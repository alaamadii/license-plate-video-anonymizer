"""Command-line interface."""

import json
import tempfile
import time
from pathlib import Path

import typer

from plate_anonymizer.benchmark.core import BenchmarkResult
from plate_anonymizer.detection.base import BaseDetector
from plate_anonymizer.detection.tiled import TiledDetector
from plate_anonymizer.detection.yolo import YoloDetector
from plate_anonymizer.evaluation.metrics import match_boxes
from plate_anonymizer.evaluation.video import evaluate_video
from plate_anonymizer.models import BoundingBox
from plate_anonymizer.pipeline import anonymize_video
from plate_anonymizer.redaction.core import RedactionMode
from plate_anonymizer.video.ffmpeg import ffmpeg_available, mux_original_audio

app = typer.Typer(
    name="plate-anonymizer",
    help="Recall-first license-plate video anonymization.",
    no_args_is_help=True,
)


@app.command()
def anonymize(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    metadata: Path | None = typer.Option(None, "--metadata"),
    device: str = typer.Option("cpu"),
    confidence: float = typer.Option(0.15, min=0.0, max=1.0),
    image_size: int = typer.Option(1280, min=320),
    inference_mode: str = typer.Option("full", help="full, tiled, or hybrid"),
    tile_size: int = typer.Option(1280, min=320),
    tile_overlap: float = typer.Option(0.2, min=0.0, max=0.9),
    box_padding: float = typer.Option(0.15, min=0.0),
    redaction: RedactionMode = typer.Option(RedactionMode.BLUR),
    temporal_max_gap: int = typer.Option(3, min=0, max=30),
    tracking: bool = typer.Option(True, "--tracking/--no-tracking"),
    preserve_audio: bool = typer.Option(False, "--preserve-audio/--no-preserve-audio"),
    plate_class_id: list[int] | None = typer.Option(None, "--plate-class-id"),
    tracker_mode: str = typer.Option("iou", help="iou baseline or experimental motion"),
    motion_max_gap_seconds: float = typer.Option(0.2, min=0, max=1),
) -> None:
    """Detect and permanently obscure plates in a video."""
    if inference_mode not in {"full", "tiled", "hybrid"}:
        raise typer.BadParameter("inference-mode must be full, tiled, or hybrid")
    if tracker_mode not in {"iou", "motion"}:
        raise typer.BadParameter("tracker-mode must be iou or motion")
    if tracker_mode == "motion" and not tracking:
        raise typer.BadParameter("motion tracking requires --tracking")

    metadata_path = metadata or output_path.with_suffix(".jsonl")
    paths = [p.resolve() for p in (input_path, output_path, metadata_path, model)]
    if len(set(paths)) != len(paths):
        raise typer.BadParameter("Input, output, metadata and model must be different files.")
    if preserve_audio and not ffmpeg_available():
        raise typer.BadParameter("ffmpeg is required for --preserve-audio; install it first.")
    typer.echo(f"Loading model: {model} (device={device}, image-size={image_size})")
    try:
        base: BaseDetector = YoloDetector(
            model, confidence=confidence, device=device, image_size=image_size,
            plate_class_ids=plate_class_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--model") from exc
    detector: BaseDetector = base
    if inference_mode in {"tiled", "hybrid"}:
        detector = TiledDetector(
            base,
            tile_size=tile_size,
            overlap=tile_overlap,
            include_full_frame=inference_mode == "hybrid",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video: Path | None = None
    processing_output = output_path
    if preserve_audio:
        suffix = output_path.suffix or ".mp4"
        temp = tempfile.NamedTemporaryFile(
            prefix="plate-anonymizer-", suffix=suffix, delete=False
        )
        temp.close()
        temporary_video = Path(temp.name)
        processing_output = temporary_video

    started = time.perf_counter()
    last_update = 0.0
    total_detections = 0

    def report_progress(done: int, total: int, detections: int) -> None:
        nonlocal last_update, total_detections
        total_detections = detections
        now = time.perf_counter()
        if done == 1 or done == total or now - last_update >= 2:
            typer.echo(
                f"Frames: {done}/{total or '?'} | Plate detections: {detections} | "
                f"Elapsed: {now - started:.1f}s"
            )
            last_update = now

    try:
        frames = anonymize_video(
            input_path,
            processing_output,
            detector,
            metadata_path,
            box_padding,
            redaction,
            tracking=tracking,
            temporal_max_gap=temporal_max_gap,
            progress=report_progress,
            tracker_mode=tracker_mode,
            motion_max_gap_seconds=motion_max_gap_seconds,
        )
        if preserve_audio and temporary_video is not None:
            mux_original_audio(temporary_video, input_path, output_path)
    finally:
        if temporary_video is not None:
            temporary_video.unlink(missing_ok=True)

    typer.echo(
        f"Processed {frames} frames in {time.perf_counter() - started:.2f}s. "
        f"Output: {output_path}. Metadata: {metadata_path}"
    )
    if total_detections == 0:
        typer.echo(
            "WARNING: No plates detected. Output contains no redactions. "
            "Check model suitability, confidence and image size.", err=True,
        )


def _load_boxes(path: Path) -> list[BoundingBox]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [BoundingBox(*map(float, item["bbox"])) for item in data]


@app.command()
def evaluate(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    ground_truth: Path = typer.Option(..., exists=True, dir_okay=False),
    threshold: float = typer.Option(0.5, min=0.0, max=1.0),
    metric: str = typer.Option("iou", help="iou or coverage"),
) -> None:
    """Evaluate a JSON list of predicted boxes against ground truth boxes."""
    result = match_boxes(_load_boxes(predictions), _load_boxes(ground_truth), threshold, metric)
    typer.echo(json.dumps({
        "tp": result.tp, "fp": result.fp, "fn": result.fn,
        "precision": result.precision, "recall": result.recall, "f1": result.f1,
    }, indent=2))


@app.command("evaluate-video")
def evaluate_video_command(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    ground_truth: Path = typer.Option(..., exists=True, dir_okay=False),
    report: Path = typer.Option(..., dir_okay=False),
    metric: str = typer.Option("iou"),
    threshold: float = typer.Option(0.5, min=0.001, max=1.0),
    box_field: str = typer.Option("bbox"),
    video: Path | None = typer.Option(None, exists=True, dir_okay=False),
    failure_dir: Path | None = typer.Option(None, file_okay=False),
) -> None:
    """Evaluate pipeline JSONL by frame and export a repeatable report with missed plates."""
    if report.resolve() in {p.resolve() for p in (predictions, ground_truth, video) if p}:
        raise typer.BadParameter("Report must not overwrite an input file.")
    try:
        result = evaluate_video(
            predictions, ground_truth, metric, threshold, box_field, video, failure_dir
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(result["overall"], indent=2))
    typer.echo(f"Report: {report}")


@app.command()
def benchmark(
    frames: int = typer.Option(..., min=1),
    source_fps: float = typer.Option(..., min=0.001),
    processing_seconds: float = typer.Option(..., min=0.001),
    gpu_hour_cost: float | None = typer.Option(None, min=0.0),
) -> None:
    """Calculate throughput and optional compute cost from a measured run."""
    result = BenchmarkResult(frames, source_fps, processing_seconds)
    payload: dict[str, float | int | None] = {
        "frames": frames,
        "processing_fps": result.processing_fps,
        "video_hours_per_gpu_hour": result.video_hours_per_gpu_hour,
    }
    if gpu_hour_cost is not None:
        payload["cost_per_video_hour"] = result.cost_per_video_hour(gpu_hour_cost)
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()

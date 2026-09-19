"""FFmpeg helpers for production video/audio output."""

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def mux_original_audio(video_only: Path, original: Path, output: Path) -> None:
    """Copy processed video and original audio into final container when audio exists.

    Video is stream-copied. Audio is copied where the output container supports it.
    If the input has no audio, FFmpeg still succeeds because the audio map is optional.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for --preserve-audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(original),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-2000:]}")

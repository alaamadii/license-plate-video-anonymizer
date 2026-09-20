"""FFmpeg helpers for production video/audio output."""

import shutil
import subprocess
from pathlib import Path


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def ffmpeg_available() -> bool:
    return ffmpeg_executable() is not None


def mux_original_audio(video_only: Path, original: Path, output: Path) -> None:
    """Copy processed video and original audio into final container when audio exists.

    MP4 output uses H.264/AAC for compatibility with Windows players and browsers.
    Other containers use stream copy.
    If the input has no audio, FFmpeg still succeeds because the audio map is optional.
    """
    executable = ffmpeg_executable()
    if executable is None:
        raise RuntimeError("ffmpeg is required for --preserve-audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "-y",
        "-i", str(video_only),
        "-i", str(original),
        "-map", "0:v:0",
        "-map", "1:a?",
    ]
    if output.suffix.lower() == ".mp4":
        command += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        ]
    else:
        command += ["-c:v", "copy", "-c:a", "copy"]
    command += [str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-2000:]}")

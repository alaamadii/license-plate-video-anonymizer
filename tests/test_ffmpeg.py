from plate_anonymizer.video.ffmpeg import ffmpeg_available


def test_ffmpeg_probe_returns_bool() -> None:
    assert isinstance(ffmpeg_available(), bool)


def test_mp4_uses_compatible_codecs_and_optional_audio(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from plate_anonymizer.video import ffmpeg

    monkeypatch.setattr(ffmpeg, "ffmpeg_executable", lambda: "bundled-ffmpeg.exe")
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(ffmpeg.subprocess, "run", run)
    ffmpeg.mux_original_audio(tmp_path / "video.mp4", tmp_path / "in.mp4", tmp_path / "out.mp4")
    args = run.call_args.args[0]
    assert args[0] == "bundled-ffmpeg.exe"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-c:a") + 1] == "aac"
    assert "1:a?" in args
    assert "-shortest" not in args  # Short audio must not truncate the processed video.

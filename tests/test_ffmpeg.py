from plate_anonymizer.video.ffmpeg import ffmpeg_available


def test_ffmpeg_probe_returns_bool() -> None:
    assert isinstance(ffmpeg_available(), bool)

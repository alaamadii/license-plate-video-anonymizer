import pytest

from plate_anonymizer.benchmark.core import BenchmarkResult


def test_benchmark_cost() -> None:
    result = BenchmarkResult(frames=3000, source_fps=30, processing_seconds=50)
    assert result.processing_fps == pytest.approx(60)
    assert result.video_hours_per_gpu_hour == pytest.approx(2)
    assert result.cost_per_video_hour(0.5) == pytest.approx(0.25)

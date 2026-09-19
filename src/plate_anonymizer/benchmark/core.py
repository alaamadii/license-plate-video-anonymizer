"""Throughput and cost calculations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    frames: int
    source_fps: float
    processing_seconds: float

    @property
    def processing_fps(self) -> float:
        return self.frames / self.processing_seconds if self.processing_seconds > 0 else 0.0

    @property
    def video_seconds(self) -> float:
        return self.frames / self.source_fps if self.source_fps > 0 else 0.0

    @property
    def video_hours_per_gpu_hour(self) -> float:
        if self.processing_seconds <= 0:
            return 0.0
        return self.video_seconds / self.processing_seconds

    def cost_per_video_hour(self, gpu_hour_cost: float) -> float | None:
        throughput = self.video_hours_per_gpu_hour
        return gpu_hour_cost / throughput if throughput > 0 else None

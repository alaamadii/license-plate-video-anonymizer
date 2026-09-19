"""OpenCV video reader/writer primitives."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if self.info.fps <= 0:
            raise RuntimeError("Input video reports an invalid FPS.")

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield index, index * 1000.0 / self.info.fps, frame
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class VideoWriter:
    def __init__(self, path: Path, info: VideoInfo, codec: str = "mp4v") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(path), fourcc, info.fps, (info.width, info.height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not create output video: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

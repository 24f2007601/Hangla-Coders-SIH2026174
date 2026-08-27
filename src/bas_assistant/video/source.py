"""Frame producers: OpenCV-backed webcam/file source and a deterministic dummy source."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class OpenCVVideoSource:
    """Reads frames from a camera index or a video file via OpenCV."""

    def __init__(self, source: int | str, width: int = 1280, height: int = 720) -> None:
        self._source = source
        self._requested_width = width
        self._requested_height = height
        self._capture: cv2.VideoCapture | None = None

    @property
    def width(self) -> int:
        if self._capture is None:
            return self._requested_width
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self._requested_width)

    @property
    def height(self) -> int:
        if self._capture is None:
            return self._requested_height
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._requested_height)

    @property
    def fps(self) -> float:
        if self._capture is None:
            return 0.0
        return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def start(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(
                f"Could not open video source {self._source!r}. "
                "Check the webcam is connected / the path is valid."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._capture = capture
        logger.info(
            "Opened video source %r (%dx%d @ %.1f fps)",
            self._source,
            self.width,
            self.height,
            self.fps,
        )

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            self.start()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Closed video source %r", self._source)


class DummyVideoSource:
    """Deterministic synthetic source so the pipeline runs with no camera/hardware.

    Produces a slowly shifting grey pattern with a labelled centre box — enough to
    exercise detection → pose → features end to end in tests and demos.
    """

    def __init__(self, width: int = 320, height: int = 240, num_frames: int = 240) -> None:
        self._width = width
        self._height = height
        self._num_frames = num_frames
        self._frame_index = 0
        self._running = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return 30.0

    def start(self) -> None:
        self._frame_index = 0
        self._running = True

    def read(self) -> np.ndarray | None:
        if not self._running:
            self.start()
        if self._frame_index >= self._num_frames:
            return None
        t = self._frame_index / max(self._num_frames, 1)
        base = int(90 + 60 * abs(np.sin(2 * np.pi * t)))
        frame = np.full((self._height, self._width, 3), base, dtype=np.uint8)
        cx, cy = self._width // 2, self._height // 2
        cv2.rectangle(frame, (cx - 30, cy - 40), (cx + 30, cy + 40), (255, 255, 255), -1)
        cv2.putText(
            frame,
            str(self._frame_index),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        self._frame_index += 1
        return frame

    def stop(self) -> None:
        self._running = False

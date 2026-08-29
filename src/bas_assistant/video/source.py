"""Frame producers: OpenCV-backed camera/file source and a deterministic dummy source.

The camera capture layer owns all platform-specific handling. It:

* resolves an explicit OpenCV backend (V4L2 on Linux, DirectShow on Windows),
* requests a configured pixel format / resolution / FPS from the driver,
* reads the negotiated values back after ``VideoCapture.set()`` (drivers may
  silently pick a different mode),
* falls back to a supported mode when the requested one is unavailable,
* logs the backend and actual mode, and tracks read success/failure + FPS.

The rest of the pipeline only ever sees plain NumPy/OpenCV frames.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BACKEND_NAMES = {
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_FFMPEG: "FFMPEG",
    cv2.CAP_ANY: "ANY",
}

# Ordered fallbacks tried when the requested mode cannot be negotiated.
DEFAULT_FALLBACK_MODES: tuple[tuple[int, int, int, str], ...] = (
    (960, 540, 30, "MJPG"),
    (640, 480, 30, "MJPG"),
    (1280, 720, 10, "YUYV"),
    (960, 540, 20, "YUYV"),
    (640, 480, 30, "YUYV"),
)


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


@dataclass(frozen=True, slots=True)
class CameraParams:
    """A camera mode: resolution, frame rate, and pixel format (FourCC string)."""

    width: int
    height: int
    fps: float
    format: str

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps:.0f} FPS {self.format or '?'}"


def _fourcc_code(fourcc: str | None) -> int | None:
    """Convert a 4-char codec string (e.g. ``'MJPG'``) to the OpenCV FOURCC int."""
    if fourcc is None or len(fourcc) != 4:
        return None
    return cv2.VideoWriter_fourcc(*fourcc)  # type: ignore[attr-defined]


def _fourcc_to_str(code: float) -> str:
    """Decode an OpenCV FOURCC int back to a readable codec string."""
    if code <= 0:
        return ""
    value = int(code)
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def _resolve_backend(backend: str, camera_source: bool) -> int:
    """Map a config backend name to an OpenCV backend constant.

    For camera sources ``auto`` resolves to the platform default backend
    (V4L2 on Linux, DirectShow on Windows) because OpenCV's ``CAP_ANY`` can pick
    a backend where requested properties silently fail to apply. File sources
    always use ``CAP_ANY`` (the FFMPEG path handles files).
    """
    if not camera_source:
        return cv2.CAP_ANY
    if backend == "auto":
        return cv2.CAP_V4L2 if sys.platform != "win32" else cv2.CAP_DSHOW
    if backend == "v4l2":
        return cv2.CAP_V4L2
    if backend == "dshow":
        return cv2.CAP_DSHOW
    if backend == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def _backend_display_name(capture) -> str:
    """Human-readable backend name; falls back to a constant map when unknown."""
    try:
        name = capture.getBackendName()
    except Exception:  # noqa: BLE001 - some backends do not expose a name
        name = ""
    return name or "unknown"


def _read_params(capture) -> CameraParams:
    """Read the currently negotiated mode back from the capture device."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fmt = _fourcc_to_str(capture.get(cv2.CAP_PROP_FOURCC))
    return CameraParams(width=width, height=height, fps=fps, format=fmt)


def _configure_once(capture, width: int, height: int, fps: int, fourcc: str | None) -> CameraParams:
    """Request a mode (FourCC first, then FPS/resolution) and read the result back."""
    if fourcc is not None:
        code = _fourcc_code(fourcc)
        if code is not None:
            capture.set(cv2.CAP_PROP_FOURCC, code)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return _read_params(capture)


def _mode_acceptable(
    actual: CameraParams, width: int, height: int, fps: int, fourcc: str | None
) -> bool:
    """True when the negotiated mode satisfies the requested resolution and frame rate.

    The FourCC is deliberately *not* a hard requirement here: several Windows
    backends do not report it reliably, so a mismatch is logged, not failed.
    """
    resolution_ok = actual.width == width and actual.height == height
    fps_ok = fps <= 0 or actual.fps >= fps * 0.7
    return resolution_ok and fps_ok


def _choose_configuration(
    capture,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None,
    fallbacks: tuple[tuple[int, int, int, str], ...],
) -> tuple[tuple[int, int, int, str], CameraParams]:
    """Try the requested mode, then each fallback; return the best found.

    Returns the mode tuple that produced an acceptable result (or the requested
    mode when none matched) together with the actual negotiated ``CameraParams``.
    """
    candidates = [(width, height, fps, fourcc or "")]
    candidates += [mode for mode in fallbacks if mode != candidates[0]]
    last_params: CameraParams | None = None
    for mode in candidates:
        params = _configure_once(capture, mode[0], mode[1], mode[2], mode[3] or None)
        if _mode_acceptable(params, mode[0], mode[1], mode[2], mode[3]):
            return mode, params
        last_params = params
    assert last_params is not None
    return (width, height, fps, fourcc or ""), last_params


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _disable_v4l2_dynamic_framerate(device: int | str) -> str:
    """Best-effort: turn off ``exposure_dynamic_framerate`` on a Linux V4L2 camera.

    The integrated UVC driver enables this control by default; combined with
    aperture-priority auto-exposure it stretches exposure and silently drops the
    delivered frame rate (measured: 30 -> ~17 fps on this class of laptop camera),
    which breaks MediaPipe hand tracking. Skipped when ``v4l2-ctl`` is unavailable.
    Returns a short human-readable status string.
    """
    tool = shutil.which("v4l2-ctl")
    if tool is None:
        return "v4l2-ctl not available; skipped"
    if not isinstance(device, int):
        return "not a camera index; skipped"
    try:
        result = subprocess.run(
            [tool, "-d", f"/dev/video{device}", "--set-ctrl", "exposure_dynamic_framerate=0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed: {exc}"
    if result.returncode != 0:
        return f"failed: {result.stderr.strip() or result.returncode}"
    return "disabled (exposure_dynamic_framerate=0)"


class OpenCVVideoSource:
    """Reads frames from a camera index or a video file via OpenCV.

    Camera sources are opened with an explicit backend and negotiated to the
    requested format/FPS/resolution, then verified by reading the values back.
    Without an explicit format on Linux, V4L2 falls back to YUYV, which often
    caps high resolutions at ~10 fps and breaks MediaPipe hand tracking;
    requesting MJPG avoids that.
    """

    def __init__(
        self,
        device: int | str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        format: str | None = "MJPG",
        backend: str = "auto",
        fallback: bool = True,
        disable_dynamic_framerate: bool = False,
        log_interval: int = 300,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._format = format
        self._backend = backend
        self._fallback = fallback
        self._disable_dynamic_framerate = disable_dynamic_framerate
        self._log_interval = log_interval
        self._capture: cv2.VideoCapture | None = None

        self._backend_name = ""
        self._requested: CameraParams | None = None
        self._actual: CameraParams | None = None

        self._frames_read = 0
        self._frames_failed = 0
        self._capture_times_ms: deque[float] = deque(maxlen=60)
        self._frame_gaps_ms: deque[float] = deque(maxlen=60)
        self._last_read_ts: float | None = None

    @property
    def width(self) -> int:
        if self._capture is None:
            return self._width
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self._width)

    @property
    def height(self) -> int:
        if self._capture is None:
            return self._height
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._height)

    @property
    def fps(self) -> float:
        if self._capture is None:
            return 0.0
        return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def requested_params(self) -> CameraParams | None:
        return self._requested

    @property
    def actual_params(self) -> CameraParams | None:
        return self._actual

    def start(self) -> None:
        if self._capture is not None:
            return
        camera_source = isinstance(self._device, int)
        backend = _resolve_backend(self._backend, camera_source)
        capture = cv2.VideoCapture(self._device, backend)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(
                f"Could not open video source {self._device!r} (backend={self._backend}). "
                "Check the webcam is connected / the path is valid."
            )
        self._backend_name = _backend_display_name(capture)
        fmt_upper = (self._format or "").upper()
        requested = CameraParams(self._width, self._height, float(self._fps), fmt_upper)

        if camera_source:
            fallbacks = DEFAULT_FALLBACK_MODES if self._fallback else ()
            chosen, actual = _choose_configuration(
                capture, self._width, self._height, self._fps, self._format, fallbacks
            )
            self._requested = CameraParams(*chosen)
            self._actual = actual
            if self._requested != requested:
                logger.warning(
                    "Camera %r: could not use requested %s; using %s",
                    self._device,
                    requested.describe(),
                    self._requested.describe(),
                )
            if self._actual != self._requested:
                logger.warning(
                    "Camera %r negotiated a different mode than requested: %s",
                    self._device,
                    self._actual.describe(),
                )
        else:
            self._requested = None
            self._actual = _read_params(capture)

        if camera_source and self._disable_dynamic_framerate:
            status = _disable_v4l2_dynamic_framerate(self._device)
            logger.info("Driver control (camera %r): %s", self._device, status)

        self._capture = capture
        self._frames_read = 0
        self._frames_failed = 0
        self._capture_times_ms.clear()
        self._frame_gaps_ms.clear()
        self._last_read_ts = None

        logger.info(
            "Camera backend=%s requested=%s actual=%s",
            self._backend_name,
            requested.describe() if camera_source else "n/a (file)",
            self._actual.describe() if self._actual else "n/a",
        )
        if (
            camera_source
            and self._format
            and self._actual
            and self._actual.format
            and self._actual.format != self._format.upper()
        ):
            logger.warning(
                "Camera %r reports pixel format %r (requested %s). On Linux, YUYV often "
                "caps to ~10 fps at high resolution, which degrades MediaPipe hand "
                "tracking. Set camera.format=MJPG in configs/default.yaml if supported.",
                self._device,
                self._actual.format,
                self._format,
            )

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            self.start()
        assert self._capture is not None
        t0 = time.perf_counter()
        ok, frame = self._capture.read()
        capture_ms = (time.perf_counter() - t0) * 1000.0
        now = time.monotonic()
        if self._last_read_ts is not None:
            self._frame_gaps_ms.append((now - self._last_read_ts) * 1000.0)
        self._last_read_ts = now

        if ok and frame is not None and frame.size > 0:
            self._frames_read += 1
            self._capture_times_ms.append(capture_ms)
            if self._log_interval and self._frames_read % self._log_interval == 0:
                logger.info("Camera stats (frame %d): %s", self._frames_read, self.diagnostics())
            return frame

        self._frames_failed += 1
        if self._log_interval and self._frames_failed % self._log_interval == 0:
            logger.warning(
                "Camera read failures: %d total (last capture took %.1f ms)",
                self._frames_failed,
                capture_ms,
            )
        return None

    def diagnostics(self) -> dict:
        """Summary of negotiated mode, read counts, and acquisition timing."""
        gaps = list(self._frame_gaps_ms)
        return {
            "backend": self._backend_name,
            "requested": self._requested.as_dict() if self._requested else None,
            "actual": self._actual.as_dict() if self._actual else None,
            "frames_read": self._frames_read,
            "frames_failed": self._frames_failed,
            "capture_ms_mean": round(_mean(list(self._capture_times_ms)), 2),
            "frame_gap_ms_mean": round(_mean(gaps), 2),
            "acquisition_fps": round(1000.0 / _mean(gaps), 2) if gaps else 0.0,
            "frame_size": [self.width, self.height],
        }

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            logger.info("Closed video source %r", self._device)
            logger.debug("Camera diagnostics: %s", self.diagnostics())
            self._capture = None


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
        self._frames_read = 0
        self._frames_failed = 0

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
        self._frames_read = 0
        self._frames_failed = 0

    def read(self) -> np.ndarray | None:
        if not self._running:
            self.start()
        if self._frame_index >= self._num_frames:
            self._frames_failed += 1
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
        self._frames_read += 1
        return frame

    def stop(self) -> None:
        self._running = False

    def diagnostics(self) -> dict:
        """Uniform diagnostics dict so scripts treat dummy and camera sources alike."""
        return {
            "backend": "dummy",
            "requested": None,
            "actual": None,
            "frames_read": self._frames_read,
            "frames_failed": self._frames_failed,
            "capture_ms_mean": 0.0,
            "frame_gap_ms_mean": 0.0,
            "acquisition_fps": 0.0,
            "frame_size": [self._width, self._height],
        }


__all__ = [
    "CameraParams",
    "DEFAULT_FALLBACK_MODES",
    "DummyVideoSource",
    "OpenCVVideoSource",
    "VideoSourceError",
    "_configure_once",
    "_fourcc_code",
    "_fourcc_to_str",
    "_mode_acceptable",
    "_read_params",
    "_resolve_backend",
]

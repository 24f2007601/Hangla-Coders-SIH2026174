"""Unit tests for video-source negotiation helpers (no camera required)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bas_assistant.video.source import (
    CameraParams,
    DummyVideoSource,
    OpenCVVideoSource,
    VideoSourceError,
    _choose_configuration,
    _configure_once,
    _disable_v4l2_dynamic_framerate,
    _fourcc_code,
    _fourcc_to_str,
    _mode_acceptable,
    _read_params,
    _resolve_backend,
)


class _FakeCapture:
    """Mimics ``cv2.VideoCapture`` for the subset of props the negotiator uses.

    ``capabilities`` is a list of (width, height, fps, fourcc_str) modes the fake
    device supports. Requesting props is clamped to the closest supported mode, so
    an unsupported 1280x720 request reports back a lower mode — exactly what a real
    driver does.
    """

    def __init__(self, capabilities: list[tuple[int, int, int, str]]) -> None:
        self._caps = capabilities
        self._req = {"fourcc": 0, "fps": 0.0, "w": 0, "h": 0}

    def set(self, prop: int, value) -> bool:
        if prop == cv2.CAP_PROP_FOURCC:
            self._req["fourcc"] = int(value)
        elif prop == cv2.CAP_PROP_FPS:
            self._req["fps"] = float(value)
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._req["w"] = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._req["h"] = int(value)
        return True

    def get(self, prop: int) -> float:
        best = self._best()
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(best[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(best[1])
        if prop == cv2.CAP_PROP_FPS:
            return float(best[2])
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*best[3]))
        return 0.0

    def _best(self) -> tuple[int, int, int, str]:
        fmt = _fourcc_to_str(self._req["fourcc"])
        matching = [c for c in self._caps if not fmt or c[3] == fmt]
        if not matching:
            matching = self._caps
        return min(
            matching,
            key=lambda c: (
                abs(c[0] - self._req["w"])
                + abs(c[1] - self._req["h"])
                + abs(c[2] - self._req["fps"])
            ),
        )


# -- FourCC conversion ---------------------------------------------------


def test_fourcc_code_mjpg() -> None:
    assert _fourcc_code("MJPG") == cv2.VideoWriter_fourcc(*"MJPG")


def test_fourcc_code_none_and_wrong_length() -> None:
    assert _fourcc_code(None) is None
    assert _fourcc_code("ABC") is None
    assert _fourcc_code("TOOLONG") is None


def test_fourcc_to_str_roundtrip() -> None:
    assert _fourcc_to_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"


def test_fourcc_to_str_zero() -> None:
    assert _fourcc_to_str(0) == ""


# -- Backend selection ---------------------------------------------------


def test_resolve_backend_auto_linux_camera(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    assert _resolve_backend("auto", camera_source=True) == cv2.CAP_V4L2


def test_resolve_backend_auto_windows_camera(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    assert _resolve_backend("auto", camera_source=True) == cv2.CAP_DSHOW


def test_resolve_backend_explicit(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    assert _resolve_backend("v4l2", camera_source=True) == cv2.CAP_V4L2
    assert _resolve_backend("dshow", camera_source=True) == cv2.CAP_DSHOW
    assert _resolve_backend("msmf", camera_source=True) == cv2.CAP_MSMF


def test_resolve_backend_file_source_ignores_backend(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    assert _resolve_backend("auto", camera_source=False) == cv2.CAP_ANY
    assert _resolve_backend("v4l2", camera_source=False) == cv2.CAP_ANY


def test_resolve_backend_unknown_falls_back_to_any(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    assert _resolve_backend("bogus", camera_source=True) == cv2.CAP_ANY


# -- Negotiation / verification / fallback -------------------------------


def test_read_params_returns_negotiated_mode() -> None:
    fake = _FakeCapture([(640, 480, 30, "MJPG")])
    fake.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    fake.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fake.set(cv2.CAP_PROP_FPS, 30)
    fake.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    params = _read_params(fake)
    assert params == CameraParams(width=640, height=480, fps=30.0, format="MJPG")


def test_mode_acceptable_exact() -> None:
    actual = CameraParams(width=1280, height=720, fps=30.0, format="MJPG")
    assert _mode_acceptable(actual, 1280, 720, 30, "MJPG")


def test_mode_acceptable_resolution_mismatch() -> None:
    actual = CameraParams(width=640, height=480, fps=30.0, format="MJPG")
    assert not _mode_acceptable(actual, 1280, 720, 30, "MJPG")


def test_mode_acceptable_fps_too_low() -> None:
    actual = CameraParams(width=1280, height=720, fps=10.0, format="YUYV")
    assert not _mode_acceptable(actual, 1280, 720, 30, "MJPG")


def test_configure_once_reads_back_requested_mode() -> None:
    fake = _FakeCapture([(1280, 720, 30, "MJPG"), (640, 480, 30, "MJPG")])
    params = _configure_once(fake, 1280, 720, 30, "MJPG")
    assert params == CameraParams(width=1280, height=720, fps=30.0, format="MJPG")


def test_choose_configuration_uses_requested_when_supported() -> None:
    fake = _FakeCapture([(1280, 720, 30, "MJPG"), (640, 480, 30, "MJPG")])
    mode, actual = _choose_configuration(fake, 1280, 720, 30, "MJPG", ())
    assert mode == (1280, 720, 30, "MJPG")
    assert actual == CameraParams(width=1280, height=720, fps=30.0, format="MJPG")


def test_choose_configuration_falls_back_to_supported_mode() -> None:
    fake = _FakeCapture([(640, 480, 30, "MJPG")])
    mode, actual = _choose_configuration(fake, 1280, 720, 30, "MJPG", ((640, 480, 30, "MJPG"),))
    assert mode == (640, 480, 30, "MJPG")
    assert actual == CameraParams(width=640, height=480, fps=30.0, format="MJPG")


def test_choose_configuration_no_fallback_returns_actual() -> None:
    fake = _FakeCapture([(640, 480, 30, "MJPG")])
    mode, actual = _choose_configuration(fake, 1280, 720, 30, "MJPG", ())
    assert mode == (1280, 720, 30, "MJPG")  # requested intent retained
    assert actual.width == 640  # but the device reported what it could do


# -- Source lifecycle (mocked cv2.VideoCapture) ---------------------------


class _SourceFakeCapture(_FakeCapture):
    """A _FakeCapture that also answers the lifecycle calls VideoCapture exposes."""

    def __init__(
        self,
        capabilities: list[tuple[int, int, int, str]],
        *,
        opened: bool = True,
        backend: str = "FAKE",
        frames: list[tuple[bool, object]] | None = None,
    ) -> None:
        super().__init__(capabilities)
        self._opened = opened
        self._backend = backend
        self._frames = list(frames or [])

    def isOpened(self) -> bool:
        return self._opened

    def getBackendName(self) -> str:
        return self._backend

    def read(self) -> tuple[bool, object]:
        if self._frames:
            return self._frames.pop(0)
        return (False, None)

    def release(self) -> None:
        self._frames.clear()


def _patch_videocapture(monkeypatch, capture) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: capture)


def test_source_start_negotiates_and_exposes_params(monkeypatch) -> None:
    capture = _SourceFakeCapture([(1280, 720, 30, "MJPG")])
    _patch_videocapture(monkeypatch, capture)

    source = OpenCVVideoSource(0, width=1280, height=720, fps=30, format="MJPG")
    source.start()

    assert source.backend_name == "FAKE"
    assert source.requested_params == CameraParams(1280, 720, 30.0, "MJPG")
    assert source.actual_params == CameraParams(1280, 720, 30.0, "MJPG")
    source.stop()


def test_source_start_falls_back_and_records_requested_intent(monkeypatch) -> None:
    capture = _SourceFakeCapture([(640, 480, 30, "MJPG")])
    _patch_videocapture(monkeypatch, capture)

    source = OpenCVVideoSource(0, width=1280, height=720, fps=30, format="MJPG")
    source.start()

    assert source.requested_params == CameraParams(640, 480, 30.0, "MJPG")  # accepted mode
    assert source.actual_params == CameraParams(640, 480, 30.0, "MJPG")
    source.stop()


def test_source_start_raises_when_camera_not_opened(monkeypatch) -> None:
    capture = _SourceFakeCapture([], opened=False)
    _patch_videocapture(monkeypatch, capture)

    with pytest.raises(VideoSourceError):
        OpenCVVideoSource(0).start()


def test_source_read_counts_success_and_failure(monkeypatch) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = _SourceFakeCapture([(1280, 720, 30, "MJPG")], frames=[(True, frame), (False, None)])
    _patch_videocapture(monkeypatch, capture)

    source = OpenCVVideoSource(0, width=1280, height=720, fps=30, format="MJPG")
    source.start()
    assert source.read() is not None
    assert source.read() is None

    diagnostics = source.diagnostics()
    assert diagnostics["frames_read"] == 1
    assert diagnostics["frames_failed"] == 1
    assert diagnostics["backend"] == "FAKE"
    assert diagnostics["frame_size"] == [1280, 720]
    source.stop()


def test_source_file_source_does_not_negotiate(monkeypatch) -> None:
    capture = _SourceFakeCapture([(1280, 720, 30, "MJPG")])
    _patch_videocapture(monkeypatch, capture)

    source = OpenCVVideoSource("data/samples/x.mp4")
    source.start()
    assert source.requested_params is None  # file sources read back, not negotiate
    assert source.actual_params is not None
    source.stop()


# -- Dummy source diagnostics ---------------------------------------------


def test_dummy_source_reads_and_diagnostics() -> None:
    source = DummyVideoSource(width=160, height=120, num_frames=3)
    source.start()
    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is None  # exhausted -> a failed read

    diagnostics = source.diagnostics()
    assert diagnostics["backend"] == "dummy"
    assert diagnostics["frames_read"] == 3
    assert diagnostics["frames_failed"] == 1
    assert diagnostics["frame_size"] == [160, 120]
    source.stop()


# -- Linux-only driver control --------------------------------------------


def test_disable_dynamic_framerate_skips_without_v4l2_ctl(monkeypatch) -> None:
    monkeypatch.setattr("bas_assistant.video.source.shutil.which", lambda _: None)
    status = _disable_v4l2_dynamic_framerate(0)
    assert "not available" in status


def test_disable_dynamic_framerate_skips_for_file_device(monkeypatch) -> None:
    monkeypatch.setattr("bas_assistant.video.source.shutil.which", lambda _: "/usr/bin/v4l2-ctl")
    status = _disable_v4l2_dynamic_framerate("path/to/video.mp4")
    assert "not a camera index" in status


def test_disable_dynamic_framerate_runs_v4l2_ctl(monkeypatch) -> None:
    monkeypatch.setattr("bas_assistant.video.source.shutil.which", lambda _: "/usr/bin/v4l2-ctl")

    class _Ok:
        returncode = 0
        stderr = ""

    monkeypatch.setattr("bas_assistant.video.source.subprocess.run", lambda *a, **k: _Ok())
    status = _disable_v4l2_dynamic_framerate(0)
    assert status == "disabled (exposure_dynamic_framerate=0)"


def test_disable_dynamic_framerate_reports_failure(monkeypatch) -> None:
    monkeypatch.setattr("bas_assistant.video.source.shutil.which", lambda _: "/usr/bin/v4l2-ctl")

    class _Fail:
        returncode = 1
        stderr = "no such control"

    monkeypatch.setattr("bas_assistant.video.source.subprocess.run", lambda *a, **k: _Fail())
    status = _disable_v4l2_dynamic_framerate(0)
    assert "failed" in status
    assert "no such control" in status

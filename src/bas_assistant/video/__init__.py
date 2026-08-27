"""Video input sources."""

from bas_assistant.video.source import (
    DummyVideoSource,
    OpenCVVideoSource,
    VideoSourceError,
)

__all__ = ["DummyVideoSource", "OpenCVVideoSource", "VideoSourceError"]

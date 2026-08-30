"""Unit tests for typed configuration loading."""

from __future__ import annotations

from pathlib import Path

from bas_assistant.config.settings import (
    CameraConfig,
    ClassifierConfig,
    PipelineConfig,
    Settings,
    load_settings,
)


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.camera.device == 0
    assert settings.camera.width == 1280
    assert settings.camera.height == 720
    assert settings.camera.fps == 30
    assert settings.camera.format == "MJPG"
    assert settings.camera.backend == "auto"
    assert settings.camera.disable_dynamic_framerate is False
    assert settings.pose.model == "mediapipe"
    assert settings.pose.min_hand_detection_confidence == 0.3
    assert settings.pose.hand_hold_seconds == 0.5
    assert settings.classifier.model_type == "dummy"
    assert settings.pipeline.sequence_length == 30
    assert settings.pipeline.metrics_enabled is False


def test_load_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "does_not_exist.yaml")
    assert settings == Settings()


def test_load_settings_reads_yaml(tmp_path: Path) -> None:
    config = tmp_path / "test.yaml"
    config.write_text(
        "camera:\n  device: 2\n  width: 640\n"
        "pipeline:\n  sequence_length: 15\n  classify_hop: 3\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.camera.device == 2
    assert settings.camera.width == 640
    assert settings.camera.format == "MJPG"
    assert settings.pipeline.sequence_length == 15
    assert settings.pipeline.classify_hop == 3
    assert settings.pose.model == "mediapipe"


def test_settings_validate_ranges() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PipelineConfig(sequence_length=0)


def test_settings_classifier_literal() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClassifierConfig(model_type="svm")  # type: ignore[arg-type]


def test_settings_camera_backend_literal() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CameraConfig(backend="gstreamer")  # type: ignore[arg-type]


def test_settings_camera_format_none_allowed() -> None:
    assert CameraConfig(format=None).format is None

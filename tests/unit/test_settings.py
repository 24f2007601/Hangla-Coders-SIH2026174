"""Unit tests for typed configuration loading."""

from __future__ import annotations

from pathlib import Path

from bas_assistant.config.settings import (
    ClassifierConfig,
    PipelineConfig,
    Settings,
    load_settings,
)


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.video.source == 0
    assert settings.pose.model == "mediapipe"
    assert settings.classifier.model_type == "dummy"
    assert settings.pipeline.sequence_length == 30


def test_load_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "does_not_exist.yaml")
    assert settings == Settings()


def test_load_settings_reads_yaml(tmp_path: Path) -> None:
    config = tmp_path / "test.yaml"
    config.write_text(
        "video:\n  source: 2\n  width: 640\n"
        "pipeline:\n  sequence_length: 15\n  classify_hop: 3\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.video.source == 2
    assert settings.video.width == 640
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

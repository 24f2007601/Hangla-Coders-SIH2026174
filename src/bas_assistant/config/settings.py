"""Typed configuration via Pydantic.

Defaults match `configs/default.yaml`. A `configs/*.yaml` file can override any field;
individual values can also be overridden with environment variables prefixed `BAS_`
(e.g. `BAS_CLASSIFIER__MODEL_TYPE=xgboost`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class CameraConfig(BaseModel):
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    format: str | None = "MJPG"
    backend: Literal["auto", "v4l2", "dshow", "msmf"] = "auto"
    disable_dynamic_framerate: bool = False


class PoseConfig(BaseModel):
    model: Literal["mediapipe", "dummy"] = "mediapipe"
    pose_model_path: Path = Path("models/pose_landmarker_lite.task")
    hand_model_path: Path = Path("models/hand_landmarker.task")
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_hand_detection_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    min_hand_presence_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    min_hand_tracking_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    hand_hold_seconds: float = Field(default=0.5, ge=0.0, le=5.0)
    with_hands: bool = True


class ClassifierConfig(BaseModel):
    model_type: Literal["dummy", "xgboost"] = "dummy"
    model_path: Path = Path("models/step_classifier.json")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///data/project.db"
    output_dir: Path = Path("data/processed")


class PipelineConfig(BaseModel):
    sequence_length: int = Field(default=30, ge=1)
    classify_hop: int = Field(default=5, ge=1)
    smoothing_window: int = Field(default=5, ge=1)
    max_fps: int = Field(default=30, ge=1)
    metrics_enabled: bool = False


class Settings(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_settings(path: Path | None = DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from a YAML file if it exists, otherwise use built-in defaults."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Settings.model_validate(data)
    return Settings()


__all__ = [
    "CameraConfig",
    "ClassifierConfig",
    "DatabaseConfig",
    "PipelineConfig",
    "PoseConfig",
    "Settings",
    "load_settings",
]

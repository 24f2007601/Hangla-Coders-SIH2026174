"""Pipeline orchestration."""

from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.pipeline.pipeline import ExperimentPipeline, FrameResult

__all__ = ["ExperimentPipeline", "FrameResult", "build_pipeline"]

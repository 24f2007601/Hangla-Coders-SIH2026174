"""Protocol-aware evidence checks for the microphone experiment."""

from __future__ import annotations

MIN_CONFIDENCE = 0.25
MIN_OBJECT_CONFIDENCE = 0.20


def _has_object(
    objects: list[dict],
    name: str,
) -> bool:
    """Return True when an object is detected confidently enough."""

    for obj in objects:
        if str(obj.get("name", "")) != name:
            continue

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence >= MIN_OBJECT_CONFIDENCE:
            return True

    return False


def confirm_step(
    candidate: str,
    confidence: float,
    current_index: int,
    objects: list[dict],
) -> str | None:
    """Confirm only the next expected step.

    XGBoost is the primary temporal step signal.
    Object detection provides permissive supporting evidence.

    The gate never allows a later step to advance the FSM.
    """

    expected_index = current_index + 1

    if expected_index > 6:
        return None

    expected_step = f"M{expected_index}"

    # --------------------------------------------------------------
    # Primary classifier signal
    # --------------------------------------------------------------

    if candidate == expected_step and confidence >= MIN_CONFIDENCE:
        return expected_step

    # --------------------------------------------------------------
    # Object-supported fallback
    #
    # These are deliberately limited to visually meaningful states.
    # --------------------------------------------------------------

    if expected_step == "M2":
        if _has_object(
            objects,
            "microphone_case_closed",
        ):
            return "M2"

    elif expected_step == "M3":
        if _has_object(
            objects,
            "microphone_case_open",
        ):
            return "M3"

    elif expected_step == "M4":
        if _has_object(
            objects,
            "microphone_case_open",
        ) and _has_object(
            objects,
            "receiver",
        ):
            return "M4"

    elif (
        expected_step == "M6"
        and _has_object(
            objects,
            "microphone",
        )
        and _has_object(
            objects,
            "microphone_case_open",
        )
    ):
        return "M6"

    return None


__all__ = ["confirm_step"]

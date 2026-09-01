"""Protocol-aware evidence checks for microphone steps."""

from __future__ import annotations

MIN_CONFIDENCE = 0.5
MIN_OBJECT_CONFIDENCE = 0.25


def _has_object(
    objects: list[dict],
    name: str,
) -> bool:
    """Return True when a sufficiently confident object is present."""
    return any(
        str(obj.get("name", "")) == name
        and float(obj.get("confidence", 0.0)) >= MIN_OBJECT_CONFIDENCE
        for obj in objects
    )


def confirm_step(
    candidate: str,
    confidence: float,
    current_index: int,
    objects: list[dict],
) -> str | None:
    """Confirm the next protocol step using classifier/object evidence."""

    expected_index = current_index + 1

    if expected_index > 6:
        return None

    expected_step = f"M{expected_index}"

    # ---------------------------------------------------------
    # M0 / M1
    # ---------------------------------------------------------
    #
    # These steps do not currently have sufficiently distinctive
    # object evidence, so require a confident classifier result.
    #
    if expected_step in {"M0", "M1"}:
        if confidence < MIN_CONFIDENCE:
            return None

        return expected_step if candidate == expected_step else None

    # ---------------------------------------------------------
    # M2
    # ---------------------------------------------------------
    #
    # A closed microphone case is strong direct evidence.
    # Allow that evidence to confirm M2 even when the classifier
    # itself is uncertain.
    #
    if expected_step == "M2":
        if _has_object(
            objects,
            "microphone_case_closed",
        ):
            return "M2"

        if confidence >= MIN_CONFIDENCE and candidate == "M2":
            return "M2"

        return None

    # ---------------------------------------------------------
    # M3
    # ---------------------------------------------------------
    #
    # Opening the microphone case is directly observable.
    #
    if expected_step == "M3":
        if _has_object(
            objects,
            "microphone_case_open",
        ):
            return "M3"

        return None

    # ---------------------------------------------------------
    # M4
    # ---------------------------------------------------------
    #
    # An open case with the receiver present is strong evidence
    # for the receiver-removal stage.
    #
    if expected_step == "M4":
        if _has_object(
            objects,
            "microphone_case_open",
        ) and _has_object(
            objects,
            "receiver",
        ):
            return "M4"

        return None

    # ---------------------------------------------------------
    # M5
    # ---------------------------------------------------------
    #
    # M4 and M5 have very similar static object states.
    # Therefore M5 still requires an explicit classifier signal
    # in addition to the receiver/case evidence.
    #
    if expected_step == "M5":
        if (
            candidate == "M5"
            and confidence >= MIN_CONFIDENCE
            and _has_object(
                objects,
                "microphone_case_open",
            )
            and _has_object(
                objects,
                "receiver",
            )
        ):
            return "M5"

        return None

    # ---------------------------------------------------------
    # M6
    # ---------------------------------------------------------
    #
    # Microphone presence provides the distinguishing object
    # evidence for microphone removal.
    #
    if expected_step == "M6":
        if (
            (
                candidate == "M6"
                or _has_object(
                    objects,
                    "microphone",
                )
            )
            and _has_object(
                objects,
                "microphone_case_open",
            )
            and _has_object(
                objects,
                "receiver",
            )
        ):
            return "M6"

        return None

    return None


__all__ = ["confirm_step"]

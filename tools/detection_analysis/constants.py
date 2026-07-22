"""Shared constants for detection analysis."""

from __future__ import annotations

ERROR_TRUE_POSITIVE = "true_positive"
ERROR_BACKGROUND = "background_false_positive"
ERROR_CLASSIFICATION = "classification_error"
ERROR_LOCALIZATION = "localization_error"
ERROR_CLASSIFICATION_LOCALIZATION = "classification_and_localization_error"
ERROR_DUPLICATE = "duplicate_detection"
ERROR_IGNORED = "matched_ignored_annotation"
ERROR_CROWD = "matched_crowd_annotation"
ERROR_LOW_CONFIDENCE = "low_confidence_prediction"
ERROR_INVALID = "invalid_prediction"

GT_CORRECT = "correctly_detected"
GT_FALSE_NEGATIVE = "false_negative"
GT_WRONG_CLASS = "detected_with_wrong_class"
GT_POOR_LOCALIZATION = "poorly_localized"
GT_DUPLICATE_ONLY = "only_duplicate_predictions"
GT_IGNORED = "ignored"
GT_CROWD = "crowd_annotation"

PREDICTION_ERROR_ORDER = [
    ERROR_INVALID,
    ERROR_LOW_CONFIDENCE,
    ERROR_IGNORED,
    ERROR_CROWD,
    ERROR_TRUE_POSITIVE,
    ERROR_DUPLICATE,
    ERROR_CLASSIFICATION,
    ERROR_LOCALIZATION,
    ERROR_CLASSIFICATION_LOCALIZATION,
    ERROR_BACKGROUND,
]

ERROR_COLORS = {
    ERROR_TRUE_POSITIVE: "#2ca02c",
    ERROR_BACKGROUND: "#d62728",
    ERROR_CLASSIFICATION: "#9467bd",
    ERROR_LOCALIZATION: "#ff7f0e",
    ERROR_CLASSIFICATION_LOCALIZATION: "#8c564b",
    ERROR_DUPLICATE: "#e377c2",
    ERROR_IGNORED: "#7f7f7f",
    ERROR_CROWD: "#17becf",
    ERROR_LOW_CONFIDENCE: "#bcbd22",
    ERROR_INVALID: "#000000",
    GT_FALSE_NEGATIVE: "#1f77b4",
}


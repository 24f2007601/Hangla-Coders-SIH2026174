"""Train and evaluate the fused hand + YOLO XGBoost step classifier."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

from bas_assistant.features.microphone import (
    MICROPHONE_FEATURE_VECTOR_SIZE,
)

DATA_ROOT = Path("data/processed")

MODEL_PATH = Path("models/step_classifier_fused.json")

SPLITS = {
    "train": DATA_ROOT / "fused_step_features_train.csv",
    "val": DATA_ROOT / "fused_step_features_val.csv",
    "test": DATA_ROOT / "fused_step_features_test.csv",
}

METADATA_COLUMNS = {
    "session_id",
    "split",
    "label",
    "frame",
}


def load_split(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one pre-built feature split."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if data.size == 0:
        raise ValueError(f"Dataset is empty: {path}")

    columns = list(data.dtype.names)

    feature_names = [name for name in columns if name not in METADATA_COLUMNS]

    if len(feature_names) != MICROPHONE_FEATURE_VECTOR_SIZE:
        raise ValueError(
            f"{path}: expected "
            f"{MICROPHONE_FEATURE_VECTOR_SIZE} features, "
            f"found {len(feature_names)}"
        )

    X = np.column_stack([data[name] for name in feature_names]).astype(float)

    labels = np.asarray(
        data["label"],
        dtype=str,
    )

    return X, labels, feature_names


def main() -> None:
    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    train_X, train_labels, feature_names = load_split(SPLITS["train"])

    val_X, val_labels, _ = load_split(SPLITS["val"])

    test_X, test_labels, _ = load_split(SPLITS["test"])

    # ---------------------------------------------------------
    # Classes
    # ---------------------------------------------------------

    classes = sorted(set(train_labels) | set(val_labels) | set(test_labels))

    class_to_id = {label: index for index, label in enumerate(classes)}

    y_train = np.asarray(
        [class_to_id[label] for label in train_labels],
        dtype=int,
    )

    y_val = np.asarray(
        [class_to_id[label] for label in val_labels],
        dtype=int,
    )

    y_test = np.asarray(
        [class_to_id[label] for label in test_labels],
        dtype=int,
    )

    # ---------------------------------------------------------
    # Dataset summary
    # ---------------------------------------------------------

    print("=" * 60)
    print("FUSED XGBOOST DATASET")
    print("=" * 60)

    print(f"Train samples : {len(train_X)}")

    print(f"Val samples   : {len(val_X)}")

    print(f"Test samples  : {len(test_X)}")

    print(f"Features      : {train_X.shape[1]}")

    print(f"Classes       : {classes}")

    # ---------------------------------------------------------
    # Verify feature dimensions
    # ---------------------------------------------------------

    if train_X.shape[1] != MICROPHONE_FEATURE_VECTOR_SIZE:
        raise ValueError("Unexpected train feature count: " f"{train_X.shape[1]}")

    if val_X.shape[1] != train_X.shape[1]:
        raise ValueError("Train/validation feature mismatch")

    if test_X.shape[1] != train_X.shape[1]:
        raise ValueError("Train/test feature mismatch")

    # ---------------------------------------------------------
    # Class-balanced sample weights
    # ---------------------------------------------------------

    class_counts = Counter(y_train.tolist())

    total_samples = len(y_train)
    number_of_classes = len(classes)

    class_weights = {
        class_id: (total_samples / (number_of_classes * count))
        for class_id, count in class_counts.items()
    }

    sample_weights = np.asarray(
        [class_weights[class_id] for class_id in y_train],
        dtype=float,
    )

    print()
    print("Training class distribution")
    print("-" * 60)

    for class_id, label in enumerate(classes):
        count = class_counts.get(
            class_id,
            0,
        )

        weight = class_weights.get(
            class_id,
            0.0,
        )

        print(f"{label:>3} : " f"{count:>4} samples | " f"weight={weight:.3f}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING XGBOOST V2")
    print("=" * 60)

    model.fit(
        train_X,
        y_train,
        sample_weight=sample_weights,
        eval_set=[
            (train_X, y_train),
            (val_X, y_val),
        ],
        verbose=False,
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(
        name: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        predictions = model.predict(X).astype(int)

        accuracy = accuracy_score(
            y,
            predictions,
        )

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            y,
            predictions,
        )

        print()
        print("=" * 60)
        print(f"{name.upper()} RESULTS")
        print("=" * 60)

        print(f"Accuracy          : " f"{accuracy:.4f}")

        print(f"Macro F1          : " f"{macro_f1:.4f}")

        print(f"Balanced Accuracy : " f"{balanced_accuracy:.4f}")

        print()

        print(
            classification_report(
                y,
                predictions,
                labels=list(range(len(classes))),
                target_names=classes,
                zero_division=0,
            )
        )

        print("Confusion matrix:")

        print(
            confusion_matrix(
                y,
                predictions,
                labels=list(range(len(classes))),
            )
        )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    evaluate(
        "Validation",
        val_X,
        y_val,
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    evaluate(
        "Test",
        test_X,
        y_test,
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(MODEL_PATH)

    # ---------------------------------------------------------
    # Save classes
    # ---------------------------------------------------------

    labels_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".labels.json")

    labels_path.write_text(
        json.dumps(
            {
                "classes": classes,
                "feature_count": len(feature_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Save feature schema
    # ---------------------------------------------------------

    feature_names_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".features.json")

    feature_names_path.write_text(
        json.dumps(
            {
                "features": feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Done
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"Model    : {MODEL_PATH}")

    print(f"Labels   : {labels_path}")

    print(f"Features : {feature_names_path}")


if __name__ == "__main__":
    main()

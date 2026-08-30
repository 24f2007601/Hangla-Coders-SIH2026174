"""Train and evaluate the XGBoost experiment-step classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from bas_assistant.features.extractor import FEATURE_VECTOR_SIZE

DATASET = Path("data/processed/synthetic_features.csv")
MODEL_PATH = Path("models/step_classifier.json")


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    # Load CSV without pandas.
    data = np.genfromtxt(
        DATASET,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    feature_names = data.dtype.names[:-1]
    label_name = data.dtype.names[-1]

    X = np.column_stack([data[name] for name in feature_names]).astype(float)
    labels = np.asarray(data[label_name], dtype=str)

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {sorted(set(labels))}")

    if X.shape[1] != FEATURE_VECTOR_SIZE:
        raise ValueError(f"Expected {FEATURE_VECTOR_SIZE} features, got {X.shape[1]}")

    classes = sorted(set(labels))
    class_to_id = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_id[label] for label in labels], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=42,
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    labels_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".labels.json")
    labels_path.write_text(
        json.dumps({"classes": classes}, indent=2),
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()

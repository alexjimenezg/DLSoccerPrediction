"""Train DeepMatch AI model for Premier League match prediction.

This script builds a neural network with team embeddings to predict match outcomes.
Uses chronological train/test split to ensure no temporal data leakage.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Input and output paths
INPUT_PATH = Path("data/features.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Model hyperparameters
EMBEDDING_DIM = 16
HIDDEN_DIM = 64
DROPOUT_RATE = 0.3
EPOCHS = 20
BATCH_SIZE = 32

# Train/test split ratio (80/20 chronological)
TRAIN_SPLIT_RATIO = 0.8


def load_features() -> pd.DataFrame:
    """Load the features CSV."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    try:
        data = pd.read_csv(INPUT_PATH)
    except Exception as exc:
        raise ValueError(f"Failed to read {INPUT_PATH}: {exc}") from exc

    return data


def identify_numerical_columns(data: pd.DataFrame) -> list[str]:
    """Identify numerical columns (excluding team identifiers and target)."""

    exclude_cols = {"Date", "Season", "HomeTeam", "AwayTeam", "target"}
    numerical_cols = [col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])]

    return numerical_cols


def prepare_data(data: pd.DataFrame) -> tuple:
    """Prepare and encode features and target.

    Returns:
        (X_numerical, X_home_team, X_away_team, y, num_cols, home_encoder, away_encoder, target_encoder)
    """

    num_cols = identify_numerical_columns(data)
    print(f"Numerical columns: {num_cols}")

    # Encode target (H, D, A -> 0, 1, 2)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data["target"])
    print(f"Target classes: {target_encoder.classes_}")

    # Encode home team
    home_encoder = LabelEncoder()
    X_home_team = home_encoder.fit_transform(data["HomeTeam"])
    num_teams = len(home_encoder.classes_)
    print(f"Number of unique teams: {num_teams}")

    # Encode away team (use same encoder to ensure consistency)
    away_encoder = LabelEncoder()
    away_encoder.classes_ = home_encoder.classes_
    X_away_team = away_encoder.transform(data["AwayTeam"])

    # Extract numerical features
    X_numerical = data[num_cols].values

    return X_numerical, X_home_team, X_away_team, y, num_cols, home_encoder, away_encoder, target_encoder


def chronological_train_test_split(
    X_numerical: np.ndarray,
    X_home_team: np.ndarray,
    X_away_team: np.ndarray,
    y: np.ndarray,
    split_ratio: float = 0.8,
) -> tuple:
    """Split data chronologically (no shuffling) to prevent data leakage.

    Args:
        X_numerical: Numerical features
        X_home_team: Encoded home team
        X_away_team: Encoded away team
        y: Target values
        split_ratio: Proportion of data for training

    Returns:
        (X_train_num, X_train_home, X_train_away, y_train,
         X_test_num, X_test_home, X_test_away, y_test)
    """

    split_idx = int(len(X_numerical) * split_ratio)

    X_train_num = X_numerical[:split_idx]
    X_test_num = X_numerical[split_idx:]

    X_train_home = X_home_team[:split_idx]
    X_test_home = X_home_team[split_idx:]

    X_train_away = X_away_team[:split_idx]
    X_test_away = X_away_team[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"\nTrain set size: {len(y_train)} ({len(y_train) / len(y) * 100:.1f}%)")
    print(f"Test set size: {len(y_test)} ({len(y_test) / len(y) * 100:.1f}%)")

    return X_train_num, X_train_home, X_train_away, y_train, X_test_num, X_test_home, X_test_away, y_test


def scale_features(
    X_train_num: np.ndarray,
    X_test_num: np.ndarray,
) -> tuple:
    """Scale numerical features using StandardScaler fit on training data only.

    Args:
        X_train_num: Training numerical features
        X_test_num: Test numerical features

    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    print(f"\nNumerical features scaled using training data statistics.")

    return X_train_scaled, X_test_scaled, scaler


def build_model(
    num_numerical_features: int,
    num_teams: int,
) -> keras.Model:
    """Build a neural network with team embeddings and numerical features.

    Args:
        num_numerical_features: Number of numerical input features
        num_teams: Number of unique teams

    Returns:
        Compiled Keras model
    """

    # Numerical input
    numerical_input = layers.Input(shape=(num_numerical_features,), name="numerical_input")
    numerical_branch = layers.Dense(32, activation="relu")(numerical_input)
    numerical_branch = layers.Dropout(DROPOUT_RATE)(numerical_branch)

    # Home team embedding
    home_team_input = layers.Input(shape=(1,), name="home_team_input")
    home_embedding = layers.Embedding(num_teams, EMBEDDING_DIM, input_length=1)(home_team_input)
    home_embedding = layers.Flatten()(home_embedding)

    # Away team embedding
    away_team_input = layers.Input(shape=(1,), name="away_team_input")
    away_embedding = layers.Embedding(num_teams, EMBEDDING_DIM, input_length=1)(away_team_input)
    away_embedding = layers.Flatten()(away_embedding)

    # Concatenate all branches
    concatenated = layers.Concatenate()([numerical_branch, home_embedding, away_embedding])

    # Dense layers
    hidden = layers.Dense(HIDDEN_DIM, activation="relu")(concatenated)
    hidden = layers.Dropout(DROPOUT_RATE)(hidden)
    hidden = layers.Dense(HIDDEN_DIM // 2, activation="relu")(hidden)
    hidden = layers.Dropout(DROPOUT_RATE)(hidden)

    # Output layer (3 classes: H, D, A)
    output = layers.Dense(3, activation="softmax", name="output")(hidden)

    # Build model
    model = keras.Model(
        inputs=[numerical_input, home_team_input, away_team_input],
        outputs=output,
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
    model: keras.Model,
    X_train_num: np.ndarray,
    X_train_home: np.ndarray,
    X_train_away: np.ndarray,
    y_train: np.ndarray,
    X_val_num: np.ndarray,
    X_val_home: np.ndarray,
    X_val_away: np.ndarray,
    y_val: np.ndarray,
) -> keras.callbacks.History:
    """Train the model.

    Args:
        model: Compiled Keras model
        X_train_num: Training numerical features
        X_train_home: Training home team encodings
        X_train_away: Training away team encodings
        y_train: Training targets
        X_val_num: Validation numerical features
        X_val_home: Validation home team encodings
        X_val_away: Validation away team encodings
        y_val: Validation targets

    Returns:
        Training history
    """

    history = model.fit(
        [X_train_num, X_train_home, X_train_away],
        y_train,
        validation_data=([X_val_num, X_val_home, X_val_away], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    return history


def evaluate_model(
    model: keras.Model,
    X_test_num: np.ndarray,
    X_test_home: np.ndarray,
    X_test_away: np.ndarray,
    y_test: np.ndarray,
    target_encoder: LabelEncoder,
) -> str:
    """Evaluate model on test set and generate metrics report.

    Args:
        model: Trained Keras model
        X_test_num: Test numerical features
        X_test_home: Test home team encodings
        X_test_away: Test away team encodings
        y_test: Test targets
        target_encoder: LabelEncoder for target classes

    Returns:
        Formatted metrics report as string
    """

    from sklearn.metrics import classification_report, confusion_matrix

    # Get predictions
    y_pred_proba = model.predict([X_test_num, X_test_home, X_test_away], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()

    # Get class names
    class_names = target_encoder.classes_

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_str = f"Confusion Matrix:\n{cm}\n"

    # Classification report
    clf_report = classification_report(y_test, y_pred, target_names=class_names)

    # Format report
    report = f"Model Evaluation Results\n"
    report += "=" * 50 + "\n"
    report += f"Accuracy: {accuracy:.4f}\n"
    report += "\n"
    report += cm_str
    report += "\n"
    report += "Classification Report:\n"
    report += clf_report

    return report


def save_artifacts(
    model: keras.Model,
    scaler: StandardScaler,
    home_encoder: LabelEncoder,
    num_cols: list[str],
    metrics_report: str,
) -> None:
    """Save trained model and preprocessing artifacts.

    Args:
        model: Trained Keras model
        scaler: StandardScaler instance
        home_encoder: LabelEncoder for teams
        num_cols: List of numerical column names
        metrics_report: Model metrics report string
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "deepmatch_model.keras"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Save team encoder
    encoder_path = MODELS_DIR / "team_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(home_encoder, f)
    print(f"Team encoder saved to {encoder_path}")

    # Save numerical columns
    cols_path = MODELS_DIR / "num_cols.pkl"
    with open(cols_path, "wb") as f:
        pickle.dump(num_cols, f)
    print(f"Numerical columns saved to {cols_path}")

    # Save metrics report
    metrics_path = REPORTS_DIR / "model_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(metrics_report)
    print(f"Metrics report saved to {metrics_path}")


def main() -> int:
    """Load data, train model, and save artifacts."""

    try:
        print("Loading features...")
        data = load_features()
        print(f"Loaded {len(data)} samples")

        print("\nPreparing data...")
        X_numerical, X_home_team, X_away_team, y, num_cols, home_encoder, away_encoder, target_encoder = prepare_data(data)

        print("\nSplitting data chronologically...")
        X_train_num, X_train_home, X_train_away, y_train, X_test_num, X_test_home, X_test_away, y_test = (
            chronological_train_test_split(X_numerical, X_home_team, X_away_team, y, TRAIN_SPLIT_RATIO)
        )

        print("\nScaling numerical features...")
        X_train_num_scaled, X_test_num_scaled, scaler = scale_features(X_train_num, X_test_num)

        # Further split training into train/validation
        val_split_idx = int(len(X_train_num_scaled) * 0.8)
        X_train_num_final = X_train_num_scaled[:val_split_idx]
        X_train_home_final = X_train_home[:val_split_idx]
        X_train_away_final = X_train_away[:val_split_idx]
        y_train_final = y_train[:val_split_idx]

        X_val_num = X_train_num_scaled[val_split_idx:]
        X_val_home = X_train_home[val_split_idx:]
        X_val_away = X_train_away[val_split_idx:]
        y_val = y_train[val_split_idx:]

        print(f"Final train set: {len(y_train_final)}")
        print(f"Validation set: {len(y_val)}")

        print("\nBuilding model...")
        num_teams = len(home_encoder.classes_)
        model = build_model(num_numerical_features=len(num_cols), num_teams=num_teams)
        print(model.summary())

        print(f"\nTraining for {EPOCHS} epochs...")
        train_model(
            model,
            X_train_num_final,
            X_train_home_final,
            X_train_away_final,
            y_train_final,
            X_val_num,
            X_val_home,
            X_val_away,
            y_val,
        )

        print("\nEvaluating on test set...")
        metrics_report = evaluate_model(model, X_test_num_scaled, X_test_home, X_test_away, y_test, target_encoder)
        print(metrics_report)

        print("\nSaving artifacts...")
        save_artifacts(model, scaler, home_encoder, num_cols, metrics_report)

        return 0

    except Exception as exc:
        print(f"Error during model training: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

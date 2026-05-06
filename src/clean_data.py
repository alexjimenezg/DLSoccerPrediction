"""Clean and validate Premier League match data for DeepMatch AI.

This script loads the raw combined dataset, filters to essential columns,
validates data types, removes incomplete or invalid rows, deduplicates,
and saves a clean dataset ready for feature engineering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Input and output paths
INPUT_PATH = Path("data/raw_matches.csv")
OUTPUT_PATH = Path("data/clean_matches.csv")

# Columns to keep for modeling
REQUIRED_COLUMNS = ("Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A")

# Valid final result values
VALID_FTR = {"H", "D", "A"}


def load_raw_data() -> pd.DataFrame:
    """Load the raw matches CSV and return as a DataFrame."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    try:
        raw = pd.read_csv(INPUT_PATH)
    except Exception as exc:
        raise ValueError(f"Failed to read {INPUT_PATH}: {exc}") from exc

    return raw


def select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the essential columns for modeling."""

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Input dataset missing required columns: {missing_columns}")

    return frame[list(REQUIRED_COLUMNS)].copy()


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse the Date column with dayfirst=True for European date format."""

    try:
        frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    except Exception as exc:
        raise ValueError(f"Failed to parse dates: {exc}") from exc

    return frame


def remove_invalid_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with valid final result values (H, D, A)."""

    before_count = len(frame)
    frame = frame[frame["FTR"].isin(VALID_FTR)].copy()
    removed = before_count - len(frame)

    if removed > 0:
        print(f"Removed {removed} rows with invalid FTR values.")

    return frame


def remove_missing_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values in critical columns."""

    critical_columns = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A")
    before_count = len(frame)

    # Keep only rows where all critical columns are non-null.
    frame = frame.dropna(subset=critical_columns)
    removed = before_count - len(frame)

    if removed > 0:
        print(f"Removed {removed} rows with missing values in critical columns.")

    return frame


def remove_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate matches (same date, teams, and result)."""

    duplicate_cols = ("Date", "HomeTeam", "AwayTeam", "FTR")
    before_count = len(frame)

    frame = frame.drop_duplicates(subset=duplicate_cols, keep="first")
    removed = before_count - len(frame)

    if removed > 0:
        print(f"Removed {removed} duplicate match rows.")

    return frame


def sort_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort the dataset chronologically by match date."""

    return frame.sort_values("Date").reset_index(drop=True)


def validate_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure goal and odds columns are numeric."""

    numeric_columns = ("FTHG", "FTAG", "B365H", "B365D", "B365A")
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Remove any rows where numeric conversion resulted in NaN
    before_count = len(frame)
    frame = frame.dropna(subset=numeric_columns)
    removed = before_count - len(frame)

    if removed > 0:
        print(f"Removed {removed} rows with non-numeric goal or odds values.")

    return frame


def report_missing_values(frame: pd.DataFrame) -> None:
    """Print a summary of missing values in the clean dataset."""

    missing = frame.isnull().sum()
    if missing.any():
        print("\nMissing values in clean dataset:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values in clean dataset.")


def main() -> int:
    """Load, clean, and save the matches dataset."""

    try:
        # Load raw data
        print("Loading raw matches data...")
        raw = load_raw_data()
        raw_shape = raw.shape
        print(f"Raw dataset shape: {raw_shape}")

        # Clean and transform
        print("\nCleaning dataset...")
        clean = select_columns(raw)
        clean = parse_dates(clean)
        clean = validate_numeric_columns(clean)
        clean = remove_invalid_results(clean)
        clean = remove_missing_data(clean)
        clean = remove_duplicates(clean)
        clean = sort_by_date(clean)

        clean_shape = clean.shape
        print(f"Clean dataset shape: {clean_shape}")

        # Report changes
        rows_removed = raw_shape[0] - clean_shape[0]
        rows_kept = clean_shape[0]
        print(f"\nRows kept: {rows_kept} (removed {rows_removed} invalid/duplicate rows)")

        # Report missing values
        report_missing_values(clean)

        # Save clean dataset
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved clean dataset to {OUTPUT_PATH}")

        # Preview
        print("\nFirst rows of clean dataset:")
        print(clean.head())

        return 0

    except Exception as exc:
        print(f"Error during data cleaning: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

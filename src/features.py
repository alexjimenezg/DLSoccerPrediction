"""Feature engineering for DeepMatch AI.

This script loads clean matches, sorts chronologically, and builds features
using only historical data for each match. Critical: Features are computed
using only previous matches for each team to prevent data leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Input and output paths
INPUT_PATH = Path("data/clean_matches.csv")
OUTPUT_PATH = Path("data/features.csv")

# Number of previous matches to use for form calculation
FORM_WINDOW = 5

# Feature columns to generate
FEATURE_COLUMNS = (
    "Date",
    "Season",
    "HomeTeam",
    "AwayTeam",
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
    "form_points_diff",
    "form_goals_for_diff",
    "form_goals_against_diff",
    "B365H",
    "B365D",
    "B365A",
)

TARGET_COLUMN = "target"


def load_clean_data() -> pd.DataFrame:
    """Load and sort clean matches chronologically."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    try:
        data = pd.read_csv(INPUT_PATH)
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.sort_values("Date").reset_index(drop=True)
    except Exception as exc:
        raise ValueError(f"Failed to read {INPUT_PATH}: {exc}") from exc

    return data


def calculate_team_form(
    team: str,
    match_index: int,
    data: pd.DataFrame,
    is_home: bool,
) -> tuple[float, float, float]:
    """Calculate form statistics for a team using only matches before the given index.

    Args:
        team: Team name
        match_index: Current match index (only use data before this)
        data: Full dataset sorted chronologically
        is_home: True if calculating for home team, False for away

    Returns:
        (avg_goals_for, avg_goals_against, avg_points)

    Note: This function ensures no data leakage by only looking at matches with index < match_index.
    """

    # Get all matches for this team before the current match
    if is_home:
        previous_matches = data[(data.index < match_index) & (data["HomeTeam"] == team)].tail(FORM_WINDOW)
    else:
        previous_matches = data[(data.index < match_index) & (data["AwayTeam"] == team)].tail(FORM_WINDOW)

    if len(previous_matches) == 0:
        # No previous matches available, return NaN
        return float("nan"), float("nan"), float("nan")

    if is_home:
        goals_for = previous_matches["FTHG"].mean()
        goals_against = previous_matches["FTAG"].mean()
        # Calculate points: H=3, D=1, A=0
        points = ((previous_matches["FTR"] == "H").sum() * 3 + (previous_matches["FTR"] == "D").sum()) / len(
            previous_matches
        )
    else:
        goals_for = previous_matches["FTAG"].mean()
        goals_against = previous_matches["FTHG"].mean()
        # Calculate points: A=3, D=1, H=0
        points = ((previous_matches["FTR"] == "A").sum() * 3 + (previous_matches["FTR"] == "D").sum()) / len(
            previous_matches
        )

    return goals_for, goals_against, points


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features for all matches in chronological order.

    Critical: For each match, features are calculated using ONLY previous matches
    to ensure no data leakage into model training.
    """

    rows = []

    for idx in range(len(data)):
        match = data.iloc[idx]

        # Get form stats for home team using only previous matches
        home_gf, home_ga, home_pts = calculate_team_form(
            match["HomeTeam"],
            idx,
            data,
            is_home=True,
        )

        # Get form stats for away team using only previous matches
        away_gf, away_ga, away_pts = calculate_team_form(
            match["AwayTeam"],
            idx,
            data,
            is_home=False,
        )

        # Skip matches where we don't have sufficient history for both teams
        if pd.isna(home_gf) or pd.isna(away_gf):
            continue

        # Calculate form differences
        form_points_diff = home_pts - away_pts
        form_goals_for_diff = home_gf - away_gf
        form_goals_against_diff = home_ga - away_ga

        # Build feature row
        row = {
            "Date": match["Date"],
            "Season": match["Season"],
            "HomeTeam": match["HomeTeam"],
            "AwayTeam": match["AwayTeam"],
            "home_avg_goals_for": home_gf,
            "home_avg_goals_against": home_ga,
            "home_avg_points": home_pts,
            "away_avg_goals_for": away_gf,
            "away_avg_goals_against": away_ga,
            "away_avg_points": away_pts,
            "form_points_diff": form_points_diff,
            "form_goals_for_diff": form_goals_for_diff,
            "form_goals_against_diff": form_goals_against_diff,
            "B365H": match["B365H"],
            "B365D": match["B365D"],
            "B365A": match["B365A"],
            "target": match["FTR"],
        }

        rows.append(row)

    # Convert to DataFrame
    features_df = pd.DataFrame(rows)

    return features_df


def main() -> int:
    """Build and save features for model training."""

    try:
        print("Loading clean matches...")
        data = load_clean_data()
        print(f"Loaded {len(data)} matches")

        print(f"\nBuilding features (using form window: {FORM_WINDOW} matches)...")
        features = build_features(data)

        print(f"\nFeature dataset shape: {features.shape}")
        print(f"Columns: {list(features.columns)}")

        # Report target distribution
        target_dist = features["target"].value_counts().sort_index()
        target_pct = (target_dist / target_dist.sum() * 100).round(2)
        print(f"\nTarget distribution:")
        print(f"  H (Home Win): {target_dist['H']} ({target_pct['H']}%)")
        print(f"  D (Draw): {target_dist['D']} ({target_pct['D']}%)")
        print(f"  A (Away Win): {target_dist['A']} ({target_pct['A']}%)")

        # Save features
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(OUTPUT_PATH, index=False)
        print(f"\nFeatures saved to {OUTPUT_PATH}")

        # Preview
        print("\nFirst rows of feature set:")
        print(features.head())

        return 0

    except Exception as exc:
        print(f"Error during feature engineering: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

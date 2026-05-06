"""Exploratory Data Analysis for DeepMatch AI.

This script loads the clean matches dataset, generates key visualizations,
computes summary statistics, and prints insights about the data distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Input and output paths
INPUT_PATH = Path("data/clean_matches.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_PATH = REPORTS_DIR / "eda_summary.csv"

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_clean_data() -> pd.DataFrame:
    """Load the clean matches dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    try:
        data = pd.read_csv(INPUT_PATH)
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    except Exception as exc:
        raise ValueError(f"Failed to read {INPUT_PATH}: {exc}") from exc

    return data


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(data: pd.DataFrame) -> None:
    """Plot the distribution of match outcomes (Home, Draw, Away)."""

    ftr_counts = data["FTR"].value_counts().sort_index()
    ftr_pct = (ftr_counts / ftr_counts.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"H": "#2ecc71", "D": "#f39c12", "A": "#e74c3c"}
    bars = ax.bar(
        ftr_counts.index,
        ftr_counts.values,
        color=[colors.get(x, "#95a5a6") for x in ftr_counts.index],
        alpha=0.8,
        edgecolor="black",
    )

    # Add percentage labels on bars
    for bar, pct in zip(bars, ftr_pct):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{pct}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("Match Outcome", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Match Outcomes (Home Win, Draw, Away Win)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Home Win", "Draw", "Away Win"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTarget Distribution:\n{ftr_counts}\n{ftr_pct}")


def plot_goals_distribution(data: pd.DataFrame) -> None:
    """Plot average home vs away goals."""

    avg_home_goals = data["FTHG"].mean()
    avg_away_goals = data["FTAG"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparing averages
    categories = ["Home Goals", "Away Goals"]
    values = [avg_home_goals, avg_away_goals]
    colors_bar = ["#3498db", "#e74c3c"]
    ax1.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("Average Goals", fontsize=12, fontweight="bold")
    ax1.set_title("Average Goals: Home vs Away", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, max(values) * 1.2)

    for i, v in enumerate(values):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")

    # Distribution histograms
    ax2.hist(data["FTHG"], bins=15, alpha=0.6, label="Home Goals", color="#3498db", edgecolor="black")
    ax2.hist(data["FTAG"], bins=15, alpha=0.6, label="Away Goals", color="#e74c3c", edgecolor="black")
    ax2.set_xlabel("Goals", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax2.set_title("Distribution of Home vs Away Goals", fontsize=14, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "goals_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nAverage Goals:\n  Home: {avg_home_goals:.3f}\n  Away: {avg_away_goals:.3f}")


def plot_results_by_season(data: pd.DataFrame) -> None:
    """Plot match outcomes aggregated by season."""

    season_ftr = pd.crosstab(data["Season"], data["FTR"])
    season_ftr_pct = season_ftr.div(season_ftr.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    season_ftr_pct.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
        edgecolor="black",
    )

    ax.set_xlabel("Season", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Match Outcome Distribution by Season", fontsize=14, fontweight="bold")
    ax.legend(["Home Win", "Draw", "Away Win"], loc="upper right")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "results_by_season.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nResults by Season (%):\n{season_ftr_pct.round(2)}")


def plot_top_teams_by_points(data: pd.DataFrame) -> None:
    """Plot top 10 teams by average points per match."""

    # Calculate points: H=3, D=1, A=0
    def calculate_points(home_wins, draws, away_losses):
        return home_wins * 3 + draws * 1

    # Home stats
    home_stats = data.groupby("HomeTeam").agg(
        home_wins=("FTR", lambda x: (x == "H").sum()),
        home_draws=("FTR", lambda x: (x == "D").sum()),
        home_matches=("FTR", "count"),
    )

    home_stats["home_points"] = home_stats["home_wins"] * 3 + home_stats["home_draws"] * 1

    # Away stats
    away_stats = data.groupby("AwayTeam").agg(
        away_wins=("FTR", lambda x: (x == "A").sum()),
        away_draws=("FTR", lambda x: (x == "D").sum()),
        away_matches=("FTR", "count"),
    )

    away_stats["away_points"] = away_stats["away_wins"] * 3 + away_stats["away_draws"] * 1

    # Combine and calculate average points per match
    combined_stats = pd.DataFrame(
        {
            "home_matches": home_stats["home_matches"],
            "home_points": home_stats["home_points"],
            "away_matches": away_stats["away_matches"],
            "away_points": away_stats["away_points"],
        }
    ).fillna(0)

    combined_stats["total_matches"] = combined_stats["home_matches"] + combined_stats["away_matches"]
    combined_stats["total_points"] = combined_stats["home_points"] + combined_stats["away_points"]
    combined_stats["avg_points_per_match"] = combined_stats["total_points"] / combined_stats["total_matches"]

    top_teams = combined_stats.nlargest(10, "avg_points_per_match")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top_teams)), top_teams["avg_points_per_match"], color="#3498db", alpha=0.8, edgecolor="black")
    ax.set_yticks(range(len(top_teams)))
    ax.set_yticklabels(top_teams.index)
    ax.set_xlabel("Average Points per Match", fontsize=12, fontweight="bold")
    ax.set_title("Top 10 Teams by Average Points per Match", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for i, v in enumerate(top_teams["avg_points_per_match"]):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_teams_by_points.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTop 10 Teams by Average Points per Match:\n{top_teams[['total_matches', 'total_points', 'avg_points_per_match']].round(3)}")

    return top_teams


def plot_odds_distribution(data: pd.DataFrame) -> None:
    """Plot distribution of Bet365 odds for all outcomes."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    odds_columns = [("B365H", "Home Win Odds"), ("B365D", "Draw Odds"), ("B365A", "Away Win Odds")]
    colors_odds = ["#2ecc71", "#f39c12", "#e74c3c"]

    for ax, (col, label), color in zip(axes, odds_columns, colors_odds):
        ax.hist(data[col], bins=40, color=color, alpha=0.8, edgecolor="black")
        ax.set_xlabel(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title(f"Distribution of {label}", fontsize=12, fontweight="bold")

        mean_odds = data[col].mean()
        ax.axvline(mean_odds, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_odds:.2f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "odds_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nBet365 Odds Statistics:")
    print(f"  Home Win (B365H): Mean={data['B365H'].mean():.3f}, Std={data['B365H'].std():.3f}")
    print(f"  Draw (B365D): Mean={data['B365D'].mean():.3f}, Std={data['B365D'].std():.3f}")
    print(f"  Away Win (B365A): Mean={data['B365A'].mean():.3f}, Std={data['B365A'].std():.3f}")


def create_summary_table(data: pd.DataFrame, top_teams: pd.DataFrame) -> None:
    """Create and save a summary statistics table."""

    summary_data = {
        "Metric": [
            "Total Matches",
            "Date Range",
            "Number of Seasons",
            "Number of Teams",
            "Home Win %",
            "Draw %",
            "Away Win %",
            "Avg Home Goals",
            "Avg Away Goals",
            "Avg Home Odds",
            "Avg Draw Odds",
            "Avg Away Odds",
        ],
        "Value": [
            len(data),
            f"{data['Date'].min().date()} to {data['Date'].max().date()}",
            data["Season"].nunique(),
            data["HomeTeam"].nunique(),
            f"{(data['FTR'] == 'H').sum() / len(data) * 100:.2f}%",
            f"{(data['FTR'] == 'D').sum() / len(data) * 100:.2f}%",
            f"{(data['FTR'] == 'A').sum() / len(data) * 100:.2f}%",
            f"{data['FTHG'].mean():.3f}",
            f"{data['FTAG'].mean():.3f}",
            f"{data['B365H'].mean():.3f}",
            f"{data['B365D'].mean():.3f}",
            f"{data['B365A'].mean():.3f}",
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"\n=== EDA Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to {SUMMARY_PATH}")


def main() -> int:
    """Run the full exploratory data analysis."""

    try:
        print("Loading clean matches data...")
        data = load_clean_data()
        print(f"Loaded {len(data)} matches from {INPUT_PATH}")

        # Create output directories
        ensure_output_dirs()

        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_target_distribution(data)
        plot_goals_distribution(data)
        plot_results_by_season(data)
        top_teams = plot_top_teams_by_points(data)
        plot_odds_distribution(data)

        # Create summary table
        print("\nCreating summary statistics...")
        create_summary_table(data, top_teams)

        print(f"\nAll figures saved to {FIGURES_DIR}")
        print("EDA complete!")

        return 0

    except Exception as exc:
        print(f"Error during EDA: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

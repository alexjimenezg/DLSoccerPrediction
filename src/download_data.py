"""Download and combine Premier League match data from Football-Data.co.uk.

This script downloads E0 CSV files for the requested seasons, keeps only the
files that load successfully, adds a Season column, concatenates them, and
writes the combined dataset to data/raw_matches.csv.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODE = "E0"
OUTPUT_PATH = Path("data/raw_matches.csv")
LATEST_SEASON_START = 2025
OLDEST_SEASON_START = 1992
MAX_CONSECUTIVE_MISSES = 3


def season_folder(start_year: int) -> str:
    """Convert a season start year into the Football-Data folder code."""

    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Format a season label such as 2019/20 or 2024/25."""

    return f"{start_year}/{(start_year + 1) % 100:02d}"


def clean_column_name(column: str) -> str:
    """Remove BOM artifacts that can appear in Football-Data headers."""

    return column.replace("\ufeff", "").replace("ï»¿", "")


def season_url(season_folder: str) -> str:
    return f"{BASE_URL}/{season_folder}/{LEAGUE_CODE}.csv"


def parse_csv(payload: bytes) -> pd.DataFrame:
    """Parse a Football-Data CSV with a strict pass and a forgiving fallback."""

    read_attempts = (
        {"encoding": "latin-1"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1", "engine": "python", "on_bad_lines": "skip"},
        {"encoding": "utf-8-sig", "engine": "python", "on_bad_lines": "skip"},
    )

    last_error: Exception | None = None
    for kwargs in read_attempts:
        try:
            return pd.read_csv(io.BytesIO(payload), **kwargs)
        except Exception as exc:
            last_error = exc

    raise last_error if last_error is not None else ValueError("Unable to parse CSV payload")


def download_season(session: requests.Session, season_label: str, season_folder: str) -> pd.DataFrame | None:
    """Download one season and return a DataFrame if the CSV loads cleanly."""

    url = season_url(season_folder)

    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Skipping {season_label}: download failed ({exc})")
        return None

    try:
        # Football-Data CSVs are usually readable with latin-1 and sometimes utf-8.
        try:
            frame = pd.read_csv(io.BytesIO(response.content), encoding="latin-1")
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
    except Exception as exc:
        print(f"Skipping {season_label}: CSV parse failed ({exc})")
        return None

    if frame.empty:
        print(f"Skipping {season_label}: no rows found")
        return None

    frame.columns = [clean_column_name(column) for column in frame.columns]

    frame["Season"] = season_label
    return frame


def parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides for the archive sweep."""

    parser = argparse.ArgumentParser(description="Download and combine Football-Data.co.uk E0 match data.")
    parser.add_argument(
        "--latest-season-start",
        type=int,
        default=LATEST_SEASON_START,
        help="Newest season start year to try, for example 2025 for 2025/26.",
    )
    parser.add_argument(
        "--oldest-season-start",
        type=int,
        default=OLDEST_SEASON_START,
        help="Oldest season start year to try, for example 1992 for 1992/93.",
    )
    parser.add_argument(
        "--max-consecutive-misses",
        type=int,
        default=MAX_CONSECUTIVE_MISSES,
        help="Stop after this many consecutive missing seasons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for the combined CSV output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure the output directory exists before writing the combined file.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "DeepMatchAI/1.0"})

    frames: list[pd.DataFrame] = []
    missed_seasons = 0
    for start_year in range(args.latest_season_start, args.oldest_season_start - 1, -1):
        current_season_label = season_label(start_year)
        current_season_folder = season_folder(start_year)
        frame = download_season(session, current_season_label, current_season_folder)
        if frame is not None:
            frames.append(frame)
            missed_seasons = 0
            continue

        missed_seasons += 1
        if missed_seasons >= args.max_consecutive_misses:
            print("Stopping after consecutive missing seasons; archive appears exhausted.")
            break

    if not frames:
        print("No valid datasets were downloaded.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)

    print(f"Saved combined dataset to {args.output}")
    print(f"Dataset shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print("First rows:")
    print(combined.head())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
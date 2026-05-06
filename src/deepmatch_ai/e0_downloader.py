"""Download Premier League CSV files from Football-Data.co.uk.

Usage:
    PYTHONPATH=src python -m deepmatch_ai.e0_downloader --seasons 2324 2223 2122
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODE = "E0"
DEFAULT_SEASONS = ("2324", "2223", "2122", "2021", "1920")
REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTR")


@dataclass(frozen=True)
class DownloadRecord:
    season: str
    url: str
    output_path: str
    status: str
    http_status: str
    bytes: str
    sha256: str
    row_count: str
    fetched_at: str
    message: str


def build_url(season: str) -> str:
    return f"{BASE_URL}/{season}/{LEAGUE_CODE}.csv"


def build_output_path(base_dir: Path, season: str) -> Path:
    return base_dir / season / f"{LEAGUE_CODE}.csv"


def build_manifest_path(base_dir: Path) -> Path:
    return base_dir / "manifest.csv"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_csv_bytes(payload: bytes) -> tuple[bool, int, str]:
    if not payload.strip():
        return False, 0, "empty payload"

    if payload.lstrip().startswith(b"<"):
        return False, 0, "response looks like HTML"

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False, 0, "payload is not valid utf-8"

    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        return False, 0, "missing CSV header"

    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        return False, 0, f"missing required columns: {', '.join(missing)}"

    row_count = sum(1 for _ in reader)
    if row_count <= 0:
        return False, 0, "no data rows found"

    return True, row_count, "ok"


def read_existing_file(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_manifest_row(manifest_path: Path, record: DownloadRecord) -> None:
    ensure_parent(manifest_path)
    fieldnames = list(record.__dataclass_fields__.keys())
    file_exists = manifest_path.exists()

    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(record))


def write_file(path: Path, payload: bytes) -> None:
    ensure_parent(path)
    path.write_bytes(payload)


def download_season(session: requests.Session, season: str, output_path: Path, force: bool) -> DownloadRecord:
    url = build_url(season)
    fetched_at = now_iso()

    if output_path.exists() and not force:
        existing_payload = read_existing_file(output_path)
        if existing_payload is not None:
            valid, row_count, message = validate_csv_bytes(existing_payload)
            if valid:
                return DownloadRecord(
                    season=season,
                    url=url,
                    output_path=str(output_path),
                    status="skipped-existing",
                    http_status="",
                    bytes=str(len(existing_payload)),
                    sha256=sha256_bytes(existing_payload),
                    row_count=str(row_count),
                    fetched_at=fetched_at,
                    message=message,
                )

    try:
        response = session.get(url, timeout=60)
    except requests.RequestException as exc:
        return DownloadRecord(
            season=season,
            url=url,
            output_path=str(output_path),
            status="failed",
            http_status="",
            bytes="0",
            sha256="",
            row_count="0",
            fetched_at=fetched_at,
            message=str(exc),
        )

    if response.status_code != 200:
        return DownloadRecord(
            season=season,
            url=url,
            output_path=str(output_path),
            status="failed",
            http_status=str(response.status_code),
            bytes=str(len(response.content)),
            sha256="",
            row_count="0",
            fetched_at=fetched_at,
            message="unexpected HTTP status",
        )

    payload = response.content
    valid, row_count, message = validate_csv_bytes(payload)
    if not valid:
        return DownloadRecord(
            season=season,
            url=url,
            output_path=str(output_path),
            status="failed",
            http_status=str(response.status_code),
            bytes=str(len(payload)),
            sha256=sha256_bytes(payload),
            row_count=str(row_count),
            fetched_at=fetched_at,
            message=message,
        )

    write_file(output_path, payload)
    return DownloadRecord(
        season=season,
        url=url,
        output_path=str(output_path),
        status="downloaded",
        http_status=str(response.status_code),
        bytes=str(len(payload)),
        sha256=sha256_bytes(payload),
        row_count=str(row_count),
        fetched_at=fetched_at,
        message=message,
    )


def download_seasons(seasons: Iterable[str], output_dir: Path, force: bool) -> list[DownloadRecord]:
    session = requests.Session()
    session.headers.update({"User-Agent": "DeepMatchAI/1.0"})

    records: list[DownloadRecord] = []
    manifest_path = build_manifest_path(output_dir)
    for season in seasons:
        output_path = build_output_path(output_dir, season)
        record = download_season(session, season, output_path, force)
        records.append(record)
        write_manifest_row(manifest_path, record)
        print(f"{season}: {record.status} ({record.message})")

    return records


def summarize(records: Iterable[DownloadRecord]) -> tuple[int, int, int]:
    downloaded = sum(1 for record in records if record.status == "downloaded")
    skipped = sum(1 for record in records if record.status == "skipped-existing")
    failed = sum(1 for record in records if record.status == "failed")
    return downloaded, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Football-Data.co.uk Premier League E0 CSVs.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="Season folders to download, for example 2324 2223 2122.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/football-data/epl/E0",
        help="Directory where raw CSV files and the manifest will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if a valid local copy already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    records = download_seasons(args.seasons, output_dir, args.force)
    downloaded, skipped, failed = summarize(records)
    print(f"Summary: downloaded={downloaded} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

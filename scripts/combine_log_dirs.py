#!/usr/bin/env python3
"""Combine multiple stag-hunt log directories into a fresh output subdirectory.

This script streams CSV rows from source directories and never modifies source
files. The merged result is written into a brand-new subdirectory under
``--logs-root``.

Usage:
    python scripts/combine_log_dirs.py \
      --logs-root logs \
      --output-subdir all_combination_20260317 \
      sweep_med amina/sweep_ernie amina/sweep_kimi
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PENDING_POINTS_SUFFIX = "_sweep_points_pending.csv"


class SQLiteDeduper:
    """Disk-backed deduper to avoid loading large key sets into memory."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = OFF")
        self.conn.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
        self.cur = self.conn.cursor()
        self.pending_writes = 0

    def seen(self, digest: str) -> bool:
        self.cur.execute("INSERT OR IGNORE INTO seen (digest) VALUES (?)", (digest,))
        inserted = self.cur.rowcount > 0
        self.pending_writes += 1
        if self.pending_writes >= 50_000:
            self.conn.commit()
            self.pending_writes = 0
        return not inserted

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
        self.db_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    default_subdir = f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(
        description=(
            "Combine CSV logs from multiple directories into one new output "
            "subdirectory."
        )
    )
    parser.add_argument(
        "source_dirs",
        nargs="+",
        help=(
            "Source directories to combine (absolute paths or paths relative to "
            "--logs-root)."
        ),
    )
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Base logs directory (default: logs).",
    )
    parser.add_argument(
        "--output-subdir",
        default=default_subdir,
        help=(
            "Name of output subdirectory under --logs-root. Must not already "
            "exist. Default: timestamped name."
        ),
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable exact-row deduplication.",
    )
    parser.add_argument(
        "--include-pending-sweep-points",
        action="store_true",
        help="Include *_sweep_points_pending.csv files (default: excluded).",
    )
    return parser.parse_args()


def resolve_source_dir(logs_root: Path, src: str) -> Path:
    candidate = Path(src)
    if candidate.is_absolute():
        return candidate
    return logs_root / candidate


def normalize_target_name(filename: str) -> str:
    if filename.endswith("_sweep_points_all.csv"):
        return filename
    if filename.endswith("_sweep_points.csv"):
        return filename[: -len("_sweep_points.csv")] + "_sweep_points_all.csv"
    return filename


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, [])


def merge_headers(paths: list[Path]) -> list[str]:
    merged: list[str] = []
    for path in paths:
        for col in read_header(path):
            if col not in merged:
                merged.append(col)
    return merged


def row_digest(row: dict[str, str], columns: list[str]) -> str:
    payload = "\x1f".join((row.get(col) or "") for col in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge_csv_group(
    source_paths: list[Path],
    output_path: Path,
    dedupe: bool,
) -> tuple[int, int]:
    columns = merge_headers(source_paths)
    if not columns:
        return (0, 0)

    deduper = SQLiteDeduper(output_path.with_suffix(".dedupe.sqlite3")) if dedupe else None
    rows_seen = 0
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for src_path in source_paths:
            with src_path.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    rows_seen += 1
                    normalized = {col: row.get(col, "") for col in columns}
                    if deduper and deduper.seen(row_digest(normalized, columns)):
                        continue
                    writer.writerow(normalized)
                    rows_written += 1

    if deduper:
        deduper.close()
    return (rows_seen, rows_written)


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root).resolve()
    output_dir = (logs_root / args.output_subdir).resolve()
    dedupe = not args.no_dedupe

    if not logs_root.exists():
        sys.exit(f"Error: logs root does not exist: {logs_root}")
    if output_dir.exists():
        sys.exit(
            f"Error: output directory already exists: {output_dir}\n"
            "Use a different --output-subdir."
        )

    source_dirs = [resolve_source_dir(logs_root, src).resolve() for src in args.source_dirs]
    for src in source_dirs:
        if not src.exists() or not src.is_dir():
            sys.exit(f"Error: source directory does not exist or is not a directory: {src}")
        if src == output_dir:
            sys.exit("Error: output directory cannot also be a source directory.")

    groups: dict[str, list[Path]] = defaultdict(list)
    for src in source_dirs:
        for csv_path in sorted(src.glob("*.csv")):
            if (
                not args.include_pending_sweep_points
                and csv_path.name.endswith(PENDING_POINTS_SUFFIX)
            ):
                continue
            target_name = normalize_target_name(csv_path.name)
            groups[target_name].append(csv_path)

    if not groups:
        sys.exit("Error: no CSV files found in source directories.")

    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Created output directory: {output_dir}")
    print(f"Sources ({len(source_dirs)}):")
    for src in source_dirs:
        print(f"  - {src}")
    print(f"Row deduplication: {'enabled' if dedupe else 'disabled'}")

    total_seen = 0
    total_written = 0
    for target_name in sorted(groups):
        target_path = output_dir / target_name
        src_paths = groups[target_name]
        seen, written = merge_csv_group(src_paths, target_path, dedupe=dedupe)
        total_seen += seen
        total_written += written
        print(
            f"- {target_name}: merged {len(src_paths)} files, "
            f"rows seen={seen}, rows written={written}"
        )

    print(
        f"Done. Wrote {len(groups)} CSV files to {output_dir} "
        f"(rows seen={total_seen}, rows written={total_written})."
    )


if __name__ == "__main__":
    main()

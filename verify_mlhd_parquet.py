#!/usr/bin/env python3
"""
Automatically checks the mlhd parquet conversion against the raw source,
with no manual comparison needed.

This uses a separate code path from MlhdToParquet.py on purpose: plain
Python line-splitting, not DuckDB/SQL. Comparing the SQL's output against
itself would not catch a bug in the SQL.

For each sampled raw file:
  1. Decompress it and split each line on tab, by hand.
  2. Pad short rows / truncate long rows to 4 fields (matches null_padding
     and strict_mode=false in the conversion script).
  3. Pull the matching rows for that exact user_mbid + source + shard out
     of the parquet.
  4. Compare the two row sets exactly (order-independent).

Prints PASS/FAIL per sampled user and a final summary. Exit code is 0 only
if every sampled user matched exactly.

Usage:
    python3 verify_mlhd_parquet.py \\
        --input /path/to/Data/mlhd \\
        --parquet '/path/to/Data/parquet/mlhd/mlhd_*.parquet' \\
        --sample 300
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import duckdb

# Shards worth checking on purpose, not just by luck of a random draw:
# the ones that were redownloaded, and the ones that used to be duplicated
# under the old mlhdplus-partial-8 before the fix.
PRIORITY_SHARDS = [
    "mlhdplus-complete-d/d3",
    "mlhdplus-complete-e/e7",
    "mlhdplus-complete-f/fa",
    "mlhdplus-partial-d/da",
    "mlhdplus-partial-e/e5",
    "mlhdplus-partial-e/eb",
    "mlhdplus-partial-f/f4",
    "mlhdplus-partial-f/fd",
    "mlhdplus-partial-8/85",
    "mlhdplus-partial-8/8e",
]


def parse_raw_file(path: Path):
    # Shell out to zstdcat rather than the zstandard Python package: some of
    # these files (streaming-compressed, no content-size in the frame header)
    # make the package's one-shot decompress() raise "could not determine
    # content size in frame header". zstdcat handles them fine either way.
    out = subprocess.run(["zstdcat", str(path)], capture_output=True, check=True)
    rows = []
    for line in out.stdout.decode("utf-8", errors="replace").split("\n"):
        if not line:
            continue
        fields = line.split("\t")
        fields = (fields + ["", "", "", ""])[
            :4
        ]  # pad short / truncate long, same as the conversion
        ts_raw, artist, release, recording = fields
        try:
            ts = int(ts_raw)
        except ValueError:
            ts = None
        rows.append((ts, artist or None, release or None, recording or None))
    return rows


def find_all_files(input_root: Path):
    files = []
    for top in sorted(input_root.glob("mlhdplus-*-*")):
        source = top.name.split("-")[1]
        for shard_dir in top.iterdir():
            if not shard_dir.is_dir():
                continue
            for f in shard_dir.glob("*.txt.zst"):
                files.append((f, source, shard_dir.name))
    return files


def sort_key(row):
    # None can't be compared to int/str in Python 3, so sort None first,
    # per field, instead of failing.
    return tuple((x is None, x) for x in row)


def check_one(con, parquet_glob: str, path: Path, source: str, shard: str):
    user_mbid = path.name.removesuffix(
        ".txt.zst"
    )  # path.stem only strips ".zst", not ".txt.zst"
    expected = sorted(parse_raw_file(path), key=sort_key)
    actual = sorted(
        con.execute(
            f"""SELECT listened_at_unix, artist_mbid, release_mbid, recording_mbid
            FROM read_parquet('{parquet_glob}')
            WHERE user_mbid = ? AND source = ? AND shard = ?""",
            [user_mbid, source, shard],
        ).fetchall(),
        key=sort_key,
    )
    return user_mbid, expected, actual


def main():
    ap = argparse.ArgumentParser(
        description="Verify mlhd parquet against raw source, automatically."
    )
    ap.add_argument("--input", required=True, help="Path to the mlhd/ directory")
    ap.add_argument(
        "--parquet",
        required=True,
        help="Glob for the converted parquet, e.g. '.../mlhd_*.parquet'",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=300,
        help="Number of additional random users to check",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_root = Path(args.input)
    random.seed(args.seed)

    all_files = find_all_files(input_root)

    targeted = []
    for rel_shard in PRIORITY_SHARDS:
        shard_dir = input_root / rel_shard
        if not shard_dir.is_dir():
            print(f"  (skip, not found: {rel_shard})")
            continue
        source = rel_shard.split("/")[0].split("-")[1]
        candidates = list(shard_dir.glob("*.txt.zst"))
        if candidates:
            f = random.choice(candidates)
            targeted.append((f, source, shard_dir.name))

    random_sample = random.sample(all_files, min(args.sample, len(all_files)))
    sample = targeted + random_sample
    print(
        f"Checking {len(targeted)} targeted users + {len(random_sample)} random users "
        f"out of {len(all_files):,} total user files\n"
    )

    con = duckdb.connect()
    ok, fail = 0, 0
    for f, source, shard in sample:
        user_mbid, expected, actual = check_one(con, args.parquet, f, source, shard)
        if expected == actual:
            ok += 1
        else:
            fail += 1
            print(f"[FAIL] {source}/{shard}/{user_mbid}")
            print(f"       raw rows: {len(expected)}   parquet rows: {len(actual)}")
            only_raw = [r for r in expected if r not in actual]
            only_parquet = [r for r in actual if r not in expected]
            for r in only_raw[:3]:
                print(f"       in raw only:     {r}")
            for r in only_parquet[:3]:
                print(f"       in parquet only: {r}")

    print(f"\n{ok}/{ok + fail} sampled users matched exactly.")
    if fail:
        print("Do not delete the raw data. Investigate the FAILs above first.")
        sys.exit(1)
    else:
        print("Safe to move on to a backup-and-delete decision on the raw data.")
        sys.exit(0)


if __name__ == "__main__":
    main()

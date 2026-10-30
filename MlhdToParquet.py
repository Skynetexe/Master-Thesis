#!/usr/bin/env python3
"""
Convert the MLHD+ dataset (mlhdplus-complete-*/ and mlhdplus-partial-*/, each
full of per-user {uuid}.txt.zst listen logs) into a small number of tidy
Parquet files, and prove that the conversion didn't drop or duplicate anything.

Background on the format (from musicbrainz.org/doc/MLHD+):
  Each user has one file per archive type:
      mlhd/mlhdplus-{complete|partial}-X/XY/{user_mbid}.txt.zst
  Each line is a tab-separated quadruple, no header:
      <unix_timestamp>\t<artist_mbid>\t<release_mbid>\t<recording_mbid>
  The split is by row, not by user: a listen only lands in -complete if it
  resolves to a canonical recording; everything else (rows missing data, or
  containing values that couldn't be matched to a valid MBID) lands in
  -partial, under the same per-user filename as the complete side. So for
  a healthy download, mlhdplus-complete-N and mlhdplus-partial-N should hold
  very close to the same number of user files for every hex digit N.

What this script checks that a naive DuckDB COPY won't:
  1. Preflight: compares shard-folder counts and file counts between each
     complete-N/partial-N pair, and flags any top-level dir that doesn't
     have all 16 shard folders. This is what catches an interrupted/partial
     download (a complete-N bucket quietly missing users that partial-N has)
     or a misplaced/duplicated shard (a bucket with more shard folders than
     it should, using names that belong to a different prefix).
  2. Reject tracking: ignore_errors/strict_mode/null_padding make the CSV
     reader forgiving of ragged partial rows, which is necessary, but it
     also means a genuinely malformed line gets dropped with zero trace.
     store_rejects logs anything DuckDB can't parse into a queryable table
     instead of silently discarding it.
  3. Row-count reconciliation: separately counts raw lines in the source
     .txt.zst files (via a delimiter that can't appear in the data, so every
     line is exactly one field, independent of every setting in #2) and
     diffs that against the parquet row count. raw == written + rejected
     means everything is accounted for; anything else is a red flag.

Usage:
    python3 MlhdToParquet.py --input /path/to/Data/mlhd --output /path/to/Data/parquet/mlhd --threads 14 --memory-limit 24GB

    # sanity-check on a tiny slice first (recommended given the dataset size):
    python3 MlhdToParquet.py --input /path/to/Data/mlhd --output /tmp/mlhd_test --test

Requires: pip install duckdb
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import duckdb

TOP_DIR_RE = re.compile(r"^mlhdplus-(complete|partial)-([0-9a-f])$")
EXPECTED_SHARD_COUNT = 16

# Output schema (same for complete & partial, so all output files can be
# read together with one glob / duckdb.read_parquet("mlhd_*.parquet")):
#   user_mbid        VARCHAR    the listener's MBID (from the filename)
#   listened_at_unix BIGINT     raw unix epoch seconds, as in the source file
#   listened_at      TIMESTAMP  same instant, as UTC wall-clock digits with
#                               no timezone attached (see note below)
#   artist_mbid      VARCHAR    nullable; on -partial rows this may hold the
#                               original unresolved text rather than a real MBID
#   release_mbid     VARCHAR    nullable, same caveat as artist_mbid
#   recording_mbid   VARCHAR    nullable, same caveat as artist_mbid
#   source           VARCHAR    'complete' or 'partial'
#   shard            VARCHAR    2-hex-char directory the file lived in (e.g. '00'..'ff')
#
# Note on listened_at: to_timestamp() returns TIMESTAMP WITH TIME ZONE, which
# displays in whatever timezone the querying session happens to have (e.g.
# your machine's local zone), even though the stored instant is correct UTC
# either way. `AT TIME ZONE 'UTC'` converts that into a plain, tz-naive
# TIMESTAMP showing the literal UTC wall-clock digits, so the column means
# the same thing regardless of who queries it or from where, which matters
# once you're joining this against other tables later.
SQL_TEMPLATE = """
COPY (
    SELECT
        regexp_extract(filename, '([0-9a-fA-F-]{{36}})\\.txt\\.zst$', 1) AS user_mbid,
        TRY_CAST(column0 AS BIGINT)                                     AS listened_at_unix,
        to_timestamp(TRY_CAST(column0 AS BIGINT)) AT TIME ZONE 'UTC'    AS listened_at,
        NULLIF(column1, '')                                             AS artist_mbid,
        NULLIF(column2, '')                                             AS release_mbid,
        NULLIF(column3, '')                                             AS recording_mbid,
        '{source}'                                                      AS source,
        regexp_extract(filename, '/([0-9a-f]{{2}})/[^/]+\\.txt\\.zst$', 1) AS shard
    FROM read_csv(
        '{glob_pattern}',
        delim = '\t',
        quote = '',
        escape = '',
        header = false,
        columns = {{'column0': 'VARCHAR', 'column1': 'VARCHAR', 'column2': 'VARCHAR', 'column3': 'VARCHAR'}},
        auto_detect = false,
        null_padding = true,
        strict_mode = false,
        ignore_errors = true,
        filename = true,
        compression = 'zstd',
        store_rejects = true,
        rejects_scan = 'reject_scans_{tag}',
        rejects_table = 'reject_errors_{tag}'
    )
)
TO '{out_file}' (FORMAT PARQUET, COMPRESSION 'zstd');
"""

# A raw line count for the same glob, independent of every CSV-parsing
# setting above: delim is a control byte that can never appear in the
# tab-separated source data, so every line becomes exactly one field and
# this can never itself throw a "wrong number of columns" error.
RAW_LINE_COUNT_SQL = """
SELECT count(*) FROM read_csv(
    '{glob_pattern}',
    delim = '\x01',
    header = false,
    columns = {{'line': 'VARCHAR'}},
    compression = 'zstd'
)
"""


def find_top_dirs(input_root: Path):
    dirs = []
    for d in sorted(input_root.iterdir()):
        if d.is_dir():
            m = TOP_DIR_RE.match(d.name)
            if m:
                dirs.append((d, m.group(1), m.group(2)))
    return dirs


def count_shard_dirs(top_dir: Path) -> int:
    return sum(1 for p in top_dir.iterdir() if p.is_dir())


def count_user_files(top_dir: Path) -> int:
    n = 0
    for _root, _dirs, files in os.walk(top_dir):
        n += sum(1 for f in files if f.endswith(".txt.zst"))
    return n


def preflight(top_dirs) -> bool:
    """Print a complete-vs-partial sanity table before any conversion runs.
    Returns True if everything looks healthy, False if anomalies were found.
    A mismatch here means the conversion will be silently incomplete or
    duplicated no matter how careful the parsing SQL is: no amount of
    reject tracking catches "the file was never there to begin with"."""
    print("Preflight: comparing shard folders and file counts per prefix\n")
    shard_counts, file_counts = {}, {}
    for d, source, digit in top_dirs:
        shard_counts[(source, digit)] = count_shard_dirs(d)
        file_counts[(source, digit)] = count_user_files(d)

    healthy = True
    header = f"  {'prefix':8s} {'complete shards':>16s} {'complete files':>15s} {'partial shards':>15s} {'partial files':>14s}"
    print(header)
    for digit in "0123456789abcdef":
        cs, cf = shard_counts.get(("complete", digit)), file_counts.get(
            ("complete", digit)
        )
        ps, pf = shard_counts.get(("partial", digit)), file_counts.get(
            ("partial", digit)
        )
        if cs is None and ps is None:
            continue
        flag = ""
        if cs is not None and cs != EXPECTED_SHARD_COUNT:
            flag += f"  <-- complete-{digit} has {cs}/{EXPECTED_SHARD_COUNT} shard dirs"
        if ps is not None and ps != EXPECTED_SHARD_COUNT:
            flag += f"  <-- partial-{digit} has {ps}/{EXPECTED_SHARD_COUNT} shard dirs"
        if cf is not None and pf is not None and cf != pf:
            flag += f"  <-- complete/partial file-count mismatch ({cf:,} vs {pf:,})"
        if flag:
            healthy = False
        print(
            f"  {digit:8s} {str(cs) if cs is not None else '-':>16s} "
            f"{f'{cf:,}' if cf is not None else '-':>15s} "
            f"{str(ps) if ps is not None else '-':>15s} "
            f"{f'{pf:,}' if pf is not None else '-':>14s}{flag}"
        )

    if not healthy:
        print(
            "\n  Anomalies found above. This usually means either an interrupted/partial"
        )
        print(
            "  download (a complete-N bucket missing users partial-N has), or shards that"
        )
        print(
            "  got nested under the wrong prefix. Converting now will silently bake that"
        )
        print(
            "  in, either as missing rows or, if any misplaced shards duplicate content,"
        )
        print("  that also exists under its correct prefix, as rows counted twice.")
    else:
        print(
            "\n  Looks healthy: every prefix has 16/16 shard dirs and matching file counts."
        )
    print()
    return healthy


def convert_dir(
    con, top_dir: Path, source: str, digit: str, output_root: Path, overwrite: bool
):
    out_file = output_root / f"mlhd_{source}_{digit}.parquet"
    if out_file.exists() and not overwrite:
        print(f"  [skip] {out_file.name} already exists (use --overwrite to redo)")
        return

    tag = f"{source}_{digit}"
    glob_pattern = str(top_dir / "*" / "*.txt.zst")
    sql = SQL_TEMPLATE.format(
        source=source, glob_pattern=glob_pattern, out_file=str(out_file), tag=tag
    )

    t0 = time.time()
    con.execute(sql)
    elapsed = time.time() - t0

    n_out = con.execute(f"SELECT count(*) FROM read_parquet('{out_file}')").fetchone()[
        0
    ]

    try:
        n_rejects = con.execute(f"SELECT count(*) FROM reject_errors_{tag}").fetchone()[
            0
        ]
    except duckdb.Error:
        n_rejects = (
            0  # table only gets created if store_rejects actually found something
        )

    raw_sql = RAW_LINE_COUNT_SQL.format(glob_pattern=glob_pattern)
    n_raw = con.execute(raw_sql).fetchone()[0]

    if source == "complete":
        # by definition every -complete row should have all 4 fields resolved;
        # a null here means either a row landed in the wrong archive, or the
        # 4-column alignment broke on that particular line (see note in the
        # module docstring: strict_mode=false silently drops >4-field rows,
        # which row-count reconciliation alone won't catch since the row
        # still gets written, just with fields missing).
        n_incomplete = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_file}') "
            "WHERE artist_mbid IS NULL OR release_mbid IS NULL OR recording_mbid IS NULL"
        ).fetchone()[0]
        if n_incomplete:
            print(
                f"    !! {n_incomplete:,} 'complete' row(s) have a null MBID. "
                f"That shouldn't happen by definition and is worth a manual look."
            )

    unaccounted = n_raw - n_out - n_rejects
    if unaccounted == 0 and n_rejects == 0:
        status = "OK"
    elif unaccounted == 0:
        status = "OK, WITH REJECTS"
    else:
        status = "MISMATCH"

    print(
        f"  [{status}] {out_file.name}: {n_out:,} rows written, {n_raw:,} raw lines, "
        f"{n_rejects:,} rejected, {elapsed:,.1f}s"
    )
    if n_rejects:
        print(
            f"    -> see: SELECT * FROM reject_errors_{tag}  (this connection only, this run)"
        )
    if unaccounted != 0:
        print(
            f"    !! {unaccounted:,} row(s) unaccounted for: not written and not logged as "
            f"a reject. Investigate {top_dir.name} before trusting this file."
        )

    return n_raw, n_out, n_rejects


def main():
    ap = argparse.ArgumentParser(
        description="Convert MLHD+ .txt.zst files into Parquet."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to the mlhd/ directory "
        "(the one containing mlhdplus-complete-0, mlhdplus-partial-0, ...)",
    )
    ap.add_argument(
        "--output", required=True, help="Directory to write .parquet files into"
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB thread count (0 = duckdb default)",
    )
    ap.add_argument(
        "--memory-limit", default=None, help="cap on the DuckDB memory use, e.g. '8GB'"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo directories that already have output",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="Only convert the first shard subfolder of the first complete and "
        "first partial top-level directory found, as a quick sanity check.",
    )
    ap.add_argument(
        "--only",
        choices=["complete", "partial"],
        default=None,
        help="Only process -complete- or -partial- directories",
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the complete/partial sanity check",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Run the full conversion even if preflight finds anomalies",
    )
    args = ap.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    top_dirs = find_top_dirs(input_root)
    if not top_dirs:
        sys.exit(
            f"No mlhdplus-{{complete,partial}}-X directories found under {input_root}"
        )

    if not args.skip_preflight:
        healthy = preflight(top_dirs)
        if not healthy and not args.test:
            if args.force:
                print("  --force given: proceeding with the full conversion anyway.\n")
            else:
                print(
                    "  Refusing to run the full conversion (use --force to override, or fix"
                )
                print("  the data first). --test is unaffected by this check.\n")
                sys.exit(1)

    if args.only:
        top_dirs = [t for t in top_dirs if t[1] == args.only]

    con = duckdb.connect()
    if args.threads:
        con.execute(f"PRAGMA threads={args.threads}")
    if args.memory_limit:
        con.execute(f"SET memory_limit='{args.memory_limit}'")

    totals = {"raw": 0, "out": 0, "rejects": 0}

    if args.test:
        picked = {}
        for d, source, digit in top_dirs:
            picked.setdefault(source, (d, source, digit))
        top_dirs = list(picked.values())
        print(
            f"[test mode] converting a single shard from: {[d.name for d, _, _ in top_dirs]}\n"
        )

        for d, source, digit in top_dirs:
            shard_dirs = sorted(p for p in d.iterdir() if p.is_dir())
            if not shard_dirs:
                continue
            shard = shard_dirs[0]
            tag = f"{source}_{digit}_TEST"
            out_file = output_root / f"mlhd_{source}_{digit}_TEST.parquet"
            glob_pattern = str(shard / "*.txt.zst")
            sql = SQL_TEMPLATE.format(
                source=source,
                glob_pattern=glob_pattern,
                out_file=str(out_file),
                tag=tag,
            )
            con.execute(sql)

            n_out = con.execute(
                f"SELECT count(*) FROM read_parquet('{out_file}')"
            ).fetchone()[0]
            try:
                n_rejects = con.execute(
                    f"SELECT count(*) FROM reject_errors_{tag}"
                ).fetchone()[0]
            except duckdb.Error:
                n_rejects = 0
            n_raw = con.execute(
                RAW_LINE_COUNT_SQL.format(glob_pattern=glob_pattern)
            ).fetchone()[0]
            unaccounted = n_raw - n_out - n_rejects
            status = "OK" if unaccounted == 0 else "MISMATCH"
            print(
                f"  [{status}] {out_file.name}: {n_out:,} rows, {n_raw:,} raw lines, "
                f"{n_rejects:,} rejected (from {shard})"
            )
            if n_rejects:
                print(
                    f"    sample reject: {con.execute(f'SELECT * FROM reject_errors_{tag} LIMIT 1').fetchone()}"
                )
        print(
            "\nInspect the _TEST.parquet file(s) above before running the full conversion."
        )
        return

    print(f"Found {len(top_dirs)} top-level directories to convert.\n")
    for d, source, digit in top_dirs:
        print(f"[{source}-{digit}] {d}")
        result = convert_dir(con, d, source, digit, output_root, args.overwrite)
        if result:
            n_raw, n_out, n_rejects = result
            totals["raw"] += n_raw
            totals["out"] += n_out
            totals["rejects"] += n_rejects

    print()
    print(
        f"TOTAL: {totals['out']:,} rows written, {totals['raw']:,} raw lines, "
        f"{totals['rejects']:,} rejected"
    )
    unaccounted_total = totals["raw"] - totals["out"] - totals["rejects"]
    if unaccounted_total == 0:
        print("Every raw line is accounted for (written or logged as a reject).")
    else:
        print(f"!! {unaccounted_total:,} row(s) unaccounted for across the whole run.")

    print("\nRead the whole dataset back with, e.g.:")
    print(f"  duckdb.read_parquet('{output_root}/mlhd_*.parquet')")


if __name__ == "__main__":
    main()

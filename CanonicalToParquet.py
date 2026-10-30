#!/usr/bin/env python3
"""Convert the canonical-data CSVs into Parquet, with row reconciliation."""

import os
from pathlib import Path

import duckdb

CODE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("THESIS_DATA_ROOT", CODE_DIR / "data"))
OUT = ROOT / "parquet/canonical"
OUT.mkdir(parents=True, exist_ok=True)

# Find the newest canonical dump present, whatever its date. A hard-coded
# dated path breaks when a new dump replaces the old one. is_dir() is
# essential: the .tar.zst and checksum files match the same glob pattern.
_dumps = sorted(
    d
    for d in (ROOT / "canonical_data").glob("musicbrainz-canonical-dump-*")
    if d.is_dir()
)
if not _dumps:
    raise SystemExit("no musicbrainz-canonical-dump-* folder under canonical_data/")
CANON = _dumps[-1] / "canonical"
print(f"using canonical dump: {_dumps[-1].name}")

jobs = [
    (
        CANON / "canonical_musicbrainz_data.csv",
        OUT / "canonical_musicbrainz_data.parquet",
    ),
    (
        CANON / "canonical_recording_redirect.csv",
        OUT / "canonical_recording_redirect.parquet",
    ),
    (
        CANON / "canonical_release_redirect.csv",
        OUT / "canonical_release_redirect.parquet",
    ),
]

con = duckdb.connect()
for src, dst in jobs:
    if not src.exists():
        print(f"[skip] {src} not found")
        continue
    raw = con.execute(f"SELECT count(*) FROM read_csv_auto('{src}')").fetchone()[0]
    con.execute(
        f"COPY (SELECT * FROM read_csv_auto('{src}')) TO '{dst}' "
        "(FORMAT PARQUET, COMPRESSION 'zstd')"
    )
    out_n = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
    print(
        f"[{'OK' if raw == out_n else 'MISMATCH'}] {dst.name}: {out_n:,} rows ({raw:,} CSV rows)"
    )
    if raw != out_n:
        raise RuntimeError(f"{dst.name}: wrote {out_n:,} rows from {raw:,} CSV rows")

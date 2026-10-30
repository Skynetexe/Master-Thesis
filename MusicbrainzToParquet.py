#!/usr/bin/env python3
"""Convert the MusicBrainz dumps into Parquet.

Input: the musicbrainz/ folder of the data folder. Two parts are handled.

  fullexport/<dumpset>/mbdump/<table>      core database, one file per table
  json-dumps/<entity>/mbdump/<entity>      JSON dump, one JSON object per line
  json-dumps/<entity>/<hh>/<mbid>.txt.zst  optional per-MBID files, if present

The fullexport files use the PostgreSQL COPY text format:

  - One row per line. Fields are separated by one tab.
  - NULL is written as the two characters \\N.
  - A raw tab, newline, carriage return, or backslash never occurs inside a
    field. COPY escapes them as \\t, \\n, \\r and \\\\. Other control bytes
    appear as octal, for example \\012.

So a plain CSV read is not lossless. This script reads every field as
VARCHAR, turns \\N into NULL, undoes the COPY escapes, and writes one Parquet
file per table. All values stay VARCHAR. Nothing is retyped, reordered, or
dropped. Cast to numbers or dates later, at query time.

Column names: the dump files carry no schema. Pass --schema-sql with SQL
files from a musicbrainz-server checkout. The script parses the CREATE TABLE
statements, matches them to the table files, and validates the column count
against the real data. A table without a match gets generic names c0, c1, ...
A wrong column count is refused, so a bad match cannot corrupt names.

Checks, in the spirit of MlhdToParquet.py:

  1. A probe file runs through the real read pipeline before any big work.
     It verifies the NULL rule, the empty-string rule, and the escape
     decoder on known tricky values.
  2. Preflight compares TIMESTAMP and SCHEMA_SEQUENCE across all dump sets.
     A mismatch means the sets come from different replication days.
  3. Every table is reconciled: raw lines == rows written + rows rejected.
     Rejected rows are saved to a sidecar Parquet file, not dropped.
  4. A manifest records every table, its columns, and its counts.
     Re-runs skip tables whose output already exists.

Usage:

    # optional, for real column names:
    git clone --depth 1 https://github.com/metabrainz/musicbrainz-server

    python3 MusicbrainzToParquet.py \
        --input  /path/Data/musicbrainz \
        --output /path/Data/parquet/musicbrainz \
        --schema-sql /path/musicbrainz-server/admin/sql \
        --threads 14 --memory-limit 24GB

    # small end-to-end sanity check first:
    python3 MusicbrainzToParquet.py --input ... --output /tmp/mb_test --test

Requires: pip install duckdb
Optional: pip install zstandard  (only for the per-MBID file fallback)
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import duckdb

TAB = chr(9)  # tab: the COPY field separator
SOH = chr(1)  # control byte: a delimiter that cannot occur in COPY text
BS = chr(92)  # backslash

HEX2_RE = re.compile(r"[0-9a-f]{2}")

CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:GLOBAL\s+|LOCAL\s+|TEMP(?:ORARY)?\s+|UNLOGGED\s+)*TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r'("[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)'
    r'(?:\.\s*("[^"]+"|[A-Za-z_][A-Za-z0-9_$]*))?'
    r"\s*\(",
    re.IGNORECASE,
)

CONSTRAINT_WORDS = {
    "PRIMARY",
    "FOREIGN",
    "UNIQUE",
    "CHECK",
    "CONSTRAINT",
    "EXCLUDE",
    "LIKE",
    "INHERITS",
    "PARTITION",
}


def sqlstr(p):
    """Escape a path for a single-quoted SQL string literal."""
    return str(p).replace("'", "''")


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"


# Discovery


def read_small(path: Path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


def discover_fullexport(input_root: Path):
    """Find every dump set: fullexport/<dumpset>/mbdump/<table file>."""
    base = input_root / "fullexport"
    if not base.is_dir():
        return {}
    dumpsets = {}
    for d in sorted(base.iterdir()):
        inner = d / "mbdump"
        if not (d.is_dir() and inner.is_dir()):
            continue
        tables = sorted(p for p in inner.iterdir() if p.is_file())
        if not tables:
            continue
        dumpsets[d.name] = {
            "path": d,
            "tables": tables,
            "timestamp": read_small(d / "TIMESTAMP"),
            "schema_sequence": read_small(d / "SCHEMA_SEQUENCE"),
        }
    return dumpsets


def discover_json_dumps(input_root: Path):
    base = input_root / "json-dumps"
    if not base.is_dir():
        return {}
    entities = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        combined = d / "mbdump" / d.name
        shard_dirs = [
            p for p in sorted(d.iterdir()) if p.is_dir() and HEX2_RE.fullmatch(p.name)
        ]
        n_shards = sum(1 for sd in shard_dirs for _ in sd.glob("*.txt.zst"))
        if not combined.is_file() and n_shards == 0:
            continue
        entities[d.name] = {
            "path": d,
            "combined": combined if combined.is_file() else None,
            "shard_glob": str(d / "*" / "*.txt.zst") if shard_dirs else None,
            "shard_dirs": [sd.name for sd in shard_dirs],
            "n_shard_files": n_shards,
            "timestamp": read_small(d / "TIMESTAMP"),
        }
    return entities


# Schema parsing: the ordered columns of each table, from the
# musicbrainz-server SQL files


def _find_body_end(text, open_idx):
    """Return the index of the closing paren that matches text[open_idx]."""
    depth = 0
    i, n = open_idx, len(text)
    state = "code"
    while i < n:
        c = text[i]
        if state == "code":
            if c == "'":
                state = "str"
            elif c == "-" and text[i : i + 2] == "--":
                state = "line"
            elif c == "/" and text[i : i + 2] == "/*":
                state = "block"
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "str":
            if c == "'":
                if text[i : i + 2] != "''":
                    state = "code"
        elif state == "line":
            if c == "\n":
                state = "code"
        else:  # block comment
            if text[i : i + 2] == "*/":
                state = "code"
                i += 1
        i += 1
    return -1


def _split_parts(body):
    """Split a CREATE TABLE body on commas outside strings, comments, parens."""
    parts, cur = [], []
    depth = 0
    i, n = 0, len(body)
    state = "code"
    while i < n:
        c = body[i]
        if state == "code":
            if c == "'":
                state = "str"
            elif c == "-" and body[i : i + 2] == "--":
                state = "line"
            elif c == "/" and body[i : i + 2] == "/*":
                state = "block"
            elif c == "(":
                depth += 1
                cur.append(c)
            elif c == ")":
                depth -= 1
                cur.append(c)
            elif c == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        elif state == "str":
            if c == "'" and body[i : i + 2] != "''":
                state = "code"
        elif state == "line":
            if c == "\n":
                state = "code"
        else:
            if body[i : i + 2] == "*/":
                state = "code"
                i += 1
        i += 1
    parts.append("".join(cur))
    return [p for p in (s.strip() for s in parts) if p]


def _column_name(part):
    m = re.match(r'"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*)', part)
    if not m:
        return None
    word = m.group(1) or m.group(2)
    if word.upper() in CONSTRAINT_WORDS:
        return None
    return word


def create_tables_in_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  [warn] cannot read schema file {path}: {e}")
        return []
    out = []
    for m in CREATE_TABLE_RE.finditer(text):
        name = m.group(1)
        if m.group(2):
            name = f"{name}.{m.group(2)}"
        name = name.replace('"', "")
        open_idx = m.end() - 1
        close_idx = _find_body_end(text, open_idx)
        if close_idx < 0:
            continue
        body = text[open_idx + 1 : close_idx]
        cols = [c for c in (_column_name(p) for p in _split_parts(body)) if c]
        if cols:
            out.append((name, cols))
    return out


def load_schemas(specs):
    files = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            found = sorted(p.glob("*.sql"))
            files.extend(found)
            print(
                f"  schema dir {p}: {len(found)} top-level .sql files (subfolders ignored)"
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"  [warn] --schema-sql path not found: {p}")
    schemas = {}
    for f in files:
        for name, cols in create_tables_in_file(f):
            schemas[name.lower()] = cols  # later files win
    print(f"  parsed {len(schemas)} CREATE TABLE definitions")
    return schemas


def lookup_schema(schemas, table):
    key = table.lower()
    if key in schemas:
        return schemas[key]
    if "." in key:
        return schemas.get(key.rsplit(".", 1)[1])
    return None


def resolve_columns(schemas, table, ncols):
    cols = lookup_schema(schemas, table)
    generic = [f"c{i}" for i in range(ncols)]
    if cols is None:
        return generic, "generic (no schema match)"
    if len(cols) != ncols:
        return (
            generic,
            f"generic (SQL schema has {len(cols)} columns, data has {ncols})",
        )
    if len(set(cols)) != len(cols):
        return generic, "generic (duplicate names in parsed schema)"
    return cols, "schema"


# Escape decoder: Postgres COPY text to plain text

REGEX_MACRO_SQL = r"""
CREATE OR REPLACE MACRO pg_unescape(s) AS
    CASE
        WHEN s IS NULL THEN NULL
        WHEN strpos(s, '\') = 0 THEN s
        ELSE regexp_replace(
            s,
            '\\[0-7]{1,3}|\\.',
            m -> CASE
                WHEN m = '\n' THEN chr(10)
                WHEN m = '\r' THEN chr(13)
                WHEN m = '\t' THEN chr(9)
                WHEN m = '\b' THEN chr(8)
                WHEN m = '\f' THEN chr(12)
                WHEN m = '\v' THEN chr(11)
                WHEN length(m) = 2 AND substr(m, 2, 1) NOT BETWEEN '0' AND '7'
                    THEN substr(m, 2, 1)
                ELSE chr(
                    CASE length(m)
                        WHEN 4 THEN 64 * CAST(substr(m, 2, 1) AS INTEGER)
                                    +  8 * CAST(substr(m, 3, 1) AS INTEGER)
                                    +      CAST(substr(m, 4, 1) AS INTEGER)
                        WHEN 3 THEN      8 * CAST(substr(m, 2, 1) AS INTEGER)
                                    +      CAST(substr(m, 3, 1) AS INTEGER)
                        ELSE CAST(substr(m, 2, 1) AS INTEGER)
                    END
                )
            END
        )
    END;
"""


def nested_unescape_macro_sql():
    """Fallback decoder. Uses only chr() calls, so it cannot be confused by
    backslash handling in SQL string literals. Octal escapes stay as written.
    chr(1) is a safe sentinel: COPY text never contains a raw chr(1)."""
    expr = "replace(s, chr(92) || chr(92), chr(1))"
    for ch, code in (("n", 10), ("t", 9), ("r", 13), ("b", 8), ("f", 12), ("v", 11)):
        expr = f"replace({expr}, chr(92) || '{ch}', chr({code}))"
    expr = f"replace({expr}, chr(1), chr(92))"
    return (
        "CREATE OR REPLACE MACRO pg_unescape(s) AS\n"
        "    CASE\n"
        "        WHEN s IS NULL THEN NULL\n"
        "        WHEN strpos(s, chr(92)) = 0 THEN s\n"
        f"        ELSE {expr}\n"
        "    END;\n"
    )


def install_unescape_macro(con):
    """Install the decoder and prove it on known tricky input."""
    raw = "a" + BS + BS + "b" + BS + "t" + BS + "n" + BS + "010" + BS + BS + "N" + "c"
    want = "a" + BS + "b" + chr(9) + chr(10) + chr(8) + BS + "N" + "c"
    try:
        con.execute(REGEX_MACRO_SQL)
        got = con.execute("SELECT pg_unescape(?)", [raw]).fetchone()[0]
        if got == want:
            print(
                "Escape decoder: full regexp decoder (handles \\n \\t \\r \\\\ and octal)."
            )
            return "regex"
        reason = f"wrong output {got!r}"
    except duckdb.Error as e:
        reason = str(e).splitlines()[0]
    # Fallback: nested replaces.
    raw2 = "a" + BS + BS + "b" + BS + "t" + BS + "n" + BS + BS + "N" + "c"
    want2 = "a" + BS + "b" + chr(9) + chr(10) + BS + "N" + "c"
    try:
        con.execute(nested_unescape_macro_sql())
        got2 = con.execute("SELECT pg_unescape(?)", [raw2]).fetchone()[0]
    except duckdb.Error as e:
        sys.exit(f"The unescape macro failed twice. Update DuckDB. ({e})")
    if got2 == want2:
        print(f"Escape decoder: nested-replace fallback ({reason}).")
        print(
            "  Octal escapes such as \\012 stay as written. They do not occur in normal"
        )
        print("  MusicBrainz data.")
        return "nested"
    sys.exit(
        f"The unescape macro does not work with this DuckDB ({got2!r}). Update DuckDB."
    )


# CSV NULL-semantics probe


def write_probe_csv(path: Path):
    rows = [
        [BS + "N", "", "x", "y"],
        ["a" + BS + BS + "N", "b" + BS + "t" + "c", "d", "e"],
        [
            "x" + BS + "t" + "y" + BS + "n" + "z",
            "q" + BS + BS + "r" + BS + "t" + "s",
            BS + "010",
            "end",
        ],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(TAB.join(r) + "\n")


def csv_mode_options(mode, ncols):
    if mode == "nullstr":
        return ", nullstr = '\\N'"
    if mode == "force_not_null":
        cols = ", ".join(f"'c{i}'" for i in range(ncols))
        return f", force_not_null = [{cols}]"
    return ""


def probe_csv_mode(con, output_root: Path, decoder):
    """Run a tiny COPY-format file through the real pipeline. The probe
    checks the NULL rule, the empty-string rule, and the escape decoder
    before any large table is touched."""
    probe_path = output_root / "_probe.csv"
    write_probe_csv(probe_path)
    octal = chr(8) if decoder == "regex" else BS + "010"
    want = [
        (None, "", "x", "y"),
        ("a" + BS + "N", "b" + TAB + "c", "d", "e"),
        ("x" + TAB + "y" + "\n" + "z", "q" + BS + "r" + TAB + "s", octal, "end"),
    ]
    notes = []
    for mode in ("nullstr", "force_not_null", "default"):
        opts = csv_mode_options(mode, 4)
        sql = f"""
            SELECT
                pg_unescape(NULLIF(c0, chr(92) || 'N')),
                pg_unescape(NULLIF(c1, chr(92) || 'N')),
                pg_unescape(NULLIF(c2, chr(92) || 'N')),
                pg_unescape(NULLIF(c3, chr(92) || 'N'))
            FROM read_csv(
                '{sqlstr(probe_path)}',
                delim = '{TAB}', quote = '', escape = '',
                header = false, auto_detect = false,
                columns = {{'c0': 'VARCHAR', 'c1': 'VARCHAR',
                            'c2': 'VARCHAR', 'c3': 'VARCHAR'}}{opts}
            )
        """
        try:
            got = con.execute(sql).fetchall()
        except duckdb.Error as e:
            notes.append(f"{mode}: not supported ({str(e).splitlines()[0]})")
            continue
        if len(got) != 3:
            notes.append(f"{mode}: expected 3 rows, got {len(got)}")
            continue
        problems = []
        empty_conflated = False
        for i, (g, w) in enumerate(zip(got, want)):
            for j, (gv, wv) in enumerate(zip(g, w)):
                if gv == wv:
                    continue
                if mode == "default" and i == 0 and j == 1 and gv is None:
                    empty_conflated = True
                    continue
                problems.append(f"row {i + 1} field {j + 1}: got {gv!r}, want {wv!r}")
        if problems:
            notes.append(f"{mode}: " + ". ".join(problems))
            continue
        probe_path.unlink(missing_ok=True)
        if mode == "nullstr":
            print(
                "CSV NULL mode: nullstr. \\N reads as NULL, an empty field stays an empty string."
            )
        elif mode == "force_not_null":
            print(
                "CSV NULL mode: force_not_null. NULLIF turns \\N into NULL in the SELECT."
            )
        else:
            print("CSV NULL mode: default.")
            if empty_conflated:
                print(
                    "  WARNING: this DuckDB reads an empty field as NULL. The dump writes"
                )
                print("  empty strings for many NOT NULL columns and NULL only as \\N.")
                print(
                    "  In this mode the two values become one. Everything else is exact."
                )
        return mode
    probe_path.unlink(missing_ok=True)
    sys.exit("No CSV configuration passed the probe. " + " | ".join(notes))


# fullexport conversion


def detect_column_count(path: Path):
    """COPY text has a fixed number of fields per row. An escaped \\t is two
    characters, so only real tab bytes count. Only the first lines are read."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(1 << 20)
            whole = b"\n" not in chunk
            if whole:
                chunk += f.read()  # the first line is over 1 MiB. Read it all.
    except OSError as e:
        return None, f"cannot read file: {e}"
    if not chunk:
        return 0, "empty file"
    lines = chunk.split(b"\n")
    if not whole and lines:
        lines = lines[:-1]  # the last piece may be cut mid-line
    counts = {}
    for ln in lines[:200]:
        if not ln.strip():
            continue
        n = ln.count(b"\t") + 1
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return 0, "no complete rows found"
    ncols = max(counts, key=lambda k: counts[k])
    if len(counts) > 1:
        detail = ", ".join(f"{k} fields x{v}" for k, v in sorted(counts.items()))
        return ncols, f"inconsistent field counts in first lines ({detail})"
    return ncols, "ok"


def convert_table(con, cfg, dumpset, table_path: Path, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    table = table_path.name
    rel = f"{dumpset}/{table}"
    out_path = out_dir / (table + ".parquet")
    if out_path.exists() and not args.overwrite:
        print(f"  [skip] {rel} (output exists)")
        return None

    ncols, note = detect_column_count(table_path)
    if ncols is None:
        print(f"  [ERROR] {rel}: {note}")
        return {"source": str(table_path), "status": "error", "error": note}
    if ncols == 0:
        print(f"  [empty] {rel}: {note}")
        return {"source": str(table_path), "status": "empty", "rows": 0, "detect": note}

    colnames, name_src = resolve_columns(cfg["schemas"], table, ncols)
    tag = re.sub(r"[^0-9A-Za-z_]", "_", rel)
    nullif = "NULLIF(c{i}, chr(92) || 'N')"
    select_items = [
        f'pg_unescape({nullif.format(i=i)}) AS "{n}"' for i, n in enumerate(colnames)
    ]
    columns_map = "{" + ", ".join(f"'c{i}': 'VARCHAR'" for i in range(ncols)) + "}"
    mode_opts = csv_mode_options(cfg["mode"], ncols)

    t0 = time.time()
    sql = f"""
        COPY (
            SELECT {", ".join(select_items)}
            FROM read_csv(
                '{sqlstr(table_path)}',
                delim = '{TAB}',
                quote = '',
                escape = '',
                header = false,
                auto_detect = false,
                columns = {columns_map}{mode_opts},
                max_line_size = 100000000,
                ignore_errors = true,
                store_rejects = true,
                rejects_scan = 'rs_{tag}',
                rejects_table = 're_{tag}'
            )
        )
        TO '{sqlstr(out_path)}'
        (FORMAT PARQUET, COMPRESSION 'zstd')
    """
    con.execute(sql)
    elapsed = time.time() - t0

    n_out = con.execute(
        f"SELECT count(*) FROM read_parquet('{sqlstr(out_path)}')"
    ).fetchone()[0]

    n_rej = None
    for tname, label in ((f"re_{tag}", "errors"), (f"rs_{tag}", "rows")):
        try:
            c = con.execute(f"SELECT count(*) FROM {tname}").fetchone()[0]
        except duckdb.Error:
            continue
        if c:
            side = out_dir / f"{table}.rejects_{label}.parquet"
            con.execute(
                f"COPY (SELECT * FROM {tname}) TO '{sqlstr(side)}' "
                "(FORMAT PARQUET, COMPRESSION 'zstd')"
            )
            print(f"    -> {c:,} rejected {label} saved to {side.name}")
        con.execute(f"DROP TABLE IF EXISTS {tname}")
        if label == "errors":
            n_rej = c
        elif n_rej is None:
            n_rej = c
    if n_rej is None:
        n_rej = 0

    # Raw line count, independent of every CSV parsing setting above.
    raw_sql = f"""
        SELECT count(*) FROM read_csv(
            '{sqlstr(table_path)}',
            delim = '{SOH}',
            header = false,
            columns = {{'line': 'VARCHAR'}},
            quote = '', escape = '', auto_detect = false,
            max_line_size = 100000000
        )
    """
    n_raw = con.execute(raw_sql).fetchone()[0]

    unaccounted = n_raw - n_out - n_rej
    status = "OK" if unaccounted == 0 else "MISMATCH"
    line = (
        f"  [{status}] {rel}: {n_out:,} rows, {n_raw:,} raw lines, "
        f"{n_rej:,} rejected, {elapsed:,.1f}s"
    )
    if note != "ok":
        line += f"   (detect: {note})"
    if name_src != "schema":
        line += f"   (columns: {name_src})"
    print(line)
    if unaccounted:
        print(f"    !! {unaccounted:,} line(s) are neither written nor rejected.")
        print(f"       Investigate {table_path} before you trust this table.")

    return {
        "source": str(table_path),
        "bytes": table_path.stat().st_size,
        "columns": colnames,
        "column_source": name_src,
        "rows": n_out,
        "raw_lines": n_raw,
        "rejected": n_rej,
        "unaccounted": unaccounted,
        "seconds": round(elapsed, 1),
        "output": str(out_path),
        "status": status,
    }


# json-dumps conversion


def _json_valid(con, s):
    return bool(con.execute("SELECT json_valid(?)", [s]).fetchone()[0])


def convert_json_combined(con, entity, info, out_dir: Path, args):
    path = info["combined"]
    out_path = out_dir / f"{entity}.parquet"
    if out_path.exists() and not args.overwrite:
        print(f"  [skip] json-dumps/{entity} (output exists)")
        return None

    read = (
        f"read_csv('{sqlstr(path)}', delim = '{SOH}', header = false, "
        f"columns = {{'line': 'VARCHAR'}}, quote = '', escape = '', "
        f"auto_detect = false, max_line_size = 100000000)"
    )

    first = con.execute(f"SELECT line FROM {read} LIMIT 1").fetchone()
    if first is None:
        print(f"  [empty] json-dumps/{entity}")
        return {"source": str(path), "status": "empty", "rows": 0}
    if not _json_valid(con, first[0]):
        print(f"  [SKIP] json-dumps/{entity}: the first line is not JSON.")
        print(f"       Inspect {path} by hand before you convert it.")
        return {"source": str(path), "status": "skipped"}

    t0 = time.time()
    # The data column holds the complete original JSON text. Nothing is lost.
    # Query it later with json_extract_string(data, '$.field').
    # release lines average ~65 KB, so at high thread counts the pipeline
    # holds more in-flight string data than the memory limit allows. On
    # out-of-memory, retry the same COPY with fewer threads until it fits.
    # preserve_insertion_order=false lets the writer flush without waiting
    # to reproduce source line order; rows are keyed by gid, so order
    # carries no information. The row-count reconciliation is unaffected.
    con.execute("SET preserve_insertion_order=false")
    sql = f"""
        COPY (
            SELECT
                json_extract_string(line, '$.gid') AS gid,
                json_extract_string(line, '$.id')  AS id,
                line AS data
            FROM {read}
            WHERE json_valid(line)
        )
        TO '{sqlstr(out_path)}'
        (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 16384)
    """
    ladder = sorted({args.threads or 14, 8, 4, 2}, reverse=True)
    for threads in ladder:
        try:
            con.execute(f"PRAGMA threads={threads}")
            con.execute(sql)
            if threads != ladder[0]:
                print(f"    (succeeded with {threads} threads)")
            break
        except duckdb.Error as e:
            if "Out of Memory" not in str(e):
                raise
            if out_path.exists():
                out_path.unlink()
            print(
                f"    [retry] out of memory with {threads} threads, retrying with fewer"
            )
    else:
        sys.exit(
            f"json-dumps/{entity}: out of memory even at 2 threads; "
            "rerun with a higher --memory-limit"
        )
    elapsed = time.time() - t0

    stats = con.execute(f"""
        SELECT
            count(*),
            count(*) FILTER (WHERE line IS NOT NULL AND json_valid(line)),
            count(*) FILTER (WHERE line IS NOT NULL AND NOT json_valid(line))
        FROM {read}
    """).fetchone()
    n_raw, n_valid, n_invalid = stats
    n_out = con.execute(
        f"SELECT count(*) FROM read_parquet('{sqlstr(out_path)}')"
    ).fetchone()[0]

    if n_invalid:
        side = out_dir / f"{entity}.invalid_lines.parquet"
        con.execute(
            f"COPY (SELECT line FROM {read} "
            f"WHERE line IS NOT NULL AND NOT json_valid(line)) "
            f"TO '{sqlstr(side)}' (FORMAT PARQUET, COMPRESSION 'zstd')"
        )
        print(f"    -> {n_invalid:,} invalid lines saved to {side.name}")

    unaccounted = n_raw - n_valid - n_invalid
    status = "OK" if (unaccounted == 0 and n_out == n_valid) else "MISMATCH"
    print(
        f"  [{status}] json-dumps/{entity}: {n_out:,} rows, {n_raw:,} lines, "
        f"{n_invalid:,} invalid, {elapsed:,.1f}s"
    )
    if unaccounted:
        print(f"    !! {unaccounted:,} line(s) unaccounted for.")

    return {
        "source": str(path),
        "rows": n_out,
        "raw_lines": n_raw,
        "rejected": n_invalid,
        "output": str(out_path),
        "status": status,
    }


def convert_json_shards(con, entity, info, out_dir: Path, args):
    out_path = out_dir / f"{entity}_shards.parquet"
    if out_path.exists() and not args.overwrite:
        print(f"  [skip] json-dumps/{entity} shards (output exists)")
        return None
    base = (
        f"read_csv('{sqlstr(info['shard_glob'])}', delim = '{SOH}', "
        f"header = false, columns = {{'column0': 'VARCHAR'}}, "
        f"quote = '', escape = '', auto_detect = false, "
        f"compression = 'zstd', filename = true, max_line_size = 100000000)"
    )

    try:
        one = con.execute(f"SELECT column0 FROM {base} LIMIT 1").fetchone()
    except duckdb.Error as e:
        print(f"  [ERROR] json-dumps/{entity} shards: {str(e).splitlines()[0]}")
        return {"source": info["shard_glob"], "status": "error"}
    if one is None:
        print(f"  [empty] json-dumps/{entity} shards")
        return {"source": info["shard_glob"], "status": "empty", "rows": 0}
    if not _json_valid(con, one[0]):
        print(f"  [SKIP] json-dumps/{entity} shards: the first file is not JSON.")
        print("       These directories do not match the official json-dumps layout.")
        print("       Inspect one file by hand before you convert them.")
        return {"source": info["shard_glob"], "status": "skipped"}

    t0 = time.time()
    n_files, n_rows, n_valid, n_invalid = con.execute(f"""
        SELECT
            count(DISTINCT filename),
            count(*),
            count(*) FILTER (WHERE column0 IS NOT NULL AND json_valid(column0)),
            count(*) FILTER (WHERE column0 IS NOT NULL AND NOT json_valid(column0))
        FROM {base}
    """).fetchone()

    if n_files != n_rows:
        print(
            f"  [WARN] json-dumps/{entity} shards: {n_rows:,} lines in {n_files:,} files."
        )
        print(
            "       A file holds more than one line. The slow per-file reader is used."
        )
        return convert_json_shards_python(con, entity, info, out_dir, args)

    sql = f"""
        COPY (
            SELECT
                regexp_extract(filename, '([0-9a-fA-F-]{{36}})\\.txt\\.zst$', 1)
                    AS gid_from_filename,
                json_extract_string(column0, '$.gid') AS gid,
                json_extract_string(column0, '$.id')  AS id,
                column0 AS data
            FROM {base}
            WHERE column0 IS NOT NULL AND json_valid(column0)
        )
        TO '{sqlstr(out_path)}'
        (FORMAT PARQUET, COMPRESSION 'zstd')
    """
    con.execute(sql)
    elapsed = time.time() - t0
    n_out = con.execute(
        f"SELECT count(*) FROM read_parquet('{sqlstr(out_path)}')"
    ).fetchone()[0]
    unaccounted = n_rows - n_valid - n_invalid
    status = "OK" if (unaccounted == 0 and n_out == n_valid) else "MISMATCH"
    print(
        f"  [{status}] json-dumps/{entity} shards: {n_out:,} rows "
        f"from {n_files:,} files, {n_invalid:,} invalid, {elapsed:,.1f}s"
    )
    return {
        "source": info["shard_glob"],
        "rows": n_out,
        "raw_lines": n_rows,
        "rejected": n_invalid,
        "output": str(out_path),
        "status": status,
    }


def convert_json_shards_python(con, entity, info, out_dir: Path, args):
    try:
        import zstandard
    except ImportError:
        print("  [ERROR] Run: pip install zstandard. Then re-run for these files.")
        return {
            "source": info["shard_glob"],
            "status": "error",
            "error": "zstandard not installed",
        }

    out_path = out_dir / f"{entity}_shards.parquet"
    tmp = "shard_" + re.sub(r"[^0-9A-Za-z_]", "_", entity)
    con.execute(
        f"CREATE OR REPLACE TABLE {tmp} (gid_from_filename VARCHAR, data VARCHAR)"
    )
    dctx = zstandard.ZstdDecompressor()
    batch, n = [], 0
    entity_dir = info["path"]
    for sd_name in info["shard_dirs"]:
        for f in sorted((entity_dir / sd_name).glob("*.txt.zst")):
            m = re.search(r"([0-9a-fA-F-]{36})\.txt\.zst\Z", f.name)
            gid = m.group(1) if m else f.name
            with open(f, "rb") as fh:
                text = dctx.stream_reader(fh).read().decode("utf-8", errors="replace")
            batch.append((gid, text))
            if len(batch) >= 5000:
                con.executemany(f"INSERT INTO {tmp} VALUES (?, ?)", batch)
                n += len(batch)
                batch = []
    if batch:
        con.executemany(f"INSERT INTO {tmp} VALUES (?, ?)", batch)
        n += len(batch)

    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT gid_from_filename,
                   json_extract_string(data, '$.gid') AS gid,
                   json_extract_string(data, '$.id')  AS id,
                   data
            FROM {tmp}
            WHERE json_valid(data)
        )
        TO '{sqlstr(out_path)}'
        (FORMAT PARQUET, COMPRESSION 'zstd')
    """)
    elapsed = time.time() - t0
    n_out = con.execute(
        f"SELECT count(*) FROM read_parquet('{sqlstr(out_path)}')"
    ).fetchone()[0]
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
    status = "OK" if n_out <= n else "MISMATCH"
    print(
        f"  [{status}] json-dumps/{entity} shards (python path): "
        f"{n_out:,} rows from {n:,} files, {n - n_out:,} invalid, {elapsed:,.1f}s"
    )
    return {
        "source": info["shard_glob"],
        "rows": n_out,
        "raw_lines": n,
        "rejected": n - n_out,
        "output": str(out_path),
        "status": status,
    }


# Preflight, manifest, main


def preflight(fullexport, jsond, schemas, output_root: Path):
    print("\nPreflight\n---------")
    healthy = True
    total_bytes, total_tables = 0, 0
    for name, ds in fullexport.items():
        b = sum(p.stat().st_size for p in ds["tables"])
        total_bytes += b
        total_tables += len(ds["tables"])
        print(
            f"  fullexport/{name}: {len(ds['tables']):3d} tables, {human(b):>10s}, "
            f"TIMESTAMP {ds['timestamp'] or '?'}"
        )

    stamps = {}
    for name, ds in fullexport.items():
        if ds["timestamp"]:
            stamps.setdefault(ds["timestamp"], []).append(f"fullexport/{name}")
    for ent, e in jsond.items():
        if e["timestamp"]:
            stamps.setdefault(e["timestamp"], []).append(f"json-dumps/{ent}")
        if e["combined"]:
            total_bytes += e["combined"].stat().st_size
    if len(stamps) > 1:
        healthy = False
        print("  !! TIMESTAMP values differ across dump sets:")
        for ts, where in stamps.items():
            print(f"       {ts}  <-  {', '.join(where)}")
        print(
            "     The sets come from different replication days. Use --force to proceed."
        )
    elif stamps:
        print(f"  All dumps share TIMESTAMP {next(iter(stamps))}.")

    seqs = set()
    for ds in fullexport.values():
        if ds["schema_sequence"]:
            seqs.add(ds["schema_sequence"])
    if len(seqs) > 1:
        healthy = False
        print(f"  !! SCHEMA_SEQUENCE values differ: {sorted(seqs)}")

    empty = [
        p.name
        for ds in fullexport.values()
        for p in ds["tables"]
        if p.stat().st_size == 0
    ]
    if empty:
        print(
            f"  note: {len(empty)} zero-byte table files: {', '.join(empty[:8])}"
            + (" ..." if len(empty) > 8 else "")
        )

    for ent, e in jsond.items():
        parts = []
        if e["combined"]:
            parts.append(f"combined file {human(e['combined'].stat().st_size)}")
        if e["n_shard_files"]:
            parts.append(
                f"{e['n_shard_files']:,} per-MBID .txt.zst files "
                f"in dirs {', '.join(e['shard_dirs'])}"
            )
        print(f"  json-dumps/{ent}: " + ". ".join(parts))
        if e["combined"] and e["n_shard_files"]:
            print("       note: both a combined file and per-MBID files exist.")
            print(
                "       The script writes them to separate outputs. Check for overlap"
            )

    if fullexport:
        if schemas:
            matched = sum(
                1
                for ds in fullexport.values()
                for p in ds["tables"]
                if lookup_schema(schemas, p.name)
            )
            print(
                f"  schema names: {matched}/{total_tables} tables have a CREATE TABLE match"
            )
        else:
            print("  schema names: none. Pass --schema-sql for real column names.")

    free = shutil.disk_usage(output_root).free
    est = int(total_bytes * 0.55)
    print(
        f"  input (without per-MBID files): {human(total_bytes)}, "
        f"free disk: {human(free)}, rough output estimate: {human(est)}"
    )
    if free < est:
        healthy = False
        print("  !! Not enough free disk space for a safe run.")
    print()
    return healthy


def save_manifest(output_root: Path, manifest):
    p = output_root / "_manifest.json"
    tmp = output_root / "_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser(
        description="Convert the MusicBrainz dumps into Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to the musicbrainz/ directory "
        "(the one containing fullexport/ and json-dumps/)",
    )
    ap.add_argument("--output", required=True, help="Directory for .parquet files")
    ap.add_argument(
        "--schema-sql",
        action="append",
        default=[],
        help="musicbrainz-server SQL file or directory. Repeatable. "
        "Gives real column names. admin/sql/CreateTables.sql and "
        "friends. Without it, columns are named c0, c1, ...",
    )
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--memory-limit", default=None, help="e.g. '24GB'")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--include",
        default=None,
        help="Regex filter on 'dumpset/table' or 'json-dumps/entity'",
    )
    ap.add_argument(
        "--parts",
        default="fullexport,json",
        help="Comma-separated: fullexport,json (default: both)",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="Convert the three smallest tables and a JSON sample only",
    )
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Run the full conversion even if preflight finds anomalies",
    )
    args = ap.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    if not input_root.is_dir():
        sys.exit(f"No such directory: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    parts = {p.strip() for p in args.parts.split(",")}
    if not parts <= {"fullexport", "json"}:
        sys.exit("--parts accepts only: fullexport, json")

    fullexport = discover_fullexport(input_root)
    jsond = discover_json_dumps(input_root)
    if not fullexport and not jsond:
        sys.exit(f"No fullexport/ or json-dumps/ dumps found under {input_root}")

    print(f"duckdb {duckdb.__version__}")
    schemas = load_schemas(args.schema_sql) if args.schema_sql else {}

    con = duckdb.connect()
    if args.threads:
        con.execute(f"PRAGMA threads={args.threads}")
    if args.memory_limit:
        con.execute(f"SET memory_limit='{args.memory_limit}'")

    decoder = install_unescape_macro(con)
    mode = probe_csv_mode(con, output_root, decoder)
    cfg = {"schemas": schemas, "mode": mode, "decoder": decoder}

    if not args.skip_preflight and not args.test:
        if not preflight(fullexport, jsond, schemas, output_root):
            if args.force:
                print("  --force given: proceeding anyway.\n")
            else:
                sys.exit(
                    "Refusing to run (use --force to override, or fix the data first)."
                )

    include_re = re.compile(args.include) if args.include else None
    manifest = {
        "input": str(input_root),
        "duckdb_version": duckdb.__version__,
        "csv_null_mode": mode,
        "escape_decoder": decoder,
        "schema_sql": [str(s) for s in args.schema_sql],
        "tables": {},
        "json": {},
    }
    _mp = output_root / "_manifest.json"
    if _mp.exists():
        try:
            _old = json.loads(_mp.read_text())
            manifest["tables"].update(_old.get("tables", {}))
            manifest["json"].update(_old.get("json", {}))
            print(
                f"  resumed manifest: {len(manifest['tables'])} table entries carried over"
            )
        except (OSError, ValueError):
            pass
    totals = {"raw": 0, "out": 0, "rej": 0}

    if args.test:
        test_dir = output_root / "_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        print("[test mode] converting the three smallest tables and one JSON sample\n")
        all_tables = [(p, ds) for ds, d in fullexport.items() for p in d["tables"]]
        all_tables.sort(key=lambda t: t[0].stat().st_size)
        for p, ds in all_tables[:3]:
            if p.stat().st_size == 0:
                continue
            convert_table(con, cfg, ds, p, test_dir / ds, args)
        if all_tables:
            p = all_tables[0][0]
            with open(p, "rb") as f:
                first = f.read(4096).split(b"\n")[0]
            print(f"\n  first raw line of {p.name}:\n    {first[:200]!r}")
            out = test_dir / all_tables[0][1] / (p.name + ".parquet")
            if out.exists():
                print("  same line after conversion:")
                print(
                    f"    {con.execute(f'SELECT * FROM read_parquet(\'{sqlstr(out)}\') LIMIT 1').fetchall()}"
                )
        for ent, e in jsond.items():
            if e["combined"]:
                sample = test_dir / f"json_{ent}.parquet"
                read = (
                    f"read_csv('{sqlstr(e['combined'])}', delim = '{SOH}', "
                    f"header = false, columns = {{'line': 'VARCHAR'}}, "
                    f"quote = '', escape = '', auto_detect = false, max_line_size = 100000000)"
                )
                con.execute(f"""
                    COPY (SELECT json_extract_string(line, '$.gid') AS gid,
                                 json_extract_string(line, '$.id') AS id,
                                 line AS data
                          FROM {read} WHERE json_valid(line) LIMIT 2000)
                    TO '{sqlstr(sample)}' (FORMAT PARQUET, COMPRESSION 'zstd')
                """)
                print(
                    f"  JSON sample ({ent}): "
                    f"{con.execute(f'SELECT count(*) FROM read_parquet(\'{sqlstr(sample)}\')').fetchone()[0]:,} rows"
                )
                break
        print("\nInspect the output under _test/ before the full run.")
        return

    if "fullexport" in parts and fullexport:
        print(
            f"Converting fullexport: {sum(len(d['tables']) for d in fullexport.values())} tables\n"
        )
        for ds_name, ds in fullexport.items():
            out_dir = output_root / "fullexport" / ds_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in ds["tables"]:
                rel = f"{ds_name}/{p.name}"
                if include_re and not include_re.search(rel):
                    continue
                entry = convert_table(con, cfg, ds_name, p, out_dir, args)
                if entry is None:
                    continue
                manifest["tables"][rel] = entry
                save_manifest(output_root, manifest)
                if entry.get("status") in ("OK", "MISMATCH"):
                    totals["raw"] += entry["raw_lines"]
                    totals["out"] += entry["rows"]
                    totals["rej"] += entry["rejected"]

    if "json" in parts and jsond:
        print(f"\nConverting json-dumps: {len(jsond)} entities\n")
        out_dir = output_root / "json-dumps"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ent, info in jsond.items():
            if include_re and not include_re.search(f"json-dumps/{ent}"):
                continue
            entries = {}
            if info["combined"]:
                entries["combined"] = convert_json_combined(
                    con, ent, info, out_dir, args
                )
            if info["shard_glob"]:
                entries["shards"] = convert_json_shards(con, ent, info, out_dir, args)
            for k, e in entries.items():
                if e is None:
                    continue
                manifest["json"][f"{ent}/{k}"] = e
                if e.get("status") in ("OK", "MISMATCH"):
                    totals["raw"] += e["raw_lines"]
                    totals["out"] += e["rows"]
                    totals["rej"] += e["rejected"]
            save_manifest(output_root, manifest)

    print()
    print(
        f"TOTAL: {totals['out']:,} rows written, {totals['raw']:,} raw lines, "
        f"{totals['rej']:,} rejected"
    )
    unaccounted = totals["raw"] - totals["out"] - totals["rej"]
    if unaccounted == 0:
        print("Every raw line is written or logged as a reject.")
    else:
        print(f"!! {unaccounted:,} line(s) unaccounted for.")
    bad = [
        k
        for k, v in {**manifest["tables"], **manifest["json"]}.items()
        if v.get("status") not in ("OK", None)
    ]
    if bad:
        print(f"Tables that need attention: {', '.join(bad)}")
    print(f"Manifest: {output_root / '_manifest.json'}")

    print(f"""
Read the result with DuckDB. Read one table per query. The schemas differ.
  SELECT * FROM read_parquet('{output_root}/fullexport/mbdump/artist.parquet') LIMIT 5;
JSON dumps. The data column holds the complete original JSON text.
  SELECT gid, json_extract_string(data, '$.name') AS name
  FROM read_parquet('{output_root}/json-dumps/artist.parquet') LIMIT 5;
""")


if __name__ == "__main__":
    main()

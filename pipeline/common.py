"""Shared mechanics: connection, thread ladder, atomic writes, lock, markers."""

import contextlib
import hashlib
import json
import logging
import os
import re
import resource
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .config import DUCKDB_SETTINGS, FEATURES_DIR, MODEL_DIR

log = logging.getLogger("pipeline")


class VerificationError(RuntimeError):
    """A frozen number or an invariant failed. Stop and investigate."""


class ResourceExhausted(RuntimeError):
    """The thread ladder could not finish within the memory limit."""


class LockHeld(RuntimeError):
    """Another live pipeline process owns the lock."""


_MEMORY_HINT = re.compile(r"memory|out of|oom|spill|temp directory", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(threads: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    for key, value in DUCKDB_SETTINGS.items():
        con.execute(f"SET {key}='{value}'")
    return con


def check(condition, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def with_thread_ladder(fn, ladder):
    """Run fn(con) at each thread count until success.

    fn must return a value that holds no open cursor, because the runner
    closes the connection after each attempt.
    Only memory-class DuckDB errors retry. All other errors stop the run.
    """
    attempts = list(ladder) or [8, 4, 2, 1]
    last_error = None
    for index, threads in enumerate(attempts):
        con = connect(threads)
        try:
            return fn(con)
        except duckdb.Error as exc:
            if not _MEMORY_HINT.search(str(exc)):
                raise
            last_error = exc
            nxt = attempts[index + 1] if index + 1 < len(attempts) else None
            log.warning(
                "memory-class error at %d threads: %s",
                threads,
                str(exc).splitlines()[0][:200],
            )
            if nxt:
                log.warning("retry with %d threads", nxt)
        finally:
            con.close()
    raise ResourceExhausted(
        f"thread ladder {attempts} exhausted. Last error: {last_error}"
    )


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require_space(directory: Path, est_bytes: int, margin: float = 1.2) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    need = int(est_bytes * margin)
    if free < need:
        raise ResourceExhausted(
            f"free space {free / 1e9:.1f} GB is below the required "
            f"{need / 1e9:.1f} GB at {directory}"
        )


def atomic_parquet_write(con, sql: str, dest: Path, expected_mb: int) -> None:
    """Write query output to dest. A crash leaves no partial file."""
    require_space(dest.parent, expected_mb * 1024 * 1024)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _fsync_path(tmp)
    os.replace(tmp, dest)
    _fsync_path(dest.parent)
    log.info("wrote %s", dest)


def atomic_json_write(doc, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    payload = json.dumps(doc, indent=2, sort_keys=True, default=str)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dest)
    _fsync_path(dest.parent)


def atomic_text_write(text: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dest)
    _fsync_path(dest.parent)


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, cache: Path) -> str:
    """SHA256 of the file bytes, cached on size and mtime."""
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    store = load_json(cache) or {}
    if store.get(key):
        return store[key]
    digest = file_sha256(path)
    store[key] = digest
    atomic_json_write(store, cache)
    return digest


def content_hash(con, sql: str, order_by: str) -> str:
    """Hash the content of a table in a thread-independent way.

    File bytes are not the contract. Rows stream in the given order.
    The order must be total: include every key column.
    """
    digest = hashlib.sha256()
    cur = con.execute(f"SELECT * FROM {sql} ORDER BY {order_by}")
    while True:
        rows = cur.fetchmany(65536)
        if not rows:
            break
        for row in rows:
            digest.update(repr(row).encode("utf-8"))
    return digest.hexdigest()


def params_digest(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()


def features_table(name: str) -> str:
    exact = FEATURES_DIR / f"{name}.parquet"
    if exact.exists():
        return str(exact)
    matches = [
        p for p in FEATURES_DIR.glob(f"{name}*.parquet") if not p.name.endswith(".tmp")
    ]
    if len(matches) != 1:
        raise VerificationError(
            f"features table '{name}': expected one parquet file, "
            f"found {sorted(p.name for p in matches)}"
        )
    return str(matches[0])


# 2005-01-01 UTC .. 2013-12-31 23:59:59 UTC, the measured listening window.
_EPOCH_S = (1104537600, 1388534399)
_EPOCH_MS = (1104537600000, 1388534399999)


def window_pred(column: str, ts_mode: str) -> str:
    """SQL predicate: the timestamp falls inside the 2005-2013 window."""
    if ts_mode == "timestamp_naive_utc":
        return (
            f"{column} BETWEEN TIMESTAMP '2005-01-01' "
            f"AND TIMESTAMP '2013-12-31 23:59:59'"
        )
    if ts_mode == "epoch_s":
        lo, hi = _EPOCH_S
    elif ts_mode == "epoch_ms":
        lo, hi = _EPOCH_MS
    else:
        raise VerificationError(f"unknown ts mode: {ts_mode}")
    return f"CAST({column} AS BIGINT) BETWEEN {lo} AND {hi}"


def year_expr(column: str, ts_mode: str) -> str:
    if ts_mode == "timestamp_naive_utc":
        return f"year({column})"
    if ts_mode == "epoch_s":
        return f"date_part('year', to_timestamp(CAST({column} AS BIGINT)))"
    if ts_mode == "epoch_ms":
        return f"date_part('year', to_timestamp(CAST({column} AS BIGINT) / 1000.0))"
    raise VerificationError(f"unknown ts mode: {ts_mode}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def pipeline_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            other = load_json(lock_path) or {}
            pid = other.get("pid")
            if isinstance(pid, int) and _pid_alive(pid):
                raise LockHeld(f"another pipeline run is active, pid {pid}")
            log.warning("stale lock from dead pid %r, the runner removes it", pid)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    if fd is None:
        raise LockHeld("the lock file could not be acquired")
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": utc_now()}))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        current = load_json(lock_path) or {}
        if current.get("pid") == os.getpid():
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def marker_path(stage_name: str) -> Path:
    return MODEL_DIR / "markers" / f"{stage_name}.json"


def read_marker(stage_name: str):
    return load_json(marker_path(stage_name))


def write_marker(stage_name, digest, inputs, outputs, row_counts=None, source_sha=None):
    doc = {
        "stage": stage_name,
        "params": digest,
        "source_sha": source_sha,
        "inputs": inputs,
        "outputs": outputs,
        "row_counts": row_counts or {},
        "peak_rss_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_048_576, 2
        ),
        "duckdb_version": duckdb.__version__,
        "finished_utc": utc_now(),
    }
    atomic_json_write(doc, marker_path(stage_name))
    log.info("marker written for %s", stage_name)

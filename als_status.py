"""Show the state of the ALS search (als_tune.py).

It only reads files, so it is safe to run while als_tune.py runs. It also
works for a run that prints nothing.

Usage, from the repository root:
    python3 als_status.py
"""

import json
import os
import re
import time
from pathlib import Path

from pipeline.config import MODEL_DIR, SCRATCH_DIR

RESULTS = MODEL_DIR / "als_tuning.json"
REPORT = MODEL_DIR / "reports" / "als_tuning.md"
LOG = MODEL_DIR / "logs" / "als_tune.log"
CACHE = SCRATCH_DIR / "csr_item_major"
WORK = SCRATCH_DIR / "als_tune"
NAME = re.compile(r"alpha([0-9.e+-]+)_reg([0-9.e+-]+)_f(\d+)$")


def ago(path):
    s = int(max(time.time() - path.stat().st_mtime, 0))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return (f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s") + " ago"


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tune_processes():
    """(main process ids, solver worker count) for python processes that
    run als_tune.py. The solver workers are forked children with the same
    command line. They exist only while a half-step runs."""
    procs = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
            stat = (d / "stat").read_text()
        except OSError:
            continue
        if any(a.endswith(b"als_tune.py") for a in argv) or (
            b"-m" in argv and (b"als_tune" in argv or b"pipeline.als_tune" in argv)
        ):
            procs[int(d.name)] = int(stat.rsplit(")", 1)[1].split()[1])
    mains = sorted(p for p, parent in procs.items() if parent not in procs)
    return mains, len(procs) - len(mains)


def transpose_state():
    meta = CACHE / "meta.json"
    if meta.exists():
        return f"built {ago(meta)}"
    tmp = sorted(CACHE.glob("*.tmp.npy")) if CACHE.exists() else []
    if tmp:
        newest = max(tmp, key=lambda p: p.stat().st_mtime)
        return (
            f"not finished. Temporary files were last written "
            f"{ago(newest)}. A stopped run builds it again from the "
            f"start."
        )
    return "not built yet"


def config_lines(configs, running):
    names = set(configs)
    if WORK.exists():
        names |= {d.name for d in WORK.iterdir() if d.is_dir()}

    def order(name):
        m = NAME.match(name)
        alpha, reg = (float(m.group(1)), float(m.group(2))) if m else (0, 0)
        entry = configs.get(name)
        # A config without an entry yet is the one running now, and only
        # phase C is left.
        phase = (entry or {}).get("phase") or "C"
        position = (entry or {}).get("position", 0 if entry else 10**6)
        return (phase, position, alpha, reg, name)

    lines = []
    for name in sorted(names, key=order):
        entry = configs.get(name, {})
        state = load(WORK / name / "fit_state.json") or {}
        history = state.get("history", [])
        phase = order(name)[0]
        if entry.get("finished"):
            lines.append(
                f"  {name} (phase {phase}): finished. "
                f"{entry['iterations_run']} iterations, best "
                f"{entry['best_iteration']}, score {entry['score']:.5f}, "
                f"val_last R@100 "
                f"{entry['best_val_last']['recall_at_100']:.4f}, "
                f"val_first R@100 "
                f"{entry['best_val_first']['recall_at_100']:.4f}"
            )
            continue
        scores = [h["score"] for h in history if "score" in h]
        fit_s = [round(h["seconds"]) for h in history]
        text = (
            f"  {name} (phase {phase}): "
            f"{'running' if running else 'stopped'}, "
            f"{len(history)} iterations committed"
        )
        if scores:
            text += ", scores " + " ".join(f"{s:.5f}" for s in scores)
        if fit_s:
            text += (
                ", fit seconds per iteration "
                + " ".join(str(s) for s in fit_s)
                + " (the ranking adds to this)"
            )
        work = list((WORK / name).glob("*.work.npy"))
        if work:
            newest = max(work, key=lambda p: p.stat().st_mtime)
            text += (
                f". Iteration {len(history) + 1} files last written " f"{ago(newest)}"
            )
        lines.append(text)
    return lines


def main():
    print(
        f"ALS search status, "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC"
    )
    mains, workers = tune_processes()
    if mains:
        busy = (
            f", {workers} solver workers, so a half-step is running"
            if workers
            else ", no solver workers, so it is ranking, " "loading or preparing"
        )
        print(
            f"als_tune.py: running, process id " f"{', '.join(map(str, mains))}{busy}"
        )
    else:
        print("als_tune.py: not running")
    print(f"Transposed item-major CSR: {transpose_state()}")

    results = load(RESULTS) or {}
    configs = results.get("configs", {})
    lines = config_lines(configs, bool(mains))
    if RESULTS.exists():
        print(f"Search results: {RESULTS} (updated {ago(RESULTS)})")
    print("Configs:" if lines else "Configs: none started yet")
    for line in lines:
        print(line)
    ref = results.get("reference")
    if ref:
        print(f"Popularity reference: score {ref['score']:.5f}")
    ab = results.get("winner_phases_ab")
    if ab:
        print(
            f"Winner of phases A and B: alpha {ab['alpha']:g}, reg "
            f"{ab['reg']:g}, {ab['iterations']} iterations, score "
            f"{ab['score']:.5f}"
        )
    win = results.get("winner")
    if win:
        print(
            f"Winner: alpha {win['alpha']:g}, reg {win['reg']:g}, "
            f"{win['iterations']} iterations, score {win['score']:.5f}"
        )
        print(
            "als_params.json for the final P6 fit: "
            + json.dumps(win["als_params_json"])
        )
    print(
        f"Report: {REPORT} ("
        + (f"written {ago(REPORT)})" if REPORT.exists() else "not written yet)")
    )
    if LOG.exists():
        tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        print(f"Log: {LOG} (last written {ago(LOG)}), last lines:")
        for line in tail:
            print("  " + line)
    else:
        print(f"Log: {LOG} does not exist.")


if __name__ == "__main__":
    main()

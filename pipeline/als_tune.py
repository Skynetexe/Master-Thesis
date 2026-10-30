"""Select the ALS hyperparameters on validation picks.

Pre-registered protocol, fixed before any result exists:

- Users: the selection sample from pipeline/selection.py. It holds
  20,000 users and does not overlap the eval sample. This script never
  reads a test pick.
- Target: each selection user's val_last and val_first picks, ranked
  under the P5 harness rules (full catalog minus train items, ties by
  item_id ascending).
- Metric: the mean of NDCG@100 over the two pick types.
- Search, phase A: alpha in {1, 5, 20, 40}, with reg 1.
- Search, phase B: reg in {0.1, 10}, at the best alpha from phase A.
- Fixed: factors 64 and seed 20260907 (P6's values).
- Iterations: up to 12. The metric is measured after each one. A config
  stops when the metric has not improved for 2 iterations. Its score is
  the score of its best iteration.
- Winner: the highest score over all configs. A tie goes to the config
  that ran first.
- Reference: popularity on the same picks. It is for comparison only and
  takes no part in the choice.

Phase C, added after phases A and B finished and before any
phase C result existed. Phases A and B put the winner at the smallest
alpha (1) and the largest reg (10) searched, so a better value can lie
beyond the grid on both axes. Rule: for each axis where the A-B winner
sits at an edge of the values searched, try the values beyond that edge
in the order listed in PHASE_C, with the other axis held at its best
value so far. An axis stops at the first value that does not beat the
best score so far. reg goes first, then alpha. The winner is the best
config over phases A, B and C. The A-B winner stays in the results for
the record.

Data: the user-major CSR from P6, and an item-major CSR built here as its
exact transpose (cached in SCRATCH_DIR/csr_item_major). P6's own
item-major CSR is measured and logged, not used: its per-file build put
other items' listeners into every item row.

Outputs:
- MODEL_DIR/als_tuning.json, updated after every iteration.
- MODEL_DIR/reports/als_tuning.md, written at the end.

A stopped run continues where it stopped: run the same command again.

Output: every line goes to the console and to MODEL_DIR/logs/als_tune.log.
Long steps print a progress line at most every 30 seconds, with the time
left. To see the state of a run from another terminal, without changing
anything: python3 als_status.py

Usage, from the repository root. This file is a module of the
pipeline package, not a runner stage:
    python3 -m pipeline.als_tune                 run or continue the search
    python3 -m pipeline.als_tune --write-params  the same, then write the
        winner to CURATED/als_params.json for P6
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from pipeline import als
from pipeline import evaluation as ev
from pipeline import selection
from pipeline.common import atomic_json_write, atomic_text_write, connect, log
from pipeline.config import CURATED, MODEL_DIR, SCRATCH_DIR

LOG_PATH = MODEL_DIR / "logs" / "als_tune.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"  # as in runner.py

PHASE_A_ALPHAS = (1.0, 5.0, 20.0, 40.0)
PHASE_A_REG = 1.0
PHASE_B_REGS = (0.1, 10.0)
FACTORS = 64
SEED = 20260907
MAX_ITER = 12
PATIENCE = 2

PROTOCOL = {
    "users": f"selection sample, {selection.SIZE} users, seed "
    f"{selection.SEED}, disjoint from the eval sample",
    "target": "val_last and val_first picks, P5 harness rules",
    "metric": "mean NDCG@100 over val_last and val_first",
    "phase_a": {"alpha": list(PHASE_A_ALPHAS), "reg": PHASE_A_REG},
    "phase_b": {"reg": list(PHASE_B_REGS), "alpha": "best of phase A"},
    "fixed": {"factors": FACTORS, "seed": SEED},
    "iterations": {"max": MAX_ITER, "patience": PATIENCE},
    "winner": "highest score, ties to the earlier config",
}

PHASE_C = {
    "declared": "after phases A and B finished and before any " "phase C result",
    "reason": "phases A and B put the winner at the smallest alpha (1) and "
    "the largest reg (10) searched",
    "rule": "for each axis where the A-B winner sits at an edge of the "
    "values searched, try the values beyond that edge in the order "
    "listed, with the other axis at its best value so far. An axis "
    "stops at the first value that does not beat the best score so "
    "far. reg goes first, then alpha. The winner is the best config "
    "over phases A, B and C.",
    "reg_up": [100.0, 1000.0, 10000.0],
    "reg_down": [0.01, 0.001],
    "alpha_up": [80.0, 160.0],
    "alpha_down": [0.25, 0.0],
}


def _config_name(alpha, reg, factors):
    return f"alpha{alpha:g}_reg{reg:g}_f{factors}"


def _best_index(scores):
    """Index of the highest score. A tie goes to the earliest."""
    best = 0
    for k, s in enumerate(scores):
        if s > scores[best]:
            best = k
    return best


def score_picks(ranks):
    last = ev.metric_block(ranks[:, 0])
    first = ev.metric_block(ranks[:, 1])
    return (last["ndcg_at_100"] + first["ndcg_at_100"]) / 2.0, last, first


def _winner_record(e):
    return {
        "alpha": e["alpha"],
        "reg": e["reg"],
        "factors": e["factors"],
        "seed": e["seed"],
        "iterations": e["best_iteration"],
        "score": e["score"],
        "phase": e["phase"],
        "als_params_json": {
            "factors": e["factors"],
            "iterations": e["best_iteration"],
            "regularization": e["reg"],
            "alpha": e["alpha"],
            "seed": e["seed"],
        },
    }


def run_search(
    inter,
    users,
    picks,
    results_path,
    work_dir,
    reference=None,
    alphas=PHASE_A_ALPHAS,
    reg_a=PHASE_A_REG,
    regs_b=PHASE_B_REGS,
    factors=FACTORS,
    seed=SEED,
    max_iter=MAX_ITER,
    patience=PATIENCE,
    workers=None,
    say=print,
    phase_c=PHASE_C,
):
    """The protocol above, with its constants as parameters so that
    test_als.py can run it on a small synthetic catalog. phase_c=None
    runs phases A and B only."""
    results_path, work_dir = Path(results_path), Path(work_dir)
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {}
    )
    results.setdefault("protocol", PROTOCOL)
    results.setdefault("configs", {})
    if reference is not None:
        # Keep the first measurement of P6's old item-major CSR. After the
        # P6 patch, P6's files are the exact transpose and would score 1.0.
        first = (results.get("reference") or {}).get("p6_item_major_real_fraction")
        results["reference"] = dict(reference)
        if first is not None:
            results["reference"]["p6_item_major_real_fraction"] = first

    def save():
        atomic_json_write(results, results_path)

    n_total = len(alphas) + len(regs_b)
    if phase_c:
        n_total += max(len(phase_c["reg_up"]), len(phase_c["reg_down"])) + max(
            len(phase_c["alpha_up"]), len(phase_c["alpha_down"])
        )
    position = {"config": 0}
    iteration_seconds = []  # fit plus ranking, for the time-left estimate

    def run_config(alpha, reg, phase):
        position["config"] += 1
        k = position["config"]
        name = _config_name(alpha, reg, factors)
        done = results["configs"].get(name)
        if done and done.get("finished"):
            say(
                f"als_tune: config {k} of at most {n_total}: {name} already "
                f"finished, score {done['score']:.5f}"
            )
            return done
        say(
            f"als_tune: config {k} of at most {n_total}: {name} (phase "
            f"{phase}) started, at most {max_iter} iterations"
        )
        params = als.ALSParams(
            factors=factors, reg=reg, alpha=alpha, iterations=max_iter, seed=seed
        )
        cfg_dir = work_dir / name

        def on_iteration(iteration, X, Y, history):
            t0 = time.time()
            ranks = als.rank_items(
                X,
                Y,
                inter,
                users,
                picks,
                log=say,
                label=f"als_tune: {name} iteration {iteration}, ranking " f"val picks",
            )
            score, last, first = score_picks(ranks)
            scores = [h["score"] for h in history[:-1]] + [score]
            best = _best_index(scores)
            since = len(scores) - 1 - best
            stop = since >= patience
            results["configs"][name] = {
                "phase": phase,
                "alpha": alpha,
                "reg": reg,
                "factors": factors,
                "seed": seed,
                "finished": False,
                "position": k,
                "iterations_run": iteration,
                "scores": scores,
            }
            save()
            iteration_seconds.append(history[-1]["seconds"] + (time.time() - t0))
            left = (0 if stop else max_iter - iteration) + (n_total - k) * max_iter
            bound = left * sum(iteration_seconds) / len(iteration_seconds)
            say(
                f"als_tune: config {k} of at most {n_total}, {name}, "
                f"iteration {iteration}: score {score:.5f}, best "
                f"{scores[best]:.5f} at "
                f"iteration {best + 1}, no gain for {since} of {patience}. "
                f"val_last R@100 {last['recall_at_100']:.4f}, val_first "
                f"R@100 {first['recall_at_100']:.4f}. Search: at most "
                f"{als._clock(bound)} left (early stopping usually ends "
                f"sooner)."
            )
            return {"score": score, "val_last": last, "val_first": first, "stop": stop}

        _, _, history = als.fit(
            inter,
            params,
            cfg_dir / "user_factors.npy",
            cfg_dir / "item_factors.npy",
            cfg_dir / "fit_state.json",
            workers=workers,
            on_iteration=on_iteration,
            log=say,
        )
        scores = [h["score"] for h in history]
        best = _best_index(scores)
        entry = {
            "phase": phase,
            "alpha": alpha,
            "reg": reg,
            "factors": factors,
            "seed": seed,
            "finished": True,
            "position": k,
            "iterations_run": len(history),
            "best_iteration": best + 1,
            "score": scores[best],
            "scores": scores,
            "best_val_last": history[best]["val_last"],
            "best_val_first": history[best]["val_first"],
            "seconds_per_iteration": [h["seconds"] for h in history],
            "max_relative_rise": max(
                (
                    h["max_relative_rise"]
                    for h in history
                    if h["max_relative_rise"] is not None
                ),
                default=None,
            ),
        }
        results["configs"][name] = entry
        save()
        for f in ("user_factors.npy", "item_factors.npy"):
            (cfg_dir / f).unlink(missing_ok=True)
        say(
            f"als_tune: config {k} of at most {n_total}: {name} finished, "
            f"best iteration {best + 1}, score {scores[best]:.5f}"
        )
        return entry

    order = []
    for alpha in alphas:
        order.append(run_config(alpha, reg_a, "A"))
    best_alpha = order[_best_index([e["score"] for e in order])]["alpha"]
    for reg in regs_b:
        order.append(run_config(best_alpha, reg, "B"))
    win_ab = order[_best_index([e["score"] for e in order])]

    if phase_c:
        results["protocol_extension"] = {"phase_c": phase_c}
        results["winner_phases_ab"] = _winner_record(win_ab)
        results.pop("winner", None)  # set again once phase C finishes
        save()
        best = {"entry": win_ab}

        def walk(values, config_of):
            for v in values:
                entry = run_config(*config_of(v), "C")
                order.append(entry)
                if entry["score"] > best["entry"]["score"]:
                    best["entry"] = entry
                else:
                    break

        ab = list(order)
        regs_seen = sorted({e["reg"] for e in ab if e["alpha"] == win_ab["alpha"]})
        if win_ab["reg"] == regs_seen[-1]:
            walk(phase_c["reg_up"], lambda r: (win_ab["alpha"], r))
        elif win_ab["reg"] == regs_seen[0]:
            walk(phase_c["reg_down"], lambda r: (win_ab["alpha"], r))
        alphas_seen = sorted({e["alpha"] for e in ab})
        best_reg = best["entry"]["reg"]
        if win_ab["alpha"] == alphas_seen[0]:
            walk(phase_c["alpha_down"], lambda a: (a, best_reg))
        elif win_ab["alpha"] == alphas_seen[-1]:
            walk(phase_c["alpha_up"], lambda a: (a, best_reg))

    win = order[_best_index([e["score"] for e in order])]
    results["winner"] = _winner_record(win)
    save()
    return results


def write_report(results, path):
    lines = ["# ALS hyperparameter selection", ""]
    add = lines.append
    p = results["protocol"]
    add(f"- users: {p['users']}")
    add(f"- metric: {p['metric']}")
    add(
        f"- search: phase A alpha {p['phase_a']['alpha']} at reg "
        f"{p['phase_a']['reg']}, then phase B reg {p['phase_b']['reg']} "
        f"at the best alpha"
    )
    add(
        f"- iterations: up to {p['iterations']['max']}, patience "
        f"{p['iterations']['patience']}"
    )
    ext = (results.get("protocol_extension") or {}).get("phase_c")
    if ext:
        add(
            f"- phase C, declared {ext['declared']}: {ext['reason']}. "
            f"Rule: {ext['rule']} Values: reg up {ext['reg_up']}, reg down "
            f"{ext['reg_down']}, alpha up {ext['alpha_up']}, alpha down "
            f"{ext['alpha_down']}. Alpha 0 gives every observed pair the "
            f"same confidence, so listen counts play no part."
        )
    add("")
    add(
        "| phase | alpha | reg | best iteration | score | val_last R@100 "
        "| val_first R@100 | largest objective rise |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for e in sorted(
        results["configs"].values(),
        key=lambda e: (e["phase"], e.get("position", 0), e["alpha"], e["reg"]),
    ):
        if not e.get("finished"):
            continue
        rise = e["max_relative_rise"]
        add(
            f"| {e['phase']} | {e['alpha']:g} | {e['reg']:g} | "
            f"{e['best_iteration']} | {e['score']:.5f} | "
            f"{e['best_val_last']['recall_at_100']:.4f} | "
            f"{e['best_val_first']['recall_at_100']:.4f} | "
            f"{'n/a' if rise is None else f'{rise:.1e}'} |"
        )
    ref = results.get("reference")
    if ref:
        add("")
        add(
            f"Popularity on the same picks: score {ref['score']:.5f}, "
            f"val_last R@100 {ref['val_last']['recall_at_100']:.4f}, "
            f"val_first R@100 {ref['val_first']['recall_at_100']:.4f}."
        )
        if "p6_item_major_real_fraction" in ref:
            add("")
            add(
                f"P6's item-major CSR: "
                f"{ref['p6_item_major_real_fraction']:.2%} of sampled "
                f"entries were real listener pairs. This search used the "
                f"exact transpose of the user-major CSR instead."
            )
    win = results.get("winner")
    ab = results.get("winner_phases_ab")
    if ab:
        add("")
        add(
            f"Winner of phases A and B: alpha {ab['alpha']:g}, reg "
            f"{ab['reg']:g}, {ab['iterations']} iterations, score "
            f"{ab['score']:.5f}."
        )
    if win:
        add("")
        add(
            f"Winner: alpha {win['alpha']:g}, reg {win['reg']:g}, "
            f"{win['iterations']} iterations, score {win['score']:.5f}."
        )
        add("")
        add(
            "A negative objective rise means the objective fell at every "
            "half-step, as exact ALS requires."
        )
    atomic_text_write("\n".join(lines) + "\n", path)


def setup_logging(path):
    """Send INFO lines to the console and append them to path.

    runner.py sets up logging for the pipeline stages. This script runs on
    its own, and Python's default drops INFO lines, so without this call
    the run prints nothing. Each handler is added once: a second call, or
    a console handler that already exists, does not double the output."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(LOG_FORMAT)
    fmt.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    if not any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
        for h in root.handlers
    ):
        to_file = logging.FileHandler(path, encoding="utf-8")
        to_file.setFormatter(fmt)
        root.addHandler(to_file)
    if not (log.isEnabledFor(logging.INFO) and log.hasHandlers()):
        raise SystemExit(
            "als_tune: the pipeline logger still drops INFO "
            "lines, so this run would print nothing. Stopping."
        )


def _stop(message):
    log.error(message)
    raise SystemExit(1)


def blas_setup():
    """One log line on the BLAS library. Stops when threadpoolctl is
    missing: pipeline/als.py then cannot hold each solver process to one
    BLAS thread, and 16 processes that each start a full set of BLAS
    threads slow each other down."""
    if als.threadpool_limits.__module__.split(".")[0] != "threadpoolctl":
        _stop(
            "als_tune: threadpoolctl is not installed, so the solver "
            "processes cannot be held to one BLAS thread each. Install it "
            "in the venv (pip install threadpoolctl), then run again."
        )
    from threadpoolctl import threadpool_info

    libs = [
        f"{d['internal_api']} {d['version']} ({d['num_threads']} " f"threads)"
        for d in threadpool_info()
        if d["user_api"] == "blas"
    ]
    return (
        f"als_tune: BLAS {', '.join(libs) or 'not found'}. The solver "
        f"uses 1 thread per process. The ranking uses "
        f"{os.cpu_count()}."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ALS hyperparameter search on the selection users' val "
        "picks (protocol in the module docstring)"
    )
    parser.add_argument(
        "--write-params",
        action="store_true",
        help="after the search, write the winner to "
        "CURATED/als_params.json, the file P6 reads",
    )
    args = parser.parse_args(argv)
    setup_logging(LOG_PATH)
    try:
        _run(write_params=args.write_params)
    except KeyboardInterrupt:
        log.info(
            "als_tune: stopped by Ctrl+C. Run python3 -m pipeline.als_tune "
            "again to continue from the last finished iteration."
        )
        raise SystemExit(130)
    except Exception:
        log.exception("als_tune: stopped by an error")
        raise SystemExit(1)


def _run(write_params=False):
    say = log.info
    t0 = time.time()
    say(f"als_tune: started, process id {os.getpid()}, log file {LOG_PATH}")
    say(blas_setup())
    p6 = als.Interactions.from_p6(MODEL_DIR)
    say(
        f"als_tune: {p6.n_users:,} user slots, {p6.n_items:,} item "
        f"slots, {p6.nnz:,} train pairs"
    )
    p6_real = als.orientation_real_fraction(p6, rows=500, per_row=200)
    say(
        f"als_tune: P6's item-major CSR: {p6_real:.2%} of sampled entries "
        f"are real listener pairs (1.0 means correct)"
    )
    say("als_tune: fingerprinting the user-major CSR (reads it once in full)")
    p6.fingerprint(log=say)
    inter = p6.with_transposed_item_major(SCRATCH_DIR / "csr_item_major", log=say)
    real = als.orientation_real_fraction(inter, rows=500, per_row=200, seed=1)
    if real < 1.0:
        _stop(
            f"als_tune: the transposed item-major CSR scored {real:.2%}, "
            f"not 1.0. Stopping."
        )
    say(
        "als_tune: transposed item-major CSR verified: 100% of sampled "
        "entries are real listener pairs"
    )
    say(
        "als_tune: loading the selection users, their val picks and the "
        "train popularity with DuckDB (no progress lines in this step)"
    )
    con = connect(4)
    try:
        users = selection.selection_user_ids(con)
        picks = selection.val_picks(con, users)
        n_slots, id_min = ev.load_id_slots()
        pop = ev.popularity_scores(con, n_slots, id_min)
    finally:
        con.close()
    say(
        f"als_tune: {len(users):,} selection users, {picks.size:,} val "
        f"picks. Ranking them by popularity for the reference score."
    )

    train = {int(u): inter.train_items(int(u)) for u in users}
    pop_ranks = ev.global_ranks(pop)
    ranks_pop = np.stack(
        [ev.user_ranks(pop_ranks, picks[:, j], users, train) for j in range(2)], axis=1
    )
    score, last, first = score_picks(ranks_pop)
    reference = {
        "model": "popularity: train listens per item",
        "score": score,
        "val_last": last,
        "val_first": first,
        "p6_item_major_real_fraction": p6_real,
    }
    say(f"als_tune: popularity reference score {score:.5f}")

    results = run_search(
        inter,
        users,
        picks,
        MODEL_DIR / "als_tuning.json",
        SCRATCH_DIR / "als_tune",
        reference=reference,
        say=say,
    )
    report = MODEL_DIR / "reports" / "als_tuning.md"
    write_report(results, report)
    win = results["winner"]
    say(
        f"als_tune: done in {(time.time() - t0) / 3600:.1f} h. Winner: "
        f"alpha {win['alpha']:g}, reg {win['reg']:g}, "
        f"{win['iterations']} iterations. Report: {report}"
    )
    say(
        "als_tune: als_params.json for the final P6 fit: "
        + json.dumps(win["als_params_json"])
    )
    target = CURATED / "als_params.json"
    if write_params:
        write_als_params(win, target, say)
    else:
        say(
            f"als_tune: {target} is unchanged. Run again with "
            f"--write-params to write the winner there for P6."
        )


def write_als_params(win, target, say):
    """Write the winner's parameters to target (P6 reads it) and log the
    content it replaced."""
    target = Path(target)
    try:
        before = json.dumps(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        before = "no readable file"
    atomic_json_write(win["als_params_json"], target)
    say(
        f"als_tune: wrote {target}: {json.dumps(win['als_params_json'])}. "
        f"Before: {before}"
    )


if __name__ == "__main__":
    main()

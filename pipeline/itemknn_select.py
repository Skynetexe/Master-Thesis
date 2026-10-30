"""The ItemKNN baseline on the selection listeners.

Protocol, fixed before any result exists:

- Model: pipeline/itemknn.py, the item-based rule of Sarwar et al. (2001)
  on P9b's two co-listening lists. It has no parameter to fit or tune.
- Users and target: the 20,000 selection users of pipeline/selection.py
  and their val_last and val_first picks, the users and picks of
  als_tune.py and P13. This script never reads a test pick.
- Metric: the mean of NDCG@100 over the two pick types, as in
  als_tune.py and P13.
- Variants: "both" (the two lists summed) with the P5 tie rule (id) and
  with popularity ties (pop), and each list alone with the P5 tie rule.
  The single lists are diagnostics.
- Choice for P14: the "both" variant with the higher score. A tie goes to
  "id", the P5 rule.
- Check: 20 users (a fixed random sample) are scored again by a second
  route (itemknn.brute_scores) and ranked again by a full sort, and the
  metrics are recomputed from the stored ranks.

Outputs:
- MODEL_DIR/itemknn_selection.json
- MODEL_DIR/itemknn_selection_ranks.npz
- MODEL_DIR/reports/itemknn_selection.md, with the comparison against
  popularity, the ALS searches and the final GAM on the same picks.

Every line goes to the console and to MODEL_DIR/logs/itemknn.log. Long
steps print a progress line at most every 30 seconds, with the time left.

Usage, from the repository root, through the runner:
    python3 -m pipeline.runner --only itemknn_select
The module's own entry runs it without a marker (the runner then asks for
--adopt):
    python3 -m pipeline.itemknn_select
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from pipeline import evaluation as ev
from pipeline import itemknn as knn
from pipeline import selection
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.config import MODEL_DIR
from pipeline.script_stage import ScriptStage, log_file

LOG_PATH = MODEL_DIR / "logs" / "itemknn.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"  # as in runner.py
VERSION = "itemknn-selection-v1"
CHECK_USERS = 20
CHECK_SEED = 2026092901
SCORE_RTOL = 1e-12

PROTOCOL = {
    "version": VERSION,
    "model": "item-based CF (Sarwar et al. 2001) on P9b's same-artist and "
    "other-artist lists, weights log1p(listens)",
    "users": f"selection sample, {selection.SIZE} users, seed "
    f"{selection.SEED}, disjoint from the eval sample",
    "target": "val_last and val_first picks, P5 candidate rule",
    "metric": "mean NDCG@100 over val_last and val_first",
    "variants": [knn.variant_name(*v) for v in knn.VARIANTS],
    "choice": "the 'both' variant with the higher score, a tie goes to id",
    "declared": "before any result",
}


def score_picks(ranks):
    """(score, val_last block, val_first block), as als_tune.score_picks."""
    last = ev.metric_block(ranks[:, 0])
    first = ev.metric_block(ranks[:, 1])
    return (last["ndcg_at_100"] + first["ndcg_at_100"]) / 2.0, last, first


def npz_key(variant):
    return variant[0] + "_" + variant[1]


def results_from_ranks(ranks, positive, n_slots):
    """The results dict for the variants, from their ranks."""
    out = {"protocol": PROTOCOL, "variants": {}}
    for v in knn.VARIANTS:
        score, last, first = score_picks(ranks[v])
        out["variants"][knn.variant_name(*v)] = {
            "lists": v[0],
            "tie": v[1],
            "score": score,
            "val_last": last,
            "val_first": first,
        }
    both = [knn.variant_name("both", "id"), knn.variant_name("both", "pop")]
    s_id = out["variants"][both[0]]["score"]
    s_pop = out["variants"][both[1]]["score"]
    out["chosen"] = both[1] if s_pop > s_id else both[0]
    out["counts"] = {
        "users": int(len(positive)),
        "users_below_100_positive": int(np.count_nonzero(positive < 100)),
        "median_positive_candidates": (
            float(np.median(positive)) if len(positive) else 0.0
        ),
        "n_slots": int(n_slots),
    }
    return out


def check_users(n_users, n_check=CHECK_USERS, seed=CHECK_SEED):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_users, size=min(n_check, n_users), replace=False))


def load_outputs(model_dir):
    """(results, {variant: ranks}, positive, users) as stored."""
    d = Path(model_dir)
    results = load_json(d / "itemknn_selection.json")
    if results is None:
        raise ValueError("itemknn_selection.json is missing")
    z = np.load(d / "itemknn_selection_ranks.npz")
    ranks = {v: np.asarray(z[npz_key(v)], dtype=np.int64) for v in knn.VARIANTS}
    return (
        results,
        ranks,
        np.asarray(z["positive"], dtype=np.int64),
        np.asarray(z["user_id"], dtype=np.int64),
    )


def verify(
    model_dir,
    t_mats,
    lists,
    u_arrays,
    users,
    picks,
    pop,
    n_slots,
    say=print,
    n_check=CHECK_USERS,
):
    """Raise ValueError unless the stored outputs follow from a second
    computation: scores by a second route, ranks by a full sort, and the
    metrics from the stored ranks. Returns the largest relative score
    gap."""
    results, ranks, positive, stored_users = load_outputs(model_dir)
    if not np.array_equal(stored_users, np.asarray(users, dtype=np.int64)):
        raise ValueError("the stored users differ from the selection users")
    for v in knn.VARIANTS:
        if ranks[v].shape != picks.shape:
            raise ValueError(
                f"{knn.variant_name(*v)}: the stored ranks do " f"not match the picks"
            )
        if int(ranks[v].min()) < 1 or int(ranks[v].max()) > n_slots:
            raise ValueError(
                f"{knn.variant_name(*v)}: a stored rank lies " f"outside 1 to {n_slots}"
            )
    worst = 0.0
    for k in check_users(len(users), n_check):
        items, weights = knn.history(*u_arrays, int(users[k]))
        fast = knn.user_scores(t_mats, items, weights)
        slow = knn.brute_scores(lists, items, weights, n_slots)
        for name in ("same_artist", "other_artists", "both"):
            scale = max(1.0, float(np.max(np.abs(slow[name]))))
            gap = float(np.max(np.abs(fast[name] - slow[name]))) / scale
            worst = max(worst, gap)
            if gap > SCORE_RTOL:
                raise ValueError(
                    f"user {int(users[k])}: the {name} scores "
                    f"differ from the second route by {gap:.1e}"
                )
        for v in knn.VARIANTS:
            s_fast = knn.candidate_scores(fast[v[0]], items)
            s_slow = knn.candidate_scores(slow[v[0]], items)
            for j, p in enumerate(picks[k]):
                got = int(ranks[v][k, j])
                p = int(p)
                if not np.isfinite(s_fast[p]):
                    if got != n_slots:
                        raise ValueError(
                            f"user {int(users[k])}: pick {p} is "
                            f"a train item but ranks {got}"
                        )
                    continue
                want = knn.sorted_rank(s_fast, p, v[1], pop)
                if got != want:
                    raise ValueError(
                        f"user {int(users[k])}, {knn.variant_name(*v)}: "
                        f"stored rank {got}, a full sort gives {want}"
                    )
                tol = SCORE_RTOL * max(1.0, abs(float(s_slow[p])))
                lo, hi = knn.rank_bounds(s_slow, p, tol)
                if not lo <= got <= hi:
                    raise ValueError(
                        f"user {int(users[k])}, {knn.variant_name(*v)}: "
                        f"stored rank {got} lies outside the second route's "
                        f"range {lo} to {hi}"
                    )
        s_both = knn.candidate_scores(fast["both"], items)
        if int(positive[k]) != int(np.count_nonzero(s_both > 0.0)):
            raise ValueError(
                f"user {int(users[k])}: the stored count of "
                f"positive candidates differs"
            )
    again = results_from_ranks(ranks, positive, n_slots)
    for key in again:
        if json.dumps(again[key], sort_keys=True) != json.dumps(
            results.get(key), sort_keys=True
        ):
            raise ValueError(
                f"the stored {key} do not follow from the stored " f"ranks"
            )
    say(
        f"itemknn: check passed on {min(n_check, len(users))} users, largest "
        f"relative score gap {worst:.1e}"
    )
    return worst


def references(model_dir):
    """Other models' selection scores on the same picks, where present."""
    rows = []
    tune = load_json(Path(model_dir) / "als_tuning.json") or {}
    ref = tune.get("reference")
    if ref:
        rows.append(
            (
                "popularity",
                ref["score"],
                ref["val_last"]["recall_at_100"],
                ref["val_first"]["recall_at_100"],
            )
        )
    win = tune.get("winner")
    if win:
        e = next(
            (
                c
                for c in (tune.get("configs") or {}).values()
                if c.get("finished")
                and c["alpha"] == win["alpha"]
                and c["reg"] == win["reg"]
                and c["factors"] == win["factors"]
            ),
            None,
        )
        if e:
            rows.append(
                (
                    f"ALS, {win['factors']} factors (phases A to C)",
                    e["score"],
                    e["best_val_last"]["recall_at_100"],
                    e["best_val_first"]["recall_at_100"],
                )
            )
    tune_d = load_json(Path(model_dir) / "als_tuning_d.json") or {}
    win_d = tune_d.get("winner")
    if win_d:
        rows.append(
            (
                f"ALS, phase D winner ({win_d['factors']} factors, "
                f"nu {win_d['nu']:g})",
                win_d["score"],
                win_d["val_last"]["recall_at_100"],
                win_d["val_first"]["recall_at_100"],
            )
        )
    doc = load_json(Path(model_dir) / "p13b2_model.json") or {}
    if doc.get("final") and doc.get("wide_fits"):
        j = doc["wide_fits"][int(doc["final"]["index"])]
        rows.append(
            (
                f"GAM, final model of P13b part 2 ({j['cell']}, "
                f"{j['arm']}, level {j['level']})",
                j["selection"]["mean"],
                None,
                None,
            )
        )
    return rows


def write_report(results, path, refs=()):
    lines = ["# ItemKNN baseline on the selection listeners", ""]
    add = lines.append
    p = results["protocol"]
    add(f"- model: {p['model']}")
    add(f"- users: {p['users']}")
    add(f"- target: {p['target']}")
    add(f"- metric: {p['metric']}")
    add(f"- choice for P14: {p['choice']}. Chosen: {results['chosen']}.")
    c = results["counts"]
    add(
        f"- users: {c['users']:,}. Candidates with a positive score: median "
        f"{c['median_positive_candidates']:,.0f} per user. "
        f"{c['users_below_100_positive']:,} users have fewer than 100, so "
        f"the tie rule reaches their top 100."
    )
    if "check" in results:
        add(
            f"- check: {results['check']['users']} users scored again by a "
            f"second route (largest relative gap "
            f"{results['check']['score_gap']:.1e}) and ranked again by a "
            f"full sort, metrics recomputed from the stored ranks."
        )
    add("")
    add(
        "| variant | score | val_last NDCG@100 | val_first NDCG@100 | "
        "val_last R@100 | val_first R@100 | val_last MRR |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|")
    for name, e in results["variants"].items():
        add(
            f"| {name} | {e['score']:.5f} | "
            f"{e['val_last']['ndcg_at_100']:.5f} | "
            f"{e['val_first']['ndcg_at_100']:.5f} | "
            f"{e['val_last']['recall_at_100']:.4f} | "
            f"{e['val_first']['recall_at_100']:.4f} | "
            f"{e['val_last']['mrr']:.5f} |"
        )
    if refs:
        add("")
        add(
            "Other models on the same picks (score: mean NDCG@100 over "
            "val_last and val_first):"
        )
        add("")
        add("| model | score | val_last R@100 | val_first R@100 |")
        add("|---|---:|---:|---:|")
        chosen = results["variants"][results["chosen"]]
        add(
            f"| ItemKNN, {results['chosen']} | {chosen['score']:.5f} | "
            f"{chosen['val_last']['recall_at_100']:.4f} | "
            f"{chosen['val_first']['recall_at_100']:.4f} |"
        )
        for name, score, rl, rf in refs:
            add(
                f"| {name} | {score:.5f} | "
                f"{'n/a' if rl is None else f'{rl:.4f}'} | "
                f"{'n/a' if rf is None else f'{rf:.4f}'} |"
            )
    atomic_text_write("\n".join(lines) + "\n", path)


def setup_logging(path):
    """INFO lines to the console and to path (as als_tune.setup_logging).
    Each handler is added once."""
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
            "itemknn: the pipeline logger drops INFO lines, so "
            "this run would print nothing. Stopping."
        )


def load_inputs(model_dir, say):
    """(n_slots, u_arrays, users, picks, pop, lists) from the real files."""
    n_slots, id_min = ev.load_id_slots()
    d = Path(model_dir)
    u_arrays = (
        np.load(d / "csr_user_indptr.npy"),
        np.load(d / "csr_user_indices.npy", mmap_mode="r"),
        np.load(d / "csr_user_data.npy", mmap_mode="r"),
    )
    say(
        "itemknn: loading the selection users, their val picks and the train "
        "popularity with DuckDB (no progress lines in this step)"
    )
    con = connect(4)
    try:
        users = np.asarray(selection.selection_user_ids(con), dtype=np.int64)
        picks = np.asarray(selection.val_picks(con, users), dtype=np.int64)
        pop = ev.popularity_scores(con, n_slots, id_min)
    finally:
        con.close()
    say("itemknn: loading P9b's co-listening lists (no progress lines in this " "step)")
    lists = knn.load_lists(d, n_slots)
    for kind in knn.KINDS:
        say(f"itemknn: {kind}: {len(lists[kind][1]):,} neighbours")
    return n_slots, u_arrays, users, picks, pop, lists


def run(
    model_dir,
    n_slots,
    u_arrays,
    users,
    picks,
    pop,
    lists,
    workers=None,
    say=print,
    n_check=CHECK_USERS,
    inputs=None,
):
    """The protocol above. Returns the results dict and writes the three
    outputs. inputs: provenance entries for the json."""
    d = Path(model_dir)
    t0 = time.time()
    say("itemknn: transposing the lists (no progress lines in this step)")
    t_mats = knn.transposed(lists, n_slots)
    say(
        f"itemknn: lists ready in {time.time() - t0:.0f} s. Ranking "
        f"{len(users):,} users' picks, {len(knn.VARIANTS)} variants."
    )
    ranks, positive = knn.rank_all(
        t_mats, u_arrays, users, picks, pop, workers=workers, log=say
    )
    results = results_from_ranks(ranks, positive, n_slots)
    results["inputs"] = dict(inputs or {})
    atomic_json_write(results, d / "itemknn_selection.json")
    np.savez_compressed(
        d / "itemknn_selection_ranks.npz",
        user_id=users,
        positive=positive,
        **{npz_key(v): ranks[v] for v in knn.VARIANTS},
    )
    gap = verify(
        d, t_mats, lists, u_arrays, users, picks, pop, n_slots, say=say, n_check=n_check
    )
    results["check"] = {"users": int(min(n_check, len(users))), "score_gap": gap}
    atomic_json_write(results, d / "itemknn_selection.json")
    write_report(results, d / "reports" / "itemknn_selection.md", references(d))
    for name, e in results["variants"].items():
        say(
            f"itemknn: {name}: score {e['score']:.5f}, val_last R@100 "
            f"{e['val_last']['recall_at_100']:.4f}, val_first R@100 "
            f"{e['val_first']['recall_at_100']:.4f}"
        )
    say(
        f"itemknn: chosen for P14: {results['chosen']}. Done in "
        f"{(time.time() - t0) / 60:.1f} min."
    )
    return results


def provenance(model_dir=None):
    """The markers of the stages behind the inputs (P6's CSR, P9b's
    lists), as the outputs record them."""
    inputs = {}
    for stage in ("p6_als", "p9b_colisten"):
        try:
            inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            inputs[f"marker:{stage}"] = None
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ItemKNN baseline on the selection users' val picks "
        "(protocol in the module docstring)"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: up to 16)"
    )
    args = parser.parse_args(argv)
    setup_logging(LOG_PATH)
    say = log.info
    try:
        say(f"itemknn: started, process id {os.getpid()}, log file " f"{LOG_PATH}")
        n_slots, u_arrays, users, picks, pop, lists = load_inputs(MODEL_DIR, say)
        run(
            MODEL_DIR,
            n_slots,
            u_arrays,
            users,
            picks,
            pop,
            lists,
            workers=args.workers,
            say=say,
            inputs=provenance(MODEL_DIR),
        )
    except KeyboardInterrupt:
        log.info(
            "itemknn: stopped by Ctrl+C. Run it again to start over (a "
            "run takes minutes)."
        )
        raise SystemExit(130)
    except Exception:
        log.exception("itemknn: stopped by an error")
        raise SystemExit(1)


# Runner stage


class StageItemKNN(ScriptStage):
    """itemknn_select as a runner stage."""

    name = "itemknn_select"
    upstream = ["p6_als", "p9b_colisten"]

    def outputs(self):
        d = self.model_dir
        return {
            "results": d / "itemknn_selection.json",
            "ranks": d / "itemknn_selection_ranks.npz",
            "report": d / "reports" / "itemknn_selection.md",
        }

    def provenance(self):
        return provenance(self.model_dir)

    def recorded_inputs(self):
        doc = load_json(self.outputs()["results"])
        return None if doc is None else doc.get("inputs")

    def _load(self):
        """(n_slots, u_arrays, users, picks, pop, lists)."""
        return load_inputs(self.model_dir, log.info)

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_PATH.name):
            n_slots, u_arrays, users, picks, pop, lists = self._load()
            run(
                self.model_dir,
                n_slots,
                u_arrays,
                users,
                picks,
                pop,
                lists,
                say=log.info,
                inputs=self.provenance(),
            )
        return self.output_info()

    def check(self):
        n_slots, u_arrays, users, picks, pop, lists = self._load()
        t_mats = knn.transposed(lists, n_slots)
        verify(
            self.model_dir,
            t_mats,
            lists,
            u_arrays,
            users,
            picks,
            pop,
            n_slots,
            say=log.info,
        )


if __name__ == "__main__":
    main()

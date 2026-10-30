"""P14: the evaluation on the eval listeners.

ANALYSIS PLAN. It was written into this docstring before P14 read any eval
listener. P5 and P6 read the eval listeners' test picks earlier, for their
baseline numbers. No choice of a model, a setting or a rule used them.
Nothing in this plan changes after a result.

Data. The eval listeners of P3's stratified sample (eval_users.parquet)
that have both test picks, which is P5's frame (19,999 listeners on the
real data), and their test_last and test_first picks (P2). Candidates and
ties follow P5: every item slot except slot 0 and the listener's train
items, the other held-out picks stay candidates, and equal scores rank by
item id. Every rank of every arm is exact over the full catalog.

Arms:
- gam: the final model, the model record of p13d.json that
  p13c_final.json names.
- artist_only: "artist terms only", the reference arm of gam_refit.
- cold_gam: the cold-track GAM (stage p14_cold_gam).
- knn_same, knn_both: ItemKNN (itemknn.py), the same-artist list and both
  lists, with P5's tie rule (itemknn_select chose both/id).
- als_d: the ALS phase D winner (als_d/best.json).
- als_p6: P6's ALS fit. P14 ranks it again and its stored eval ranks must
  equal the result.
- popularity: P5's baseline. P14 ranks it again and its stored eval ranks
  must equal the result.

Metrics, per pick, then the listener's mean over the two picks: NDCG@100
(1 / log2(rank + 1) up to rank 100, else 0), NDCG@10, R@10 and R@100 (the
pick within the cut) and the reciprocal rank (MRR). An arm's score is the
mean over the listeners. NDCG@100 is the primary metric, as in selection.

SESOI: 2 percent of the GAM's eval score on the metric of the comparison
(the precedent of P13d and the tree-GAM audit: 2 percent of the model's
score), so SESOI = 0.02 x mean(g) on NDCG@100 for the confirmatory family.
A difference is "above" when its mean exceeds SESOI and its 95 percent
interval (gam_audit.paired_summary) lies above 0, "below" in the mirror
case, and "within" otherwise.

Confirmatory family: Holm at alpha 0.05 over four paired t-tests across
the listeners (df = n - 1) on the per-listener NDCG@100, g for the GAM:
- C1, non-inferiority to ItemKNN same-artist (k): H0: mean(g - k) <=
  -SESOI, that is mean(1.02 g - k) <= 0, tested one-sided on d = 1.02 g -
  k. The margin enters as a fixed multiple, so no estimated margin is
  plugged in. Non-inferiority holds when Holm's adjusted p is at most
  0.05.
- C2, C3, C4: the GAM against the ALS phase D winner, popularity and
  "artist terms only", two-sided on d = g - a.
Exploratory family: Benjamini-Yekutieli at q 0.05 over 31 two-sided paired
t-tests of the GAM against an arm:
- E1 to E3: NDCG@100 against knn_both, als_p6 and cold_gam.
- E4 to E31: NDCG@10, R@10, R@100 and MRR against each of the seven other
  arms (knn_same, knn_both, als_d, als_p6, popularity, artist_only,
  cold_gam).
Permutation check (Smucker, Allan and Carterette 2007): every test runs
again as a paired randomization test. The signs of the listeners'
differences flip at random, N_PERM = 10,000 draws (seed PERM_SEED and the
test's position), p = (1 + count) / (N_PERM + 1), with the test's own
alternative. The t-test decides. The report gives both p-values and both
decisions, and flags every test whose decisions differ.

Slices, descriptive (no test): per pick, with 95 percent intervals for
listener clusters (gam_knn_diag.cluster_mean):
- pick type,
- popularity band (P5's bands of pop_centile),
- history length (tertiles of the listeners' train pairs),
- cold tracks (P5's bands of the pick's train pairs: 0, 01-05, 06-99,
  100+),
- same-artist neighbour (the pick's colisten_same_artist value above 0 for
  the listener, or not),
- pick year (the anchor below in 2013, or earlier).
Per level: each arm's NDCG@100 and R@100, and the GAM minus each arm. On
the picks without a same-artist neighbour the report states the cold-track
GAM minus the GAM.

Drop from selection to eval, descriptive: per arm, the eval NDCG@100 (test
picks of the eval listeners) minus the selection NDCG@100 (val picks of the
selection listeners), with the interval of a difference of two independent
means. The selection ranks are the stored ones (gam, artist_only,
cold_gam, ItemKNN) or a recomputation (ALS, popularity).

Reasons, on each eval listener's top 10 of the GAM:
- the contribution of term j to track i is f_j(x_i) minus its mean over
  the listener's candidate set (background 1 of the P13 design, exact:
  every candidate of the full catalog with equal weight). artist_affinity
  and colisten_same_artist form the artist reason (their sum). Every other
  term is a reason of its own. A reason is a candidate when one of its
  terms is on its shape at the track (as p13c.Reasons).
- D1 (P13c's rule) from the 100 listener replicates of the final model
  (theta_boot_listener of the arrays that p13c_final.json names): shown
  when a candidate, positive under the model, positive in at least 95
  percent of the replicates and in the top 3 in at least 80 percent of
  them. The 50 twoway replicates give a sensitivity row, as in P13c.
- S4 (P13d's rule) from the S3 set (p13d_model_set.npz, member_tables):
  positive under every member and in the top 3 of at least 80 percent of
  them, on the same background. A reason is shown when D1 and S4 hold.
- strength: the odds factor exp(contribution), with no minimum.
- evidence, per candidate reason (concepts.evidence): the matched tags or
  genres with their terms, the audio dimensions from the best fit down,
  and for the artist reason the credited artist, the number of history
  tracks by that artist and the co-listened history tracks. Up to
  EVIDENCE_K units each, largest first, with the count of all units.
Checks of the reasons, each one stops the run:
- the contributions of the model's terms sum to the track's score minus
  the mean score over the candidates, and the term values sum to the
  score (CHECK_ATOL, SUM_ATOL).
- the raw and the centered tables give the same contributions (a shift of
  a term moves its background by the same constant).
- template direction: every shown reason is positive, and every shown
  content reason (a bounded term) sits on a non-decreasing part of its
  shape at the track, so the template "more of this raises the track"
  holds.
Reported, descriptive: the share of tracks with each reason shown, the
mean contribution and odds factor when shown, the reasons per track, the
share of tracks that leave the top 10 when their top shown reason falls to
its background value (the score minus the contribution, against the 11th
track, P5's tie rule), and for the artist reason the share whose
artist_affinity part is negative or whose artist_affinity shape falls at
the track (where the suppression of P13a hides).

Leakage (Ji et al., TOIS 2023), descriptive. A pick's anchor is the time
that P2's split orders it by: the last play (last_ts) of test_last and the
first play (first_ts) of test_first. Per pick:
- global: the share of all training pairs, and of their listens, that
  begin after the anchor (first_ts > anchor: every play after it, a lower
  bound) and that end after it (last_ts > anchor: some play after it, an
  upper bound). The global table counts pairs by minute, so a pair in the
  anchor's own minute counts in the upper bound only.
- own: the same shares of the listener's own training pairs. P2's split
  makes the lower bound 0 for both picks and the upper bound 0 for
  test_last, and P14 checks both.
The report gives quantiles per pick type and the share of anchors in 2013.

Cold-start curve (cold_curve.py), declared here because it reads the eval
listeners:
- listeners: the eval listeners with at least 100 training pairs, the same
  set at every point.
- history: the listener's first N training pairs in time order (first_ts,
  ties by item id), N in 5, 10, 25 and 100. Each kept pair counts as one
  play (weight round(1000 log1p(1)) / 1000), because the pair table cannot
  tell which later plays fell before the cut.
- candidates: P5's candidates of the full history (the whole training set
  excluded), so every point ranks the same candidates and only the
  evidence changes. Picks: the same test picks.
- arms: the GAM, the cold-track GAM, ItemKNN both lists and popularity
  (flat), with the frozen tables and lists. ALS needs a fold-in and stays
  out.
- metrics: NDCG@100 and R@100 per N with 95 percent listener intervals,
  and P14's full-history value as the last point. Descriptive, no test.

Ties, descriptive (added with the change note below): per
GAM arm, the share of picks whose score equals the score of another
candidate, the size of those ties, NDCG@100 and R@100 at the best and at
the worst end of the tie spans (the best and the worst rank among the
candidates with the pick's exact score), and the share of listeners whose
10th and 11th tracks tie.

Change to the computation only, made before any result was read.
The first run stopped at its dense check (listener 82624, the cold-track
GAM, pick 1: rank 71,855, the reference's rank 71,940). That pass used the
arithmetic of catalog.CatalogScorer: an item part plus sparse corrections.
This arithmetic can give two tracks with equal term values two scores that
differ in the last bits, for example two values on one flat part of a
bounded shape. So it ordered such ties by rounding and not by item id. The
pass now adds the term values as COMPUTATION says. The rules of the plan
did not change. The stored selection ranks of the three GAM arms (P13c,
gam_refit and p14_cold_gam) come from CatalogScorer, so the selection side
of the drop orders exact ties by rounding. The tie spans show how far any
order of the ties can move the eval scores.

COMPUTATION.
- The GAM arms: one pass over the eval listeners scores the three tables
  over the whole catalog, one slot range at a time. The tables are the
  final model's (six terms on the real data) and the two other models,
  placed into its layout with zeros for the terms that they lack. The
  score of a slot is the sum, in term order, of its term values by the
  expressions of the dense reference (catalog.reference_scores: np.interp
  on the knots, the state values, 0 outside the tiers of a term), on the
  concepts of the range (CatalogScorer.concept_range). Each expression
  works slot by slot. So equal term values give equal scores, and equal
  scores rank by item id, as P5 says.
- Per listener the pass keeps the exact ranks of the two picks, their
  scores and tie spans, the top TOP_KEEP of each table and the score sum
  over the candidates. It also keeps the sum of the basis vectors over the
  candidates (from CatalogScorer's corrections: the background of every
  table of the final layout), the concept values at the picks, and for
  the GAM's top 10 the slots, values and evidence. Blocks of EVAL_BLOCK
  listeners are saved as they finish.
- ItemKNN: itemknn.rank_all. ALS: als.rank_items. Popularity: P5's route.
- The inputs from DuckDB (the frame, the covariates, the popularity and
  the leakage counts) and the ItemKNN and ALS ranks are saved under
  SCRATCH_DIR/p14, keyed by their inputs.
Checks while it runs, each one stops the run:
- the frame equals P5's route (evaluation.load_eval_frame) and the user
  ids of P5's and P6's stored ranks.
- the train pairs per listener equal the CSR, and the leakage table
  counts every training pair once.
- in every slot range, the scores of every CHECK_STRIDE-th slot, of the
  picks and of the top candidates of the range equal
  catalog.reference_scores() on those slots exactly (np.array_equal). The
  scores of the picks and of the GAM's top 10 do so again from the
  concepts of their own one-slot ranges.
- the pool gives the probe listeners the results of this process.
- VERIFY_USERS listeners get their ranks, tie spans, pick scores and top
  lists again from the dense reference over the full catalog
  (concepts.full_catalog, catalog.reference_scores) exactly, and their
  score means and per-term backgrounds within CHECK_ATOL.
- a second ItemKNN route on CHECK_USERS listeners (as
  itemknn_select.verify).
- P5's and P6's stored eval ranks equal the recomputation, and the phase D
  winner's selection score from the recomputed ranks equals best.json's.
- the reason checks above, and a read-back of every result from the
  stored outputs.
The runner checks that the markers of the upstream stages are current.

Outputs: MODEL_DIR/p14.json (the plan, the results, the checks),
MODEL_DIR/p14.npz (per listener and pick: the ranks of every arm, the
slice labels, the leakage shares, the top lists, the reasons and the
evidence) and MODEL_DIR/reports/p14_report.md. Every line also goes to
MODEL_DIR/logs/p14.log. Long steps print a line at most every 30 seconds
with the share done and the time left.

Resume: after Ctrl+C the same command continues: the saved inputs, ranks
and pass blocks stay.

Expected run on the real data: DuckDB 15 to 30 minutes (the leakage
counts scan the 16 train files three times), ItemKNN about 10 minutes, ALS
about 8 minutes, the pass about 2 to 3 hours (a full scan per listener,
the probe prints an estimate), the dense check a few minutes, the reasons,
tests and read-back a few minutes. Peak memory near P13's selection pass.
The saved DuckDB inputs and ranks of the first run stay valid. Its pass
blocks stay in SCRATCH_DIR/p14/pass, and the pass writes to pass_v2.

Usage, from the repository root, after the stage p14_cold_gam:
    python3 -m pipeline.runner --only p14_eval
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only p14_eval
"""

import glob as globlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from pipeline import als
from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import evaluation as ev
from pipeline import gam
from pipeline import gam_arms
from pipeline import gam_audit as ga
from pipeline import itemknn as knn
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    connect,
    file_fingerprint,
    file_sha256,
    load_json,
    log,
    marker_path,
)

from pipeline import gam_knn_diag as gkd
from pipeline import gam_refit as gr
from pipeline import itemknn_select as iks
from pipeline import p13c as pc
from pipeline import p13d as pd13
from pipeline import p14_cold as cold
from pipeline.script_stage import ScriptStage, log_file

VERSION = "p14-v2"
INPUTS_VERSION = "p14-v1"  # the keys of the saved DuckDB inputs and ranks
PASS_VERSION = "p14-pass-v2"  # the key of the pass blocks
PASS_DIR = "pass_v2"
WORK_NAME = "p14"
DECLARED = "before any eval listener was read for P14"
OUT_FILES = ("p14.json", "p14.npz")
REPORT_NAME = "p14_report.md"
LOG_NAME = "p14.log"
SESOI_SHARE = 0.02
ALPHA = 0.05
N_PERM = 10_000
PERM_SEED = 2026100402
PERM_CHUNK = 250
TOP_KEEP = 20
REASON_K = 10
EVIDENCE_K = 5
EVAL_CHUNK = 1 << 18
EVAL_BLOCK = 25
PROBE_USERS = 4
CHECK_STRIDE = 64  # every 64th slot of a range against the reference
VERIFY_USERS = 6
VERIFY_SEED = 2026100403
CHECK_USERS = 20
CHECK_SEED = 2026100404
CHECK_ATOL = 1e-8  # a mean over millions of candidates
SUM_ATOL = 1e-9  # the term values of one track against its score
Z95 = 1.959963984540054
PICK_TYPES = ("test_last", "test_first")
METRICS = ("ndcg_at_100", "ndcg_at_10", "recall_at_10", "recall_at_100", "mrr")
SECONDARY = METRICS[1:]
ARMS = (
    "gam",
    "artist_only",
    "cold_gam",
    "knn_same",
    "knn_both",
    "als_d",
    "als_p6",
    "popularity",
)
ARM_LABELS = {
    "gam": "GAM (final model)",
    "artist_only": "artist terms only",
    "cold_gam": "cold-track GAM",
    "knn_same": "ItemKNN, same-artist list",
    "knn_both": "ItemKNN, both lists",
    "als_d": "ALS, phase D winner",
    "als_p6": "ALS, P6 fit",
    "popularity": "popularity",
}
GAM_ARMS = ("gam", "artist_only", "cold_gam")
OTHERS = (
    "knn_same",
    "knn_both",
    "als_d",
    "als_p6",
    "popularity",
    "artist_only",
    "cold_gam",
)
CONFIRMATORY = (
    ("C1", "knn_same", "greater"),
    ("C2", "als_d", "two-sided"),
    ("C3", "popularity", "two-sided"),
    ("C4", "artist_only", "two-sided"),
)
EXPLORATORY = tuple(
    [
        (f"E{i + 1}", a, "ndcg_at_100")
        for i, a in enumerate(("knn_both", "als_p6", "cold_gam"))
    ]
    + [
        (f"E{4 + i * len(OTHERS) + j}", a, m)
        for i, m in enumerate(SECONDARY)
        for j, a in enumerate(OTHERS)
    ]
)
SAME = "colisten_same_artist"
POP_BANDS = gkd.POP_BANDS
COLD_BANDS = ("0", "01-05", "06-99", "100+")
HISTORY_LABELS = ("short", "middle", "long")
NEIGHBOUR_LABELS = ("with a same-artist neighbour", "without a same-artist neighbour")
YEAR_LABELS = ("before 2013", "2013")
SLICE_ORDER = (
    "pick type",
    "popularity band",
    "history length",
    "cold tracks",
    "same-artist neighbour",
    "pick year",
)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

PROTOCOL = {
    "version": VERSION,
    "declared": DECLARED,
    "listeners": "eval listeners with both test picks (P5's frame)",
    "arms": {a: ARM_LABELS[a] for a in ARMS},
    "metrics": list(METRICS),
    "primary": "ndcg_at_100",
    "sesoi": "0.02 x the GAM's eval score on the metric of the comparison",
    "confirmatory": {
        "family": "Holm, alpha 0.05",
        "tests": [
            {"id": i, "arm": a, "alternative": alt} for i, a, alt in CONFIRMATORY
        ],
        "C1": "non-inferiority: mean(1.02 g - k) > 0, " "one-sided",
    },
    "exploratory": {
        "family": "Benjamini-Yekutieli, q 0.05",
        "tests": [{"id": i, "arm": a, "metric": m} for i, a, m in EXPLORATORY],
    },
    "test": "paired t-test over the listeners, permutation check with "
    f"{N_PERM} sign flips",
    "reasons": "D1 from the listener replicates, then S4 from the S3 set, "
    "background: the listener's candidate set",
    "leakage": "anchor: last_ts of test_last, first_ts of test_first",
}


# Helpers


def _num(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _clean(x):
    """x with every non-finite float as None and numpy scalars as Python
    values, as the json file holds it."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _num(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def _norm(x):
    return json.loads(json.dumps(_clean(x), sort_keys=True))


def offsets_of(terms):
    return [int(x) for x in np.cumsum([0] + [t.size for t in terms])[:-1]]


def _strip(t):
    d = t.to_json()
    d.pop("monotone", None)
    return d


def place(terms, src_terms, src_tables):
    """Tables of another layout in the layout of terms: each term's entries
    by name, zeros for the terms that the other layout lacks. A shared term
    must have the same knots and states."""
    offs, soffs = offsets_of(terms), offsets_of(src_terms)
    by = {t.name: (t, o) for t, o in zip(src_terms, soffs)}
    extra = sorted(set(by) - {t.name for t in terms})
    if extra:
        raise ValueError(f"the terms {extra} are not in the final model")
    out = np.zeros(int(sum(t.size for t in terms)), dtype=np.float64)
    src = np.asarray(src_tables, dtype=np.float64)
    for t, o in zip(terms, offs):
        if t.name not in by:
            continue
        s, so = by[t.name]
        if _strip(s) != _strip(t):
            raise ValueError(
                f"term {t.name} has other knots or states than " f"the final model's"
            )
        out[o : o + t.size] = src[so : so + s.size]
    return out


class TrainRows:
    """The train items of a user from P6's user-major CSR (the part of
    als.Interactions that als.rank_items reads)."""

    def __init__(self, u_indptr, u_indices):
        self.indptr, self.indices = u_indptr, u_indices

    def train_items(self, user):
        return np.asarray(
            self.indices[int(self.indptr[user]) : int(self.indptr[user + 1])]
        )


def _array_key(*arrays):
    return s13._array_sha(*[np.ascontiguousarray(a) for a in arrays])


# Statistics


def pick_metrics(ranks):
    """{metric: (n, 2) per-pick values} from exact ranks."""
    r = np.asarray(ranks, dtype=np.float64)
    gain = 1.0 / np.log2(r + 1.0)
    return {
        "ndcg_at_100": np.where(r <= 100, gain, 0.0),
        "ndcg_at_10": np.where(r <= 10, gain, 0.0),
        "recall_at_10": (r <= 10).astype(np.float64),
        "recall_at_100": (r <= 100).astype(np.float64),
        "mrr": 1.0 / r,
    }


def listener_metrics(ranks):
    return {k: v.mean(axis=1) for k, v in pick_metrics(ranks).items()}


def t_test(d, alternative):
    """The paired t-test of mean(d) against 0 (df = n - 1)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    t = m / se if se > 0 else math.nan
    if alternative == "greater":
        p = float(stats.t.sf(t, n - 1))
    elif alternative == "two-sided":
        p = float(2.0 * stats.t.sf(abs(t), n - 1))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return {"mean": m, "se": se, "t": t, "df": n - 1, "p": p, "n": n}


def perm_test(d, alternative, seed, n_perm=None):
    """The paired randomization test of mean(d): random sign flips of each
    listener's difference. Returns (1 + count) / (n_perm + 1)."""
    n_perm = N_PERM if n_perm is None else int(n_perm)
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    obs = float(d.mean())
    tol = 1e-12 * max(abs(obs), float(np.abs(d).mean()), 1e-300)
    rng = np.random.default_rng(seed)
    count = done = 0
    while done < n_perm:
        k = min(PERM_CHUNK, n_perm - done)
        s = rng.integers(0, 2, size=(k, n), dtype=np.int8)
        means = ((2.0 * s - 1.0) @ d) / n
        if alternative == "greater":
            count += int(np.count_nonzero(means >= obs - tol))
        else:
            count += int(np.count_nonzero(np.abs(means) >= abs(obs) - tol))
        done += k
    return (1 + count) / (n_perm + 1)


def holm(p):
    """Holm's adjusted p-values (step-down)."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def by_adjust(p):
    """Benjamini-Yekutieli adjusted p-values (step-up, c(m) = sum 1/i)."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    c = sum(1.0 / i for i in range(1, m + 1))
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 1.0
    for i in range(m - 1, -1, -1):
        k = order[i]
        run = min(run, min(1.0, m * c * p[k] / (i + 1)))
        adj[k] = run
    return adj


def sesoi_label(dd, sesoi):
    m, lo, hi = dd.get("mean"), dd.get("lo"), dd.get("hi")
    if m is None or lo is None or hi is None:
        return "within"
    if m > sesoi and lo > 0:
        return "above"
    if m < -sesoi and hi < 0:
        return "below"
    return "within"


def paired(a, b):
    return {
        k: (int(v) if k == "n" else _num(v)) for k, v in ga.paired_summary(a, b).items()
    }


def run_tests(lm):
    """The confirmatory and exploratory families from the per-listener
    metrics {arm: {metric: (n,)}}. A pure function."""
    g = lm["gam"]["ndcg_at_100"]
    sesoi = SESOI_SHARE * float(g.mean())
    conf = []
    for pos, (tid, arm, alt) in enumerate(CONFIRMATORY):
        a = lm[arm]["ndcg_at_100"]
        d = (1.0 + SESOI_SHARE) * g - a if alt == "greater" else g - a
        tt = t_test(d, alt)
        conf.append(
            {
                "id": tid,
                "arm": arm,
                "metric": "ndcg_at_100",
                "alternative": alt,
                "kind": ("non-inferiority" if alt == "greater" else "difference"),
                "difference": paired(g, a),
                "statistic": tt,
                "p": tt["p"],
                "p_perm": perm_test(d, alt, [PERM_SEED, 0, pos]),
            }
        )
    expl = []
    for pos, (tid, arm, met) in enumerate(EXPLORATORY):
        x, y = lm["gam"][met], lm[arm][met]
        tt = t_test(x - y, "two-sided")
        expl.append(
            {
                "id": tid,
                "arm": arm,
                "metric": met,
                "alternative": "two-sided",
                "kind": "difference",
                "difference": paired(x, y),
                "statistic": tt,
                "p": tt["p"],
                "p_perm": perm_test(x - y, "two-sided", [PERM_SEED, 1, pos]),
            }
        )
    for fam, adjust in ((conf, holm), (expl, by_adjust)):
        a_t = adjust([r["p"] for r in fam])
        a_p = adjust([r["p_perm"] for r in fam])
        for r, x, y in zip(fam, a_t, a_p):
            r["p_adjusted"], r["p_perm_adjusted"] = float(x), float(y)
            r["reject"] = bool(x <= ALPHA)
            r["reject_perm"] = bool(y <= ALPHA)
            r["perm_agrees"] = r["reject"] == r["reject_perm"]
            r["sesoi"] = SESOI_SHARE * float(lm["gam"][r["metric"]].mean())
            r["label"] = sesoi_label(r["difference"], r["sesoi"])
            if r["kind"] == "non-inferiority":
                r["decision"] = (
                    "non-inferior" if r["reject"] else "non-inferiority not shown"
                )
                r["margin"] = -sesoi
            else:
                r["decision"] = "differs" if r["reject"] else "no difference shown"
    return {
        "sesoi": sesoi,
        "confirmatory": conf,
        "exploratory": expl,
        "disagreements": [r["id"] for r in conf + expl if not r["perm_agrees"]],
    }


def arm_block(ranks):
    lm = listener_metrics(ranks)
    pm = pick_metrics(ranks)
    r = np.asarray(ranks, dtype=np.float64)
    out = {k: float(v.mean()) for k, v in lm.items()}
    out["types"] = {
        name: dict(
            {k: float(pm[k][:, p].mean()) for k in pm},
            median_rank=float(np.median(r[:, p])),
        )
        for p, name in enumerate(PICK_TYPES)
    }
    out["listeners"] = int(len(r))
    return out


def drop_row(e, s):
    """Eval minus selection: two independent means."""
    e, s = np.asarray(e, dtype=np.float64), np.asarray(s, dtype=np.float64)
    me, ms = float(e.mean()), float(s.mean())
    se = math.sqrt(float(e.var(ddof=1)) / len(e) + float(s.var(ddof=1)) / len(s))
    d = me - ms
    return {
        "selection": ms,
        "eval": me,
        "drop": d,
        "se": se,
        "lo": d - Z95 * se,
        "hi": d + Z95 * se,
        "relative": d / ms if ms else None,
        "listeners": [int(len(s)), int(len(e))],
    }


# Slices


def slice_labels(L):
    """{slice: ((n, 2) codes, level names)} from the label inputs."""
    pc_, tp = L["pop_centile"], L["train_pairs"]
    hist = np.asarray(L["history"], dtype=np.float64)
    q1, q2 = (float(x) for x in np.quantile(hist, [1 / 3, 2 / 3]))
    h = np.where(hist <= q1, 0, np.where(hist <= q2, 1, 2))
    cold_c = np.where(tp <= 0, 0, np.where(tp <= 5, 1, np.where(tp <= 99, 2, 3)))
    years = (
        np.asarray(L["anchor"], dtype=np.int64)
        .astype("datetime64[s]")
        .astype("datetime64[Y]")
        .astype(np.int64)
        + 1970
    )
    n = len(hist)
    out = {
        "pick type": (np.tile(np.arange(2), (n, 1)), list(PICK_TYPES)),
        "popularity band": (gkd.pop_bands(pc_), list(POP_BANDS)),
        "history length": (np.repeat(h[:, None], 2, axis=1), list(HISTORY_LABELS)),
        "cold tracks": (cold_c, list(COLD_BANDS)),
        "same-artist neighbour": (
            np.where(L["neighbour"], 0, 1),
            list(NEIGHBOUR_LABELS),
        ),
        "pick year": (np.where(years >= 2013, 1, 0), list(YEAR_LABELS)),
    }
    return out, {"history_tertiles": [q1, q2]}


def slice_tables(pm, labels):
    """Per slice and level: each arm's NDCG@100 and R@100, the GAM minus
    each arm, and the cold-track GAM minus the GAM."""
    out = {}
    for name, (codes, levels) in labels.items():
        rows = []
        for c, lev in enumerate(levels):
            mask = codes == c
            row = {
                "level": lev,
                "picks": int(mask.sum()),
                "listeners": int(mask.any(axis=1).sum()),
                "arms": {},
                "gam_minus": {},
            }
            for a in ARMS:
                row["arms"][a] = {
                    "ndcg_at_100": gkd.cluster_mean(pm[a]["ndcg_at_100"], mask),
                    "recall_at_100": gkd.cluster_mean(pm[a]["recall_at_100"], mask),
                }
                if a != "gam":
                    row["gam_minus"][a] = gkd.cluster_mean(
                        pm["gam"]["ndcg_at_100"] - pm[a]["ndcg_at_100"], mask
                    )
            row["cold_minus_gam"] = gkd.cluster_mean(
                pm["cold_gam"]["ndcg_at_100"] - pm["gam"]["ndcg_at_100"], mask
            )
            rows.append(row)
        out[name] = rows
    return out


def summarize_scores(ranks, sel_ranks, L):
    """Every accuracy result from the stored ranks and labels: a pure
    function, so check_outputs() recomputes it."""
    lm = {a: listener_metrics(ranks[a]) for a in ARMS}
    pm = {a: pick_metrics(ranks[a]) for a in ARMS}
    labels, cuts = slice_labels(L)
    drop = {}
    for a in ARMS:
        if a in sel_ranks:
            drop[a] = drop_row(
                lm[a]["ndcg_at_100"], ga.per_listener_scores(sel_ranks[a])
            )
    return {
        "arms": {a: arm_block(ranks[a]) for a in ARMS},
        "tests": run_tests(lm),
        "slices": slice_tables(pm, labels),
        "slice_cuts": cuts,
        "drop": drop,
    }


def tie_summary(ranks, span_lo, span_hi, top_score):
    """The ties of the GAM arms (module docstring), from the stored ranks,
    tie spans (n, arms, 2) and top scores: a pure function, so
    check_outputs() recomputes it."""
    out = {}
    for c, arm in enumerate(GAM_ARMS):
        lo, hi, rk = span_lo[:, c, :], span_hi[:, c, :], ranks[arm]
        if np.any(lo > rk) or np.any(rk > hi):
            raise ValueError(f"{arm}: a rank lies outside its tie span")
        size = hi - lo + 1
        tied = size > 1
        row = {
            "picks": int(size.size),
            "tied": float(tied.mean()),
            "size_median": (float(np.median(size[tied])) if tied.any() else None),
            "size_max": int(size.max()),
            "boundary_tie": float(
                np.mean(top_score[:, c, REASON_K - 1] == top_score[:, c, REASON_K])
            ),
        }
        for end, r in (("best", lo), ("p5", rk), ("worst", hi)):
            m = listener_metrics(r)
            row[end] = {
                k: float(m[k].mean())
                for k in ("ndcg_at_100", "recall_at_100", "ndcg_at_10")
            }
        out[arm] = row
    return out


# Leakage


def leakage_counts(user, first_ts, last_ts, listens, frame_users, anchor):
    """The aggregates that Inputs.leakage() gets from DuckDB, from a pair
    table in memory: the minute tables of first_ts and last_ts (count and
    listens per minute) and, per frame user and anchor column, the own
    pairs and listens with first_ts or last_ts after the anchor. The smoke
    test builds its inputs with it."""
    user = np.asarray(user, dtype=np.int64)
    f, l_ = (np.asarray(x, dtype=np.int64) for x in (first_ts, last_ts))
    w = np.asarray(listens, dtype=np.int64)
    mf, ml = f // 60, l_ // 60
    m0 = int(min(mf.min(), ml.min()))
    M = int(max(mf.max(), ml.max())) - m0 + 1
    out = {
        "m0": m0,
        "first_n": np.bincount(mf - m0, minlength=M),
        "first_l": np.bincount(mf - m0, weights=w, minlength=M).astype(np.int64),
        "last_n": np.bincount(ml - m0, minlength=M),
        "last_l": np.bincount(ml - m0, weights=w, minlength=M).astype(np.int64),
        "total_n": int(len(f)),
        "total_l": int(w.sum()),
        "bad_order": int(np.count_nonzero(f > l_)),
    }
    fu = np.asarray(frame_users, dtype=np.int64)
    pos = np.searchsorted(fu, user)
    pos_c = np.minimum(pos, len(fu) - 1)
    inn = fu[pos_c] == user
    k = pos_c[inn]
    out["own_n"] = np.bincount(k, minlength=len(fu))
    out["own_l"] = np.bincount(k, weights=w[inn], minlength=len(fu)).astype(np.int64)
    a = np.asarray(anchor, dtype=np.int64)
    for key, ts in (("lo", f[inn]), ("hi", l_[inn])):
        nn = np.zeros((len(fu), 2), dtype=np.int64)
        ll = np.zeros((len(fu), 2), dtype=np.int64)
        for c in range(2):
            after = ts > a[k, c]
            nn[:, c] = np.bincount(k[after], minlength=len(fu))
            ll[:, c] = np.bincount(
                k[after], weights=w[inn][after], minlength=len(fu)
            ).astype(np.int64)
        out[f"own_{key}_n"], out[f"own_{key}_l"] = nn, ll
    return out


def leakage_shares(anchor, C):
    """Per pick: the global and own shares (module docstring, leakage)."""
    a = np.asarray(anchor, dtype=np.int64)
    M = len(C["first_n"])
    mi = a // 60 - int(C["m0"])
    out = {}
    for unit, tot in (("n", C["total_n"]), ("l", C["total_l"])):
        sf_first = np.concatenate((np.cumsum(C[f"first_{unit}"][::-1])[::-1], [0]))
        sf_last = np.concatenate((np.cumsum(C[f"last_{unit}"][::-1])[::-1], [0]))
        lo = sf_first[np.clip(mi + 1, 0, M)]
        hi = sf_last[np.clip(mi, 0, M)]
        name = "pairs" if unit == "n" else "listens"
        out[f"global_lo_{name}"] = lo / float(tot)
        out[f"global_hi_{name}"] = hi / float(tot)
        own_tot = np.asarray(C[f"own_{unit}"], dtype=np.float64)[:, None]
        out[f"own_lo_{name}"] = C[f"own_lo_{unit}"] / own_tot
        out[f"own_hi_{name}"] = C[f"own_hi_{unit}"] / own_tot
    return out


def summarize_leakage(shares, anchor):
    """Quantiles per pick type and the share of anchors in 2013."""
    years = (
        np.asarray(anchor, dtype=np.int64)
        .astype("datetime64[s]")
        .astype("datetime64[Y]")
        .astype(np.int64)
        + 1970
    )
    out = {"quantiles": list(QUANTILES), "types": {}}
    for p, name in enumerate(PICK_TYPES):
        row = {
            "share_2013": float(np.mean(years[:, p] == 2013)),
            "year_min": int(years[:, p].min()),
            "year_max": int(years[:, p].max()),
        }
        for k, v in sorted(shares.items()):
            x = np.asarray(v[:, p], dtype=np.float64)
            row[k] = {
                "mean": float(x.mean()),
                "q": [float(q) for q in np.quantile(x, QUANTILES)],
            }
        out["types"][name] = row
    return out


def check_leakage(C, shares, own_pairs):
    """P2's split and the counts (module docstring, leakage)."""
    if int(C["bad_order"]):
        raise ValueError(
            f"{C['bad_order']} training pairs have first_ts " f"after last_ts"
        )
    if int(np.sum(C["first_n"])) != int(C["total_n"]) or int(
        np.sum(C["last_n"])
    ) != int(C["total_n"]):
        raise ValueError("the minute tables do not count every training " "pair once")
    if not np.array_equal(
        np.asarray(C["own_n"], dtype=np.int64), np.asarray(own_pairs, dtype=np.int64)
    ):
        raise ValueError("the train pairs per eval listener differ from the " "CSR")
    if np.any(C["own_lo_n"] != 0):
        raise ValueError(
            "a training pair of an eval listener begins after "
            "its pick's anchor, which P2's split rules out"
        )
    if np.any(C["own_hi_n"][:, 0] != 0):
        raise ValueError(
            "a training pair of an eval listener ends after "
            "its test_last pick, which P2's split rules out"
        )
    for k, v in shares.items():
        if not (np.all(v >= 0) and np.all(v <= 1)):
            raise ValueError(f"a leakage share {k} lies outside 0 to 1")
    if np.any(shares["global_lo_pairs"] > shares["global_hi_pairs"]):
        raise ValueError("a lower leakage bound exceeds its upper bound")


class Inputs:
    """The real inputs that P13's loaders do not give: DuckDB queries,
    P9b's lists, the ALS factors and P5's and P6's stored eval ranks. The
    smoke test replaces this class."""

    def __init__(self, model_dir, work):
        self.d = Path(model_dir)
        self.work = Path(work)

    def key(self):
        """What the saved DuckDB inputs depend on."""
        return {
            name: file_sha256(self.d / name)
            for name in (
                "splits.parquet",
                "eval_users.parquet",
                "item_covariates.parquet",
                "stats_item_train.parquet",
                "train_manifest.json",
            )
        }

    def frame(self):
        """(dict of the eval frame, P5's frame) from splits.parquet."""
        con = connect(4)
        try:
            rows = con.execute(f"""
                WITH picks AS (
                    SELECT user_id,
                           max(CASE WHEN is_test_last THEN item_id END) AS tl,
                           max(CASE WHEN is_test_last THEN first_ts END)
                               AS tl_f,
                           max(CASE WHEN is_test_last THEN last_ts END) AS tl_l,
                           max(CASE WHEN is_test_first THEN item_id END) AS tf,
                           max(CASE WHEN is_test_first THEN first_ts END)
                               AS tf_f,
                           max(CASE WHEN is_test_first THEN last_ts END) AS tf_l
                    FROM read_parquet('{self.d / "splits.parquet"}')
                    GROUP BY user_id)
                SELECT e.user_id, e.activity_decile, p.tl, p.tf, p.tl_f,
                       p.tl_l, p.tf_f, p.tf_l
                FROM read_parquet('{self.d / "eval_users.parquet"}') e
                LEFT JOIN picks p ON e.user_id = p.user_id
                ORDER BY e.user_id""").fetchall()
            p5 = ev.load_eval_frame(con)
        finally:
            con.close()
        keep = [r for r in rows if all(x is not None for x in r[2:])]
        fr = {
            "user": np.array([r[0] for r in keep], dtype=np.int64),
            "decile": np.array([r[1] for r in keep], dtype=np.int64),
            "picks": np.array([[r[2], r[3]] for r in keep], dtype=np.int64),
            "ts_first": np.array([[r[4], r[6]] for r in keep], dtype=np.int64),
            "ts_last": np.array([[r[5], r[7]] for r in keep], dtype=np.int64),
            "dropped": int(len(rows) - len(keep)),
        }
        return fr, p5

    def covariates(self, n_slots):
        con = connect(4)
        try:
            return ev.covariate_arrays(con, n_slots)
        finally:
            con.close()

    def popularity(self, n_slots, id_min):
        con = connect(4)
        try:
            return ev.popularity_scores(con, n_slots, id_min)
        finally:
            con.close()

    def leakage(self, frame, say):
        """The aggregates of leakage_counts() from the 16 train files."""
        import pandas as pd

        files = sorted(globlib.glob(str(self.d / "interactions_train_*.parquet")))
        if not files:
            raise ValueError("no interactions_train_*.parquet files")
        users = frame["user"]
        anchor = anchor_of(frame)
        n_u = len(users)
        parts = {"first": [], "last": []}
        bad = total_n = total_l = 0
        own = {
            k: np.zeros((n_u, 2) if "_" in k else n_u, dtype=np.int64)
            for k in ("n", "l", "lo_n", "lo_l", "hi_n", "hi_l")
        }
        con = connect(8)
        try:
            con.register(
                "p14_anchor",
                pd.DataFrame(
                    {"user_id": users, "a_last": anchor[:, 0], "a_first": anchor[:, 1]}
                ),
            )
            prog = Progress(
                f"{WORK_NAME}: leakage counts (three scans per " f"file)",
                len(files),
                say,
                unit=" files",
            )
            for path in files:
                for col in ("first", "last"):
                    z = con.execute(f"""
                        SELECT {col}_ts // 60 AS m, count(*) AS n,
                               CAST(sum(listens) AS BIGINT) AS l,
                               count(*) FILTER (WHERE first_ts > last_ts)
                                   AS bad
                        FROM read_parquet('{path}')
                        GROUP BY 1""").fetchnumpy()
                    if np.ma.is_masked(z["m"]) or np.ma.is_masked(z["l"]):
                        raise ValueError(
                            f"{path}: a training pair has no " f"{col}_ts or no listens"
                        )
                    parts[col].append(
                        {k: np.asarray(z[k], dtype=np.int64) for k in ("m", "n", "l")}
                    )
                    if col == "first":
                        bad += int(np.sum(z["bad"]))
                        total_n += int(np.sum(z["n"]))
                        total_l += int(np.sum(z["l"]))
                z = con.execute(f"""
                    SELECT t.user_id AS user_id, count(*) AS n,
                           CAST(sum(t.listens) AS BIGINT) AS l,
                           count(*) FILTER (WHERE t.first_ts > a.a_last) AS lo0,
                           coalesce(CAST(sum(t.listens) FILTER (
                               WHERE t.first_ts > a.a_last) AS BIGINT), 0)
                               AS lol0,
                           count(*) FILTER (WHERE t.first_ts > a.a_first)
                               AS lo1,
                           coalesce(CAST(sum(t.listens) FILTER (
                               WHERE t.first_ts > a.a_first) AS BIGINT), 0)
                               AS lol1,
                           count(*) FILTER (WHERE t.last_ts > a.a_last) AS hi0,
                           coalesce(CAST(sum(t.listens) FILTER (
                               WHERE t.last_ts > a.a_last) AS BIGINT), 0)
                               AS hil0,
                           count(*) FILTER (WHERE t.last_ts > a.a_first) AS hi1,
                           coalesce(CAST(sum(t.listens) FILTER (
                               WHERE t.last_ts > a.a_first) AS BIGINT), 0)
                               AS hil1
                    FROM read_parquet('{path}') t
                    JOIN p14_anchor a ON t.user_id = a.user_id
                    GROUP BY t.user_id""").fetchnumpy()
                u = np.asarray(z["user_id"], dtype=np.int64)
                k = np.searchsorted(users, u)
                if len(u) and not np.array_equal(users[np.minimum(k, n_u - 1)], u):
                    raise ValueError("the join returned a user outside the " "frame")
                own["n"][k] += np.asarray(z["n"], dtype=np.int64)
                own["l"][k] += np.asarray(z["l"], dtype=np.int64)
                for c in range(2):
                    own["lo_n"][k, c] += np.asarray(z[f"lo{c}"], np.int64)
                    own["lo_l"][k, c] += np.asarray(z[f"lol{c}"], np.int64)
                    own["hi_n"][k, c] += np.asarray(z[f"hi{c}"], np.int64)
                    own["hi_l"][k, c] += np.asarray(z[f"hil{c}"], np.int64)
                prog.advance(1)
            prog.finish()
        finally:
            con.close()
        out = {"total_n": total_n, "total_l": total_l, "bad_order": bad}
        m0 = min(int(p["m"].min()) for col in parts for p in parts[col])
        m1 = max(int(p["m"].max()) for col in parts for p in parts[col])
        for col in ("first", "last"):
            for unit in ("n", "l"):
                arr = np.zeros(m1 - m0 + 1, dtype=np.int64)
                for p in parts[col]:
                    np.add.at(arr, p["m"] - m0, p[unit])
                out[f"{col}_{unit}"] = arr
        out["m0"] = m0
        out.update(
            own_n=own["n"],
            own_l=own["l"],
            own_lo_n=own["lo_n"],
            own_lo_l=own["lo_l"],
            own_hi_n=own["hi_n"],
            own_hi_l=own["hi_l"],
        )
        return out

    def lists(self, n_slots):
        return knn.load_lists(self.d, n_slots)

    def als(self, which):
        """(X, Y, a fingerprint) of the phase D winner ("d") or of P6
        ("p6"), memory-mapped. The phase D files must match best.json."""
        if which == "d":
            best = load_json(self.d / "als_d" / "best.json")
            if best is None:
                raise ValueError(
                    "als_d/best.json is missing: run python3 -m "
                    "pipeline.runner --only als_phase_d first"
                )
            paths = {}
            for tag in ("user", "item"):
                p = self.d / "als_d" / best["files"][tag]["file"]
                sha = file_fingerprint(p, self.work / "fingerprints.json")
                if sha != best["files"][tag]["sha256"]:
                    raise ValueError(
                        f"{p.name} does not match its SHA-256 in " f"best.json"
                    )
                paths[tag] = p
            fpr = {k: best["files"][k]["sha256"] for k in ("user", "item")}
        elif which == "p6":
            paths = {
                "user": self.d / "als_user_factors.npy",
                "item": self.d / "als_item_factors.npy",
            }
            fpr = {"marker:p6_als": file_sha256(marker_path("p6_als"))}
        else:
            raise ValueError(which)
        return (
            np.load(paths["user"], mmap_mode="r"),
            np.load(paths["item"], mmap_mode="r"),
            fpr,
        )

    def als_reference(self, which):
        """The selection score that best.json records for the phase D
        winner, or None (P6 records none)."""
        if which != "d":
            return None
        return float(load_json(self.d / "als_d" / "best.json")["score"])

    def stored_ranks(self, which):
        """(user ids, (n, 2) test ranks) as P5 or P6 stored them."""
        if which == "p5":
            z = np.load(self.d / "eval_baseline_ranks.npz")
            return (
                np.asarray(z["user_id"], dtype=np.int64),
                np.stack([z["rank_last"], z["rank_first"]], axis=1).astype(np.int64),
            )
        z = np.load(self.d / "eval_als_ranks.npz")
        return (
            np.asarray(z["user_id"], dtype=np.int64),
            np.stack([z["rank_test_last"], z["rank_test_first"]], axis=1).astype(
                np.int64
            ),
        )


def anchor_of(frame):
    """(n, 2) anchors: last_ts of test_last, first_ts of test_first."""
    return np.stack([frame["ts_last"][:, 0], frame["ts_first"][:, 1]], axis=1).astype(
        np.int64
    )


FRAME_KEYS = ("user", "decile", "picks", "ts_first", "ts_last")
LEAK_KEYS = (
    "first_n",
    "first_l",
    "last_n",
    "last_l",
    "own_n",
    "own_l",
    "own_lo_n",
    "own_lo_l",
    "own_hi_n",
    "own_hi_l",
)
LEAK_SCALARS = ("m0", "total_n", "total_l", "bad_order")


def duck_inputs(io, work, n_slots, id_min, say):
    """The frame, the covariates, the popularity and the leakage counts,
    saved in work and keyed by io.key()."""
    path, meta = work / "inputs.npz", work / "inputs.json"
    k = {"version": INPUTS_VERSION, "inputs": io.key(), "n_slots": int(n_slots)}
    if path.exists() and load_json(meta) == {"k": s13._sha(k)}:
        z = np.load(path)
        frame = {key: np.asarray(z[f"frame_{key}"]) for key in FRAME_KEYS}
        frame["dropped"] = int(z["frame_dropped"])
        leak = {key: np.asarray(z[f"leak_{key}"]) for key in LEAK_KEYS}
        leak.update({key: int(z[f"leak_{key}"]) for key in LEAK_SCALARS})
        say(f"{WORK_NAME}: the DuckDB inputs are saved, so no query runs")
        return (
            frame,
            (z["cov_pop"], z["cov_pairs"], z["cov_tier"]),
            np.asarray(z["pop"]),
            leak,
        )
    t0 = time.time()
    say(
        f"{WORK_NAME}: loading the eval frame, the covariates and the train "
        f"popularity with DuckDB (no progress lines in this step)"
    )
    frame, p5 = io.frame()
    if len(frame["user"]) < 2 or np.any(np.diff(frame["user"]) <= 0):
        raise ValueError("the eval frame's listeners are not sorted and " "distinct")
    user, decile, tl, tf, dropped = p5
    if not (
        np.array_equal(frame["user"], user)
        and np.array_equal(frame["decile"], decile)
        and np.array_equal(frame["picks"][:, 0], tl)
        and np.array_equal(frame["picks"][:, 1], tf)
        and frame["dropped"] == int(dropped)
    ):
        raise ValueError(
            "the eval frame differs from P5's route " "(evaluation.load_eval_frame)"
        )
    cov = io.covariates(n_slots)
    pop = np.asarray(io.popularity(n_slots, id_min), dtype=np.int64)
    say(
        f"{WORK_NAME}: {len(frame['user']):,} eval listeners with both test "
        f"picks ({frame['dropped']} without), loaded in "
        f"{_clock(time.time() - t0)}"
    )
    leak = io.leakage(frame, say)
    arrays = {f"frame_{key}": frame[key] for key in FRAME_KEYS}
    arrays["frame_dropped"] = np.int64(frame["dropped"])
    arrays.update({f"leak_{key}": np.asarray(leak[key]) for key in LEAK_KEYS})
    arrays.update({f"leak_{key}": np.int64(leak[key]) for key in LEAK_SCALARS})
    arrays.update(cov_pop=cov[0], cov_pairs=cov[1], cov_tier=cov[2], pop=pop)
    gr._atomic_savez(path, **arrays)
    atomic_json_write({"k": s13._sha(k)}, meta)
    return frame, cov, pop, leak


# Models


def load_models(d, base):
    """The final model, its replicates, the S3 set, artist terms only and
    the cold-track GAM, checked, with the three GAM arms' tables in the
    final layout."""
    M, rec, ptr, final_ranks, bounded = cold.load_final(d, base)
    terms = list(M["lay"].terms)
    z = gkd.load_npz(d / ptr["files"]["arrays"])
    boot = {s: np.asarray(z[f"theta_boot_{s}"], dtype=np.float64) for s in pc.SCHEMES}
    for s, th in boot.items():
        if th.ndim != 2 or th.shape[1] != len(M["theta"]):
            raise ValueError(f"the {s} replicates do not fit the final model")
    zs = gkd.load_npz(d / pc.P13D_FILES[1])
    if not np.array_equal(np.asarray(zs["idx"]), np.asarray(M["idx"])):
        raise ValueError(
            "p13d_model_set.npz holds tables in another index "
            "space than the final model's"
        )
    members = [str(x) for x in zs["members"].tolist()]
    mtab = np.asarray(zs["member_tables"], dtype=np.float64)
    if mtab.shape != (len(members), M["lay"].L):
        raise ValueError("the S3 member tables do not fit the final layout")
    gdoc = load_json(d / "gam_refit.json")
    if gdoc is None or gdoc.get("version") != gr.VERSION:
        raise ValueError("gam_refit.json is missing or has another version")
    c_ref = next(i for i, f in enumerate(gdoc["fits"]) if f["name"] == gr.REFERENCE)
    fa = gdoc["fits"][c_ref]
    lay_a, _ = gr.layout_of(base, fa["kept"])
    if [t.to_json() for t in lay_a.terms] != fa["terms"] or gam_arms.layout_fingerprint(
        lay_a
    ) != fa["fingerprint"]:
        raise ValueError(
            "the layout of artist terms only, rebuilt here, "
            "differs from gam_refit.json's"
        )
    if not np.array_equal(
        lay_a.tables(np.asarray(fa["theta"])), np.asarray(fa["tables_raw"])
    ):
        raise ValueError("the raw tables of artist terms only are not M " "theta")
    zr = gkd.load_npz(d / "gam_refit.npz")
    if not np.array_equal(np.asarray(zr["user_id"]), base["sel_users"]):
        raise ValueError("gam_refit.npz holds other listeners than P13's")
    cdoc = load_json(d / cold.OUT_FILES[0])
    if cdoc is None or cdoc.get("version") != cold.VERSION:
        raise ValueError(
            f"{cold.OUT_FILES[0]} is missing: run python3 -m "
            f"pipeline.runner --only p14_cold_gam first"
        )
    cold.check_outputs(d, base, M, bounded, final_ranks)
    _, _, lay_c, _ = cold.cold_layout(base, M, bounded)
    zc = gkd.load_npz(d / cold.OUT_FILES[1])
    tables = np.stack(
        [
            np.asarray(M["tables"], dtype=np.float64),
            place(terms, lay_a.terms, fa["tables"]),
            place(terms, lay_c.terms, cdoc["model"]["tables"]),
        ]
    )
    sel = {
        "gam": final_ranks,
        "artist_only": np.asarray(zr["ranks"][c_ref]),
        "cold_gam": np.asarray(zc["ranks"][0]),
    }
    return {
        "M": M,
        "rec": rec,
        "ptr": ptr,
        "terms": terms,
        "bounded": bounded,
        "boot": boot,
        "members": members,
        "member_tables": mtab,
        "tables": tables,
        "sel": sel,
        "artist": {"name": fa["name"], "kept": list(fa["kept"])},
        "cold": {
            "kept": list(cdoc["model"]["kept"]),
            "bounded": list(cdoc["model"]["bounded"]),
        },
    }


# The pass over the listeners
_E = {}  # the pass context, set before a fork


def term_sums(terms, tables, conc, tier):
    """(C, n) scores of n slots: for each table, the sum in term order of
    the term values, by the expressions of catalog.reference_scores().
    conc: {source: (value, has)} of the slots. tier: their tier codes.

    Each expression works slot by slot, so the result equals
    reference_scores() on the same slots, whatever slots come with them.
    Two shortcuts keep the cost down: np.interp runs only on the slots on
    a shape, and a term whose table is zero in a configuration is skipped.
    Its values are +0.0, so the skip can only leave -0.0 where the
    reference has +0.0. The two compare equal, so every comparison and
    np.array_equal give the same result."""
    T = np.atleast_2d(np.asarray(tables, dtype=np.float64))
    n = len(tier)
    S = np.zeros((T.shape[0], n), dtype=np.float64)
    for t, off in zip(terms, offsets_of(terms)):
        tab = T[:, off : off + t.size]
        live = [c for c in range(T.shape[0]) if np.any(tab[c] != 0.0)]
        if not live:
            continue
        if t.multiplier is not None:
            raise ValueError("the pass supports no term with a multiplier")
        v, h = conc[t.source]
        K, kn = t.K, t.knot_array
        app = None if t.tiers is None else np.isin(tier, np.asarray(t.tiers))
        on = h if app is None else h & app
        # the state of each slot: 0 not applicable (or on the shape, which
        # np.interp fills), 1 no data, 2 exactly zero
        code = (~h if app is None else app & ~h).astype(np.intp)
        if t.zero_state:
            zero = on & (v == 0.0)
            code += 2 * zero
            on = on & ~zero
        idx = np.flatnonzero(on)
        x = np.clip(v[idx], kn[0], kn[-1])
        if np.isnan(x).any():
            raise ValueError(f"{t.source}: a slot with data has no value")
        for c in live:
            lut = np.array(
                [
                    0.0,
                    tab[c, K + gam.MISS] if t.missing_state else 0.0,
                    tab[c, K + gam.ZERO] if t.zero_state else 0.0,
                ]
            )
            val = lut[code]
            val[idx] = np.interp(x, kn, tab[c, :K])
            S[c] += val
    return S


def _range_scores(c, state, pre, a, b):
    """(scores (C, b - a), corrections, concepts) of slots a..b-1: the term
    sums of term_sums() and CatalogScorer's corrections of each listener
    term, for the basis sums."""
    scorer, space = c["scorer"], c["space"]
    conc = {}
    for t in scorer.terms:
        if t.source not in conc:
            conc[t.source] = scorer.concept_range(state, space, pre, t.source, a, b)
    S = term_sums(scorer.terms, scorer.T, conc, scorer.tier[a:b])
    corr = {
        k: scorer._corrections(k, *conc[scorer.terms[k].source], a, b)
        for k in scorer.user_terms
    }
    return S, corr, conc


def _corr_basis(scorer, corr, out_mask):
    """The sum of the correction triples (module catalog) over the slots
    of a range that out_mask leaves in: (1 - w) at the slot, w at the next
    slot and -1 at the item-level slot."""
    L = scorer.L
    out = np.zeros(L)
    for k in scorer.user_terms:
        rows, slot, w, bslot = corr[k]
        if not len(rows):
            continue
        keep = ~out_mask[rows]
        off = int(scorer.offsets[k])
        s_ = off + slot[keep].astype(np.int64)
        w_ = w[keep]
        out += np.bincount(s_, weights=1.0 - w_, minlength=L)
        out += np.bincount(s_ + 1, weights=w_, minlength=L)
        out -= np.bincount(off + bslot[keep].astype(np.int64), minlength=L)
    return out


def _reference_check(c, S, conc, sub, a, u):
    """Raise RuntimeError unless the scores S[:, sub] of the range that
    starts at slot a equal catalog.reference_scores() on those slots."""
    scorer = c["scorer"]
    full = {src: (v[sub], h[sub]) for src, (v, h) in conc.items()}
    ref = ct.reference_scores(full, scorer.terms, scorer.T, 0.0, scorer.tier[a + sub])
    if not np.array_equal(S[:, sub], ref):
        gap = float(np.max(np.abs(S[:, sub] - ref)))
        raise RuntimeError(
            f"listener {u}: the pass's scores differ from "
            f"catalog.reference_scores near slot {a} (largest "
            f"gap {gap:.1e})"
        )


def _range_top(s, a, keep, floor=-np.inf):
    """(slots, scores) of the keep best finite scores of one range at or
    above floor (the current list's last score), ties by slot: every entry
    at or above the keep-th largest score, then a sort."""
    if floor > -np.inf:
        idx = np.flatnonzero(s >= floor)
        if len(idx) > 4 * keep:
            sub = s[idx]
            thr = np.partition(sub, len(sub) - keep)[len(sub) - keep]
            idx = idx[sub >= thr]
    else:
        fin = np.isfinite(s)
        n_fin = int(fin.sum())
        if n_fin == 0:
            return np.empty(0, dtype=np.int64), np.empty(0)
        if n_fin <= keep:
            idx = np.flatnonzero(fin)
        else:
            thr = np.partition(s[fin], n_fin - keep)[n_fin - keep]
            idx = np.flatnonzero(s >= thr)
    idx = idx[np.lexsort((idx, -s[idx]))[:keep]]
    return a + idx.astype(np.int64), s[idx]


def _merge_top(ids, sc, new_ids, new_sc, keep):
    ids = np.concatenate((ids, new_ids))
    sc = np.concatenate((sc, new_sc))
    order = np.lexsort((ids, -sc))[:keep]
    return ids[order], sc[order]


def _concepts_at(c, state, pre, slots):
    """(values, has) of the terms' sources at single slots, from one-slot
    ranges (equal to the catalog's values bit for bit)."""
    scorer, space = c["scorer"], c["space"]
    J = len(scorer.terms)
    val = np.full((len(slots), J), np.nan)
    has = np.zeros((len(slots), J), dtype=bool)
    for i, s in enumerate(slots):
        cache = {}
        for j, t in enumerate(scorer.terms):
            if t.source not in cache:
                cache[t.source] = scorer.concept_range(
                    state, space, pre, t.source, int(s), int(s) + 1
                )
            v, h = cache[t.source]
            val[i, j], has[i, j] = v[0], h[0]
    return val, has


def _evidence(c, state, item):
    """The evidence units of each reason of one track (module docstring)."""
    ev_ = cn.evidence(state, c["space"], int(item))
    G, EK = len(c["groups"]), EVIDENCE_K
    count = np.zeros(G, dtype=np.int32)
    ids = np.full((G, EK), -1, dtype=np.int64)
    vals = np.full((G, EK), np.nan)
    credit, credit_n = -1, 0
    for g, (_, js) in enumerate(c["groups"]):
        rows = None
        for j in js:
            src = c["sources"][j]
            e = ev_[src]
            if src in cn.CREDIT_CONCEPTS:
                if e["has_data"] and e["credit"] is not None:
                    credit, credit_n = int(e["credit"]), len(e["terms"])
                continue
            if rows is not None:
                raise ValueError("a reason with two lists of evidence units")
            rows = list(e["terms"]) if e["has_data"] else []
            if src == "audio_fit":
                rows = [
                    {"id": cn.AUDIO_DIMS.index(r["id"]), "term": r["term"]}
                    for r in sorted(rows, key=lambda r: (-r["term"], r["id"]))
                ]
        rows = rows or []
        count[g] = len(rows)
        for i, r in enumerate(rows[:EK]):
            ids[g, i] = int(r["id"])
            vals[g, i] = float(r["term"])
    return count, ids, vals, credit, credit_n


def _reference_at(c, val, has, slots):
    """catalog.reference_scores() (C, len(slots)) at single slots from the
    concept values per term that _concepts_at() gives."""
    scorer = c["scorer"]
    full = {}
    for j, t in enumerate(scorer.terms):
        full.setdefault(t.source, (val[:, j], has[:, j]))
    return ct.reference_scores(
        full, scorer.terms, scorer.T, 0.0, c["space"].tier[np.asarray(slots)]
    )


def _eval_one(c, k):
    """Everything the pass keeps for eval listener k (module docstring)."""
    scorer, space = c["scorer"], c["space"]
    u = int(c["users"][k])
    picks = np.asarray(c["picks"][k], dtype=np.int64)
    items, weights = cd.history_of(c["u_indptr"], c["u_indices"], c["u_data"], u)
    state = cn.build_user_state(items, weights, space)
    excl = np.unique(items)
    scorer._check_picks(picks, excl)
    pre = scorer.prepare(state, space)
    n, chunk, C, L = scorer.n, int(c["chunk"]), scorer.C, scorer.L
    first = list(dict.fromkeys(int(t) // chunk for t in picks))
    held = {
        r: _range_scores(c, state, pre, r * chunk, min((r + 1) * chunk, n))
        for r in first
    }
    s_pick = np.stack(
        [held[int(t) // chunk][0][:, int(t) % chunk] for t in picks]
    )  # (picks, C)
    n_p = len(picks)
    greater = np.zeros((n_p, C), dtype=np.int64)
    equal = np.zeros((n_p, C), dtype=np.int64)  # the pick counts itself
    before = np.zeros((n_p, C), dtype=np.int64)  # equal, lower slot
    top = [(np.empty(0, dtype=np.int64), np.empty(0)) for _ in range(C)]
    corr_sum = np.zeros(L)
    score_sum = np.zeros(C)
    n_checked = 0
    for r in range(-(-n // chunk)):
        a, b = r * chunk, min((r + 1) * chunk, n)
        S, corr, conc = held.pop(r) if r in held else _range_scores(c, state, pre, a, b)
        lo, hi = np.searchsorted(excl, (a, b))
        out_mask = np.zeros(b - a, dtype=bool)
        out_mask[excl[lo:hi] - a] = True
        if a == 0:
            out_mask[0] = True
        corr_sum += _corr_basis(scorer, corr, out_mask)
        score_sum += S[:, ~out_mask].sum(axis=1)
        S[:, out_mask] = -np.inf
        for p, t in enumerate(picks):
            sp_ = s_pick[p][:, None]
            greater[p] += np.count_nonzero(S > sp_, axis=1)
            eq = S == sp_
            equal[p] += np.count_nonzero(eq, axis=1)
            e = min(max(int(t) - a, 0), b - a)
            if e:
                before[p] += np.count_nonzero(eq[:, :e], axis=1)
        sub = [np.arange(0, b - a, CHECK_STRIDE), picks[(picks >= a) & (picks < b)] - a]
        for j in range(C):
            floor = top[j][1][-1] if len(top[j][1]) == TOP_KEEP else -np.inf
            ids, sc = _range_top(S[j], a, TOP_KEEP, floor)
            top[j] = _merge_top(top[j][0], top[j][1], ids, sc, TOP_KEEP)
            sub.append(ids - a)
        sub = np.unique(np.concatenate(sub))
        sub = sub[~out_mask[sub]]
        _reference_check(c, S, conc, sub, a, u)
        n_checked += len(sub)
    if np.any(equal < 1):
        raise RuntimeError(f"listener {u}: a pick does not equal its own " f"score")
    base_cand = c["base_counts"].copy()
    ex_all = np.concatenate(([0], excl))
    for kk in scorer.user_terms:
        t, off = scorer.terms[kk], int(scorer.offsets[kk])
        base_cand[off : off + t.size] -= np.bincount(
            scorer.base_slot[kk][ex_all], minlength=t.size
        )
    for idx, w in c["item_basis"]:
        base_cand -= np.bincount(
            idx[ex_all], weights=1.0 - w[ex_all], minlength=L
        ) + np.bincount(idx[ex_all] + 1, weights=w[ex_all], minlength=L)
    n_cand = int(n - len(ex_all))
    top_id = np.full((C, TOP_KEEP), -1, dtype=np.int64)
    top_sc = np.full((C, TOP_KEEP), -np.inf)
    for j in range(C):
        top_id[j, : len(top[j][0])] = top[j][0]
        top_sc[j, : len(top[j][1])] = top[j][1]
    R = min(REASON_K, int(np.count_nonzero(top_id[0] >= 0)))
    if R < REASON_K:
        raise ValueError(f"listener {u} has fewer than {REASON_K} candidates")
    t_slots = top_id[0, :REASON_K]
    t_val, t_has = _concepts_at(c, state, pre, t_slots)
    p_val, p_has = _concepts_at(c, state, pre, picks)
    if not (
        np.array_equal(_reference_at(c, t_val, t_has, t_slots)[0], top_sc[0, :REASON_K])
        and np.array_equal(_reference_at(c, p_val, p_has, picks), s_pick.T)
    ):
        raise RuntimeError(
            f"listener {u}: the scores of the picks or of the "
            f"GAM's top {REASON_K} differ from "
            f"catalog.reference_scores on their own concepts"
        )
    J = len(scorer.terms)
    t_slot = np.empty((REASON_K, J), dtype=np.int16)
    t_w = np.empty((REASON_K, J))
    tier = space.tier[t_slots]
    for j, t in enumerate(scorer.terms):
        t_slot[:, j], t_w[:, j] = gam.encode(t, t_val[:, j], t_has[:, j], tier)
    T0, D0 = scorer.T[0], scorer.D[0]
    s_terms = np.zeros(REASON_K)
    for j, off in enumerate(scorer.offsets):
        s_terms += gam.pwl_value(
            T0, D0, int(off) + t_slot[:, j].astype(np.int64), t_w[:, j]
        )
    gap = np.abs(s_terms - top_sc[0, :REASON_K])
    if np.any(gap > SUM_ATOL * np.maximum(1.0, np.abs(s_terms))):
        raise ValueError(
            f"listener {u}: the term values of a top track do "
            f"not sum to its score (gap {gap.max():.1e})"
        )
    G_ = len(c["groups"])
    ev_count = np.zeros((REASON_K, G_), dtype=np.int32)
    ev_id = np.full((REASON_K, G_, EVIDENCE_K), -1, dtype=np.int64)
    ev_val = np.full((REASON_K, G_, EVIDENCE_K), np.nan)
    ev_credit = np.full(REASON_K, -1, dtype=np.int64)
    ev_credit_n = np.zeros(REASON_K, dtype=np.int64)
    for i, s in enumerate(t_slots):
        ev_count[i], ev_id[i], ev_val[i], ev_credit[i], ev_credit_n[i] = _evidence(
            c, state, s
        )
    return {
        "ranks": (1 + greater + before).T,
        "span_lo": (1 + greater).T,
        "span_hi": (greater + equal).T,
        "pick_score": s_pick.T,
        "n_checked": np.int64(n_checked),
        "top_id": top_id,
        "top_score": top_sc,
        "n_cand": np.int64(n_cand),
        "score_sum": score_sum,
        "bsum": base_cand + corr_sum,
        "top_slot": t_slot,
        "top_w": t_w,
        "top_val": t_val,
        "top_has": t_has,
        "pick_val": p_val,
        "pick_has": p_has,
        "ev_count": ev_count,
        "ev_id": ev_id,
        "ev_val": ev_val,
        "ev_credit": ev_credit,
        "ev_credit_tracks": ev_credit_n,
    }


def basis_counts(scorer, space):
    """The sum of the basis vectors over every item slot of the item-level
    part: per listener term the count of its item-level slot, per item term
    (popularity) its (1 - w, w) weights. Returns (sums, [(flat index, w)]
    of each item term, for the per-listener subtraction)."""
    out = np.zeros(scorer.L)
    item = []
    for kk, t in enumerate(scorer.terms):
        off = int(scorer.offsets[kk])
        if kk in scorer.user_terms:
            out[off : off + t.size] = np.bincount(
                scorer.base_slot[kk], minlength=t.size
            )
            continue
        if t.source != "popularity" or t.tiers is not None:
            raise ValueError(
                f"term {t.name} is neither a listener term nor " f"the popularity term"
            )
        slot, w = gam.encode(
            t, space.popularity, scorer.item_has[t.source], scorer.tier
        )
        idx = off + slot.astype(np.int64)
        out += np.bincount(idx, weights=1.0 - w, minlength=scorer.L) + np.bincount(
            idx + 1, weights=w, minlength=scorer.L
        )
        item.append((idx, w))
    return out, item


def _eval_task(task):
    lo, hi = task
    with threadpool_limits(1):  # many workers, one BLAS thread each
        outs = [_eval_one(_E, k) for k in range(lo, hi)]
    return {key: np.stack([o[key] for o in outs]) for key in outs[0]}


def _same(a, b):
    return all(
        np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == "f")) for k in a
    )


def eval_pass(st, terms, tables, groups, users, picks, work, key, say, workers=None):
    """The pass over the eval listeners (module docstring): per-listener
    arrays, listeners first. Blocks are saved as they finish, so a stopped
    run continues with the blocks still missing."""
    n_slots, _ = st._id_slots()
    u_indptr, u_indices, u_data = st._csr()
    n_blocks = max(1, -(-len(users) // EVAL_BLOCK))
    tasks = s13._blocks(np.ones(len(users), dtype=np.int64), n_blocks)
    pdir = work / PASS_DIR
    state_path = pdir / "state.json"
    k = {
        "key": s13._sha(key),
        "tables": s13._array_sha(tables),
        "users": _array_key(users, picks),
        "blocks": [list(t) for t in tasks],
        "settings": [TOP_KEEP, REASON_K, EVIDENCE_K, EVAL_CHUNK, CHECK_STRIDE],
    }
    state = load_json(state_path) or {}
    if state.get("k") != k:
        pdir.mkdir(parents=True, exist_ok=True)
        for f in pdir.glob("part_*.npz"):
            f.unlink()
        state = {"k": k, "parts": {}}
        atomic_json_write(state, state_path)

    def part(i):
        return pdir / f"part_{i:05d}.npz"

    done = {
        int(i)
        for i, sha in state["parts"].items()
        if part(int(i)).exists() and file_sha256(part(int(i))) == sha
    }
    pending = [i for i in range(len(tasks)) if i not in done]
    if done:
        say(f"{WORK_NAME}: pass resume, {len(done)} of {len(tasks)} blocks " f"done")
    if pending:
        say(
            f"{WORK_NAME}: loading the item side with the transposed "
            f"co-listening lists (no progress lines in this step)"
        )
        t0 = time.time()
        space = st._space(st._sources(n_slots))
        scorer = ct.CatalogScorer(terms, tables, space, chunk=EVAL_CHUNK)
        if scorer.tilt is not None:
            raise ValueError("the pass supports no term with a multiplier")
        base_counts, item_basis = basis_counts(scorer, space)
        say(
            f"{WORK_NAME}: item side and scorer ready in " f"{_clock(time.time() - t0)}"
        )
        _E.update(
            scorer=scorer,
            space=space,
            users=users,
            picks=picks,
            u_indptr=u_indptr,
            u_indices=u_indices,
            u_data=u_data,
            chunk=EVAL_CHUNK,
            base_counts=base_counts,
            item_basis=item_basis,
            groups=groups,
            sources=[t.source for t in terms],
        )
        try:
            left = np.concatenate([np.arange(*tasks[i]) for i in pending])
            if workers is None:
                with cold.relabel_logs(
                    WORK_NAME, "python3 -m pipeline.runner " "--only p14_eval"
                ):
                    workers = st._selection_workers(
                        space, users[left], u_indptr, u_indices, n_slots
                    )
            take = np.unique(
                np.round(
                    np.linspace(0, len(left) - 1, min(PROBE_USERS, len(left)))
                ).astype(np.int64)
            )
            probe, secs = {}, []
            with threadpool_limits(1):
                for j, i in enumerate(take, start=1):
                    t1 = time.time()
                    probe[int(left[i])] = _eval_one(_E, int(left[i]))
                    secs.append(time.time() - t1)
                    say(
                        f"{WORK_NAME}: probe listener {j} of {len(take)}: "
                        f"{secs[-1]:.2f} s"
                    )
            par = max(
                1.0,
                min(float(workers), s13b.PASS_SPEEDUP_PER_CORE * s13._physical_cores()),
            )
            say(
                f"{WORK_NAME}: probe mean {np.mean(secs):.2f} s per listener "
                f"in one process. Rough estimate for {len(left):,} listeners "
                f"on {workers} workers, with a parallel speedup of {par:.1f}: "
                f"{_clock(np.mean(secs) * len(left) / par)}. The progress "
                f"lines below measure the real rate"
            )
            prog = Progress(
                f"{WORK_NAME}: eval pass, {len(tables)} "
                f"configurations, {workers} workers",
                len(left),
                say,
                unit=" listeners",
            )
            with s13._Pool(workers) as pool:
                for j, out in pool.unordered(_eval_task, [tasks[i] for i in pending]):
                    i = pending[j]
                    s13._atomic_savez(part(i), **out)
                    state["parts"][str(i)] = file_sha256(part(i))
                    atomic_json_write(state, state_path)
                    prog.advance(tasks[i][1] - tasks[i][0])
            prog.finish()
            starts = np.array([t[0] for t in tasks])
            for kk, res in probe.items():
                i = int(np.searchsorted(starts, kk, side="right") - 1)
                z = np.load(part(i))
                j = kk - tasks[i][0]
                if not _same(res, {key: z[key][j] for key in res}):
                    raise ValueError(
                        f"eval listener {int(users[kk])}: the "
                        f"pool's results differ from the probe's "
                        f"in this process"
                    )
        finally:
            _E.clear()
    zs = [np.load(part(i)) for i in range(len(tasks))]
    out = {key: np.concatenate([z[key] for z in zs]) for key in zs[0].files}
    if len(out["ranks"]) != len(users):
        raise ValueError("the pass blocks do not cover the eval listeners")
    return out


def verify_dense(st, terms, tables, users, picks, P, say, n_verify):
    """Raise ValueError unless n_verify listeners get their ranks, tie
    spans, pick scores and top lists again from catalog.reference_scores()
    over the full catalog exactly, their candidate counts too, and their
    score means and per-term backgrounds within CHECK_ATOL."""
    n_slots, _ = st._id_slots()
    space = st._space(st._sources(n_slots))
    u_indptr, u_indices, u_data = st._csr()
    rng = np.random.default_rng(VERIFY_SEED)
    take = np.sort(
        rng.choice(len(users), size=min(n_verify, len(users)), replace=False)
    )
    offs = offsets_of(terms)
    near, tied, worst = 0, 0, 0.0
    ids_all = np.arange(n_slots)
    prog = Progress(
        f"{WORK_NAME}: the pass against the dense reference",
        len(take),
        say,
        unit=" listeners",
    )
    for k in take:
        u = int(users[k])
        items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
        state = cn.build_user_state(items, weights, space)
        full = cn.full_catalog(state, space)
        ref = ct.reference_scores(full, terms, tables, 0.0, space.tier)
        cand = np.ones(n_slots, dtype=bool)
        cand[0] = False
        cand[items] = False
        nc = int(cand.sum())
        if nc != int(P["n_cand"][k]):
            raise ValueError(
                f"listener {u}: {int(P['n_cand'][k])} candidates "
                f"stored, {nc} in the dense reference"
            )
        for c in range(len(tables)):
            s = np.where(cand, ref[c], -np.inf)
            for p in range(2):
                t = int(picks[k, p])
                sp_ = s[t]
                g = int(np.count_nonzero(s > sp_))
                e = int(np.count_nonzero(s == sp_))
                want = (1 + g + int(np.count_nonzero(s[:t] == sp_)), 1 + g, g + e)
                got = (
                    int(P["ranks"][k, c, p]),
                    int(P["span_lo"][k, c, p]),
                    int(P["span_hi"][k, c, p]),
                )
                if not (P["pick_score"][k, c, p] == sp_ and got == want):
                    raise ValueError(
                        f"listener {u}, table {c}, pick {p}: "
                        f"rank and tie span {got} (score "
                        f"{P['pick_score'][k, c, p]!r}), the "
                        f"dense reference {want} (score "
                        f"{sp_!r})"
                    )
                tied += int(e > 1)
                lo, hi = ct.reference_rank(ref[c], t, items, s13.SCORE_TOL)
                near += int(hi > lo)
            order = np.lexsort((ids_all, -s))[:TOP_KEEP]
            if not (
                np.array_equal(P["top_id"][k, c], order)
                and np.array_equal(P["top_score"][k, c], s[order])
            ):
                raise ValueError(
                    f"listener {u}, table {c}: the top list "
                    f"differs from the dense reference"
                )
            gap = abs(float(ref[c][cand].mean()) - float(P["score_sum"][k, c]) / nc)
            worst = max(worst, gap)
            if gap > CHECK_ATOL:
                raise ValueError(
                    f"listener {u}, table {c}: the stored score "
                    f"sum differs from the dense one ({gap:.1e})"
                )
        for j, (t, off) in enumerate(zip(terms, offs)):
            Tj = np.zeros_like(tables[0])
            Tj[off : off + t.size] = tables[0][off : off + t.size]
            vj = ct.reference_scores(full, terms, Tj, 0.0, space.tier)[0]
            bg = (
                float(P["bsum"][k, off : off + t.size] @ tables[0][off : off + t.size])
                / nc
            )
            gap = abs(float(vj[cand].mean()) - bg)
            worst = max(worst, gap)
            if gap > CHECK_ATOL:
                raise ValueError(
                    f"listener {u}, term {t.name}: the basis "
                    f"background differs from the dense mean "
                    f"over the candidates ({gap:.1e})"
                )
        del full, ref
        prog.advance(1)
    prog.finish()
    return {
        "listeners": int(len(take)),
        "tied_picks": int(tied),
        "near_ties": int(near),
        "largest_mean_gap": worst,
    }


# Reasons


class TopReasons:
    """The reason values of any table of the final layout on the eval
    listeners' top lists, against each listener's candidate set."""

    def __init__(self, terms, kept, slot, w, bsum, n_cand):
        self.terms = list(terms)
        self.offsets = offsets_of(terms)
        self.groups = pc.reason_groups(list(kept))
        jx = {t.name: j for j, t in enumerate(self.terms)}
        self.gix = [[jx[n] for n in g] for _, g in self.groups]
        self.slot = np.asarray(slot)
        self.w = np.asarray(w, dtype=np.float64)
        self.bsum = np.asarray(bsum, dtype=np.float64)
        self.n_cand = np.asarray(n_cand, dtype=np.float64)
        self.on = np.stack([self.slot[:, :, j] < t.K for j, t in enumerate(self.terms)])
        self.cand = np.stack([self.on[g].any(axis=0) for g in self.gix])

    def parts(self, T):
        """(term values (J, n, R), backgrounds (J, n))."""
        T = np.asarray(T, dtype=np.float64)
        D = gam.diff_table(T, self.terms, self.offsets)
        n, R = self.slot.shape[:2]
        v = np.empty((len(self.terms), n, R))
        bg = np.empty((len(self.terms), n))
        for j, (t, off) in enumerate(zip(self.terms, self.offsets)):
            v[j] = gam.pwl_value(
                T, D, off + self.slot[:, :, j].astype(np.int64), self.w[:, :, j]
            )
            bg[j] = (
                self.bsum[:, off : off + t.size] @ T[off : off + t.size] / self.n_cand
            )
        return v, bg

    def contributions(self, T):
        v, bg = self.parts(T)
        return v - bg[:, :, None]

    def values(self, T):
        phi = self.contributions(T)
        return np.stack([phi[g].sum(axis=0) for g in self.gix])

    def ranks(self, val):
        G = len(self.groups)
        r = pc.ranks_of(val.reshape(G, -1), self.cand.reshape(G, -1))
        return r.reshape(val.shape)


def d1_rule(R, val, tables_iter, B):
    """D1 over B replicate tables: (shown, positive counts, top counts)."""
    pos = np.zeros(R.cand.shape, dtype=np.int64)
    top = np.zeros(R.cand.shape, dtype=np.int64)
    n = 0
    for T in tables_iter:
        v = R.values(T)
        pos += R.cand & (v > 0)
        top += R.cand & (R.ranks(v) <= pc.K_TOP)
        n += 1
    if n != B:
        raise ValueError(f"{n} replicate tables, expected {B}")
    shown = (
        R.cand
        & (val > 0)
        & (pos >= math.ceil(pc.SIGN_SHARE * B - 1e-9))
        & (top >= math.ceil(pc.TOP_SHARE * B - 1e-9))
    )
    return shown, pos, top


def s4_rule(R, member_tables):
    """S4 over the S3 members: (passes, top counts, needed count)."""
    ok = R.cand.copy()
    top = np.zeros(R.cand.shape, dtype=np.int64)
    for T in member_tables:
        v = R.values(T)
        ok &= v > 0
        top += R.cand & (R.ranks(v) <= pc.K_TOP)
    need = math.ceil(pd13.S4_TOP_SHARE * len(member_tables) - 1e-9)
    return ok & (top >= need), top, need


def reason_arrays(R, G, terms):
    """Every reason array from the tables: the model's values, D1 (listener
    and twoway), S4 and the shown mask. A pure function of its inputs."""
    M = G["M"]
    lay = M["lay"]
    val = R.values(M["tables"])
    d1, pos, top = d1_rule(
        R,
        val,
        (lay.tables(th) for th in G["boot"]["listener"]),
        len(G["boot"]["listener"]),
    )
    d1_tw, _, _ = d1_rule(
        R, val, (lay.tables(th) for th in G["boot"]["twoway"]), len(G["boot"]["twoway"])
    )
    s4, s4_top, need = s4_rule(R, list(G["member_tables"]))
    return {
        "reason_value": val,
        "candidate": R.cand,
        "shown_d1": d1,
        "shown_d1_twoway": d1_tw,
        "d1_positive": pos,
        "d1_top": top,
        "s4_pass": s4,
        "s4_top": s4_top,
        "shown": d1 & s4,
        "shown_twoway": d1_tw & s4,
    }, need


def check_reasons(R, G, P):
    """The sum, shift and direction checks (module docstring). Returns
    their largest gaps."""
    M = G["M"]
    T0 = np.asarray(M["tables"], dtype=np.float64)
    phi = R.contributions(T0)
    s = P["top_score"][:, 0, :REASON_K]
    mean_s = P["score_sum"][:, 0] / P["n_cand"]
    gap_sum = float(np.max(np.abs(phi.sum(axis=0) - (s - mean_s[:, None]))))
    if gap_sum > CHECK_ATOL:
        raise ValueError(
            f"the contributions do not sum to the score minus "
            f"the mean score over the candidates "
            f"({gap_sum:.1e})"
        )
    v, _ = R.parts(T0)
    gap_v = float(np.max(np.abs(v.sum(axis=0) - s) / np.maximum(1.0, np.abs(s))))
    if gap_v > SUM_ATOL:
        raise ValueError(f"the term values do not sum to the score " f"({gap_v:.1e})")
    raw = R.contributions(M["lay"].tables(M["theta"]))
    gap_shift = float(np.max(np.abs(raw - phi)))
    if gap_shift > CHECK_ATOL:
        raise ValueError(
            f"the raw and the centered tables give other "
            f"contributions ({gap_shift:.1e})"
        )
    return {"sum": gap_sum, "term_values": gap_v, "shift": gap_shift}


def local_slope(R, T, j):
    """(n, R) sign class of term j's shape at each top track: 1 rising, 0
    flat (including the clamp at the top knot), -1 falling, -2 off the
    shape (a zero or missing state)."""
    t, off = R.terms[j], R.offsets[j]
    D = gam.diff_table(np.asarray(T, dtype=np.float64), R.terms, R.offsets)
    s = R.slot[:, :, j].astype(np.int64)
    d = D[off + np.minimum(s, t.K - 1)]
    out = np.where(d > s13b.MONOTONE_TOL, 1, np.where(d < -s13b.MONOTONE_TOL, -1, 0))
    out = np.where(s == t.K - 1, 0, out)
    return np.where(s < t.K, out, -2)


def summarize_reasons(R, A, need, P, G):
    """The reason results from the reason arrays: a pure function, so
    check_outputs() recomputes it."""
    M = G["M"]
    T0 = np.asarray(M["tables"], dtype=np.float64)
    val, cand, shown = A["reason_value"], A["candidate"], A["shown"]
    names = [g for g, _ in R.groups]
    rows = []
    for g, name in enumerate(names):
        sg = shown[g]
        rows.append(
            {
                "reason": name,
                "terms": list(R.groups[g][1]),
                "candidate": float(cand[g].mean()),
                "shown_d1": float(A["shown_d1"][g].mean()),
                "shown": float(sg.mean()),
                "removed_by_s4": float((A["shown_d1"][g] & ~sg).mean()),
                "shown_twoway": float(A["shown_twoway"][g].mean()),
                "mean_shown": _num(val[g][sg].mean()) if sg.any() else None,
                "odds_mean": _num(np.exp(val[g][sg]).mean()) if sg.any() else None,
                "odds_of_mean": _num(np.exp(val[g][sg].mean())) if sg.any() else None,
                "mean_abs_candidate": (
                    _num(np.abs(val[g][cand[g]]).mean()) if cand[g].any() else None
                ),
            }
        )
    k = shown.sum(axis=0)
    # top shown reason set to its background: does the track leave?
    v_sh = np.where(shown, val, -np.inf)
    g_top = np.argmax(v_sh, axis=0)
    has = shown.any(axis=0)
    s = P["top_score"][:, 0, :REASON_K]
    s11 = P["top_score"][:, 0, REASON_K][:, None]
    id11 = P["top_id"][:, 0, REASON_K][:, None]
    ids = P["top_id"][:, 0, :REASON_K]
    s_new = s - np.take_along_axis(val, g_top[None], axis=0)[0]
    leaves = (s_new < s11) | ((s_new == s11) & (ids > id11))
    leave = {
        "tracks": int(has.sum()),
        "no_shown_reason": int((~has).sum()),
        "share": _num(leaves[has].mean()) if has.any() else None,
        "by_reason": {},
    }
    for g, name in enumerate(names):
        m = has & (g_top == g)
        leave["by_reason"][name] = {
            "tracks": int(m.sum()),
            "share": _num(leaves[m].mean()) if m.any() else None,
        }
    # template direction
    direction = {}
    for g, (name, terms_g) in enumerate(R.groups):
        js = R.gix[g]
        if all(R.terms[j].monotone for j in js):
            cls = local_slope(R, T0, js[0])[shown[g]]
            if np.any(cls == -1):
                raise ValueError(
                    f"a shown {name} reason sits on a falling "
                    f"part of a bounded shape"
                )
            direction[name] = {
                "shown": int(shown[g].sum()),
                "rising": _num(np.mean(cls == 1)) if cls.size else None,
                "flat": _num(np.mean(cls == 0)) if cls.size else None,
            }
    if "artist" in names:
        g = names.index("artist")
        jx = {t.name: j for j, t in enumerate(R.terms)}
        if "artist_affinity" in jx:
            ja = jx["artist_affinity"]
            phi_a = R.contributions(T0)[ja]
            sg = shown[g]
            cls = local_slope(R, T0, ja)
            on = sg & (cls != -2)
            direction["artist"] = {
                "shown": int(sg.sum()),
                "affinity_part_negative": (
                    _num(np.mean(phi_a[sg] < 0)) if sg.any() else None
                ),
                "affinity_on_shape": int(on.sum()),
                "affinity_slope_falling": (
                    _num(np.mean(cls[on] == -1)) if on.any() else None
                ),
            }
    if np.any(val[shown] <= 0):
        raise ValueError("a shown reason is not positive")
    return {
        "tracks": int(cand.shape[1] * cand.shape[2]),
        "listeners": int(cand.shape[1]),
        "rows": rows,
        "n_shown": {str(j): float(np.mean(k == j)) for j in range(len(names) + 1)},
        "s4": {"members": len(G["members"]), "need": int(need)},
        "d1": {
            "replicates": int(len(G["boot"]["listener"])),
            "twoway": int(len(G["boot"]["twoway"])),
        },
        "leave_top10": leave,
        "template_direction": direction,
    }


# Arms


def popularity_ranks(pop, users, picks, u_indptr, u_indices):
    """P5's route (evaluation.global_ranks and user_ranks)."""
    g = ev.global_ranks(np.asarray(pop))
    train = {
        int(u): np.asarray(u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])])
        for u in users
    }
    return np.stack(
        [ev.user_ranks(g, picks[:, p], users, train) for p in range(picks.shape[1])],
        axis=1,
    )


def check_knn(t_mats, lists, u_arrays, users, picks, pop, ranks, n_slots, n_check):
    """itemknn_select.verify's second route on n_check listeners: scores
    by brute_scores, ranks by a full sort and within the second route's
    range. Returns the largest relative score gap."""
    worst = 0.0
    for k in iks.check_users(len(users), n_check, CHECK_SEED):
        items, weights = knn.history(*u_arrays, int(users[k]))
        fast = knn.user_scores(t_mats, items, weights)
        slow = knn.brute_scores(lists, items, weights, n_slots)
        for name in ("same_artist", "both"):
            scale = max(1.0, float(np.max(np.abs(slow[name]))))
            gap = float(np.max(np.abs(fast[name] - slow[name]))) / scale
            worst = max(worst, gap)
            if gap > iks.SCORE_RTOL:
                raise ValueError(
                    f"ItemKNN, listener {int(users[k])}: the "
                    f"{name} scores differ from the second route"
                )
            s_fast = knn.candidate_scores(fast[name], items)
            s_slow = knn.candidate_scores(slow[name], items)
            arm = "knn_same" if name == "same_artist" else "knn_both"
            for j, p in enumerate(picks[k]):
                got = int(ranks[arm][k, j])
                want = knn.sorted_rank(s_fast, int(p), "id")
                tol = iks.SCORE_RTOL * max(1.0, abs(float(s_slow[int(p)])))
                lo, hi = knn.rank_bounds(s_slow, int(p), tol)
                if got != want or not lo <= got <= hi:
                    raise ValueError(
                        f"ItemKNN, listener {int(users[k])}, "
                        f"{name}: rank {got}, a full sort gives "
                        f"{want}, the second route {lo} to {hi}"
                    )
    return worst


def cached(path, key, fn):
    """fn()'s dict of arrays, saved at path and keyed by key."""
    meta = path.with_suffix(".json")
    k = s13._sha(key)
    if path.exists() and load_json(meta) == {"k": k}:
        z = np.load(path)
        return {name: np.asarray(z[name]) for name in z.files}, True
    out = fn()
    gr._atomic_savez(path, **out)
    atomic_json_write({"k": k}, meta)
    return out, False


def other_arms(io, base, frame, pop, n_slots, u_arrays, work, say, workers=None):
    """The eval and selection ranks of ItemKNN, ALS and popularity, with
    their checks. Returns (eval ranks, selection ranks, check record)."""
    users, picks = frame["user"], frame["picks"]
    s_users, s_picks = base["sel_users"], base["sel_picks"]
    u_indptr, u_indices, u_data = u_arrays
    ranks, sel, chk = {}, {}, {}
    ranks["popularity"] = popularity_ranks(pop, users, picks, u_indptr, u_indices)
    sel["popularity"] = popularity_ranks(pop, s_users, s_picks, u_indptr, u_indices)
    st_u, st_r = io.stored_ranks("p5")
    if not (np.array_equal(st_u, users) and np.array_equal(st_r, ranks["popularity"])):
        raise ValueError("P5's stored eval ranks differ from the " "recomputation")
    chk["popularity_equals_p5"] = True
    rows = TrainRows(u_indptr, u_indices)
    for arm, which in (("als_d", "d"), ("als_p6", "p6")):
        X, Y, fpr = io.als(which)
        key = {
            "version": INPUTS_VERSION,
            "arm": arm,
            "factors": fpr,
            "users": _array_key(users, picks),
            "sel": _array_key(s_users, s_picks),
        }

        def compute():
            say(
                f"{WORK_NAME}: ranking with {ARM_LABELS[arm]} "
                f"({X.shape[1]} factors)"
            )
            return {
                "eval": als.rank_items(
                    X,
                    Y,
                    rows,
                    users,
                    picks,
                    log=say,
                    label=f"{WORK_NAME}: {arm}, eval picks",
                ),
                "sel": als.rank_items(
                    X,
                    Y,
                    rows,
                    s_users,
                    s_picks,
                    log=say,
                    label=f"{WORK_NAME}: {arm}, selection picks",
                ),
            }

        out, hit = cached(work / f"ranks_{arm}.npz", key, compute)
        if hit:
            say(f"{WORK_NAME}: the {arm} ranks are saved")
        ranks[arm], sel[arm] = out["eval"], out["sel"]
        del X, Y
        ref = io.als_reference(which)
        if ref is not None:
            gap = abs(float(ga.per_listener_scores(sel[arm]).mean()) - ref)
            if gap > 1e-12:
                raise ValueError(
                    f"{arm}: the selection score from the "
                    f"recomputed ranks differs from the stored "
                    f"one by {gap:.1e}"
                )
            chk[f"{arm}_selection_gap"] = gap
    st_u, st_r = io.stored_ranks("p6")
    if not (np.array_equal(st_u, users) and np.array_equal(st_r, ranks["als_p6"])):
        raise ValueError("P6's stored eval ranks differ from the " "recomputation")
    chk["als_p6_equals_p6"] = True
    lists = io.lists(n_slots)
    t_mats = knn.transposed(lists, n_slots)
    key = {
        "version": INPUTS_VERSION,
        "arm": "itemknn",
        "users": _array_key(users, picks),
        "pop": _array_key(pop),
        "code": file_sha256(Path(knn.__file__)),
        "lists": [_array_key(*lists[k]) for k in knn.KINDS],
    }

    def compute_knn():
        say(f"{WORK_NAME}: ranking with ItemKNN")
        r, positive = knn.rank_all(
            t_mats,
            u_arrays,
            users,
            picks,
            pop,
            workers=workers,
            log=say,
            label=f"{WORK_NAME}: ItemKNN, eval picks",
        )
        return {
            "same": r[("same_artist", "id")],
            "both": r[("both", "id")],
            "positive": positive,
        }

    out, hit = cached(work / "ranks_itemknn.npz", key, compute_knn)
    if hit:
        say(f"{WORK_NAME}: the ItemKNN ranks are saved")
    ranks["knn_same"], ranks["knn_both"] = out["same"], out["both"]
    chk["itemknn_score_gap"] = check_knn(
        t_mats, lists, u_arrays, users, picks, pop, ranks, n_slots, CHECK_USERS
    )
    chk["itemknn_listeners_checked"] = int(min(CHECK_USERS, len(users)))
    _, k_ranks, _, k_users = iks.load_outputs(Path(io.d))
    if not np.array_equal(k_users, s_users):
        raise ValueError("ItemKNN's stored selection listeners differ from " "P13's")
    sel["knn_same"] = k_ranks[("same_artist", "id")]
    sel["knn_both"] = k_ranks[("both", "id")]
    return ranks, sel, chk


# Run


def provenance(model_dir):
    """The SHA-256 of the input files and the markers behind the large
    ones."""
    d = Path(model_dir)
    out = cold.provenance(d)
    for name in (
        "gam_refit.json",
        "gam_refit.npz",
        "p14_cold_gam.json",
        "p14_cold_gam.npz",
        "itemknn_selection.json",
        "itemknn_selection_ranks.npz",
        "als_d/best.json",
        "eval_baseline_ranks.npz",
        "eval_als_ranks.npz",
        "eval_users.parquet",
        "splits.parquet",
        "item_covariates.parquet",
        "stats_item_train.parquet",
        "train_manifest.json",
        "id_spaces.json",
        "csr_meta.json",
        "p13d_model_set.npz",
    ):
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    for stage in (
        "p4_features",
        "p5_eval_baseline",
        "p6_als",
        "p9b_colisten",
        "p25_train",
    ):
        try:
            out[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            out[f"marker:{stage}"] = None
    return out


def run(
    st, io, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None, workers=None
):
    """P14 (module docstring). st: a StageP13b2 whose loaders give the
    item side, the CSR and the slots. io: an Inputs. Returns the output
    doc."""
    prev = gr.WORK_NAME
    gr.WORK_NAME = WORK_NAME
    try:
        return _run(st, io, Path(model_dir), Path(work), say, n_verify, inputs, workers)
    finally:
        gr.WORK_NAME = prev
        _E.clear()


def labels_of(frame, cov, P, terms, u_indptr):
    picks = frame["picks"]
    j_same = [t.name for t in terms].index(SAME)
    return {
        "pop_centile": np.asarray(cov[0])[picks],
        "train_pairs": np.asarray(cov[1])[picks],
        "history": (
            np.asarray(u_indptr[frame["user"] + 1], np.int64)
            - np.asarray(u_indptr[frame["user"]], np.int64)
        ),
        "anchor": anchor_of(frame),
        "neighbour": P["pick_has"][:, :, j_same]
        & (np.nan_to_num(P["pick_val"][:, :, j_same]) > 0.0),
    }


def _run(st, io, d, work, say, n_verify, inputs, workers):
    t0 = time.time()
    work.mkdir(parents=True, exist_ok=True)
    base = st._base_context()
    n_slots, id_min = st._id_slots()
    G = load_models(d, base)
    terms = G["terms"]
    if SAME not in [t.name for t in terms]:
        raise ValueError(f"the final model has no {SAME} term")
    say(
        f"{WORK_NAME}: final model {G['rec']['fit_name']} "
        f"({len(terms)} terms), {len(G['boot']['listener'])} listener "
        f"replicates, S3 set of {len(G['members'])}, artist terms only "
        f"({', '.join(G['artist']['kept'])}), cold-track GAM "
        f"({len(G['cold']['kept'])} terms). Saved work: {work}"
    )
    frame, cov, pop, C = duck_inputs(io, work, n_slots, id_min, say)
    users, picks = frame["user"], frame["picks"]
    u_arrays = st._csr()
    u_indptr = u_arrays[0]
    if int(users.max()) >= len(u_indptr) - 1 or int(users.min()) < 1:
        raise ValueError("an eval listener lies outside the user CSR")
    own_pairs = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
        u_indptr[users], dtype=np.int64
    )
    if int(C["total_n"]) != int(u_indptr[-1]):
        raise ValueError(
            f"the train files hold {C['total_n']:,} pairs, the "
            f"CSR {int(u_indptr[-1]):,}"
        )
    shares = leakage_shares(anchor_of(frame), C)
    check_leakage(C, shares, own_pairs)
    say(
        f"{WORK_NAME}: {len(users):,} eval listeners. Leakage counts checked "
        f"against the CSR and P2's split"
    )
    ranks, sel, chk_arms = other_arms(
        io, base, frame, pop, n_slots, u_arrays, work, say, workers
    )
    groups = [
        (name, [[t.name for t in terms].index(n) for n in g])
        for name, g in pc.reason_groups(G["M"]["kept"])
    ]
    key = {
        "version": PASS_VERSION,
        "p13a_params": s13._params(),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, gam, s13)
        },
    }
    t1 = time.time()
    P = eval_pass(st, terms, G["tables"], groups, users, picks, work, key, say, workers)
    pass_secs = time.time() - t1
    dense = verify_dense(st, terms, G["tables"], users, picks, P, say, n_verify)
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    for c, arm in enumerate(GAM_ARMS):
        ranks[arm] = P["ranks"][:, c, :]
    sel.update(G["sel"])
    say(
        f"{WORK_NAME}: reasons on the top {REASON_K} of "
        f"{len(users):,} listeners: D1 over "
        f"{len(G['boot']['listener'])} + {len(G['boot']['twoway'])} "
        f"replicates, S4 over {len(G['members'])} members"
    )
    R = TopReasons(
        terms, G["M"]["kept"], P["top_slot"], P["top_w"], P["bsum"], P["n_cand"]
    )
    gaps = check_reasons(R, G, P)
    A, need = reason_arrays(R, G, terms)
    L = labels_of(frame, cov, P, terms, u_indptr)
    say(
        f"{WORK_NAME}: metrics, tests ({len(CONFIRMATORY)} confirmatory, "
        f"{len(EXPLORATORY)} exploratory, {N_PERM:,} permutations each) and "
        f"slices"
    )
    out = summarize_scores(ranks, sel, L)
    out["ties"] = tie_summary(ranks, P["span_lo"], P["span_hi"], P["top_score"])
    out["reasons"] = summarize_reasons(R, A, need, P, G)
    out["leakage"] = summarize_leakage(shares, L["anchor"])
    out.update(
        _norm(
            {
                "version": VERSION,
                "declared": DECLARED,
                "protocol": PROTOCOL,
                "data": {
                    "listeners": int(len(users)),
                    "dropped_without_picks": int(frame["dropped"]),
                    "selection_listeners": int(len(base["sel_users"])),
                    "n_slots": int(n_slots),
                },
                "models": {
                    "final": {k: G["rec"][k] for k in ("fit_name", "tag", "changed")},
                    "final_kept": list(G["M"]["kept"]),
                    "bounded": list(G["bounded"]),
                    "artist": G["artist"],
                    "cold": G["cold"],
                    "members": list(G["members"]),
                },
                "check": dict(
                    chk_arms,
                    dense=dense,
                    reasons=gaps,
                    frame_equals_p5=True,
                    leakage_counts=True,
                    scores_equal_reference=True,
                    slots_checked=int(P["n_checked"].sum()),
                    probe_equal=True,
                ),
                "inputs": dict(inputs or {}),
                "seconds": {"pass_this_run": pass_secs},
            }
        )
    )
    arrays = {
        "user_id": users,
        "picks": picks,
        "anchor": L["anchor"],
        "decile": frame["decile"],
        "sel_user_id": base["sel_users"],
        "pop_centile": L["pop_centile"],
        "train_pairs": L["train_pairs"],
        "history": L["history"],
        "neighbour": L["neighbour"],
        "tables": G["tables"],
    }
    arrays.update({f"rank_{a}": np.asarray(ranks[a], dtype=np.int64) for a in ARMS})
    arrays.update({f"sel_rank_{a}": np.asarray(sel[a], dtype=np.int64) for a in sel})
    arrays.update({f"leak_{k}": v for k, v in shares.items()})
    arrays.update({k: P[k] for k in P if k != "ranks"})
    arrays.update(A)
    gr._atomic_savez(d / OUT_FILES[1], **arrays)
    atomic_json_write(_clean(out), d / OUT_FILES[0])
    say(
        f"{WORK_NAME}: reading the outputs back (the reasons and every test "
        f"again, about a minute)"
    )
    check_outputs(d, base, G)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(_clean(out), d / OUT_FILES[0])
    write_report(_norm(out), d / "reports" / REPORT_NAME)
    t = out["tests"]["confirmatory"]
    say(
        f"{WORK_NAME}: GAM {out['arms']['gam']['ndcg_at_100']:.5f}. "
        + ", ".join(
            f"{r['id']} {r['arm']}: {r['decision']} (adjusted p "
            f"{r['p_adjusted']:.2g})"
            for r in t
        )
        + f". Done in {_clock(time.time() - t0)}"
    )
    return out


def check_outputs(d, base, G):
    """Read the outputs back: the reason arrays follow from the stored top
    lists and the models' tables, and every result from the stored ranks,
    labels, shares and reason arrays."""
    d = Path(d)
    doc = load_json(d / OUT_FILES[0])
    if doc is None:
        raise ValueError(f"{OUT_FILES[0]} is missing")
    z = gkd.load_npz(d / OUT_FILES[1])
    if not np.array_equal(z["tables"], G["tables"]):
        raise ValueError("the stored tables differ from the models' tables")
    if not np.array_equal(z["sel_user_id"], base["sel_users"]):
        raise ValueError("the stored selection listeners differ from P13's")
    R = TopReasons(
        G["terms"], G["M"]["kept"], z["top_slot"], z["top_w"], z["bsum"], z["n_cand"]
    )
    P = {k: z[k] for k in ("top_score", "top_id", "score_sum", "n_cand")}
    gaps = check_reasons(R, G, P)
    A, need = reason_arrays(R, G, G["terms"])
    for k, v in A.items():
        if not np.array_equal(v, z[k]):
            raise ValueError(
                f"the stored reason array {k} does not follow "
                f"from the stored top lists and the tables"
            )
    ranks = {a: z[f"rank_{a}"] for a in ARMS}
    sel = {a: z[f"sel_rank_{a}"] for a in ARMS if f"sel_rank_{a}" in z}
    L = {
        k: z[k]
        for k in ("pop_centile", "train_pairs", "history", "anchor", "neighbour")
    }
    again = summarize_scores(ranks, sel, L)
    again["ties"] = tie_summary(ranks, z["span_lo"], z["span_hi"], z["top_score"])
    again["reasons"] = summarize_reasons(R, A, need, P, G)
    shares = {k[len("leak_") :]: z[k] for k in z if k.startswith("leak_")}
    again["leakage"] = summarize_leakage(shares, z["anchor"])
    again = _norm(again)
    for k in (
        "arms",
        "tests",
        "slices",
        "slice_cuts",
        "drop",
        "ties",
        "reasons",
        "leakage",
    ):
        if again[k] != doc[k]:
            raise ValueError(
                f"the stored results '{k}' do not follow from " f"the stored arrays"
            )
    return gaps


# Report


def _f(x, fmt=".5f"):
    return "" if x is None else format(x, fmt)


def _iv(dd, fmt="+.5f"):
    if dd is None or dd.get("mean") is None:
        return "n/a"
    if dd.get("lo") is None:
        return format(dd["mean"], fmt)
    return (
        f"{format(dd['mean'], fmt)} ({format(dd['lo'], fmt)} to "
        f"{format(dd['hi'], fmt)})"
    )


def _p(x):
    return "" if x is None else (f"{x:.2e}" if x < 1e-3 else f"{x:.4f}")


def write_report(out, path):
    a, t = out["arms"], out["tests"]
    L = [
        "# P14: the evaluation on the eval listeners",
        "",
        f"- Plan: the docstring of p14.py, " f"declared {out['declared']}.",
        f"- Listeners: {out['data']['listeners']:,} eval listeners with "
        f"both test picks ({out['data']['dropped_without_picks']} without). "
        f"Every rank is exact over the full catalog "
        f"({out['data']['n_slots']:,} slots), P5's candidate and tie rules.",
        f"- Final model: {out['models']['final']['fit_name']}. Terms: "
        f"{', '.join(out['models']['final_kept'])}. Bounded: "
        f"{', '.join(out['models']['bounded']) or 'none'}. S3 set: "
        f"{len(out['models']['members'])} members.",
        f"- Checks: P5's and P6's stored eval ranks equal the "
        f"recomputation, the ItemKNN second route on "
        f"{out['check']['itemknn_listeners_checked']} listeners (largest "
        f"relative gap {out['check']['itemknn_score_gap']:.1e}), "
        f"{out['check']['slots_checked']:,} sampled slot scores equal the "
        f"dense reference exactly, "
        f"{out['check']['dense']['listeners']} listeners equal it over the "
        f"full catalog ({out['check']['dense']['tied_picks']} tied picks, "
        f"{out['check']['dense']['near_ties']} with near ties, largest "
        f"mean gap {out['check']['dense']['largest_mean_gap']:.1e}), "
        f"contributions sum to the margin over the mean candidate (largest "
        f"gap {out['check']['reasons']['sum']:.1e}), and the results read "
        f"back.",
        "",
        "## Accuracy",
        "",
        "| arm | NDCG@100 | NDCG@10 | R@10 | R@100 | MRR | median rank, "
        "test_last | median rank, test_first |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        b = a[arm]
        L.append(
            f"| {ARM_LABELS[arm]} | {b['ndcg_at_100']:.5f} | "
            f"{b['ndcg_at_10']:.5f} | {b['recall_at_10']:.4f} | "
            f"{b['recall_at_100']:.4f} | {b['mrr']:.5f} | "
            f"{b['types']['test_last']['median_rank']:,.0f} | "
            f"{b['types']['test_first']['median_rank']:,.0f} |"
        )
    L += [
        "",
        "| arm | test_last NDCG@100 | test_last R@100 | test_first "
        "NDCG@100 | test_first R@100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        ty = a[arm]["types"]
        L.append(
            f"| {ARM_LABELS[arm]} | "
            f"{ty['test_last']['ndcg_at_100']:.5f} | "
            f"{ty['test_last']['recall_at_100']:.4f} | "
            f"{ty['test_first']['ndcg_at_100']:.5f} | "
            f"{ty['test_first']['recall_at_100']:.4f} |"
        )
    L += [
        "",
        "### Ties of the GAM arms (descriptive)",
        "",
        "P5 ranks equal scores by item id. A tie span runs from the best "
        "to the worst rank among the candidates with the pick's exact "
        "score. Size: the candidates in the tie, the pick included. "
        "Boundary: the share of listeners whose 10th and 11th tracks "
        "tie.",
        "",
        "| arm | picks in a tie | median size | largest | NDCG@100, best "
        "end | NDCG@100, by id | NDCG@100, worst end | R@100, best end | "
        "R@100, worst end | boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in GAM_ARMS:
        r = out["ties"][arm]
        L.append(
            f"| {ARM_LABELS[arm]} | {r['tied']:.1%} | "
            f"{_f(r['size_median'], ',.0f') or 'n/a'} | "
            f"{r['size_max']:,} | {r['best']['ndcg_at_100']:.5f} | "
            f"{r['p5']['ndcg_at_100']:.5f} | "
            f"{r['worst']['ndcg_at_100']:.5f} | "
            f"{r['best']['recall_at_100']:.4f} | "
            f"{r['worst']['recall_at_100']:.4f} | "
            f"{r['boundary_tie']:.1%} |"
        )
    L += [
        "",
        "## Confirmatory tests (Holm, alpha 0.05)",
        "",
        f"SESOI: 2 percent of the GAM's NDCG@100 = {t['sesoi']:.5f}. "
        f"Differences are GAM minus the arm, per-listener NDCG@100, with "
        f"95 percent intervals. C1 tests mean(1.02 g - k) > 0.",
        "",
        "| test | arm | difference (95% interval) | SESOI label | t | p | "
        "Holm p | permutation p | Holm p, permutation | decision |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in t["confirmatory"]:
        flag = "" if r["perm_agrees"] else " (permutation differs)"
        L.append(
            f"| {r['id']} | {ARM_LABELS[r['arm']]} | "
            f"{_iv(r['difference'])} | {r['label']} | "
            f"{_f(r['statistic']['t'], '.2f')} | {_p(r['p'])} | "
            f"{_p(r['p_adjusted'])} | {_p(r['p_perm'])} | "
            f"{_p(r['p_perm_adjusted'])} | {r['decision']}{flag} |"
        )
    L += [
        "",
        "## Exploratory tests (Benjamini-Yekutieli, q 0.05)",
        "",
        "The SESOI of each test is 2 percent of the GAM's score on its " "metric.",
        "",
        "| test | metric | arm | difference (95% interval) | SESOI label "
        "| p | BY p | permutation BY p | decision |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in t["exploratory"]:
        flag = "" if r["perm_agrees"] else " (permutation differs)"
        L.append(
            f"| {r['id']} | {r['metric']} | {ARM_LABELS[r['arm']]} | "
            f"{_iv(r['difference'])} | {r['label']} | {_p(r['p'])} | "
            f"{_p(r['p_adjusted'])} | {_p(r['p_perm_adjusted'])} | "
            f"{r['decision']}{flag} |"
        )
    L += [
        "",
        "Tests whose t-test and permutation decisions differ: "
        + (", ".join(t["disagreements"]) or "none")
        + ".",
        "",
        "## Drop from selection to eval",
        "",
        "NDCG@100 on the val picks of the selection listeners and on the "
        "test picks of the eval listeners (two samples, independent "
        "means). The selection ranks of the three GAM arms come from "
        "catalog.CatalogScorer, which orders exact ties by rounding and "
        "not by item id. The tie spans above show how far an order of the "
        "ties can move NDCG@100.",
        "",
        "| arm | selection | eval | drop (95% interval) | relative |",
        "|---|---:|---:|---|---:|",
    ]
    for arm in [x for x in ARMS if x in out["drop"]]:
        r = out["drop"][arm]
        dd = {"mean": r["drop"], "lo": r["lo"], "hi": r["hi"]}
        L.append(
            f"| {ARM_LABELS[arm]} | {r['selection']:.5f} | "
            f"{r['eval']:.5f} | {_iv(dd)} | "
            f"{_f(r['relative'], '+.1%')} |"
        )
    L += [
        "",
        "## Slices (descriptive)",
        "",
        f"Per pick, 95 percent intervals for listener clusters. History "
        f"tertiles at {out['slice_cuts']['history_tertiles'][0]:,.0f} and "
        f"{out['slice_cuts']['history_tertiles'][1]:,.0f} train pairs.",
    ]
    for name in SLICE_ORDER:
        rows = out["slices"][name]
        L += [
            "",
            f"### {name}",
            "",
            "| level | picks | " + " | ".join(ARM_LABELS[x] for x in ARMS) + " |",
            "|---|---:|" + "---:|" * len(ARMS),
        ]
        for row in rows:
            L.append(
                f"| {row['level']} | {row['picks']:,} | "
                + " | ".join(_f(row["arms"][x]["ndcg_at_100"]["mean"]) for x in ARMS)
                + " |"
            )
        L += [
            "",
            "GAM minus each arm, NDCG@100:",
            "",
            "| level | " + " | ".join(ARM_LABELS[x] for x in ARMS[1:]) + " |",
            "|---|" + "---|" * (len(ARMS) - 1),
        ]
        for row in rows:
            L.append(
                f"| {row['level']} | "
                + " | ".join(_iv(row["gam_minus"][x]) for x in ARMS[1:])
                + " |"
            )
    nb = out["slices"]["same-artist neighbour"]
    L += [
        "",
        "Cold-track GAM minus the GAM, NDCG@100: "
        + ", ".join(f"{row['level']} {_iv(row['cold_minus_gam'])}" for row in nb)
        + ".",
    ]
    rs = out["reasons"]
    L += [
        "",
        "## Reasons on the GAM's top 10",
        "",
        f"{rs['tracks']:,} tracks of {rs['listeners']:,} listeners. "
        f"Background: each listener's candidate set. D1 over "
        f"{rs['d1']['replicates']} listener replicates (sensitivity: "
        f"{rs['d1']['twoway']} twoway), S4 over {rs['s4']['members']} "
        f"members (top 3 in at least {rs['s4']['need']}).",
        "",
        "| reason | candidate | shown by D1 | shown | removed by S4 | "
        "shown, twoway | mean value when shown | odds factor of the mean | "
        "mean odds factor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rs["rows"]:
        L.append(
            f"| {r['reason']} | {r['candidate']:.1%} | "
            f"{r['shown_d1']:.1%} | {r['shown']:.1%} | "
            f"{r['removed_by_s4']:.1%} | {r['shown_twoway']:.1%} | "
            f"{_f(r['mean_shown'], '+.3f')} | "
            f"{_f(r['odds_of_mean'], '.2f')} | "
            f"{_f(r['odds_mean'], '.2f')} |"
        )
    L += [
        "",
        "Reasons shown per track: "
        + ", ".join(f"{k}: {v:.1%}" for k, v in rs["n_shown"].items())
        + ".",
    ]
    lv = rs["leave_top10"]
    L += [
        "",
        f"Leave the top 10 when the top shown reason falls to its "
        f"background value: {_f(lv['share'], '.1%') or 'n/a'} of "
        f"{lv['tracks']:,} tracks ({lv['no_shown_reason']:,} tracks show "
        f"no reason). By the top reason: "
        + ", ".join(
            f"{k} {_f(lv['by_reason'][k]['share'], '.1%') or 'n/a'}"
            f" of {lv['by_reason'][k]['tracks']:,}"
            for k in [r["reason"] for r in rs["rows"]]
        )
        + ".",
    ]
    td = rs["template_direction"]
    for name in [r_["reason"] for r_ in rs["rows"] if r_["reason"] in td]:
        r = td[name]
        if r["shown"] == 0:
            L += ["", f"Template direction, {name}: no reason shown."]
            continue
        if name == "artist":
            L.append("")
            L.append(
                f"Artist reason: of {r['shown']:,} shown, "
                f"{_f(r['affinity_part_negative'], '.1%')} have a "
                f"negative artist_affinity part. Of "
                f"{r['affinity_on_shape']:,} with artist_affinity on "
                f"its shape, {_f(r['affinity_slope_falling'], '.1%')} "
                f"sit on a falling part of it."
            )
        else:
            L.append("")
            L.append(
                f"Template direction, {name}: {r['shown']:,} shown, "
                f"{_f(r['rising'], '.1%')} on a rising part and "
                f"{_f(r['flat'], '.1%')} on a flat part of the bounded "
                f"shape, none on a falling part."
            )
    lk = out["leakage"]
    L += [
        "",
        "## Leakage (descriptive)",
        "",
        "Anchor: last_ts of test_last, first_ts of test_first. Global: "
        "the share of all training pairs (listens) that begin after the "
        "anchor (lower bound) or end after it (upper bound). Own: the "
        "listener's own training pairs. Values: mean, then the 5th, 50th "
        "and 95th percentiles.",
        "",
        "| pick type | anchors in 2013 | global pairs, lower | global "
        "pairs, upper | global listens, lower | global listens, upper | "
        "own pairs, upper | own listens, upper |",
        "|---|---:|---|---|---|---|---|---|",
    ]

    def qs(x):
        return (
            f"{x['mean']:.3f} ({x['q'][0]:.3f}, {x['q'][2]:.3f}, " f"{x['q'][4]:.3f})"
        )

    for name in PICK_TYPES:
        r = lk["types"][name]
        L.append(
            f"| {name} | {r['share_2013']:.1%} | "
            f"{qs(r['global_lo_pairs'])} | {qs(r['global_hi_pairs'])} | "
            f"{qs(r['global_lo_listens'])} | "
            f"{qs(r['global_hi_listens'])} | {qs(r['own_hi_pairs'])} | "
            f"{qs(r['own_hi_listens'])} |"
        )
    L += [
        "",
        "Own lower bounds are 0 for both pick types and own upper "
        "bounds 0 for test_last, as P2's split makes them (checked).",
    ]
    atomic_text_write("\n".join(L).replace(";", ",") + "\n", path)


# Runner stage


class StageP14(ScriptStage):
    """P14 as a runner stage."""

    name = "p14_eval"
    upstream = [
        "p13c_final",
        "p14_cold_gam",
        "gam_refit",
        "itemknn_select",
        "als_phase_d",
        "p5_eval_baseline",
        "p6_als",
        "p9b_colisten",
    ]

    def __init__(self, model_dir=None, scratch_dir=None):
        super().__init__(model_dir, scratch_dir)
        self.n_verify = VERIFY_USERS
        self.workers = None

    def _stage(self):
        return s13b2.StageP13b2()

    def _io(self):
        return Inputs(self.model_dir, self.scratch_dir / WORK_NAME)

    def outputs(self):
        d = self.model_dir
        return {
            "results": d / OUT_FILES[0],
            "arrays": d / OUT_FILES[1],
            "report": d / "reports" / REPORT_NAME,
        }

    def provenance(self):
        return provenance(self.model_dir)

    def recorded_inputs(self):
        doc = load_json(self.outputs()["results"])
        return None if doc is None else doc.get("inputs")

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_NAME):
            run(
                self._stage(),
                self._io(),
                self.model_dir,
                self.scratch_dir / WORK_NAME,
                say=log.info,
                n_verify=self.n_verify,
                inputs=self.provenance(),
                workers=self.workers,
            )
        return self.output_info()

    def check(self):
        st = self._stage()
        base = st._base_context()
        check_outputs(self.model_dir, base, load_models(self.model_dir, base))

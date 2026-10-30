"""Beyond accuracy: the recommended lists themselves. How narrow, varied,
broad, concentrated and new each arm's lists are, how far down the first
track by a new artist sits, and what a cap per artist costs in accuracy.

RULES. The stage comes after the main evaluation
(pipeline/p14.py), and these rules were written before its first run on
real data. Every number is descriptive: no test and no SESOI decision.
The model stays frozen.

From P14, exactly as P14 uses them:
- listeners: P14's eval listeners with both test picks, and their
  test_last and test_first picks, as p14.npz stores them.
- candidates: every item slot except slot 0 and the listener's training
  items. The other held-out picks stay candidates.
- scorers: the GAM arms by p14.term_sums() on the concepts of each slot
  range (the arithmetic of the dense reference, so equal term values give
  equal scores), ItemKNN by itemknn.user_scores(), ALS phase D by the
  product of als.rank_items() over its chunks of ALS_CHUNK listeners
  (factors from P14's loader), popularity by P5's train listens.
- tie rule: equal scores rank by item id.

Arms: gam, artist_only, cold_gam, knn_same, knn_both, als_d (256 factors)
and popularity, with P14's labels. als_p6 stays out.

Per listener and arm, from one pass over the full catalogue:
- the top K = 200 list: item ids in rank order (score descending, then
  item id ascending), stored as int32, with their scores.
- first_new: the rank of the best-ranked candidate whose artist is new to
  the listener, exact over the full catalogue with the tie rule: 1 + the
  candidates with a higher score + the candidates with the same score and
  a smaller item id. -1 when no such candidate exists.
- the ranks of both test picks, for the checks.

Definitions:
- artists of a track: its credited artist ids under the rule of
  pipeline/artists.py, as artist affinity reads them.
- primary artist: the first credited artist id. artists.py keeps the ids
  of a track sorted, so it is the smallest. The cap and the count of
  distinct artists use it.
- known artist: at least one credited artist of the track occurs among
  the credited artists of the listener's training tracks. This is the
  condition under which artist affinity is above 0 (concepts.py), and
  same-artist co-listening needs a shared credit (P9b).
- new artist: a track with at least one credited artist and no known
  artist. A track without a credit is neither known nor new.
- popularity of a track: n_i, the training listeners who played it (the
  pairs of stats_item_train: one pair per listener and track), and P5's
  pop_centile (item_covariates). N: the training listeners
  (stats_user_train).

Metrics per arm. Each per-listener value has its mean over the listeners
with a 95 percent normal interval (the mean plus or minus 1.96 standard
errors).
1. the known-artist share of the top 10, and the share of listeners whose
   top 10 holds at least one new artist.
2. the distinct primary artists in the top 10.
3. coverage: the distinct tracks in the union of all top-10 lists, as a
   count and as a share of the catalogue's tracks (item slots 1 to n - 1),
   and the distinct primary artists in that union.
4. concentration: the Gini index of how often each catalogue track appears
   in the top-10 lists, every track included, zeros too.
5. novelty: the mean self-information of the top-10 tracks,
   -log2((n_i + 1) / (N + 1)), and their mean pop_centile. Reference row:
   the same two numbers over each listener's own training tracks.
6. overlap: the mean Jaccard index of the top-10 sets of PAIRS = 100,000
   random pairs of distinct listeners (seed PAIR_SEED, the same pairs for
   every arm). Listeners repeat across the pairs, so no interval.
7. first new artist: the 5th, 25th, 50th, 75th and 95th percentiles of
   first_new over the listeners that have a new-artist candidate (the
   inverted empirical distribution: the smallest rank that at least that
   share of these listeners reaches), and the shares of all listeners with
   first_new within 10 and within 100.
8. caps per primary artist of 1, 2, 3 and 5 (5 is the demo's default), and
   no cap. The walk: along the top-K list in rank order, keep a track while
   its primary artist has fewer kept tracks than the cap, and stop at 10
   kept. A track without a credit shares no artist with any other track
   (artists.py), so no cap removes it. Per cap:
   - NDCG@10 and R@10 on the test picks by P14's per-pick rule: the mean
     over the two picks, then over the listeners. A pick that the cap
     removes, or that falls below the 10 kept tracks, gains 0.
   - the paired difference to no cap (gam_audit.paired_summary).
   - the distinct primary artists and the known-artist share of the
     capped list.
   - the share of listeners whose capped list holds fewer than 10 tracks
     within the top K.
   With them, per arm: the share of the top-10 tracks that credit more
   than one artist. The demo's filter caps every credited artist of a
   track, and this walk only the primary one, so the two walks differ
   only on such tracks.

Checks, each one stops the run:
- for every arm and listener, the ranks of both test picks equal P14's
  stored ranks.
- with no cap, each arm's NDCG@10 and R@10 equal p14.json's values
  exactly.
- the top 20 of the three GAM arms (ids and scores) equal P14's stored
  top lists.
- known artists: in every slot range of the GAM pass, the known tracks
  equal the tracks whose artist affinity is above 0. On the GAM's top 10,
  the known flags equal P14's stored artist affinity above 0, and when P14
  stored that for every top-10 track, the GAM's known-artist share is 1.0.
- every list is free of training items and duplicates, and is sorted by
  score descending, then item id ascending.
- the popularity arm gives every listener the same order, apart from the
  removed training items: each list equals P5's global order
  (evaluation.global_ranks) without the listener's training items.
- n_i equals the user CSR's count of listeners per track and the
  covariates' train pairs, and N equals the CSR's listeners with a pair.
- the ALS product over column ranges equals als.rank_items()'s product of
  the whole chunk bit for bit, on the first chunk.
- in every slot range, the GAM scores of every CHECK_STRIDE-th slot, of
  the picks and of the slots that enter a top list equal
  catalog.reference_scores() exactly (P14's check).
- the pool gives the probe listeners the results of this process.
- VERIFY_USERS listeners: each arm's list, scores, first_new, pick ranks
  and known flags equal a full sort of a second route:
  concepts.full_catalog() with catalog.reference_scores() for the GAM
  arms and the artist-affinity concept for the known flags,
  itemknn.user_scores() (with itemknn.brute_scores() within
  itemknn_select.SCORE_RTOL), the ALS product of the whole chunk, and P5's
  scores.
- the outputs read back: every result follows from the stored arrays.

COMPUTATION.
- one pass over the listeners in blocks of BLOCK listeners, saved as they
  finish. BLOCK divides ALS_CHUNK, so a block lies inside one chunk of
  als.rank_items(). Workers run with one BLAS thread.
- per block: the ALS product of the rows of the whole chunk (the shape of
  als.rank_items()'s product) over column ranges of CHUNK slots, twice:
  first the ranges that hold a pick, for the picks' scores, then every
  range. Per listener: the GAM arms over the slot ranges (the ranges that
  hold a pick first, as in P14), ItemKNN and popularity as dense vectors.
- every arm feeds its scores, range by range in slot order, to Scan,
  which keeps the top K, counts the candidates above each pick and finds
  the best-ranked new-artist candidate with the candidates above it.
- known tracks: the tracks of the history's credited artists, from the
  credit matrix of the item side and its transpose.

Outputs: MODEL_DIR/beyond.json (the rules, every metric per arm and per
cap, the checks), MODEL_DIR/beyond.npz (per listener and arm: the top-K
lists and scores, the known flags, first_new, the pick ranks and the
per-listener metrics, and per track the primary artist, n_i and
pop_centile), MODEL_DIR/reports/beyond_report.md and
MODEL_DIR/figures/tab_beyond_arms.csv, tab_beyond_first_new.csv and
tab_beyond_caps.csv. The files hold no time and no worker count, so the
same inputs give the same bytes. Every line also goes to
MODEL_DIR/logs/beyond.log. Long steps print a line at most every 30
seconds with the share done, the time elapsed and the time left.

Resume: after Ctrl+C the same command continues: the saved inputs and the
finished blocks stay in SCRATCH_DIR/beyond_accuracy.

Expected run on the real data: P14's pass took 2 to 3 hours. This pass
scores the same three GAM tables plus four dense arms, so about 2 to 4
hours (the probe prints an estimate). The input checks (one scan of the
user CSR) and the dense checks take a few minutes.

Usage, from the repository root, after the stage p14_eval:
    python3 -m pipeline.runner --only beyond_accuracy
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only beyond_accuracy
"""

import csv
import io as iolib
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import evaluation as ev
from pipeline import gam_audit as ga
from pipeline import gam_knn_diag as gkd
from pipeline import itemknn as knn
from pipeline import itemknn_select as iks
from pipeline import p14
from pipeline import p14_cold as cold
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.script_stage import ScriptStage, log_file

VERSION = "beyond-v1"
INPUTS_VERSION = "beyond-inputs-v1"
# The key of the pass blocks: a change to Scan or to the arms' scoring
# bumps it.
PASS_VERSION = "beyond-pass-v1"
WORK_NAME = "beyond_accuracy"
DECLARED = "after the main evaluation (P14) and before the " "first run on real data"
OUT_FILES = ("beyond.json", "beyond.npz")
REPORT_NAME = "beyond_report.md"
TAB_FILES = ("tab_beyond_arms.csv", "tab_beyond_first_new.csv", "tab_beyond_caps.csv")
FIG_DIR = "figures"
LOG_NAME = "beyond.log"
COMMAND = "python3 -m pipeline.runner --only beyond_accuracy"
ARMS = ("gam", "artist_only", "cold_gam", "knn_same", "knn_both", "als_d", "popularity")
GAM_ARMS = p14.GAM_ARMS  # the rows of p14.load_models' tables
ARM_LABELS = {a: p14.ARM_LABELS[a] for a in ARMS}
K = 200  # the stored list of each listener and arm
TOP = 10  # the list that the metrics read
CAPS = (1, 2, 3, 5, None)
CUTS = (10, 100)  # first_new within these ranks
QUANTILES = (5, 25, 50, 75, 95)
PAIRS = 100_000
PAIR_SEED = 2026101501
P14_TOP = 20  # P14 stored the top 20 of each GAM table
AFFINITY = "artist_affinity"
CHUNK = p14.EVAL_CHUNK  # slots per range, as P14's pass
ALS_CHUNK = 64  # listeners per product, as als.rank_items()
BLOCK = ALS_CHUNK  # listeners per task, a divisor of ALS_CHUNK
CHECK_STRIDE = p14.CHECK_STRIDE
PROBE_USERS = 4
VERIFY_USERS = 6
VERIFY_SEED = 2026101502
PRUNE = 1 << 16  # first-new buffer entries before a prune
Z95 = p14.Z95


def protocol():
    return {
        "version": VERSION,
        "declared": DECLARED,
        "listeners": "P14's eval listeners with both test picks",
        "candidates": "every item slot except slot 0 and the listener's "
        "training items, ties by item id (P5, P14)",
        "arms": {a: ARM_LABELS[a] for a in ARMS},
        "k": K,
        "top": TOP,
        "caps": [c for c in CAPS if c is not None],
        "known": "a credited artist of the track occurs among the credited "
        "artists of the listener's training tracks",
        "primary_artist": "the first (smallest) credited artist id",
        "novelty": "-log2((n_i + 1) / (N + 1)) and pop_centile, top 10",
        "overlap": {"pairs": PAIRS, "seed": PAIR_SEED},
        "first_new": {
            "quantiles": list(QUANTILES),
            "within": list(CUTS),
            "percentile_method": "inverted_cdf",
        },
        "intervals": "mean plus or minus 1.96 standard errors over the " "listeners",
        "tests": "none, descriptive",
    }


class Inputs(p14.Inputs):
    """P14's inputs (its ALS loader, P5's popularity and covariates) and the
    training listeners per track (stats_item_train) and in all
    (stats_user_train), from DuckDB. The smoke test replaces this class."""

    def key(self):
        """What the saved inputs depend on."""
        return {
            name: file_sha256(self.d / name)
            for name in (
                "stats_item_train.parquet",
                "stats_user_train.parquet",
                "item_covariates.parquet",
            )
        }

    def item_listeners(self, n_slots):
        """n_i per item slot: the pairs of stats_item_train (one pair per
        listener and track), 0 for a slot without a row."""
        con = connect(4)
        try:
            cols = con.execute(
                f"SELECT item_id, pairs FROM read_parquet("
                f"'{self.d / 'stats_item_train.parquet'}')"
            ).fetchnumpy()
        finally:
            con.close()
        out = np.zeros(n_slots, dtype=np.int64)
        out[np.asarray(cols["item_id"], dtype=np.int64)] = np.asarray(
            cols["pairs"], dtype=np.int64
        )
        return out

    def train_listeners(self):
        """N: the listeners of stats_user_train with a training pair."""
        con = connect(4)
        try:
            return int(
                con.execute(
                    f"SELECT count(*) FROM read_parquet("
                    f"'{self.d / 'stats_user_train.parquet'}') "
                    f"WHERE pairs > 0"
                ).fetchone()[0]
            )
        finally:
            con.close()


def load_inputs(io, work, n_slots, id_min, say):
    """(pop, pop_centile, n_i, N): P5's popularity scores and covariates and
    the training listeners, saved in work and keyed by io.key()."""
    path, meta = work / "inputs.npz", work / "inputs.json"
    k = {
        "version": INPUTS_VERSION,
        "inputs": io.key(),
        "n_slots": int(n_slots),
        "id_min": int(id_min),
    }
    if path.exists() and load_json(meta) == {"k": s13._sha(k)}:
        z = np.load(path)
        say(f"{WORK_NAME}: the DuckDB inputs are saved, so no query runs")
        return (
            np.asarray(z["pop"]),
            np.asarray(z["pop_centile"]),
            np.asarray(z["n_i"]),
            np.asarray(z["cov_pairs"]),
            int(z["N"]),
        )
    say(
        f"{WORK_NAME}: reading P5's popularity and covariates and the "
        f"training listeners with DuckDB (no progress lines in this step)"
    )
    t0 = time.time()
    pop = np.asarray(io.popularity(n_slots, id_min), dtype=np.int64)
    cov = io.covariates(n_slots)
    pop_c = np.asarray(cov[0], dtype=np.int64)
    pairs = np.asarray(cov[1], dtype=np.int64)
    n_i = np.asarray(io.item_listeners(n_slots), dtype=np.int64)
    N = int(io.train_listeners())
    s13._atomic_savez(
        path, pop=pop, pop_centile=pop_c, n_i=n_i, cov_pairs=pairs, N=np.int64(N)
    )
    atomic_json_write({"k": s13._sha(k)}, meta)
    say(f"{WORK_NAME}: inputs read in {_clock(time.time() - t0)}")
    return pop, pop_c, n_i, pairs, N


def csr_listeners(u_indptr, u_indices, n_slots, say, step=1 << 24):
    """The training listeners of each item slot, counted over the user CSR
    (one entry per listener and track), and the listeners with a pair."""
    total = int(u_indptr[-1])
    out = np.zeros(n_slots, dtype=np.int64)
    prog = Progress(
        f"{WORK_NAME}: training listeners per track from the " f"user CSR",
        total,
        say,
        unit=" million pairs",
        scale=1_000_000,
    )
    for a in range(0, total, step):
        b = min(a + step, total)
        out += np.bincount(np.asarray(u_indices[a:b]), minlength=n_slots)
        prog.advance(b - a)
    prog.finish()
    users = int(np.count_nonzero(np.diff(np.asarray(u_indptr))))
    return out, users


def check_inputs(pop, pop_c, n_i, cov_pairs, N, u_indptr, u_indices, n_slots, say):
    """n_i and N against the user CSR, the covariates' train pairs against
    n_i, and the ranges of the popularity inputs (module docstring)."""
    for name, arr in (
        ("pop", pop),
        ("pop_centile", pop_c),
        ("n_i", n_i),
        ("cov_pairs", cov_pairs),
    ):
        if len(arr) != n_slots:
            raise ValueError(f"{name} has {len(arr)} slots, the catalog " f"{n_slots}")
    counts, users = csr_listeners(u_indptr, u_indices, n_slots, say)
    if not np.array_equal(counts, n_i):
        bad = int(np.flatnonzero(counts != n_i)[0])
        raise ValueError(
            f"track {bad}: {int(n_i[bad])} training listeners "
            f"in stats_item_train, {int(counts[bad])} in the "
            f"user CSR"
        )
    if not np.array_equal(cov_pairs[1:], n_i[1:]):
        raise ValueError(
            "the covariates' train pairs differ from the "
            "training listeners per track"
        )
    if users != N:
        raise ValueError(
            f"{N:,} training listeners in stats_user_train, "
            f"{users:,} in the user CSR"
        )
    if int(n_i.sum()) != int(u_indptr[-1]):
        raise ValueError(
            "the training listeners per track do not sum to " "the training pairs"
        )
    if pop_c.min() < 0 or pop_c.max() > 100:
        raise ValueError("a pop_centile lies outside 0 to 100")
    return {
        "train_pairs": int(u_indptr[-1]),
        "train_listeners": int(N),
        "tracks_with_listeners": int(np.count_nonzero(n_i)),
    }


# Artists


def primary_artists(credit):
    """The first credited artist of each track: the smallest column of its
    row of the credit matrix, -1 for a track without a credit. The columns
    are the sorted distinct artist ids (concept_data.compact), so the
    smallest column is the smallest id, the first id of the track's sorted
    row in pipeline/artists.py."""
    out = np.full(credit.shape[0], -1, dtype=np.int32)
    nz = np.flatnonzero(np.diff(credit.indptr) > 0)
    if len(nz):
        out[nz] = np.minimum.reduceat(
            np.asarray(credit.indices), np.asarray(credit.indptr[:-1])[nz]
        )
    out[0] = -1  # slot 0 is the phantom slot
    return out


def known_slots(credit, credit_t, items):
    """Sorted slots of the tracks by a known artist: the rows of credit_t
    (artist column -> tracks) of every column that a history track
    credits. concepts.build_user_state reads the same columns for artist
    affinity, and a column of the history has a share above 0, so these
    are the tracks whose artist affinity is above 0."""
    rows = credit[np.asarray(items, dtype=np.int64)]
    cols = np.unique(rows.indices)
    if not len(cols):
        return np.zeros(0, dtype=np.int32)
    return np.unique(credit_t[cols].indices).astype(np.int32)


def range_masks(known, has_credit, a, b):
    """(known, new) masks of slots a..b-1 from the sorted known slots: new
    where the track has a credit and no known artist."""
    lo, hi = np.searchsorted(known, (a, b))
    kr = np.zeros(b - a, dtype=bool)
    kr[known[lo:hi] - a] = True
    return kr, has_credit[a:b] & ~kr


class Scan:
    """The top list, the ranks of the picks and the best-ranked new-artist
    candidate of one listener and arm, from the scores of consecutive slot
    ranges fed in slot order. A score of -inf marks a non-candidate.

    - picks: P5's rule, 1 + #(score > s_p) + #(score == s_p, slot < p),
      summed over the ranges, so s_pick must be known before the first
      range.
    - top list: each range offers its best `keep` candidates (equal scores
      by slot), and a merge keeps the best `keep` of the list and the
      offer. Once the list is full, a range offers only scores above the
      list's last score: an equal score in a later range has a larger slot
      and ranks after the last entry.
    - first new: the best new-artist candidate so far (bn_s, bn_id) and a
      buffer of the candidates that rank above it. A candidate above the
      final best also ranked above the best at the time of its range (the
      best only improves), so it entered the buffer, and a prune removes
      only entries at or below the best. No new-artist candidate ranks
      above the best one, so the count of the buffer entries above the
      final best is the count of candidates above it."""

    def __init__(self, keep, picks, s_pick):
        self.keep = int(keep)
        self.picks = np.asarray(picks, dtype=np.int64)
        self.s_pick = np.asarray(s_pick, dtype=np.float64)
        self.greater = np.zeros(len(self.picks), dtype=np.int64)
        self.before = np.zeros(len(self.picks), dtype=np.int64)
        self.top_id = np.empty(0, dtype=np.int64)
        self.top_s = np.empty(0, dtype=np.float64)
        self.bn_s, self.bn_id = np.float64(-np.inf), -1
        self.buf_s, self.buf_id, self.n_buf = [], [], 0
        self.limit = PRUNE
        self.end = 0

    def feed(self, s, a, new):
        """Scores s of slots a..a+len(s)-1 (-inf: not a candidate) and the
        new-artist mask of those slots. Returns the slots that entered the
        top list."""
        n = len(s)
        if a != self.end:
            raise ValueError("Scan: the ranges must come in slot order")
        self.end = a + n
        for p in range(len(self.picks)):
            sp = self.s_pick[p]
            self.greater[p] += int(np.count_nonzero(s > sp))
            e = min(max(int(self.picks[p]) - a, 0), n)
            if e:
                self.before[p] += int(np.count_nonzero(s[:e] == sp))
        if len(self.top_id) == self.keep:
            idx = np.flatnonzero(s > self.top_s[-1])
        else:
            idx = np.flatnonzero(s > -np.inf)
        if len(idx) > self.keep:
            sub = s[idx]
            thr = np.partition(sub, len(sub) - self.keep)[len(sub) - self.keep]
            gt = idx[sub > thr]
            idx = np.concatenate((gt, idx[sub == thr][: self.keep - len(gt)]))
        if len(idx):
            ids = np.concatenate((self.top_id, a + idx))
            sc = np.concatenate((self.top_s, s[idx].astype(np.float64)))
            order = np.lexsort((ids, -sc))[: self.keep]
            self.top_id, self.top_s = ids[order], sc[order]
        if new.any():
            sn = np.where(new, s, -np.inf)
            j = int(np.argmax(sn))  # the first maximum: the smallest slot
            if sn[j] > self.bn_s:  # a tie with an earlier range keeps it
                self.bn_s, self.bn_id = np.float64(sn[j]), a + j
                self._prune()
        if self.bn_s == -np.inf:
            sel = np.flatnonzero(s > -np.inf)
        else:
            sel = np.flatnonzero(s > self.bn_s)
            if self.bn_id >= a:  # equal scores before it rank above
                sel = np.concatenate(
                    (sel, np.flatnonzero(s[: self.bn_id - a] == self.bn_s))
                )
        if len(sel):
            self.buf_s.append(s[sel].astype(np.float64))
            self.buf_id.append(a + sel.astype(np.int64))
            self.n_buf += len(sel)
            if self.n_buf > self.limit:
                self._prune()
                self.limit = max(PRUNE, 2 * self.n_buf)
        return a + idx

    def _above(self, sc, ids):
        return (sc > self.bn_s) | ((sc == self.bn_s) & (ids < self.bn_id))

    def _prune(self):
        if self.bn_s == -np.inf or not self.buf_s:
            return
        sc, ids = np.concatenate(self.buf_s), np.concatenate(self.buf_id)
        keep = self._above(sc, ids)
        self.buf_s, self.buf_id = [sc[keep]], [ids[keep]]
        self.n_buf = int(keep.sum())

    def result(self):
        if self.bn_s == -np.inf:
            first = -1
        else:
            self._prune()
            first = 1 + self.n_buf
        return {
            "top": self.top_id,
            "score": self.top_s,
            "first_new": int(first),
            "ranks": 1 + self.greater + self.before,
        }


def mask_out(s, excl, a):
    """s (the scores of slots a..a+len(s)-1) with -inf on the listener's
    training items (sorted) and on slot 0, in place."""
    lo, hi = np.searchsorted(excl, (a, a + len(s)))
    s[excl[lo:hi] - a] = -np.inf
    if a == 0:
        s[0] = -np.inf
    return s


# The pass over the listeners
_E = {}  # the pass context, set before a fork


def _range(c, state, pre, a, b):
    """(scores (C, b - a), concepts) of slots a..b-1: p14.term_sums on the
    concepts of the range, as P14's pass."""
    scorer, space = c["scorer"], c["space"]
    conc = {}
    for t in scorer.terms:
        if t.source not in conc:
            conc[t.source] = scorer.concept_range(state, space, pre, t.source, a, b)
    return p14.term_sums(scorer.terms, scorer.T, conc, scorer.tier[a:b]), conc


def _prepare(c, k):
    """Listener k's history, training items, picks and known tracks."""
    u = int(c["users"][k])
    items, weights = cd.history_of(c["u_indptr"], c["u_indices"], c["u_data"], u)
    excl = np.unique(items)
    return {
        "u": u,
        "picks": np.asarray(c["picks"][k], dtype=np.int64),
        "items": items,
        "weights": weights,
        "excl": excl,
        "known": known_slots(c["credit"], c["credit_t"], excl),
    }


def _dense(c, s, L):
    """Scan of one dense score vector with -inf on the non-candidates."""
    scan = Scan(K, L["picks"], s[L["picks"]])
    n = len(s)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        scan.feed(s[a:b], a, range_masks(L["known"], c["has_credit"], a, b)[1])
    return scan.result()


def _gam(c, L):
    """The three GAM tables over the slot ranges (module docstring), with
    P14's reference check and the check of the known tracks against the
    artist-affinity concept. Returns ([results], slots checked)."""
    scorer, space = c["scorer"], c["space"]
    picks, excl, u = L["picks"], L["excl"], L["u"]
    state = cn.build_user_state(L["items"], L["weights"], space)
    scorer._check_picks(picks, excl)
    pre = scorer.prepare(state, space)
    n = scorer.n
    first = list(dict.fromkeys(int(t) // CHUNK for t in picks))
    held = {r: _range(c, state, pre, r * CHUNK, min((r + 1) * CHUNK, n)) for r in first}
    s_pick = np.stack(
        [held[int(t) // CHUNK][0][:, int(t) % CHUNK] for t in picks], axis=1
    )  # (C, picks)
    scans = [Scan(K, picks, s_pick[j]) for j in range(scorer.C)]
    checked = 0
    for r in range(-(-n // CHUNK)):
        a, b = r * CHUNK, min((r + 1) * CHUNK, n)
        S, conc = held.pop(r) if r in held else _range(c, state, pre, a, b)
        kr, new = range_masks(L["known"], c["has_credit"], a, b)
        if AFFINITY in conc:
            v, h = conc[AFFINITY]
            if not np.array_equal(kr, h & (v > 0.0)):
                raise RuntimeError(
                    f"listener {u}: the known tracks differ "
                    f"from the tracks whose artist affinity "
                    f"is above 0, near slot {a}"
                )
        out = np.zeros(b - a, dtype=bool)
        lo, hi = np.searchsorted(excl, (a, b))
        out[excl[lo:hi] - a] = True
        if a == 0:
            out[0] = True
        S[:, out] = -np.inf
        took = [scans[j].feed(S[j], a, new) for j in range(scorer.C)]
        sub = np.unique(
            np.concatenate(
                [
                    np.arange(0, b - a, CHECK_STRIDE),
                    picks[(picks >= a) & (picks < b)] - a,
                ]
                + [t - a for t in took]
            )
        )
        sub = sub[~out[sub]]
        p14._reference_check(c, S, conc, sub, a, u)
        checked += len(sub)
    return [s.result() for s in scans], checked


def _knn(c, L):
    """ItemKNN's two arms: itemknn.user_scores on the sorted history, the
    non-candidates at -inf (itemknn.candidate_scores), as rank_all."""
    items, weights = knn.history(c["u_indptr"], c["u_indices"], c["u_data"], L["u"])
    sc = knn.user_scores(c["t_mats"], items, weights)
    sc.pop("other_artists")
    out = {}
    for kind, arm in (("same_artist", "knn_same"), ("both", "knn_both")):
        out[arm] = _dense(c, knn.candidate_scores(sc.pop(kind), items), L)
    return out


def _popularity(c, L):
    """P5's scores (train listens) with the non-candidates at -inf."""
    s = np.array(c["pop_f"], dtype=np.float64)
    return _dense(c, mask_out(s, L["excl"], 0), L)


def _als(c, lo, hi, Ls):
    """The ALS arm of listeners lo..hi-1, which lie in one chunk of
    ALS_CHUNK listeners of als.rank_items(). The product of the chunk's
    rows runs over column ranges: first the ranges that hold a pick, for
    the picks' scores, then every range for the scans."""
    users, n = c["users"], c["n_slots"]
    c0 = (lo // ALS_CHUNK) * ALS_CHUNK
    c1 = min(c0 + ALS_CHUNK, len(users))
    if hi > c1:
        raise ValueError("a block spans two ALS chunks")
    Xc = np.asarray(c["als_x"][users[c0:c1]], dtype=np.float32)
    Yd = c["als_y"]
    ks = range(lo, hi)
    s_pick = {k: np.empty(len(Ls[k]["picks"]), dtype=np.float32) for k in ks}
    need = sorted({int(t) // CHUNK for k in ks for t in Ls[k]["picks"]})
    for r in need:
        a, b = r * CHUNK, min((r + 1) * CHUNK, n)
        S = Xc @ Yd[a:b].T
        for k in ks:
            for p, t in enumerate(Ls[k]["picks"]):
                if a <= t < b:
                    s_pick[k][p] = S[k - c0, int(t) - a]
    scans = {k: Scan(K, Ls[k]["picks"], s_pick[k]) for k in ks}
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        S = Xc @ Yd[a:b].T
        for k in ks:
            row = mask_out(S[k - c0], Ls[k]["excl"], a)
            scans[k].feed(row, a, range_masks(Ls[k]["known"], c["has_credit"], a, b)[1])
    return {k: scans[k].result() for k in ks}


def _block(c, lo, hi):
    """Every arm of listeners lo..hi-1: per-listener arrays (module
    docstring)."""
    Ls = {k: _prepare(c, k) for k in range(lo, hi)}
    res_als = _als(c, lo, hi, Ls)
    m = hi - lo
    n_slots = c["n_slots"]
    out = {
        "n_cand": np.empty(m, dtype=np.int64),
        "n_known_cand": np.empty(m, dtype=np.int64),
        "n_new_cand": np.empty(m, dtype=np.int64),
        "checked": np.empty(m, dtype=np.int64),
    }
    for arm in ARMS:
        out[f"top_{arm}"] = np.full((m, K), -1, dtype=np.int32)
        out[f"score_{arm}"] = np.full((m, K), -np.inf)
        out[f"known_{arm}"] = np.zeros((m, K), dtype=bool)
        out[f"first_new_{arm}"] = np.empty(m, dtype=np.int64)
        out[f"rank_{arm}"] = np.empty((m, 2), dtype=np.int64)
    for j, k in enumerate(range(lo, hi)):
        L = Ls.pop(k)
        g, out["checked"][j] = _gam(c, L)
        res = dict(zip(GAM_ARMS, g))
        res.update(_knn(c, L))
        res["als_d"] = res_als.pop(k)
        res["popularity"] = _popularity(c, L)
        known, excl = L["known"], L["excl"]
        n_known = len(known) - int(np.count_nonzero(np.isin(known, excl)))
        out["n_cand"][j] = n_slots - 1 - len(excl)
        out["n_known_cand"][j] = n_known
        out["n_new_cand"][j] = (
            c["n_credited"] - int(np.count_nonzero(c["has_credit"][excl])) - n_known
        )
        for arm in ARMS:
            r = res[arm]
            t = len(r["top"])
            out[f"top_{arm}"][j, :t] = r["top"]
            out[f"score_{arm}"][j, :t] = r["score"]
            out[f"known_{arm}"][j, :t] = is_known(known, r["top"])
            out[f"first_new_{arm}"][j] = r["first_new"]
            out[f"rank_{arm}"][j] = r["ranks"]
    return out


def _task(task):
    lo, hi = task
    with threadpool_limits(1):  # many workers, one BLAS thread each
        return _block(_E, lo, hi)


def _same(a, b):
    return all(
        np.array_equal(a[k], b[k], equal_nan=(a[k].dtype.kind == "f")) for k in a
    )


def block_tasks(n):
    """The blocks of BLOCK listeners. BLOCK divides ALS_CHUNK, so no block
    spans two chunks of als.rank_items()."""
    if BLOCK < 1 or ALS_CHUNK % BLOCK:
        raise ValueError("BLOCK must divide ALS_CHUNK")
    return [(lo, min(lo + BLOCK, n)) for lo in range(0, n, BLOCK)]


def is_known(known, ids):
    """For each id of ids, whether it lies in the sorted array known."""
    ids = np.asarray(ids, dtype=np.int64)
    if not len(known):
        return np.zeros(len(ids), dtype=bool)
    pos = np.minimum(np.searchsorted(known, ids), len(known) - 1)
    return known[pos] == ids


def check_block(out, lo, p14_ranks, p14_top, p14_score):
    """The pick ranks of every arm against P14's and the GAM arms' top 20
    against P14's top lists, for the listeners of one block."""
    m = len(out["n_cand"])
    for arm in ARMS:
        want = p14_ranks[arm][lo : lo + m]
        got = out[f"rank_{arm}"]
        if not np.array_equal(got, want):
            j = int(np.flatnonzero(np.any(got != want, axis=1))[0])
            raise ValueError(
                f"{arm}, eval listener {lo + j}: the ranks of "
                f"the picks {got[j].tolist()} differ from P14's "
                f"{want[j].tolist()}"
            )
    w = min(P14_TOP, K)
    for c_, arm in enumerate(GAM_ARMS):
        ids = out[f"top_{arm}"][:, :w].astype(np.int64)
        sc = out[f"score_{arm}"][:, :w]
        if not (
            np.array_equal(ids, p14_top[lo : lo + m, c_, :w])
            and np.array_equal(sc, p14_score[lo : lo + m, c_, :w])
        ):
            raise ValueError(
                f"{arm}: the top {w} of the eval listeners "
                f"{lo} to {lo + m - 1} differ from P14's stored "
                f"top lists"
            )


def als_split_check(X, Yd, users, n_slots, say):
    """The ALS product of the first chunk of ALS_CHUNK listeners over
    column ranges with one BLAS thread (the pass) against
    als.rank_items()'s product of the whole chunk with every thread, bit
    for bit."""
    uc = users[:ALS_CHUNK]
    Xc = np.asarray(X[uc], dtype=np.float32)
    t0 = time.time()
    with threadpool_limits(int(os.cpu_count() or 1)):
        S = np.asarray(X[uc], dtype=np.float32) @ Yd.T
    with threadpool_limits(1):
        for a in range(0, n_slots, CHUNK):
            b = min(a + CHUNK, n_slots)
            if not np.array_equal(Xc @ Yd[a:b].T, S[:, a:b]):
                raise ValueError(
                    f"the ALS product over column ranges differs "
                    f"from als.rank_items()'s product near slot "
                    f"{a}, so the pass cannot reproduce P14's "
                    f"ALS scores"
                )
    say(
        f"{WORK_NAME}: the ALS product over column ranges equals "
        f"als.rank_items()'s product bit for bit on the first chunk "
        f"({len(uc)} listeners, {_clock(time.time() - t0)})"
    )
    return {"listeners": int(len(uc)), "ranges": -(-n_slots // CHUNK)}


def workers_for(st, space, users, u_indptr, u_indices, n_slots, say):
    """P13's worker setting (memory-checked for the GAM part), capped so
    that the dense arms' vectors of every worker fit in the free memory:
    six float64 vectors over the slots, the ALS product of one range and
    the masks."""
    with cold.relabel_logs(WORK_NAME, COMMAND):
        w = st._selection_workers(space, users, u_indptr, u_indices, n_slots)
    extra = 51 * n_slots + 4 * ALS_CHUNK * CHUNK + (64 << 20)
    mem = s13._memory()
    if mem is None:
        return w
    fit = max(1, int((mem[0] - s13.MEMORY_RESERVE) // extra))
    if fit < w:
        say(
            f"{WORK_NAME}: the dense arms need about {extra / 2**30:.2f} GB "
            f"per worker, so {fit} workers instead of {w}"
        )
    return max(1, min(w, fit))


def beyond_pass(st, ctx, users, picks, work, key, say, p14_cmp, workers=None):
    """The pass over the listeners (module docstring): per-listener arrays,
    saved in blocks as they finish, so a stopped run continues with the
    blocks still missing."""
    n_slots = ctx["n_slots"]
    u_indptr, u_indices = ctx["u_indptr"], ctx["u_indices"]
    tasks = block_tasks(len(users))
    pdir = work / "pass"
    state_path = pdir / "state.json"
    k = {
        "key": s13._sha(key),
        "users": p14._array_key(users, picks),
        "blocks": [list(t) for t in tasks],
        "settings": [K, CHUNK, ALS_CHUNK, CHECK_STRIDE],
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
        _E.update(ctx, users=users, picks=picks)
        try:
            left = np.concatenate([np.arange(*tasks[i]) for i in pending])
            if workers is None:
                workers = workers_for(
                    st, ctx["space"], users[left], u_indptr, u_indices, n_slots, say
                )
            take = np.unique(
                np.round(
                    np.linspace(0, len(left) - 1, min(PROBE_USERS, len(left)))
                ).astype(np.int64)
            )
            probe, secs = {}, []
            with threadpool_limits(1):
                for j, i in enumerate(take, start=1):
                    kk = int(left[i])
                    t1 = time.time()
                    probe[kk] = _block(_E, kk, kk + 1)
                    secs.append(time.time() - t1)
                    say(
                        f"{WORK_NAME}: probe listener {j} of {len(take)}: "
                        f"{secs[-1]:.2f} s for {len(ARMS)} arms (the ALS "
                        f"product of its whole chunk included)"
                    )
            par = max(
                1.0,
                min(float(workers), s13b.PASS_SPEEDUP_PER_CORE * s13._physical_cores()),
            )
            say(
                f"{WORK_NAME}: probe mean {np.mean(secs):.2f} s per listener "
                f"in one process. Rough estimate for {len(left):,} listeners "
                f"on {workers} workers, with a parallel speedup of {par:.1f}: "
                f"{_clock(np.mean(secs) * len(left) / par)}. The ALS product "
                f"is shared by the {ALS_CHUNK} listeners of a block, so the "
                f"real rate is higher. The progress lines below measure it"
            )
            prog = Progress(
                f"{WORK_NAME}: pass, {len(ARMS)} arms, " f"{workers} workers",
                len(left),
                say,
                unit=" listeners",
            )
            with s13._Pool(workers) as pool:
                for j, out in pool.unordered(_task, [tasks[i] for i in pending]):
                    i = pending[j]
                    check_block(out, tasks[i][0], *p14_cmp)
                    s13._atomic_savez(part(i), **out)
                    state["parts"][str(i)] = file_sha256(part(i))
                    atomic_json_write(state, state_path)
                    prog.advance(tasks[i][1] - tasks[i][0])
            prog.finish()
            starts = np.array([t[0] for t in tasks])
            for kk, res in probe.items():
                i = int(np.searchsorted(starts, kk, side="right") - 1)
                z = np.load(part(i))
                jj = kk - tasks[i][0]
                if not _same(
                    {key_: v[0] for key_, v in res.items()},
                    {key_: z[key_][jj] for key_ in res},
                ):
                    raise ValueError(
                        f"eval listener {int(users[kk])}: the "
                        f"pool's results differ from the "
                        f"probe's in this process"
                    )
        finally:
            _E.clear()
    zs = [np.load(part(i)) for i in range(len(tasks))]
    out = {key_: np.concatenate([z[key_] for z in zs]) for key_ in zs[0].files}
    if len(out["n_cand"]) != len(users):
        raise ValueError("the pass blocks do not cover the listeners")
    for i, (lo, hi) in enumerate(tasks):
        check_block({key_: v[lo:hi] for key_, v in out.items()}, lo, *p14_cmp)
    return out


# Checks


def full_sort(s, picks, new, known, keep=K):
    """A second route for one listener and arm: a full sort of the
    candidates (score descending, then slot ascending). Returns the top
    list, its scores and known flags, first_new and the picks' ranks."""
    cand = np.flatnonzero(s > -np.inf)
    order = cand[np.lexsort((cand, -s[cand]))]
    pos = np.empty(len(s), dtype=np.int64)
    pos[order] = np.arange(1, len(order) + 1)
    first = np.flatnonzero(new[order])
    return {
        "top": order[:keep],
        "score": s[order[:keep]].astype(np.float64),
        "known": known[order[:keep]],
        "first_new": int(first[0]) + 1 if len(first) else -1,
        "ranks": pos[np.asarray(picks, dtype=np.int64)],
    }


def verify_dense(ctx, P, users, picks, terms, tables, say, n_verify):
    """VERIFY_USERS listeners against the full sorts of the second routes
    (module docstring). Returns the check record."""
    space, n_slots = ctx["space"], ctx["n_slots"]
    u_indptr, u_indices, u_data = (ctx["u_indptr"], ctx["u_indices"], ctx["u_data"])
    X, Yd = ctx["als_x"], ctx["als_y"]
    lists = {
        kind: (
            np.asarray(space.colisten[kind].indptr, np.int64),
            np.asarray(space.colisten[kind].indices, np.int32),
            np.asarray(space.colisten[kind].data, np.float64),
        )
        for kind in knn.KINDS
    }
    rng = np.random.default_rng(VERIFY_SEED)
    take = np.sort(
        rng.choice(len(users), size=min(n_verify, len(users)), replace=False)
    )
    prog = Progress(
        f"{WORK_NAME}: the pass against full sorts of second " f"routes",
        len(take),
        say,
        unit=" listeners",
    )
    worst = 0.0
    for k in take:
        u = int(users[k])
        items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
        state = cn.build_user_state(items, weights, space)
        full = cn.full_catalog(state, space)
        v, h = full[AFFINITY]
        known, new = h & (v > 0.0), h & (v == 0.0)
        cand = np.ones(n_slots, dtype=bool)
        cand[0] = False
        cand[items] = False
        want = {}
        ref = ct.reference_scores(full, terms, tables, 0.0, space.tier)
        for c_, arm in enumerate(GAM_ARMS):
            want[arm] = full_sort(
                np.where(cand, ref[c_], -np.inf), picks[k], new, known
            )
        del full, ref
        hi_, hw = knn.history(u_indptr, u_indices, u_data, u)
        fast = knn.user_scores(ctx["t_mats"], hi_, hw)
        slow = knn.brute_scores(lists, hi_, hw, n_slots)
        for kind, arm in (("same_artist", "knn_same"), ("both", "knn_both")):
            scale = max(1.0, float(np.max(np.abs(slow[kind]))))
            gap = float(np.max(np.abs(fast[kind] - slow[kind]))) / scale
            worst = max(worst, gap)
            if gap > iks.SCORE_RTOL:
                raise ValueError(
                    f"listener {u}: ItemKNN's {kind} scores "
                    f"differ from itemknn.brute_scores"
                )
            want[arm] = full_sort(
                knn.candidate_scores(fast[kind], hi_), picks[k], new, known
            )
        del fast, slow
        c0 = (int(k) // ALS_CHUNK) * ALS_CHUNK
        with threadpool_limits(int(os.cpu_count() or 1)):
            S = np.asarray(X[users[c0 : c0 + ALS_CHUNK]], dtype=np.float32) @ Yd.T
        row = np.where(cand, S[int(k) - c0], -np.inf)
        del S
        want["als_d"] = full_sort(row, picks[k], new, known)
        want["popularity"] = full_sort(
            np.where(cand, ctx["pop_f"], -np.inf), picks[k], new, known
        )
        for arm in ARMS:
            w = want[arm]
            t = len(w["top"])
            got_top = P[f"top_{arm}"][k].astype(np.int64)
            ok = (
                np.array_equal(got_top[:t], w["top"])
                and np.all(got_top[t:] == -1)
                and np.array_equal(P[f"score_{arm}"][k][:t], w["score"])
                and np.array_equal(P[f"known_{arm}"][k][:t], w["known"])
                and int(P[f"first_new_{arm}"][k]) == w["first_new"]
                and np.array_equal(P[f"rank_{arm}"][k], w["ranks"])
            )
            if not ok:
                raise ValueError(
                    f"listener {u}, {arm}: the pass's list, "
                    f"scores, known flags, first_new or ranks "
                    f"differ from a full sort of the second "
                    f"route"
                )
        prog.advance(1)
    prog.finish()
    return {"listeners": int(len(take)), "knn_score_gap": worst}


def check_popularity(pop, users, picks, u_indptr, u_indices, P):
    """Each listener's popularity list equals P5's global order
    (evaluation.global_ranks: score descending, slot ascending) without
    slot 0 and the listener's training items, with the scores of P5."""
    rank = ev.global_ranks(np.asarray(pop))
    order = np.empty(len(rank), dtype=np.int64)
    order[rank - 1] = np.arange(len(rank))  # the slot at each rank
    for k, u in enumerate(users):
        excl = np.asarray(
            u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])], dtype=np.int64
        )
        head = order[: K + len(excl) + 1]
        head = head[(head != 0) & ~np.isin(head, excl)][:K]
        got = P["top_popularity"][k].astype(np.int64)
        if not (
            np.array_equal(got[: len(head)], head)
            and np.array_equal(
                P["score_popularity"][k][: len(head)], np.asarray(pop, np.float64)[head]
            )
        ):
            raise ValueError(
                f"listener {int(u)}: the popularity list is not "
                f"P5's global order without the training items"
            )
    return True


def check_lists(P, users, u_indptr, u_indices, n_slots, primary):
    """Every list: ids inside the slots, no training item, no duplicate,
    sorted by score descending then id ascending, -1 only as padding at
    the end, and known flags only on credited tracks."""
    for arm in ARMS:
        ids = P[f"top_{arm}"].astype(np.int64)
        sc = P[f"score_{arm}"]
        kn = P[f"known_{arm}"]
        valid = ids >= 0
        if np.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError(f"{arm}: a list has a gap before its last entry")
        if (
            np.any(valid & ((ids < 1) | (ids >= n_slots)))
            or np.any(valid & ~np.isfinite(sc))
            or np.any(~valid & kn)
        ):
            raise ValueError(f"{arm}: a list entry lies outside the slots")
        a, b = ids[:, :-1], ids[:, 1:]
        sa, sb = sc[:, :-1], sc[:, 1:]
        both = valid[:, 1:]
        if np.any(both & ((sb > sa) | ((sb == sa) & (b <= a)))):
            raise ValueError(
                f"{arm}: a list is not sorted by score "
                f"descending, then item id ascending"
            )
        srt = np.sort(np.where(valid, ids, -1 - np.arange(ids.shape[1])), axis=1)
        if np.any(srt[:, 1:] == srt[:, :-1]):
            raise ValueError(f"{arm}: a list holds a track twice")
        if np.any(kn & valid & (primary[np.maximum(ids, 0)] < 0)):
            raise ValueError(f"{arm}: a track without a credit is known")
        for k, u in enumerate(users):
            excl = np.asarray(u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])])
            if np.any(np.isin(ids[k][valid[k]], excl)):
                raise ValueError(
                    f"{arm}, listener {int(u)}: a list holds a " f"training item"
                )
    return True


def check_gam_known(P, z14, terms):
    """The GAM's top-10 known flags against P14's stored artist affinity
    (module docstring). Returns the record."""
    names = [t.name for t in terms]
    if AFFINITY not in names:
        return {"affinity_term": False}
    j = names.index(AFFINITY)
    has = np.asarray(z14["top_has"])[:, :TOP, j]
    val = np.asarray(z14["top_val"])[:, :TOP, j]
    match = has & (np.nan_to_num(val) > 0.0)
    if not np.array_equal(P["known_gam"][:, :TOP], match):
        raise ValueError(
            "the GAM's top-10 known flags differ from P14's "
            "stored artist affinity above 0"
        )
    share = float(P["known_gam"][:, :TOP].mean())
    if bool(match.all()) and share != 1.0:
        raise ValueError(
            f"P14 stored an artist match for every top-10 track "
            f"of the GAM, but its known-artist share is "
            f"{share}"
        )
    return {
        "affinity_term": True,
        "p14_match_share": float(match.mean()),
        "known_share": share,
        "all_known": bool(match.all()),
    }


# Results


def interval(x):
    """The mean of x over the listeners with a 95 percent normal
    interval."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {
        "mean": m,
        "se": p14._num(se),
        "lo": p14._num(m - Z95 * se),
        "hi": p14._num(m + Z95 * se),
        "n": int(n),
    }


def paired(a, b):
    return {
        k: (int(v) if k == "n" else p14._num(v))
        for k, v in ga.paired_summary(a, b).items()
    }


def gini(counts):
    """The Gini index of non-negative counts, zeros included:
    sum_i (2 i - n - 1) c_(i) / (n sum c), with c sorted ascending and i
    from 1. Integer sums, so the order of the additions cannot change it."""
    c = np.sort(np.asarray(counts, dtype=np.int64))
    n, tot = len(c), int(c.sum())
    if n == 0 or tot == 0:
        return None
    w = 2 * np.arange(1, n + 1, dtype=np.int64) - n - 1
    return int(np.dot(w, c)) / (n * tot)


def jaccard(A, B):
    """The Jaccard index of the sets in the rows of A and B (-1: no
    entry). Rows hold no duplicate."""
    va, vb = A >= 0, B >= 0
    inter = ((A[:, :, None] == B[:, None, :]) & va[:, :, None]).any(axis=2).sum(axis=1)
    union = va.sum(axis=1) + vb.sum(axis=1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def occurrence(prim):
    """For each entry of each row, how many earlier entries of the row have
    the same primary artist. Entries without one (-1) get their count among
    such entries, which the cap walk never reads: it keeps them all."""
    n, k = prim.shape
    rows = np.repeat(np.arange(n), k)
    pos = np.tile(np.arange(k), n)
    p = prim.ravel()
    order = np.lexsort((pos, p, rows))
    sp, sr = p[order], rows[order]
    start = np.ones(len(order), dtype=bool)
    start[1:] = (sr[1:] != sr[:-1]) | (sp[1:] != sp[:-1])
    first = np.maximum.accumulate(np.where(start, np.arange(len(order)), 0))
    occ = np.empty(len(order), dtype=np.int64)
    occ[order] = np.arange(len(order)) - first
    return occ.reshape(n, k)


def cap_positions(ids, prim, cap, keep=TOP, occ=None):
    """The cap walk (module docstring): (n, keep) positions in the top-K
    list of the kept tracks, -1 where fewer were kept. An artist's
    occurrences are kept in order up to the cap, so a track is kept when
    fewer than cap earlier tracks share its primary artist. occ: the
    occurrence() of prim, when the caller has it."""
    valid = ids >= 0
    if cap is not None and occ is None:
        occ = occurrence(prim)
    ok = valid if cap is None else valid & ((prim < 0) | (occ < cap))
    take = ok & (np.cumsum(ok, axis=1) <= keep)
    out = np.full((ids.shape[0], keep), -1, dtype=np.int64)
    r, c_ = np.nonzero(take)
    slot = np.cumsum(take, axis=1)[r, c_] - 1
    out[r, slot] = c_
    return out


def distinct(prim):
    """The distinct primary artists in each row (-1: none)."""
    s = np.sort(prim, axis=1)
    first = np.ones(s.shape, dtype=bool)
    first[:, 1:] = s[:, 1:] != s[:, :-1]
    return ((s >= 0) & first).sum(axis=1)


def capped_metrics(ids, known, prim, occ, picks, cap):
    """Per listener under one cap: (ndcg@10, r@10, distinct artists,
    known-artist share, kept tracks). prim: the primary artist of each
    list entry (-1: none), occ: its occurrence(). The picks' ranks are
    their positions in the capped list, and K + 1 when the list lacks
    them, so P14's pick_metrics give them no gain at 10."""
    pos = cap_positions(ids, prim, cap, occ=occ)
    got = pos >= 0
    pc_ = np.maximum(pos, 0)
    lst = np.where(got, np.take_along_axis(ids, pc_, axis=1), -1)
    kn = got & np.take_along_axis(known, pc_, axis=1)
    pr = np.where(got, np.take_along_axis(prim, pc_, axis=1), -1)
    eq = (lst[:, None, :] == np.asarray(picks)[:, :, None]) & got[:, None, :]
    found = eq.any(axis=2)
    r = np.where(found, np.argmax(eq, axis=2) + 1, K + 1)
    lm = p14.listener_metrics(r)
    kept = got.sum(axis=1)
    return (
        lm["ndcg_at_10"],
        lm["recall_at_10"],
        distinct(pr),
        kn.sum(axis=1) / np.maximum(kept, 1),
        kept,
    )


def reference_rows(users, u_indptr, u_indices, selfinfo, pop_c):
    """Each listener's mean self-information and mean pop_centile over its
    training tracks."""
    si = np.empty(len(users))
    pc_ = np.empty(len(users))
    for k, u in enumerate(users):
        row = np.asarray(
            u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])], dtype=np.int64
        )
        si[k] = float(selfinfo[row].mean())
        pc_[k] = float(np.asarray(pop_c[row], dtype=np.float64).mean())
    return si, pc_


def self_information(n_i, N):
    return -np.log2((np.asarray(n_i, dtype=np.float64) + 1.0) / (N + 1.0))


def pairs_of(n):
    """PAIRS pairs of distinct listeners, seeded."""
    rng = np.random.default_rng(PAIR_SEED)
    i = rng.integers(0, n, size=PAIRS)
    j = rng.integers(0, n - 1, size=PAIRS)
    return i, j + (j >= i)


def summarize(A, glob):
    """Every result and every per-listener metric from the stored arrays: a
    pure function, so the read-back recomputes it. A: the per-listener
    arrays of the pass. glob: picks, the per-track arrays, N and the
    reference rows."""
    prim_of = glob["primary"]
    multi_of = np.asarray(glob["credits"]) > 1
    si_of = self_information(glob["n_i"], glob["N"])
    pc_of = np.asarray(glob["pop_centile"], dtype=np.float64)
    n_tracks = len(prim_of) - 1
    picks = glob["picks"]
    pi, pj = pairs_of(len(picks))
    res = {
        "arms": {},
        "reference": {
            "self_information": interval(glob["ref_selfinfo"]),
            "pop_centile": interval(glob["ref_pop_centile"]),
        },
    }
    per = {"pair_i": pi, "pair_j": pj}
    for arm in ARMS:
        ids = A[f"top_{arm}"].astype(np.int64)
        known = A[f"known_{arm}"]
        top = ids[:, :TOP]
        v = top >= 0
        prim = np.where(v, prim_of[np.maximum(top, 0)], -1)
        new = v & (prim >= 0) & ~known[:, :TOP]
        nv = np.maximum(v.sum(axis=1), 1)
        m_known = (known[:, :TOP] & v).sum(axis=1) / nv
        m_new = new.any(axis=1).astype(np.float64)
        m_dist = distinct(prim).astype(np.float64)
        m_si = np.where(v, si_of[np.maximum(top, 0)], 0.0).sum(axis=1) / nv
        m_pc = np.where(v, pc_of[np.maximum(top, 0)], 0.0).sum(axis=1) / nv
        m_multi = (v & multi_of[np.maximum(top, 0)]).sum(axis=1) / nv
        flat = top[v]
        union = np.unique(flat)
        up = prim_of[union]
        counts = np.bincount(flat, minlength=len(prim_of))[1:]
        jac = jaccard(top[pi], top[pj])
        fn = A[f"first_new_{arm}"]
        has_new = fn >= 1
        fq = (
            np.percentile(fn[has_new], QUANTILES, method="inverted_cdf")
            if has_new.any()
            else [None] * len(QUANTILES)
        )
        caps = {}
        ndcg0, r0 = None, None
        cap_arrays = []
        prim_k = np.where(ids >= 0, prim_of[np.maximum(ids, 0)], -1)
        occ = occurrence(prim_k)
        for cap in reversed(CAPS):  # no cap first: the reference
            nd, rc, di, kn, kept = capped_metrics(ids, known, prim_k, occ, picks, cap)
            if cap is None:
                ndcg0, r0 = nd, rc
            cap_arrays.append((cap, nd, rc, di, kn, kept))
            caps["none" if cap is None else str(cap)] = {
                "ndcg_at_10": interval(nd),
                "recall_at_10": interval(rc),
                "ndcg_minus_none": paired(nd, ndcg0),
                "recall_minus_none": paired(rc, r0),
                "distinct_artists": interval(di),
                "known_share": interval(kn),
                "unfilled": interval((kept < TOP).astype(np.float64)),
            }
        cap_arrays.reverse()  # the order of CAPS
        res["arms"][arm] = {
            "label": ARM_LABELS[arm],
            "known_share_top10": interval(m_known),
            "new_artist_listeners": interval(m_new),
            "distinct_artists_top10": interval(m_dist),
            "multi_artist_share_top10": interval(m_multi),
            "coverage": {
                "tracks": int(len(union)),
                "share": len(union) / n_tracks,
                "artists": int(len(np.unique(up[up >= 0]))),
            },
            "gini": gini(counts),
            "novelty": {
                "self_information": interval(m_si),
                "pop_centile": interval(m_pc),
            },
            "overlap": {
                "jaccard": float(jac.mean()),
                "pairs": int(PAIRS),
                "identical_pairs": float(np.mean(jac == 1.0)),
            },
            "first_new": {
                "listeners_with_new": int(has_new.sum()),
                "listeners_without_new": int((~has_new).sum()),
                "quantiles": {
                    f"p{q}": (None if x is None else int(x))
                    for q, x in zip(QUANTILES, fq)
                },
                "within": {
                    str(c): interval((has_new & (fn <= c)).astype(np.float64))
                    for c in CUTS
                },
                "mean_known_candidates": float(np.mean(A["n_known_cand"])),
                "mean_new_candidates": float(np.mean(A["n_new_cand"])),
            },
            "caps": {
                key: caps[key]
                for key in [("none" if c is None else str(c)) for c in CAPS]
            },
            "uncapped": {
                "ndcg_at_10": float(ndcg0.mean()),
                "recall_at_10": float(r0.mean()),
            },
        }
        per.update(
            {
                f"{arm}_known10": m_known,
                f"{arm}_new10": m_new,
                f"{arm}_distinct10": m_dist,
                f"{arm}_multi10": m_multi,
                f"{arm}_selfinfo10": m_si,
                f"{arm}_popc10": m_pc,
                f"{arm}_jaccard": jac,
                f"{arm}_cap_ndcg10": np.stack([x[1] for x in cap_arrays], axis=1),
                f"{arm}_cap_r10": np.stack([x[2] for x in cap_arrays], axis=1),
                f"{arm}_cap_distinct": np.stack([x[3] for x in cap_arrays], axis=1),
                f"{arm}_cap_known": np.stack([x[4] for x in cap_arrays], axis=1),
                f"{arm}_cap_kept": np.stack([x[5] for x in cap_arrays], axis=1),
            }
        )
    return res, per


def check_uncapped(res, doc14):
    """With no cap, each arm's NDCG@10 and R@10 equal p14.json's values
    exactly."""
    for arm in ARMS:
        a, b = res["arms"][arm]["uncapped"], doc14["arms"][arm]
        if a["ndcg_at_10"] != b["ndcg_at_10"] or a["recall_at_10"] != b["recall_at_10"]:
            raise ValueError(
                f"{arm}: NDCG@10 {a['ndcg_at_10']!r} and R@10 "
                f"{a['recall_at_10']!r} of the uncapped list "
                f"differ from P14's {b['ndcg_at_10']!r} and "
                f"{b['recall_at_10']!r}"
            )
    return True


# Run


def provenance(model_dir):
    """The SHA-256 of the input files and the markers behind the large
    ones."""
    d = Path(model_dir)
    out = cold.provenance(d)
    for name in (
        "p14.json",
        "p14.npz",
        "p14_cold_gam.json",
        "p14_cold_gam.npz",
        "gam_refit.json",
        "gam_refit.npz",
        "als_d/best.json",
        "stats_item_train.parquet",
        "stats_user_train.parquet",
        "item_covariates.parquet",
        "train_manifest.json",
        "csr_meta.json",
        "id_spaces.json",
    ):
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    for stage in (
        "p4_features",
        "p5_eval_baseline",
        "p6_als",
        "p9b_colisten",
        "p25_train",
        "als_phase_d",
    ):
        try:
            out[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            out[f"marker:{stage}"] = None
    return out


def load_p14(d):
    """p14.json and the arrays of p14.npz that this stage reads."""
    doc = load_json(d / p14.OUT_FILES[0])
    if doc is None or doc.get("version") != p14.VERSION:
        raise ValueError(
            f"{p14.OUT_FILES[0]} is missing or has another "
            f"version: run python3 -m pipeline.runner --only "
            f"p14_eval first"
        )
    z = np.load(d / p14.OUT_FILES[1])
    keys = [
        "user_id",
        "picks",
        "history",
        "tables",
        "top_id",
        "top_score",
        "top_has",
        "top_val",
    ] + [f"rank_{a}" for a in ARMS]
    return doc, {k: np.asarray(z[k]) for k in keys}


def write_npz(path, arrays):
    """The arrays as an .npz file with a fixed time stamp on every member,
    so equal arrays give equal bytes, written under a temporary name and
    renamed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(
                    fh, np.asanyarray(arrays[name]), allow_pickle=False
                )
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run(
    st, io, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None, workers=None
):
    """The stage (module docstring). st: a StageP13b2 whose loaders give the
    item side, the CSR and the slots. io: an Inputs. Returns the output
    doc."""
    try:
        return _run(st, io, Path(model_dir), Path(work), say, n_verify, inputs, workers)
    finally:
        _E.clear()


def _run(st, io, d, work, say, n_verify, inputs, workers):
    t0 = time.time()
    work.mkdir(parents=True, exist_ok=True)
    say(f"{WORK_NAME}: {VERSION}, rules declared {DECLARED}. Saved work: " f"{work}")
    doc14, z14 = load_p14(d)
    base = st._base_context()
    n_slots, id_min = st._id_slots()
    G = p14.load_models(d, base)
    terms = G["terms"]
    tables = np.asarray(G["tables"], dtype=np.float64)
    if not np.array_equal(z14["tables"], tables):
        raise ValueError(
            "p14.npz holds other tables than the models of " "p14.load_models"
        )
    users = np.asarray(z14["user_id"], dtype=np.int64)
    picks = np.asarray(z14["picks"], dtype=np.int64)
    if (
        len(users) < 2
        or np.any(np.diff(users) <= 0)
        or int(doc14["data"]["listeners"]) != len(users)
    ):
        raise ValueError("P14's listeners are not sorted, distinct and " "complete")
    u_indptr, u_indices, u_data = st._csr()
    if int(users.max()) >= len(u_indptr) - 1 or int(users.min()) < 1:
        raise ValueError("an eval listener lies outside the user CSR")
    if not np.array_equal(
        np.asarray(u_indptr[users + 1], np.int64)
        - np.asarray(u_indptr[users], np.int64),
        np.asarray(z14["history"], np.int64),
    ):
        raise ValueError("p14.npz's history lengths differ from the CSR")
    say(
        f"{WORK_NAME}: {len(users):,} eval listeners of P14, {len(ARMS)} "
        f"arms, top {K}, {n_slots - 1:,} tracks"
    )
    pop, pop_c, n_i, cov_pairs, N = load_inputs(io, work, n_slots, id_min, say)
    chk_in = check_inputs(
        pop, pop_c, n_i, cov_pairs, N, u_indptr, u_indices, n_slots, say
    )
    say(
        f"{WORK_NAME}: n_i equals the CSR's listeners per track, N "
        f"{N:,}. Loading the item side (no progress lines in this step)"
    )
    t1 = time.time()
    src = st._sources(n_slots)
    space = st._space(src)
    credit = space.credit_mats["artists"]
    credit_t = credit.T.tocsr()
    primary = primary_artists(credit)
    has_credit = np.asarray(space.has_credit["artists"], dtype=bool)
    if not np.array_equal(primary >= 0, has_credit):
        raise ValueError("the primary artists do not cover the credited " "tracks")
    scorer = ct.CatalogScorer(terms, tables, space, chunk=CHUNK)
    if scorer.tilt is not None:
        raise ValueError("the pass supports no term with a multiplier")
    X, Y, fpr = io.als("d")
    if Y.dtype != np.float32 or X.dtype != np.float32:
        raise ValueError("the ALS factors are not float32")
    Yd = np.asarray(Y, dtype=np.float32)
    if (
        Yd.shape[0] != n_slots
        or X.shape[1] != Yd.shape[1]
        or X.shape[0] <= int(users.max())
    ):
        raise ValueError("the ALS factors do not fit the slots or the " "listeners")
    say(
        f"{WORK_NAME}: item side, scorer and ALS factors ({Yd.shape[1]}) "
        f"ready in {_clock(time.time() - t1)}. {int(has_credit.sum()):,} "
        f"tracks with a credit, {credit.shape[1]:,} artists"
    )
    ctx = {
        "scorer": scorer,
        "space": space,
        "n_slots": int(n_slots),
        "u_indptr": u_indptr,
        "u_indices": u_indices,
        "u_data": u_data,
        "credit": credit,
        "credit_t": credit_t,
        "has_credit": has_credit,
        "n_credited": int(has_credit.sum()),
        "t_mats": {kind: space.colisten_t[kind] for kind in knn.KINDS},
        "pop_f": np.asarray(pop, dtype=np.float64),
        "als_x": X,
        "als_y": Yd,
    }
    chk_als = als_split_check(X, Yd, users, n_slots, say)
    key = {
        "version": PASS_VERSION,
        "p13a_params": s13._params(),
        "inputs": io.key(),
        "als": fpr,
        "tables": s13._array_sha(tables),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, knn, p14)
        },
    }
    p14_cmp = (
        {a: np.asarray(z14[f"rank_{a}"], dtype=np.int64) for a in ARMS},
        np.asarray(z14["top_id"], dtype=np.int64),
        np.asarray(z14["top_score"], dtype=np.float64),
    )
    t2 = time.time()
    P = beyond_pass(st, ctx, users, picks, work, key, say, p14_cmp, workers)
    say(
        f"{WORK_NAME}: pass done in {_clock(time.time() - t2)}. Every pick "
        f"rank equals P14's, and the GAM arms' top {P14_TOP} equal P14's "
        f"top lists"
    )
    chk_dense = verify_dense(ctx, P, users, picks, terms, tables, say, n_verify)
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    check_popularity(pop, users, picks, u_indptr, u_indices, P)
    check_lists(P, users, u_indptr, u_indices, n_slots, primary)
    chk_known = check_gam_known(P, z14, terms)
    say(
        f"{WORK_NAME}: the lists pass their checks. Popularity follows P5's "
        f"global order. The GAM's top-10 known flags equal P14's artist "
        f"matches"
    )
    n_credits = np.diff(credit.indptr).astype(np.int16)
    n_credits[0] = 0
    glob = {
        "picks": picks,
        "primary": primary,
        "credits": n_credits,
        "n_i": n_i,
        "pop_centile": pop_c,
        "N": N,
    }
    glob["ref_selfinfo"], glob["ref_pop_centile"] = reference_rows(
        users, u_indptr, u_indices, self_information(n_i, N), pop_c
    )
    res, per = summarize(P, glob)
    check_uncapped(res, doc14)
    out = p14._norm(
        {
            "version": VERSION,
            "declared": DECLARED,
            "protocol": protocol(),
            "data": {
                "listeners": int(len(users)),
                "n_slots": int(n_slots),
                "tracks": int(n_slots - 1),
                "train_listeners": int(N),
                "credited_tracks": int(has_credit.sum()),
                "artists": int(credit.shape[1]),
            },
            "models": {
                "final": G["rec"]["fit_name"],
                "terms": [t.name for t in terms],
                "artist": G["artist"],
                "cold": G["cold"],
            },
            "results": res,
            "check": {
                "inputs": chk_in,
                "als_split": chk_als,
                "dense": chk_dense,
                "gam_known": chk_known,
                "ranks_equal_p14": True,
                "gam_top_equal_p14": P14_TOP,
                "uncapped_equal_p14": True,
                "lists": True,
                "popularity_order": True,
                "probe_equal": True,
                "slots_checked": int(P["checked"].sum()),
            },
            "inputs": dict(inputs or {}),
        }
    )
    arrays = {
        "user_id": users,
        "picks": picks,
        "track_primary": primary,
        "track_credits": n_credits,
        "track_listeners": n_i.astype(np.int32),
        "track_pop_centile": pop_c.astype(np.int16),
        "artist_ids": np.asarray(src.credit_ids["artists"], dtype=np.int64),
        "train_listeners": np.int64(N),
        "caps": np.asarray([0 if c is None else c for c in CAPS], dtype=np.int64),
        "ref_selfinfo": glob["ref_selfinfo"],
        "ref_pop_centile": glob["ref_pop_centile"],
    }
    arrays.update({k: v for k, v in P.items() if k != "checked"})
    arrays.update(per)
    write_npz(d / OUT_FILES[1], arrays)
    atomic_json_write(out, d / OUT_FILES[0])
    write_report(out, d / "reports" / REPORT_NAME)
    write_tables(out, d / FIG_DIR)
    say(f"{WORK_NAME}: reading the outputs back")
    check_outputs(d, (u_indptr, u_indices, u_data))
    g = res["arms"]["gam"]
    say(
        f"{WORK_NAME}: GAM known-artist share of the top 10 "
        f"{g['known_share_top10']['mean']:.3f}, median first new artist at "
        f"rank {g['first_new']['quantiles']['p50']}. Done in "
        f"{_clock(time.time() - t0)}"
    )
    return out


def check_outputs(d, u_arrays):
    """Read the outputs back: the listeners, picks and ranks against P14,
    the lists' checks, and every result and per-listener metric recomputed
    from the stored arrays."""
    d = Path(d)
    doc = load_json(d / OUT_FILES[0])
    if doc is None:
        raise ValueError(f"{OUT_FILES[0]} is missing")
    doc14, z14 = load_p14(d)
    z = gkd.load_npz(d / OUT_FILES[1])
    u_indptr, u_indices = u_arrays[0], u_arrays[1]
    users = np.asarray(z["user_id"], dtype=np.int64)
    if not (
        np.array_equal(users, z14["user_id"])
        and np.array_equal(z["picks"], z14["picks"])
    ):
        raise ValueError("the stored listeners or picks are not P14's")
    if [int(x) for x in z["caps"]] != [0 if c is None else c for c in CAPS]:
        raise ValueError("the stored caps differ from CAPS")
    p14_cmp = (
        {a: np.asarray(z14[f"rank_{a}"], dtype=np.int64) for a in ARMS},
        np.asarray(z14["top_id"], dtype=np.int64),
        np.asarray(z14["top_score"], dtype=np.float64),
    )
    for lo, hi in block_tasks(len(users)):
        check_block(
            {
                k: v[lo:hi]
                for k, v in z.items()
                if k.startswith(("top_", "score_", "rank_", "n_cand"))
            },
            lo,
            *p14_cmp,
        )
    n_slots = len(z["track_primary"])
    check_lists(z, users, u_indptr, u_indices, n_slots, z["track_primary"])
    stored_order_check(z, users, u_indptr, u_indices)
    names = list(doc["models"]["terms"])
    if AFFINITY in names:
        j = names.index(AFFINITY)
        match = z14["top_has"][:, :TOP, j] & (
            np.nan_to_num(z14["top_val"][:, :TOP, j]) > 0.0
        )
        if not np.array_equal(z["known_gam"][:, :TOP], match):
            raise ValueError(
                "the stored GAM known flags differ from P14's " "artist matches"
            )
    N = int(z["train_listeners"])
    si, pc_ = reference_rows(
        users,
        u_indptr,
        u_indices,
        self_information(z["track_listeners"], N),
        z["track_pop_centile"],
    )
    if not (
        np.array_equal(si, z["ref_selfinfo"])
        and np.array_equal(pc_, z["ref_pop_centile"])
    ):
        raise ValueError(
            "the stored reference rows do not follow from the " "training tracks"
        )
    if not np.array_equal(z["track_credits"] > 0, z["track_primary"] >= 0):
        raise ValueError("the stored credit counts and primary artists " "disagree")
    glob = {
        "picks": np.asarray(z["picks"], dtype=np.int64),
        "primary": z["track_primary"],
        "credits": z["track_credits"],
        "n_i": z["track_listeners"],
        "pop_centile": z["track_pop_centile"],
        "N": N,
        "ref_selfinfo": z["ref_selfinfo"],
        "ref_pop_centile": z["ref_pop_centile"],
    }
    A = {
        k: z[k]
        for k in z
        if k.startswith(("top_", "known_", "first_new_", "n_known_cand", "n_new_cand"))
    }
    res, per = summarize(A, glob)
    for k, v in per.items():
        if not np.array_equal(v, z[k]):
            raise ValueError(
                f"the stored per-listener array {k} does not "
                f"follow from the stored lists"
            )
    if p14._norm(res) != doc["results"]:
        raise ValueError("the stored results do not follow from the stored " "arrays")
    check_uncapped(res, doc14)
    return True


def stored_order_check(z, users, u_indptr, u_indices):
    """The popularity lists from the stored arrays alone: the same score
    for a track in every list, and each list equal to the merged order of
    all lists (score descending, id ascending) without the listener's
    training items. A track before a list's last entry that no list holds
    would sit in that list, so the merged order covers every list."""
    ids = z["top_popularity"].astype(np.int64)
    sc = z["score_popularity"]
    v = ids >= 0
    flat_i, flat_s = ids[v], sc[v]
    uniq, first = np.unique(flat_i, return_index=True)
    s_u = flat_s[first]
    pos = np.searchsorted(uniq, flat_i)
    if not np.array_equal(s_u[pos], flat_s):
        raise ValueError("a track has two popularity scores in the lists")
    G = uniq[np.lexsort((uniq, -s_u))]
    for k, u in enumerate(users):
        excl = np.asarray(
            u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])], dtype=np.int64
        )
        m = int(v[k].sum())
        head = G[: m + len(excl)]
        head = head[~np.isin(head, excl)][:m]
        if not np.array_equal(ids[k][:m], head):
            raise ValueError(
                f"listener {int(u)}: the popularity list does "
                f"not follow the order of the other lists"
            )
    return True


# Report


def _iv(x, fmt=".3f"):
    if x is None or x.get("mean") is None:
        return "n/a"
    if x.get("lo") is None:
        return format(x["mean"], fmt)
    return (
        f"{format(x['mean'], fmt)} ({format(x['lo'], fmt)} to "
        f"{format(x['hi'], fmt)})"
    )


def _f(x, fmt):
    return "n/a" if x is None else format(x, fmt)


def write_report(out, path):
    res, data = out["results"], out["data"]
    chk = out["check"]
    L = [
        "# Beyond accuracy: the recommended lists",
        "",
        f"- Rules: the docstring of pipeline/beyond.py, declared "
        f"{out['declared']}. Every number is descriptive: no test.",
        f"- Listeners: {data['listeners']:,} eval listeners of P14 with "
        f"both test picks. Each arm's top {out['protocol']['k']} over the "
        f"full catalogue ({data['tracks']:,} tracks), P5's candidate and "
        f"tie rules. The metrics read the top {TOP}.",
        f"- Known artist: a credited artist of the track occurs among the "
        f"credited artists of the listener's training tracks. New artist: "
        f"a credited track without a known artist. Primary artist: the "
        f"first credited artist id. {data['credited_tracks']:,} tracks "
        f"have a credit, by {data['artists']:,} artists.",
        f"- Novelty: self-information -log2((n_i + 1) / (N + 1)) with N = "
        f"{data['train_listeners']:,} training listeners, and P5's "
        f"popularity centile.",
        f"- Checks: every pick rank of every arm equals P14's, the GAM "
        f"arms' top {chk['gam_top_equal_p14']} equal P14's top lists, the "
        f"uncapped NDCG@10 and R@10 equal P14's, "
        f"{chk['slots_checked']:,} sampled GAM scores equal the dense "
        f"reference, {chk['dense']['listeners']} listeners equal full "
        f"sorts of second routes, popularity follows P5's global order, "
        f"and the results read back.",
        "",
        "Values: the mean over the listeners with a 95 percent interval.",
        "",
        f"## The top {TOP} per arm",
        "",
        "| arm | known-artist share | listeners with a new artist | "
        "distinct artists | coverage, tracks | coverage, share | coverage, "
        "artists | Gini | self-information, bits | popularity centile | "
        "Jaccard of two listeners |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for arm in ARMS:
        a = res["arms"][arm]
        cv = a["coverage"]
        L.append(
            f"| {a['label']} | {_iv(a['known_share_top10'])} | "
            f"{_iv(a['new_artist_listeners'])} | "
            f"{_iv(a['distinct_artists_top10'], '.2f')} | "
            f"{cv['tracks']:,} | {cv['share']:.2e} | "
            f"{cv['artists']:,} | {_f(a['gini'], '.4f')} | "
            f"{_iv(a['novelty']['self_information'], '.2f')} | "
            f"{_iv(a['novelty']['pop_centile'], '.1f')} | "
            f"{a['overlap']['jaccard']:.4f} |"
        )
    rf = res["reference"]
    L.append(
        f"| training tracks (reference) | | | | | | | | "
        f"{_iv(rf['self_information'], '.2f')} | "
        f"{_iv(rf['pop_centile'], '.1f')} | |"
    )
    L += [
        "",
        f"Coverage: the distinct tracks and primary artists in the "
        f"union of all top-{TOP} lists, the share of the "
        f"{data['tracks']:,} tracks. Gini: how often each track appears "
        f"in the top-{TOP} lists, every track included. Jaccard: the mean "
        f"over {res['arms']['gam']['overlap']['pairs']:,} random pairs of "
        f"listeners.",
        "",
        "## First new artist",
        "",
        "first_new: the rank of the best-ranked candidate by a new artist, "
        "exact over the full catalogue. Percentiles over the listeners "
        "with a new-artist candidate.",
        "",
        "| arm | p5 | p25 | p50 | p75 | p95 | within 10 | within 100 | "
        "listeners without | known-artist candidates, mean |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for arm in ARMS:
        f = res["arms"][arm]["first_new"]
        q = f["quantiles"]
        L.append(
            f"| {ARM_LABELS[arm]} | "
            + " | ".join(_f(q[f"p{x}"], ",d") for x in QUANTILES)
            + f" | {_iv(f['within']['10'])} | "
            f"{_iv(f['within']['100'])} | "
            f"{f['listeners_without_new']:,} | "
            f"{f['mean_known_candidates']:,.0f} |"
        )
    L += [
        "",
        "## Caps per primary artist",
        "",
        f"The walk keeps a track while its primary artist has fewer kept "
        f"tracks than the cap and stops at {TOP} kept. NDCG@10 and R@10 "
        f"on the test picks, P14's per-pick rule. Differences: capped "
        f"minus no cap, paired over the listeners.",
        "",
        f"Top-{TOP} tracks that credit more than one artist (the demo's "
        f"filter caps each credited artist, this walk only the primary "
        f"one): "
        + ", ".join(
            f"{ARM_LABELS[a]} " f"{_iv(res['arms'][a]['multi_artist_share_top10'])}"
            for a in ARMS
        )
        + ".",
        "",
        "| arm | cap | NDCG@10 | R@10 | NDCG@10 minus no cap | R@10 minus "
        "no cap | distinct artists | known-artist share | not filled |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        for c in CAPS:
            key = "none" if c is None else str(c)
            r = res["arms"][arm]["caps"][key]
            L.append(
                f"| {ARM_LABELS[arm]} | {key} | "
                f"{_iv(r['ndcg_at_10'], '.5f')} | "
                f"{_iv(r['recall_at_10'], '.4f')} | "
                f"{_iv(r['ndcg_minus_none'], '+.5f')} | "
                f"{_iv(r['recall_minus_none'], '+.4f')} | "
                f"{_iv(r['distinct_artists'], '.2f')} | "
                f"{_iv(r['known_share'])} | {_iv(r['unfilled'])} |"
            )
    atomic_text_write("\n".join(L).replace(";", ",") + "\n", path)


def _csv(rows, header):
    buf = iolib.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _ivc(x):
    return [x.get("mean"), x.get("lo"), x.get("hi")]


def table_rows(out):
    """The three tables of the figures notebook: (file name, header, rows)."""
    res = out["results"]
    arms = []
    for arm in ARMS + ("reference",):
        if arm == "reference":
            rf = res["reference"]
            arms.append(
                [arm, "training tracks (reference)"]
                + [None] * 16
                + _ivc(rf["self_information"])
                + _ivc(rf["pop_centile"])
                + [None]
            )
            continue
        a = res["arms"][arm]
        arms.append(
            [arm, a["label"]]
            + _ivc(a["known_share_top10"])
            + _ivc(a["new_artist_listeners"])
            + _ivc(a["distinct_artists_top10"])
            + _ivc(a["multi_artist_share_top10"])
            + [
                a["coverage"]["tracks"],
                a["coverage"]["share"],
                a["coverage"]["artists"],
                a["gini"],
            ]
            + _ivc(a["novelty"]["self_information"])
            + _ivc(a["novelty"]["pop_centile"])
            + [a["overlap"]["jaccard"]]
        )
    h_arms = (
        ["arm", "label"]
        + [
            f"{m}_{s}"
            for m in (
                "known_share",
                "new_artist_listeners",
                "distinct_artists",
                "multi_artist_share",
            )
            for s in ("mean", "lo", "hi")
        ]
        + ["coverage_tracks", "coverage_share", "coverage_artists", "gini"]
        + [
            f"{m}_{s}"
            for m in ("self_information", "pop_centile")
            for s in ("mean", "lo", "hi")
        ]
        + ["jaccard"]
    )
    first = []
    for arm in ARMS:
        f = res["arms"][arm]["first_new"]
        first.append(
            [arm, ARM_LABELS[arm]]
            + [f["quantiles"][f"p{q}"] for q in QUANTILES]
            + _ivc(f["within"]["10"])
            + _ivc(f["within"]["100"])
            + [f["listeners_without_new"], f["mean_known_candidates"]]
        )
    h_first = (
        ["arm", "label"]
        + [f"p{q}" for q in QUANTILES]
        + [f"within_{c}_{s}" for c in CUTS for s in ("mean", "lo", "hi")]
        + ["listeners_without_new", "mean_known_candidates"]
    )
    caps = []
    for arm in ARMS:
        for c in CAPS:
            key = "none" if c is None else str(c)
            r = res["arms"][arm]["caps"][key]
            caps.append(
                [arm, ARM_LABELS[arm], key]
                + _ivc(r["ndcg_at_10"])
                + _ivc(r["recall_at_10"])
                + _ivc(r["ndcg_minus_none"])
                + _ivc(r["recall_minus_none"])
                + _ivc(r["distinct_artists"])
                + _ivc(r["known_share"])
                + _ivc(r["unfilled"])
            )
    h_caps = ["arm", "label", "cap"] + [
        f"{m}_{s}"
        for m in (
            "ndcg_at_10",
            "recall_at_10",
            "ndcg_minus_none",
            "recall_minus_none",
            "distinct_artists",
            "known_share",
            "unfilled",
        )
        for s in ("mean", "lo", "hi")
    ]
    return [
        (TAB_FILES[0], h_arms, arms),
        (TAB_FILES[1], h_first, first),
        (TAB_FILES[2], h_caps, caps),
    ]


def write_tables(out, folder):
    for name, header, rows in table_rows(out):
        atomic_text_write(
            _csv([["" if x is None else x for x in r] for r in rows], header),
            Path(folder) / name,
        )


# Runner stage


class StageBeyond(ScriptStage):
    """The beyond-accuracy measures of the lists as a runner stage."""

    name = "beyond_accuracy"
    upstream = ["p14_eval", "p14_cold_gam"]

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
        out = {
            "results": d / OUT_FILES[0],
            "arrays": d / OUT_FILES[1],
            "report": d / "reports" / REPORT_NAME,
        }
        for name in TAB_FILES:
            out[name.split(".")[0]] = d / FIG_DIR / name
        return out

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
        check_outputs(self.model_dir, self._stage()._csr())

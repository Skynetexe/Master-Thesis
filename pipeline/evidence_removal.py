"""Evidence removal: do the shown reasons point at the evidence that drives
a recommendation? For tracks of the GAM's top 10, the stage removes the
evidence that a shown reason cites from the listener's history or
profile, ranks the full catalogue again with the frozen model, and gives
the track's new rank. It also keeps the artist evidence alone and checks
whether the track stays.

The test follows the probabilities of necessity and sufficiency of
counterfactual explanation (Tan et al.), with removal from the history
(Tran et al.) at the level of single tracks and of the whole list (Nguyen
et al.). It acts on the evidence that the listener sees, which is the
claim that an explanation makes.

RULES, declared before the first run on real data. Every number is
descriptive: no test and no SESOI decision. The model stays frozen.

From P14, exactly as P14 uses them:
- the model: the final model's tables (row 0 of p14.load_models).
- the lists: each eval listener's top 10 of the GAM (p14.npz top_id),
  the reasons shown on them (shown: D1 and S4), and their evidence lists
  (ev_count, ev_id, ev_credit, ev_credit_tracks: up to EVIDENCE_K units
  per reason, largest first, with the count of all units).
- candidates: every item slot except slot 0 and the listener's training
  items. Every pass takes them from the full history, so a removed
  history track does not become a candidate and only the evidence
  changes (as in the cold-start curve).
- tie rule: equal scores rank by item id. A track leaves the top 10 when
  its new rank is above 10.
- scores: p14.term_sums on the concepts of the slots (the arithmetic of
  catalog.reference_scores).

Listeners: P14's eval listeners in a random order (numpy default_rng with
seed ORDER_SEED, permutation). The first SAMPLE_SIZE listeners of that
order run in every case, and the pool measures its rate on them. When
that rate predicts at most HOURS_LIMIT hours for all eval listeners, the
other listeners run too. Otherwise the stage keeps the first SAMPLE_SIZE.
When no rate was measured (the first SAMPLE_SIZE finished in a run that
stopped before the choice), the estimate of the timed probe decides. The
choice is saved with the pass, so a resumed run keeps it.

History and profile. The history is the listener's training tracks with
their play weights (P6's CSR). The profile is what the concepts read from
the history (concepts.build_user_state). A removed track keeps nothing. A
kept track keeps its play weight, so the play-count problem of the
cold-start curve does not arise. Removing a unit (a track tag, an artist
tag or a genre) from the profile sets its share to 0 and computes the
profile's norm again. The history's total weight stays, so the shares of
the other units stay. "No audio data" gives the profile no audio
dimension (as build_user_state does for a history without audio), so
audio fit has no data at any track.

The artist evidence of a track: the history tracks that credit one of
its credited artists (artist affinity) and the history tracks among its
same-artist neighbours (co-listening). P9b's same-artist neighbours share
an artist credit with the track, so on the real data the second set lies
inside the first. The stage counts the neighbours outside it.

Counterfactuals, on the GAM's top 10 of each listener. A pass ranks the
full catalogue with the changed history or profile.
1. artist_shown: the artist reason's shown evidence removed. For each of
   the top 3 tracks whose artist reason is shown and cites at least one
   co-listened history track: one pass without those history tracks (up
   to EVIDENCE_K, as P14 stores them).
2. artist_all: all artist evidence removed. One pass per distinct set of
   credited artists among the top-10 tracks with the artist reason shown,
   without the artist evidence of every such track with that set. Each of
   those tracks takes its result from that pass.
3. artist_kept (sufficiency): the same sets. The history keeps only that
   artist evidence.
4. content: per content reason (track tags, artist tags, genres) shown on
   at least one top-10 track: one pass without the union of the units
   that the reason cites on the whole top 10. Each track with the reason
   shown takes its result from that pass, so it loses its own cited units
   and those of the other tracks.
5. audio: when audio fit is shown on at least one top-10 track: one pass
   with no audio data in the profile.
6. list: one pass without the evidence of every shown reason of the top
   10: the history without the artist evidence of 2 for every artist set,
   then the profile without the units of 4 for every content reason, and
   no audio data when 5 runs. All 10 tracks are tested.
A pass whose history keeps no track does not run. Its tracks count as "no
history left" and stay out of the shares. A reason with no rule above
(none in the final model: its reasons are the artist reason, the three
content reasons and audio fit) is not removed, and the outputs list it.

Measures per counterfactual, descriptive:
- the tracks tested and the listeners with at least one.
- necessity: the share of tested tracks that leave the top 10. For 3,
  sufficiency: the share that stay. Each with a 95 percent interval for
  listener clusters (gam_knn_diag.cluster_mean), cut at 0 and 1.
- the new rank: its 5th, 25th, 50th, 75th and 95th percentiles over the
  tested tracks (the inverted empirical distribution, so each one is a
  rank that occurs).
- for 4: each content reason, and the three together.
- for 6: the mean number of the original 10 tracks that stay in the new
  top 10, over the listeners, with a 95 percent interval cut at 0 and
  10.
- for 1 and 2: the same figures per tertile of the number of history
  tracks by the track's artist (P14's ev_credit_tracks). The cuts are the
  1/3 and 2/3 quantiles over the tested tracks (np.quantile, as P14's
  slices): low up to the first cut, middle up to the second, high above.
- for 1 and 4: the share of tested tracks whose concept value rose (the
  track's co-listening for 1, the reason's concept for 4), and for 4 the
  share whose concept has no data after the removal.

Checks, each one stops the run:
- with nothing removed: for every listener, the pass ranks P14's top 10
  at 1 to 10 with P14's scores, the concepts at those tracks equal P14's
  stored values, and the ranks of both test picks equal P14's.
- the evidence units: P14's own function (p14._evidence, through
  concepts.evidence) on the full history gives each top-10 track exactly
  P14's stored evidence lists. Passes 1, 4 and 6 remove those units.
- removal never raises an evidence sum: the sum of the evidence terms
  before the division by the history's total weight or by the profile's
  norm. For 1, the track's co-listening sum and the history weight behind
  each of its credited artists do not rise. For 2, and for 6 on the
  tracks with the artist reason shown, both are 0. For 4 and 6, the
  removed units have no share in the profile, and for 4 the track's
  cosine numerator does not rise. For 5, and for 6 when audio is removed,
  no dimension has audio data. The value itself also divides by the total
  weight or the norm, which falls with the removal, so the value can
  rise. The report gives how often.
- for 3, the track's artist evidence sums equal those of the full
  history.
- in every pass, the scores of every CHECK_STRIDE-th scored slot and of
  the tested tracks equal catalog.reference_scores() on those slots
  exactly, and the tested tracks' scores equal their scores from one-slot
  ranges.
- VERIFY_USERS listeners: every pass gives the same new ranks, scores and
  concept values as a full sort of the dense reference
  (concepts.full_catalog, catalog.reference_scores).
- the pool gives the probe listeners the results of this process.
- the outputs read back: every result follows from the stored arrays,
  the listeners follow the rule, and the tables follow from the results.

COMPUTATION.
- per listener, the passes above. A pass first scores its tested tracks
  from one-slot ranges (CatalogScorer.concept_range), then counts the
  candidates above each of them: rank = 1 + the candidates with a higher
  score + the candidates with the same score and a smaller item id.
- the bound. A slot without artist evidence for the pass's history (no
  credited artist known to the history, no co-listened history track) has
  artist affinity and co-listening "exactly zero" or "no data". Its score
  is at most U: the sum in term order of the largest value that each term
  can take at such a slot (the two artist terms' state values, the
  largest table entry of every other term, and 0). When every tested
  track of a pass scores above U, no such slot can rank above or tie with
  them, so the pass scores only the slots with artist evidence. Otherwise
  it scores every slot range of CHUNK slots.
- the listeners go in blocks of BLOCK listeners of the random order, saved
  as they finish. Workers run with one BLAS thread.

Outputs: MODEL_DIR/evidence_removal.json (the rules, the results, the
checks), MODEL_DIR/evidence_removal.npz (per listener: the new ranks and
scores of every pass, the removed counts and the flags),
MODEL_DIR/reports/evidence_removal_report.md and
MODEL_DIR/figures/tab_evidence_removal.csv and tab_evidence_tertiles.csv.
The files hold no time and no worker count, so the same inputs give the
same bytes. Every line also goes to MODEL_DIR/logs/evidence_removal.log.
Long steps print a line at most every 30 seconds with the share done, the
time elapsed and the time left.

Resume: after Ctrl+C the same command continues: the finished blocks and
the choice of listeners stay in SCRATCH_DIR/evidence_removal.

Expected run on the real data: a full pass over the catalogue costs about
as much as one of P14's (1.5 s per listener in one process). A listener
needs a full pass for the base ranks, one per artist set for 2 and one for
6 (about 6 in all), and short bounded passes for the others. The first
SAMPLE_SIZE listeners take about 2 to 3 hours on 16 workers. The stage
then prints its estimate for all listeners and its choice.

Usage, from the repository root, after the stage p14_eval:
    python3 -m pipeline.runner --only evidence_removal
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only evidence_removal
"""

import csv
import dataclasses
import io as iolib
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pipeline import beyond as by
from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import gam
from pipeline import gam_knn_diag as gkd
from pipeline import p13c as pc
from pipeline import p14
from pipeline import p14_cold as cold
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.script_stage import ScriptStage, log_file

VERSION = "evidence-removal-v1"
# The key of the pass blocks: a change to the passes bumps it.
PASS_VERSION = "evidence-removal-pass-v1"
WORK_NAME = "evidence_removal"
DECLARED = "before the first run on real data"
OUT_FILES = ("evidence_removal.json", "evidence_removal.npz")
REPORT_NAME = "evidence_removal_report.md"
TAB_FILES = ("tab_evidence_removal.csv", "tab_evidence_tertiles.csv")
FIG_DIR = "figures"
LOG_NAME = "evidence_removal.log"
COMMAND = "python3 -m pipeline.runner --only evidence_removal"
TOP = p14.REASON_K  # the list that P14 explains
SHOWN_TOP = 3  # counterfactual 1 runs on the top 3
EVIDENCE_K = p14.EVIDENCE_K  # P14's evidence units per reason
ORDER_SEED = 4_183_067
SAMPLE_SIZE = 5_000
HOURS_LIMIT = 8.0
CHUNK = p14.EVAL_CHUNK  # slots per range, as P14's pass
BLOCK = 10  # listeners per saved block
CHECK_STRIDE = p14.CHECK_STRIDE
PROBE_USERS = 4
VERIFY_USERS = 4
VERIFY_SEED = 7_290_311
NOT_TESTED = 0  # rank codes below 1
NO_HISTORY = -1
QUANTILES = (5, 25, 50, 75, 95)
CREDIT_SRC = "artist_affinity"
SAME_SRC = "colisten_same_artist"
GATE = (CREDIT_SRC, SAME_SRC)  # the concepts of the artist reason
CREDIT = cn.CREDIT_CONCEPTS[CREDIT_SRC]
SAME = cn.COLISTEN_CONCEPTS[SAME_SRC]
KINDS = ("artist_shown", "artist_all", "artist_kept", "content", "audio", "list")
ROW_OF = {
    "artist_shown": "1",
    "artist_all": "2",
    "artist_kept": "3",
    "content": "4",
    "audio": "5",
    "list": "6",
}
KIND_LABELS = {
    "artist_shown": "artist reason, shown evidence removed",
    "artist_all": "artist reason, all evidence removed",
    "artist_kept": "artist reason, evidence kept alone",
    "content": "content reasons, cited units removed",
    "audio": "audio fit, no audio data",
    "list": "every shown reason of the top 10 removed",
}
REASON_LABELS = {
    "tag_affinity": "track tags",
    "artist_tag_affinity": "artist tags",
    "genre_affinity": "genres",
    "audio_fit": "audio fit",
}
TERTILES = ("low", "middle", "high")
SKIP_DENSE = ("checked", "full_scans")  # counts of the route, not results
Z95 = p14.Z95


def protocol():
    return {
        "version": VERSION,
        "declared": DECLARED,
        "listeners": {
            "order_seed": ORDER_SEED,
            "sample_size": SAMPLE_SIZE,
            "hours_limit": HOURS_LIMIT,
            "rule": "all eval listeners when the rate of the "
            "first sample_size predicts at most "
            "hours_limit hours, else the first "
            "sample_size of the random order",
        },
        "top": TOP,
        "shown_top": SHOWN_TOP,
        "evidence_k": EVIDENCE_K,
        "candidates": "every item slot except slot 0 and the training "
        "items of the full history, ties by item id",
        "leave": f"new rank above {TOP}",
        "counterfactuals": {k: KIND_LABELS[k] for k in KINDS},
        "intervals": "95 percent, listener clusters",
        "quantiles": list(QUANTILES),
        "tests": "none, descriptive",
    }


# Context


def bound_of(terms, T):
    """U of the module docstring: the sum in term order of the largest
    value of each term at a slot without artist evidence. The scores of
    such slots add values at or below these maxima in the same order, and
    a rounded sum never falls when an addend grows, so U bounds them."""
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    if T.shape[0] != 1:
        raise ValueError("the bound reads one table")
    total = 0.0
    for t, off in zip(terms, p14.offsets_of(terms)):
        tab = T[0, off : off + t.size]
        K, kn = t.K, t.knot_array
        if t.source in GATE:
            vals = [
                (
                    float(tab[K + gam.ZERO])
                    if t.zero_state
                    else float(np.interp(np.clip(0.0, kn[0], kn[-1]), kn, tab[:K]))
                ),
                float(tab[K + gam.MISS]) if t.missing_state else 0.0,
            ]
            if t.tiers is not None:
                vals.append(0.0)
        else:
            vals = [float(np.max(tab[:K])), 0.0]
            if t.zero_state:
                vals.append(float(tab[K + gam.ZERO]))
            if t.missing_state:
                vals.append(float(tab[K + gam.MISS]))
        total = total + max(vals)
    return total


def reason_kinds(groups, terms):
    """The removal rule of each reason group: artist, content, audio, or
    other (no rule, not removed)."""
    out = []
    for name, js in groups:
        src = [terms[j].source for j in js]
        if name == "artist":
            out.append("artist")
        elif len(js) == 1 and src[0] in cn.TERM_CONCEPTS:
            out.append("content")
        elif len(js) == 1 and src[0] == "audio_fit":
            out.append("audio")
        else:
            out.append("other")
    return out


def context(
    scorer, space, terms, T, groups, z, u_arrays, chunk=CHUNK, stride=CHECK_STRIDE
):
    """The pass context: the scorer, the item side, the model, the reason
    groups and P14's arrays of every eval listener (z, keyed as
    p14.npz)."""
    terms = list(terms)
    kinds = reason_kinds(groups, terms)
    credit = space.credit_mats[CREDIT]
    srcs = list(dict.fromkeys(t.source for t in terms))
    g_of = {
        kind: [g for g, k_ in enumerate(kinds) if k_ == kind]
        for kind in ("artist", "content", "audio")
    }
    c = {
        "scorer": scorer,
        "space": space,
        "terms": terms,
        "T": np.atleast_2d(np.asarray(T, dtype=np.float64)),
        "tier": space.tier,
        "credit": credit,
        "credit_t": credit.T.tocsr(),
        "groups": [(name, list(js)) for name, js in groups],
        "names": [name for name, _ in groups],
        "kinds": kinds,
        "sources": [t.source for t in terms],
        "src_list": srcs,
        "g_artist": g_of["artist"][0] if g_of["artist"] else None,
        "g_content": g_of["content"],
        "g_audio": g_of["audio"][0] if g_of["audio"] else None,
        "bound": bound_of(terms, T),
        "chunk": int(chunk),
        "stride": int(stride),
        "u_indptr": u_arrays[0],
        "u_indices": u_arrays[1],
        "u_data": u_arrays[2],
    }
    c.update(arrays_of(z))
    if c["shown"].shape[0] != len(groups):
        raise ValueError("P14's shown reasons hold other groups than the " "model's")
    return c


def arrays_of(z):
    """P14's arrays that the passes read, with the GAM's top 10."""
    return {
        "users": np.asarray(z["user_id"], dtype=np.int64),
        "picks": np.asarray(z["picks"], dtype=np.int64),
        "top": np.asarray(z["top_id"], dtype=np.int64)[:, 0, :TOP],
        "top_score": np.asarray(z["top_score"], dtype=np.float64)[:, 0, :TOP],
        "top_val": np.asarray(z["top_val"], dtype=np.float64),
        "top_has": np.asarray(z["top_has"], dtype=bool),
        "shown": np.asarray(z["shown"], dtype=bool),
        "ev_count": np.asarray(z["ev_count"]),
        "ev_id": np.asarray(z["ev_id"]),
        "ev_val": np.asarray(z["ev_val"]),
        "ev_credit": np.asarray(z["ev_credit"]),
        "ev_credit_tracks": np.asarray(z["ev_credit_tracks"]),
        "rank_gam": np.asarray(z["rank_gam"], dtype=np.int64),
    }


# History, profile


def without_units(state, space, units):
    """The state with the units {concept: sorted unit ids} removed from the
    profile: their shares set to 0 and the norm computed again by the
    expression of concepts.build_user_state. The total weight stays."""
    share, norm = dict(state.term_share), dict(state.term_norm)
    for src, ids in units.items():
        name = cn.TERM_CONCEPTS[src]
        p = np.array(share[name], dtype=np.float64)
        p[np.asarray(ids, dtype=np.int64)] = 0.0
        share[name] = p
        norm[name] = float(np.sqrt(np.sum((space.idf[name] * p) ** 2)))
    return dataclasses.replace(state, term_share=share, term_norm=norm)


def without_audio(state):
    """The state with no audio dimension, as concepts.build_user_state
    gives a history without audio."""
    D = len(state.audio_dim_ok)
    return dataclasses.replace(
        state,
        audio_mu=np.zeros(D),
        audio_sigma=np.ones(D),
        audio_dim_ok=np.zeros(D, dtype=bool),
    )


def isin_sorted(a, sorted_b):
    """For each entry of a, whether it lies in the sorted array sorted_b."""
    a = np.asarray(a, dtype=np.int64)
    if not len(sorted_b):
        return np.zeros(len(a), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_b, a), len(sorted_b) - 1)
    return sorted_b[pos] == a


def artist_evidence(c, items, t):
    """(sorted history tracks with artist evidence for track t, sorted
    history tracks that credit one of t's artists): the second set and the
    history tracks among t's same-artist neighbours make the first."""
    credit, credit_t = c["credit"], c["credit_t"]
    cols = credit.indices[credit.indptr[t] : credit.indptr[t + 1]]
    by_art = (
        np.unique(credit_t[cols].indices) if len(cols) else np.zeros(0, dtype=np.int64)
    )
    a = items[isin_sorted(items, by_art)]
    m = c["space"].colisten[SAME]
    nb = np.sort(m.indices[m.indptr[t] : m.indptr[t + 1]].astype(np.int64))
    b = items[isin_sorted(items, nb)]
    return np.union1d(a, b).astype(np.int64), a.astype(np.int64)


def artist_sums(c, state, t):
    """The evidence sums of track t's artist concepts: the co-listening sum
    over the history (the numerator of co-listening) and the history
    weight behind each credited artist (the numerators of artist
    affinity)."""
    m = c["space"].colisten[SAME]
    s, e = int(m.indptr[t]), int(m.indptr[t + 1])
    w = cn._history_weight(state, m.indices[s:e].astype(np.int64))
    col = float(np.sum(m.data[s:e] * w))
    credit = c["credit"]
    cols = credit.indices[credit.indptr[t] : credit.indptr[t + 1]]
    sub = credit[state.items][:, cols]
    return col, np.asarray(sub.T @ state.weights, dtype=np.float64).ravel()


def content_numer(c, state, src, t):
    """The numerator of track t's cosine for a content concept."""
    name = cn.TERM_CONCEPTS[src]
    space = c["space"]
    m = space.term_mats[name]
    s, e = int(m.indptr[t]), int(m.indptr[t + 1])
    cols = m.indices[s:e]
    coef = space.idf[name][cols] ** 2 * state.term_share[name][cols]
    return float(np.dot(m.data[s:e], coef))


# Plans


class Plan:
    """One pass: its history, its state (None when no track is left), the
    tested slots and where their results go."""

    def __init__(
        self,
        kind,
        index,
        pos,
        slots,
        items,
        weights,
        state,
        removed=0,
        kept=0,
        neighbour_only=0,
        units=None,
        audio=False,
    ):
        self.kind, self.index = kind, int(index)
        self.pos = np.asarray(pos, dtype=np.int64)
        self.slots = np.asarray(slots, dtype=np.int64)
        self.items, self.weights, self.state = items, weights, state
        self.removed, self.kept = int(removed), int(kept)
        self.neighbour_only = int(neighbour_only)
        self.units = dict(units or {})
        self.audio = bool(audio)
        self.same_history = items is None
        self.res = None


def listener_inputs(c, k):
    """Listener k's history, state and top 10, and the check of the
    evidence units against P14's stored lists."""
    space = c["space"]
    u = int(c["users"][k])
    items, weights = cd.history_of(c["u_indptr"], c["u_indices"], c["u_data"], u)
    state = cn.build_user_state(items, weights, space)
    top = np.asarray(c["top"][k], dtype=np.int64)
    picks = np.asarray(c["picks"][k], dtype=np.int64)
    for what, arr in (("a test pick", picks), ("a top-10 track", top)):
        if (
            np.any(isin_sorted(arr, state.items))
            or np.any(arr < 1)
            or np.any(arr >= space.n_slots)
        ):
            raise RuntimeError(
                f"listener {u}: {what} is a training item or " f"lies outside the slots"
            )
    if len(np.unique(top)) != TOP:
        raise RuntimeError(f"listener {u}: P14's top {TOP} repeats a track")
    for i, t in enumerate(top):
        cnt, ids, vals, cr, crn = p14._evidence(c, state, int(t))
        if not (
            np.array_equal(cnt, c["ev_count"][k, i])
            and np.array_equal(ids, c["ev_id"][k, i])
            and np.array_equal(vals, c["ev_val"][k, i], equal_nan=True)
            and cr == int(c["ev_credit"][k, i])
            and crn == int(c["ev_credit_tracks"][k, i])
        ):
            raise RuntimeError(
                f"listener {u}, top track {i + 1}: "
                f"concepts.evidence on the full history "
                f"differs from P14's stored evidence lists"
            )
    return {
        "u": u,
        "k": int(k),
        "state": state,
        "items": state.items,
        "weights": state.weights,
        "top": top,
        "picks": picks,
        "shown": np.asarray(c["shown"][:, k, :], dtype=bool),
    }


def cited(c, k, i, g):
    """The units that reason g cites on top track i, as P14 stores them."""
    n = min(int(c["ev_count"][k, i, g]), EVIDENCE_K)
    ids = np.asarray(c["ev_id"][k, i, g, :n], dtype=np.int64)
    if np.any(ids < 0):
        raise RuntimeError(
            f"listener {int(c['users'][k])}: P14's evidence "
            f"list of top track {i + 1} has a gap"
        )
    return ids


def plans_of(c, L):
    """The passes of one listener (module docstring), in a fixed order:
    the base, 1 (top 3), 2 and 3 (per artist set), 4 (per content
    reason), 5 and 6."""
    space = c["space"]
    k, top, shown = L["k"], L["top"], L["shown"]
    items, weights, base = L["items"], L["weights"], L["state"]
    ga = c["g_artist"]

    def make(kind, index, pos, slots, keep=None, units=None, audio=False, **info):
        if keep is None:
            h_i, h_w, state = None, None, base
        else:
            h_i, h_w = items[keep], weights[keep]
            state = cn.build_user_state(h_i, h_w, space) if len(h_i) else None
        if state is not None and units:
            state = without_units(state, space, units)
        if state is not None and audio:
            state = without_audio(state)
        return Plan(
            kind, index, pos, slots, h_i, h_w, state, units=units, audio=audio, **info
        )

    plans = [make("base", 0, np.arange(TOP), np.concatenate((top, L["picks"])))]
    art = np.zeros(TOP, dtype=bool) if ga is None else np.asarray(shown[ga], dtype=bool)
    for i in range(SHOWN_TOP):
        if not art[i] or int(c["ev_count"][k, i, ga]) == 0:
            continue
        R = cited(c, k, i, ga)
        if not np.all(isin_sorted(R, items)):
            raise RuntimeError(
                f"listener {L['u']}: a cited co-listened "
                f"track of top track {i + 1} is not in the "
                f"history"
            )
        keep = ~isin_sorted(items, np.unique(R))
        plans.append(
            make("artist_shown", i, [i], top[[i]], keep, removed=int((~keep).sum()))
        )
    sets = {}
    for i in np.flatnonzero(art):
        t = int(top[i])
        cr = c["credit"]
        key = tuple(
            int(x) for x in np.sort(cr.indices[cr.indptr[t] : cr.indptr[t + 1]])
        )
        sets.setdefault(key, []).append(int(i))
    ev_all = []
    for j, pos in enumerate(sets.values()):
        evs = [artist_evidence(c, items, int(top[i])) for i in pos]
        E = np.unique(np.concatenate([e for e, _ in evs]))
        A = np.unique(np.concatenate([a for _, a in evs]))
        ev_all.append(E)
        inE = isin_sorted(items, E)
        nb_only = int(len(E) - len(A))
        plans.append(
            make(
                "artist_all",
                j,
                pos,
                top[pos],
                ~inE,
                removed=int(inE.sum()),
                neighbour_only=nb_only,
            )
        )
        plans.append(make("artist_kept", j, pos, top[pos], inE, kept=int(inE.sum())))
    units_all = {}
    for g in c["g_content"]:
        pos = np.flatnonzero(shown[g])
        if not len(pos):
            continue
        if np.any(c["ev_count"][k, pos, g] < 1):
            raise RuntimeError(
                f"listener {L['u']}: a shown {c['names'][g]} " f"reason cites no unit"
            )
        U = np.unique(np.concatenate([cited(c, k, int(i), g) for i in pos]))
        src = c["terms"][c["groups"][g][1][0]].source
        units_all[src] = U
        plans.append(make("content", g, pos, top[pos], units={src: U}, removed=len(U)))
    g5 = c["g_audio"]
    audio = g5 is not None and bool(shown[g5].any())
    if audio:
        pos = np.flatnonzero(shown[g5])
        plans.append(make("audio", g5, pos, top[pos], audio=True))
    E_list = (
        np.unique(np.concatenate(ev_all)) if ev_all else np.zeros(0, dtype=np.int64)
    )
    if len(E_list) or units_all or audio:
        keep = ~isin_sorted(items, E_list)
        plans.append(
            make(
                "list",
                0,
                np.arange(TOP),
                top,
                keep,
                units=units_all,
                audio=audio,
                removed=int((~keep).sum()),
            )
        )
    return plans


# Scans


def concept_rows(scorer, state, space, pre, source, rows):
    """(value, has) of one concept on the sorted slots rows: the
    expressions of CatalogScorer.concept_range() row by row, so the values
    equal those of the catalogue's ranges bit for bit (a sparse row product
    sums each row in its stored order, whatever rows come with it)."""
    rows = np.asarray(rows, dtype=np.int64)
    if source in cn.TERM_CONCEPTS:
        name = cn.TERM_CONCEPTS[source]
        coef, norm = pre[source]
        numer = space.term_mats[name][rows] @ coef
        has = space.has_terms[name][rows] & (norm > 0)
        denom = np.where(has, space.item_norm[name][rows] * norm, 1.0)
        return np.where(has, numer / denom, np.nan), has
    if source in cn.CREDIT_CONCEPTS:
        name = cn.CREDIT_CONCEPTS[source]
        sub = space.credit_mats[name][rows]
        best = cn._row_max(pre[source][sub.indices], sub.indptr)
        has = space.has_credit[name][rows]
        return np.where(has, best, np.nan), has
    if source in cn.COLISTEN_CONCEPTS:
        name = cn.COLISTEN_CONCEPTS[source]
        has = space.has_colisten[name][rows]
        return np.where(has, pre[source][rows] / state.total, np.nan), has
    if source == "audio_fit":
        value = np.full(len(rows), np.nan, dtype=np.float64)
        has = np.zeros(len(rows), dtype=bool)
        hit = np.flatnonzero(isin_sorted(rows, scorer.audio_rows))
        if len(hit):
            value[hit], has[hit] = cn._audio_fit(state, space, rows[hit])
        return value, has
    if source == "popularity":
        has = rows != 0
        return np.where(has, space.popularity[rows], np.nan), has
    raise KeyError(source)


def evidence_slots(c, pre):
    """The sorted slots with artist evidence for a pass: the tracks that
    credit an artist of the history (artist affinity above 0) and the
    tracks with a co-listened history track (co-listening above 0)."""
    parts = []
    if CREDIT_SRC in pre:
        cols = np.flatnonzero(pre[CREDIT_SRC] > 0)
        if len(cols):
            parts.append(np.asarray(c["credit_t"][cols].indices, dtype=np.int64))
    if SAME_SRC in pre:
        parts.append(np.flatnonzero(pre[SAME_SRC] > 0).astype(np.int64))
    X = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
    return X[X > 0]


def _ref_check(c, S, conc, sub, tier_sub, u, where):
    """Raise RuntimeError unless S[:, sub] equals catalog.reference_scores()
    on those slots."""
    full = {src: (v[sub], h[sub]) for src, (v, h) in conc.items()}
    ref = ct.reference_scores(full, c["terms"], c["T"], 0.0, tier_sub)
    if not np.array_equal(S[:, sub], ref):
        gap = float(np.max(np.abs(S[:, sub] - ref)))
        raise RuntimeError(
            f"listener {u}: the pass's scores differ from "
            f"catalog.reference_scores near slot {where} "
            f"(largest gap {gap:.1e})"
        )


def _count(vals, at, slots, s_t, greater, before):
    """Add, for each tested slot, the scored candidates above it: vals the
    scores at or above the lowest tested score, at their slots."""
    for q in range(len(slots)):
        greater[q] += int(np.count_nonzero(vals > s_t[q]))
        before[q] += int(np.count_nonzero((vals == s_t[q]) & (at < slots[q])))


def _piece(c, L, p, S, conc, ids, out, s_t, greater, before):
    """The checks and counts of one scored piece: S (1, n) the scores of
    the sorted slots ids, out the non-candidates among them. Returns the
    slots checked against the reference."""
    S[:, out] = -np.inf
    hit = isin_sorted(p.slots, ids)
    at = np.searchsorted(ids, p.slots[hit])
    sub = np.unique(np.concatenate((np.arange(0, len(ids), c["stride"]), at)))
    sub = sub[~out[sub]]
    _ref_check(c, S, conc, sub, c["tier"][ids[sub]], L["u"], int(ids[0]))
    Sv = S[0]
    if not np.array_equal(Sv[at], s_t[hit]):
        raise RuntimeError(
            f"listener {L['u']}: a tested track's score in "
            f"its range differs from its one-slot score"
        )
    sel = np.flatnonzero(Sv >= s_t.min())
    _count(Sv[sel], ids[sel], p.slots, s_t, greater, before)
    return len(sub)


def scan_full(c, L, p, pre, s_t):
    """Counts over every slot range of the catalogue."""
    scorer, space = c["scorer"], c["space"]
    n, chunk = scorer.n, c["chunk"]
    excl = L["items"]
    greater = np.zeros(len(p.slots), dtype=np.int64)
    before = np.zeros(len(p.slots), dtype=np.int64)
    checked = 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        conc = {
            src: scorer.concept_range(p.state, space, pre, src, a, b)
            for src in c["src_list"]
        }
        S = p14.term_sums(c["terms"], c["T"], conc, c["tier"][a:b])
        out = np.zeros(b - a, dtype=bool)
        lo, hi = np.searchsorted(excl, (a, b))
        out[excl[lo:hi] - a] = True
        if a == 0:
            out[0] = True
        checked += _piece(c, L, p, S, conc, np.arange(a, b), out, s_t, greater, before)
    return greater, before, checked


def scan_rows(c, L, p, pre, s_t, X):
    """Counts over the slots with artist evidence only (the bound of the
    module docstring)."""
    scorer, space = c["scorer"], c["space"]
    chunk = c["chunk"]
    greater = np.zeros(len(p.slots), dtype=np.int64)
    before = np.zeros(len(p.slots), dtype=np.int64)
    checked = 0
    for lo in range(0, len(X), chunk):
        rows = X[lo : lo + chunk]
        conc = {
            src: concept_rows(scorer, p.state, space, pre, src, rows)
            for src in c["src_list"]
        }
        S = p14.term_sums(c["terms"], c["T"], conc, c["tier"][rows])
        out = isin_sorted(rows, L["items"]) | (rows == 0)
        checked += _piece(c, L, p, S, conc, rows, out, s_t, greater, before)
    return greater, before, checked


def slot_concepts(c, state, pre, slots):
    """{source: (value, has)} of the tested slots, each from its one-slot
    range."""
    scorer, space = c["scorer"], c["space"]
    out = {}
    for src in c["src_list"]:
        v = np.empty(len(slots), dtype=np.float64)
        h = np.empty(len(slots), dtype=bool)
        for q, t in enumerate(slots):
            vv, hh = scorer.concept_range(state, space, pre, src, int(t), int(t) + 1)
            v[q], h[q] = vv[0], hh[0]
        out[src] = (v, h)
    return out


def prepare_sources(scorer, state, space, sources):
    """CatalogScorer.prepare() for some of its sources only: the same code
    on a shorter list."""
    saved = scorer.user_sources
    scorer.user_sources = [s for s in saved if s in sources]
    try:
        return scorer.prepare(state, space)
    finally:
        scorer.user_sources = saved


def plan_inputs(c, L, p):
    """(the scorer's inputs of a pass, its slots with artist evidence or
    None). A pass on the full history shares the base's inputs, apart from
    the sources whose units it removes, and the base's evidence slots."""
    scorer, space = c["scorer"], c["space"]
    if not p.same_history:
        return scorer.prepare(p.state, space), None
    if "pre" not in L:
        L["pre"] = scorer.prepare(L["state"], space)
    pre = dict(L["pre"])
    if p.units:
        pre.update(prepare_sources(scorer, p.state, space, list(p.units)))
    return pre, "X"


def scan_plan(c, L, p):
    """The new ranks, scores and concepts of a pass's tested slots."""
    pre, share = plan_inputs(c, L, p)
    conc = slot_concepts(c, p.state, pre, p.slots)
    s_t = p14.term_sums(c["terms"], c["T"], conc, c["tier"][p.slots])[0]
    bounded = bool(s_t.min() > c["bound"])
    if bounded:
        if share is None:
            X = evidence_slots(c, pre)
        else:
            if "X" not in L:
                L["X"] = evidence_slots(c, L["pre"])
            X = L["X"]
        if not np.all(isin_sorted(p.slots, X)):
            raise RuntimeError(
                f"listener {L['u']}: a tested track above "
                f"the bound has no artist evidence"
            )
        g, b, chk = scan_rows(c, L, p, pre, s_t, X)
    else:
        g, b, chk = scan_full(c, L, p, pre, s_t)
    return {
        "rank": 1 + g + b,
        "score": s_t,
        "conc": conc,
        "checked": chk,
        "bounded": bounded,
    }


def dense_plan(c, L, p):
    """The same as scan_plan() from the dense reference over the whole
    catalogue (for the checks)."""
    space = c["space"]
    full = cn.full_catalog(p.state, space)
    s = ct.reference_scores(full, c["terms"], c["T"], 0.0, space.tier)[0]
    s[0] = -np.inf
    s[L["items"]] = -np.inf
    s_t = s[p.slots]
    rank = np.array(
        [
            1
            + int(np.count_nonzero(s > s_t[q]))
            + int(np.count_nonzero(s[: int(t)] == s_t[q]))
            for q, t in enumerate(p.slots)
        ],
        dtype=np.int64,
    )
    conc = {
        src: (full[src][0][p.slots], full[src][1][p.slots]) for src in c["src_list"]
    }
    return {
        "rank": rank,
        "score": s_t.astype(np.float64),
        "conc": conc,
        "checked": 0,
        "bounded": False,
    }


# Checks


def check_base(c, L, res):
    """With nothing removed, the pass gives P14's top 10, scores, concepts
    and pick ranks."""
    k, u = L["k"], L["u"]
    if not np.array_equal(res["rank"][:TOP], np.arange(1, TOP + 1)):
        raise RuntimeError(
            f"listener {u}: with nothing removed, P14's top "
            f"{TOP} rank at {res['rank'][:TOP].tolist()}"
        )
    if not np.array_equal(res["score"][:TOP], c["top_score"][k]):
        raise RuntimeError(
            f"listener {u}: with nothing removed, the scores "
            f"of the top {TOP} differ from P14's"
        )
    for j, src in enumerate(c["sources"]):
        v, h = res["conc"][src]
        if not (
            np.array_equal(v[:TOP], c["top_val"][k, :, j], equal_nan=True)
            and np.array_equal(h[:TOP], c["top_has"][k, :, j])
        ):
            raise RuntimeError(
                f"listener {u}: with nothing removed, the "
                f"concept {src} at the top {TOP} differs "
                f"from P14's"
            )
    if not np.array_equal(res["rank"][TOP:], c["rank_gam"][k]):
        raise RuntimeError(
            f"listener {u}: with nothing removed, the test "
            f"picks rank at {res['rank'][TOP:].tolist()}, "
            f"P14 at {c['rank_gam'][k].tolist()}"
        )


def check_sums(c, L, plans):
    """The evidence sums of the module docstring."""
    u, top, base = L["u"], L["top"], L["state"]
    ga = c["g_artist"]
    for p in plans:
        if p.state is None or p.kind == "base":
            continue
        if (
            p.kind in ("artist_shown", "artist_all", "artist_kept", "list")
            and ga is not None
        ):
            for i in p.pos:
                if p.kind == "list" and not L["shown"][ga, i]:
                    continue
                t = int(top[i])
                col0, cr0 = artist_sums(c, base, t)
                col1, cr1 = artist_sums(c, p.state, t)
                if p.kind == "artist_shown":
                    ok = col1 <= col0 and bool(np.all(cr1 <= cr0))
                elif p.kind == "artist_kept":
                    ok = col1 == col0 and np.array_equal(cr1, cr0)
                else:
                    ok = col1 == 0.0 and not np.any(cr1 != 0.0)
                if not ok:
                    raise RuntimeError(
                        f"listener {u}, top track {i + 1}, "
                        f"{p.kind}: the artist evidence sums "
                        f"break the rule (co-listening {col0}"
                        f" to {col1})"
                    )
        for src, U in p.units.items():
            name = cn.TERM_CONCEPTS[src]
            if np.any(p.state.term_share[name][U] != 0.0):
                raise RuntimeError(
                    f"listener {u}, {p.kind}: a removed " f"{src} unit keeps a share"
                )
            if p.kind == "content":
                for i in p.pos:
                    t = int(top[i])
                    if content_numer(c, p.state, src, t) > content_numer(
                        c, base, src, t
                    ):
                        raise RuntimeError(
                            f"listener {u}, top track "
                            f"{i + 1}: removing {src} units "
                            f"raised the cosine numerator"
                        )
        if p.audio and p.state.audio_dim_ok.any():
            raise RuntimeError(
                f"listener {u}, {p.kind}: the profile keeps " f"audio data"
            )


# Listener


def empty_out(GC):
    f = np.float64
    return {
        "rank_base": np.zeros(TOP, np.int64),
        "score_base": np.full(TOP, np.nan, f),
        "rank_pick": np.zeros(2, np.int64),
        "rank_shown": np.zeros(SHOWN_TOP, np.int64),
        "score_shown": np.full(SHOWN_TOP, np.nan, f),
        "removed_shown": np.zeros(SHOWN_TOP, np.int64),
        "rose_shown": np.zeros(SHOWN_TOP, bool),
        "rank_all": np.zeros(TOP, np.int64),
        "score_all": np.full(TOP, np.nan, f),
        "set_of": np.full(TOP, -1, np.int64),
        "removed_all": np.zeros(TOP, np.int64),
        "neighbour_only": np.int64(0),
        "rank_kept": np.zeros(TOP, np.int64),
        "score_kept": np.full(TOP, np.nan, f),
        "kept_tracks": np.zeros(TOP, np.int64),
        "rank_content": np.zeros((GC, TOP), np.int64),
        "score_content": np.full((GC, TOP), np.nan, f),
        "units_content": np.zeros(GC, np.int64),
        "rose_content": np.zeros((GC, TOP), bool),
        "nodata_content": np.zeros((GC, TOP), bool),
        "rank_audio": np.zeros(TOP, np.int64),
        "score_audio": np.full(TOP, np.nan, f),
        "rank_list": np.zeros(TOP, np.int64),
        "score_list": np.full(TOP, np.nan, f),
        "removed_list": np.int64(0),
        "units_list": np.zeros(GC, np.int64),
        "audio_list": np.bool_(False),
        "plans": np.int64(0),
        "full_scans": np.int64(0),
        "checked": np.int64(0),
    }


def assemble(c, L, plans):
    """The fixed-shape arrays of one listener from its passes."""
    GC = len(c["g_content"])
    gpos = {g: j for j, g in enumerate(c["g_content"])}
    out = empty_out(GC)
    base = plans[0].res
    out["rank_base"][:] = base["rank"][:TOP]
    out["score_base"][:] = base["score"][:TOP]
    out["rank_pick"][:] = base["rank"][TOP:]
    full, checked = 0, 0
    for p in plans:
        r = p.res
        if r is not None:
            full += int(not r["bounded"])
            checked += int(r["checked"])
        if p.kind == "base":
            continue
        rank = np.full(len(p.pos), NO_HISTORY, np.int64) if r is None else r["rank"]
        score = np.full(len(p.pos), np.nan) if r is None else r["score"]
        if p.kind == "artist_shown":
            i = int(p.pos[0])
            out["rank_shown"][i], out["score_shown"][i] = rank[0], score[0]
            out["removed_shown"][i] = p.removed
            if r is not None and SAME_SRC in r["conc"]:
                v1, h1 = r["conc"][SAME_SRC]
                v0, h0 = base["conc"][SAME_SRC]
                out["rose_shown"][i] = bool(h1[0] and h0[i] and v1[0] > v0[i])
        elif p.kind in ("artist_all", "artist_kept"):
            tag = "all" if p.kind == "artist_all" else "kept"
            out[f"rank_{tag}"][p.pos] = rank
            out[f"score_{tag}"][p.pos] = score
            if tag == "all":
                out["set_of"][p.pos] = p.index
                out["removed_all"][p.pos] = p.removed
                out["neighbour_only"] += p.neighbour_only
            else:
                out["kept_tracks"][p.pos] = p.kept
        elif p.kind == "content":
            j = gpos[p.index]
            out["rank_content"][j, p.pos] = rank
            out["score_content"][j, p.pos] = score
            out["units_content"][j] = p.removed
            if r is not None:
                (src,) = p.units
                v1, h1 = r["conc"][src]
                v0, h0 = base["conc"][src]
                out["nodata_content"][j, p.pos] = ~h1
                out["rose_content"][j, p.pos] = (
                    h1 & h0[p.pos] & (np.nan_to_num(v1) > np.nan_to_num(v0[p.pos]))
                )
        elif p.kind == "audio":
            out["rank_audio"][p.pos] = rank
            out["score_audio"][p.pos] = score
        elif p.kind == "list":
            out["rank_list"][p.pos] = rank
            out["score_list"][p.pos] = score
            out["removed_list"] = np.int64(p.removed)
            for src, U in p.units.items():
                g = next(
                    g
                    for g in c["g_content"]
                    if c["terms"][c["groups"][g][1][0]].source == src
                )
                out["units_list"][gpos[g]] = len(U)
            out["audio_list"] = np.bool_(p.audio)
    out["plans"] = np.int64(len(plans))
    out["full_scans"] = np.int64(full)
    out["checked"] = np.int64(checked)
    return out


def listener(c, k, dense=False):
    """Every pass of P14 listener k (module docstring): fixed-shape arrays.
    dense: the dense reference route instead of the scans."""
    L = listener_inputs(c, k)
    plans = plans_of(c, L)
    route = dense_plan if dense else scan_plan
    for p in plans:
        if p.state is not None:
            p.res = route(c, L, p)
    L.pop("pre", None)
    L.pop("X", None)
    check_base(c, L, plans[0].res)
    check_sums(c, L, plans)
    return assemble(c, L, plans)


# The pass over the listeners
_E = {}  # the pass context, set before a fork


def _task(task):
    lo, hi = task
    with threadpool_limits(1):  # many workers, one BLAS thread each
        outs = [listener(_E, int(_E["order"][j])) for j in range(lo, hi)]
    return {key: np.stack([o[key] for o in outs]) for key in outs[0]}


def _same(a, b):
    return all(
        np.array_equal(
            np.asarray(a[k]),
            np.asarray(b[k]),
            equal_nan=np.asarray(a[k]).dtype.kind == "f",
        )
        for k in a
    )


def workers_for(st, space, users, u_indptr, u_indices, n_slots, say):
    """P13's worker setting (memory-checked for the co-listening gathers),
    capped so that each worker's two co-listening sums over the catalogue,
    one range of concepts and the passes' profiles fit in the free
    memory."""
    with cold.relabel_logs(WORK_NAME, COMMAND):
        w = st._selection_workers(space, users, u_indptr, u_indices, n_slots)
    extra = 16 * n_slots + 160 * CHUNK + (128 << 20)
    mem = s13._memory()
    if mem is None:
        return w
    fit = max(1, int((mem[0] - s13.MEMORY_RESERVE) // extra))
    if fit < w:
        say(
            f"{WORK_NAME}: a worker needs about {extra / 2**30:.2f} GB, so "
            f"{fit} workers instead of {w}"
        )
    return max(1, min(w, fit))


def removal_pass(st, ctx, work, key, say, workers=None):
    """The pass over the listeners (module docstring): the rule of the
    listeners, then per-listener arrays sorted by P14's listener index.
    Blocks are saved as they finish, so a stopped run continues with the
    blocks still missing. Returns (rule, P14 indices, arrays)."""
    users = ctx["users"]
    n_all = len(users)
    order = np.random.default_rng(ORDER_SEED).permutation(n_all)
    n1 = min(SAMPLE_SIZE, n_all)
    tasks = [(lo, min(lo + BLOCK, n1)) for lo in range(0, n1, BLOCK)] + [
        (lo, min(lo + BLOCK, n_all)) for lo in range(n1, n_all, BLOCK)
    ]
    first = [i for i, t in enumerate(tasks) if t[1] <= n1]
    rest = [i for i, t in enumerate(tasks) if t[0] >= n1]
    pdir = work / "pass"
    state_path = pdir / "state.json"
    k = {
        "key": s13._sha(key),
        "users": p14._array_key(users),
        "order_seed": ORDER_SEED,
        "sample_size": SAMPLE_SIZE,
        "hours_limit": HOURS_LIMIT,
        "blocks": [list(t) for t in tasks],
        "settings": [TOP, SHOWN_TOP, EVIDENCE_K, CHUNK, CHECK_STRIDE],
    }
    state = load_json(state_path) or {}
    if state.get("k") != k:
        pdir.mkdir(parents=True, exist_ok=True)
        for f in pdir.glob("part_*.npz"):
            f.unlink()
        state = {"k": k, "parts": {}, "rule": None}
        atomic_json_write(state, state_path)

    def part(i):
        return pdir / f"part_{i:05d}.npz"

    def pending(idx):
        return [
            i
            for i in idx
            if not (
                str(i) in state["parts"]
                and part(i).exists()
                and file_sha256(part(i)) == state["parts"][str(i)]
            )
        ]

    run = {"workers": workers}

    def probe(left):
        take = np.unique(
            np.round(np.linspace(0, len(left) - 1, min(PROBE_USERS, len(left)))).astype(
                np.int64
            )
        )
        out, secs = {}, []
        with threadpool_limits(1):
            for j, i in enumerate(take, start=1):
                kk = int(order[int(left[i])])
                t1 = time.time()
                out[int(left[i])] = listener(_E, kk)
                secs.append(time.time() - t1)
                say(
                    f"{WORK_NAME}: probe listener {j} of {len(take)}: "
                    f"{secs[-1]:.2f} s for {int(out[int(left[i])]['plans'])} "
                    f"passes"
                )
        return out, float(np.mean(secs))

    def par():
        return max(
            1.0,
            min(
                float(run["workers"]),
                s13b.PASS_SPEEDUP_PER_CORE * s13._physical_cores(),
            ),
        )

    def go(idx, label):
        todo = pending(idx)
        if not todo:
            return 0, 0.0
        if len(todo) < len(idx):
            say(
                f"{WORK_NAME}: {label}: resume, {len(idx) - len(todo)} of "
                f"{len(idx)} blocks done"
            )
        left = np.concatenate([np.arange(*tasks[i]) for i in todo])
        if run["workers"] is None:
            run["workers"] = workers_for(
                st,
                ctx["space"],
                users[order[left]],
                ctx["u_indptr"],
                ctx["u_indices"],
                ctx["space"].n_slots,
                say,
            )
        pr, mean = probe(left)
        say(
            f"{WORK_NAME}: probe mean {mean:.2f} s per listener in one "
            f"process. Rough estimate for {len(left):,} listeners on "
            f"{run['workers']} workers, with a parallel speedup of "
            f"{par():.1f}: {_clock(mean * len(left) / par())}. The progress "
            f"lines below measure the real rate"
        )
        prog = Progress(
            f"{WORK_NAME}: {label}, {run['workers']} workers",
            len(left),
            say,
            unit=" listeners",
        )
        t0 = time.time()
        with s13._Pool(run["workers"]) as pool:
            for j, out in pool.unordered(_task, [tasks[i] for i in todo]):
                i = todo[j]
                s13._atomic_savez(part(i), **out)
                state["parts"][str(i)] = file_sha256(part(i))
                atomic_json_write(state, state_path)
                prog.advance(tasks[i][1] - tasks[i][0])
        secs = time.time() - t0
        prog.finish()
        starts = np.array([t[0] for t in tasks])
        for pos, res in pr.items():
            i = int(np.searchsorted(starts, pos, side="right") - 1)
            with np.load(part(i)) as z:
                stored = {key_: z[key_][pos - tasks[i][0]] for key_ in res}
            if not _same(res, stored):
                raise ValueError(
                    f"eval listener "
                    f"{int(users[order[pos]])}: the pool's "
                    f"results differ from the probe's in this "
                    f"process"
                )
        return len(left), secs

    _E.update(ctx, order=order)
    try:
        done1, secs1 = go(first, f"the first {n1:,} listeners of the " f"random order")
        if state.get("rule") is None:
            if n_all <= n1:
                rule = "all"
                say(
                    f"{WORK_NAME}: all {n_all:,} eval listeners lie in the "
                    f"first {SAMPLE_SIZE:,}"
                )
            else:
                if done1:
                    pred = secs1 / done1 * n_all
                    how = (
                        f"the first {n1:,} listeners ran at "
                        f"{done1 / secs1 * 3600:,.0f} listeners per hour"
                    )
                else:
                    left = np.concatenate([np.arange(*tasks[i]) for i in rest])
                    if run["workers"] is None:
                        run["workers"] = s13.SELECTION_WORKERS
                    _, mean = probe(left)
                    pred = mean * n_all / par()
                    how = f"the probe took {mean:.2f} s per listener"
                rule = "all" if pred <= HOURS_LIMIT * 3600.0 else "sample"
                say(
                    f"{WORK_NAME}: {how}, so all {n_all:,} eval listeners "
                    f"would take about {_clock(pred)}. The limit is "
                    f"{HOURS_LIMIT:g} hours, so "
                    + (
                        "every eval listener runs"
                        if rule == "all"
                        else f"the stage keeps the first {n1:,} of the random " f"order"
                    )
                )
            state["rule"] = rule
            atomic_json_write(state, state_path)
        if state["rule"] == "all":
            go(rest, f"the other {n_all - n1:,} listeners")
    finally:
        _E.clear()
    use = first + (rest if state["rule"] == "all" else [])
    out = load_parts([part(i) for i in use])
    n_used = int(sum(tasks[i][1] - tasks[i][0] for i in use))
    if len(out["plans"]) != n_used:
        raise ValueError("the pass blocks do not cover the listeners")
    idx = order[:n_used]
    srt = np.argsort(idx, kind="stable")
    return (
        state["rule"],
        idx[srt].astype(np.int64),
        {key_: v[srt] for key_, v in out.items()},
    )


def load_parts(paths):
    """The arrays of the saved blocks, concatenated in order. Each file is
    read and closed before the next one opens, so the open-file limit of
    the shell does not bound the number of blocks."""
    out = {}
    for path in paths:
        with np.load(path) as z:
            for key_ in z.files:
                out.setdefault(key_, []).append(np.asarray(z[key_]))
    return {key_: np.concatenate(v) for key_, v in out.items()}


def chosen(rule, n_all):
    """The P14 indices of the listeners under the rule, sorted."""
    if rule == "all":
        return np.arange(n_all, dtype=np.int64)
    order = np.random.default_rng(ORDER_SEED).permutation(n_all)
    return np.sort(order[: min(SAMPLE_SIZE, n_all)]).astype(np.int64)


def verify_dense(ctx, idx, P, say, n_verify):
    """n_verify listeners again through the dense reference: every array
    of the pass except the route's counts."""
    rng = np.random.default_rng(VERIFY_SEED)
    take = np.sort(rng.choice(len(idx), size=min(n_verify, len(idx)), replace=False))
    prog = Progress(
        f"{WORK_NAME}: the passes against the dense reference",
        len(take),
        say,
        unit=" listeners",
    )
    for j in take:
        res = listener(ctx, int(idx[j]), dense=True)
        got = {key: P[key][j] for key in res if key not in SKIP_DENSE}
        want = {key: v for key, v in res.items() if key not in SKIP_DENSE}
        if not _same(want, got):
            bad = [key for key in want if not _same({key: want[key]}, {key: got[key]})]
            raise ValueError(
                f"listener {int(ctx['users'][idx[j]])}: the "
                f"passes differ from the dense reference in "
                f"{bad}"
            )
        prog.advance(1)
    prog.finish()
    return {"listeners": int(len(take))}


# Results


def quantiles(r):
    r = np.asarray(r, dtype=np.int64)
    if not len(r):
        return {f"p{q}": None for q in QUANTILES}
    v = np.percentile(r, QUANTILES, method="inverted_cdf")
    return {f"p{q}": int(x) for q, x in zip(QUANTILES, v)}


def share(hit, mask):
    """The share of hit over mask with its listener-cluster interval, cut
    at 0 and 1."""
    out = gkd.cluster_mean(np.asarray(hit, dtype=np.float64), mask)
    out = {
        key: (p14._num(v) if key in ("mean", "se", "lo", "hi") else int(v))
        for key, v in out.items()
    }
    if out["lo"] is not None:
        out["lo"] = max(0.0, out["lo"])
    if out["hi"] is not None:
        out["hi"] = min(1.0, out["hi"])
    return out


def block(ranks, measure, x=None):
    """The measures of one counterfactual from its (n, m) rank codes."""
    r = np.asarray(ranks, dtype=np.int64)
    tested = r >= 1
    hit = (r > TOP) if measure == "leave" else (r <= TOP)
    out = {
        "measure": measure,
        "tested": int(tested.sum()),
        "listeners": int(tested.any(axis=1).sum()),
        "no_history": int((r == NO_HISTORY).sum()),
        "share": share(hit, tested),
        "rank": quantiles(r[tested]),
    }
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        if tested.any():
            q1, q2 = (float(v) for v in np.quantile(x[tested], [1 / 3, 2 / 3]))
            lev = np.where(x <= q1, 0, np.where(x <= q2, 1, 2))
        else:
            q1 = q2 = None
            lev = np.zeros(x.shape, dtype=np.int64)
        rows = []
        for j, name in enumerate(TERTILES):
            m = tested & (lev == j)
            rows.append(
                {
                    "level": name,
                    "tested": int(m.sum()),
                    "x_min": int(x[m].min()) if m.any() else None,
                    "x_max": int(x[m].max()) if m.any() else None,
                    "share": share(hit, m),
                    "rank": quantiles(r[m]),
                }
            )
        out["tertiles"] = {"cuts": [q1, q2], "levels": rows}
    return out


def summarize(A, glob):
    """Every result from the stored arrays: a pure function, so the
    read-back recomputes it. glob: the reason names and P14's counts of
    history tracks by the artist."""
    x = np.asarray(glob["credit_tracks"], dtype=np.int64)
    res = {}
    res["artist_shown"] = block(A["rank_shown"], "leave", x[:, :SHOWN_TOP])
    t1 = A["rank_shown"] >= 1
    res["artist_shown"]["value_rose"] = (
        p14._num(A["rose_shown"][t1].mean()) if t1.any() else None
    )
    res["artist_shown"]["shown_without_colistening"] = int(
        glob["shown_top3_without_colistening"]
    )
    res["artist_all"] = block(A["rank_all"], "leave", x)
    res["artist_all"]["neighbour_only_tracks"] = int(np.sum(A["neighbour_only"]))
    res["artist_kept"] = block(A["rank_kept"], "stay")
    res["content"] = {}
    for j, name in enumerate(glob["content"]):
        r = A["rank_content"][:, j, :]
        b = block(r, "leave")
        t = r >= 1
        b["value_rose"] = (
            p14._num(A["rose_content"][:, j, :][t].mean()) if t.any() else None
        )
        b["no_data"] = (
            p14._num(A["nodata_content"][:, j, :][t].mean()) if t.any() else None
        )
        b["units_mean"] = (
            p14._num(A["units_content"][:, j][t.any(axis=1)].mean())
            if t.any()
            else None
        )
        res["content"][name] = b
    n = len(A["rank_list"])
    res["content_all"] = block(A["rank_content"].reshape(n, -1), "leave")
    res["audio"] = block(A["rank_audio"], "leave")
    res["list"] = block(A["rank_list"], "leave")
    ran = np.all(A["rank_list"] >= 1, axis=1)
    kept = (A["rank_list"] <= TOP).sum(axis=1)[ran]
    kt = by.interval(kept) if len(kept) > 1 else None
    if kt is not None and kt["lo"] is not None:
        kt["lo"], kt["hi"] = max(0.0, kt["lo"]), min(float(TOP), kt["hi"])
    res["list"]["kept_in_top"] = kt
    res["list"]["removed_tracks_mean"] = (
        p14._num(A["removed_list"][ran].mean()) if ran.any() else None
    )
    res["passes"] = {
        "per_listener": p14._num(np.mean(A["plans"])),
        "full_scans_per_listener": p14._num(np.mean(A["full_scans"])),
    }
    return res


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
        "p13d_model_set.npz",
        "train_manifest.json",
        "csr_meta.json",
        "id_spaces.json",
    ):
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    for stage in ("p4_features", "p6_als", "p9b_colisten", "p25_train"):
        try:
            out[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            out[f"marker:{stage}"] = None
    return out


P14_KEYS = (
    "user_id",
    "picks",
    "history",
    "tables",
    "top_id",
    "top_score",
    "top_val",
    "top_has",
    "shown",
    "ev_count",
    "ev_id",
    "ev_val",
    "ev_credit",
    "ev_credit_tracks",
    "rank_gam",
)


def load_p14(d):
    """p14.json and the arrays of p14.npz that this stage reads."""
    doc = load_json(d / p14.OUT_FILES[0])
    if doc is None or doc.get("version") != p14.VERSION:
        raise ValueError(
            f"{p14.OUT_FILES[0]} is missing or has another "
            f"version: run python3 -m pipeline.runner --only "
            f"p14_eval first"
        )
    with np.load(d / p14.OUT_FILES[1]) as z:
        return doc, {k: np.asarray(z[k]) for k in P14_KEYS}


def glob_of(z14, idx, names, kinds, ga):
    """The inputs of summarize() besides the pass's arrays."""
    content = [names[g] for g, kd in enumerate(kinds) if kd == "content"]
    sh = np.asarray(z14["shown"], dtype=bool)
    if ga is None:
        n_wo = 0
    else:
        art = sh[ga][idx, :SHOWN_TOP]
        n_wo = int(
            (art & (np.asarray(z14["ev_count"])[idx, :SHOWN_TOP, ga] == 0)).sum()
        )
    return {
        "credit_tracks": np.asarray(z14["ev_credit_tracks"])[idx],
        "content": content,
        "shown_top3_without_colistening": n_wo,
    }


def run(
    st, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None, workers=None
):
    """The stage (module docstring). st: a StageP13b2 whose loaders give the
    item side, the CSR and the slots. Returns the output doc."""
    try:
        return _run(st, Path(model_dir), Path(work), say, n_verify, inputs, workers)
    finally:
        _E.clear()


def _run(st, d, work, say, n_verify, inputs, workers):
    t0 = time.time()
    work.mkdir(parents=True, exist_ok=True)
    say(f"{WORK_NAME}: {VERSION}, rules declared {DECLARED}. Saved work: " f"{work}")
    doc14, z14 = load_p14(d)
    base = st._base_context()
    n_slots, _ = st._id_slots()
    G = p14.load_models(d, base)
    terms = G["terms"]
    if not np.array_equal(z14["tables"], G["tables"]):
        raise ValueError(
            "p14.npz holds other tables than the models of " "p14.load_models"
        )
    T = np.asarray(G["tables"][:1], dtype=np.float64)
    users = np.asarray(z14["user_id"], dtype=np.int64)
    if (
        len(users) < 2
        or np.any(np.diff(users) <= 0)
        or int(doc14["data"]["listeners"]) != len(users)
    ):
        raise ValueError("P14's listeners are not sorted, distinct and " "complete")
    u_arrays = st._csr()
    u_indptr = u_arrays[0]
    if int(users.max()) >= len(u_indptr) - 1 or int(users.min()) < 1:
        raise ValueError("an eval listener lies outside the user CSR")
    if not np.array_equal(
        np.asarray(u_indptr[users + 1], np.int64)
        - np.asarray(u_indptr[users], np.int64),
        np.asarray(z14["history"], np.int64),
    ):
        raise ValueError("p14.npz's history lengths differ from the CSR")
    groups = [
        (name, [[t.name for t in terms].index(n) for n in g])
        for name, g in pc.reason_groups(G["M"]["kept"])
    ]
    names = [name for name, _ in groups]
    if names != [r["reason"] for r in doc14["reasons"]["rows"]]:
        raise ValueError("the model's reason groups differ from P14's")
    say(
        f"{WORK_NAME}: {len(users):,} eval listeners of P14, reasons "
        f"{', '.join(names)}. Loading the item side (no progress lines in "
        f"this step)"
    )
    t1 = time.time()
    space = st._space(st._sources(n_slots))
    scorer = ct.CatalogScorer(terms, T, space, chunk=CHUNK)
    if scorer.tilt is not None:
        raise ValueError("the pass supports no term with a multiplier")
    ctx = context(
        scorer, space, terms, T, groups, z14, u_arrays, chunk=CHUNK, stride=CHECK_STRIDE
    )
    say(
        f"{WORK_NAME}: item side and scorer ready in "
        f"{_clock(time.time() - t1)}. The bound U for slots without artist "
        f"evidence: {ctx['bound']:+.4f}"
    )
    key = {
        "version": PASS_VERSION,
        "p13a_params": s13._params(),
        "tables": s13._array_sha(T),
        "p14": p14._array_key(*[z14[k] for k in P14_KEYS]),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, gam, p14)
        },
    }
    t2 = time.time()
    rule, idx, P = removal_pass(st, ctx, work, key, say, workers)
    say(
        f"{WORK_NAME}: pass done in {_clock(time.time() - t2)}, "
        f"{len(idx):,} listeners. With nothing removed, every listener's "
        f"top {TOP}, concepts and pick ranks equal P14's"
    )
    chk_dense = verify_dense(ctx, idx, P, say, n_verify)
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    glob = glob_of(z14, idx, names, ctx["kinds"], ctx["g_artist"])
    res = summarize(P, glob)
    out = p14._norm(
        {
            "version": VERSION,
            "declared": DECLARED,
            "protocol": protocol(),
            "data": {
                "eval_listeners": int(len(users)),
                "listeners": int(len(idx)),
                "rule": rule,
                "n_slots": int(n_slots),
                "reasons": names,
                "kinds": ctx["kinds"],
                "content": glob["content"],
                "not_removed": [
                    n for n, kd in zip(names, ctx["kinds"]) if kd == "other"
                ],
                "bound": ctx["bound"],
            },
            "models": {"final": G["rec"]["fit_name"], "terms": [t.name for t in terms]},
            "results": res,
            "check": {
                "base_equals_p14": True,
                "evidence_units_equal_p14": True,
                "evidence_sums": True,
                "scores_equal_reference": True,
                "slots_checked": int(np.sum(P["checked"])),
                "dense": chk_dense,
                "probe_equal": True,
            },
            "inputs": dict(inputs or {}),
        }
    )
    arrays = {
        "user_id": users[idx],
        "p14_index": idx,
        "picks": z14["picks"][idx],
        "top": np.asarray(z14["top_id"])[idx, 0, :TOP],
        "credit_tracks": glob["credit_tracks"],
    }
    arrays.update(P)
    by.write_npz(d / OUT_FILES[1], arrays)
    atomic_json_write(out, d / OUT_FILES[0])
    write_report(out, d / "reports" / REPORT_NAME)
    write_tables(out, d / FIG_DIR)
    say(f"{WORK_NAME}: reading the outputs back")
    check_outputs(d)
    r2 = res["artist_all"]["share"]["mean"]
    say(
        f"{WORK_NAME}: all artist evidence removed: "
        f"{'n/a' if r2 is None else f'{r2:.1%}'} of "
        f"{res['artist_all']['tested']:,} tracks leave the top {TOP}. Done "
        f"in {_clock(time.time() - t0)}"
    )
    return out


def check_outputs(d):
    """Read the outputs back: the listeners against P14 and the rule, the
    base ranks, every result from the stored arrays, and the tables from
    the results."""
    d = Path(d)
    doc = load_json(d / OUT_FILES[0])
    if doc is None:
        raise ValueError(f"{OUT_FILES[0]} is missing")
    _, z14 = load_p14(d)
    z = gkd.load_npz(d / OUT_FILES[1])
    idx = np.asarray(z["p14_index"], dtype=np.int64)
    n_all = len(z14["user_id"])
    if int(doc["data"]["eval_listeners"]) != n_all or not np.array_equal(
        idx, chosen(doc["data"]["rule"], n_all)
    ):
        raise ValueError("the stored listeners do not follow the rule")
    if not (
        np.array_equal(z["user_id"], z14["user_id"][idx])
        and np.array_equal(z["picks"], z14["picks"][idx])
        and np.array_equal(z["top"], z14["top_id"][idx, 0, :TOP])
        and np.array_equal(z["credit_tracks"], z14["ev_credit_tracks"][idx])
    ):
        raise ValueError("the stored listeners, picks or top lists are not " "P14's")
    if not (
        np.all(z["rank_base"] == np.arange(1, TOP + 1))
        and np.array_equal(z["rank_pick"], z14["rank_gam"][idx])
    ):
        raise ValueError("the stored base ranks differ from P14's")
    for key in [k for k in z if k.startswith("rank_")]:
        if np.any(z[key] < NO_HISTORY):
            raise ValueError(f"{key} holds a code below {NO_HISTORY}")
        s_key = "score_" + key[len("rank_") :]
        if s_key in z and not np.array_equal(np.isnan(z[s_key]), z[key] < 1):
            raise ValueError(f"{s_key} has a score where {key} has none")
    names, kinds = doc["data"]["reasons"], doc["data"]["kinds"]
    ga = kinds.index("artist") if "artist" in kinds else None
    glob = glob_of(z14, idx, names, kinds, ga)
    A = {
        k: z[k]
        for k in z
        if k not in ("user_id", "p14_index", "picks", "top", "credit_tracks")
    }
    if p14._norm(summarize(A, glob)) != doc["results"]:
        raise ValueError("the stored results do not follow from the stored " "arrays")
    for name, header, rows in table_rows(doc):
        p = d / FIG_DIR / name
        if not p.exists() or p.read_text(encoding="utf-8") != _csv(rows, header):
            raise ValueError(f"{name} does not follow from the results")
    return True


# Report


def _pct(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _shr(s):
    if s is None or s.get("mean") is None:
        return "n/a"
    if s.get("lo") is None:
        return _pct(s["mean"])
    return f"{_pct(s['mean'])} ({_pct(s['lo'])} to {_pct(s['hi'])})"


def _rk(q, key):
    x = q.get(key)
    return "n/a" if x is None else f"{x:,}"


def rows_of(res, content_labels):
    """(row id, label, measure name, block) per row of the main table."""
    out = []
    for kind in KINDS:
        if kind == "content":
            out.append(
                ("4", KIND_LABELS["content"] + ", all three", res["content_all"])
            )
            for name, b in res["content"].items():
                lab = content_labels.get(name, name)
                out.append((f"4 ({lab})", f"cited {lab} removed", b))
            continue
        out.append((ROW_OF[kind], KIND_LABELS[kind], res[kind]))
    return out


def write_report(out, path):
    res, data, chk = out["results"], out["data"], out["check"]
    lab = {n: REASON_LABELS.get(n, n) for n in data["content"]}
    L = [
        "# Evidence removal: do the reasons point at the evidence?",
        "",
        f"- Rules: the docstring of pipeline/evidence_removal.py, declared "
        f"{out['declared']}. Every number is descriptive: no test.",
        f"- Listeners: {data['listeners']:,} of {data['eval_listeners']:,} "
        f"eval listeners of P14 ("
        + (
            "all of them"
            if data["rule"] == "all"
            else f"the first {out['protocol']['listeners']['sample_size']:,} of a "
            f"random order with seed "
            f"{out['protocol']['listeners']['order_seed']}"
        )
        + f"). Each pass ranks the full catalogue ({data['n_slots'] - 1:,} "
        f"tracks) with the candidates of the full history, ties by item "
        f"id. A track leaves the top {TOP} when its new rank is above "
        f"{TOP}.",
        f"- Model: {out['models']['final']}. Reasons: "
        f"{', '.join(data['reasons'])}. Not removed (no rule): "
        f"{', '.join(data['not_removed']) or 'none'}.",
        f"- Checks: with nothing removed every listener's top {TOP}, "
        f"scores, concepts and pick ranks equal P14's. The evidence units "
        f"equal P14's lists. Removal never raised an evidence sum. "
        f"{chk['slots_checked']:,} sampled slot scores equal the dense "
        f"reference exactly, "
        + (
            f"{chk['dense']['listeners']} listeners equal"
            if chk["dense"]["listeners"] != 1
            else "1 listener equals"
        )
        + " it in every pass, and the results read back.",
        "",
        "## Counterfactuals",
        "",
        "Necessity: the share of tested tracks that leave the top 10. Row "
        "3, sufficiency: the share that stay. 95 percent intervals for "
        "listener clusters. The new rank over the tested tracks: median, "
        "and the 5th and 95th percentiles.",
        "",
        "| row | counterfactual | tracks tested | listeners | measure | "
        "share | new rank, median | 5th to 95th percentile | no history "
        "left |",
        "|---|---|---:|---:|---|---|---:|---|---:|",
    ]
    for rid, label, b in rows_of(res, lab):
        q = b["rank"]
        L.append(
            f"| {rid} | {label} | {b['tested']:,} | "
            f"{b['listeners']:,} | {b['measure']} | {_shr(b['share'])} "
            f"| {_rk(q, 'p50')} | {_rk(q, 'p5')} to {_rk(q, 'p95')} | "
            f"{b['no_history']:,} |"
        )
    a1 = res["artist_shown"]
    L += [
        "",
        f"Row 1 removes up to {out['protocol']['evidence_k']} "
        f"co-listened history tracks, the shown evidence. "
        f"{a1['shown_without_colistening']:,} top-3 tracks show the "
        f"artist reason without a co-listened history track and are not "
        f"tested. Row 4 removes, per reason, the units cited on the whole "
        f"top {TOP}: each track loses its own cited units and those of "
        f"the other tracks.",
        "",
        "## By the number of history tracks by the artist",
        "",
        "Tertiles of P14's count of history tracks by the track's "
        "credited artist, over the tested tracks of each row.",
        "",
        "| row | tertile | history tracks by the artist | tracks tested "
        "| leave the top 10 | new rank, median | 5th to 95th percentile |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for kind in ("artist_shown", "artist_all"):
        for t in res[kind]["tertiles"]["levels"]:
            rng = (
                "n/a"
                if t["x_min"] is None
                else (
                    f"{t['x_min']:,}"
                    if t["x_min"] == t["x_max"]
                    else f"{t['x_min']:,} to {t['x_max']:,}"
                )
            )
            L.append(
                f"| {ROW_OF[kind]} | {t['level']} | {rng} | "
                f"{t['tested']:,} | {_shr(t['share'])} | "
                f"{_rk(t['rank'], 'p50')} | {_rk(t['rank'], 'p5')} to "
                f"{_rk(t['rank'], 'p95')} |"
            )
    lst = res["list"]
    kept = lst["kept_in_top"]
    kept_txt = (
        "n/a"
        if kept is None
        else f"{kept['mean']:.2f} ({kept['lo']:.2f} to {kept['hi']:.2f})"
    )
    rem = lst["removed_tracks_mean"]
    L += [
        "",
        "## The whole list",
        "",
        f"With the evidence of every shown reason removed together, "
        f"{kept_txt} of the original {TOP} tracks stay in the new top "
        f"{TOP} (mean over the listeners). The history loses "
        f"{'n/a' if rem is None else f'{rem:,.1f}'} tracks on average.",
        "",
        "## Concept values after the removal",
        "",
        "Removal lowers the sum of a concept's evidence terms. The value "
        "also divides by the history's total weight or the profile's "
        "norm, which fall too, so it can rise.",
    ]
    L.append(
        f"- row 1, the track's co-listening rose: "
        f"{_pct(a1['value_rose'])} of the tested tracks."
    )
    for name, b in res["content"].items():
        um = "n/a" if b["units_mean"] is None else f"{b['units_mean']:.1f}"
        L.append(
            f"- row 4, {lab[name]}: the concept rose at "
            f"{_pct(b['value_rose'])} and has no data at "
            f"{_pct(b['no_data'])} of the tested tracks. Units removed "
            f"per listener: {um}."
        )
    pa = res["passes"]
    L += [
        "",
        f"Passes per listener: {pa['per_listener']:.2f}, of which "
        f"{pa['full_scans_per_listener']:.2f} score the whole catalogue. "
        f"Same-artist neighbours outside the tracks by the artist: "
        f"{res['artist_all']['neighbour_only_tracks']:,}.",
    ]
    atomic_text_write("\n".join(L).replace(";", ",") + "\n", path)


def _csv(rows, header):
    buf = iolib.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows([["" if x is None else x for x in r] for r in rows])
    return buf.getvalue()


def table_rows(out):
    """The two tables of the figures notebook: (file name, header, rows)."""
    res, data = out["results"], out["data"]
    lab = {n: REASON_LABELS.get(n, n) for n in data["content"]}
    main = []
    for rid, label, b in rows_of(res, lab):
        s, q = b["share"], b["rank"]
        kept = b.get("kept_in_top") or {}
        main.append(
            [
                rid,
                label,
                b["measure"],
                b["tested"],
                b["listeners"],
                b["no_history"],
                s.get("mean"),
                s.get("lo"),
                s.get("hi"),
            ]
            + [q[f"p{x}"] for x in QUANTILES]
            + [kept.get("mean"), kept.get("lo"), kept.get("hi")]
        )
    h_main = (
        [
            "row",
            "counterfactual",
            "measure",
            "tested",
            "listeners",
            "no_history",
            "share_mean",
            "share_lo",
            "share_hi",
        ]
        + [f"rank_p{x}" for x in QUANTILES]
        + ["kept_top10_mean", "kept_top10_lo", "kept_top10_hi"]
    )
    ter = []
    for kind in ("artist_shown", "artist_all"):
        for t in res[kind]["tertiles"]["levels"]:
            s, q = t["share"], t["rank"]
            ter.append(
                [
                    ROW_OF[kind],
                    kind,
                    t["level"],
                    t["x_min"],
                    t["x_max"],
                    t["tested"],
                    s.get("mean"),
                    s.get("lo"),
                    s.get("hi"),
                    q["p5"],
                    q["p50"],
                    q["p95"],
                ]
            )
    h_ter = [
        "row",
        "counterfactual",
        "tertile",
        "history_tracks_min",
        "history_tracks_max",
        "tested",
        "leave_mean",
        "leave_lo",
        "leave_hi",
        "rank_p5",
        "rank_p50",
        "rank_p95",
    ]
    return [(TAB_FILES[0], h_main, main), (TAB_FILES[1], h_ter, ter)]


def write_tables(out, folder):
    for name, header, rows in table_rows(out):
        atomic_text_write(_csv(rows, header), Path(folder) / name)


# Runner stage


class StageEvidenceRemoval(ScriptStage):
    """The evidence-removal test of the explanations as a runner stage."""

    name = "evidence_removal"
    upstream = ["p14_eval", "p14_cold_gam"]

    def __init__(self, model_dir=None, scratch_dir=None):
        super().__init__(model_dir, scratch_dir)
        self.n_verify = VERIFY_USERS
        self.workers = None

    def _stage(self):
        return s13b2.StageP13b2()

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
                self.model_dir,
                self.scratch_dir / WORK_NAME,
                say=log.info,
                n_verify=self.n_verify,
                inputs=self.provenance(),
                workers=self.workers,
            )
        return self.output_info()

    def check(self):
        check_outputs(self.model_dir)

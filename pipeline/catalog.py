"""Full-catalog ranking with P13 model tables, in bounded memory.

P13's selection pass and P14's evaluation rank every item slot for one
listener under the P5 harness rules (pipeline/evaluation.py): the
candidates are all slots except slot 0 and the listener's train items,
held-out picks stay candidates, and equal scores rank by item id,
ascending. Many model configurations share one pass: the concepts of a
listener are computed once, and each configuration adds only table
lookups.

Why not concepts.full_catalog() and a dense sum: full_catalog() computes
audio fit for all 6.6 million slots at once, with about 1.6 GB of
temporary arrays per listener, and a dense sum reads every term of every
slot for every configuration. Here:

1. concept_catalog() computes one concept at a time, audio fit in row
   chunks and only on the tracks with audio. Its values equal
   full_catalog() bit for bit (test_gam.py).
2. CatalogScorer splits each configuration's score into an item part,
   built once for all listeners, and a listener part:
   - the item part: popularity, and for every concept term the value of
     its item-level state ("exactly zero" where the track has data,
     "no data" where it has none, "not applicable" for the audio terms,
     whose shape values are listener-specific).
   - the tilt table value f_1(pop) of each track, multiplied by the
     listener's mainstreaminess.
   - per listener and term, a correction on the tracks whose state
     differs from the item-level state: the tracks on the shape, and the
     tracks with no data for this listener only (a history without tags,
     for example). Each correction adds three entries to its slot's
     sparse row: (1 - w) at the slot, w at the slot after it and -1 at
     the item-level slot. One sparse product of a chunk's rows with the
     (entries, configurations) tables then gives the corrections of
     every configuration at once.
   Most tracks carry no correction for the sparse concepts (artist,
   co-listening), so the pass reads far fewer entries than a dense sum.
3. The ranks follow the P5 rule exactly:
   rank = 1 + #(score > s_pick) + #(score == s_pick, id < pick id).
   The pick's score is computed by the same function on a one-slot range,
   so it equals the entry of its chunk bit for bit (checked every time).
4. ranks_cut() gives the same ranks up to a cut (100 for NDCG@100) and
   cut + 1 beyond it, at a fraction of the cost. It computes the concepts
   of one slot range at a time (concept_range(), equal to
   concept_catalog() on that range bit for bit), scores the ranges that
   hold the picks first, and stops as soon as every pick has more than
   cut - 1 better candidates in every configuration: the counts only
   grow, so the rank is then known to exceed the cut. A listener whose
   picks rank far down stops after a few ranges. Memory per listener
   stays near one range plus two co-listening vectors.

The score is the sum of the term values, the same numbers that the
explanation cites. The addition order here (item part, tilt, corrections)
differs from a plain sum in term order, so the two can differ in the last
bits (about 1e-15). verify() compares this scorer with an independent
dense reference within a stated tolerance and checks every rank.
"""

import numpy as np
import scipy.sparse as sp

from . import concepts as cn
from .gam import MISS, NA, ZERO, diff_table, encode, encode_continuous, pwl_value

AUDIO_CHUNK = 1 << 18  # rows of audio fit per step, about 21 MB
SCORE_CHUNK = 1 << 20  # item slots per scoring step
RANK_CHUNK = 1 << 16  # item slots per step of ranks_cut()


def concept_catalog(state, space, concept, audio_chunk=AUDIO_CHUNK, audio_rows=None):
    """(value, has) of one concept of concepts.full_catalog() for every
    item slot. The same expressions as full_catalog(), with audio fit in
    row chunks through concepts._audio_fit(), which works row by row.
    audio_rows: the sorted slots with any audio dimension. Audio fit is
    then computed on those rows only, and every other slot keeps NaN and
    no data, which is what full_catalog() gives a slot without audio."""
    if concept in cn.TERM_CONCEPTS:
        name = cn.TERM_CONCEPTS[concept]
        m = space.term_mats[name]
        coef = space.idf[name] ** 2 * state.term_share[name]
        numer = m @ coef
        has = space.has_terms[name] & (state.term_norm[name] > 0)
        denom = np.where(has, space.item_norm[name] * state.term_norm[name], 1.0)
        return np.where(has, numer / denom, np.nan), has
    if concept in cn.CREDIT_CONCEPTS:
        name = cn.CREDIT_CONCEPTS[concept]
        m = space.credit_mats[name]
        dense = np.zeros(m.shape[1], dtype=np.float64)
        dense[state.credit_cols[name]] = state.credit_share[name]
        best = cn._row_max(dense[m.indices], m.indptr)
        has = space.has_credit[name]
        return np.where(has, best, np.nan), has
    if concept in cn.COLISTEN_CONCEPTS:
        if not space.transposed:
            raise ValueError(
                "co-listening over the catalog needs an "
                "ItemSide built with transposed=True"
            )
        name = cn.COLISTEN_CONCEPTS[concept]
        numer = np.asarray(
            space.colisten_t[name][state.items].T @ state.weights, dtype=np.float64
        ).ravel()
        has = space.has_colisten[name]
        return np.where(has, numer / state.total, np.nan), has
    if concept == "audio_fit":
        n = space.n_slots
        value = np.full(n, np.nan, dtype=np.float64)
        has = np.zeros(n, dtype=bool)
        if audio_rows is None:
            for a in range(0, n, audio_chunk):
                b = min(a + audio_chunk, n)
                value[a:b], has[a:b] = cn._audio_fit(state, space, slice(a, b))
        else:
            for a in range(0, len(audio_rows), audio_chunk):
                rows = audio_rows[a : a + audio_chunk]
                value[rows], has[rows] = cn._audio_fit(state, space, rows)
        return value, has
    if concept == "popularity":
        has = np.ones(space.n_slots, dtype=bool)
        has[0] = False
        return np.where(has, space.popularity, np.nan), has
    raise KeyError(concept)


def item_has(space, concept):
    """The item-level data flag of a concept: True where the track itself
    has the data. A listener can only lack more (a history without tags
    or audio), never add any."""
    if concept in cn.TERM_CONCEPTS:
        return space.has_terms[cn.TERM_CONCEPTS[concept]]
    if concept in cn.CREDIT_CONCEPTS:
        return space.has_credit[cn.CREDIT_CONCEPTS[concept]]
    if concept in cn.COLISTEN_CONCEPTS:
        return space.has_colisten[cn.COLISTEN_CONCEPTS[concept]]
    if concept == "audio_fit":
        return space.audio_mask.any(axis=1)
    if concept == "popularity":
        has = np.ones(space.n_slots, dtype=bool)
        has[0] = False
        return has
    raise KeyError(concept)


def base_slots(term, has_item, tier):
    """The item-level slot of a term for every track: "not applicable"
    outside the term's tiers, "no data" (or "not applicable" without a
    missing state) where the track has no data, "exactly zero" (or "not
    applicable" without a zero state) where it has data."""
    K = term.K
    n = len(has_item)
    slot = np.full(n, K + NA, dtype=np.int16)
    applies = (
        np.ones(n, dtype=bool)
        if term.tiers is None
        else np.isin(tier, np.asarray(term.tiers))
    )
    if term.missing_state:
        slot[applies & ~has_item] = K + MISS
    if term.zero_state:
        slot[applies & has_item] = K + ZERO
    return slot


def _merge_sorted(a, b):
    """Positions that merge two disjoint sorted index arrays: returns
    (order positions of a, of b) in the merged array."""
    pa = np.arange(len(a)) + np.searchsorted(b, a)
    pb = np.arange(len(b)) + np.searchsorted(a, b)
    return pa, pb


class CatalogScorer:
    """Scores and ranks of every item slot for many configurations.

    terms: the model's terms (gam.Term), in order. tables: (C, L) flat
    tables, one row per configuration (gam.Layout.tables). A term whose
    source is "popularity" and that has no multiplier is item-only. A
    term with the "mainstreaminess" multiplier is the tilt. Every other
    term is a listener term. Scores are held as (slots, C) arrays, so one
    slot's configurations sit together in memory."""

    def __init__(self, terms, tables, space, chunk=SCORE_CHUNK):
        self.terms = tuple(terms)
        self.T = np.atleast_2d(np.asarray(tables, dtype=np.float64))
        self.C, self.L = self.T.shape
        self.offsets = np.cumsum([0] + [t.size for t in self.terms])[:-1]
        self.D = diff_table(self.T, self.terms, self.offsets)
        self.TT = np.ascontiguousarray(self.T.T)
        self.n = space.n_slots
        self.chunk = int(chunk)
        self.tier = space.tier
        self.audio_rows = np.flatnonzero(space.audio_mask.any(axis=1))
        self.base = np.zeros((self.n, self.C), dtype=np.float64)
        self.tilt = None
        self.user_terms, self.base_slot, self.applies = [], {}, {}
        self.user_sources = []
        self.item_has = {}
        for k, (t, off) in enumerate(zip(self.terms, self.offsets)):
            if t.source not in self.item_has:
                self.item_has[t.source] = item_has(space, t.source)
            h_item = self.item_has[t.source]
            if t.multiplier is not None:
                if t.source != "popularity" or self.tilt is not None:
                    raise ValueError(
                        "the scorer supports one multiplier " "term, on popularity"
                    )
                slot, w = encode(t, space.popularity, h_item, self.tier)
                self.tilt = self._item_values(slot, w, int(off))
            elif t.source == "popularity" and t.tiers is None:
                slot, w = encode(t, space.popularity, h_item, self.tier)
                self.base += self._item_values(slot, w, int(off))
            else:
                bs = base_slots(t, h_item, self.tier)
                self.base_slot[k] = bs
                self.base += self.TT[int(off) + bs.astype(np.int64)]
                self.applies[k] = (
                    None if t.tiers is None else np.isin(self.tier, np.asarray(t.tiers))
                )
                self.user_terms.append(k)
                if t.source not in self.user_sources:
                    self.user_sources.append(t.source)

    def _item_values(self, slot, w, off):
        idx = off + slot.astype(np.int64)
        out = np.empty((len(idx), self.C), dtype=np.float64)
        for c in range(self.C):
            out[:, c] = pwl_value(self.T[c], self.D[c], idx, w)
        return out

    # Per listener

    def listener(self, state, space, m_u):
        """The corrections of one listener over the whole catalog: for
        each listener term, the tracks whose slot differs from the
        item-level slot, with the slot, the position w and the item-level
        slot. Concept arrays are freed term by term, so memory stays near
        one concept."""
        corr = {}
        src, cached = None, None
        for k in self.user_terms:
            t = self.terms[k]
            if t.source != src:
                cached = None
                cached = concept_catalog(
                    state, space, t.source, audio_rows=self.audio_rows
                )
                src = t.source
            corr[k] = self._corrections(k, cached[0], cached[1], 0, self.n)
        return {"m_u": float(m_u), "corr": corr}

    def _corrections(self, k, value, has, a, b):
        """The corrections of listener term k on slots a..b-1, from the
        concept's (value, has) on those slots: (rows, slot, w, item-level
        slot) of the slots whose slot differs from the item-level slot,
        with rows counted from a (int32, increasing). Every step works slot
        by slot, so a range gives the entries of the whole catalog on it."""
        t = self.terms[k]
        h_item = self.item_has[t.source][a:b]
        if np.any(has & ~h_item):
            raise ValueError(
                f"{t.source}: the listener has data where the " f"track has none"
            )
        ap = self.applies[k]
        if ap is not None:
            ap = ap[a:b]
        has_a = has if ap is None else has & ap
        cont = has_a & (value != 0.0) if t.zero_state else has_a
        ic = np.flatnonzero(cont)
        sc, wc = encode_continuous(t.knot_array, value[ic])
        lack = h_item & ~has
        im = np.flatnonzero(lack if ap is None else lack & ap)
        K = t.K
        sm = np.full(len(im), K + (MISS if t.missing_state else NA), dtype=np.int16)
        if len(im):
            pa, pb = _merge_sorted(ic, im)
            n_all = len(ic) + len(im)
            items = np.empty(n_all, dtype=np.int64)
            slot = np.empty(n_all, dtype=np.int16)
            w = np.zeros(n_all, dtype=np.float64)
            items[pa], items[pb] = ic, im
            slot[pa], slot[pb] = sc, sm
            w[pa] = wc
        else:
            items, slot, w = ic, sc, wc
        bslot = self.base_slot[k][a:b][items]
        keep = slot != bslot
        if not keep.all():
            items, slot, w, bslot = (items[keep], slot[keep], w[keep], bslot[keep])
        return items.astype(np.int32), slot, w, bslot

    def _chunk_matrix(self, corr, a, b):
        """The corrections of slots a..b-1 as one sparse (b - a, L) matrix:
        each corrected slot holds, for each term in order, (1 - w) at its
        slot, w at the next slot and -1 at its item-level slot. Slots
        without a correction have empty rows. None when no slot has one.
        corr: {term: (rows, slot, w, item-level slot)} with increasing
        rows on the same scale as a and b. Every correction adds one
        triple, so the entries are placed as rows of an (n, 3) array."""
        n_c = b - a
        parts = []
        count = np.zeros(n_c + 1, dtype=np.int64)
        for k in self.user_terms:
            items, slot, w, bslot = corr[k]
            lo, hi = np.searchsorted(items, (a, b))
            if hi <= lo:
                continue
            rows = items[lo:hi].astype(np.int64) - a
            off = int(self.offsets[k])
            cc = np.empty((hi - lo, 3), dtype=np.int32)
            cc[:, 0] = slot[lo:hi]
            cc[:, 0] += off
            cc[:, 1] = cc[:, 0] + 1
            cc[:, 2] = bslot[lo:hi]
            cc[:, 2] += off
            vv = np.empty((hi - lo, 3), dtype=np.float64)
            vv[:, 1] = w[lo:hi]
            vv[:, 0] = 1.0 - vv[:, 1]
            vv[:, 2] = -1.0
            parts.append((rows, cc, vv))
            count[rows + 1] += 1
        if not parts:
            return None
        triples = np.cumsum(count)
        n_t = int(triples[-1])
        if len(parts) == 1:
            cols, vals = parts[0][1], parts[0][2]
        else:
            cols = np.empty((n_t, 3), dtype=np.int32)
            vals = np.empty((n_t, 3), dtype=np.float64)
            cursor = triples[:-1].copy()
            for rows, cc, vv in parts:  # term order
                pos = cursor[rows]
                cols[pos] = cc
                vals[pos] = vv
                cursor[rows] += 1
        idx = np.int32 if 3 * n_t < np.iinfo(np.int32).max else np.int64
        return sp.csr_matrix(
            (vals.ravel(), cols.ravel(), (3 * triples).astype(idx)), shape=(n_c, self.L)
        )

    def score_range(self, lst, a, b):
        """(b - a, C) scores of slots a..b-1: the item part, the tilt times
        the listener's mainstreaminess, and one sparse product for the
        corrections of every configuration. The arithmetic per slot is the
        same for any range, so a one-slot range gives the same bits as the
        chunk that holds the slot."""
        S = self.base[a:b].copy()
        if self.tilt is not None:
            S += lst["m_u"] * self.tilt[a:b]
        G = self._chunk_matrix(lst["corr"], a, b)
        if G is not None:
            S += G @ self.TT
        return S

    # Per listener, by range

    def prepare(self, state, space):
        """The listener's inputs that every slot range reads, computed
        once: the term weights and the norm of each term source, the dense
        credit shares, and the co-listening numerators over the catalog
        (the history's rows of the transposed lists, as in
        concept_catalog())."""
        pre = {}
        for src in self.user_sources:
            if src in cn.TERM_CONCEPTS:
                name = cn.TERM_CONCEPTS[src]
                pre[src] = (
                    space.idf[name] ** 2 * state.term_share[name],
                    state.term_norm[name],
                )
            elif src in cn.CREDIT_CONCEPTS:
                name = cn.CREDIT_CONCEPTS[src]
                dense = np.zeros(space.credit_mats[name].shape[1], dtype=np.float64)
                dense[state.credit_cols[name]] = state.credit_share[name]
                pre[src] = dense
            elif src in cn.COLISTEN_CONCEPTS:
                if not space.transposed:
                    raise ValueError(
                        "co-listening over the catalog needs "
                        "an ItemSide built with transposed=True"
                    )
                name = cn.COLISTEN_CONCEPTS[src]
                pre[src] = np.asarray(
                    space.colisten_t[name][state.items].T @ state.weights,
                    dtype=np.float64,
                ).ravel()
            elif src in ("audio_fit", "popularity"):
                pre[src] = None
            else:
                raise KeyError(src)
        return pre

    def concept_range(self, state, space, pre, source, a, b):
        """(value, has) of one concept on slots a..b-1, equal to
        concept_catalog() on those slots bit for bit: the same expressions
        on the rows of the range (a sparse row product sums each row in
        its stored order, whatever rows come with it)."""
        if source in cn.TERM_CONCEPTS:
            name = cn.TERM_CONCEPTS[source]
            coef, norm = pre[source]
            m = space.term_mats[name]
            s, e = int(m.indptr[a]), int(m.indptr[b])
            sub = sp.csr_matrix(
                (m.data[s:e], m.indices[s:e], m.indptr[a : b + 1] - m.indptr[a]),
                shape=(b - a, m.shape[1]),
            )
            numer = sub @ coef
            has = space.has_terms[name][a:b] & (norm > 0)
            denom = np.where(has, space.item_norm[name][a:b] * norm, 1.0)
            return np.where(has, numer / denom, np.nan), has
        if source in cn.CREDIT_CONCEPTS:
            name = cn.CREDIT_CONCEPTS[source]
            m = space.credit_mats[name]
            s, e = int(m.indptr[a]), int(m.indptr[b])
            best = cn._row_max(
                pre[source][m.indices[s:e]], m.indptr[a : b + 1] - m.indptr[a]
            )
            has = space.has_credit[name][a:b]
            return np.where(has, best, np.nan), has
        if source in cn.COLISTEN_CONCEPTS:
            name = cn.COLISTEN_CONCEPTS[source]
            has = space.has_colisten[name][a:b]
            return np.where(has, pre[source][a:b] / state.total, np.nan), has
        if source == "audio_fit":
            value = np.full(b - a, np.nan, dtype=np.float64)
            has = np.zeros(b - a, dtype=bool)
            lo, hi = np.searchsorted(self.audio_rows, (a, b))
            if hi > lo:
                rows = self.audio_rows[lo:hi]
                value[rows - a], has[rows - a] = cn._audio_fit(state, space, rows)
            return value, has
        if source == "popularity":
            has = np.ones(b - a, dtype=bool)
            if a == 0:
                has[0] = False
            return np.where(has, space.popularity[a:b], np.nan), has
        raise KeyError(source)

    def chunk_scores(self, state, space, pre, m_u, a, b):
        """(b - a, C) scores of slots a..b-1 from the concepts of that
        range alone. Bit for bit the scores of score_range() on
        listener(): the same corrections enter the same sparse rows in the
        same order."""
        S = self.base[a:b].copy()
        if self.tilt is not None:
            S += m_u * self.tilt[a:b]
        corr = {}
        src, cached = None, None
        for k in self.user_terms:
            t = self.terms[k]
            if t.source != src:
                cached = None
                cached = self.concept_range(state, space, pre, t.source, a, b)
                src = t.source
            corr[k] = self._corrections(k, cached[0], cached[1], a, b)
        G = self._chunk_matrix(corr, 0, b - a)
        if G is not None:
            S += G @ self.TT
        return S

    def _masked(self, S, exclude, a, b):
        """S with the listener's train items and slot 0 set to -inf."""
        lo, hi = np.searchsorted(exclude, (a, b))
        S[exclude[lo:hi] - a] = -np.inf
        if a == 0:
            S[0] = -np.inf
        return S

    def _check_picks(self, picks, exclude):
        if (
            np.isin(picks, exclude).any()
            or (picks <= 0).any()
            or (picks >= self.n).any()
        ):
            raise ValueError("a pick is a train item or outside the slots")

    def ranks_cut(self, state, space, m_u, picks, exclude, cut, chunk=None):
        """(C, len(picks)) ranks under the P5 rule, exact up to cut and
        cut + 1 beyond, and the number of slots scored.

        The slots go in ranges of chunk slots: first the ranges that hold
        a pick, whose scores give the picks' own scores, then the others
        in slot order. Per pick and configuration the count is
        #(score >= s_pick) on slots before the pick plus #(score > s_pick)
        on the pick and after it, which is the P5 count. The pass stops
        when 1 + count exceeds cut for every pick and configuration. The
        counts only grow, so each rank then exceeds cut, and the result
        does not depend on where the pass stopped or on chunk."""
        chunk = self.chunk if chunk is None else int(chunk)
        cut = int(cut)
        if chunk < 1 or cut < 1:
            raise ValueError("chunk and cut must be positive")
        picks = np.asarray(picks, dtype=np.int64)
        exclude = np.unique(np.asarray(exclude, dtype=np.int64))
        self._check_picks(picks, exclude)
        pre = self.prepare(state, space)
        n_ranges = -(-self.n // chunk)
        first = list(dict.fromkeys(int(t) // chunk for t in picks))
        held = {}
        for r in first:
            a, b = r * chunk, min((r + 1) * chunk, self.n)
            held[r] = self._masked(
                self.chunk_scores(state, space, pre, m_u, a, b), exclude, a, b
            )
        n_p = len(picks)
        s_pick = np.empty((n_p, self.C), dtype=np.float64)
        for p, t in enumerate(picks):
            r = int(t) // chunk
            s_pick[p] = held[r][int(t) - r * chunk]
        finite = np.isfinite(s_pick)
        count = np.zeros((n_p, self.C), dtype=np.int64)
        scored = 0
        in_first = set(first)
        order = first + [r for r in range(n_ranges) if r not in in_first]
        for r in order:
            a, b = r * chunk, min((r + 1) * chunk, self.n)
            S = (
                held.pop(r)
                if r in held
                else self._masked(
                    self.chunk_scores(state, space, pre, m_u, a, b), exclude, a, b
                )
            )
            scored += b - a
            for p, t in enumerate(picks):
                sp_ = s_pick[p]
                e = min(max(int(t) - a, 0), b - a)
                if e:
                    count[p] += np.count_nonzero(S[:e] >= sp_, axis=0)
                if e < b - a:
                    count[p] += np.count_nonzero(S[e:] > sp_, axis=0)
            if np.all(~finite | (1 + count > cut)):
                break
        rank = np.where(finite, 1 + count, self.n)
        return np.minimum(rank, cut + 1).T, scored

    def scores(self, lst):
        """(C, n_slots) scores, slot 0 included (for checks and top-K)."""
        return np.concatenate(
            [
                self.score_range(lst, a, min(a + self.chunk, self.n))
                for a in range(0, self.n, self.chunk)
            ],
            axis=0,
        ).T

    def ranks(self, lst, picks, exclude):
        """(C, len(picks)) ranks under the P5 rule. exclude: the
        listener's train items (sorted or not). Slot 0 never ranks."""
        picks = np.asarray(picks, dtype=np.int64)
        exclude = np.unique(np.asarray(exclude, dtype=np.int64))
        self._check_picks(picks, exclude)
        n_p = len(picks)
        s_pick = np.empty((n_p, self.C), dtype=np.float64)
        for p, t in enumerate(picks):
            s_pick[p] = self.score_range(lst, int(t), int(t) + 1)[0]
        finite = np.isfinite(s_pick)
        greater = np.zeros((n_p, self.C), dtype=np.int64)
        ties = np.zeros((n_p, self.C), dtype=np.int64)
        for a in range(0, self.n, self.chunk):
            b = min(a + self.chunk, self.n)
            S = self.score_range(lst, a, b)
            lo, hi = np.searchsorted(exclude, (a, b))
            S[exclude[lo:hi] - a] = -np.inf
            if a == 0:
                S[0] = -np.inf
            for p, t in enumerate(picks):
                sp_ = s_pick[p]
                if a <= t < b and not np.array_equal(S[t - a], sp_):
                    raise RuntimeError(
                        "a pick's score differs between its "
                        "chunk and its own evaluation"
                    )
                greater[p] += np.count_nonzero(S > sp_, axis=0)
                if t > a:
                    e = min(int(t), b) - a
                    ties[p] += np.count_nonzero(S[:e] == sp_, axis=0)
        out = np.where(finite, 1 + greater + ties, self.n)
        return out.T


# Reference


def reference_scores(full, terms, tables, m_u, tier):
    """An independent dense scorer for checks: the plain sum, in term
    order, of each term's value, with the shape values from np.interp on
    the raw concept values of concepts.full_catalog() (full). No slot
    encoding, no item part, no corrections. Returns (C, n_slots)."""
    T = np.atleast_2d(np.asarray(tables, dtype=np.float64))
    offsets = np.cumsum([0] + [t.size for t in terms])[:-1]
    n = len(tier)
    out = np.zeros((T.shape[0], n), dtype=np.float64)
    for t, off in zip(terms, offsets):
        v, h = full[t.source]
        K = t.K
        kn = t.knot_array
        applies = (
            np.ones(n, dtype=bool)
            if t.tiers is None
            else np.isin(tier, np.asarray(t.tiers))
        )
        x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
        for c in range(T.shape[0]):
            tab = T[c, off : off + t.size]
            val = np.interp(x, kn, tab[:K])
            if t.zero_state:
                val = np.where(h & (v == 0.0), tab[K + ZERO], val)
            miss = tab[K + MISS] if t.missing_state else 0.0
            val = np.where(h, val, miss)
            val = np.where(applies, val, 0.0)
            if t.multiplier is not None:
                val = m_u * val
            out[c] += val
    return out


def reference_rank(scores, pick, exclude, tol):
    """(lowest, highest) rank of pick under the P5 rule when scores within
    tol of the pick's score may order either way. Without such near ties
    both bounds equal the exact rank."""
    s = np.array(scores, dtype=np.float64)
    s[0] = -np.inf
    s[np.asarray(exclude, dtype=np.int64)] = -np.inf
    t = int(pick)
    st = s[t]
    s[t] = -np.inf  # the pick never counts itself
    greater = int(np.count_nonzero(s > st + tol))
    near = np.abs(s - st) <= tol
    exact_tie_before = int(np.count_nonzero(s[:t] == st))
    n_near = int(np.count_nonzero(near))
    n_near_exact = int(np.count_nonzero(s == st))
    if n_near == n_near_exact:  # only exact ties are near
        r = 1 + int(np.count_nonzero(s > st)) + exact_tie_before
        return r, r
    return 1 + greater, 1 + greater + n_near

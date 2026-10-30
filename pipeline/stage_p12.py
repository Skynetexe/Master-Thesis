"""Stage P12: the concept features of every P11 training pair.

For each user in fusion_pairs.npz, the stage builds the user's state from
the train history (P6's user CSR) and computes every concept of
pipeline/concepts.py for the user's positives and negatives, through
concept_data.user_rows(). P14 ranks through the same function, so the
features the model learns from are computed exactly as the features it
ranks with (Breck et al. 2017, "training and serving features compute
the same values").

Output: fusion_features.npz, one row per row of fusion_pairs.npz, in the
same order:
- params: JSON string. concepts lists the columns in order.
- pairs_sha256: evaluation.npz_hash() of the fusion_pairs.npz it was
  built from. verify() refuses a file whose pairs changed, and P13 must
  check the same hash.
- values (rows x concepts) float32, NaN where a concept has no data.
- has (rows x concepts) bool. False means "no data for this track", not
  "no match": a present value of 0.0 is a real mismatch.
- match_tier (rows,) int8, codes into concepts.MATCH_TIERS.

Resume: the users run in blocks. Each finished block is saved in
SCRATCH_DIR/p12, keyed by the parameters, the pairs' SHA-256, the
upstream stages' output SHA-256 values, the code of this stage and of
concepts.py, concept_data.py and artists.py, and the block layout. A
stopped run continues after the last saved block. A code change starts
again from the first block.

verify():
- the file matches the pairs and the parameters, and NaN marks exactly
  the missing states;
- for every row, the missing states agree with the item side (a credit,
  a co-listening list, audio, terms), and popularity and tier equal the
  item's own values;
- for VERIFY_USERS random users, every concept of four rows each equals
  concept_data.Reference, a second implementation that shares no code
  with concepts.py;
- for the same rows, every evidence list of concepts.evidence() sums to
  the concept value. An explanation therefore never cites a number that
  the feature did not contain. The report prints the largest gaps that
  verify() measured.

reports/p12_report.md, the concept audit. It extends P11's sampler audit
from data coverage to the concepts themselves:
1. Three states per concept and row group: no data, data but zero (a
   real mismatch), and the value. Explanation coverage: the share of
   rows that the listener's own history can explain at all, and the
   stricter share with a history match, an artist, a work or a track
   that the listener played. A tag match can rest on a broad term such
   as "rock". Value range: percentiles 1, 50 and 99 per concept, which
   show the scale of each shape function and any heavy tail.
2. Each concept alone, within users: every positive against the same
   user's negatives. The pairs split into four states (both rows have
   data, only the positive, only the negative, neither). The AUC among
   the pairs where both have data is the concept's own separation, BPR's
   criterion for one concept (Rendle et al. 2009) in the Mann-Whitney
   reading of the AUC (Hanley & McNeil 1982). The AUC with a missing
   state ranked last adds the separation that comes from missing data.
   A concept that separates well only in the second form separates
   through its missing state, and its explanation line must say "no
   data" (Caruana et al. 2015 show why a readable model needs this
   check).
3. Overlap: Spearman correlations between concepts over the training
   rows, and for each concept the AUC of popularity between rows with
   and without data. Correlated concepts can trade credit between
   training runs (concurvity, Siems et al. 2023). A data flag that
   tracks popularity lets a missing state carry popularity (Josse et al.
   2024 on informative missingness).
4. Evidence size: how many units (tags, tracks, audio dimensions) carry
   each concept, and the share of the value that the top one and the
   top three carry. The top-3 share is the sufficiency of a three-item
   explanation (DeYoung et al. 2020). Here it is exact, because each
   value is the sum of its units.
5. Named examples: the concepts of a few positive pairs and of two
   uniform negatives, one drawn at random and the one with the strongest
   history match, with their top evidence. These are the raw material of
   an explanation. A negative with a history match shows that a track
   the listener did not play is not a track the listener dislikes (Pan
   et al. 2008).
"""

import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import artists as ar
from . import concept_data as cd
from . import concepts as cn
from . import evaluation as ev
from .als import Progress
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    connect,
    features_table,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from .config import MODEL_DIR, SCRATCH_DIR

VERSION = "pair-features-v1"
TASKS_PER_WORKER = 8
WORKERS = max(1, min(16, os.cpu_count() or 1))
SEED = 2026092412
VERIFY_USERS = 40
VERIFY_RTOL = 1e-6  # float32 storage holds about 7 digits
VERIFY_ATOL = 1e-9
EVIDENCE_TOL = 1e-9
REPORT_CORR_ROWS = 2_000_000
REPORT_EVIDENCE_USERS = 300
REPORT_EVIDENCE_NEGATIVES = 4  # per kind and user
REPORT_EXAMPLES = 3
GROUPS = (
    ("positives, train", 0, 0),
    ("positives, selection", 0, 1),
    ("uniform negatives", 1, None),
    ("p_pop negatives", 2, None),
)
PERSONAL = (
    list(cn.TERM_CONCEPTS) + list(cn.CREDIT_CONCEPTS) + list(cn.COLISTEN_CONCEPTS)
)
HISTORY = list(cn.CREDIT_CONCEPTS) + list(cn.COLISTEN_CONCEPTS)

_CTX = {}


def _params():
    return {
        "version": VERSION,
        "concepts": list(cn.CONTINUOUS),
        "match_tiers": list(cn.MATCH_TIERS),
        "audio_dims": list(cn.AUDIO_DIMS),
        "sigma_floor": cn.SIGMA_FLOOR,
        "weight_scale": cd.WEIGHT_SCALE,
    }


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez keeps .npz
    np.savez(tmp, **arrays)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _path_sha(path):
    """SHA-256 of a file, or of a directory's files in name order."""
    import hashlib

    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(f.relative_to(path)).encode("utf-8"))
            digest.update(file_sha256(f).encode("utf-8"))
        return digest.hexdigest()
    return file_sha256(path)


def _blocks(costs, n_tasks):
    """Contiguous index ranges with near-equal summed cost."""
    cum = np.cumsum(np.asarray(costs, dtype=np.int64))
    if len(cum) == 0:
        return []
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n_tasks) / n_tasks)
    edges = np.unique(np.concatenate(([0], cuts, [len(cum)])))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _block(task):
    """The rows of users lo..hi-1 of the pairs file."""
    lo, hi = task
    c = _CTX
    p_indptr = c["p_indptr"]
    r0, r1 = int(p_indptr[lo]), int(p_indptr[hi])
    n_c = len(cn.CONTINUOUS)
    values = np.empty((r1 - r0, n_c), dtype=np.float32)
    has = np.empty((r1 - r0, n_c), dtype=bool)
    tier = np.empty(r1 - r0, dtype=np.int8)
    for k in range(lo, hi):
        a, b = int(p_indptr[k]), int(p_indptr[k + 1])
        items, weights = cd.history_of(
            c["u_indptr"], c["u_indices"], c["u_data"], int(c["users"][k])
        )
        v, h, t = cd.user_rows(c["space"], items, weights, c["p_items"][a:b])
        values[a - r0 : b - r0] = v
        has[a - r0 : b - r0] = h
        tier[a - r0 : b - r0] = t
    return values, has, tier


def pair_states(v, h, pos_rows, neg_rows):
    """One concept over the (positive, negative) pairs of the same user.
    pos_rows: (P,) rows. neg_rows: (P, K) rows of that user's negatives.
    Returns the pair counts per state ("pairs", "both", "pos_only",
    "neg_only", "neither"), "auc_both", the AUC among the pairs where
    both rows have data, and "auc_missing_last", the AUC over every pair
    when a missing state ranks below any value. Ties count half. An AUC
    without pairs is NaN."""
    vp, hp = v[pos_rows][:, None], h[pos_rows][:, None]
    vn, hn = v[neg_rows], h[neg_rows]
    both = hp & hn
    out = {
        "pairs": int(both.size),
        "both": int(both.sum()),
        "pos_only": int((hp & ~hn).sum()),
        "neg_only": int((~hp & hn).sum()),
        "neither": int((~hp & ~hn).sum()),
    }
    win = float((both & (vp > vn)).sum())
    tie = float((both & (vp == vn)).sum())
    out["auc_both"] = (win + 0.5 * tie) / out["both"] if out["both"] else math.nan
    out["auc_missing_last"] = (
        (win + 0.5 * tie + out["pos_only"] + 0.5 * out["neither"]) / out["pairs"]
        if out["pairs"]
        else math.nan
    )
    return out


def rank_auc(ranks, labels):
    """P(a labelled row outranks an unlabelled row), ties counting half:
    the Mann-Whitney U statistic over n1 * n0 (Hanley & McNeil 1982).
    ranks: average ranks of the scores (scipy.stats.rankdata). NaN when a
    class is empty."""
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    # Ranks are multiples of 0.5, so the float64 sum is exact up to 2**52.
    return (float(np.asarray(ranks)[labels].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(x, y):
    """Spearman's rho with average ranks for ties (Spearman 1904)."""
    if len(x) < 3:
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry) / denom if denom > 0 else math.nan


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def _fmt(x, digits=3):
    return (
        "n/a"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.{digits}f}"
    )


class StageP12:
    name = "p12_pair_features"
    upstream = [
        "p25_train",
        "p3_sampling",
        "p4_features",
        "p6_als",
        "p9b_colisten",
        "p11_fusion_pairs",
    ]

    def __init__(self):
        self._cache = {}
        self._verified = None

    def outputs(self):
        return {
            "features": MODEL_DIR / "fusion_features.npz",
            "report": MODEL_DIR / "reports" / "p12_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        for label, mod in (
            ("concepts_module", cn),
            ("concept_data_module", cd),
            ("artists_module", ar),
        ):
            inputs[label] = file_sha256(Path(mod.__file__))
        for table in ("item_artists", "item_works"):
            try:
                inputs[table] = _path_sha(features_table(table))
            except Exception:  # recorded as absent
                inputs[table] = None
        return inputs

    # The loaders below read the model folder and DuckDB. test_p12.py
    # replaces them.
    def _pairs(self):
        if "pairs" not in self._cache:
            path = MODEL_DIR / "fusion_pairs.npz"
            z = np.load(path)
            self._cache["pairs"] = (
                {
                    k: np.asarray(z[k])
                    for k in ("user_id", "split", "indptr", "item", "kind")
                },
                json.loads(str(z["params"])),
                ev.npz_hash(path),
            )
        return self._cache["pairs"]

    def _id_slots(self):
        return ev.load_id_slots()

    def _sources(self, n_slots):
        if "sources" not in self._cache:
            self._cache["sources"] = cd.load_sources(n_slots, log.info)
        return self._cache["sources"]

    def _space(self, src):
        if "space" not in self._cache:
            self._cache["space"] = cd.item_side(src, transposed=False)
        return self._cache["space"]

    def _csr(self):
        mm = {"mmap_mode": "r"}
        return (
            np.load(MODEL_DIR / "csr_user_indptr.npy"),
            np.load(MODEL_DIR / "csr_user_indices.npy", **mm),
            np.load(MODEL_DIR / "csr_user_data.npy", **mm),
        )

    def _upstream_outputs(self):
        """The output SHA-256 values that the upstream markers record. A
        rerun that writes the same files keeps the saved blocks valid."""
        out = {}
        for stage in self.upstream:
            if stage == "p11_fusion_pairs":
                continue  # the pairs' own SHA-256 stands for it
            doc = load_json(marker_path(stage)) or {}
            out[stage] = {
                k: (v.get("sha256") if isinstance(v, dict) else v)
                for k, v in sorted((doc.get("outputs") or {}).items())
            }
        return out

    def _track_names(self, ids):
        """{item_id: "artist - title"}, best effort. item_catalog has one
        row per raw recording MBID, so one row per item is chosen the
        way the data-prep scripts choose it: names present first, then
        the most listens, then the smallest MBID."""
        ids = sorted({int(i) for i in ids})
        if not ids:
            return {}
        con = connect(4)
        try:
            rows = con.execute(f"""
                SELECT i.item_id, c.artist_name, c.recording_name
                FROM read_parquet('{features_table("item_ids")}') AS i
                JOIN read_parquet('{features_table("item_catalog")}') AS c
                  ON c.canonical_recording_mbid = i.canonical_recording_mbid
                WHERE i.item_id IN ({", ".join(map(str, ids))})
                QUALIFY row_number() OVER (
                    PARTITION BY i.item_id
                    ORDER BY c.recording_name IS NOT NULL DESC,
                             c.artist_name IS NOT NULL DESC,
                             c.listens DESC, c.recording_mbid) = 1
            """).fetchall()
        except Exception as exc:  # names are optional
            log.warning("p12: no track names for the examples: %s", exc)
            return {}
        finally:
            con.close()
        return {int(r[0]): f"{r[1]} - {r[2]}" for r in rows}

    def _entity_names(self, kind, ids):
        """{MusicBrainz row id: name} for artists or works, best effort."""
        ids = sorted({int(i) for i in ids})
        if not ids:
            return {}
        table, id_col, name_col = {
            "artists": ("item_artists", "artist_id", "artist_name"),
            "works": ("item_works", "work_id", "work_name"),
        }[kind]
        con = connect(4)
        try:
            rows = con.execute(f"""
                SELECT CAST(t.{id_col} AS BIGINT) AS id, min(t.{name_col})
                FROM read_parquet('{features_table(table)}') AS t
                WHERE CAST(t.{id_col} AS BIGINT)
                      IN ({", ".join(map(str, ids))})
                GROUP BY 1
            """).fetchall()
        except Exception as exc:  # names are optional
            log.warning("p12: no %s names for the examples: %s", kind, exc)
            return {}
        finally:
            con.close()
        return {int(r[0]): str(r[1]) for r in rows}

    def run(self, params, files):
        outputs = self.outputs()
        p, p_params, pairs_sha = self._pairs()
        users, p_indptr, p_items = p["user_id"], p["indptr"], p["item"]
        n_slots, id_min = self._id_slots()
        u_indptr, u_indices, u_data = self._csr()
        check(
            len(u_indptr) - 1 > int(users.max()),
            "a pair's user falls outside the user CSR",
        )
        check(
            int(p_items.min()) >= id_min and int(p_items.max()) < n_slots,
            "a pair's item falls outside the item slots",
        )
        hist = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
            u_indptr[users], dtype=np.int64
        )
        costs = hist + 50 * np.diff(p_indptr)
        tasks = _blocks(costs, WORKERS * TASKS_PER_WORKER)
        work = SCRATCH_DIR / "p12"
        state_path = work / "state.json"
        key = {
            "params": _params(),
            "pairs_sha256": pairs_sha,
            "upstream": self._upstream_outputs(),
            "code": {
                Path(f).name: file_sha256(Path(f))
                for f in (__file__, cn.__file__, cd.__file__, ar.__file__)
            },
            "blocks": [list(t) for t in tasks],
        }
        state = load_json(state_path) or {}
        if state.get("key") != key:
            for f in work.glob("part_*.npz"):
                f.unlink()
            state = {"key": key, "parts": {}}
            atomic_json_write(state, state_path)
        elif (
            state.get("final_sha")
            and outputs["features"].exists()
            and file_sha256(outputs["features"]) == state["final_sha"]
        ):
            log.info("p12: features resume, every block is done")
            return self._info(outputs)

        def part_path(k):
            return work / f"part_{k:05d}.npz"

        done = {
            int(k)
            for k, sha in state["parts"].items()
            if part_path(int(k)).exists() and file_sha256(part_path(int(k))) == sha
        }
        pending = [k for k in range(len(tasks)) if k not in done]
        if done:
            log.info("p12: blocks resume, %d of %d done", len(done), len(tasks))
        if pending:
            src = self._sources(n_slots)
            space = self._space(src)
            _CTX.update(
                space=space,
                users=users,
                p_indptr=p_indptr,
                p_items=p_items,
                u_indptr=u_indptr,
                u_indices=u_indices,
                u_data=u_data,
            )
            n_users_left = sum(tasks[k][1] - tasks[k][0] for k in pending)
            log.info(
                "p12: %d users, %d rows, %d concepts, %d blocks left, " "%d workers",
                len(users),
                len(p_items),
                len(cn.CONTINUOUS),
                len(pending),
                WORKERS,
            )
            prog = Progress(
                "p12: concept features", n_users_left, log.info, unit=" users"
            )
            try:
                if WORKERS <= 1 or len(pending) <= 1:
                    results = (_block(tasks[k]) for k in pending)
                    self._save_parts(
                        pending, results, tasks, part_path, state, state_path, prog
                    )
                else:
                    with ProcessPoolExecutor(
                        max_workers=WORKERS, mp_context=mp.get_context("fork")
                    ) as pool:
                        results = pool.map(_block, [tasks[k] for k in pending])
                        self._save_parts(
                            pending, results, tasks, part_path, state, state_path, prog
                        )
            finally:
                _CTX.clear()
            prog.finish()

        parts = [np.load(part_path(k)) for k in range(len(tasks))]
        values = np.concatenate([z["values"] for z in parts])
        has = np.concatenate([z["has"] for z in parts])
        tier = np.concatenate([z["tier"] for z in parts])
        check(
            len(values) == len(p_items),
            f"the blocks hold {len(values)} rows, the pairs {len(p_items)}",
        )
        _atomic_savez(
            outputs["features"],
            params=np.array(json.dumps(_params())),
            pairs_sha256=np.array(pairs_sha),
            values=values,
            has=has,
            match_tier=tier,
        )
        state["final_sha"] = file_sha256(outputs["features"])
        atomic_json_write(state, state_path)
        for k in range(len(tasks)):
            part_path(k).unlink()
        log.info("p12: %d rows written", len(values))
        return self._info(outputs)

    def _save_parts(self, pending, results, tasks, part_path, state, state_path, prog):
        for k, (values, has, tier) in zip(pending, results):
            path = part_path(k)
            _atomic_savez(path, values=values, has=has, tier=tier)
            state["parts"][str(k)] = file_sha256(path)
            atomic_json_write(state, state_path)
            prog.advance(tasks[k][1] - tasks[k][0])

    def _info(self, outputs):
        z = np.load(outputs["features"])
        return {
            "features": {
                "path": str(outputs["features"]),
                "rows": int(len(z["match_tier"])),
                "sha256": ev.npz_hash(outputs["features"]),
            }
        }

    def _load(self):
        z = np.load(self.outputs()["features"])
        return (
            {k: z[k] for k in ("values", "has", "match_tier")},
            json.loads(str(z["params"])),
            str(z["pairs_sha256"]),
        )

    def verify(self, params):
        self._verified = None
        d, stored, pairs_sha = self._load()
        p, _, current_sha = self._pairs()
        check(
            stored == _params(),
            f"fusion_features.npz was built with "
            f"{stored}, the stage now says "
            f"{_params()}",
        )
        check(
            pairs_sha == current_sha,
            "fusion_features.npz was built from other fusion pairs",
        )
        values, has, tier = d["values"], d["has"], d["match_tier"]
        items, p_indptr = p["item"].astype(np.int64), p["indptr"]
        n, n_c = len(items), len(cn.CONTINUOUS)
        check(
            values.shape == (n, n_c)
            and has.shape == (n, n_c)
            and tier.shape == (n,)
            and values.dtype == np.float32
            and has.dtype == bool,
            "the feature arrays do not match the pairs in shape or type",
        )
        check(
            np.array_equal(np.isnan(values), ~has),
            "NaN does not mark exactly the missing states",
        )
        n_slots, _ = self._id_slots()
        src = self._sources(n_slots)
        space = self._space(src)
        col = {c: j for j, c in enumerate(cn.CONTINUOUS)}
        lens = np.diff(p_indptr)

        # Missing states against the item side, for every row.
        for concept, name in cn.CREDIT_CONCEPTS.items():
            check(
                np.array_equal(has[:, col[concept]], space.has_credit[name][items]),
                f"{concept}: a missing state disagrees with the credits",
            )
        for concept, name in cn.COLISTEN_CONCEPTS.items():
            check(
                np.array_equal(has[:, col[concept]], space.has_colisten[name][items]),
                f"{concept}: a missing state disagrees with the " f"co-listening lists",
            )
        for concept, name in cn.TERM_CONCEPTS.items():
            h = has[:, col[concept]]
            user_has = np.maximum.reduceat(h, p_indptr[:-1])
            want = space.has_terms[name][items] & np.repeat(user_has, lens)
            check(
                np.array_equal(h, want),
                f"{concept}: a missing state disagrees with the item's "
                f"terms or with the user's other rows",
            )
        item_audio = space.audio_mask[items].any(axis=1)
        check(
            not (has[:, col["audio_fit"]] & ~item_audio).any(),
            "audio_fit has data for an item without audio",
        )
        pop = col["popularity"]
        check(
            bool(has[:, pop].all())
            and np.array_equal(
                values[:, pop], space.popularity[items].astype(np.float32)
            ),
            "popularity differs from log1p(train listens) of the item",
        )
        check(
            np.array_equal(tier.astype(np.int64), space.tier[items]),
            "match_tier differs from the item's bridge tier",
        )
        ranges = {
            **{c: (0.0, 1.0 + 1e-6) for c in cn.TERM_CONCEPTS},
            **{c: (0.0, 1.0 + 1e-6) for c in cn.CREDIT_CONCEPTS},
            **{c: (0.0, np.inf) for c in cn.COLISTEN_CONCEPTS},
            "audio_fit": (-np.inf, 0.0),
            "popularity": (0.0, np.inf),
        }
        for concept, (lo, hi) in ranges.items():
            v = values[has[:, col[concept]], col[concept]]
            check(
                len(v) == 0 or (float(v.min()) >= lo and float(v.max()) <= hi),
                f"{concept}: a value falls outside [{lo}, {hi}]",
            )

        # Recompute sampled rows with the second implementation, and check
        # that every evidence list sums to its value.
        u_indptr, u_indices, u_data = self._csr()
        ref = cd.Reference(src)
        rng = np.random.default_rng(SEED)
        kinds = p["kind"]
        users_idx = rng.choice(
            len(p["user_id"]), size=min(VERIFY_USERS, len(p["user_id"])), replace=False
        )
        n_rows = n_ev = 0
        ref_gap = ev_gap = 0.0
        for k in np.sort(users_idx):
            s, e = int(p_indptr[k]), int(p_indptr[k + 1])
            rows = []
            for kd in (0, 1, 2):
                cand = np.flatnonzero(kinds[s:e] == kd)
                if len(cand):
                    rows.append(s + int(rng.choice(cand)))
            rows.append(s + int(rng.integers(0, e - s)))
            h_items, h_weights = cd.history_of(
                u_indptr, u_indices, u_data, int(p["user_id"][k])
            )
            u_ref = ref.user(h_items, h_weights)
            state = cn.build_user_state(h_items, h_weights, space)
            for r in rows:
                want = ref.values(u_ref, int(items[r]))
                for concept in cn.CONTINUOUS:
                    j = col[concept]
                    wv, wh = want[concept]
                    got_v = float(values[r, j])
                    ok = bool(has[r, j]) == wh and (
                        not wh or abs(got_v - wv) <= VERIFY_RTOL * abs(wv) + VERIFY_ATOL
                    )
                    check(
                        ok,
                        f"user {int(p['user_id'][k])}, item "
                        f"{int(items[r])}: {concept} is "
                        f"{got_v if has[r, j] else 'missing'}, the "
                        f"reference gives "
                        f"{wv if wh else 'missing'}",
                    )
                    if wh:
                        ref_gap = max(
                            ref_gap, abs(got_v - wv) / max(abs(wv), VERIFY_ATOL)
                        )
                check(
                    int(tier[r]) == want["match_tier"],
                    f"item {int(items[r])}: match_tier differs from the " f"reference",
                )
                evd = cn.evidence(state, space, int(items[r]))
                for concept in PERSONAL + ["audio_fit"]:
                    entry = evd[concept]
                    if not entry["has_data"]:
                        continue
                    total = math.fsum(t["term"] for t in entry["terms"])
                    gap = abs(total - entry["value"])
                    check(
                        gap <= EVIDENCE_TOL * max(1.0, abs(entry["value"])),
                        f"item {int(items[r])}: the {concept} evidence "
                        f"sums to {total}, the value is {entry['value']}",
                    )
                    ev_gap = max(ev_gap, gap)
                    n_ev += 1
                n_rows += 1
        self._verified = {
            "rows": n,
            "reference_rows": n_rows,
            "reference_gap": ref_gap,
            "evidence_lists": n_ev,
            "evidence_gap": ev_gap,
        }
        log.info(
            "p12: verify passed. %d rows, %d rows recomputed by the "
            "reference (largest relative gap %.1e), %d evidence lists "
            "sum to their values (largest gap %.1e)",
            n,
            n_rows,
            ref_gap,
            n_ev,
            ev_gap,
        )

    # Report

    def _neg_rows(self, p):
        """Rows of the positives and, per positive, its user's negatives
        of each kind, from P11's layout: positives, then K uniform draws,
        then K p_pop draws."""
        kinds, indptr = p["kind"], p["indptr"]
        lens = np.diff(indptr)
        n_neg = np.add.reduceat((kinds == 1).astype(np.int64), indptr[:-1])
        k = int(n_neg[0])
        check(
            bool((n_neg == k).all())
            and bool(
                (np.add.reduceat((kinds == 2).astype(np.int64), indptr[:-1]) == k).all()
            ),
            "the users hold different numbers of negatives",
        )
        pos = np.flatnonzero(kinds == 0)
        user = np.searchsorted(indptr, pos, side="right") - 1
        base = indptr[user] + (lens[user] - 2 * k)
        uni = base[:, None] + np.arange(k)[None, :]
        check(
            bool((kinds[uni] == 1).all()) and bool((kinds[uni + k] == 2).all()),
            "the negatives do not follow P11's row layout",
        )
        return pos, {1: uni, 2: uni + k}, k

    def report(self, params):
        d, stored, pairs_sha = self._load()
        p, p_params, _ = self._pairs()
        values, has = d["values"], d["has"]
        kinds, split = p["kind"], p["split"]
        lens = np.diff(p["indptr"])
        row_split = np.repeat(split, lens)
        n, concepts = len(values), list(cn.CONTINUOUS)
        col = {c: j for j, c in enumerate(concepts)}
        n_slots, _ = self._id_slots()
        src = self._sources(n_slots)
        space = self._space(src)
        lines = [
            "# P12 report: concept features of the fusion pairs",
            "",
            f"- parameters: {json.dumps(stored)}",
            f"- rows: {n:,}, from fusion_pairs.npz with SHA-256 "
            f"{pairs_sha[:16]}..., built by P11 with "
            f"{json.dumps(p_params)}",
            "- every value comes from concept_data.user_rows(), the "
            "same function that P14 ranks with.",
        ]
        v = self._verified
        if v:
            lines.append(
                f"- verify(): {v['reference_rows']:,} rows recomputed by "
                f"the independent reference, largest relative gap "
                f"{v['reference_gap']:.1e} (float32 storage). "
                f"{v['evidence_lists']:,} evidence lists sum to their "
                f"concept values, largest gap {v['evidence_gap']:.1e}."
            )
        lines.append("")

        masks = []
        for _, kd, sp_code in GROUPS:
            m = kinds == kd
            if sp_code is not None:
                m &= row_split == sp_code
            masks.append(m)
        lines += self._states_section(values, has, masks, concepts, col)
        lines += self._alone_section(p, values, has, concepts)
        lines += self._overlap_section(values, has, masks, concepts, col)
        lines += self._evidence_sections(p, space, src)
        atomic_text_write("\n".join(lines) + "\n", self.outputs()["report"])
        log.info("p12 report written")

    def _states_section(self, values, has, masks, concepts, col):
        lines = [
            "## 1. Three states per concept",
            "",
            "Each cell: share of rows with data / share of rows with "
            "data and a value of exactly 0 (a real mismatch) / median "
            'value among the rows with data. "No data" and "no '
            'match" are different statements. The model gives each '
            "missing state its own learned value, and the "
            "explanation names the state.",
            "",
            "| concept | " + " | ".join(g[0] for g in GROUPS) + " |",
            "|---|" + "---:|" * len(GROUPS),
        ]
        for j, c in enumerate(concepts):
            cells = []
            for m in masks:
                rows_n = int(m.sum())
                h = has[m, j]
                v = values[m, j][h]
                med = float(np.median(v)) if len(v) else None
                cells.append(
                    f"{_pct(int(h.sum()), rows_n)} / "
                    f"{_pct(int((v == 0).sum()), rows_n)} / "
                    f"{_fmt(med, 4)}"
                )
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        q_groups = [
            (label, m)
            for (label, kd, _), m in zip(GROUPS, masks)
            if label != "positives, selection"
        ]
        lines += [
            "",
            "### Value range",
            "",
            "Percentiles 1, 50 and 99 of each concept among the rows "
            "with data. They show the scale that each shape function "
            "of P13 must cover, and heavy tails: a few extreme values "
            "that could dominate a concept.",
            "",
            "| concept | "
            + " | ".join(f"{label}: p1 / p50 / p99" for label, _ in q_groups)
            + " |",
            "|---|" + "---:|" * len(q_groups),
        ]
        for j, c in enumerate(concepts):
            cells = []
            for _, m in q_groups:
                v = values[m & has[:, j], j].astype(np.float64)
                cells.append(
                    " / ".join(_fmt(float(x), 4) for x in np.percentile(v, [1, 50, 99]))
                    if len(v)
                    else "n/a"
                )
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "### Explanation coverage",
            "",
            "A personal match: one of "
            + ", ".join(PERSONAL)
            + " has data and a value above 0, "
            "so the explanation can name a tag, an artist, a work or "
            "a track from the listener's own history. A history "
            "match is the stricter kind: one of "
            + ", ".join(HISTORY)
            + " is above 0, so the explanation "
            "names an artist, a work or a track that the listener "
            "played. A tag or genre match can rest on a broad term "
            'such as "rock". Rows with no personal match and no '
            "audio data can only be explained by popularity and the "
            "bridge tier.",
            "",
            "| rows | rows in group | at least one personal match | "
            "history match | two or more personal matches | audio "
            "fit only | popularity and tier only |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        match = np.zeros(len(values), dtype=np.int8)
        for c in PERSONAL:  # NaN > 0 is False
            match += has[:, col[c]] & (values[:, col[c]] > 0)
        history = np.zeros(len(values), dtype=bool)
        for c in HISTORY:
            history |= has[:, col[c]] & (values[:, col[c]] > 0)
        audio = has[:, col["audio_fit"]]
        for (label, _, _), m in zip(GROUPS, masks):
            rows_n = int(m.sum())
            mm, au = match[m], audio[m]
            lines.append(
                f"| {label} | {rows_n:,} | {_pct(int((mm >= 1).sum()), rows_n)}"
                f" | {_pct(int(history[m].sum()), rows_n)}"
                f" | {_pct(int((mm >= 2).sum()), rows_n)} | "
                f"{_pct(int(((mm == 0) & au).sum()), rows_n)} | "
                f"{_pct(int(((mm == 0) & ~au).sum()), rows_n)} |"
            )
        lines.append("")
        return lines

    def _alone_section(self, p, values, has, concepts):
        pos, negs, k = self._neg_rows(p)
        lines = [
            "## 2. Each concept alone, within users",
            "",
            f"Each positive against the {k} negatives of each kind of "
            "the same user. The pairs fall into four states: both "
            "rows have data, only the positive, only the negative, "
            'neither. "AUC, both" uses the pairs where both rows '
            'have data, so it measures the value alone. "AUC, '
            'missing last" uses every pair and ranks a missing state '
            "below any value. When the second is far above the first, "
            "the concept separates through missing data, and its "
            'explanation line must say "no data", not "no '
            'match". Ties count half.',
            "",
        ]
        for kd, label in ((1, "uniform negatives"), (2, "p_pop negatives")):
            lines += [
                f"### Against {label}",
                "",
                "| concept | AUC, both | both | positive only | "
                "negative only | neither | AUC, missing last |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for j, c in enumerate(concepts):
                s = pair_states(values[:, j], has[:, j], pos, negs[kd])
                lines.append(
                    f"| {c} | {_fmt(s['auc_both'])} | "
                    f"{_pct(s['both'], s['pairs'])} | "
                    f"{_pct(s['pos_only'], s['pairs'])} | "
                    f"{_pct(s['neg_only'], s['pairs'])} | "
                    f"{_pct(s['neither'], s['pairs'])} | "
                    f"{_fmt(s['auc_missing_last'])} |"
                )
            lines.append("")
        return lines

    def _overlap_section(self, values, has, masks, concepts, col):
        n = len(values)
        rng = np.random.default_rng(SEED + 3)
        sample = np.sort(rng.choice(n, size=min(REPORT_CORR_ROWS, n), replace=False))
        sv, sh = values[sample], has[sample]
        nc = len(concepts)
        rho = np.full((nc, nc), np.nan)
        for i in range(nc):
            rho[i, i] = 1.0
            for j in range(i + 1, nc):
                m = sh[:, i] & sh[:, j]
                rho[i, j] = rho[j, i] = spearman(sv[m, i], sv[m, j])
        lines = [
            "## 3. Overlap between concepts",
            "",
            f"Spearman correlations over {len(sample):,} random "
            "training rows (every row group). Each cell uses the rows "
            "where both concepts have data. Two strongly correlated "
            "concepts can trade credit between training runs, so an "
            "explanation should not rank them against each other.",
            "",
            "| concept | " + " | ".join(c.replace("_", " ") for c in concepts) + " |",
            "|---|" + "---:|" * nc,
        ]
        for i, ci in enumerate(concepts):
            lines.append(
                f"| {ci} | "
                + " | ".join(_fmt(float(rho[i, j]), 2) for j in range(nc))
                + " |"
            )
        lines += [
            "",
            "### Popularity behind the missing states",
            "",
            "AUC of popularity between rows with data and rows "
            "without: the chance that a row with data is the more "
            "popular one. 0.5 means that the missing state carries "
            "no popularity. Near 1, a missing state marks a rare "
            "track, and a learned missing value can act as a "
            "popularity penalty.",
            "",
            "| concept | " + " | ".join(g[0] for g in GROUPS) + " |",
            "|---|" + "---:|" * len(GROUPS),
        ]
        pop = col["popularity"]
        ranks = [rankdata(values[m, pop]) for m in masks]
        for c in concepts:
            if c in cn.NEVER_MISSING:
                continue
            cells = [_fmt(rank_auc(r, has[m, col[c]])) for r, m in zip(ranks, masks)]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    def _evidence_sections(self, p, space, src):
        u_indptr, u_indices, u_data = self._csr()
        rng = np.random.default_rng(SEED + 5)
        indptr, kinds, items = p["indptr"], p["kind"], p["item"]
        n_users = len(p["user_id"])
        pick = np.sort(
            rng.choice(n_users, size=min(REPORT_EVIDENCE_USERS, n_users), replace=False)
        )
        evid = [c for c in cn.CONTINUOUS if c != "popularity"]
        stats = {g: {c: [] for c in evid} for g in ("positive", "negative")}
        examples, contrast = [], []
        for k in pick:
            s, e = int(indptr[k]), int(indptr[k + 1])
            rows = list(s + np.flatnonzero(kinds[s:e] == 0))
            for kd in (1, 2):
                rows += list(
                    s + np.flatnonzero(kinds[s:e] == kd)[:REPORT_EVIDENCE_NEGATIVES]
                )
            h_items, h_weights = cd.history_of(
                u_indptr, u_indices, u_data, int(p["user_id"][k])
            )
            state = cn.build_user_state(h_items, h_weights, space)
            for r in rows:
                evd = cn.evidence(state, space, int(items[r]))
                group = "positive" if kinds[r] == 0 else "negative"
                for c in evid:
                    entry = evd[c]
                    if not entry["has_data"]:
                        continue
                    sizes = [abs(t["term"]) for t in entry["terms"]]
                    total = math.fsum(sizes)
                    top = sorted(sizes, reverse=True)
                    top1, top3 = (
                        (top[0] / total, math.fsum(top[:3]) / total)
                        if total > 0
                        else (math.nan, math.nan)
                    )
                    stats[group][c].append((len(sizes), top1, top3))
                entry = {
                    "n_data": sum(bool(evd[c]["has_data"]) for c in evid),
                    "key": float(rng.random()),
                    "user": int(p["user_id"][k]),
                    "item": int(items[r]),
                    "evd": evd,
                    "history": max(
                        (float(evd[c]["value"]) for c in HISTORY if evd[c]["has_data"]),
                        default=0.0,
                    ),
                }
                if kinds[r] == 0:
                    examples.append(entry)
                elif kinds[r] == 1:
                    contrast.append(entry)
        lines = [
            "## 4. Evidence size",
            "",
            f"{len(pick):,} random users: all their positives and the "
            f"first {REPORT_EVIDENCE_NEGATIVES} negatives of each kind. "
            "Units: shared terms, history tracks (co-listening, and "
            "the tracks that carry the best artist or work) or audio "
            "dimensions. Top-3 share: the part of the concept's value "
            "that its three largest units carry. Each value is the "
            "exact sum of its units, so a three-unit explanation with "
            "a top-3 share of 0.9 states 90 percent of the concept. A "
            "top-1 share near 1 means that one unit decides the "
            "concept.",
            "",
            "| concept | group | rows with data | with units | "
            "median units | p90 units | median top-1 share | median "
            "top-3 share |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for c in evid:
            for g in ("positive", "negative"):
                rows = stats[g][c]
                if not rows:
                    lines.append(f"| {c} | {g} | 0 | n/a | n/a | n/a | n/a " f"| n/a |")
                    continue
                units = np.array([r[0] for r in rows])
                meds = []
                for pos in (1, 2):
                    sh = np.array([r[pos] for r in rows if not math.isnan(r[pos])])
                    meds.append(float(np.median(sh)) if len(sh) else None)
                lines.append(
                    f"| {c} | {g} | {len(rows):,} | "
                    f"{_pct(int((units > 0).sum()), len(rows))} | "
                    f"{np.median(units):.0f} | {np.percentile(units, 90):.0f} "
                    f"| {_fmt(meds[0], 2)} | {_fmt(meds[1], 2)} |"
                )
        lines.append("")
        examples.sort(key=lambda t: (-t["n_data"], t["key"]))
        chosen = [dict(t, role="positive") for t in examples[:REPORT_EXAMPLES]]
        if contrast:
            typical = min(contrast, key=lambda t: t["key"])
            chosen.append(dict(typical, role="uniform negative, drawn at " "random"))
            matched = [t for t in contrast if t["history"] > 0 and t is not typical]
            if matched:
                best = max(matched, key=lambda t: (t["history"], -t["key"]))
                chosen.append(
                    dict(best, role="uniform negative with a " "history match")
                )
        lines += self._examples(chosen, src)
        return lines

    def _examples(self, chosen, src):
        tracks, ents = set(), {"artists": set(), "works": set()}
        for ex in chosen:
            item, evd = ex["item"], ex["evd"]
            tracks.add(item)
            for c, name in cn.CREDIT_CONCEPTS.items():
                if evd[c]["has_data"] and evd[c]["credit"] is not None:
                    ents[name].add(int(src.credit_ids[name][evd[c]["credit"]]))
                tracks.update(t["id"] for t in evd[c]["terms"][:3])
            for c in cn.COLISTEN_CONCEPTS:
                tracks.update(t["id"] for t in evd[c]["terms"][:3])
        tnames = self._track_names(tracks)
        enames = {kd: self._entity_names(kd, ids) for kd, ids in ents.items()}

        def track(i):
            return tnames.get(int(i), f"item {int(i)}")

        def units(n, word):
            return f"{n} {word}" + ("" if n == 1 else "s")

        lines = [
            "## 5. Named examples",
            "",
            "Positive pairs with the most concepts that have data. "
            "Then two uniform negatives: one drawn at random, the "
            "usual case, and the one with the strongest history "
            "match. A negative with a history match is a track that "
            "the listener did not play in the data, not a track that "
            "the listener dislikes (Pan et al. 2008). Each line shows "
            "the concept's state, its value, how many units carry it "
            'and the three largest. "no match": the data exists '
            "and shares nothing with the listener's history.",
            "",
        ]
        for ex in chosen:
            user, item, evd, role = (ex["user"], ex["item"], ex["evd"], ex["role"])
            lines += [
                f"### {track(item)} (item {item}, listener {user}, " f"{role})",
                "",
                "| concept | state | value | top evidence |",
                "|---|---|---:|---|",
            ]
            for c in cn.CONTINUOUS:
                entry = evd[c]
                if not entry["has_data"]:
                    lines.append(f"| {c} | no data | | |")
                    continue
                value = float(entry["value"])
                state = "no match" if value == 0 else "match"
                if c == "popularity":
                    state = "value"
                    top = f"{math.expm1(value):,.0f} train listens"
                elif c in cn.TERM_CONCEPTS:
                    names = src.term_names.get(cn.TERM_CONCEPTS[c], [])
                    terms = entry["terms"]
                    top = (
                        f"{units(len(terms), 'shared term')}: "
                        + ", ".join(
                            f"{names[t['id']] if t['id'] < len(names) else t['id']}"
                            f" ({t['term']:.3f}, share {t['listening_share']:.3f})"
                            for t in terms[:3]
                        )
                        if terms
                        else "no term shared with the listener's history"
                    )
                elif c in cn.CREDIT_CONCEPTS:
                    kd = cn.CREDIT_CONCEPTS[c]
                    mb_id = int(src.credit_ids[kd][entry["credit"]])
                    name = enames[kd].get(mb_id, f"{kd[:-1]} {mb_id}")
                    terms = entry["terms"]
                    top = (
                        f"{name}, {units(len(terms), 'history track')}: "
                        + ", ".join(
                            f"{track(t['id'])} ({t['term']:.4f})" for t in terms[:3]
                        )
                        if terms
                        else f"{name}: not in the listener's history"
                    )
                elif c in cn.COLISTEN_CONCEPTS:
                    terms = entry["terms"]
                    top = (
                        f"{units(len(terms), 'history track')}: "
                        + ", ".join(
                            f"{track(t['id'])} ({t['term']:.5f})" for t in terms[:3]
                        )
                        if terms
                        else "no neighbour in the listener's history"
                    )
                else:
                    state = "value"
                    top = ", ".join(
                        f"{t['id'][2:]} (z {t['z']:+.2f})" for t in entry["terms"][:3]
                    )
                lines.append(
                    f"| {c} | {state} | {value:.4f} | " f"{top.replace('|', '/')} |"
                )
            lines.append(f"| match_tier | {evd['match_tier']['value']} | | |")
            lines.append("")
        return lines

"""Stage P11: the fusion model's training pairs. pipeline/fusion_pairs.py
holds the method and its reasons.

Output: fusion_pairs.npz, grouped by user:
- params: JSON string of the stage parameters.
- counts: JSON string of the rule counts, for the report.
- user_id (int64, sorted), split (int8: 0 train, 1 selection).
- indptr (int64): rows of user k are indptr[k]..indptr[k+1]-1.
- item (int32), kind (int8: 0 positive, 1 uniform negative, 2 p_pop
  negative), pick (int8: for a positive, 1 = val_last, 2 = val_first,
  3 = both), draw (int16: a negative's place in its user's stream).
  Each user holds its positives first (sorted by item), then
  NEG_PER_KIND uniform draws, then NEG_PER_KIND p_pop draws.
- n_allowed (int64), pop_allowed (float64): per user, the candidate
  slots and the p_pop mass left after the exclusions. They give each
  negative's exact sampling probability (fusion_pairs.py).
- consumed (int64, users x 2): raw draws used per proposal, so the
  report can show how often a draw hit an excluded track.

reports/p11_report.md: the counts, the sampling diagnostics, and a
sampler audit. The audit compares positives, both negative kinds and
the training users' test picks on popularity, bridge tier and data
coverage. The fusion model learns its shape functions and its
missing-state values from the contrast between positives and negatives.
So the audit shows, before any training, which contrasts each sampler
offers the model, and which of them a learned missing state could turn
into a popularity signal.

The eval users do not appear in any output. Their rows of
splits.parquet are dropped right after loading, before any pick is read.
The report reads the test picks of the training users only.

Known limit, shared by every model in the comparison: P2 holds out the
last picks of each user, not a global time cut. So train holds some
listens that come after another user's val pick, and features such as
popularity and co-listening see them (Ji et al. 2023). The fusion model
sees the same features as the baselines, so the comparison stays fair.
"""

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import evaluation as ev
from . import fusion_pairs as fp
from . import selection as sel
from .als import Progress
from .common import atomic_text_write, check, connect, file_sha256, log, marker_path
from .config import MODEL_DIR

NEG_PER_KIND = 16  # negatives per user and proposal
SEED = 20260924
VERSION = "fusion-pairs-v1"
TASKS_PER_WORKER = 8  # user blocks per worker, for load balance
VERIFY_REDRAW_USERS = 200  # users recomputed by the slow reference
VERIFY_SPOT_PAIRS = 2000  # negatives, and positives, in the train files
WORKERS = max(1, min(16, os.cpu_count() or 1))
BANDS = ("01-10", "11-50", "51-90", "91-100")
TIER_GROUPS = (
    ("isrc, isrc+url, url", (1, 2, 3)),
    ("fuzzy_dur", (4,)),
    ("fuzzy_name", (5,)),
    ("unmatched", (6,)),
)
COVERAGE = (
    ("audio", "has_audio"),
    ("direct tags", "has_tags_direct"),
    ("artist tags", "has_tags_artist"),
    ("genres", "has_genres"),
    ("work", "has_work"),
)

_CTX = {}


def _params():
    return {
        "version": VERSION,
        "neg_per_kind": NEG_PER_KIND,
        "seed": SEED,
        "selection_seed": sel.SEED,
        "selection_size": sel.SIZE,
    }


def _atomic_savez(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez keeps .npz
    np.savez(tmp, **arrays)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _run_tasks(fn, tasks, costs, workers, prog):
    """fn over tasks, results in task order. Forked workers share _CTX.
    Each result depends only on its task, so the worker count cannot
    change the output."""
    results = []
    if workers <= 1 or len(tasks) <= 1:
        for task, cost in zip(tasks, costs):
            results.append(fn(task))
            prog.advance(cost)
        return results
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("fork")
    ) as pool:
        for out, cost in zip(pool.map(fn, tasks), costs):
            results.append(out)
            prog.advance(cost)
    return results


def _blocks(costs, n_tasks):
    """Contiguous index ranges with near-equal summed cost."""
    cum = np.cumsum(np.asarray(costs, dtype=np.int64))
    if len(cum) == 0:
        return []
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n_tasks) / n_tasks)
    edges = np.unique(np.concatenate(([0], cuts, [len(cum)])))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _user_block(task):
    """Rows for the users lo..hi-1 of _CTX["users"]."""
    lo, hi = task
    c = _CTX
    u_indptr, u_indices = c["u_indptr"], c["u_indices"]
    items, kinds, picks, draws = [], [], [], []
    n = hi - lo
    lens = np.zeros(n, dtype=np.int64)
    n_allowed = np.zeros(n, dtype=np.int64)
    pop_allowed = np.zeros(n, dtype=np.float64)
    consumed = np.zeros((n, 2), dtype=np.int64)
    for j, k in enumerate(range(lo, hi)):
        u = int(c["users"][k])
        row = u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])]
        a, b = int(c["p_indptr"][k]), int(c["p_indptr"][k + 1])
        out = fp.user_rows(
            c["seed"],
            u,
            c["p_items"][a:b],
            c["p_flags"][a:b],
            row,
            c["held"][k],
            c["k"],
            c["samplers"],
            c["n_candidates"],
            c["p_pop_slot"],
            c["pop_total"],
        )
        items.append(out[0])
        kinds.append(out[1])
        picks.append(out[2])
        draws.append(out[3])
        lens[j] = len(out[0])
        n_allowed[j], pop_allowed[j] = out[4], out[5]
        consumed[j] = (out[6][fp.KIND_UNIFORM], out[6][fp.KIND_POP])
    return (
        np.concatenate(items),
        np.concatenate(kinds),
        np.concatenate(picks),
        np.concatenate(draws),
        lens,
        n_allowed,
        pop_allowed,
        consumed,
    )


def _train_check_block(task):
    """Rows lo..hi-1 of the stored users: count the stored items found in
    the user's train row, and the users with an empty train row."""
    lo, hi = task
    c = _CTX
    u_indptr, u_indices = c["u_indptr"], c["u_indices"]
    hits = empty = 0
    for k in range(lo, hi):
        u = int(c["users"][k])
        row = np.asarray(u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])])
        empty += int(len(row) == 0)
        got = c["item"][int(c["indptr"][k]) : int(c["indptr"][k + 1])]
        hits += int(fp.member(got.astype(row.dtype), row).sum())
    return hits, empty


def _band(pop_centile):
    c = np.asarray(pop_centile)
    return np.where(c <= 10, 0, np.where(c <= 50, 1, np.where(c <= 90, 2, 3)))


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


class StageP11:
    name = "p11_fusion_pairs"
    upstream = ["p2_split", "p3_sampling", "p6_als"]

    def outputs(self):
        return {
            "pairs": MODEL_DIR / "fusion_pairs.npz",
            "report": MODEL_DIR / "reports" / "p11_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        # evaluation.py supplies only file paths and the slot range, which
        # the upstream markers pin. Its hash would rerun P11, and every
        # stage after it, for each unrelated change to the P14 code.
        for label, mod in (("fusion_pairs_module", fp), ("selection_module", sel)):
            inputs[label] = file_sha256(Path(mod.__file__))
        return inputs

    # The loaders below read DuckDB tables and model files. test_p11.py
    # replaces them.
    def _splits(self):
        con = connect(4)
        try:
            cols = con.execute(
                f"SELECT user_id, item_id, is_test_last, is_val_last, "
                f"is_test_first, is_val_first "
                f"FROM read_parquet('{ev.SPLITS}')"
            ).fetchnumpy()
        finally:
            con.close()
        return tuple(
            np.asarray(cols[k])
            for k in (
                "user_id",
                "item_id",
                "is_test_last",
                "is_val_last",
                "is_test_first",
                "is_val_first",
            )
        )

    def _eval_users(self):
        con = connect(4)
        try:
            return np.asarray(
                con.execute(
                    f"SELECT user_id FROM read_parquet('{ev.EVAL_USERS}')"
                ).fetchnumpy()["user_id"],
                dtype=np.int64,
            )
        finally:
            con.close()

    def _selection_users(self):
        """selection.py's own DuckDB path, for verify()."""
        con = connect(4)
        try:
            return np.asarray(sel.selection_user_ids(con), dtype=np.int64)
        finally:
            con.close()

    def _proposals(self):
        z = np.load(MODEL_DIR / "neg_proposals.npz")
        return (
            np.asarray(z["item_ids"], dtype=np.int64),
            np.asarray(z["p_pop"], dtype=np.float64),
            np.asarray(z["cum_pop"], dtype=np.float64),
        )

    def _csr(self):
        return (
            np.load(MODEL_DIR / "csr_user_indptr.npy"),
            np.load(MODEL_DIR / "csr_user_indices.npy", mmap_mode="r"),
        )

    def _id_slots(self):
        return ev.load_id_slots()

    def _covariates(self, n_slots):
        """Per item slot: pop centile, tier code (1..6, 0 for the phantom
        slot), train pairs and the coverage flags of COVERAGE."""
        con = connect(4)
        try:
            cols = con.execute(f"""
                SELECT item_id, pop_centile, train_pairs,
                       CASE tier WHEN 'isrc' THEN 1 WHEN 'isrc+url' THEN 2
                                 WHEN 'url' THEN 3 WHEN 'fuzzy_dur' THEN 4
                                 WHEN 'fuzzy_name' THEN 5 ELSE 6 END AS tier,
                       has_audio, has_tags_direct, has_tags_artist,
                       has_genres, has_work
                FROM read_parquet('{ev.COVARIATES}')
            """).fetchnumpy()
        finally:
            con.close()
        ids = np.asarray(cols["item_id"], dtype=np.int64)
        out = {}
        for key, dtype in (
            ("pop_centile", np.int64),
            ("train_pairs", np.int64),
            ("tier", np.int64),
        ):
            arr = np.zeros(n_slots, dtype=dtype)
            arr[ids] = np.asarray(cols[key], dtype=dtype)
            out[key] = arr
        for _, key in COVERAGE:
            arr = np.zeros(n_slots, dtype=bool)
            arr[ids] = np.asarray(cols[key], dtype=bool)
            out[key] = arr
        return out

    def _train_hits(self, users, items):
        """How many of the (user, item) pairs exist in the train parquet
        files. An independent check of the CSR route."""
        con = connect(4)
        try:
            con.execute(
                "CREATE OR REPLACE TABLE p11_spot " "(user_id BIGINT, item_id BIGINT)"
            )
            con.executemany(
                "INSERT INTO p11_spot VALUES (?, ?)",
                list(zip(np.asarray(users).tolist(), np.asarray(items).tolist())),
            )
            return int(con.execute(f"""
                SELECT count(*) FROM p11_spot s
                JOIN read_parquet('{ev.TRAIN_GLOB}') t
                  ON s.user_id = t.user_id AND s.item_id = t.item_id
            """).fetchone()[0])
        finally:
            con.close()

    # Run

    def _roles(self):
        """The users of the output with their split, picks and positives,
        and the rule counts. The eval users' rows of splits.parquet are
        dropped before any pick is read."""
        cols = self._splits()
        evals = np.unique(self._eval_users())
        eligible = np.unique(np.asarray(cols[0], dtype=np.int64))
        selection = sel.select(eligible, evals, seed=sel.SEED, size=sel.SIZE)
        keep_rows = ~np.isin(np.asarray(cols[0], dtype=np.int64), evals)
        users, picks = fp.pick_table(*(np.asarray(c)[keep_rows] for c in cols))
        p_indptr, p_items, p_flags, counts = fp.positives(picks)
        has_pos = np.diff(p_indptr) > 0
        rows = np.repeat(has_pos, np.diff(p_indptr))
        p_items, p_flags = p_items[rows], p_flags[rows]
        users, picks = users[has_pos], picks[has_pos]
        p_indptr = np.concatenate(([0], np.cumsum(np.diff(p_indptr)[has_pos]))).astype(
            np.int64
        )
        split = np.where(
            np.isin(users, selection), fp.SPLIT_SELECTION, fp.SPLIT_TRAIN
        ).astype(np.int8)
        counts.update(
            eligible_users=int(len(eligible)),
            eval_users_left_out=int(np.isin(eligible, evals).sum()),
            selection_users=int(len(selection)),
        )
        return users, split, picks, (p_indptr, p_items, p_flags), counts

    def run(self, params, files):
        outputs = self.outputs()
        users, split, picks, (p_indptr, p_items, p_flags), counts = self._roles()
        n_slots, id_min = self._id_slots()
        u_indptr, u_indices = self._csr()
        check(
            len(u_indptr) - 1 > int(users.max()),
            "a user id falls outside the user-major CSR",
        )
        item_ids, p_pop, cum_pop = self._proposals()
        check(
            int(item_ids.min()) >= id_min and int(item_ids.max()) < n_slots,
            "a proposal item falls outside the item slots",
        )
        p_pop_slot = np.zeros(n_slots, dtype=np.float64)
        p_pop_slot[item_ids] = p_pop
        samplers = {
            fp.KIND_UNIFORM: fp.uniform_sampler(id_min, n_slots),
            fp.KIND_POP: fp.pop_sampler(item_ids, cum_pop),
        }
        hist = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
            u_indptr[users], dtype=np.int64
        )
        costs = hist + 2000  # the draws cost about 2,000 entries
        tasks = _blocks(costs, WORKERS * TASKS_PER_WORKER)
        _CTX.update(
            users=users,
            u_indptr=u_indptr,
            u_indices=u_indices,
            p_indptr=p_indptr,
            p_items=p_items,
            p_flags=p_flags,
            held=picks,
            k=NEG_PER_KIND,
            seed=SEED,
            samplers=samplers,
            n_candidates=n_slots - id_min,
            p_pop_slot=p_pop_slot,
            pop_total=float(np.sum(p_pop)),
        )
        log.info(
            "p11: %d users (%d train, %d selection), %d positives, "
            "%d negatives per user and proposal, %d workers",
            len(users),
            int((split == fp.SPLIT_TRAIN).sum()),
            int((split == fp.SPLIT_SELECTION).sum()),
            len(p_items),
            NEG_PER_KIND,
            WORKERS,
        )
        prog = Progress(
            "p11: sampling negatives",
            int(costs.sum()),
            log.info,
            unit=" M work units",
            scale=10**6,
        )
        try:
            parts = _run_tasks(
                _user_block,
                tasks,
                [int(costs[a:b].sum()) for a, b in tasks],
                WORKERS,
                prog,
            )
        finally:
            _CTX.clear()
        prog.finish()
        item = np.concatenate([p[0] for p in parts])
        lens = np.concatenate([p[4] for p in parts])
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        check(item.max() < np.iinfo(np.int32).max, "an item id does not fit int32")
        _atomic_savez(
            outputs["pairs"],
            params=np.array(json.dumps(_params())),
            counts=np.array(json.dumps(counts)),
            user_id=users,
            split=split,
            indptr=indptr,
            item=item.astype(np.int32),
            kind=np.concatenate([p[1] for p in parts]),
            pick=np.concatenate([p[2] for p in parts]),
            draw=np.concatenate([p[3] for p in parts]),
            n_allowed=np.concatenate([p[5] for p in parts]),
            pop_allowed=np.concatenate([p[6] for p in parts]),
            consumed=np.concatenate([p[7] for p in parts]),
        )
        log.info("p11: %d rows written", len(item))
        return {
            "pairs": {
                "path": str(outputs["pairs"]),
                "rows": int(len(item)),
                "sha256": ev.npz_hash(outputs["pairs"]),
            }
        }

    def _load(self):
        z = np.load(self.outputs()["pairs"])
        out = {k: z[k] for k in z.files if k not in ("params", "counts")}
        return out, json.loads(str(z["params"])), json.loads(str(z["counts"]))

    def verify(self, params):
        d, stored, counts = self._load()
        check(
            stored == _params(),
            f"fusion_pairs.npz was built with "
            f"{stored}, the stage now says "
            f"{_params()}",
        )
        users, split, indptr = d["user_id"], d["split"], d["indptr"]
        item = d["item"].astype(np.int64)
        kind, pick, draw = d["kind"], d["pick"], d["draw"]
        n_users, n_rows, k = len(users), len(item), NEG_PER_KIND
        check(
            n_users > 0 and bool((np.diff(users) > 0).all()),
            "user ids are not sorted and unique",
        )
        check(
            len(indptr) == n_users + 1
            and indptr[0] == 0
            and indptr[-1] == n_rows
            and bool((np.diff(indptr) > 0).all()),
            "indptr does not cover the rows",
        )
        for name in ("kind", "pick", "draw"):
            check(len(d[name]) == n_rows, f"{name} has the wrong length")
        for name in ("split", "n_allowed", "pop_allowed", "consumed"):
            check(len(d[name]) == n_users, f"{name} has the wrong length")

        # The roles, from the loaders, with selection.py's own route.
        evals = np.unique(self._eval_users())
        selection = np.unique(self._selection_users())
        check(
            not np.isin(users, evals).any(), "an eval user appears in the fusion pairs"
        )
        in_sel = np.isin(users, selection)
        check(
            bool(np.array_equal(split == fp.SPLIT_SELECTION, in_sel)),
            "the selection flag disagrees with selection.py",
        )
        r_users, r_split, picks, (p_indptr, p_items, p_flags), _ = self._roles()
        check(
            np.array_equal(users, r_users) and np.array_equal(split, r_split),
            "the stored users differ from the eligible users outside the "
            "eval sample that have a positive",
        )

        # Layout: positives, then k uniform draws, then k p_pop draws.
        lens = np.diff(indptr)
        n_pos = np.diff(p_indptr)
        check(
            np.array_equal(lens, n_pos + 2 * k),
            "a user does not hold its positives plus two sets of draws",
        )
        start = np.repeat(indptr[:-1], lens)
        pos = np.arange(n_rows, dtype=np.int64) - start
        npos_row = np.repeat(n_pos, lens)
        rel = pos - npos_row
        want_kind = np.where(
            rel < 0, fp.KIND_POSITIVE, np.where(rel < k, fp.KIND_UNIFORM, fp.KIND_POP)
        )
        want_draw = np.where(rel < 0, 0, rel % k)
        check(
            np.array_equal(kind, want_kind.astype(kind.dtype))
            and np.array_equal(draw, want_draw.astype(draw.dtype)),
            "a row has the wrong kind or draw number for its place",
        )
        is_pos = kind == fp.KIND_POSITIVE
        check(
            np.array_equal(item[is_pos], p_items)
            and np.array_equal(pick[is_pos], p_flags)
            and not pick[~is_pos].any(),
            "the positives differ from the val picks under the rules",
        )

        n_slots, id_min = self._id_slots()
        check(
            int(item.min()) >= id_min and int(item.max()) < n_slots,
            "an item falls outside the candidate slots",
        )
        item_ids, p_pop, cum_pop = self._proposals()
        check(
            bool(np.isin(item[kind == fp.KIND_POP], item_ids).all()),
            "a p_pop negative is not a P3 proposal item",
        )

        # No negative is a held-out pick of its user.
        row_user = np.repeat(np.arange(n_users), lens)
        neg = ~is_pos
        held_hit = np.zeros(int(neg.sum()), dtype=bool)
        for col in range(4):
            held_hit |= item[neg] == picks[row_user[neg], col]
        check(
            not held_hit.any(),
            f"{int(held_hit.sum())} negatives are held-out picks of their " f"user",
        )

        # No stored item is in its user's train row: all users, via the
        # CSR.
        u_indptr, u_indices = self._csr()
        hist = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
            u_indptr[users], dtype=np.int64
        )
        costs = hist + lens
        tasks = _blocks(costs, WORKERS * TASKS_PER_WORKER)
        _CTX.update(
            users=users,
            u_indptr=u_indptr,
            u_indices=u_indices,
            item=item,
            indptr=indptr,
        )
        prog = Progress(
            "p11 verify: no stored item in train",
            int(costs.sum()),
            log.info,
            unit=" M entries",
            scale=10**6,
        )
        try:
            res = _run_tasks(
                _train_check_block,
                tasks,
                [int(costs[a:b].sum()) for a, b in tasks],
                WORKERS,
                prog,
            )
        finally:
            _CTX.clear()
        prog.finish()
        hits, empty = sum(r[0] for r in res), sum(r[1] for r in res)
        check(empty == 0, f"{empty} stored users have no train items")
        check(hits == 0, f"{hits} stored items are train items of their " f"user")

        # Independent route: the train parquet files, in one scan.
        rng = np.random.default_rng(SEED + 1)
        take = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(mask),
                    size=min(VERIFY_SPOT_PAIRS, int(mask.sum())),
                    replace=False,
                )
                for mask in (neg, is_pos)
            ]
        )
        n_hit = self._train_hits(users[row_user[take]], item[take])
        check(
            n_hit == 0,
            f"{n_hit} of {len(take)} sampled pairs (negatives "
            f"and positives) exist in the train parquet files",
        )

        # Exact recomputation of whole users by the slow reference.
        p_pop_slot = np.zeros(n_slots, dtype=np.float64)
        p_pop_slot[item_ids] = p_pop
        pop_total = float(np.sum(p_pop))
        samplers = {
            fp.KIND_UNIFORM: fp.uniform_sampler(id_min, n_slots),
            fp.KIND_POP: fp.pop_sampler(item_ids, cum_pop),
        }
        pick_users = rng.choice(
            n_users, size=min(VERIFY_REDRAW_USERS, n_users), replace=False
        )
        for j in np.sort(pick_users):
            u = int(users[j])
            row = np.asarray(u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])])
            excl = set(row.tolist()) | set(picks[j].tolist())
            s = int(indptr[j]) + int(n_pos[j])
            for col, kd in enumerate((fp.KIND_UNIFORM, fp.KIND_POP)):
                want, used = fp.reference_draws(SEED, u, kd, k, samplers[kd], excl)
                got = item[s + col * k : s + (col + 1) * k]
                check(
                    np.array_equal(got, want) and int(d["consumed"][j, col]) == used,
                    f"user {u}: the {fp.KIND_NAMES[kd]} draws differ "
                    f"from a fresh recomputation",
                )
            check(
                int(d["n_allowed"][j]) == n_slots - id_min - len(excl),
                f"user {u}: n_allowed is wrong",
            )
            want_mass = pop_total - float(
                np.sum(p_pop_slot[np.fromiter(excl, dtype=np.int64)])
            )
            check(
                abs(float(d["pop_allowed"][j]) - want_mass)
                <= 1e-12 * max(1.0, pop_total),
                f"user {u}: pop_allowed is wrong",
            )
        check(
            bool((d["n_allowed"] > 0).all())
            and bool((d["pop_allowed"] > 0).all())
            and bool((d["consumed"] >= k).all()),
            "a user has no allowed candidates or too few raw draws",
        )
        log.info(
            "p11: verify passed. %d users, %d rows, %d users " "recomputed in full",
            n_users,
            n_rows,
            len(pick_users),
        )

    def report(self, params):
        d, stored, counts = self._load()
        users, split, indptr = d["user_id"], d["split"], d["indptr"]
        item = d["item"].astype(np.int64)
        kind, pick = d["kind"], d["pick"]
        lens = np.diff(indptr)
        n_users, k = len(users), NEG_PER_KIND
        row_user = np.repeat(np.arange(n_users), lens)
        row_split = split[row_user]
        n_slots, id_min = self._id_slots()
        cov = self._covariates(n_slots)
        r_users, _, picks, _, _ = self._roles()
        check(
            np.array_equal(r_users, users),
            "the stored users differ from the users the rules give",
        )
        is_pos = kind == fp.KIND_POSITIVE
        n_train = int((split == fp.SPLIT_TRAIN).sum())
        n_sel = int((split == fp.SPLIT_SELECTION).sum())
        both = fp.PICK_VAL_LAST | fp.PICK_VAL_FIRST

        lines = [
            "# P11 report: fusion training pairs",
            "",
            f"- parameters: {json.dumps(stored)}",
            f"- eligible users (P2): {counts['eligible_users']:,}. "
            f"Eval users left out: {counts['eval_users_left_out']:,}. "
            f"Selection users: {counts['selection_users']:,}.",
            f"- users with pairs: {n_users:,} ({n_train:,} train, "
            f"{n_sel:,} selection). Users without a positive after "
            f"the rules: {counts['no_positive']:,}.",
            f"- rows: {len(item):,}. Positives: {int(is_pos.sum()):,}. "
            f"Negatives per user: {k} uniform and {k} p_pop.",
            f"- rule 3: val_last dropped because it is the test_first "
            f"pick for {counts['val_last_is_test_first']:,} users. "
            f"val_first dropped because it is the test_last pick for "
            f"{counts['val_first_is_test_last']:,} users.",
            f"- rule 4: val_last and val_first are one track for "
            f"{counts['val_last_is_val_first']:,} users. That track is "
            f"one positive with both flags.",
            f"- positives by pick: val_last only "
            f"{int((pick[is_pos] == fp.PICK_VAL_LAST).sum()):,}, "
            f"val_first only "
            f"{int((pick[is_pos] == fp.PICK_VAL_FIRST).sum()):,}, both "
            f"{int((pick[is_pos] == both).sum()):,}.",
            "",
        ]

        # Sampling diagnostics.
        lines += ["## Sampling diagnostics", ""]
        for col, kd in enumerate((fp.KIND_UNIFORM, fp.KIND_POP)):
            rate = k / d["consumed"][:, col]
            sel_rows = kind == kd
            key = row_user[sel_rows] * np.int64(n_slots) + item[sel_rows]
            dup = len(key) - len(np.unique(key))
            lines.append(
                f"- {fp.KIND_NAMES[kd]}: {int(sel_rows.sum()):,} draws. "
                f"Share of raw draws accepted, per user: median "
                f"{np.median(rate):.3f}, p10 {np.percentile(rate, 10):.3f}, "
                f"p1 {np.percentile(rate, 1):.3f}, minimum "
                f"{rate.min():.3f}. Repeated tracks within a user: {dup:,} "
                f"({_pct(dup, len(key))})."
            )
        pop_total = float(np.sum(self._proposals()[1]))
        excl = 1.0 - d["pop_allowed"] / pop_total
        lines += [
            f"- p_pop mass on tracks a user may not draw (own train items "
            f"and held-out picks), per user: median {np.median(excl):.3f}, "
            f"p90 {np.percentile(excl, 90):.3f}, p99 "
            f"{np.percentile(excl, 99):.3f}, maximum {excl.max():.3f}.",
            f"- candidate slots left per user: minimum "
            f"{int(d['n_allowed'].min()):,} of {n_slots - id_min:,}.",
            "",
        ]

        # Sampler audit.
        groups = [
            ("positives, train users", item[is_pos & (row_split == fp.SPLIT_TRAIN)]),
            (
                "positives, selection users",
                item[is_pos & (row_split == fp.SPLIT_SELECTION)],
            ),
            ("uniform negatives", item[kind == fp.KIND_UNIFORM]),
            ("p_pop negatives", item[kind == fp.KIND_POP]),
            ("test_last picks, train users", picks[split == fp.SPLIT_TRAIN, 0]),
            ("test_first picks, train users", picks[split == fp.SPLIT_TRAIN, 2]),
        ]
        lines += [
            "## Sampler audit",
            "",
            "The model learns each shape function and each missing "
            "state from the contrast between positives and "
            "negatives. Where a negative kind differs from the "
            "positives in coverage, a learned missing state can "
            "absorb the difference and act as a popularity signal. "
            "P13 compares its learned missing values with these "
            "rows. The last two rows are the test picks of the "
            "training users, never of the eval users. They show how "
            "far the val picks, which the model learns from, sit "
            "from the test picks, which it is judged on.",
            "",
            "Popularity (P3 pop centile of train pairs):",
            "",
            "| group | rows | "
            + " | ".join(f"band {b}" for b in BANDS)
            + " | 0 train pairs | median train pairs |",
            "|---|---:|" + "---:|" * (len(BANDS) + 2),
        ]
        for label, its in groups:
            n = len(its)
            band = _band(cov["pop_centile"][its])
            tp = cov["train_pairs"][its]
            lines.append(
                f"| {label} | {n:,} | "
                + " | ".join(_pct(int((band == b).sum()), n) for b in range(len(BANDS)))
                + f" | {_pct(int((tp == 0).sum()), n)} | "
                f"{np.median(tp) if n else 0:,.0f} |"
            )
        lines += [
            "",
            "Bridge tier and data coverage:",
            "",
            "| group | "
            + " | ".join(t for t, _ in TIER_GROUPS)
            + " | "
            + " | ".join(c for c, _ in COVERAGE)
            + " |",
            "|---|" + "---:|" * (len(TIER_GROUPS) + len(COVERAGE)),
        ]
        for label, its in groups:
            n = len(its)
            tier = cov["tier"][its]
            lines.append(
                f"| {label} | "
                + " | ".join(
                    _pct(int(np.isin(tier, codes).sum()), n) for _, codes in TIER_GROUPS
                )
                + " | "
                + " | ".join(_pct(int(cov[key][its].sum()), n) for _, key in COVERAGE)
                + " |"
            )
        lines.append("")
        atomic_text_write("\n".join(lines) + "\n", self.outputs()["report"])
        log.info("p11 report written")

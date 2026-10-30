"""Stage P11b: more negatives for the fusion training pairs.

P13b part 1 found that the loss form decides the popularity and the
missing-data lines of the explanations, and that form A gains from more
negatives. Part 2 needs two sets of pairs:
1. wide: every train-split listener of fusion_pairs.npz with WIDE_PER_KIND
   draws from each proposal, for the sweep over the number of negatives.
2. ref: REF_USERS of those listeners with REF_PER_KIND draws from each
   proposal, for the large-m reference of the loss form.

The draws come from P11's own random streams, fusion_pairs.user_rng(SEED,
user, kind) with P11's SEED, through P11's own function
fusion_pairs.user_rows(). draw_allowed() reads each stream one double per
draw, so a longer draw starts with the shorter one: the first 16 draws of
each kind equal P11's stored draws, and the first WIDE_PER_KIND draws of a
reference listener equal the listener's wide draws. verify() checks both
prefixes row by row. The exclusions (the listener's train items and four
held-out picks), n_allowed and pop_allowed are P11's.

Outputs, with the arrays of fusion_pairs.npz (params, counts, user_id,
split, indptr, item, kind, pick, draw, n_allowed, pop_allowed, consumed)
and pairs_sha256, the npz hash of the fusion_pairs.npz they extend. Each
listener holds its positives first (sorted by item), then the uniform
draws, then the p_pop draws, each in draw order:
- fusion_pairs_wide.npz,
- fusion_pairs_ref.npz,
- reports/p11b_report.md.

The reference listeners: the REF_USERS train-split listeners with the
smallest SHA-256 of "REF_SALT:user_id". The rule needs no stored random
state, as selection.py.

verify():
- the parameters, and the hash of the fusion_pairs.npz the files extend.
- the layout of every listener (kind and draw number per place).
- the positives, n_allowed and pop_allowed equal P11's, and the first 16
  draws of each kind equal P11's rows.
- the first WIDE_PER_KIND draws of each reference listener equal the
  wide rows.
- no negative is a held-out pick of its listener, no stored item is a
  train item of its listener (all rows, through the user CSR), every
  p_pop draw is a P3 proposal item.
- a slow one-double-at-a-time recomputation of whole listeners
  (fusion_pairs.reference_draws) gives the same draws and raw counts.
- the reference listeners follow the SHA-256 rule.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import evaluation as ev
from . import fusion_pairs as fp
from . import selection as sel
from . import stage_p11 as s11
from .als import Progress
from .common import atomic_text_write, check, file_sha256, log, marker_path
from .config import MODEL_DIR

VERSION = "fusion-pairs-more-v1"
WIDE_PER_KIND = 64  # negatives per listener and proposal, wide
REF_PER_KIND = 4096  # negatives per listener and proposal, ref
REF_USERS = 5000
REF_SALT = "p13b2-reference-20260928"
DRAW_COST = 125  # work units per draw (P11: 2,000 for 16)
VERIFY_WIDE_USERS = 40  # listeners recomputed by the slow reference
VERIFY_REF_USERS = 4
VERIFY_SEED = 2026092811


def _params():
    return {
        "version": VERSION,
        "wide_per_kind": WIDE_PER_KIND,
        "ref_per_kind": REF_PER_KIND,
        "ref_users": REF_USERS,
        "ref_salt": REF_SALT,
        "p11": s11._params(),
    }


def ref_order(users, salt=None):
    """The listeners sorted by SHA-256 of "salt:user_id", the order of the
    reference rule."""
    salt = REF_SALT if salt is None else salt
    keys = [
        hashlib.sha256(f"{salt}:{int(u)}".encode("utf-8")).hexdigest()
        for u in np.asarray(users)
    ]
    return np.asarray(users)[np.argsort(np.array(keys), kind="stable")]


def subset_csr(indptr, cols, mask):
    """The rows of a CSR (indptr over rows, then arrays of entries) whose
    mask is set, as a new CSR."""
    indptr = np.asarray(indptr, dtype=np.int64)
    lens = np.diff(indptr)[mask]
    take = np.repeat(mask, np.diff(indptr))
    new_ptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    return (new_ptr,) + tuple(np.asarray(c)[take] for c in cols)


def layout_arrays(indptr, n_pos, k):
    """(kind, draw) that P11's layout gives each row: positives, then k
    uniform draws, then k p_pop draws."""
    indptr = np.asarray(indptr, dtype=np.int64)
    lens = np.diff(indptr)
    start = np.repeat(indptr[:-1], lens)
    rel = (
        np.arange(int(indptr[-1]), dtype=np.int64)
        - start
        - np.repeat(np.asarray(n_pos, dtype=np.int64), lens)
    )
    kind = np.where(
        rel < 0, fp.KIND_POSITIVE, np.where(rel < k, fp.KIND_UNIFORM, fp.KIND_POP)
    )
    draw = np.where(rel < 0, 0, rel % k)
    return kind.astype(np.int8), draw.astype(np.int16)


def prefix_rows(indptr, n_pos, k_have, k_want):
    """Row indices of each listener's positives and first k_want draws of
    each kind, in a file with k_have draws per kind, in layout order."""
    indptr = np.asarray(indptr, dtype=np.int64)
    n_pos = np.asarray(n_pos, dtype=np.int64)
    per = n_pos + 2 * k_want
    start = np.repeat(indptr[:-1], per)
    npos_r = np.repeat(n_pos, per)
    rel = np.arange(int(per.sum()), dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(per)[:-1])), per
    )
    # rel < n_pos: positive rel. Then k_want uniform, then k_want p_pop.
    off = np.where(
        rel < npos_r, rel, np.where(rel < npos_r + k_want, rel, rel - k_want + k_have)
    )
    return start + off


class StageP11b(s11.StageP11):
    name = "p11b_more_pairs"
    upstream = ["p2_split", "p3_sampling", "p6_als", "p11_fusion_pairs"]

    def __init__(self):
        self._verified = None

    def outputs(self):
        return {
            "wide": MODEL_DIR / "fusion_pairs_wide.npz",
            "ref": MODEL_DIR / "fusion_pairs_ref.npz",
            "report": MODEL_DIR / "reports" / "p11b_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        for label, mod in (
            ("fusion_pairs_module", fp),
            ("selection_module", sel),
            ("stage_p11_module", s11),
        ):
            inputs[label] = file_sha256(Path(mod.__file__))
        return inputs

    def _p11(self):
        """P11's stored pairs, its parameters and its npz hash."""
        path = MODEL_DIR / "fusion_pairs.npz"
        z = np.load(path)
        out = {k: np.asarray(z[k]) for k in z.files if k not in ("params", "counts")}
        return out, json.loads(str(z["params"])), ev.npz_hash(path)

    # Run

    def _train_roles(self, p11):
        """The train-split listeners of P11 with their held-out picks and
        positives, through P11's own rules."""
        users, split, picks, (p_indptr, p_items, p_flags), counts = self._roles()
        check(
            np.array_equal(users, p11["user_id"])
            and np.array_equal(split, p11["split"]),
            "P11's rules now give other listeners than fusion_pairs.npz " "holds",
        )
        tr = split == fp.SPLIT_TRAIN
        new_ptr, items, flags = subset_csr(p_indptr, (p_items, p_flags), tr)
        return users[tr], picks[tr], new_ptr, items, flags, counts

    def _draw(self, users, picks, p_indptr, p_items, p_flags, k, label):
        """P11's rows for the listeners with k draws per kind."""
        n_slots, id_min = self._id_slots()
        u_indptr, u_indices = self._csr()
        check(
            len(u_indptr) - 1 > int(users.max()),
            "a user id falls outside the user-major CSR",
        )
        item_ids, p_pop, cum_pop = self._proposals()
        p_pop_slot = np.zeros(n_slots, dtype=np.float64)
        p_pop_slot[item_ids] = p_pop
        samplers = {
            fp.KIND_UNIFORM: fp.uniform_sampler(id_min, n_slots),
            fp.KIND_POP: fp.pop_sampler(item_ids, cum_pop),
        }
        hist = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
            u_indptr[users], dtype=np.int64
        )
        costs = hist + DRAW_COST * 2 * k
        tasks = s11._blocks(costs, s11.WORKERS * s11.TASKS_PER_WORKER)
        s11._CTX.update(
            users=users,
            u_indptr=u_indptr,
            u_indices=u_indices,
            p_indptr=p_indptr,
            p_items=p_items,
            p_flags=p_flags,
            held=picks,
            k=k,
            seed=s11.SEED,
            samplers=samplers,
            n_candidates=n_slots - id_min,
            p_pop_slot=p_pop_slot,
            pop_total=float(np.sum(p_pop)),
        )
        prog = Progress(
            f"p11b: {label} draws, {k} per kind",
            int(costs.sum()),
            log.info,
            unit=" M work units",
            scale=10**6,
        )
        try:
            parts = s11._run_tasks(
                s11._user_block,
                tasks,
                [int(costs[a:b].sum()) for a, b in tasks],
                s11.WORKERS,
                prog,
            )
        finally:
            s11._CTX.clear()
        prog.finish()
        item = np.concatenate([p[0] for p in parts])
        lens = np.concatenate([p[4] for p in parts])
        check(item.max() < np.iinfo(np.int32).max, "an item id does not fit int32")
        return {
            "indptr": np.concatenate(([0], np.cumsum(lens))).astype(np.int64),
            "item": item.astype(np.int32),
            "kind": np.concatenate([p[1] for p in parts]),
            "pick": np.concatenate([p[2] for p in parts]),
            "draw": np.concatenate([p[3] for p in parts]),
            "n_allowed": np.concatenate([p[5] for p in parts]),
            "pop_allowed": np.concatenate([p[6] for p in parts]),
            "consumed": np.concatenate([p[7] for p in parts]),
        }

    def run(self, params, files):
        outputs = self.outputs()
        p11, p11_params, p11_sha = self._p11()
        check(
            p11_params == s11._params(),
            "fusion_pairs.npz was built with other P11 parameters",
        )
        users, picks, p_indptr, p_items, p_flags, counts = self._train_roles(p11)
        check(
            len(users) >= REF_USERS,
            f"only {len(users)} train listeners, fewer than REF_USERS",
        )
        log.info(
            "p11b: %d train listeners, %d per kind (wide), %d "
            "reference listeners with %d per kind",
            len(users),
            WIDE_PER_KIND,
            REF_USERS,
            REF_PER_KIND,
        )
        wide = self._draw(
            users, picks, p_indptr, p_items, p_flags, WIDE_PER_KIND, "wide"
        )
        ref_users = np.sort(ref_order(users)[:REF_USERS])
        at = np.searchsorted(users, ref_users)
        mask = np.zeros(len(users), dtype=bool)
        mask[at] = True
        r_ptr, r_items, r_flags = subset_csr(p_indptr, (p_items, p_flags), mask)
        ref = self._draw(
            ref_users, picks[mask], r_ptr, r_items, r_flags, REF_PER_KIND, "reference"
        )
        for label, users_, d in (("wide", users, wide), ("ref", ref_users, ref)):
            s11._atomic_savez(
                outputs[label],
                params=np.array(json.dumps(_params())),
                counts=np.array(json.dumps(counts)),
                pairs_sha256=np.array(p11_sha),
                user_id=users_,
                split=np.full(len(users_), fp.SPLIT_TRAIN, dtype=np.int8),
                **d,
            )
            log.info(
                "p11b: %s: %d listeners, %d rows written",
                label,
                len(users_),
                len(d["item"]),
            )
        return self._info(outputs)

    def _info(self, outputs):
        out = {}
        for label in ("wide", "ref"):
            z = np.load(outputs[label])
            out[label] = {
                "path": str(outputs[label]),
                "rows": int(len(z["item"])),
                "sha256": ev.npz_hash(outputs[label]),
            }
        return out

    def _load_b(self, label):
        z = np.load(self.outputs()[label])
        out = {
            k: np.asarray(z[k])
            for k in z.files
            if k not in ("params", "counts", "pairs_sha256")
        }
        return (
            out,
            json.loads(str(z["params"])),
            json.loads(str(z["counts"])),
            str(z["pairs_sha256"]),
        )

    def verify(self, params):
        self._verified = None
        p11, p11_params, p11_sha = self._p11()
        check(
            p11_params == s11._params(),
            "fusion_pairs.npz was built with other P11 parameters",
        )
        users, picks, p_indptr, p_items, p_flags, _ = self._train_roles(p11)
        n_pos = np.diff(p_indptr)
        files = {}
        for label, k in (("wide", WIDE_PER_KIND), ("ref", REF_PER_KIND)):
            d, stored, _, sha = self._load_b(label)
            check(
                stored == _params(),
                f"fusion_pairs_{label}.npz was built with {stored}, the "
                f"stage now says {_params()}",
            )
            check(
                sha == p11_sha,
                f"fusion_pairs_{label}.npz extends another " f"fusion_pairs.npz",
            )
            files[label] = (d, k)

        # The wide file: P11's train listeners, positives and exclusions.
        wide, kw = files["wide"]
        check(
            np.array_equal(wide["user_id"], users)
            and bool((wide["split"] == fp.SPLIT_TRAIN).all()),
            "the wide file holds other listeners than P11's train split",
        )
        tr11 = np.flatnonzero(p11["split"] == fp.SPLIT_TRAIN)
        for name in ("n_allowed", "pop_allowed"):
            check(
                np.array_equal(wide[name], p11[name][tr11]),
                f"the wide file's {name} differs from P11's",
            )
        self._check_layout(wide, n_pos, kw, p_items, p_flags, "wide")
        # Prefix: P11's rows of the train listeners are the first 16 draws.
        k11 = int(p11_params["neg_per_kind"])
        rows11 = np.flatnonzero(
            np.repeat(p11["split"] == fp.SPLIT_TRAIN, np.diff(p11["indptr"]))
        )
        rows_w = prefix_rows(wide["indptr"], n_pos, kw, k11)
        for name in ("item", "kind", "pick", "draw"):
            check(
                np.array_equal(wide[name][rows_w], p11[name][rows11]),
                f"the first {k11} draws of the wide file differ from "
                f"P11's rows ({name})",
            )

        # The reference file: the SHA rule and the wide prefix.
        ref, kr = files["ref"]
        want_ref = np.sort(ref_order(users)[:REF_USERS])
        check(
            np.array_equal(ref["user_id"], want_ref),
            "the reference listeners do not follow the SHA-256 rule",
        )
        at = np.searchsorted(users, ref["user_id"])
        mask = np.zeros(len(users), dtype=bool)
        mask[at] = True
        r_ptr, r_items, r_flags = subset_csr(p_indptr, (p_items, p_flags), mask)
        r_npos = np.diff(r_ptr)
        for name in ("n_allowed", "pop_allowed"):
            check(
                np.array_equal(ref[name], wide[name][at]),
                f"the reference file's {name} differs from the wide " f"file's",
            )
        self._check_layout(ref, r_npos, kr, r_items, r_flags, "ref")
        rows_r = prefix_rows(ref["indptr"], r_npos, kr, kw)
        rows_wr = np.flatnonzero(np.repeat(mask, np.diff(wide["indptr"])))
        for name in ("item", "kind", "pick", "draw"):
            check(
                np.array_equal(ref[name][rows_r], wide[name][rows_wr]),
                f"the first {kw} draws of the reference file differ from "
                f"the wide rows ({name})",
            )

        # Exclusions, proposal items and the train rows, for both files.
        u_indptr, u_indices = self._csr()
        n_slots, id_min = self._id_slots()
        item_ids, p_pop, cum_pop = self._proposals()
        for label, (d, k) in files.items():
            item = d["item"].astype(np.int64)
            kind = d["kind"]
            at_u = np.searchsorted(users, d["user_id"])
            lens = np.diff(d["indptr"])
            row_user = np.repeat(np.arange(len(d["user_id"])), lens)
            neg = kind != fp.KIND_POSITIVE
            check(
                int(item.min()) >= id_min and int(item.max()) < n_slots,
                f"{label}: an item falls outside the candidate slots",
            )
            check(
                bool(np.isin(item[kind == fp.KIND_POP], item_ids).all()),
                f"{label}: a p_pop draw is not a P3 proposal item",
            )
            hit = np.zeros(int(neg.sum()), dtype=bool)
            for col in range(4):
                hit |= item[neg] == picks[at_u[row_user[neg]], col]
            check(
                not hit.any(),
                f"{label}: {int(hit.sum())} negatives are "
                f"held-out picks of their listener",
            )
            hist = np.asarray(u_indptr[d["user_id"] + 1], dtype=np.int64) - np.asarray(
                u_indptr[d["user_id"]], dtype=np.int64
            )
            costs = hist + lens
            tasks = s11._blocks(costs, s11.WORKERS * s11.TASKS_PER_WORKER)
            s11._CTX.update(
                users=d["user_id"],
                u_indptr=u_indptr,
                u_indices=u_indices,
                item=item,
                indptr=d["indptr"],
            )
            prog = Progress(
                f"p11b verify: no {label} item in train",
                int(costs.sum()),
                log.info,
                unit=" M entries",
                scale=10**6,
            )
            try:
                res = s11._run_tasks(
                    s11._train_check_block,
                    tasks,
                    [int(costs[a:b].sum()) for a, b in tasks],
                    s11.WORKERS,
                    prog,
                )
            finally:
                s11._CTX.clear()
            prog.finish()
            check(
                sum(r[1] for r in res) == 0, f"{label}: a listener has no train items"
            )
            check(
                sum(r[0] for r in res) == 0,
                f"{label}: {sum(r[0] for r in res)} stored items are "
                f"train items of their listener",
            )

        # Slow recomputation of whole listeners, one double at a time.
        samplers = {
            fp.KIND_UNIFORM: fp.uniform_sampler(id_min, n_slots),
            fp.KIND_POP: fp.pop_sampler(item_ids, cum_pop),
        }
        rng = np.random.default_rng(VERIFY_SEED)
        n_redrawn = 0
        for label, n_check in (("wide", VERIFY_WIDE_USERS), ("ref", VERIFY_REF_USERS)):
            d, k = files[label]
            npos = np.diff(r_ptr) if label == "ref" else n_pos
            take = rng.choice(
                len(d["user_id"]), size=min(n_check, len(d["user_id"])), replace=False
            )
            for j in np.sort(take):
                u = int(d["user_id"][j])
                row = np.asarray(u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])])
                jj = int(np.searchsorted(users, u))
                excl = set(row.tolist()) | set(picks[jj].tolist())
                s = int(d["indptr"][j]) + int(npos[j])
                for col, kd in enumerate((fp.KIND_UNIFORM, fp.KIND_POP)):
                    want, used = fp.reference_draws(
                        s11.SEED, u, kd, k, samplers[kd], excl
                    )
                    got = d["item"][s + col * k : s + (col + 1) * k]
                    check(
                        np.array_equal(got.astype(np.int64), want)
                        and int(d["consumed"][j, col]) == used,
                        f"{label} user {u}: the {fp.KIND_NAMES[kd]} draws "
                        f"differ from a fresh recomputation",
                    )
                n_redrawn += 1
        self._verified = {
            "k11": k11,
            "wide_rows": int(len(wide["item"])),
            "ref_rows": int(len(ref["item"])),
            "p11_rows_matched": int(len(rows11)),
            "wide_rows_matched": int(len(rows_wr)),
            "redrawn": n_redrawn,
        }
        log.info(
            "p11b: verify passed. %d wide rows, %d reference rows, the "
            "first draws match %d P11 rows and %d wide rows, %d "
            "listeners recomputed in full",
            len(wide["item"]),
            len(ref["item"]),
            len(rows11),
            len(rows_wr),
            n_redrawn,
        )

    def _check_layout(self, d, n_pos, k, p_items, p_flags, label):
        indptr = d["indptr"]
        n_users, n_rows = len(d["user_id"]), len(d["item"])
        check(
            len(indptr) == n_users + 1
            and indptr[0] == 0
            and indptr[-1] == n_rows
            and np.array_equal(np.diff(indptr), n_pos + 2 * k),
            f"{label}: a listener does not hold its positives plus two "
            f"sets of {k} draws",
        )
        for name in ("kind", "pick", "draw"):
            check(len(d[name]) == n_rows, f"{label}: {name} has the wrong " f"length")
        for name in ("split", "n_allowed", "pop_allowed", "consumed"):
            check(len(d[name]) == n_users, f"{label}: {name} has the wrong " f"length")
        kind, draw = layout_arrays(indptr, n_pos, k)
        check(
            np.array_equal(d["kind"], kind) and np.array_equal(d["draw"], draw),
            f"{label}: a row has the wrong kind or draw number for its " f"place",
        )
        is_pos = d["kind"] == fp.KIND_POSITIVE
        check(
            np.array_equal(d["item"][is_pos].astype(np.int64), p_items)
            and np.array_equal(d["pick"][is_pos], p_flags)
            and not d["pick"][~is_pos].any(),
            f"{label}: the positives differ from the val picks under the " f"rules",
        )
        check(
            bool((d["consumed"] >= k).all())
            and bool((d["n_allowed"] > 0).all())
            and bool((d["pop_allowed"] > 0).all()),
            f"{label}: a listener has too few raw draws or no allowed " f"candidates",
        )

    def report(self, params):
        lines = [
            "# P11b report: more negatives for the fusion pairs",
            "",
            f"- parameters: {json.dumps(_params())}",
        ]
        if self._verified:
            v = self._verified
            lines.append(
                f"- verify(): the first {v['k11']} draws of each kind equal "
                f"P11's "
                f"{v['p11_rows_matched']:,} rows of the train listeners, "
                f"the first {WIDE_PER_KIND} draws of each reference listener "
                f"equal its {v['wide_rows_matched']:,} wide rows, and "
                f"{v['redrawn']} listeners recomputed one double at a time "
                f"give the same draws."
            )
        lines.append("")
        lines.append(
            "| file | listeners | draws per kind | rows | raw draws "
            "per accepted draw, uniform | same, p_pop | repeated "
            "p_pop draws within a listener |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for label, k in (("wide", WIDE_PER_KIND), ("ref", REF_PER_KIND)):
            d, _, _, _ = self._load_b(label)
            cons = d["consumed"].astype(np.float64)
            lens = np.diff(d["indptr"])
            n_pos = lens - 2 * k
            starts = d["indptr"][:-1] + n_pos + k
            rep = 0
            for j in range(len(d["user_id"])):
                seg = d["item"][starts[j] : starts[j] + k]
                rep += k - len(np.unique(seg))
            lines.append(
                f"| {label} | {len(d['user_id']):,} | {k:,} | "
                f"{len(d['item']):,} | {cons[:, 0].sum() / (k * len(cons)):.4f}"
                f" | {cons[:, 1].sum() / (k * len(cons)):.4f} | {rep:,} |"
            )
        lines.append("")
        lines.append(
            "Draws are independent and with replacement, so a "
            "listener can hold a track twice among its p_pop draws. "
            "The sampling probabilities of fusion_pairs.py hold "
            "for every draw."
        )
        text = "\n".join(lines) + "\n"
        check(";" not in text, "the report holds a semicolon")
        atomic_text_write(text, self.outputs()["report"])
        log.info("p11b report written")

"""Stage P12b: the concept features of P11b's pairs.

For each pairs file of P11b (wide and ref), the stage computes the concept
features of every row with P12's own worker (stage_p12._block), which
calls concept_data.user_rows(), the function that P12 and P14 use. The
new rows therefore get their features exactly as P12's rows did, and
verify() proves it on the rows that the files share with P12: the first
16 draws of each kind of every train listener are P11's rows, and their
features equal P12's stored features bit for bit. The first WIDE draws of
each reference listener are its wide rows, with the same check.

Outputs, in P12's format (params, pairs_sha256, values, has, match_tier),
one row per row of the pairs file, in the same order:
- fusion_features_wide.npz, for fusion_pairs_wide.npz,
- fusion_features_ref.npz, for fusion_pairs_ref.npz,
- reports/p12b_report.md.

Resume: as in P12, the listeners run in blocks. Each finished block is
saved under SCRATCH_DIR/p12b_wide or SCRATCH_DIR/p12b_ref, keyed by P12's
parameters, the pairs' SHA-256, the upstream outputs, the code and the
block layout. A stopped run continues after the last saved block.

verify():
- P12's verify() on each file: the parameters and the pairs' hash, NaN
  exactly at the missing states, the missing states of every row against
  the item side, popularity and tier, the value ranges, 40 listeners
  recomputed by concept_data.Reference (a second implementation that
  shares no code with concepts.py), and every evidence list summing to
  its value.
- the rows shared with P12 and between the two files are equal bit for
  bit (values, data flags and tiers).
"""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import artists as ar
from . import concept_data as cd
from . import concepts as cn
from . import evaluation as ev
from . import fusion_pairs as fp
from . import stage_p11b as s11b
from . import stage_p12 as s12
from .als import Progress
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from .config import MODEL_DIR, SCRATCH_DIR

WHICH = ("wide", "ref")


def n_pos_of(kind, indptr):
    """Positives per listener from the kind column and the listener
    offsets."""
    return np.add.reduceat(
        (np.asarray(kind) == fp.KIND_POSITIVE).astype(np.int64), np.asarray(indptr)[:-1]
    )


def same_rows(a, b):
    """Bit-for-bit equality of two feature row sets (values, has, tier),
    NaN included."""
    va, ha, ta = a
    vb, hb, tb = b
    return (
        va.shape == vb.shape
        and np.array_equal(va.view(np.uint32), vb.view(np.uint32))
        and np.array_equal(ha, hb)
        and np.array_equal(ta, tb)
    )


class StageP12b(s12.StageP12):
    name = "p12b_more_features"
    upstream = s12.StageP12.upstream + ["p11b_more_pairs", "p12_pair_features"]

    def __init__(self):
        super().__init__()
        self._which = "wide"
        self._verified_b = {}

    def outputs(self):
        return {
            "wide": MODEL_DIR / "fusion_features_wide.npz",
            "ref": MODEL_DIR / "fusion_features_ref.npz",
            "report": MODEL_DIR / "reports" / "p12b_report.md",
        }

    def current_inputs(self, files):
        inputs = super().current_inputs(files)
        for label, mod in (("stage_p12_module", s12), ("stage_p11b_module", s11b)):
            inputs[label] = file_sha256(Path(mod.__file__))
        return inputs

    # The loaders below read the model folder. test_p13b2.py runs the
    # stages before this one on a synthetic world in a temporary folder.
    def _pairs(self):
        key = f"pairs_{self._which}"
        if key not in self._cache:
            path = MODEL_DIR / f"fusion_pairs_{self._which}.npz"
            z = np.load(path)
            self._cache[key] = (
                {
                    k: np.asarray(z[k])
                    for k in ("user_id", "split", "indptr", "item", "kind")
                },
                json.loads(str(z["params"])),
                ev.npz_hash(path),
            )
        return self._cache[key]

    def _p11_pairs(self):
        z = np.load(MODEL_DIR / "fusion_pairs.npz")
        return {k: np.asarray(z[k]) for k in ("user_id", "split", "indptr", "kind")}

    def _p12_features(self):
        z = np.load(MODEL_DIR / "fusion_features.npz")
        return (
            {k: z[k] for k in ("values", "has", "match_tier")},
            json.loads(str(z["params"])),
            str(z["pairs_sha256"]),
        )

    def _upstream_outputs(self):
        out = {}
        for stage in self.upstream:
            if stage in ("p11_fusion_pairs", "p11b_more_pairs"):
                continue  # the pairs' own SHA-256 stands for it
            doc = load_json(marker_path(stage)) or {}
            out[stage] = {
                k: (v.get("sha256") if isinstance(v, dict) else v)
                for k, v in sorted((doc.get("outputs") or {}).items())
            }
        return out

    def run(self, params, files):
        outputs = self.outputs()
        for which in WHICH:
            self._which = which
            self._run_one(which, outputs[which])
        return self._info(outputs)

    def _run_one(self, which, out_path):
        """P12's run() for one pairs file, with its own scratch folder."""
        p, p_params, pairs_sha = self._pairs()
        users, p_indptr, p_items = p["user_id"], p["indptr"], p["item"]
        n_slots, id_min = self._id_slots()
        u_indptr, u_indices, u_data = self._csr()
        check(
            len(u_indptr) - 1 > int(users.max()),
            f"{which}: a pair's user falls outside the user CSR",
        )
        check(
            int(p_items.min()) >= id_min and int(p_items.max()) < n_slots,
            f"{which}: a pair's item falls outside the item slots",
        )
        hist = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
            u_indptr[users], dtype=np.int64
        )
        costs = hist + 50 * np.diff(p_indptr)
        tasks = s12._blocks(costs, s12.WORKERS * s12.TASKS_PER_WORKER)
        work = SCRATCH_DIR / f"p12b_{which}"
        state_path = work / "state.json"
        key = {
            "params": s12._params(),
            "pairs_sha256": pairs_sha,
            "upstream": self._upstream_outputs(),
            "code": {
                Path(f).name: file_sha256(Path(f))
                for f in (__file__, s12.__file__, cn.__file__, cd.__file__, ar.__file__)
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
            and out_path.exists()
            and file_sha256(out_path) == state["final_sha"]
        ):
            log.info("p12b: %s features resume, every block is done", which)
            return

        def part_path(k):
            return work / f"part_{k:05d}.npz"

        done = {
            int(k)
            for k, sha in state["parts"].items()
            if part_path(int(k)).exists() and file_sha256(part_path(int(k))) == sha
        }
        pending = [k for k in range(len(tasks)) if k not in done]
        if done:
            log.info(
                "p12b: %s blocks resume, %d of %d done", which, len(done), len(tasks)
            )
        if pending:
            src = self._sources(n_slots)
            space = self._space(src)
            s12._CTX.update(
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
                "p12b: %s: %d listeners, %d rows, %d blocks left, %d " "workers",
                which,
                len(users),
                len(p_items),
                len(pending),
                s12.WORKERS,
            )
            prog = Progress(
                f"p12b: {which} concept features",
                n_users_left,
                log.info,
                unit=" listeners",
            )
            try:
                if s12.WORKERS <= 1 or len(pending) <= 1:
                    results = (s12._block(tasks[k]) for k in pending)
                    self._save_parts(
                        pending, results, tasks, part_path, state, state_path, prog
                    )
                else:
                    with ProcessPoolExecutor(
                        max_workers=s12.WORKERS, mp_context=mp.get_context("fork")
                    ) as pool:
                        results = pool.map(s12._block, [tasks[k] for k in pending])
                        self._save_parts(
                            pending, results, tasks, part_path, state, state_path, prog
                        )
            finally:
                s12._CTX.clear()
            prog.finish()
        parts = [np.load(part_path(k)) for k in range(len(tasks))]
        values = np.concatenate([z["values"] for z in parts])
        has = np.concatenate([z["has"] for z in parts])
        tier = np.concatenate([z["tier"] for z in parts])
        del parts
        check(
            len(values) == len(p_items),
            f"{which}: the blocks hold {len(values)} rows, the pairs "
            f"{len(p_items)}",
        )
        s12._atomic_savez(
            out_path,
            params=np.array(json.dumps(s12._params())),
            pairs_sha256=np.array(pairs_sha),
            values=values,
            has=has,
            match_tier=tier,
        )
        state["final_sha"] = file_sha256(out_path)
        atomic_json_write(state, state_path)
        for k in range(len(tasks)):
            part_path(k).unlink()
        log.info("p12b: %s: %d rows written", which, len(values))

    def _info(self, outputs):
        out = {}
        for which in WHICH:
            z = np.load(outputs[which])
            out[which] = {
                "path": str(outputs[which]),
                "rows": int(len(z["match_tier"])),
                "sha256": ev.npz_hash(outputs[which]),
            }
        return out

    def _load(self):
        z = np.load(self.outputs()[self._which])
        return (
            {k: z[k] for k in ("values", "has", "match_tier")},
            json.loads(str(z["params"])),
            str(z["pairs_sha256"]),
        )

    def verify(self, params):
        self._verified = None
        self._verified_b = {}
        for which in WHICH:
            self._which = which
            log.info("p12b: P12's checks on fusion_features_%s.npz", which)
            s12.StageP12.verify(self, params)
            self._verified_b[which] = dict(self._verified)
        self._verified = None
        self._verified_b.update(self._verify_shared())
        log.info(
            "p12b: verify passed. Shared rows equal bit for bit: %d "
            "with P12, %d between the reference and the wide file",
            self._verified_b["shared_p12"],
            self._verified_b["shared_wide"],
        )

    def _verify_shared(self):
        wide_p = self._pairs_of("wide")
        ref_p = self._pairs_of("ref")
        k_wide = int(self._pairs_params("wide")["wide_per_kind"])
        k_ref = int(self._pairs_params("ref")["ref_per_kind"])
        k11 = int(self._pairs_params("wide")["p11"]["neg_per_kind"])
        # P12's rows of the train listeners against the wide file.
        p11 = self._p11_pairs()
        f12, _, f12_pairs_sha = self._p12_features()
        check(
            f12_pairs_sha == self._base_pairs_sha(),
            "fusion_features.npz was built from other pairs than the ones "
            "P11b extends",
        )
        tr = p11["split"] == fp.SPLIT_TRAIN
        check(
            np.array_equal(p11["user_id"][tr], wide_p["user_id"]),
            "the wide file holds other listeners than P11's train split",
        )
        rows12 = np.flatnonzero(np.repeat(tr, np.diff(p11["indptr"])))
        rows_w = s11b.prefix_rows(
            wide_p["indptr"], n_pos_of(wide_p["kind"], wide_p["indptr"]), k_wide, k11
        )
        fw = self._features_of("wide")
        a = tuple(f12[k][rows12] for k in ("values", "has", "match_tier"))
        b = tuple(fw[k][rows_w] for k in ("values", "has", "match_tier"))
        check(
            same_rows(a, b),
            "the wide features of P11's rows differ from " "P12's stored features",
        )
        n12 = len(rows12)
        del a, b, f12
        # The reference file's first draws against the wide rows.
        at = np.searchsorted(wide_p["user_id"], ref_p["user_id"])
        check(
            np.array_equal(wide_p["user_id"][at], ref_p["user_id"]),
            "a reference listener is not a wide listener",
        )
        mask = np.zeros(len(wide_p["user_id"]), dtype=bool)
        mask[at] = True
        rows_wr = np.flatnonzero(np.repeat(mask, np.diff(wide_p["indptr"])))
        rows_r = s11b.prefix_rows(
            ref_p["indptr"], n_pos_of(ref_p["kind"], ref_p["indptr"]), k_ref, k_wide
        )
        fr = self._features_of("ref")
        a = tuple(fw[k][rows_wr] for k in ("values", "has", "match_tier"))
        b = tuple(fr[k][rows_r] for k in ("values", "has", "match_tier"))
        check(
            same_rows(a, b),
            "the reference features of the first draws "
            "differ from the wide features",
        )
        return {"shared_p12": int(n12), "shared_wide": int(len(rows_wr))}

    def _pairs_of(self, which):
        keep = self._which
        self._which = which
        try:
            return self._pairs()[0]
        finally:
            self._which = keep

    def _pairs_params(self, which):
        keep = self._which
        self._which = which
        try:
            return self._pairs()[1]
        finally:
            self._which = keep

    def _base_pairs_sha(self):
        """The npz hash of the fusion_pairs.npz that P11b's files extend."""
        z = np.load(MODEL_DIR / "fusion_pairs_wide.npz")
        return str(z["pairs_sha256"])

    def _features_of(self, which):
        z = np.load(self.outputs()[which])
        return {k: z[k] for k in ("values", "has", "match_tier")}

    def report(self, params):
        lines = [
            "# P12b report: the concept features of P11b's pairs",
            "",
            f"- parameters: {json.dumps(s12._params())}",
        ]
        v = self._verified_b
        if v:
            lines.append(
                f"- verify(): P12's checks passed on both files. The "
                f"features of the {v['shared_p12']:,} rows that the wide "
                f"file shares with P11 equal P12's stored features bit for "
                f"bit, and the {v['shared_wide']:,} rows that the reference "
                f"file shares with the wide file equal the wide features."
            )
            for which in WHICH:
                w = v.get(which) or {}
                if w:
                    lines.append(
                        f"- {which}: {w['rows']:,} rows. "
                        f"{w['reference_rows']} rows recomputed by the "
                        f"reference implementation (largest relative gap "
                        f"{w['reference_gap']:.1e}), {w['evidence_lists']} "
                        f"evidence lists sum to their values (largest gap "
                        f"{w['evidence_gap']:.1e})."
                    )
        lines.append("")
        lines.append("| file | rows | share with data per concept |")
        lines.append("|---|---:|---|")
        for which in WHICH:
            f = self._features_of(which)
            shares = ", ".join(
                f"{c} {100 * float(f['has'][:, j].mean()):.1f}" f"%"
                for j, c in enumerate(cn.CONTINUOUS)
            )
            lines.append(f"| {which} | {len(f['match_tier']):,} | {shares} |")
        lines.append("")
        text = "\n".join(lines) + "\n"
        check(";" not in text, "the report holds a semicolon")
        atomic_text_write(text, self.outputs()["report"])
        log.info("p12b report written")

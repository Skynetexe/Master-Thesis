"""Stage P6: the ALS collaborative-filtering baseline.

Builds the interaction CSR as memory-mapped artifacts in both orientations,
fits implicit-feedback ALS (Hu, Koren, Volinsky 2008) through
pipeline/als.py, and evaluates under the harness rules: full catalog minus
the user's train items, held-out picks stay candidates, ties break by
item_id ascending.

Revised in the repair of the ALS baseline:

1. The item-major CSR is the exact transpose of the user-major CSR. The
   old build sorted each train file by item and concatenated the files.
   The files split users, not items, so every item row held other items'
   listeners. Rebuilds of that logic on synthetic data gave 0.7 to 1.3
   percent real pairs. als_tune.py measured the share on the real data
   and recorded it in reports/als_tuning.md. The transpose comes from
   the cache that als_tune.py built when the cache key matches
   (transpose version, SHA-256 of the user-major arrays, item slots). It
   is hard-linked into MODEL_DIR, or copied when the two folders sit on
   different file systems.
2. The fit runs through pipeline/als.py: exact row solves, the exact
   objective checked after every half-step, rows without data solved to
   exactly zero, and a resume state bound to the parameters, the solver
   version and a SHA-256 of the data. The old resume compared only
   als_params.json, so it kept stale factors after the CSR was rebuilt.
   The old hundredfold loss guard is gone: an objective rise of more than
   1e-4 relative now stops the fit. The divergence recorded in an early
   run (reg 0.01) came before the transpose fix.
3. The parameters come from CURATED/als_params.json. als_tune.py
   --write-params writes it from the pre-registered search.
   current_inputs() hashes that file (it hashed MODEL_DIR/als_params.json,
   which run() never read) and pipeline/als.py, because the runner's
   source hash covers only this module, common.py and config.py. A change
   to either now reruns this stage.
4. The model label in eval_als.json and in the report comes from the
   parameters. It was a fixed string that named other parameters.
5. verify() checks the item-major orientation on a sample, that the
   factors are a finished fit of these parameters on this data, and that
   rows without train data are exactly zero.

Confidence: c = 1 + alpha * log1p(listens), from the w_log column scaled
by 1000 into uint16. The listens_u8 column is never read here. It exists
only as a compact representation and saturates at 255 against a max pair
of 223,244, so a confidence built on it would flatten every heavy pair.

The factors and the CSR derive from MLHD interactions only, so the
Spotify license list does not apply to them.
"""

import os
import shutil
from pathlib import Path

import numpy as np

from . import als
from . import evaluation as ev
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
    require_space,
    with_thread_ladder,
)
from .config import CURATED, FROZEN, MODEL_DIR, SCRATCH_DIR

ITEM_MAJOR_FILES = (
    ("csr_indptr_t", "csr_item_indptr.npy"),
    ("csr_indices_t", "csr_item_indices.npy"),
    ("csr_data_t", "csr_item_data.npy"),
)
ALS_KEYS = ("factors", "iterations", "regularization", "alpha", "seed")


def _params_path():
    return CURATED / "als_params.json"


def _fit_state_path():
    return SCRATCH_DIR / "p6_als_fit_state.json"


def _csr_state_path():
    return SCRATCH_DIR / "p6_csr_state.json"


def _transpose_cache():
    return SCRATCH_DIR / "csr_item_major"  # shared with als_tune.py


def _als_params(ap):
    return als.ALSParams(
        factors=int(ap["factors"]),
        reg=float(ap["regularization"]),
        alpha=float(ap["alpha"]),
        iterations=int(ap["iterations"]),
        seed=int(ap["seed"]),
    )


def _label(ap):
    return (
        f"als: factors {int(ap['factors'])}, alpha {float(ap['alpha']):g} "
        f"on log1p, reg {float(ap['regularization']):g}, "
        f"{int(ap['iterations'])} iterations"
    )


def _link_or_copy(src, dst):
    """Place src at dst. A hard link when both sit on one file system,
    else a copy. dst changes only by an atomic rename of a complete file.

    A hard link shares its data with the cache file. Nothing may open dst
    for writing in place (np.save or open_memmap mode "w+" on dst), or
    the cache changes with it. This module and pipeline/als.py only ever
    replace these files by rename."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.stem + ".tmp.npy")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        require_space(dst.parent, int(src.stat().st_size * 1.05))
        shutil.copyfile(src, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    os.replace(tmp, dst)
    # When dst already was a link to src, rename() leaves both names in
    # place (POSIX), so the extra name goes here.
    tmp.unlink(missing_ok=True)


class StageP6:
    name = "p6_als"
    upstream = ["p5_eval_baseline"]

    def outputs(self):
        return {
            "csr_indices": MODEL_DIR / "csr_user_indices.npy",
            "csr_data": MODEL_DIR / "csr_user_data.npy",
            "csr_indptr": MODEL_DIR / "csr_user_indptr.npy",
            "csr_indices_t": MODEL_DIR / "csr_item_indices.npy",
            "csr_data_t": MODEL_DIR / "csr_item_data.npy",
            "csr_indptr_t": MODEL_DIR / "csr_item_indptr.npy",
            "csr_meta": MODEL_DIR / "csr_meta.json",
            "user_factors": MODEL_DIR / "als_user_factors.npy",
            "item_factors": MODEL_DIR / "als_item_factors.npy",
            "metrics": MODEL_DIR / "eval_als.json",
            "ranks": MODEL_DIR / "eval_als_ranks.npz",
            "report": MODEL_DIR / "reports" / "p6_als.md",
        }

    def current_inputs(self, files):
        inputs = {}
        try:
            inputs["marker:p5_eval_baseline"] = file_sha256(
                marker_path("p5_eval_baseline")
            )
        except FileNotFoundError:
            inputs["marker:p5_eval_baseline"] = None
        for key, path in [
            ("als_params", _params_path()),
            ("train_manifest", MODEL_DIR / "train_manifest.json"),
            ("als_module", Path(als.__file__)),
        ]:
            inputs[key] = file_sha256(path) if path.exists() else None
        return inputs

    def run(self, params, files):
        outputs = self.outputs()
        ap = load_json(_params_path())
        check(ap is not None, f"{_params_path()} is missing")
        missing = [k for k in ALS_KEYS if k not in ap]
        check(not missing, f"{_params_path()} lacks {missing}")
        self._ap = ap

        found = {}

        def build_work(con):
            found["fingerprint"] = self._build_csr(con, outputs)

        with_thread_ladder(build_work, params["ladder"])

        inter = self._interactions(outputs, found.get("fingerprint"))
        self._fit(inter, outputs)

        con = connect(params["ladder"][0])
        try:
            self._evaluate(con, inter, outputs)
        finally:
            con.close()

        info = {}
        for key in (
            "csr_indices",
            "csr_data",
            "csr_indices_t",
            "csr_data_t",
            "user_factors",
            "item_factors",
        ):
            info[key] = {
                "path": str(outputs[key]),
                "rows": 0,
                "sha256": file_sha256(outputs[key]),
            }
        for key in ("csr_indptr", "csr_indptr_t"):
            info[key] = {
                "path": str(outputs[key]),
                "rows": 0,
                "sha256": file_sha256(outputs[key]),
            }
        info["csr_meta"] = {
            "path": str(outputs["csr_meta"]),
            "rows": 0,
            "sha256": file_sha256(outputs["csr_meta"]),
        }
        info["metrics"] = {
            "path": str(outputs["metrics"]),
            "rows": 0,
            "sha256": file_sha256(outputs["metrics"]),
        }
        info["ranks"] = {
            "path": str(outputs["ranks"]),
            "rows": 0,
            "sha256": ev.npz_hash(outputs["ranks"]),
        }
        return info

    def _expected_nnz(self, con):
        held = con.execute(
            f"SELECT count(*) FROM read_parquet('{ev.SPLITS}')"
        ).fetchone()[0]
        return FROZEN["pairs"] - held

    def _build_csr(self, con, outputs):
        """The user-major CSR from the train files, and the item-major CSR
        as its exact transpose. Returns the user-major SHA-256 when this
        call computed it, else None."""
        state_path = _csr_state_path()
        manifest_path = MODEL_DIR / "train_manifest.json"
        manifest_sha = file_sha256(manifest_path)
        state = load_json(state_path) or {}
        user_paths = [outputs[k] for k in ("csr_indices", "csr_data", "csr_indptr")]
        item_paths = [outputs[k] for k, _ in ITEM_MAJOR_FILES]
        user_ok = (
            state.get("manifest") == manifest_sha
            and all(p.exists() for p in user_paths)
            and outputs["csr_meta"].exists()
        )
        if (
            user_ok
            and state.get("item_major") == als.TRANSPOSE_VERSION
            and all(p.exists() for p in item_paths)
        ):
            log.info("p6: csr artifacts resume")
            return None
        if user_ok:
            meta = load_json(outputs["csr_meta"])
            log.info(
                "p6: user-major csr resume. The item-major csr is "
                "rebuilt as its exact transpose."
            )
        else:
            require_space(MODEL_DIR, 48 * 2**30)
            manifest = load_json(manifest_path) or {}
            nnz = self._expected_nnz(con)
            ids = load_json(MODEL_DIR / "id_spaces.json") or {}
            user_slots = int(ids["users"]["slots"])
            item_slots = int(ids["items"]["slots"])
            self._write_orientation(
                con,
                "user_id, item_id",
                "user_id",
                "item_id",
                user_slots,
                outputs["csr_indices"],
                outputs["csr_data"],
                outputs["csr_indptr"],
                manifest,
                nnz,
                "user-major",
            )
            meta = {
                "rows": user_slots,
                "cols": item_slots,
                "nnz": int(nnz),
                "note": "user-major rows are users and indices are "
                "item_id. item-major rows are items and indices "
                "are user_id, built as the exact transpose of "
                "the user-major csr. data = 1000 * "
                "log1p(listens) as uint16.",
            }
            atomic_json_write(meta, outputs["csr_meta"])
            # No item_major key yet: a stop before the transpose is placed
            # resumes at the transpose.
            atomic_json_write({"manifest": manifest_sha, "nnz": int(nnz)}, state_path)
        u_arrays = (
            np.load(outputs["csr_indptr"]),
            np.load(outputs["csr_indices"], mmap_mode="r"),
            np.load(outputs["csr_data"], mmap_mode="r"),
        )
        check(
            int(u_arrays[0][-1]) == int(meta["nnz"]),
            f"user-major: indptr end {int(u_arrays[0][-1])} != nnz " f"{meta['nnz']}",
        )
        inter = als.Interactions.from_user_major(
            u_arrays, int(meta["cols"]), _transpose_cache(), log=log.info
        )
        for key, name in ITEM_MAJOR_FILES:
            _link_or_copy(_transpose_cache() / name, outputs[key])
        atomic_json_write(
            {
                "manifest": manifest_sha,
                "nnz": int(meta["nnz"]),
                "item_major": als.TRANSPOSE_VERSION,
                "user_major_sha256": inter.fingerprint(),
            },
            state_path,
        )
        log.info(
            "p6: item-major csr placed as the exact transpose, %d " "nonzeros",
            int(meta["nnz"]),
        )
        return inter.fingerprint()

    def _write_orientation(
        self,
        con,
        order_sql,
        count_key,
        index_key,
        slots,
        indices_path,
        data_path,
        indptr_path,
        manifest,
        nnz,
        label,
    ):
        """Correct for the user-major orientation only: the train files
        hold ascending user ranges (_check_file_order), so the files in
        name order, each sorted by user_id, item_id, form one global
        order."""
        indices = np.lib.format.open_memmap(
            indices_path, mode="w+", dtype=np.int32, shape=(nnz,)
        )
        data = np.lib.format.open_memmap(
            data_path, mode="w+", dtype=np.uint16, shape=(nnz,)
        )
        counts = np.zeros(slots, dtype=np.int64)
        offset = 0
        for name in sorted(manifest):
            path = MODEL_DIR / name
            expected_rows = manifest[name]["rows"]
            cols = con.execute(f"""
                SELECT user_id, item_id, w_log FROM read_parquet('{path}')
                ORDER BY {order_sql}
            """).fetchnumpy()
            n = len(cols["user_id"])
            check(
                n == expected_rows,
                f"{name}: the csr scan found {n} rows, the manifest "
                f"says {expected_rows}",
            )
            indices[offset : offset + n] = np.asarray(cols[index_key], dtype=np.int32)
            data[offset : offset + n] = np.round(
                np.asarray(cols["w_log"], dtype=np.float64) * 1000.0
            ).astype(np.uint16)
            counts += np.bincount(
                np.asarray(cols[count_key], dtype=np.int64), minlength=slots
            )
            offset += n
            log.info("p6: %s csr %s done, %d rows", label, name, n)
        check(offset == nnz, f"{label}: nnz {offset} != expected {nnz}")
        check(
            int(data.min()) >= 693 and int(data.max()) <= 65535,
            f"{label}: a data value falls outside the log1p range",
        )
        check(int(counts[0]) == 0, f"{label}: the phantom slot is not empty")
        indices.flush()
        data.flush()
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        np.save(indptr_path, indptr)

    def _interactions(self, outputs, fingerprint=None):
        def mm(key):
            return np.load(outputs[key], mmap_mode="r")

        return als.Interactions(
            np.load(outputs["csr_indptr"]),
            mm("csr_indices"),
            mm("csr_data"),
            np.load(outputs["csr_indptr_t"]),
            mm("csr_indices_t"),
            mm("csr_data_t"),
            fingerprint=fingerprint,
        )

    def _fit(self, inter, outputs):
        params = _als_params(self._ap)
        log.info("p6: fitting %s", _label(self._ap))
        _, _, history = als.fit(
            inter,
            params,
            outputs["user_factors"],
            outputs["item_factors"],
            _fit_state_path(),
            log=log.info,
        )
        check(
            len(history) == params.iterations,
            f"the fit stopped after {len(history)} of {params.iterations} "
            f"iterations",
        )
        self._history = history

    def _load_frame(self, con):
        rows = con.execute(f"""
            WITH picks AS (
                SELECT user_id,
                       max(CASE WHEN is_test_last THEN item_id END) AS test_last,
                       max(CASE WHEN is_test_first THEN item_id END) AS test_first,
                       max(CASE WHEN is_val_last THEN item_id END) AS val_last
                FROM read_parquet('{ev.SPLITS}')
                GROUP BY user_id)
            SELECT e.user_id, e.activity_decile,
                   p.test_last, p.test_first, p.val_last
            FROM read_parquet('{ev.EVAL_USERS}') e
            LEFT JOIN picks p ON e.user_id = p.user_id
            ORDER BY e.user_id
        """).fetchall()
        dropped = sum(1 for r in rows if r[2] is None or r[3] is None)
        rows = [r for r in rows if r[2] is not None and r[3] is not None]
        user = np.array([r[0] for r in rows], dtype=np.int64)
        decile = np.array([r[1] for r in rows], dtype=np.int64)
        test_last = np.array([r[2] for r in rows], dtype=np.int64)
        test_first = np.array([r[3] for r in rows], dtype=np.int64)
        val_last = np.array([r[4] for r in rows], dtype=np.int64)
        return user, decile, test_last, test_first, val_last, dropped

    def _evaluate(self, con, inter, outputs):
        meta = load_json(MODEL_DIR / "csr_meta.json") or {}
        item_slots = int(meta["cols"])
        X = np.load(outputs["user_factors"])
        Y = np.load(outputs["item_factors"])
        user, decile, test_last, test_first, val_last, dropped = self._load_frame(con)
        n = len(user)
        ranks = als.rank_items(
            X,
            Y,
            inter,
            user,
            np.stack([test_last, test_first, val_last], axis=1),
            log=log.info,
            label="p6: ranking the eval users' picks",
        )
        ranks_last, ranks_first, ranks_val = (ranks[:, 0], ranks[:, 1], ranks[:, 2])

        pop, pairs, tier = ev.covariate_arrays(con, item_slots)
        decile_labels = np.array([f"d{int(d):02d}" for d in decile])

        def variant_block(test_items, rk):
            tp = pairs[test_items]
            cold = np.where(
                tp <= 0,
                "0",
                np.where(tp <= 5, "01-05", np.where(tp <= 99, "06-99", "100+")),
            )
            pc = pop[test_items]
            band = np.where(
                pc <= 10,
                "01-10",
                np.where(pc <= 50, "11-50", np.where(pc <= 90, "51-90", "91-100")),
            )
            tier_labels = np.array([ev.TIER_NAMES[int(c)] for c in tier[test_items]])
            return {
                "overall": ev.metric_block(rk),
                "slices": {
                    "tier": ev.slice_blocks(rk, tier_labels),
                    "pop_band": ev.slice_blocks(rk, band),
                    "cold_test_item": ev.slice_blocks(rk, cold),
                    "user_decile": ev.slice_blocks(rk, decile_labels),
                },
            }

        history = getattr(self, "_history", None) or (
            load_json(_fit_state_path()) or {}
        ).get("history", [])
        doc = {
            "model": _label(self._ap),
            "als_params": {k: self._ap[k] for k in ALS_KEYS},
            "fit": {
                "solver": als.SOLVER_VERSION,
                "item_major": als.TRANSPOSE_VERSION,
                "objective_after_items": [h["objective_after_items"] for h in history],
                "seconds": [h["seconds"] for h in history],
            },
            "candidate_rule": "full catalog minus the user's train items, "
            "held-out picks stay candidates",
            "tie_rule": "score desc, then item_id asc",
            "n_eval_users": int(n),
            "dropped_no_test_pick": int(dropped),
            "variants": {
                "last_ts": variant_block(test_last, ranks_last),
                "first_ts": variant_block(test_first, ranks_first),
                "val_last": {"overall": ev.metric_block(ranks_val)},
            },
        }
        atomic_json_write(doc, outputs["metrics"])
        np.savez_compressed(
            outputs["ranks"],
            user_id=user,
            rank_test_last=ranks_last,
            rank_test_first=ranks_first,
            rank_val_last=ranks_val,
        )
        log.info("p6: eval artifacts written")

    def _verify_als(self, inter, ap, outputs):
        """The checks that the old verify() lacked (the repair above).
        They need no DuckDB, so test_p6.py runs them directly."""
        real = als.orientation_real_fraction(
            inter, rows=500, per_row=200, seed=int(ap.get("seed", 0))
        )
        check(
            real == 1.0,
            f"item-major: {real:.2%} of sampled entries are real pairs of "
            f"the user-major csr. The exact transpose scores 100%.",
        )
        state = load_json(_csr_state_path()) or {}
        check(
            state.get("item_major") == als.TRANSPOSE_VERSION,
            "the csr state does not record the transpose build of the "
            "item-major csr",
        )
        missing = [k for k in ALS_KEYS if k not in ap]
        check(not missing, f"{_params_path()} lacks {missing}")
        problems = als.fit_state_problems(
            inter,
            _als_params(ap),
            outputs["user_factors"],
            outputs["item_factors"],
            _fit_state_path(),
        )
        check(
            not problems,
            "the ALS factors are not a finished fit of "
            "als_params.json on this data: " + " ".join(problems),
        )
        doc = load_json(outputs["metrics"]) or {}
        check(
            doc.get("model") == _label(ap),
            f"eval_als.json names the model {doc.get('model')!r}, the "
            f"parameters say {_label(ap)!r}",
        )

    def verify(self, params):
        con = connect(4)
        outputs = self.outputs()
        try:
            meta = load_json(MODEL_DIR / "csr_meta.json") or {}
            ap = load_json(_params_path()) or {}
            user_slots = int(meta["rows"])
            item_slots = int(meta["cols"])
            nnz = int(meta["nnz"])
            f = int(ap.get("factors", 64))
            _check_file_order(con)

            for ind_key, idx_key, row_slots, index_bound, label in [
                ("csr_indptr", "csr_indices", user_slots, item_slots, "user-major"),
                ("csr_indptr_t", "csr_indices_t", item_slots, user_slots, "item-major"),
            ]:
                indptr = np.load(outputs[ind_key])
                indices = np.load(outputs[idx_key], mmap_mode="r")
                check(
                    len(indptr) == row_slots + 1,
                    f"{label}: indptr length {len(indptr)}",
                )
                check(
                    int(indptr[-1]) == nnz,
                    f"{label}: indptr end {indptr[-1]} != nnz {nnz}",
                )
                check(int(indptr[1]) == 0, f"{label}: the phantom slot is not empty")
                lo = int(indices.min())
                hi = int(indices.max())
                check(
                    lo >= 1 and hi < index_bound,
                    f"{label}: an index falls outside the slot range",
                )

            self._verify_als(self._interactions(outputs), ap, outputs)

            spot = np.random.default_rng(int(ap.get("seed", 20260907)))
            k = spot.integers(0, nnz, size=1000)
            indptr = np.load(outputs["csr_indptr"])
            indices = np.load(outputs["csr_indices"], mmap_mode="r")
            u = np.searchsorted(indptr, k, side="right") - 1
            i = np.asarray(indices[k], dtype=np.int64)
            con.execute(
                "CREATE OR REPLACE TABLE spot_pairs " "(user_id BIGINT, item_id BIGINT)"
            )
            con.executemany(
                "INSERT INTO spot_pairs VALUES (?, ?)",
                list(zip(u.tolist(), i.tolist())),
            )
            hits = con.execute(f"""
                SELECT count(*) FROM spot_pairs s
                JOIN read_parquet('{MODEL_DIR}/interactions_train_*.parquet') t
                  ON s.user_id = t.user_id AND s.item_id = t.item_id
            """).fetchone()[0]
            check(
                hits == 1000,
                f"{hits} of 1000 sampled csr pairs exist in the train " f"table",
            )

            X = np.load(outputs["user_factors"])
            Y = np.load(outputs["item_factors"])
            check(
                X.shape == (user_slots, f) and X.dtype == np.float32,
                f"user factors have shape {X.shape}",
            )
            check(
                Y.shape == (item_slots, f) and Y.dtype == np.float32,
                f"item factors have shape {Y.shape}",
            )
            check(
                bool(np.isfinite(X).all()) and bool(np.isfinite(Y).all()),
                "a factor value is not finite",
            )
            norm2 = float(np.sum(X * X, dtype=np.float64)) + float(
                np.sum(Y * Y, dtype=np.float64)
            )
            check(norm2 < 1e12, f"factor norms exploded: squared norm {norm2:.3e}.")

            doc = load_json(outputs["metrics"])
            data = np.load(outputs["ranks"])
            user = np.asarray(data["user_id"])
            rl = np.asarray(data["rank_test_last"])
            rf = np.asarray(data["rank_test_first"])
            rv = np.asarray(data["rank_val_last"])
            n = len(user)
            check(
                n == doc["n_eval_users"] and n > 0,
                "the json and the npz disagree on the user count",
            )
            for rk in (rl, rf, rv):
                check(
                    int(rk.min()) >= 1
                    and int(rk.max()) <= nnz
                    and int(rk.max()) <= FROZEN["items"],
                    "a rank falls outside the candidate range",
                )
            for variant in ("last_ts", "first_ts"):
                ov = doc["variants"][variant]["overall"]
                check(
                    0.0 <= ov["recall_at_10"] <= ov["recall_at_100"] <= 1.0,
                    f"{variant}: the recall ordering is wrong",
                )
                check(
                    ov["ndcg_at_100"] <= ov["recall_at_100"] + 1e-12,
                    f"{variant}: ndcg@100 exceeds recall@100",
                )
                n_slice = sum(
                    sb["n_users"]
                    for sb in doc["variants"][variant]["slices"]["user_decile"].values()
                )
                check(n_slice == n, f"{variant}: slice counts do not sum")

            frame = self._load_frame(con)
            check_users = user[:50]
            S50 = X[check_users] @ Y.T
            S50[:, 0] = -np.inf
            for k2 in range(50):
                u = int(check_users[k2])
                s, e = int(indptr[u]), int(indptr[u + 1])
                if e > s:
                    S50[k2, indices[s:e]] = -np.inf
                for rk, t in [
                    (rl, frame[2][k2]),
                    (rv, frame[4][k2]),
                    (rf, frame[3][k2]),
                ]:
                    order = np.argsort(-S50[k2], kind="stable")
                    pos = int(np.flatnonzero(order == t)[0]) + 1
                    check(
                        pos == int(rk[k2]),
                        f"user {u}: rank mismatch, count {int(rk[k2])}, "
                        f"argsort {pos}",
                    )
            log.info("p6: argsort brute-force check passed on 50 users")
        finally:
            con.close()

    def report(self, params):
        doc = load_json(self.outputs()["metrics"]) or {}
        base = load_json(MODEL_DIR / "eval_baseline.json") or {}
        ap = load_json(_params_path()) or {}
        tuning = load_json(MODEL_DIR / "als_tuning.json") or {}
        lines = ["# P6 ALS baseline", ""]
        add = lines.append
        add(f"- model: {doc.get('model')}")
        add(
            f"- config: factors {ap.get('factors')}, iterations "
            f"{ap.get('iterations')}, reg {ap.get('regularization')}, "
            f"alpha {ap.get('alpha')}, seed {ap.get('seed')}"
        )
        win = tuning.get("winner")
        if win:
            same = win.get("als_params_json") == {k: ap.get(k) for k in ALS_KEYS}
            add(
                f"- selection: als_tune.py, winner of phase "
                f"{win.get('phase', 'A or B')}, val score "
                f"{win['score']:.5f} (mean NDCG@100 over val_last and "
                f"val_first of the 20,000 selection users). "
                + (
                    "als_params.json matches it."
                    if same
                    else "als_params.json does not match it."
                )
            )
        fit = doc.get("fit") or {}
        if fit:
            path = ", ".join(f"{j:.6e}" for j in fit.get("objective_after_items", []))
            add(
                f"- fit: solver {fit.get('solver')}, item-major csr "
                f"{fit.get('item_major')}, objective after each iteration: "
                f"{path}"
            )
        add(
            f"- eval users: {doc.get('n_eval_users')} "
            f"(dropped {doc.get('dropped_no_test_pick')})"
        )
        add(f"- candidate rule: {doc.get('candidate_rule')}")
        add(f"- tie rule: {doc.get('tie_rule')}")
        add("")
        for variant in ("last_ts", "first_ts", "val_last"):
            block = (doc.get("variants") or {}).get(variant) or {}
            ov = block.get("overall") or {}
            base_block = (base.get("variants") or {}).get(variant) or {}
            base_ov = base_block.get("overall") or {}
            add(f"## Variant {variant}")
            add("")
            add("| metric | popularity | ALS |")
            add("|---|---|---|")
            for metric in ("recall_at_10", "recall_at_100", "ndcg_at_100", "mrr"):
                add(
                    f"| {metric} | {base_ov.get(metric, float('nan')):.4f} | "
                    f"{ov.get(metric, 0):.4f} |"
                )
            add(
                f"| median rank | "
                f"{base_ov.get('median_rank', float('nan')):.0f} | "
                f"{ov.get('median_rank', 0):.0f} |"
            )
            add("")
            slices = block.get("slices") or {}
            if slices:
                for slice_name in ("tier", "pop_band", "cold_test_item", "user_decile"):
                    add(f"### {variant} by {slice_name}")
                    add("")
                    add("| value | users | R@10 | R@100 | NDCG@100 |")
                    add("|---|---|---|---|---|")
                    for value, sb in sorted((slices.get(slice_name) or {}).items()):
                        add(
                            f"| {value} | {sb['n_users']} | "
                            f"{sb['recall_at_10']:.4f} | "
                            f"{sb['recall_at_100']:.4f} | "
                            f"{sb['ndcg_at_100']:.4f} |"
                        )
                    add("")
        atomic_text_write("\n".join(lines) + "\n", self.outputs()["report"])
        log.info("p6 report written")


def _check_file_order(con):
    """The 16 train files must partition users into ordered ranges.
    Proven on the real data: zero-gap partition, 1..594415."""
    rows = con.execute(f"""
        SELECT filename, min(user_id) AS lo, max(user_id) AS hi
        FROM read_parquet('{MODEL_DIR}/interactions_train_*.parquet', filename=true)
        GROUP BY filename ORDER BY filename
    """).fetchall()
    check(len(rows) == 16, f"expected 16 train files, found {len(rows)}")
    prev = None
    for name, lo, hi in rows:
        check(
            prev is None or lo > prev,
            f"train files overlap or disorder at {name}: lo {lo}, prev {prev}",
        )
        prev = hi

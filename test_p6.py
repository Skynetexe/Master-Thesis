"""Checks for the revised pipeline/stage_p6.py on synthetic data.

Every path points into a temporary folder, so nothing touches the real
model folder. No DuckDB is needed: the DuckDB steps (_write_orientation,
_expected_nnz, _load_frame and evaluation.covariate_arrays) are replaced
by synthetic stand-ins.

1. Start from the state on disk today: a correct user-major CSR next to
   the old per-file item-major CSR. _build_csr() keeps the user-major
   CSR and replaces the item-major CSR with the exact transpose from the
   transpose cache, as hard links.
2. A second call resumes without work.
3. _fit() equals a direct als.fit() with the same parameters, bit for bit.
4. _evaluate() ranks equal a brute-force stable argsort, and the model
   label comes from als_params.json.
5. current_inputs() hashes CURATED/als_params.json and pipeline/als.py.
6. _verify_als() passes the good state. It fails for the old item-major
   CSR, for parameters that changed after the fit, and for a label from
   other parameters. Swapping the old CSR in by rename leaves the cache
   intact.
7. report() names the parameters and says whether they match the
   tuning winner.
8. A new train manifest rebuilds the user-major CSR (stand-in) and reuses
   the cache for the same data. Changed data rebuild the cache, and the
   item-major CSR follows.
9. _link_or_copy() copies when a hard link is impossible.

Usage, from the repository root:
    python3 test_p6.py
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from pipeline import als, stage_p6
from pipeline import evaluation as ev

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def raises(fn, text):
    try:
        fn()
    except Exception as exc:  # the real check() raises
        return text in str(exc)
    return False


tmp = Path(tempfile.mkdtemp(prefix="test_p6_"))
stage_p6.MODEL_DIR = tmp / "model"
stage_p6.SCRATCH_DIR = tmp / "scratch"
stage_p6.CURATED = tmp / "curated"
M, S, CUR = stage_p6.MODEL_DIR, stage_p6.SCRATCH_DIR, stage_p6.CURATED
for d in (M / "reports", S, CUR):
    d.mkdir(parents=True)
CACHE = S / "csr_item_major"

lines = []


class Keep(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


stage_p6.log.addHandler(Keep())
stage_p6.log.setLevel(logging.INFO)


def user_major(P, R):
    rows, cols = np.nonzero(P)
    indptr = np.zeros(P.shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    return (
        np.cumsum(indptr),
        cols.astype(np.int32),
        np.round(1000.0 * np.log1p(R[rows, cols])).astype(np.uint16),
    )


def p6_style_item_major(u, n_items, n_files=4):
    """The old _write_orientation for the item-major side."""
    indptr, indices, data = u
    n_users = len(indptr) - 1
    rows = np.repeat(np.arange(n_users), np.diff(indptr))
    edges = np.linspace(0, n_users, n_files + 1).astype(np.int64)
    idx, dat = [], []
    counts = np.zeros(n_items, dtype=np.int64)
    for f in range(n_files):
        m = (rows >= edges[f]) & (rows < edges[f + 1])
        o = np.lexsort((rows[m], indices[m]))
        idx.append(rows[m][o])
        dat.append(data[m][o])
        counts += np.bincount(indices[m], minlength=n_items)
    return (
        np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        np.concatenate(idx).astype(np.int32),
        np.concatenate(dat),
    )


def replace_array(path, arr):
    """Write by rename, as the pipeline does, so a hard link breaks."""
    t = path.with_name(path.stem + ".inject.npy")
    np.save(t, arr)
    os.replace(t, path)


rng = np.random.default_rng(6)
NU, NI = 90, 70
P = rng.random((NU, NI)) < 0.15
P[0, :] = P[:, 0] = False
P[4, :] = False  # a user without train data
P[:, 9] = False  # an item without train data
R = np.where(P, rng.pareto(1.2, (NU, NI)) * 20 + 1, 0.0)
u = user_major(P, R)
exact = user_major(P.T, R.T)
nnz = int(u[0][-1])

stage = stage_p6.StageP6()
out = stage.outputs()
for key, arr in zip(("csr_indptr", "csr_indices", "csr_data"), u):
    np.save(out[key], arr)
for (key, _), arr in zip(stage_p6.ITEM_MAJOR_FILES, p6_style_item_major(u, NI)):
    np.save(out[key], arr)
(M / "csr_meta.json").write_text(
    json.dumps({"rows": NU, "cols": NI, "nnz": nnz, "note": "old build"})
)
manifest = M / "train_manifest.json"
manifest.write_text(json.dumps({"interactions_train_00.parquet": {"rows": nnz}}))
(S / "p6_csr_state.json").write_text(
    json.dumps({"manifest": stage_p6.file_sha256(manifest), "nnz": nnz})
)
ap = {"factors": 6, "iterations": 3, "regularization": 0.5, "alpha": 5.0, "seed": 11}
(CUR / "als_params.json").write_text(json.dumps(ap))


def item_major_now():
    return [np.load(out[k]) for k, _ in stage_p6.ITEM_MAJOR_FILES]


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


# 1: today's state on disk.
check(
    not same(item_major_now(), exact),
    "setup: the old per-file item-major CSR differs from the transpose",
)
fp = stage._build_csr(None, out)
check(
    same(item_major_now(), exact),
    "_build_csr() replaces the old item-major CSR with the exact transpose",
)
check(
    same([np.load(out[k]) for k in ("csr_indptr", "csr_indices", "csr_data")], u)
    and any("user-major csr resume" in m for m in lines),
    "_build_csr() keeps the user-major CSR",
)
state = json.loads((S / "p6_csr_state.json").read_text())
check(
    state.get("item_major") == als.TRANSPOSE_VERSION
    and state.get("user_major_sha256") == fp
    and fp == als.user_major_fingerprint(u),
    "the csr state records the transpose build and the user-major SHA-256",
)
check(
    all(
        os.stat(out[k]).st_ino == os.stat(CACHE / name).st_ino
        for k, name in stage_p6.ITEM_MAJOR_FILES
    )
    and not list(M.glob("*.tmp.npy")),
    "the item-major files are hard links into the transpose cache, with "
    "no temporary names left behind",
)

# 2: resume.
lines.clear()
check(
    stage._build_csr(None, out) is None
    and any("csr artifacts resume" in m for m in lines),
    "a second _build_csr() call resumes without work",
)

# 3: fit.
stage._ap = ap
inter = stage._interactions(out)
stage._fit(inter, out)
Xd, Yd, _ = als.fit(
    als.Interactions(*u, *exact),
    stage_p6._als_params(ap),
    tmp / "ref" / "x.npy",
    tmp / "ref" / "y.npy",
    tmp / "ref" / "s.json",
    log=lambda m: None,
)
X6, Y6 = np.load(out["user_factors"]), np.load(out["item_factors"])
check(
    np.array_equal(X6, Xd) and np.array_equal(Y6, Yd),
    "_fit() equals a direct als.fit() with the same parameters, bit for bit",
)

# 4: evaluation.
users_e = np.array([x for x in range(1, NU) if P[x].sum() >= 3][:25])
picks_e = np.array(
    [rng.choice(np.flatnonzero(~P[x])[1:], size=3, replace=False) for x in users_e]
)
stage._load_frame = lambda con: (
    users_e,
    np.ones(len(users_e), dtype=int),
    picks_e[:, 0],
    picks_e[:, 1],
    picks_e[:, 2],
    0,
)
real_cov = getattr(ev, "covariate_arrays", None)
ev.covariate_arrays = lambda con, n: (
    np.full(n, 95),
    np.full(n, 200),
    np.ones(n, dtype=np.int64),
)
try:
    stage._evaluate(None, inter, out)
finally:
    if real_cov is not None:
        ev.covariate_arrays = real_cov
doc = json.loads(out["metrics"].read_text())
check(
    doc["model"] == "als: factors 6, alpha 5 on log1p, reg 0.5, " "3 iterations",
    f"the model label comes from als_params.json ({doc['model']})",
)
rk = np.load(out["ranks"])
bad = 0
for k, x in enumerate(users_e):
    s = (X6[x] @ Y6.T).astype(np.float64)
    s[0] = -np.inf
    s[np.flatnonzero(P[x])] = -np.inf
    order = np.argsort(-s, kind="stable")
    for j, name in enumerate(("rank_test_last", "rank_test_first", "rank_val_last")):
        bad += int(
            int(np.flatnonzero(order == picks_e[k, j])[0]) + 1 != int(rk[name][k])
        )
check(bad == 0, f"_evaluate() ranks equal a stable argsort ({bad} " f"mismatches)")

# 5: marker inputs.
ci = stage.current_inputs([])
check(
    ci["als_params"] == stage_p6.file_sha256(CUR / "als_params.json")
    and ci["als_module"] == stage_p6.file_sha256(Path(als.__file__)),
    "current_inputs() hashes CURATED/als_params.json and pipeline/als.py",
)

# 6: the new verify checks.
inter = stage._interactions(out)
check(
    not raises(lambda: stage._verify_als(inter, ap, out), ""),
    "_verify_als() passes the good state",
)
for (key, _), arr in zip(stage_p6.ITEM_MAJOR_FILES, p6_style_item_major(u, NI)):
    replace_array(out[key], arr)
check(
    raises(lambda: stage._verify_als(stage._interactions(out), ap, out), "item-major"),
    "_verify_als() fails for the old per-file item-major CSR",
)
check(
    same([np.load(CACHE / n) for _, n in stage_p6.ITEM_MAJOR_FILES], exact),
    "replacing the item-major files by rename leaves the cache intact",
)
for key, name in stage_p6.ITEM_MAJOR_FILES:
    stage_p6._link_or_copy(CACHE / name, out[key])
ap2 = dict(ap, regularization=0.6)
check(
    raises(
        lambda: stage._verify_als(stage._interactions(out), ap2, out),
        "not a finished fit",
    ),
    "_verify_als() fails when the parameters changed after the fit",
)
doc_old = dict(doc, model="als: factors 64, alpha 40 on log1p, reg 1.0")
out["metrics"].write_text(json.dumps(doc_old))
check(
    raises(
        lambda: stage._verify_als(stage._interactions(out), ap, out),
        "eval_als.json names the model",
    ),
    "_verify_als() fails for a model label from other parameters",
)
out["metrics"].write_text(json.dumps(doc))

# 7: report.
(M / "eval_baseline.json").write_text(json.dumps({"variants": {}}))
(M / "als_tuning.json").write_text(
    json.dumps({"winner": {"phase": "C", "score": 0.0171, "als_params_json": ap}})
)
stage.report({})
text = out["report"].read_text()
check(
    doc["model"] in text
    and "als_params.json matches it." in text
    and "objective after each iteration" in text,
    "report() names the parameters, the fit path and the tuning match",
)
(M / "als_tuning.json").write_text(
    json.dumps(
        {
            "winner": {
                "phase": "C",
                "score": 0.0171,
                "als_params_json": dict(ap, alpha=1.0),
            }
        }
    )
)
stage.report({})
check(
    "does not match" in out["report"].read_text(),
    "report() flags parameters that differ from the tuning winner",
)

# 8: a new manifest.
current = {"u": u}


def stand_in_write(
    con,
    order_sql,
    count_key,
    index_key,
    slots,
    indices_path,
    data_path,
    indptr_path,
    manifest_doc,
    n,
    label,
):
    np.save(indptr_path, current["u"][0])
    np.save(indices_path, current["u"][1])
    np.save(data_path, current["u"][2])


stage._write_orientation = stand_in_write
stage._expected_nnz = lambda con: int(current["u"][0][-1])
space_calls = []  # the real check wants 48 GiB free, more than /tmp has
stage_p6.require_space = lambda path, n: space_calls.append((path, n))
(M / "id_spaces.json").write_text(
    json.dumps({"users": {"slots": NU}, "items": {"slots": NI}})
)
manifest.write_text(
    json.dumps({"interactions_train_00.parquet": {"rows": nnz}, "rebuild": 1})
)
lines.clear()
stage._build_csr(None, out)
check(
    any("reusing" in m for m in lines)
    and same(item_major_now(), exact)
    and space_calls == [(M, 48 * 2**30)],
    "a new manifest with the same data checks disk space, rebuilds the "
    "user-major CSR and reuses the transpose cache",
)
R2 = R.copy()
k0 = tuple(np.argwhere(P)[3])
R2[k0] += 40.0
current["u"] = user_major(P, R2)
manifest.write_text(
    json.dumps({"interactions_train_00.parquet": {"rows": nnz}, "rebuild": 2})
)
lines.clear()
stage._build_csr(None, out)
check(
    any("building" in m for m in lines)
    and same(item_major_now(), user_major(P.T, R2.T)),
    "changed data rebuild the cache, and the item-major CSR follows",
)

# 9: copy fallback.
real_link = os.link


def no_link(*a, **k):
    raise OSError("cross-device link")


os.link = no_link
try:
    stage_p6._link_or_copy(CACHE / "csr_item_data.npy", tmp / "copied.npy")
finally:
    os.link = real_link
check(
    np.array_equal(np.load(tmp / "copied.npy"), np.load(CACHE / "csr_item_data.npy"))
    and os.stat(tmp / "copied.npy").st_ino
    != os.stat(CACHE / "csr_item_data.npy").st_ino,
    "_link_or_copy() copies when a hard link is impossible",
)
stage_p6._link_or_copy(CACHE / "csr_item_data.npy", out["csr_data_t"])
stage_p6._link_or_copy(CACHE / "csr_item_data.npy", out["csr_data_t"])
check(
    not list(M.glob("*.tmp.npy"))
    and os.stat(out["csr_data_t"]).st_ino
    == os.stat(CACHE / "csr_item_data.npy").st_ino,
    "placing a link that already exists leaves no temporary name behind",
)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL P6 CHECKS PASSED")

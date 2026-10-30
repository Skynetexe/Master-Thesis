"""Smoke test of the runner stages after P13b part 2 on the P13 chain of
p13_chain.py. Every path points into a temporary folder, and nothing
touches the real model folder.

The modules first write their outputs as their own entry points write
them. Then, through pipeline.runner.execute():
1. A plain call stops at the guard and writes nothing.
2. --adopt writes the markers of the five stages with outputs, finds
   nothing to adopt for p13c_final, and changes no output byte.
3. A plain call skips those five stages and runs p13c_final. P13d keeps
   the model on the chain, so p13c_final writes the pointer only. A third
   call skips all six.
4. A changed hash of gam_refit.py (the test changes the hash, not the
   file) makes the markers of the stages that run gam_refit.py stale, and
   only those. A plain call then stops at the guard.
5. A changed upstream output and a corrupted output each stop the
   adoption, and no output changes. With both files restored, the
   adoption passes.
6. A forced bounded P13d model (as test_p13d.py forces it): the adoption,
   then p13c_final runs P13c on the bounded model through the runner and
   writes the tagged outputs and the pointer.
7. ALS phase D: the adoption of a tiny synthetic run of run_phase(), and
   three refusals: a corrupted best.json, a changed als_tuning.json and
   changed P6 data.

Usage, from the repository root:
    python3 test_runner_p13.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import als
from pipeline import als_freq
from pipeline import als_tune_d as TD
from pipeline import common
from pipeline import config
from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
from pipeline import itemknn_select as IK
from pipeline import p13c as PC
from pipeline import p13d as D
from pipeline import runner as RN
from pipeline import script_stage as SS
from pipeline import stage_p13b2 as s13b2
from pipeline.common import VerificationError, file_sha256

failures = []
T0 = time.time()
NBOOT = {"listener": 6, "twoway": 3}
NBOOT_SWITCH = {"listener": 4, "twoway": 2}
UPSTREAM = ("p6_als", "p9b_colisten", "p13b2_loss")
SIX = [
    "itemknn_select",
    "gam_knn_slices",
    "gam_refit",
    "p13c_stability",
    "p13d_bounds",
    "p13c_final",
]
FIVE = SIX[:5]
PARAMS = dict(config.DEFAULT_PARAMS)


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(msg):
    pass


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


CAP = Capture()
common.log.addHandler(CAP)
common.log.setLevel(logging.INFO)
logging.getLogger().addHandler(logging.NullHandler())  # no stderr copies


def stage():
    return C.p13_loaders(s13b2.StageP13b2())


def fake_marker(name):
    """A current marker of an upstream stage that the chain does not run."""
    st = RN.STAGE_BY_NAME[name]
    common.write_marker(
        name,
        RN.stage_params_digest(st, PARAMS),
        {"test": name},
        {},
        source_sha=RN.stage_source_sha(st),
    )


def stages(model, scratch):
    """The six stages on model and scratch, with the world's loaders."""
    ik = IK.StageItemKNN(model, scratch)
    ik._load = lambda: (
        W.NI,
        C.U_ARRAYS,
        W.SEL.copy(),
        W.SEL_PICKS.copy(),
        C.POP,
        C.LISTS,
    )
    sl = G.StageGamKnnSlices(model, scratch)
    sl._load = lambda: (W.NI, C.U_ARRAYS, C.LISTS, C.pop_centiles())
    out = [
        ik,
        sl,
        GR.StageGamRefit(model, scratch),
        PC.StageP13c("gam_refit", model, scratch),
        D.StageP13d(model, scratch),
        PC.StageP13c("p13d", model, scratch),
    ]
    for s in out[2:]:
        s._stage = stage
    out[3].n_boot, out[5].n_boot = NBOOT, NBOOT_SWITCH
    out[3].n_verify = out[5].n_verify = 6
    return out


def call(sts, **kw):
    """runner.execute() on the test stages. Returns (exit code, lines)."""
    CAP.lines.clear()
    by_name = dict(RN.STAGE_BY_NAME)
    by_name.update({s.name: s for s in sts})
    rc = RN.execute(sts, PARAMS, [], by_name=by_name, **kw)
    return rc, list(CAP.lines)


def refusal(sts, **kw):
    """The message of the VerificationError that execute() raises, or
    None."""
    try:
        call(sts, **kw)
    except VerificationError as exc:
        return str(exc)
    return None


def snapshot(model):
    """SHA-256 of every file in the model folder, the markers aside."""
    return {
        p.relative_to(model).as_posix(): file_sha256(p)
        for p in sorted(model.rglob("*"))
        if p.is_file() and "markers" not in p.relative_to(model).parts
    }


def markers(model):
    return {p.stem for p in (model / "markers").glob("*.json")}


def skipped(lines):
    return [ln.split()[1] for ln in lines if ln.startswith("skip ")]


def current(sts):
    return [
        s.name
        for s in sts
        if RN._stage_current(
            s, RN.stage_params_digest(s, PARAMS), RN.stage_source_sha(s), []
        )
    ]


def clock():
    return f"{time.time() - T0:.0f} s"


print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_runner_p13_"))
ROOT = TMP / "chain"
CHAIN = C.chain_into(ROOT)
SCR = ROOT / "scratch"
common.MODEL_DIR = CHAIN
for name in UPSTREAM:
    fake_marker(name)

# The modules write their outputs as their own entry points write them,
# each with its provenance.
IK.run(
    CHAIN,
    W.NI,
    C.U_ARRAYS,
    W.SEL.copy(),
    W.SEL_PICKS.copy(),
    C.POP,
    C.LISTS,
    workers=1,
    say=quiet,
    n_check=12,
    inputs=IK.provenance(CHAIN),
)
G.run_slices(
    CHAIN,
    W.NI,
    C.U_ARRAYS,
    C.LISTS,
    C.pop_centiles(),
    workers=1,
    say=quiet,
    n_check=5,
    inputs=G.provenance(CHAIN),
)
GR.run(
    stage(),
    CHAIN,
    SCR / "gam_refit",
    say=quiet,
    n_verify=6,
    inputs=GR.provenance(CHAIN),
)
PC.run(
    stage(),
    CHAIN,
    SCR / "p13c",
    say=quiet,
    n_verify=6,
    n_boot=NBOOT,
    inputs=PC.provenance(CHAIN),
)
D.run(stage(), CHAIN, SCR / "p13d", say=quiet, n_verify=6, inputs=D.provenance(CHAIN))
P13C_SHA = {f: file_sha256(CHAIN / f) for f in PC.OUT_FILES}
print(f"      outputs written ({clock()})")
STS = stages(CHAIN, SCR)
names = {s.name: [p.name for p in s.code_files()] for s in STS}
check(
    all("script_stage.py" in v and "runner.py" not in v for v in names.values())
    and "gam_refit.py" in names["p13c_stability"]
    and "p13c.py" in names["p13d_bounds"]
    and "gam_refit.py" not in names["gam_knn_slices"]
    and "itemknn_select.py" in names["gam_refit"],
    "code_files(): each stage hashes the pipeline modules it can run "
    "(p13c_stability holds gam_refit.py, p13d_bounds holds p13c.py), "
    "script_stage.py, and never runner.py",
)

# 1: the guard
before = snapshot(CHAIN)
rc, lines = call(STS)
check(
    rc == 2
    and any(
        "stage itemknn_select found its outputs without a current marker" in ln
        and "--only itemknn_select --adopt" in ln
        and "--only itemknn_select --force" in ln
        for ln in lines
    )
    and snapshot(CHAIN) == before
    and markers(CHAIN) == set(UPSTREAM),
    "1: a plain call stops at the guard of the first stage, names --adopt "
    "and --force, and writes no output and no marker",
)

# 2: the adoption
rc, lines = call(STS, adopt=True)
mk = {n: json.loads(common.marker_path(n).read_text()) for n in FIVE}
by = {s.name: s for s in STS}
check(
    rc == 0
    and markers(CHAIN) == set(UPSTREAM) | set(FIVE)
    and [ln.split()[1] for ln in lines if ln.startswith("adopted ")] == FIVE
    and any("nothing to adopt for p13c_final" in ln for ln in lines)
    and snapshot(CHAIN) == before,
    f"2: --adopt writes the five markers, finds nothing to adopt for "
    f"p13c_final and changes no output byte ({clock()})",
)
rc_only, lines_only = call(STS, adopt=True, only="p13c_final")
check(
    rc_only == 2
    and any("nothing to adopt for p13c_final" in ln for ln in lines_only)
    and not common.marker_path("p13c_final").exists(),
    "2: --adopt --only p13c_final stops: p13c_final has no outputs yet",
)
check(
    all(
        mk[n]["inputs"] == by[n].current_inputs([])
        and all(
            v["sha256"] == file_sha256(Path(v["path"]))
            for v in mk[n]["outputs"].values()
        )
        and mk[n]["source_sha"] == RN.stage_source_sha(by[n])
        for n in FIVE
    ),
    "2: each adopted marker records the stage's inputs, its source hash "
    "and the SHA-256 of each output",
)

# 3: skips, then p13c_final's pointer
rec = json.loads((CHAIN / D.OUT_FILES[0]).read_text())["model"]
rc, lines = call(STS)
ptr = json.loads((CHAIN / PC.FINAL_FILE).read_text())
check(
    rec["changed"] is False
    and rc == 0
    and skipped(lines) == FIVE
    and "run p13c_final" in lines
    and "done p13c_final" in lines
    and any("P13d kept gam_refit's model" in ln for ln in lines)
    and ptr["model"]["changed"] is False
    and ptr["files"]
    == {
        "results": "p13c_model.json",
        "arrays": "p13c_arrays.npz",
        "report": "reports/p13c_report.md",
    }
    and ptr["sha256"]["results"] == P13C_SHA["p13c_model.json"]
    and snapshot(CHAIN)
    == dict(before, **{PC.FINAL_FILE: file_sha256(CHAIN / PC.FINAL_FILE)}),
    "3: a plain call skips the five adopted stages, and p13c_final writes "
    "only the pointer to p13c_stability's outputs (P13d kept the model)",
)
rc, lines = call(STS)
SNAP3 = snapshot(CHAIN)
check(
    rc == 0
    and skipped(lines) == SIX
    and not any(ln.startswith(("run ", "adopt")) for ln in lines),
    "3: a third call skips all six stages",
)

# 4: a changed shared module
real_sha = SS.file_sha256
SS.file_sha256 = lambda p: ("0" * 64 if Path(p).name == "gam_refit.py" else real_sha(p))
try:
    now = current(STS)
    rc, lines = call(STS)
    snap4 = snapshot(CHAIN)
finally:
    SS.file_sha256 = real_sha
check(
    now == ["itemknn_select", "gam_knn_slices"]
    and rc == 2
    and skipped(lines) == ["itemknn_select", "gam_knn_slices"]
    and any(
        "stage gam_refit found its outputs without a current marker" in ln
        for ln in lines
    )
    and snap4 == SNAP3,
    "4: a changed gam_refit.py makes gam_refit, p13c_stability, "
    "p13d_bounds and p13c_final stale (not itemknn_select or "
    "gam_knn_slices), and a plain call stops at the guard",
)
check(current(STS) == SIX, "4: with the real hash, all six are current " "again")

# 5: a changed input, a corrupted output
mk_d = common.marker_path("p13d_bounds")
old_time = json.loads(mk_d.read_text())["finished_utc"]
mk_d.unlink()
arr_path = CHAIN / PC.OUT_FILES[1]
arr_bytes = arr_path.read_bytes()
z = dict(np.load(arr_path))
z["extra"] = np.zeros(1)
np.savez(arr_path, **z)
snap5 = snapshot(CHAIN)
msg = refusal(STS, adopt=True)
check(
    msg is not None
    and "p13c_arrays.npz" in msg
    and "--only p13d_bounds --force" in msg
    and not mk_d.exists()
    and snapshot(CHAIN) == snap5,
    "5: an edited p13c_arrays.npz (an extra array) stops the adoption of "
    "p13d_bounds, names the stale input and writes nothing",
)
arr_path.write_bytes(arr_bytes)
d_path = CHAIN / D.OUT_FILES[0]
d_bytes = d_path.read_bytes()
doc = json.loads(d_bytes)
doc["s1"]["bounds_kept"] = not doc["s1"]["bounds_kept"]
d_path.write_text(json.dumps(doc))
snap5 = snapshot(CHAIN)
msg = refusal(STS, adopt=True)
check(
    msg is not None
    and msg.startswith("p13d_bounds: ")
    and "'s1'" in msg
    and not mk_d.exists()
    and snapshot(CHAIN) == snap5,
    "5: a p13d.json with S1 flipped fails the read-back, so the adoption " "stops",
)
d_path.write_bytes(d_bytes)
# A marker holds its time to the second. On a fast machine the adoption
# below can fall in the second of the first one, write the same bytes and
# leave p13c_final current. So wait for the next second.
while common.utc_now() == old_time:
    time.sleep(0.05)
rc, lines = call(STS, adopt=True)
check(
    rc == 0
    and mk_d.exists()
    and [ln.split()[1] for ln in lines if ln.startswith("adopted ")]
    == ["p13d_bounds", "p13c_final"]
    and snapshot(CHAIN) == SNAP3,
    f"5: with both files restored, p13d_bounds is adopted again, and so is "
    f"p13c_final, whose marker held the old p13d marker ({clock()})",
)

# 6: the bounded model through p13c_final
F = TMP / "forced"
shutil.copytree(ROOT, F)
C.configure(F)
FM = F / "model"
common.MODEL_DIR = FM
for name in SIX:
    common.marker_path(name).unlink()
(FM / PC.FINAL_FILE).unlink()
STF = stages(FM, F / "scratch")
real_decide = D.decide


def forced(scores, names_, model_arm):
    res = real_decide(scores, names_, model_arm)
    res["s2"] = dict(res["s2"], chosen=D.PILOT)
    return res


D.decide = forced
try:
    D.run(
        stage(),
        FM,
        F / "scratch" / "p13d",
        say=quiet,
        n_verify=6,
        inputs=D.provenance(FM),
    )
    rc_a, lines_a = call(STF, adopt=True)
finally:
    D.decide = real_decide
recf = json.loads((FM / D.OUT_FILES[0]).read_text())["model"]
tag = recf["tag"]
check(
    recf["changed"]
    and recf["fit_name"] == D.PILOT
    and rc_a == 0
    and [ln.split()[1] for ln in lines_a if ln.startswith("adopted ")] == FIVE
    and any("nothing to adopt for p13c_final" in ln for ln in lines_a),
    "6: with the forced bounded model, the five stages are adopted, and "
    "p13c_final has nothing to adopt",
)
rc, lines = call(STF)
files, report = PC.out_names(tag)
ptr = json.loads((FM / PC.FINAL_FILE).read_text())
outc = json.loads((FM / files[0]).read_text())
mk_f = json.loads(common.marker_path("p13c_final").read_text())
log_f = FM / "logs" / f"p13c_{tag}.log"
check(
    rc == 0
    and skipped(lines) == FIVE
    and "done p13c_final" in lines
    and ptr["model"]["changed"]
    and ptr["model"]["fit_name"] == D.PILOT
    and ptr["files"]
    == {"results": files[0], "arrays": files[1], "report": f"reports/{report}"}
    and ptr["sha256"]["arrays"] == file_sha256(FM / files[1])
    and outc["model_source"]["fit"] == D.PILOT
    and outc["check"].get("ranks_equal_p13d") is True
    and outc["inputs"] == ptr["inputs"] == STF[5].provenance()
    and sorted(mk_f["outputs"]) == ["arrays", "pointer", "report", "results"]
    and log_f.exists()
    and "p13c: the model switch" in log_f.read_text(),
    f"6: p13c_final runs P13c on the bounded model through the runner: "
    f"tagged outputs, their log, the pointer to them and a marker over "
    f"all four ({clock()})",
)
check(
    {f: file_sha256(FM / f) for f in PC.OUT_FILES} == P13C_SHA,
    "6: the outputs of p13c_stability stay unchanged",
)
rc, lines = call(STF)
check(rc == 0 and skipped(lines) == SIX, "6: a further call skips all six")

# 7: ALS phase D


def build(P, R):
    """Interactions in P6's layout (as in test_als_d.py)."""
    data_dense = np.where(P > 0, np.round(1000.0 * np.log1p(R)), 0)

    def csr(Mx, Dx):
        rows, cols = np.nonzero(Mx)
        indptr = np.zeros(Mx.shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr)
        return (indptr, cols.astype(np.int32), Dx[rows, cols].astype(np.uint16))

    return als.Interactions(*csr(P, data_dense), *csr(P.T, data_dense.T))


A = TMP / "als"
AM = A / "model"
(AM / "reports").mkdir(parents=True)
common.MODEL_DIR = AM
rng = np.random.default_rng(29)
NU, NI = 120, 95
Pm = (rng.random((NU, NI)) < 0.14).astype(np.float64)
Pm[0, :] = 0
Pm[:, 0] = 0
Rm = np.where(Pm > 0, rng.pareto(1.2, (NU, NI)) * 20 + 1, 0.0)
inter = build(Pm, Rm)
inter2 = build(Pm, Rm * 2.0)
users = np.arange(1, 60)
picks = np.stack(
    [rng.integers(1, NI, len(users)), rng.integers(1, NI, len(users))], axis=1
)
ac = {
    "winner": {
        "alpha": 2.0,
        "reg": 0.3,
        "score": 0.0,
        "seed": 4,
        "factors": 64,
        "iterations": 2,
    },
    "reference": {"score": 0.001},
}
ac_path = AM / "als_tuning.json"
ac_path.write_text(json.dumps(ac))
masses = als_freq.mass_of(inter, A / "mass.npz")
real_f = (TD.FACTORS_128, TD.FACTORS_256)
real_steps = dict(TD.MAX_STEPS)
TD.FACTORS_128, TD.FACTORS_256 = 5, 7
TD.MAX_STEPS.clear()
TD.MAX_STEPS.update({5: 2, 7: 1})
try:
    res = TD.run_phase(
        inter,
        users,
        picks,
        masses,
        ac,
        AM / "als_tuning_d.json",
        A / "scratch" / "als_tune_d",
        AM / "als_d",
        workers=1,
        say=quiet,
        max_iter=5,
        patience=2,
    )
finally:
    TD.FACTORS_128, TD.FACTORS_256 = real_f
    TD.MAX_STEPS.clear()
    TD.MAX_STEPS.update(real_steps)
TD.write_report(res, AM / "reports" / "als_tuning_d.md")
fake_marker("p6_als")
sa = TD.StageALSD(AM, A / "scratch")
sa._p6 = lambda: inter
snap7 = snapshot(AM)
rc, lines = call([sa])
check(
    rc == 2 and any("stage als_phase_d found its outputs" in ln for ln in lines),
    "7: a plain call stops at the guard of als_phase_d",
)
rc, lines = call([sa], adopt=True)
mk_a = common.marker_path("als_phase_d")
check(
    rc == 0
    and mk_a.exists()
    and "adopted als_phase_d" in lines
    and len([k for k in sa.outputs() if k.startswith("factors_")]) >= 1
    and snapshot(AM) == snap7,
    "7: the tiny phase D run is adopted: the freshness rule (phase A to C "
    "winner, P6's shape and fingerprint) and the read-back pass, and no "
    "output changes",
)
mk_a.unlink()
best_path = AM / "als_d" / "best.json"
best_bytes = best_path.read_bytes()
best = json.loads(best_bytes)
best["score"] = best["score"] + 1e-6
best_path.write_text(json.dumps(best))
msg_best = refusal([sa], adopt=True)
best_path.write_bytes(best_bytes)
ac2 = json.loads(json.dumps(ac))
ac2["winner"]["reg"] = 0.31
ac_path.write_text(json.dumps(ac2))
msg_ac = refusal([sa], adopt=True)
ac_path.write_text(json.dumps(ac))
sa._p6 = lambda: inter2
msg_p6 = refusal([sa], adopt=True)
sa._p6 = lambda: inter
check(
    msg_best is not None
    and "best.json" in msg_best
    and msg_ac is not None
    and "als_tuning.json (the phase A to C winner's reg)" in msg_ac
    and msg_p6 is not None
    and "P6's interactions (their fingerprint)" in msg_p6
    and not mk_a.exists()
    and snapshot(AM) == snap7,
    "7: a corrupted best.json, a changed phase A to C winner and changed "
    "P6 data each stop the adoption, and nothing is written",
)
rc, lines = call([sa], adopt=True)
rc2, lines2 = call([sa])
check(
    rc == 0 and rc2 == 0 and skipped(lines2) == ["als_phase_d"],
    "7: with the files restored, the adoption passes, and a plain call "
    "skips als_phase_d",
)

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)

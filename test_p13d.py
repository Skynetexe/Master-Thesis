"""Smoke test of p13d.py and of the model switch of p13c.py on the P13
chain of p13_chain.py (P11 to P13b part 2 and ItemKNN on the synthetic
world of p13_world.py). Every path points into a temporary folder, and
nothing touches the real model folder.

1. The chain, the slices of gam_knn_diag.py, gam_refit.py and p13c.py
   (the default model, a few replicates).
2. p13d.py with pilot: the model checks and one bounded fit are saved,
   and no output is written.
3. p13d.py: the ten fits, the rules, the outputs and the read-back.
   Checked here: the layouts, the bounds of every fit from its tables,
   rules S1 to S3 from the stored ranks with an NDCG written here, S4
   from the stored member tables, the model record through p13c.py's own
   loader, and a resumed run.
4. A forced bounded model (S2's choice replaced, as p13_chain.py forces
   P13b's choice), then p13c.py with --model p13d: tagged outputs, the
   order of every fit, and the outputs of the first p13c run unchanged.

Usage, from the repository root:
    python3 test_p13d.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.common import file_sha256

from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
from pipeline import p13c as PC
from pipeline import p13d as D

failures = []
lines = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(msg):
    pass


def keep(msg):
    lines.append(msg)


def stage():
    return C.p13_loaders(s13b2.StageP13b2())


def ndcg(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    return np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0).mean(axis=-1)


def interval(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    return m, m - 1.959963984540054 * se


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def order_break(T, lay):
    """The largest break of the content terms' order, from the tables."""
    worst = 0.0
    for t, off in zip(lay.terms, lay.offsets):
        if t.name not in D.CONTENT:
            continue
        h = np.asarray(T[off : off + t.K])
        worst = max(worst, float(np.max(np.maximum(h[:-1] - h[1:], 0.0))))
        if t.zero_state:
            worst = max(worst, float(T[off + t.K] - h[0]))
    return worst


print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_p13d_"))
CHAIN = C.chain_into(TMP / "chain")
SCR = TMP / "chain" / "scratch"
G.run_slices(
    CHAIN, W.NI, C.U_ARRAYS, C.LISTS, C.pop_centiles(), workers=1, say=quiet, n_check=5
)

# 1: gam_refit and p13c
GR.run(stage(), CHAIN, SCR / "gam_refit", say=quiet, n_verify=6, inputs={})
PC.run(
    stage(),
    CHAIN,
    SCR / "p13c",
    say=quiet,
    n_verify=6,
    n_boot={"listener": 6, "twoway": 3},
)
P13C_SHA = {f: file_sha256(CHAIN / f) for f in PC.OUT_FILES}
doc_c = json.loads((CHAIN / "p13c_model.json").read_text())
check(
    doc_c["check"]["ranks_equal_gam_refit"] is True
    and "model_source" not in doc_c
    and set(doc_c["exports"]) == set(PC.OUT_FILES),
    "p13c.py's default path writes the first run's names and checks",
)
BASE = stage()._base_context()
M, _ = PC.load_model(CHAIN, BASE)
print(
    f"      model {M['name']} ({len(M['kept'])} terms), "
    f"{'-'.join(M['arm'])}, level {M['level']}, lambda {M['lambda']}"
)

# 2: the pilot
WORK = SCR / "p13d"
pilot = D.run(stage(), CHAIN, WORK, say=keep, n_verify=6, pilot=True)
saved = json.loads((WORK / "fits.json").read_text())["fits"]
check(
    sorted(saved) == sorted(["check|model", "check|mix32-B", D.PILOT])
    and not (CHAIN / D.OUT_FILES[0]).exists()
    and not (CHAIN / D.OUT_FILES[1]).exists(),
    "the pilot saves the model checks and the bounded p_pop-B at lambda "
    "only, and writes no output",
)
check(
    pilot["pilot"] == D.PILOT
    and pilot["projected_total"] > 0
    and any(
        "pilot done" in m
        and "Next: python3 -m pipeline.runner " "--only p13d_bounds" in m
        for m in lines
    ),
    "the pilot prints its time and the projected total",
)

# 3: the full run
lines.clear()
out = D.run(stage(), CHAIN, WORK, say=keep, n_verify=6)
doc = json.loads((CHAIN / D.OUT_FILES[0]).read_text())
z = np.load(CHAIN / D.OUT_FILES[1])
names = [str(x) for x in z["names"]]
check(
    names == D.ALL_FITS
    and len(names) == 14
    and len(D.NEW_FITS) == 10
    and len(D.PASS_FITS) == 12,
    "fourteen fits in the declared order, ten of them new, twelve in the " "pass",
)
lay_b, idx_b = PC.layout_of(BASE, M["kept"], D.CONTENT)
check(
    all(t.monotone == (1 if t.name in D.CONTENT else 0) for t in lay_b.terms)
    and all(t.monotone == 0 for t in M["lay"].terms)
    and np.array_equal(idx_b, M["idx"])
    and all("artist" not in n or n == "artist_tag_affinity" for n in D.CONTENT),
    "the bounded layout bounds the four content terms only, and the artist "
    "terms stay free",
)
lays = {"free": M["lay"], "bounded": lay_b}
breaks, kkt = [], []
for i, n in enumerate(names):
    fam = D.parse(n)[0]
    T = lays[fam].tables(z["theta"][i])
    if fam == "bounded":
        breaks.append(order_break(T, lay_b))
    if n in D.NEW_FITS:
        kkt.append(doc["fits"][n]["proj_grad_max"])
check(
    len(breaks) == 6 and max(breaks) <= s13b.MONOTONE_TOL,
    f"every bounded fit keeps the content terms' order, by its tables "
    f"(largest break {max(breaks):.1e})",
)
check(
    max(kkt) <= s13.KKT_TOL, f"every new fit reached KKT_TOL (largest {max(kkt):.1e})"
)
check(
    doc["fits"][D.PILOT]["start"] == D.MODEL
    and doc["fits"]["bounded|p_pop-B|lambda*10"]["start"] == D.PILOT
    and doc["fits"]["free|mix32-B|lambda/10"]["start"] == D.MIX,
    "the warm starts follow the protocol",
)

S = np.stack([ndcg(z["ranks"][i]) for i in range(len(names))])
ix = {n: i for i, n in enumerate(names)}
ses = 0.02 * S[ix[D.MODEL]].mean()
_, lo1 = interval(S[ix[D.PILOT]], S[ix[D.MODEL]])
keep_b = lo1 > -ses
check(
    doc["s1"]["bounds_kept"] == bool(keep_b)
    and close(doc["s1"]["difference"]["lo"], lo1, 1e-9)
    and close(doc["sesoi"]["abs"], ses, 1e-12),
    f"S1 from the stored ranks (lower end {lo1:+.5f}, threshold "
    f"{-ses:+.5f}, bounds kept: {keep_b})",
)
fam = "bounded" if keep_b else "free"
start = D.PILOT if keep_b else D.MODEL
family = [n for n in names if n.startswith(fam + "|") and "-B|" in n]
elig = []
for n in family:
    if n != start:
        m, lo = interval(S[ix[n]], S[ix[start]])
        if m > ses and lo > 0:
            elig.append(n)
check(
    len(family) == 6
    and doc["s2"]["family"] == fam
    and doc["s2"]["start"] == start
    and sorted(doc["s2"]["eligible"]) == sorted(elig)
    and (elig or doc["s2"]["chosen"] == start),
    f"S2 from the stored ranks (eligible: {elig or 'none'}, chosen "
    f"{doc['s2']['chosen']})",
)
means = S.mean(axis=1)
members = [n for i, n in enumerate(names) if means[i] >= means.max() - ses]
check(
    doc["s3"]["members"] == members and [str(x) for x in z["members"]] == members,
    f"S3 from the stored ranks ({len(members)} members)",
)
check(
    np.array_equal(
        z["ranks"][ix[D.MODEL]], np.load(CHAIN / "p13c_arrays.npz")["ranks"][0]
    ),
    "the model's ranks equal P13c's",
)

Sr = PC.reason_sample(CHAIN, BASE)
Rr = PC.Reasons(BASE, M, Sr)
vals = [Rr.values(z["member_tables"][k])[0] for k in range(len(members))]
pos_all = np.logical_and.reduce([Rr.cand & (v > 0) for v in vals])
top = np.sum([Rr.cand & (PC.ranks_of(v, Rr.cand) <= 3) for v in vals], axis=0)
need = math.ceil(0.8 * len(vals) - 1e-9)
d1 = z["d1_shown"]
s4 = d1 & pos_all & (top >= need)
check(
    np.array_equal(z["s4_shown"], s4)
    and not np.any(s4 & ~d1)
    and np.array_equal(d1, np.load(CHAIN / "p13c_arrays.npz")["shown"]),
    f"S4 from the stored member tables, inside D1 ({s4.sum()} of "
    f"{d1.sum()} shown reasons stay)",
)

M2, ranks2, tag2 = PC.load_p13d_model(CHAIN, BASE)
check(
    M2["fingerprint"] == doc["model"]["fingerprint"]
    and np.array_equal(ranks2, z["ranks"][ix[doc["model"]["fit_name"]]])
    and tag2 == D.tag_of(doc["model"]["fit_name"]),
    "p13c.py's loader accepts the model record",
)
rep = (CHAIN / "reports" / D.REPORT_NAME).read_text()
check(
    all(
        h in rep
        for h in (
            "## Rules",
            "## Scores",
            "## S4",
            "## Bounded shapes",
            "## Concurvity",
            "## Agreement",
        )
    )
    and ";" not in rep,
    "the report has its sections and no semicolon",
)


def table_rows_ok(text):
    """Every row of a Markdown table has as many cells as its header."""
    width = None
    for line in text.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        n = line.count("|")
        if width is None:
            width = n
        elif n != width:
            return False
    return True


def numbers_as_text(x):
    """Values that json holds as text but that read as numbers."""
    if isinstance(x, dict):
        return sum(numbers_as_text(v) for v in x.values())
    if isinstance(x, list):
        return sum(numbers_as_text(v) for v in x)
    if isinstance(x, str):
        try:
            float(x)
            return 1
        except ValueError:
            return int(x in ("True", "False"))
    return 0


check(
    table_rows_ok(rep) and numbers_as_text(doc) == 0,
    "the report's tables keep their columns, and p13d.json holds no number " "as text",
)
conc = doc["concurvity"]
check(
    set(conc["terms"]) == set(M["kept"])
    and all(0.0 <= r["worst"] <= 1.0 for r in conc["terms"].values())
    and set(conc["observed"]) == {"free model", "bounded p_pop-B at lambda"}
    and conc["rows"] > 0,
    f"concurvity for every term on {conc['rows']:,} rows of the loss",
)
ag = doc["agreement"]
check(
    len(ag["reference"]) == 4
    and all(0.0 <= v <= 1.0 for v in ag["reference"].values()),
    "agreement with the four fits at the largest reference level",
)

lines.clear()
again = D.run(stage(), CHAIN, WORK, say=keep, n_verify=6)
check(
    any("every fit at level" in m for m in lines)
    and all(again[k] == out[k] for k in ("s1", "s2", "s3")),
    "a resumed run loads no level and gives the same rules",
)

# 4: forced bounded model
F = TMP / "forced"
shutil.copytree(TMP / "chain", F)
C.configure(F)
FM = F / "model"
real = D.decide


def forced(scores, names, model_arm):
    res = real(scores, names, model_arm)
    res["s2"] = dict(res["s2"], chosen=D.PILOT)
    return res


D.decide = forced
try:
    D.run(stage(), FM, F / "scratch" / "p13d", say=quiet, n_verify=6)
finally:
    D.decide = real
docf = json.loads((FM / D.OUT_FILES[0]).read_text())
tag = docf["model"]["tag"]
check(
    docf["model"]["fit_name"] == D.PILOT
    and docf["model"]["changed"]
    and docf["model"]["bounded"] == list(D.CONTENT)
    and tag == "bounded_p_pop-B_lambda",
    "the forced run records the bounded model and its tag",
)
outc = PC.run(
    stage(),
    FM,
    F / "scratch" / f"p13c_{tag}",
    say=quiet,
    n_verify=6,
    n_boot={"listener": 4, "twoway": 2},
    model="p13d",
)
files, report = PC.out_names(tag)
check(
    all((FM / f).exists() for f in files)
    and (FM / "reports" / report).exists()
    and files
    == (
        "p13c_bounded_p_pop-B_lambda_model.json",
        "p13c_bounded_p_pop-B_lambda_arrays.npz",
    ),
    "the model switch writes tagged outputs",
)
check(
    {f: file_sha256(FM / f) for f in PC.OUT_FILES} == P13C_SHA,
    "the first p13c run's outputs stay unchanged",
)
zc = np.load(FM / files[1])
thetas = np.concatenate(
    [
        zc["theta_boot_listener"],
        zc["theta_boot_twoway"],
        zc["theta_family"],
        zc["theta_ref"],
    ]
)
worst = max(order_break(lay_b.tables(th), lay_b) for th in thetas)
check(
    outc["check"].get("ranks_equal_p13d") is True
    and outc["model"]["fingerprint"] == docf["model"]["fingerprint"]
    and outc["model_source"]["fit"] == D.PILOT
    and worst <= s13b.MONOTONE_TOL,
    f"the switched run keeps the bounds in all {len(thetas)} fits (largest "
    f"break {worst:.1e}), and its ranks equal P13d's",
)
repc = (FM / "reports" / report).read_text()
check(
    "model source: p13d.json" in repc and "equal P13d's" in repc,
    "the switched report names its source",
)

print(f"\n{len(failures)} failures")
for f in failures:
    print("  FAIL", f)
if os.environ.get("KEEP_TMP"):
    print(f"      outputs kept in {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)

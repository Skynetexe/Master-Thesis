"""Smoke test of pipeline/figures.py, on the outputs of the smoke test
test_demo_and_cold_curve.py (on the P13 chain of p13_chain.py). Every
path points into a temporary folder, and nothing touches the real model
folder.

1. test_demo_and_cold_curve.py runs in a child process with KEEP_TMP=1.
   Its kept folder holds the chain with its markers, P14, the cold-start
   curve and the demo page. With DEMO_CURVE_TMP=<folder>, a copy of a
   kept folder is used instead.
2. The marker guard: when the cold-start outputs differ from their marker
   (the demo test runs the module again after the runner, for its resume
   check), figures.run() must refuse them. Then the marker goes, and the
   runner adopts the outputs (freshness rule and read-back).
3. figures.run() on the chain: the checks pass, and each output exists and
   is in the manifest with its SHA-256.
4. Checks written here: the data files against the sources (the curve at
   the knots equals P13c's stored heights, the bands at the shape quantiles
   and the states equal P13c's C1 bands, the free twin equals the model
   set, the cold-start values equal its JSON file, the explanation rows
   equal the demo's shown reasons, the pipeline rows equal the runner's
   stages with a second graph walk), every table of tables.html against its
   report table after a label map written here, and the example rule
   against a second formulation.
5. Resume and determinism: a second call keeps every group. A Ctrl+C inside
   the third group keeps the first two, and the next call draws only the
   rest. Every file then equals the first run's, byte for byte.
6. The symmetric log axis: with TAIL_RATIO 5 the long-tailed terms of the
   chain get it, and the band checks pass on its grid.
7. Background means from the level's rows (P13c's saved work moved away)
   give the same bands.
8. Refusals, each with its message: background means of another fit, a
   changed C1 band, a changed model-set table, a demo of another model, a
   report that does not follow from its JSON file, a p14.json and report
   that differ from the ranks, a cold-start JSON and report that differ from
   the ranks, PHASES without a stage.
9. main(): --demo-user picks the folder, the only folder is the default,
   and an unknown user stops with exit code 1.

Usage, from the repository root:
    python3 test_figures.py
With DEMO_CURVE_TMP=<folder> (a folder that test_demo_and_cold_curve.py
kept with KEEP_TMP=1), step 1 copies it. P13B2_BASE_CACHE and
P13_CHAIN_CACHE pass to test_demo_and_cold_curve.py. With KEEP_TMP=1 the
temporary folder stays.
"""

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T0 = time.time()
failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def clock():
    return f"{time.time() - T0:.0f} s"


def raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        if text in str(exc):
            return True
        print(f"      other message: {exc}")
        return False
    return False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


TMP = Path(tempfile.mkdtemp(prefix="test_figures_"))

# 1: the demo test's outputs
if os.environ.get("DEMO_CURVE_TMP"):
    B2 = TMP / "demo_curve"
    shutil.copytree(os.environ["DEMO_CURVE_TMP"], B2)
    check(
        (B2 / "chain" / "model" / "p14.json").exists(),
        f"1: a copy of the kept folder {os.environ['DEMO_CURVE_TMP']}",
    )
else:
    print("      running test_demo_and_cold_curve.py (about 3 minutes)", flush=True)
    env = dict(os.environ, KEEP_TMP="1", TMPDIR=str(TMP))
    r = subprocess.run(
        [sys.executable, "test_demo_and_cold_curve.py"],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
    )
    kept = sorted(TMP.glob("test_demo_and_cold_curve_*"))
    check(
        r.returncode == 0 and len(kept) == 1,
        f"1: test_demo_and_cold_curve.py passes and keeps its folder " f"({clock()})",
    )
    if r.returncode != 0 or len(kept) != 1:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(1)
    B2 = kept[0]

import p13_chain as C
from pipeline import cold_curve as CC
from pipeline import common
from pipeline import config
from pipeline import figures as F
from pipeline import lb_demo as LB
from pipeline import p13c as PC
from pipeline import p14 as P
from pipeline import runner as RN
from pipeline import stage_p13b2 as s13b2

ROOT = B2 / "chain"
CHAIN = ROOT / "model"
SCR = ROOT / "scratch"
DEMO = B2 / "demo" / "lb_demo.json"
C.configure(ROOT)
common.MODEL_DIR = CHAIN
LINES = []
OUT = TMP / "figures"


def say(msg):
    LINES.append(msg)


def stage():
    return C.p13_loaders(s13b2.StageP13b2())


def run(out=OUT, force=False, demo=DEMO):
    return F.run(stage(), CHAIN, SCR, out, demo, say=say, force=force)


def marker_set(name, key, path):
    """Record path's SHA-256 under key in the marker of stage name, as a run
    of the stage would."""
    p = common.marker_path(name)
    doc = json.loads(p.read_text())
    doc["outputs"][key]["sha256"] = sha(path)
    p.write_text(json.dumps(doc))


class Saved:
    """The bytes of some files, put back on exit."""

    def __init__(self, *paths):
        self.paths = [Path(p) for p in paths]

    def __enter__(self):
        self.data = {p: p.read_bytes() for p in self.paths}
        return self

    def __exit__(self, *exc):
        for p, b in self.data.items():
            p.write_bytes(b)
        return False


# 2: the marker guard
mk = common.marker_path("cold_start_curve")
mdoc = json.loads(mk.read_text())
rep = CHAIN / "reports" / CC.REPORT_NAME
natural = mdoc["outputs"]["report"]["sha256"] != sha(rep)
if not natural:
    mdoc["outputs"]["report"]["sha256"] = "0" * 64
    mk.write_text(json.dumps(mdoc))
refused = raises(
    run, "is not the output that the marker of " "cold_start_curve recorded"
)
st_c = CC.StageColdCurve(CHAIN, SCR)
mk.unlink()
by = dict(RN.STAGE_BY_NAME)
by[st_c.name] = st_c
rc = RN.execute([st_c], dict(config.DEFAULT_PARAMS), [], by_name=by, adopt=True)
mdoc = json.loads(mk.read_text())
check(
    refused
    and rc == 0
    and mdoc["outputs"]["report"]["sha256"] == sha(rep)
    and mdoc["outputs"]["results"]["sha256"] == sha(CHAIN / CC.OUT_FILES[0]),
    f"2: outputs that differ from their marker "
    f"({'from the demo test' if natural else 'a changed marker'}) "
    "stop the run, "
    f"and the runner adopts them after the marker goes",
)

# 3: the full run
LINES.clear()
man = run()
stems = ("fig_pipeline", "fig_shapes", "fig_cold_start", "fig_explanations")
want = {f"{s}.{e}" for s in stems for e in ("png", "pdf", "svg", "csv")}
want |= {"tables.html", "tables.md"}
have = {n for g in man["groups"].values() for n in g["outputs"]}
tabs = sorted(n for n in have if n.startswith("tab_"))
check(
    sorted(man["groups"])
    == sorted(["pipeline", "shapes", "cold_start", "explanations", "tables"])
    and want <= have
    and len(tabs) >= 3
    and all(
        sha(OUT / n) == s
        for g in man["groups"].values()
        for n, s in g["outputs"].items()
    )
    and (OUT / "captions.md").exists()
    and man["captions"] == sha(OUT / "captions.md"),
    f"3: the run writes {len(have)} files in 5 groups, each in the "
    f"manifest with its SHA-256 ({clock()})",
)
check(
    any("equal their markers" in ln for ln in LINES)
    and any("give P13c's stored bands exactly" in ln for ln in LINES)
    and any("equal P13c's C1 bands at every shape quantile" in ln for ln in LINES)
    and any("equals its JSON file" in ln for ln in LINES),
    "3: the log names each check of the sources",
)


# 4: checks written here


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


ptr = json.loads((CHAIN / PC.FINAL_FILE).read_text())
res = json.loads((CHAIN / ptr["files"]["results"]).read_text())
td = res["model"]["table_dict"]
bands = res["bootstrap"]["bands"]
base = stage()._base_context()
rows = read_csv(OUT / "fig_shapes.csv")
ok_k = ok_q = ok_s = True
n_q = n_s = 0
for name, tab in td.items():
    mine = [r for r in rows if r["term"] == name]
    curve = {float(r["x"]): r for r in mine if r["kind"] == "curve"}
    for x, h in zip(tab["knots"], tab["heights"]):
        ok_k &= x in curve and float(curve[x]["value"]) == h
    kn = tab["knots"]
    brow = {b["point"]: b for b in bands[name]}
    for p, x in enumerate((base["shape_q"].get(name) or [])[:5]):
        xc = min(max(float(x), kn[0]), kn[-1])
        b = brow[PC.POINTS[p]]
        r = curve.get(xc)
        ok_q &= r is not None and [
            float(r["listener_lo"]),
            float(r["listener_hi"]),
            float(r["twoway_lo"]),
            float(r["twoway_hi"]),
        ] == [
            b["listener"]["lo"],
            b["listener"]["hi"],
            b["twoway"]["lo"],
            b["twoway"]["hi"],
        ]
        n_q += 1
    for r in mine:
        if r["kind"] in ("zero", "missing"):
            b = brow[r["kind"]]
            ok_s &= float(r["value"]) == tab[r["kind"]] and [
                float(r["listener_lo"]),
                float(r["listener_hi"]),
            ] == [b["listener"]["lo"], b["listener"]["hi"]]
            n_s += 1
check(
    ok_k and ok_q and ok_s and n_q > 0 and n_s > 0,
    f"4: fig_shapes.csv: the curve at every knot equals P13c's stored "
    f"heights, and the bands at {n_q} shape quantiles and {n_s} states "
    f"equal P13c's C1 bands exactly",
)
zs = np.load(CHAIN / PC.P13D_FILES[1])
names = [str(x) for x in zs["names"].tolist()]
twin = "free|" + ptr["model"]["fit_name"].split("|", 1)[1]
T_free = zs["tables"][names.index(twin)]
off, ok_f, n_f = 0, True, 0
for t in res["model"]["terms"]:
    K = len(t["knots"])
    mine = {
        float(r["x"]): r
        for r in rows
        if r["term"] == t["name"] and r["kind"] == "curve"
    }
    if t["name"] in ptr["model"]["bounded"]:
        for k, x in enumerate(t["knots"]):
            ok_f &= float(mine[x]["free_fit"]) == float(T_free[off + k])
            n_f += 1
    else:
        ok_f &= all(r["free_fit"] == "" for r in mine.values())
    off += K + 4
check(
    ok_f and n_f > 0,
    f"4: fig_shapes.csv: the free twin {twin} at {n_f} knots equals "
    f"p13d_model_set.npz, and the free terms have none",
)

cdoc = json.loads((CHAIN / CC.OUT_FILES[0]).read_text())
crow = read_csv(OUT / "fig_cold_start.csv")
pts = cdoc["results"]["points"]
gap = 0.0
for r in crow:
    k = pts.index(r["N"] if r["N"] == "full" else int(r["N"]))
    s = cdoc["results"]["arms"][r["arm"]][k][r["metric"]]
    gap = max(
        gap,
        abs(float(r["mean"]) - s["mean"]),
        abs(float(r["lo"]) - s["lo"]),
        abs(float(r["hi"]) - s["hi"]),
    )
check(
    len(crow) == len(CC.ARMS) * len(pts) * 2 and gap <= 1e-12,
    f"4: fig_cold_start.csv equals cold_start_curve.json ({len(crow)} "
    f"values, largest gap {gap:.1e})",
)

ddoc = json.loads(DEMO.read_text())


def first_by_artist(gam_list, k=3):
    """The example rule as a second formulation: the first track of each
    artist in rank order, the first k of them."""
    first = {}
    for r in gam_list:
        key = next(
            (
                ("credit", x["credit"])
                for x in r["reasons"]
                if x["reason"] == "artist" and x.get("credit") is not None
            ),
            ("name", r.get("artist")),
        )
        first[key] = min(first.get(key, 10**9), int(r["rank"]))
    return sorted(first.values())[:k]


ex_ranks = first_by_artist(ddoc["gam"])
erows = read_csv(OUT / "fig_explanations.csv")
want_e = [
    (str(r["rank"]), str(r["item"]), x["reason"], repr(x["value"]), repr(x["odds"]))
    for r in sorted(ddoc["gam"], key=lambda r: r["rank"])
    if r["rank"] in ex_ranks
    for x in r["reasons"]
]
check(
    [(r["rank"], r["item"], r["reason"], r["value"], r["odds"]) for r in erows]
    == want_e
    and len(want_e) > 0,
    f"4: fig_explanations.csv holds the shown reasons of ranks {ex_ranks}, "
    f"value and odds exact",
)
mod = json.loads(json.dumps(ddoc["gam"]))
for x in mod[1]["reasons"]:
    if x["reason"] == "artist":
        x["credit"] = next(
            y["credit"] for y in mod[0]["reasons"] if y["reason"] == "artist"
        )
got = [int(r["rank"]) for r in F.pick_examples(mod)]
check(
    got == first_by_artist(mod) and got[0] == 1 and 2 not in got,
    f"4: the example rule skips a track whose artist an earlier example "
    f"has ({got}), as the second formulation does",
)

prow = read_csv(OUT / "fig_pipeline.csv")
up = {s.name: list(s.upstream) for s in RN.STAGES}
down = {n: [] for n in up}
for n, ups in up.items():
    for u in ups:
        down[u].append(n)
evals = set(F.PHASES[-1][1])


def reaches(n):
    seen, stack = set(), [n]
    while stack:
        x = stack.pop()
        if x in evals:
            return True
        for y in down[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


used_here = {n: reaches(n) for n in up}
stage_rows = {r["name"]: r for r in prow if r["name"] in up}
check(
    sorted(stage_rows) == sorted(up)
    and all((stage_rows[n]["used_by_final_model"] == "yes") == used_here[n] for n in up)
    and sorted(n for n in up if not used_here[n])
    == sorted(["p7_graph", "p8_fusion_features", "p9_cooccurrence", "p10_profiles"])
    and {r["name"] for r in prow} - set(up) == {f for f, _ in F.SCRIPTS},
    "4: fig_pipeline.csv holds every runner stage once, and a downstream "
    "walk written here marks the same 4 stages as not used",
)


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.cell, self.head = [], None, False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.out.append({"id": dict(attrs).get("id"), "head": [], "rows": []})
        elif tag == "thead":
            self.head = True
        elif tag == "tbody":
            self.head = False
        elif tag == "tr" and not self.head and self.out:
            self.out[-1]["rows"].append([])
        elif tag in ("th", "td"):
            self.cell = ""

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            (self.out[-1]["head"] if self.head else self.out[-1]["rows"][-1]).append(
                self.cell
            )
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data


def md_tables(text):
    out, lines = [], text.splitlines()
    for i, s in enumerate(lines):
        if set(s.replace("|", "").replace(":", "").strip()) == {"-"}:
            head = [c.strip() for c in lines[i - 1].strip()[1:-1].split("|")]
            body = []
            for t in lines[i + 1 :]:
                if not t.startswith("|"):
                    break
                body.append([c.strip() for c in t.strip()[1:-1].split("|")])
            out.append((head, body))
    return out


LAB = {v: v[0].upper() + v[1:] for v in P.ARM_LABELS.values()}
LAB.update({v: v[0].upper() + v[1:] for v in CC.ARM_LABELS.values()})
LAB.update(
    {
        "ndcg_at_100": "NDCG@100",
        "ndcg_at_10": "NDCG@10",
        "recall_at_10": "R@10",
        "recall_at_100": "R@100",
        "mrr": "MRR",
    }
)
LAB.update(LB.REASON_LABELS)


def head_label(h):
    w = h.replace(",", " ").split(" ")[0]
    return h if len(w) <= 1 or "_" in w or "@" in w else h[0].upper() + h[1:]


tp = Tables()
tp.feed((OUT / "tables.html").read_text(encoding="utf-8"))
reports = md_tables((CHAIN / "reports" / P.REPORT_NAME).read_text()) + md_tables(
    (CHAIN / "reports" / CC.REPORT_NAME).read_text()
)
same = len(tp.out) == len(reports) + 1
for t, (head, body) in zip(tp.out, reports):
    same &= t["head"] == [head_label(h) for h in head] and t["rows"] == [
        [LAB.get(c, c) for c in r] for r in body
    ]
ex_t = tp.out[-1]
want_t = []
for r in sorted(ddoc["gam"], key=lambda r: r["rank"]):
    if r["rank"] in ex_ranks:
        for x in r["reasons"]:
            want_t.append(
                [
                    str(r["rank"]),
                    f"“{r['title']}” by {r['artist']}",
                    LB.REASON_LABELS.get(x["reason"], x["reason"]),
                    f"{x['value']:+.3f}",
                    LB._odds(x["odds"]),
                    F.evidence_text(x),
                ]
            )
csv_same = all(
    list(csv.reader(open(OUT / f"tab_{t['id']}.csv", newline="", encoding="utf-8")))
    == [t["head"]] + t["rows"]
    for t in tp.out
)
check(
    same and ex_t["rows"] == want_t and csv_same,
    f"4: tables.html holds every table of the two reports cell for cell "
    f"after the label map ({len(reports)} tables), the example table, and "
    f"each CSV equals its table",
)
check(
    "t" in tp.out[3]["head"]
    and "p" in tp.out[3]["head"]
    and "test_last NDCG@100" in tp.out[1]["head"]
    and tp.out[0]["head"][0] == "Arm",
    "4: column heads get a capital letter, but not t, p or a key",
)

# 5: resume and determinism
first = {n: sha(OUT / n) for g in man["groups"].values() for n in g["outputs"]}
LINES.clear()
run()
check(
    sum("complete from an earlier run, kept" in ln for ln in LINES) == 5
    and all(sha(OUT / n) == s for n, s in first.items()),
    "5: a second call keeps all 5 groups, and no file changes",
)
OUT3 = TMP / "figures_resume"
real_draw = F.draw_cold


def stopping(*a, **k):
    raise KeyboardInterrupt


F.draw_cold = stopping
try:
    run(out=OUT3)
    stopped = False
except KeyboardInterrupt:
    stopped = True
finally:
    F.draw_cold = real_draw
part = json.loads((OUT3 / "manifest.json").read_text())
LINES.clear()
run(out=OUT3)
kept_l = [ln for ln in LINES if "complete from an earlier run" in ln]
check(
    stopped
    and sorted(part["groups"]) == ["pipeline", "shapes"]
    and len(kept_l) == 2
    and all(("pipeline" in ln or "shapes" in ln) for ln in kept_l)
    and all((OUT3 / n).read_bytes() == (OUT / n).read_bytes() for n in first),
    "5: Ctrl+C in the third group keeps the first two, the next call "
    "draws the rest, and every file equals the first run's byte for byte",
)
LINES.clear()
run(force=True)
check(
    all(sha(OUT / n) == s for n, s in first.items())
    and not any("kept" in ln for ln in LINES),
    "5: force draws every group again, with the same bytes",
)

# 6: the symmetric log axis
real_ratio = F.TAIL_RATIO
F.TAIL_RATIO = 5.0
try:
    logs = [
        t["name"] for t in res["model"]["terms"] if F.x_scale(t["knots"])[0] == "symlog"
    ]
    LINES.clear()
    ok6 = True
    try:
        run(out=TMP / "figures_symlog", force=True)
    except ValueError as exc:
        ok6 = False
        print(f"      {exc}")
finally:
    F.TAIL_RATIO = real_ratio
check(
    ok6 and len(logs) >= 2,
    f"6: with TAIL_RATIO 5 the terms {logs} get the symmetric log axis, "
    f"and the bands on its grid equal the stored C1 bands",
)
t1 = F.symlog_ticks(0.0, 0.1167, 0.00022)
t2 = F.symlog_ticks(-34.1, 0.0, 0.067)
check(
    t1 == [0.0, 0.001, 0.01, 0.1]
    and t2 == [-10.0, -1.0, -0.1, 0.0]
    and F.x_scale([0.0, 0.1, 0.2, 0.3])[0] == "linear"
    and F.x_scale([0.0, 0.0, 0.0, 0.5])[0] == "symlog",
    f"6: the log axis ticks are 0 and the powers of ten outside the "
    f"linear part ({t1}, {t2}), and a median knot of 0 gives the log axis",
)

# 7: means from the level's rows
fits = SCR / f"{PC.WORK_NAME}_{ptr['model']['tag']}" / "fits.json"
away = fits.with_name("fits.json.away")
os.replace(fits, away)
try:
    LINES.clear()
    run(out=TMP / "figures_means", force=True)
    ok7 = (TMP / "figures_means" / "fig_shapes.csv").read_bytes() == (
        OUT / "fig_shapes.csv"
    ).read_bytes()
except ValueError as exc:
    ok7 = False
    print(f"      {exc}")
finally:
    os.replace(away, fits)
check(
    ok7 and any("come from the level's rows" in ln for ln in LINES),
    "7: without P13c's saved means, the means from the level's rows give "
    "the same bands, byte for byte",
)

# 8: refusals
NEG = TMP / "figures_neg"
lvl = f"wide:{res['model']['level']}"
with Saved(fits):
    doc = json.loads(fits.read_text())
    name0 = res["model"]["terms"][0]["name"]
    doc["means"][lvl][name0] = [1.001 * v for v in doc["means"][lvl][name0]]
    fits.write_text(json.dumps(doc))
    check(
        raises(lambda: run(out=NEG, force=True), "centered with the background means"),
        "8: background means of another fit stop the run",
    )
rpath = CHAIN / ptr["files"]["results"]
fpath = CHAIN / PC.FINAL_FILE
with Saved(rpath, fpath, common.marker_path("p13c_final")):
    doc = json.loads(rpath.read_text())
    doc["bootstrap"]["bands"][name0][0]["listener"]["lo"] += 1e-9
    rpath.write_text(json.dumps(doc))
    pdoc = json.loads(fpath.read_text())
    pdoc["sha256"]["results"] = sha(rpath)
    fpath.write_text(json.dumps(pdoc))
    marker_set("p13c_final", "results", rpath)
    marker_set("p13c_final", "pointer", fpath)
    check(
        raises(lambda: run(out=NEG, force=True), "do not give the stored C1 bands"),
        "8: a changed C1 band (pointer and marker made to fit) stops the " "run",
    )
spath = CHAIN / PC.P13D_FILES[1]
with Saved(spath, common.marker_path("p13d_bounds")):
    z = dict(np.load(spath))
    z["tables"] = z["tables"].copy()
    z["tables"][names.index(ptr["model"]["fit_name"]), 0] += 1e-6
    with open(spath, "wb") as f:
        np.savez(f, **z)
    marker_set("p13d_bounds", "model_set", spath)
    check(
        raises(lambda: run(out=NEG, force=True), "other tables for the final fit"),
        "8: a model set whose final tables differ stops the run",
    )
bad = TMP / "bad_demo.json"
dd = json.loads(DEMO.read_text())
dd["model"]["final"] = "free|other"
bad.write_text(json.dumps(dd))
check(
    raises(lambda: run(out=NEG, force=True, demo=bad), "another model"),
    "8: a demo page of another model stops the run",
)
rp14 = CHAIN / "reports" / P.REPORT_NAME
j14 = CHAIN / P.OUT_FILES[0]
with Saved(rp14, j14, common.marker_path("p14_eval")):
    rp14.write_text(rp14.read_text() + "\nA line that the writer never " "wrote.\n")
    marker_set("p14_eval", "report", rp14)
    check(
        raises(lambda: run(out=NEG, force=True), "does not follow from p14.json"),
        "8: a P14 report that does not follow from p14.json stops the run",
    )
    doc = json.loads(j14.read_text())
    doc["arms"]["gam"]["ndcg_at_100"] += 1e-6
    j14.write_text(json.dumps(doc))
    P.write_report(doc, rp14)
    marker_set("p14_eval", "results", j14)
    marker_set("p14_eval", "report", rp14)
    check(
        raises(lambda: run(out=NEG, force=True), "differ from p14.json"),
        "8: a p14.json and report that differ from the ranks of p14.npz "
        "stop the run",
    )
jc = CHAIN / CC.OUT_FILES[0]
with Saved(rep, jc, common.marker_path("cold_start_curve")):
    doc = json.loads(jc.read_text())
    doc["results"]["arms"]["gam"][0]["ndcg_at_100"]["mean"] += 1e-6
    jc.write_text(json.dumps(doc))
    CC.write_report(doc, rep)
    marker_set("cold_start_curve", "results", jc)
    marker_set("cold_start_curve", "report", rep)
    check(
        raises(lambda: run(out=NEG, force=True), "differs from cold_start_curve.json"),
        "8: a cold-start JSON and report that differ from the ranks stop " "the run",
    )
real_phases = F.PHASES
F.PHASES = tuple((t, tuple(m for m in ms if m != "p7_graph")) for t, ms in real_phases)
try:
    check(
        raises(lambda: run(out=NEG, force=True), "does not hold every runner stage"),
        "8: PHASES without a runner stage stops the run",
    )
finally:
    F.PHASES = real_phases
LINES.clear()
run()
check(
    all(sha(OUT / n) == s for n, s in first.items()),
    "8: after the refusals, with every file back, the outputs are " "unchanged",
)

# 9: main()
dd_dir = CHAIN / "demo" / "tester"
dd_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(DEMO, dd_dir / "lb_demo.json")
real_m, real_s = F.MODEL_DIR, F.SCRATCH_DIR
F.MODEL_DIR, F.SCRATCH_DIR = CHAIN, SCR
root = logging.getLogger()
before = list(root.handlers)


def main_quiet(argv):
    """F.main() with its console lines in a buffer, and its handler gone
    after the call."""
    try:
        return F.main(argv)
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)


saved_err = sys.stderr
try:
    sys.stderr = open(TMP / "main_console.txt", "w")
    rc1 = main_quiet(["--demo-user", "tester"])
    rc2 = main_quiet([])
    rc3 = main_quiet(["--demo-user", "nobody"])
finally:
    sys.stderr.close()
    sys.stderr = saved_err
    F.MODEL_DIR, F.SCRATCH_DIR = real_m, real_s
console = (TMP / "main_console.txt").read_text()
check(
    "fontTools" not in console
    and "pruned" not in console
    and console.count("figures: done in") == 2
    and "stopped: demo/nobody/lb_demo.json is missing" in console,
    "9: the console shows the script's lines once each, and no line of "
    "the PDF backend's font subsetting",
)
check(
    rc1 == 0
    and rc2 == 0
    and rc3 == 1
    and (CHAIN / "figures" / "fig_shapes.png").read_bytes()
    == (OUT / "fig_shapes.png").read_bytes()
    and (CHAIN / "logs" / F.LOG_NAME).exists(),
    "9: main() with --demo-user and with the only folder exits 0 with the "
    "same figures and its log file, and an unknown user exits 1",
)

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)

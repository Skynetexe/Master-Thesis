"""The figures and tables of the thesis.
They come from the stored outputs of the frozen model, P14, the cold-start
curve and the ListenBrainz demo. This is a plain script, because no later
step reads its outputs.

RULES, written before any run on real data. The script draws
and tables stored results. It fits nothing, selects nothing and tests
nothing. No figure or table changes a number of its source.

Sources. The script checks each one before it uses it (Checks).
- the final model: p14.load_models() (p13c_final.json, the model record of
  p13d.json, the P13c exports that the pointer names, and the 100 listener
  and 50 twoway replicates), and the P13c results file of that model.
- the background means of the model's level of the wide file, from P13c's
  saved work (SCRATCH_DIR/p13c_<tag>/fits.json) through
  p13c.stored_means().
- p13d_model_set.npz: the centered tables of every fit of P13d.
- p14.json, p14.npz and reports/p14_report.md (stage p14_eval).
- cold_start_curve.json, cold_start_curve.npz and
  reports/cold_start_curve.md (stage cold_start_curve).
- MODEL_DIR/demo/<user>/lb_demo.json (pipeline/lb_demo.py).
- the stages of pipeline/runner.py and their upstream lists.

Figure fig_shapes, the shapes with bootstrap bands. One panel for each term
of the final model, in the model's term order.
- curve: the stored centered knot heights of the final model, linear
  between the knots and flat beyond the end knots, as the model scores.
- band: at each point of the panel's grid, the 2.5 and 97.5 percentiles
  (p13c.BAND) of the term's value over the 100 listener replicates. Each
  replicate's tables are centered with P13c's frozen rule
  (p13c.centered(): the level's p_pop negatives). The grid holds every
  knot, the picks' five shape quantiles, and GRID_POINTS points with equal
  spacing on the panel's x scale. The twoway band goes into the data file
  only.
- free twin: when the final model is a bounded fit of P13d, each bounded
  term also shows the same term of P13d's free fit with the same arm and
  penalty ("free|<arm>|<lambda>" in p13d_model_set.npz) as a thin gray
  line. So the panel shows what the bound changes.
- states: the zero state ("no match": data, and the value is exactly 0)
  and the missing state ("no data") are points beside the curve, with
  their bands, for the terms that have them.
- picks: below the curve, the median, the 25th to 75th and the 5th to
  95th percentiles of the term's value over the selection picks (P13a's
  shape quantiles).
- x scale: linear, but symmetric log when the largest absolute knot is
  more than TAIL_RATIO times the median absolute knot (a long tail). The
  linear part of that scale ends at LINEAR_SHARE times the median absolute
  knot, or at the smallest nonzero absolute knot when the median is 0. Its
  ticks are 0 and the powers of ten outside the linear part.
- y scale: one for each panel, because the terms differ by an order of
  magnitude. The panel title gives the term's importance (P13c: the mean
  absolute contribution against the picks' p_pop draws). This is the
  number to compare across panels.

Figure fig_cold_start, the cold-start curve. Two panels, NDCG@100 and
R@100, against the history length N for the four arms of
cold_start_curve.json, with the stored 95 percent intervals as error bars.
The x axis is logarithmic. The full-history point stands apart at the
right, and no line joins it to the curve. Popularity ignores the history,
so it is a flat gray line with its interval as a gray band. The y axis is
logarithmic, because the values span more than a decade.

Figure fig_explanations and table tab_explanations, the example
explanations. EXAMPLES = 3 tracks of the demo's GAM top 25: the track at
rank 1, then the next tracks by rank whose artist differs from the artist
of every earlier example. The artist is the credit of the artist reason,
or the track's artist name when the track shows no artist reason. For each
track, one bar for each shown reason (the demo keeps only the reasons that
D1 and S4 show) with its contribution in log-odds and its odds factor
exp(value). The panels share one x axis, so the size of the artist reason
against the content reasons stays visible. The candidate reasons that the
rules do not show are named under the bars. The table gives the evidence
of each reason in the words of the demo page. Genre evidence shows only
the count of matched genres, because the genres come from the Spotify
copy, and that copy is for local research use only.

Figure fig_pipeline, the pipeline. The stages of pipeline/runner.py in
PHASES, after one box for the data and the ETL scripts and before one box
for the plain scripts. A stage is gray when no stage of the evaluation
phase depends on it, directly or through other stages (not used by the
final model).

Tables (tables.html, tables.md and one CSV file for each table): every
table of reports/p14_report.md and reports/cold_start_curve.md, cell for
cell, with the thesis labels: the arm labels with a capital first letter,
the metric keys as NDCG@100, NDCG@10, R@10, R@100 and MRR, the reason
names as on the demo page (lb_demo.REASON_LABELS), and a capital first
letter in each column head, but not in a statistic symbol (t, p) or a key
(test_last). The caption of a table is its report heading, plus the line
that ends in a colon just before the table. When two tables of a report
get the same caption, the later ones get (2), (3). The other lines of its
section, and the notes of the report's opening, are its notes, under the
table. Then the table of the example explanations. tables.html opens in
Word or LibreOffice Writer, and its tables copy into the thesis document.

Checks, each one stops the run:
- markers: the outputs of the stages p13d_bounds, p13c_final, p14_eval and
  cold_start_curve equal the SHA-256 that their markers recorded. Each
  stage's read-back passed before its marker was written.
- the final model: p14.load_models() checks the pointer, the model record
  and the replicates. The demo names the same model, with the same numbers
  of replicates and S3 members.
- the means: the final model's tables, centered with them, equal the
  stored tables within s13b2.CENTER_TOL, and p13c.band_rows() over the
  replicates gives the stored bands of P13c's C1 exactly.
- the bands: at the picks' shape quantiles and at the states, the panel's
  band equals the stored C1 band exactly, and the curve equals the stored
  C1 value within s13b2.CENTER_TOL.
- the free twin: the model set's tables of the final fit equal the final
  model's stored tables exactly.
- P14: NDCG@100, NDCG@10, R@10, R@100, MRR and the median ranks of every
  arm, computed here from the ranks of p14.npz, equal p14.json within
  CHECK_ATOL, and so do the means of the confirmatory differences. The
  report follows from p14.json (p14.write_report() gives the same text),
  and the cells of the accuracy table equal the values computed here, in
  the report's formats.
- the cold-start curve: each mean and interval, computed here from the
  ranks of cold_start_curve.npz, equals cold_start_curve.json within
  CHECK_ATOL. The report follows from the JSON file, and its cells equal
  the values computed here, in the report's formats.
- the demo: each example's shown reasons have positive values in
  descending order and odds factors equal to exp(value), each evidence
  count covers its units, and the evidence text equals the text of the
  demo page for every reason but genre.
- the pipeline: PHASES holds every stage of the runner exactly once, no
  upstream list names a stage of a later phase, and each plain script on
  the figure exists in the pipeline package.
- the read-back: each CSV file and the HTML and Markdown tables are read
  again and equal the values that were drawn or tabled, and each figure
  file opens again (the PNG width equals the text width at DPI).

Outputs, in MODEL_DIR/figures (the data folder, never the repository: the
explanations hold the demo listener's tracks):
- fig_shapes, fig_cold_start, fig_explanations, fig_pipeline: .png (DPI
  dots per inch), .pdf and .svg, and .csv with the numbers drawn.
- tables.html, tables.md and tab_<id>.csv.
- captions.md: draft captions with their sources.
- manifest.json: the SHA-256 of every input and output, the checks, the
  versions and the font.
Each figure is TEXT_WIDTH_MM wide (A4 minus the 30 mm margins of the
Capstone Project Module Handbook, section 4.3), so it goes into the
document at 100 percent, and its text prints at 6.5 to 9 pt in the first
sans-serif font of FONTS that the computer has. Lines go to the console
and to MODEL_DIR/logs/figures.log.

Resume: each group of outputs (a figure with its data file, or the tables)
is written atomically and recorded in manifest.json with the SHA-256 of its
inputs and of this script. After Ctrl+C, the same command keeps the groups
that are complete and draws the rest. --force draws every group again.

Expected run on the real data: under two minutes (P13's base context and
the model, 150 replicate centerings, four figures, the tables, the SHA-256
of p14.npz).

Usage, from the repository root:
    python3 -m pipeline.figures --demo-user NAME
"""

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import sys
import tempfile
import textwrap
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mimage
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib import scale as mscale
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from pipeline import cold_curve as cc
from pipeline import gam
from pipeline import lb_demo as lb
from pipeline import p13c as pc
from pipeline import p13d as pd13
from pipeline import p14
from pipeline import runner
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.config import MODEL_DIR, SCRATCH_DIR
from pipeline.script_stage import LOG_FORMAT, log_file

VERSION = "figures-v1"
DECLARED = "before any run on real data"
WORK_NAME = "figures"
OUT_NAME = "figures"
LOG_NAME = "figures.log"
COMMAND = "python3 -m pipeline.figures"
TEXT_WIDTH_MM = 150.0  # A4 (210 mm) minus two 30 mm margins
MM = 1.0 / 25.4  # inches per mm
DPI = 300
FORMATS = ("png", "pdf", "svg")
FONTS = ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans")
GRID_POINTS = 400
TAIL_RATIO = 20.0
LINEAR_SHARE = 0.1
EXAMPLES = 3
CHECK_ATOL = 1e-12
TITLE_CHARS = 70

# Three categorical colours, one per arm, and neutral greys for the text,
# the grid and the axes. Popularity is a reference line in the muted ink.
SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C, FILL_C = "#e1e0d9", "#c3c2b7", "#f4f3ef"
ARM_STYLE = {
    "gam": (SLOT[0], "o", -0.025),
    "knn_both": (SLOT[1], "s", 0.0),
    "cold_gam": (SLOT[2], "^", 0.025),
    "popularity": (MUTED, None, 0.0),
}
META = {
    "png": {"Software": None},
    "pdf": {"Creator": None, "Producer": None, "CreationDate": None},
    "svg": {"Creator": None, "Date": None},
}

TERM_LABELS = {
    "tag_affinity": "Track tags",
    "artist_tag_affinity": "Artist tags",
    "genre_affinity": "Genres",
    "artist_affinity": "Artist share",
    "colisten_same_artist": "Co-listening, same artist",
    "colisten_other_artists": "Co-listening, other artists",
    "work_affinity": "Work share",
    "audio_fit": "Sound",
    "popularity": "Popularity",
}
TERM_X = {
    "tag_affinity": "TF-IDF cosine of the track tags",
    "artist_tag_affinity": "TF-IDF cosine of the artist tags",
    "genre_affinity": "TF-IDF cosine of the genres",
    "artist_affinity": "listening share of the track's artist",
    "colisten_same_artist": "co-listening score, same artist",
    "colisten_other_artists": "co-listening score, other artists",
    "work_affinity": "listening share of the track's work",
    "audio_fit": "audio fit (minus the mean squared z-score)",
    "popularity": "log(1 + train listens)",
}
STATE_LABELS = {"zero": "no match", "missing": "no data"}

# The pipeline figure: the runner's stages in phases, each with a short
# label. The checks hold these against pipeline/runner.py.
PHASES = (
    ("Preparation", ("p1_audit", "p2_split", "p25_train", "p3_sampling")),
    (
        "Features and baselines",
        (
            "p4_features",
            "p5_eval_baseline",
            "p6_als",
            "p7_graph",
            "p8_fusion_features",
            "p9_cooccurrence",
            "p9b_colisten",
            "p10_profiles",
            "als_phase_d",
            "itemknn_select",
        ),
    ),
    (
        "Training pairs and model fits",
        (
            "p11_fusion_pairs",
            "p12_pair_features",
            "p13a_fit",
            "p13b_arms",
            "p11b_more_pairs",
            "p12b_more_features",
            "p13b2_loss",
            "gam_knn_slices",
            "gam_refit",
        ),
    ),
    ("Stability and the frozen model", ("p13c_stability", "p13d_bounds", "p13c_final")),
    (
        "Evaluation",
        (
            "p14_cold_gam",
            "p14_eval",
            "cold_start_curve",
            "beyond_accuracy",
            "evidence_removal",
        ),
    ),
)
STAGE_TEXT = {
    "p1_audit": ("P1", "corpus audit"),
    "p2_split": ("P2", "train, validation, test split"),
    "p25_train": ("P2.5", "training table, statistics"),
    "p3_sampling": ("P3", "sampling tables"),
    "p4_features": ("P4", "item feature matrices"),
    "p5_eval_baseline": ("P5", "evaluation harness, popularity"),
    "p6_als": ("P6", "ALS baseline"),
    "p7_graph": ("P7", "item-item relational graph"),
    "p8_fusion_features": ("P8", "extra fusion features"),
    "p9_cooccurrence": ("P9", "item co-occurrence"),
    "p9b_colisten": ("P9b", "exact co-listening neighbours"),
    "p10_profiles": ("P10", "user profile vectors"),
    "als_phase_d": ("ALS-D", "ALS phase D tuning"),
    "itemknn_select": ("KNN", "ItemKNN baseline"),
    "p11_fusion_pairs": ("P11", "training pairs"),
    "p12_pair_features": ("P12", "concept features of the pairs"),
    "p13a_fit": ("P13a", "audit and base GAM fit"),
    "p13b_arms": ("P13b", "experiment arms"),
    "p11b_more_pairs": ("P11b", "more negatives"),
    "p12b_more_features": ("P12b", "their concept features"),
    "p13b2_loss": ("P13b2", "loss form, negatives"),
    "gam_knn_slices": ("Diag", "GAM against ItemKNN slices"),
    "gam_refit": ("Refit", "GAM refit, 2 × 2 design"),
    "p13c_stability": ("P13c", "bootstrap stability"),
    "p13d_bounds": ("P13d", "bounded content shapes"),
    "p13c_final": ("P13c", "stability of the final model"),
    "p14_cold_gam": ("P14", "cold-track GAM"),
    "p14_eval": ("P14", "evaluation on the eval listeners"),
    "cold_start_curve": ("Curve", "cold-start curve"),
    "beyond_accuracy": ("Beyond", "novelty, coverage, diversity"),
    "evidence_removal": ("Removal", "evidence removal of the reasons"),
}
DATA_BOX = (
    "Data and ETL scripts (outside the runner)",
    "MLHD+ listening histories, the MusicBrainz database and the "
    "Spotify catalogue, converted to Parquet and bridged into one "
    "item space.",
)
SCRIPTS = (
    ("lb_demo.py", "ListenBrainz demo"),
    ("lb_outside.py", "tracks outside MLHD"),
    ("tree_audit.py", "tree-GAM audit"),
    ("figures.py", "thesis figures and tables"),
)
SCRIPTS_TITLE = "Plain scripts on the stored outputs"

METRIC_LABELS = {
    "ndcg_at_100": "NDCG@100",
    "ndcg_at_10": "NDCG@10",
    "recall_at_10": "R@10",
    "recall_at_100": "R@100",
    "mrr": "MRR",
}
ACCURACY_HEAD = [
    "arm",
    "NDCG@100",
    "NDCG@10",
    "R@10",
    "R@100",
    "MRR",
    "median rank, test_last",
    "median rank, test_first",
]


# Helpers


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def _rel(d, p):
    p, d = Path(p), Path(d)
    try:
        return str(p.relative_to(d))
    except ValueError:
        return str(p)


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def style(font=None):
    """The rcParams of every figure. font: the family to use (the first of
    FONTS found when None)."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [font] if font else list(FONTS),
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS_C,
        "axes.linewidth": 0.6,
        "axes.titlepad": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": VERSION,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


def resolved_font():
    """(family name, file) of the sans-serif font the figures use."""
    with matplotlib.rc_context(style()):
        path = font_manager.findfont(
            font_manager.FontProperties(family=["sans-serif"]), fallback_to_default=True
        )
    name = font_manager.FontProperties(fname=path).get_name()
    return name, Path(path).name


def save_figure(fig, out, stem):
    """The figure in FORMATS, each written atomically. Returns {file:
    SHA-256}."""
    shas = {}
    for ext in FORMATS:
        p = Path(out) / f"{stem}.{ext}"
        tmp = p.with_name(p.name + ".part")
        fig.savefig(tmp, format=ext, dpi=DPI, metadata=META[ext])
        os.replace(tmp, p)
        shas[p.name] = file_sha256(p)
    plt.close(fig)
    return shas


def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(
            [
                "" if x is None else (repr(x) if isinstance(x, float) else str(x))
                for x in r
            ]
        )
    return buf.getvalue()


def write_csv(path, header, rows):
    """Write the rows and read them back. Returns the file's SHA-256."""
    text = csv_text(header, rows)
    atomic_text_write(text, path)
    with open(path, newline="", encoding="utf-8") as f:
        back = list(csv.reader(f))
    want = list(csv.reader(io.StringIO(text)))
    if back != want:
        raise ValueError(f"{Path(path).name} does not read back")
    return file_sha256(path)


def check_figure_files(out, stem):
    """Each figure file opens again: the PNG has the text width at DPI, the
    PDF starts as a PDF and the SVG parses."""
    img = mimage.imread(Path(out) / f"{stem}.png")
    want = int(round(TEXT_WIDTH_MM * MM * DPI))
    if abs(img.shape[1] - want) > 1:
        raise ValueError(f"{stem}.png is {img.shape[1]} px wide, not {want}")
    if not (Path(out) / f"{stem}.pdf").read_bytes().startswith(b"%PDF"):
        raise ValueError(f"{stem}.pdf is no PDF file")
    ET.parse(Path(out) / f"{stem}.svg")
    return {"png_px": [int(img.shape[1]), int(img.shape[0])]}


# Sources


def marker_file(d, name):
    """The marker of a stage in the model folder d, in common.marker_path()'s
    layout."""
    return Path(d) / marker_path(name).relative_to(marker_path(name).parents[1])


def check_marker(d, stage):
    """Raise ValueError unless the stage's outputs equal the SHA-256 that
    its marker recorded. Returns {file: SHA-256}."""
    mk = load_json(marker_file(d, stage.name))
    if mk is None:
        raise ValueError(
            f"the stage {stage.name} has no marker: run python3 "
            f"-m pipeline.runner --only {stage.name}"
        )
    rec = mk.get("outputs") or {}
    out = {}
    for key, path in sorted(stage.outputs().items()):
        want = (rec.get(key) or {}).get("sha256")
        have = file_sha256(path) if Path(path).is_file() else None
        if want is None or have != want:
            raise ValueError(
                f"{_rel(d, path)} is not the output that the marker of "
                f"{stage.name} recorded, so the stage's read-back has not "
                f"checked this file. To check the outputs again and record "
                f"them, remove the marker and adopt the outputs: rm "
                f"'{marker_file(d, stage.name)}' and then python3 -m "
                f"pipeline.runner --only {stage.name} --adopt"
            )
        out[_rel(d, path)] = have
    return out


def demo_file(d, user):
    """lb_demo.json of the user, or of the only demo folder when user is
    None."""
    root = Path(d) / "demo"
    if user is not None:
        p = root / lb._slug(user) / "lb_demo.json"
        if not p.is_file():
            raise ValueError(
                f"{_rel(d, p)} is missing: run python3 -m "
                f"pipeline.lb_demo --user {user}"
            )
        return p
    found = sorted(root.glob("*/lb_demo.json")) if root.is_dir() else []
    if len(found) != 1:
        raise ValueError(f"give --demo-user: {len(found)} demo pages under " f"{root}")
    return found[0]


# Model


def model_part(st, d, scratch, say):
    """The final model, its replicates and the background means, checked
    (module docstring). Returns a dict for the shapes and the demo."""
    t0 = time.time()
    base = st._base_context()
    G = p14.load_models(d, base)
    M, rec, ptr = G["M"], G["rec"], G["ptr"]
    res = load_json(Path(d) / ptr["files"]["results"])
    tag = rec["tag"] if rec["changed"] else None
    work = Path(scratch) / (f"{pc.WORK_NAME}_{tag}" if tag else pc.WORK_NAME)
    saved = ((load_json(work / "fits.json") or {}).get("means") or {}).get(
        f"wide:{M['level']}"
    )
    say(
        f"{WORK_NAME}: final model {rec['fit_name']}, "
        f"{len(M['lay'].terms)} terms, "
        f"{len(G['boot']['listener'])} listener and "
        f"{len(G['boot']['twoway'])} twoway replicates. Background means "
        + (
            "from P13c's saved work"
            if saved is not None
            else "not saved, so they come from the level's rows (a few minutes, "
            "memory near P13b part 2's peak)"
        )
    )
    means = pc.stored_means(st, base, M, work)
    lay = M["lay"]
    names = [t.name for t in lay.terms]
    stored = np.asarray(M["tables"], dtype=np.float64)
    gap = float(np.max(np.abs(pc.centered(M, M["theta"], means) - stored)))
    if gap > s13b2.CENTER_TOL:
        raise ValueError(
            f"the model's tables, centered with the background "
            f"means, differ from the stored tables by {gap:.1e}"
        )
    sq = base["shape_q"]
    full = pc.value_array(M, [M["theta"]], means, sq)[0]
    reps = {s: pc.value_array(M, list(G["boot"][s]), means, sq) for s in pc.SCHEMES}
    bands = res["bootstrap"]["bands"]
    if pc.band_rows(names, full, reps) != bands:
        raise ValueError(
            "the replicates and means do not give the stored C1 " "bands of P13c"
        )
    n_rep = sum(len(G["boot"][s]) for s in pc.SCHEMES)
    prog = Progress(
        f"{WORK_NAME}: centering the replicates", n_rep, say, unit=" replicates"
    )
    boot_T = {}
    for s in pc.SCHEMES:
        rows = []
        for th in G["boot"][s]:
            rows.append(pc.centered(M, th, means))
            prog.advance(1)
        boot_T[s] = np.array(rows, dtype=np.float64)
    prog.finish()
    free = None
    zs = np.load(Path(d) / pc.P13D_FILES[1])
    fits = [str(x) for x in zs["names"].tolist()]
    tabs = np.asarray(zs["tables"], dtype=np.float64)
    if rec["fit_name"] not in fits or not np.array_equal(
        tabs[fits.index(rec["fit_name"])], stored
    ):
        raise ValueError(
            "p13d_model_set.npz holds other tables for the final "
            "fit than the final model's"
        )
    family, _, rest = rec["fit_name"].partition("|")
    if family == "bounded":
        twin = f"free|{rest}"
        if twin not in fits:
            raise ValueError(f"p13d_model_set.npz holds no fit {twin}")
        free = {"name": twin, "tables": tabs[fits.index(twin)]}
    say(
        f"{WORK_NAME}: the model, its {n_rep} replicates and the means give "
        f"P13c's stored bands exactly (centering gap {gap:.1e}), in "
        f"{_clock(time.time() - t0)}"
    )
    return {
        "G": G,
        "M": M,
        "rec": rec,
        "res": res,
        "base": base,
        "means": means,
        "names": names,
        "boot_T": boot_T,
        "free": free,
        "bounded": list(G["bounded"]),
        "centering_gap": gap,
        "work": work,
    }


# Shapes


def x_scale(knots):
    """("linear", None) or ("symlog", linear part) for a term's knots
    (module docstring)."""
    a = np.abs(np.asarray(knots, dtype=np.float64))
    med, top = float(np.median(a)), float(a.max())
    if top > TAIL_RATIO * med:
        lt = LINEAR_SHARE * med if med > 0 else float(a[a > 0].min())
        return "symlog", lt
    return "linear", None


def grid_of(knots, q, scale):
    """The panel's x grid: every knot, the picks' quantiles (clamped into
    the knots, as p13c.value_array() reads them) and GRID_POINTS points with
    equal spacing on the x scale."""
    kn = np.asarray(knots, dtype=np.float64)
    if scale[0] == "symlog":
        tr = mscale.SymmetricalLogTransform(10, scale[1], 1.0)
        a, b = tr.transform(np.array([kn[0], kn[-1]]))
        even = tr.inverted().transform(np.linspace(a, b, GRID_POINTS))
    else:
        even = np.linspace(kn[0], kn[-1], GRID_POINTS)
    qc = [min(max(float(x), kn[0]), kn[-1]) for x in q]
    g = np.unique(
        np.concatenate(
            [kn, np.asarray(qc, dtype=np.float64), np.clip(even, kn[0], kn[-1])]
        )
    )
    return g, qc


def term_values(t, off, T, x):
    """The term's value at x from the flat tables T, as the model scores:
    np.interp on the knots, flat beyond the end knots."""
    kn = t.knot_array
    return np.interp(np.clip(x, kn[0], kn[-1]), kn, np.asarray(T)[off : off + t.K])


def shapes_data(m):
    """Everything fig_shapes draws, with the checks of the bands (module
    docstring)."""
    M, res, base = m["M"], m["res"], m["base"]
    lay = M["lay"]
    stored = np.asarray(M["tables"], dtype=np.float64)
    bands = res["bootstrap"]["bands"]
    imp = res["bootstrap"]["importance"]
    out = []
    worst = 0.0
    for t, off in zip(lay.terms, lay.offsets):
        rows = {r["point"]: r for r in bands[t.name]}
        q = [float(x) for x in (base["shape_q"].get(t.name) or [])[:5]]
        scale = x_scale(t.knot_array)
        grid, qc = grid_of(t.knot_array, q, scale)
        curve = term_values(t, off, stored, grid)
        V = {
            s: np.array([term_values(t, off, T, grid) for T in m["boot_T"][s]])
            for s in pc.SCHEMES
        }
        band = {s: np.quantile(V[s], pc.BAND, axis=0) for s in pc.SCHEMES}
        for p, x in enumerate(qc):
            i = int(np.searchsorted(grid, x))
            r = rows.get(pc.POINTS[p])
            if r is None or grid[i] != x:
                raise ValueError(f"{t.name}: no stored C1 band at " f"{pc.POINTS[p]}")
            for s in pc.SCHEMES:
                if [float(band[s][0, i]), float(band[s][1, i])] != [
                    r[s]["lo"],
                    r[s]["hi"],
                ]:
                    raise ValueError(
                        f"{t.name}: the {s} band at "
                        f"{pc.POINTS[p]} differs from P13c's C1"
                    )
            worst = max(worst, abs(float(curve[i]) - r["value"]))
        free = None
        if m["free"] is not None and t.name in m["bounded"]:
            free = term_values(t, off, m["free"]["tables"], grid)
        states = []
        for key, flag, slot in (
            ("zero", t.zero_state, gam.ZERO),
            ("missing", t.missing_state, gam.MISS),
        ):
            if not flag:
                continue
            j = off + t.K + slot
            r = rows.get(key)
            sb = {s: np.quantile(m["boot_T"][s][:, j], pc.BAND) for s in pc.SCHEMES}
            if r is None:
                raise ValueError(f"{t.name}: no stored C1 band for the " f"{key} state")
            for s in pc.SCHEMES:
                if [float(sb[s][0]), float(sb[s][1])] != [r[s]["lo"], r[s]["hi"]]:
                    raise ValueError(
                        f"{t.name}: the {s} band of the {key} "
                        f"state differs from P13c's C1"
                    )
            worst = max(worst, abs(float(stored[j]) - r["value"]))
            states.append(
                {
                    "key": key,
                    "label": STATE_LABELS[key],
                    "value": float(stored[j]),
                    "lo": float(sb["listener"][0]),
                    "hi": float(sb["listener"][1]),
                    "lo2": float(sb["twoway"][0]),
                    "hi2": float(sb["twoway"][1]),
                    "free": (None if free is None else float(m["free"]["tables"][j])),
                }
            )
        if worst > s13b2.CENTER_TOL:
            raise ValueError(
                f"{t.name}: the stored curve differs from the C1 "
                f"values by {worst:.1e}"
            )
        out.append(
            {
                "name": t.name,
                "label": TERM_LABELS.get(t.name, t.name),
                "xlabel": TERM_X.get(t.name, t.source),
                "scale": scale,
                "grid": grid,
                "curve": curve,
                "lo": band["listener"][0],
                "hi": band["listener"][1],
                "lo2": band["twoway"][0],
                "hi2": band["twoway"][1],
                "free": free,
                "states": states,
                "q": qc,
                "importance": float(imp[t.name]),
                "bounded": t.name in m["bounded"],
                "knots": t.knot_array,
                "heights": stored[off : off + t.K],
            }
        )
    return {
        "terms": out,
        "n_listener": int(len(m["boot_T"]["listener"])),
        "n_twoway": int(len(m["boot_T"]["twoway"])),
        "free_name": None if m["free"] is None else m["free"]["name"],
        "curve_gap": worst,
    }


def _plain(v, _pos=None):
    if v == 0:
        return "0"
    return f"{v:g}"


def symlog_ticks(lo, hi, lt):
    """0 and the powers of ten outside the linear part (|x| >= lt), inside
    [lo, hi]: a tick inside the linear part would crowd the 0."""
    out = [0.0] if lo <= 0.0 <= hi else []
    top = max(abs(lo), abs(hi))
    for k in range(math.ceil(math.log10(lt)), math.floor(math.log10(top)) + 1):
        v = 10.0**k
        out += [x for x in (-v, v) if lo <= x <= hi]
    return sorted(out)


def draw_shapes(S, out):
    """A fixed layout in mm, so the panels of a column line up whatever the
    tick labels' widths: per panel a state axis, then the curve axis."""
    terms = S["terms"]
    ncol = 2
    nrow = math.ceil(len(terms) / ncol)
    W, row_h, legend_h = TEXT_WIDTH_MM, 47.0, 13.0
    H = 1.0 + row_h * nrow + legend_h
    col_x, sw, gap, cw, ah = (14.0, 88.0), 9.5, 3.0, 47.5, 26.0

    def rect(x, y_top, w, h):
        return (x / W, 1.0 - (y_top + h) / H, w / W, h / H)

    fig = plt.figure(figsize=(W * MM, H * MM))
    for k, T in enumerate(terms):
        r, c = divmod(k, ncol)
        y0 = 1.0 + r * row_h + 6.0
        ax_s = fig.add_axes(rect(col_x[c], y0, sw, ah))
        ax = fig.add_axes(rect(col_x[c] + sw + gap, y0, cw, ah), sharey=ax_s)
        ax.fill_between(
            T["grid"], T["lo"], T["hi"], color=SLOT[0], alpha=0.22, lw=0, zorder=1
        )
        if T["free"] is not None:
            ax.plot(T["grid"], T["free"], color=MUTED, lw=0.8, zorder=2)
        ax.plot(T["grid"], T["curve"], color=SLOT[0], lw=1.3, zorder=3)
        ax.axhline(0.0, color=AXIS_C, lw=0.6, zorder=0)
        if T["scale"][0] == "symlog":
            ax.set_xscale("symlog", linthresh=T["scale"][1], linscale=1.0)
            ax.xaxis.set_major_locator(
                ticker.FixedLocator(
                    symlog_ticks(T["grid"][0], T["grid"][-1], T["scale"][1])
                )
            )
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_plain))
        ax.set_xlim(T["grid"][0], T["grid"][-1])
        ax.yaxis.grid(True, color=GRID_C, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelleft=False, length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(T["xlabel"])
        # the label starts over the state axis, so it has the panel's width
        ax_s.set_title(T["label"], loc="left", fontweight="bold")
        ax.set_title(
            f"importance {T['importance']:.2f}", loc="right", color=INK2, fontsize=7.0
        )
        q = T["q"]
        if len(q) == 5:
            tr = ax.get_xaxis_transform()
            ax.plot(
                [q[0], q[4]],
                [0.07, 0.07],
                transform=tr,
                color=INK2,
                lw=0.8,
                solid_capstyle="butt",
                zorder=4,
            )
            ax.plot(
                [q[1], q[3]],
                [0.07, 0.07],
                transform=tr,
                color=INK2,
                lw=2.6,
                solid_capstyle="butt",
                zorder=4,
            )
            ax.plot(
                [q[2]],
                [0.07],
                transform=tr,
                marker="o",
                ms=3.4,
                color="white",
                mec=INK2,
                mew=0.9,
                ls="none",
                zorder=5,
            )
        for i, s in enumerate(T["states"]):
            # the band as a line from lo to hi: a percentile band need not
            # hold the fit, so no error bar of signed lengths
            ax_s.vlines([i], [s["lo"]], [s["hi"]], color=SLOT[0], lw=1.0, zorder=3)
            ax_s.plot(
                [i],
                [s["value"]],
                marker="o",
                ms=3.2,
                color=SLOT[0],
                mec="white",
                mew=0.5,
                ls="none",
                zorder=4,
            )
            if s["free"] is not None:
                ax_s.plot(
                    [i + 0.34],
                    [s["free"]],
                    marker="o",
                    ms=2.4,
                    color=MUTED,
                    ls="none",
                    zorder=2,
                )
        ns = len(T["states"])
        ax_s.set_xlim(-0.6, max(ns, 1) - 0.25)
        ax_s.set_xticks(list(range(ns)))
        ax_s.set_xticklabels(
            [s["label"] for s in T["states"]], rotation=90, fontsize=6.5
        )
        ax_s.axhline(0.0, color=AXIS_C, lw=0.6, zorder=0)
        ax_s.yaxis.grid(True, color=GRID_C, lw=0.5)
        ax_s.set_axisbelow(True)
        if k % ncol == 0:
            ax_s.set_ylabel("centered effect (log-odds)")
        vals = [T["lo"], T["hi"], T["curve"]] + (
            [T["free"]] if T["free"] is not None else []
        )
        lo = min(float(np.min(v)) for v in vals)
        hi = max(float(np.max(v)) for v in vals)
        for s in T["states"]:
            lo = min(lo, s["lo"], s["free"] if s["free"] is not None else lo)
            hi = max(hi, s["hi"], s["free"] if s["free"] is not None else hi)
        rng = (hi - lo) or 1.0
        ax_s.set_ylim(lo - 0.22 * rng, hi + 0.08 * rng)
    handles = [
        Line2D([], [], color=SLOT[0], lw=1.3),
        Patch(facecolor=SLOT[0], alpha=0.22, lw=0),
        Line2D([], [], color=INK2, lw=2.2, marker="o", ms=3.2, mfc="white", mec=INK2),
    ]
    labels = [
        "final model",
        f"95% band, {S['n_listener']} listener " f"bootstrap replicates",
        "selection picks: median, 25-75% (thick), 5-95% (thin)",
    ]
    if S["free_name"] is not None:
        handles.insert(2, Line2D([], [], color=MUTED, lw=0.8))
        labels.insert(2, "free fit without bounds")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.5 / H),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.8,
    )
    return fig


def shapes_rows(S):
    rows = []
    for T in S["terms"]:
        free = T["free"] if T["free"] is not None else [None] * len(T["grid"])
        for i, x in enumerate(T["grid"]):
            rows.append(
                [
                    T["name"],
                    "curve",
                    float(x),
                    float(T["curve"][i]),
                    float(T["lo"][i]),
                    float(T["hi"][i]),
                    float(T["lo2"][i]),
                    float(T["hi2"][i]),
                    None if free[i] is None else float(free[i]),
                ]
            )
        for s in T["states"]:
            rows.append(
                [
                    T["name"],
                    s["key"],
                    None,
                    s["value"],
                    s["lo"],
                    s["hi"],
                    s["lo2"],
                    s["hi2"],
                    s["free"],
                ]
            )
    return (
        [
            "term",
            "kind",
            "x",
            "value",
            "listener_lo",
            "listener_hi",
            "twoway_lo",
            "twoway_hi",
            "free_fit",
        ],
        rows,
    )


# Cold start


def _interval(r):
    """(mean, lo, hi) of a listener metric from (n, 2) ranks, written here:
    the listener's mean over its picks, then the mean over the listeners
    with cold_curve.Z95 standard errors."""
    r = np.asarray(r, dtype=np.float64)
    gain = np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0)
    out = {}
    for k, v in (
        ("ndcg_at_100", gain),
        ("recall_at_100", (r <= 100).astype(np.float64)),
    ):
        x = v.mean(axis=1)
        m = float(x.mean())
        se = float(x.std(ddof=1) / math.sqrt(len(x)))
        out[k] = (m, m - cc.Z95 * se, m + cc.Z95 * se)
    return out


def cold_data(d):
    """The curve, computed here from cold_start_curve.npz and checked
    against cold_start_curve.json (module docstring)."""
    doc = load_json(Path(d) / cc.OUT_FILES[0])
    res = doc["results"]
    with np.load(Path(d) / cc.OUT_FILES[1]) as z:
        points = [int(x) for x in z["points"]]
        ranks = {a: np.asarray(z[f"rank_{a}"]) for a in cc.CURVE_ARMS}
        full = {a: np.asarray(z[f"p14_rank_{a}"]) for a in cc.ARMS}
    if points + ["full"] != res["points"]:
        raise ValueError(
            "the cut points of cold_start_curve.npz differ from " "its JSON file"
        )
    arms = {}
    gap = 0.0
    for a in cc.ARMS:
        got = []
        for j in range(len(points)):
            r = full["popularity"] if a == "popularity" else ranks[a][:, j, :]
            got.append(_interval(r))
        got.append(_interval(full[a]))
        for g, row in zip(got, res["arms"][a]):
            for k in cc.METRICS:
                s = row[k]
                for v, w in zip(g[k], (s["mean"], s["lo"], s["hi"])):
                    gap = max(gap, abs(v - w))
        arms[a] = got
    if gap > CHECK_ATOL:
        raise ValueError(
            f"the cold-start curve computed from its ranks "
            f"differs from cold_start_curve.json by {gap:.1e}"
        )
    return {
        "doc": doc,
        "points": points,
        "arms": arms,
        "gap": gap,
        "listeners": int(doc["data"]["listeners"]),
    }


def draw_cold(C, out):
    pts = C["points"]
    xs = [math.log10(n) for n in pts]
    xfull = xs[-1] + 0.55
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_MM * MM, 64 * MM), layout="constrained"
    )
    for ax, metric, title in zip(axes, cc.METRICS, ("NDCG@100", "R@100")):
        for a in cc.ARMS:
            color, marker, dx = ARM_STYLE[a]
            rows = C["arms"][a]
            if a == "popularity":
                m, lo, hi = rows[-1][metric]
                span = [xs[0] - 0.12, xfull + 0.12]
                ax.fill_between(
                    span, [lo, lo], [hi, hi], color=MUTED, alpha=0.15, lw=0, zorder=1
                )
                ax.plot(span, [m, m], color=MUTED, lw=1.0, zorder=2)
                continue
            m = np.array([r[metric][0] for r in rows])
            lo = np.array([r[metric][1] for r in rows])
            hi = np.array([r[metric][2] for r in rows])
            x = np.array(xs + [xfull]) + dx
            ax.plot(x[:-1], m[:-1], color=color, lw=1.3, zorder=3)
            ax.errorbar(
                x,
                m,
                yerr=[m - lo, hi - m],
                fmt=marker,
                ms=3.6,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=1.8,
                mec="white",
                mew=0.6,
                zorder=4,
            )
        ax.axvline((xs[-1] + xfull) / 2.0, color=AXIS_C, lw=0.6, zorder=0)
        ax.set_yscale("log")
        pos = [v for a in cc.ARMS for r in C["arms"][a] for v in r[metric] if v > 0]
        ax.set_ylim(min(pos) / 1.4, max(pos) * 1.4)
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.grid(True, color=GRID_C, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(xs + [xfull])
        ax.set_xticklabels([str(n) for n in pts] + ["full"])
        ax.set_xlim(xs[0] - 0.12, xfull + 0.12)
        ax.set_xlabel("training pairs kept per listener (N)")
        ax.set_title(title, loc="left", fontweight="bold")
    handles, labels = [], []
    for a in cc.ARMS:
        color, marker, _ = ARM_STYLE[a]
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                lw=1.3 if marker else 1.0,
                marker=marker,
                ms=3.6,
                mec="white",
                mew=0.6,
            )
        )
        labels.append(
            _cap(cc.ARM_LABELS[a]) + (" (no history)" if a == "popularity" else "")
        )
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        handlelength=2.2,
        columnspacing=1.6,
    )
    return fig


def cold_rows(C):
    rows = []
    for a in cc.ARMS:
        for n, r in zip(C["points"] + ["full"], C["arms"][a]):
            for k in cc.METRICS:
                rows.append([a, k, n, *[float(v) for v in r[k]]])
    return ["arm", "metric", "N", "mean", "lo", "hi"], rows


# Explanations


def _artist_key(r):
    for x in r["reasons"]:
        if x["reason"] == "artist" and x.get("credit") is not None:
            return ("credit", int(x["credit"]))
    return ("name", r.get("artist"))


def pick_examples(gam_list, k=EXAMPLES):
    """The examples of the module docstring: rank 1, then the next tracks by
    rank whose artist differs from every earlier example's."""
    out, seen = [], set()
    for r in sorted(gam_list, key=lambda r: int(r["rank"])):
        key = _artist_key(r)
        if out and key in seen:
            continue
        out.append(r)
        seen.add(key)
        if len(out) == k:
            break
    return out


def evidence_text(x):
    """The evidence of a shown reason in the words of the demo page
    (lb_demo._reason_html), but genre names withheld."""
    ev_ = x.get("evidence") or {}
    units = ev_.get("units") or []
    more = max(0, int(ev_.get("count") or 0) - len(units))
    if x["reason"] == "artist":
        who = x.get("credit_name") or "this artist"
        n = int(x.get("credit_tracks") or 0)
        parts = [f"You play {n} track{'' if n == 1 else 's'} by {who}"]
        co = [u.get("title") or f"item {u['id']}" for u in units]
        if co:
            parts.append(
                "Listeners of " + ", ".join(f"“{c}”" for c in co[:3]) + " also play it"
            )
        hist = x.get("history_labels")
        if hist and not co:
            parts.append(
                "For example "
                + ", ".join(f"“{h.get('title') or ''}”" for h in hist[:2])
            )
        return ". ".join(parts) + "."
    if x["reason"] == "audio_fit":
        return (
            ("Close to your usual " + ", ".join(u["name"] for u in units[:4]) + ".")
            if units
            else ""
        )
    if x["reason"] == "genre_affinity":
        n = int(ev_.get("count") or 0)
        return (
            f"{n} matched genre{'' if n == 1 else 's'} (names withheld: "
            f"Spotify-derived)"
        )
    body = ", ".join(u["name"] for u in units)
    if more:
        body += f" and {more} more"
    return body


def _page_body(x):
    """The evidence text of the demo page's chip for a reason."""
    s = lb._reason_html(x)
    s = re.sub(r"<b>.*?</b>", "", s, count=1)
    s = re.sub(r'<span class="odds">.*?</span>', "", s, count=1)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return s[2:] if s.startswith(": ") else s


def demo_data(m, demo_path):
    """The example explanations, checked (module docstring)."""
    doc = load_json(demo_path)
    if doc is None or not doc.get("gam"):
        raise ValueError(
            f"{demo_path} holds no model list: run the demo " f"with a history first"
        )
    G = m["G"]
    want = {
        "final": m["rec"]["fit_name"],
        "replicates": int(len(G["boot"]["listener"])),
        "members": int(len(G["members"])),
    }
    have = {k: doc["model"].get(k) for k in want}
    if have != want:
        raise ValueError(
            f"the demo page was made with another model "
            f"({have}, not {want}): run the demo again"
        )
    ex = pick_examples(doc["gam"])
    for r in ex:
        vals = [x["value"] for x in r["reasons"]]
        if any(v <= 0 for v in vals) or vals != sorted(vals, reverse=True):
            raise ValueError(
                f"rank {r['rank']}: the shown reasons are not "
                f"positive in descending order"
            )
        for x in r["reasons"]:
            if x["odds"] != lb._exp(x["value"]):
                raise ValueError(
                    f"rank {r['rank']}: an odds factor is not " f"exp(value)"
                )
            ev_ = x.get("evidence") or {}
            if int(ev_.get("count") or 0) < len(ev_.get("units") or []):
                raise ValueError(
                    f"rank {r['rank']}: an evidence count is " f"below its units"
                )
            if x["reason"] != "genre_affinity" and evidence_text(x) != _page_body(x):
                raise ValueError(
                    f"rank {r['rank']}: the evidence text of "
                    f"{x['reason']} differs from the demo page"
                )
    out = []
    for r in ex:
        shown = [x["reason"] for x in r["reasons"]]
        out.append(
            {
                "rank": int(r["rank"]),
                "item": int(r["item"]),
                "title": r.get("title") or f"item {r['item']}",
                "artist": r.get("artist") or "unknown artist",
                "reasons": r["reasons"],
                "not_shown": [c for c in r.get("candidates", []) if c not in shown],
            }
        )
    return {"examples": out, "doc": doc}


def _short(s, n=TITLE_CHARS):
    return s if len(s) <= n else s[: n - 1] + "…"


def draw_explanations(E, out):
    ex = E["examples"]
    sizes = [max(1, len(e["reasons"])) + (0.6 if e["not_shown"] else 0.0) for e in ex]
    H = sum(9.0 + 6.0 * k for k in sizes) + 12.0
    fig = plt.figure(figsize=(TEXT_WIDTH_MM * MM, H * MM), layout="constrained")
    gs = fig.add_gridspec(len(ex), 1, height_ratios=[k + 0.9 for k in sizes])
    top = max(x["value"] for e in ex for x in e["reasons"])
    axes = []
    for i, e in enumerate(ex):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        vals = [x["value"] for x in e["reasons"]]
        y = np.arange(len(vals))[::-1].astype(float)
        ax.barh(y, vals, height=0.55, color=SLOT[0], zorder=2)
        for yi, x in zip(y, e["reasons"]):
            ax.text(
                x["value"] + 0.012 * top,
                yi,
                f"{x['value']:+.2f} (odds {lb._odds(x['odds'])})",
                va="center",
                ha="left",
                fontsize=6.8,
                color=INK,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [lb.REASON_LABELS.get(x["reason"], x["reason"]) for x in e["reasons"]]
        )
        ax.set_ylim(-1.3 if e["not_shown"] else -0.7, len(vals) - 0.3)
        ax.set_title(
            _short(f"{e['rank']}. “{e['title']}” by " f"{e['artist']}"),
            loc="left",
            fontweight="bold",
        )
        if e["not_shown"]:
            # a row of its own under the bars
            ax.text(
                top * 1.34,
                -0.95,
                "candidate reasons that the rules "
                "do not show: "
                + ", ".join(lb.REASON_LABELS.get(c, c) for c in e["not_shown"]),
                ha="right",
                va="center",
                fontsize=6.5,
                color=MUTED,
            )
        ax.xaxis.grid(True, color=GRID_C, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if i < len(ex) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[0].set_xlim(0.0, top * 1.34)
    axes[-1].set_xlabel(
        "contribution to the track's log-odds against the " "mean candidate"
    )
    return fig


def explanation_rows(E):
    rows = []
    for e in E["examples"]:
        for x in e["reasons"]:
            rows.append(
                [
                    e["rank"],
                    e["item"],
                    e["title"],
                    e["artist"],
                    x["reason"],
                    float(x["value"]),
                    float(x["odds"]),
                    evidence_text(x),
                ]
            )
    return (
        ["rank", "item", "title", "artist", "reason", "value", "odds", "evidence"],
        rows,
    )


def explanation_table(E):
    """The example explanations as a table (the tables' layout)."""
    rows = []
    for e in E["examples"]:
        for x in e["reasons"]:
            rows.append(
                [
                    str(e["rank"]),
                    f"“{e['title']}” by {e['artist']}",
                    lb.REASON_LABELS.get(x["reason"], x["reason"]),
                    f"{x['value']:+.3f}",
                    lb._odds(x["odds"]),
                    evidence_text(x),
                ]
            )
    return {
        "id": "demo_01_explanations",
        "source": "demo",
        "caption": "Example explanations of the demo's top 25",
        "notes": [
            "The tracks: rank 1, then the next tracks whose artist "
            "differs from every earlier example. Contribution: "
            "log-odds against the mean candidate. Odds factor: "
            "exp(contribution). Only the reasons that D1 and S4 "
            "show."
        ],
        "header": [
            "Rank",
            "Track",
            "Reason",
            "Contribution",
            "Odds factor",
            "Evidence",
        ],
        "align": ["r", "l", "l", "r", "r", "l"],
        "rows": rows,
    }


# Pipeline


def pipeline_data():
    """The runner's stages in PHASES, checked (module docstring)."""
    stages = list(runner.STAGES)
    names = [s.name for s in stages]
    order = {n: i for i, n in enumerate(names)}
    up = {s.name: list(getattr(s, "upstream", [])) for s in stages}
    placed = [n for _, members in PHASES for n in members]
    if sorted(placed) != sorted(names) or len(set(placed)) != len(placed):
        raise ValueError(
            f"PHASES does not hold every runner stage exactly "
            f"once: missing {sorted(set(names) - set(placed))}, "
            f"unknown {sorted(set(placed) - set(names))}"
        )
    if sorted(STAGE_TEXT) != sorted(names):
        raise ValueError("STAGE_TEXT does not label every runner stage")
    phase = {n: k for k, (_, members) in enumerate(PHASES) for n in members}
    for n, ups in up.items():
        late = [u for u in ups if phase[u] > phase[n]]
        if late:
            raise ValueError(f"{n} depends on {late}, which sit in a later " f"phase")
    used = set(PHASES[-1][1])
    stack = list(used)
    while stack:
        for u in up[stack.pop()]:
            if u not in used:
                used.add(u)
                stack.append(u)
    here = Path(__file__).resolve().parent
    missing = [f for f, _ in SCRIPTS if not (here / f).is_file()]
    if missing:
        raise ValueError(
            f"the plain scripts {missing} are not in the " f"pipeline package"
        )
    boxes = [{"title": DATA_BOX[0], "text": DATA_BOX[1], "items": []}]
    for title, members in PHASES:
        items = [
            (n,) + STAGE_TEXT[n] + (n in used,) for n in sorted(members, key=order.get)
        ]
        boxes.append({"title": title, "text": None, "items": items})
    boxes.append(
        {
            "title": SCRIPTS_TITLE,
            "text": None,
            "items": [(f, f, txt, True) for f, txt in SCRIPTS],
        }
    )
    return {
        "boxes": boxes,
        "upstream": up,
        "phase": phase,
        "unused": sorted(set(names) - used, key=order.get),
        "order": names,
    }


def draw_pipeline(P, out):
    W = TEXT_WIDTH_MM
    cols, row_h, head, pad, gap = 3, 4.1, 6.0, 2.2, 5.0
    heights = []
    for b in P["boxes"]:
        lines = textwrap.wrap(b["text"], 112) if b["text"] else []
        n_rows = math.ceil(len(b["items"]) / cols)
        heights.append(head + pad + row_h * (n_rows + len(lines)))
    H = sum(heights) + gap * (len(heights) - 1) + 9.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, W)
    ax.set_ylim(H, 0.0)
    ax.axis("off")
    y = 1.0
    colw = (W - 8.0) / cols
    for i, (b, h) in enumerate(zip(P["boxes"], heights)):
        runner_box = 0 < i < len(P["boxes"]) - 1
        ax.add_patch(
            FancyBboxPatch(
                (1.0, y),
                W - 2.0,
                h,
                boxstyle="round,pad=0,rounding_size=1.6",
                fc=FILL_C if runner_box else "white",
                ec=AXIS_C,
                lw=0.7,
            )
        )
        ax.text(
            3.5,
            y + 3.6,
            b["title"],
            fontsize=8.0,
            fontweight="bold",
            va="center",
            ha="left",
            color=INK,
        )
        yy = y + head + 1.4
        for line in (textwrap.wrap(b["text"], 112) if b["text"] else []):
            ax.text(3.5, yy, line, fontsize=6.8, va="center", ha="left", color=INK2)
            yy += row_h
        for k, (name, sid, desc, used) in enumerate(b["items"]):
            r, c = divmod(k, cols)
            x0 = 3.5 + c * colw
            y0 = yy + r * row_h
            ink = INK if used else MUTED
            if name == sid:
                ax.text(
                    x0,
                    y0,
                    sid,
                    fontsize=6.6,
                    va="center",
                    ha="left",
                    color=ink,
                    fontweight="bold",
                )
                ax.text(
                    x0 + 17.0,
                    y0,
                    desc,
                    fontsize=6.6,
                    va="center",
                    ha="left",
                    color=INK2,
                )
            else:
                ax.text(
                    x0,
                    y0,
                    sid,
                    fontsize=6.6,
                    va="center",
                    ha="left",
                    color=ink,
                    fontweight="bold",
                )
                ax.text(
                    x0 + 10.5,
                    y0,
                    desc,
                    fontsize=6.6,
                    va="center",
                    ha="left",
                    color=INK2 if used else MUTED,
                )
        y += h
        if i < len(P["boxes"]) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (W / 2.0, y + 0.5),
                    (W / 2.0, y + gap - 0.5),
                    arrowstyle="-|>",
                    mutation_scale=7.0,
                    color=INK2,
                    lw=0.8,
                )
            )
            y += gap
    ax.text(
        1.5,
        H - 3.5,
        "Gray: no evaluation stage depends on these "
        "stages (not used by the final model). Arrows: the order of "
        "the phases.",
        fontsize=6.3,
        va="center",
        ha="left",
        color=MUTED,
    )
    return fig


def pipeline_rows(P):
    rows = []
    for b in P["boxes"]:
        for name, sid, desc, used in b["items"]:
            rows.append(
                [
                    b["title"],
                    name,
                    sid,
                    desc,
                    "yes" if used else "no",
                    " ".join(P["upstream"].get(name, [])),
                ]
            )
    return ["box", "name", "id", "label", "used_by_final_model", "upstream"], rows


# Tables
_RULE = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$")


def _cells(line):
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        raise ValueError(f"a table row does not start and end with |: {s}")
    return [c.strip() for c in s[1:-1].split("|")]


def parse_report(text, source):
    """The tables of a stage report: [{id, source, caption, notes, header,
    align, rows}] (module docstring)."""
    lines = text.splitlines()
    title = next((s[2:].strip() for s in lines if s.startswith("# ")), source)
    tables, heading, paras, last, intro = [], title, [], None, []
    i = 0
    while i < len(lines):
        s = lines[i].rstrip()
        if s.startswith("#"):
            if last is not None and paras:
                last["notes"] += paras
            heading, paras, last = s.lstrip("#").strip(), [], None
            i += 1
            continue
        if (
            s.startswith("|")
            and i + 1 < len(lines)
            and _RULE.match(lines[i + 1].strip())
        ):
            header = _cells(s)
            rule = _cells(lines[i + 1])
            align = ["r" if c.endswith(":") else "l" for c in rule]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                row = _cells(lines[j])
                if len(row) != len(header):
                    raise ValueError(
                        f"{source}: a row of the table under "
                        f"'{heading}' has {len(row)} cells, "
                        f"not {len(header)}"
                    )
                rows.append(row)
                j += 1
            sub = paras[-1] if paras and paras[-1].endswith(":") else None
            notes = paras[:-1] if sub else list(paras)
            if not tables:
                notes = intro + notes
            cap = heading + (f": {sub[:-1]}" if sub else "")
            last = {
                "source": source,
                "report": title,
                "caption": cap,
                "notes": notes,
                "header": header,
                "align": align,
                "rows": rows,
            }
            tables.append(last)
            paras = []
            i = j
            continue
        if s.strip() and heading != title:
            paras.append(s.strip())
        elif s.strip() and not s.startswith(("# ", "- ")):
            intro.append(s.strip())  # a note of the report's opening
        i += 1
    if last is not None and paras:
        last["notes"] += paras
    n_rules = sum(1 for s in lines if _RULE.match(s.strip()))
    if n_rules != len(tables):
        raise ValueError(
            f"{source}: {n_rules} tables in the report, " f"{len(tables)} parsed"
        )
    seen = {}
    for t in tables:
        seen[t["caption"]] = seen.get(t["caption"], 0) + 1
        if seen[t["caption"]] > 1:
            t["caption"] += f" ({seen[t['caption']]})"
    for k, t in enumerate(tables, start=1):
        slug = re.sub(r"[^a-z0-9]+", "_", t["caption"].lower()).strip("_")
        t["id"] = f"{source}_{k:02d}_{slug[:40].strip('_')}"
    return tables


def _thesis_labels():
    out = {}
    for lab in list(p14.ARM_LABELS.values()) + list(cc.ARM_LABELS.values()):
        out[lab] = _cap(lab)
    out.update(METRIC_LABELS)
    out.update(lb.REASON_LABELS)
    return out


def _head(h):
    """A column head with a capital first letter, but not a statistic
    symbol (t, p) or a head that starts with a key (test_last)."""
    first = re.split(r"[ ,]", h, maxsplit=1)[0]
    if len(first) <= 1 or "_" in first or "@" in first:
        return h
    return _cap(h)


def thesis_table(t):
    """The report table with the thesis labels (module docstring)."""
    lab = _thesis_labels()
    return dict(
        t,
        caption=_cap(t["caption"]),
        header=[_head(h) for h in t["header"]],
        rows=[[lab.get(c, c) for c in r] for r in t["rows"]],
    )


def check_report(mod, d, name, report):
    """The stored report equals the text that the module's writer makes from
    the JSON file (the report follows from the results)."""
    doc = load_json(Path(d) / name)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "report.md"
        mod.write_report(doc, p)
        again = p.read_text(encoding="utf-8")
    text = (Path(d) / "reports" / report).read_text(encoding="utf-8")
    if again != text:
        raise ValueError(f"reports/{report} does not follow from {name}")
    return doc, text


def p14_check(d, doc, tables):
    """P14's accuracy, computed here from the ranks of p14.npz, against
    p14.json and the cells of the report's accuracy table (module
    docstring). Returns the largest gap."""
    with np.load(Path(d) / p14.OUT_FILES[1]) as z:
        ranks = {a: np.asarray(z[f"rank_{a}"], dtype=np.float64) for a in p14.ARMS}
    gap, ndcg = 0.0, {}
    got = {}
    for a, r in ranks.items():
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError(f"p14.npz: rank_{a} is not (listeners, 2)")
        gain = 1.0 / np.log2(r + 1.0)
        lm = {
            "ndcg_at_100": np.where(r <= 100, gain, 0.0).mean(axis=1),
            "ndcg_at_10": np.where(r <= 10, gain, 0.0).mean(axis=1),
            "recall_at_10": (r <= 10).astype(np.float64).mean(axis=1),
            "recall_at_100": (r <= 100).astype(np.float64).mean(axis=1),
            "mrr": (1.0 / r).mean(axis=1),
        }
        ndcg[a] = lm["ndcg_at_100"]
        row = {k: float(v.mean()) for k, v in lm.items()}
        for p, ty in enumerate(p14.PICK_TYPES):
            row[ty] = float(np.median(r[:, p]))
        st = doc["arms"][a]
        if int(st["listeners"]) != len(r):
            raise ValueError(
                f"p14.npz holds {len(r)} listeners for {a}, "
                f"p14.json {st['listeners']}"
            )
        for k in METRIC_LABELS:
            gap = max(gap, abs(row[k] - st[k]))
        for ty in p14.PICK_TYPES:
            gap = max(gap, abs(row[ty] - st["types"][ty]["median_rank"]))
        got[a] = row
    for t in doc["tests"]["confirmatory"]:
        m = float((ndcg["gam"] - ndcg[t["arm"]]).mean())
        gap = max(gap, abs(m - t["difference"]["mean"]))
    if gap > CHECK_ATOL:
        raise ValueError(
            f"P14's metrics computed from the ranks of p14.npz "
            f"differ from p14.json by {gap:.1e}"
        )
    acc = [t for t in tables if t["header"] == ACCURACY_HEAD]
    if len(acc) != 1:
        raise ValueError("the P14 report holds no single accuracy table")
    arm_of = {v: k for k, v in p14.ARM_LABELS.items()}
    if sorted(arm_of.get(r[0], r[0]) for r in acc[0]["rows"]) != sorted(p14.ARMS):
        raise ValueError("the accuracy table does not hold every P14 arm")
    for r in acc[0]["rows"]:
        g = got[arm_of[r[0]]]
        want = [
            r[0],
            f"{g['ndcg_at_100']:.5f}",
            f"{g['ndcg_at_10']:.5f}",
            f"{g['recall_at_10']:.4f}",
            f"{g['recall_at_100']:.4f}",
            f"{g['mrr']:.5f}",
            f"{g['test_last']:,.0f}",
            f"{g['test_first']:,.0f}",
        ]
        if r != want:
            raise ValueError(
                f"the accuracy table's row {r[0]} differs from "
                f"the values computed from the ranks"
            )
    return gap


def cold_table_check(C, tables):
    """The report's cells equal the curve computed here, in its format."""
    lab = {v: k for k, v in cc.ARM_LABELS.items()}
    n = 0
    for t in tables:
        metric = {"NDCG@100": "ndcg_at_100", "R@100": "recall_at_100"}.get(t["caption"])
        if metric is None:
            continue
        for r in t["rows"]:
            got = C["arms"][lab[r[0]]]
            want = [r[0]] + [
                f"{g[metric][0]:.5f} ({g[metric][1]:.5f} to " f"{g[metric][2]:.5f})"
                for g in got
            ]
            if r != want:
                raise ValueError(
                    f"the cold-start table {t['caption']}, row "
                    f"{r[0]}, differs from the curve computed "
                    f"from the ranks"
                )
            n += 1
    if n != 2 * len(cc.ARMS):
        raise ValueError("the cold-start report does not hold both tables")


def table_md(t):
    L = [f"### {t['id']}: {t['caption']}", ""]
    L.append("| " + " | ".join(t["header"]) + " |")
    L.append("|" + "|".join("---:" if a == "r" else "---" for a in t["align"]) + "|")
    L += ["| " + " | ".join(r) + " |" for r in t["rows"]]
    L += ([""] + list(t["notes"])) if t["notes"] else []
    return "\n".join(L) + "\n"


CSS = """body{font-family:"Times New Roman",Times,serif;font-size:12pt;
margin:24px;color:#000;background:#fff}
h1{font-size:14pt}h2{font-size:13pt;margin-top:28pt}
table{border-collapse:collapse;font-size:10pt;margin:4pt 0 4pt}
th,td{padding:1pt 6pt;vertical-align:top}
thead th{border-top:1.5pt solid #000;border-bottom:0.75pt solid #000;
font-weight:bold}
tbody tr:last-child td{border-bottom:1.5pt solid #000}
.r{text-align:right;font-variant-numeric:tabular-nums}.l{text-align:left}
p.cap{font-weight:bold;margin:16pt 0 2pt}p.note{font-size:9pt;margin:2pt 0}
p.id{font-size:8pt;color:#555;margin:0}"""


def tables_html(groups, intro):
    e = html.escape
    L = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Thesis tables</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
        "<h1>Thesis tables</h1>",
    ]
    L += [f'<p class="note">{e(x)}</p>' for x in intro]
    for title, tables in groups:
        L.append(f"<h2>{e(title)}</h2>")
        for t in tables:
            L.append(f'<p class="cap">{e(t["caption"])}</p>')
            L.append(f'<p class="id">{e(t["id"])}</p>')
            L.append(
                f'<table id="{e(t["id"])}"><thead><tr>'
                + "".join(
                    f'<th class="{a}">{e(h)}</th>'
                    for h, a in zip(t["header"], t["align"])
                )
                + "</tr></thead><tbody>"
            )
            for r in t["rows"]:
                L.append(
                    "<tr>"
                    + "".join(
                        f'<td class="{a}">{e(c)}</td>' for c, a in zip(r, t["align"])
                    )
                    + "</tr>"
                )
            L.append("</tbody></table>")
            L += [f'<p class="note">{e(x)}</p>' for x in t["notes"]]
    L.append("</body></html>")
    return "\n".join(L) + "\n"


class _TableReader(HTMLParser):
    """The tables of an HTML file: {id: {"header": [...], "rows": [[...]]}}."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.cur, self.cell, self.in_head = {}, None, None, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.cur = {"header": [], "rows": []}
            self.tables[a.get("id")] = self.cur
        elif tag == "thead":
            self.in_head = True
        elif tag == "tbody":
            self.in_head = False
        elif tag == "tr" and self.cur is not None and not self.in_head:
            self.cur["rows"].append([])
        elif tag in ("th", "td") and self.cur is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            text = "".join(self.cell)
            if self.in_head:
                self.cur["header"].append(text)
            else:
                self.cur["rows"][-1].append(text)
            self.cell = None
        elif tag == "table":
            self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def tables_part(d, C, E, out):
    """The tables (module docstring), written and read back. Returns
    ({file: SHA-256}, checks)."""
    doc14, text14 = check_report(p14, d, p14.OUT_FILES[0], p14.REPORT_NAME)
    docc, textc = check_report(cc, d, cc.OUT_FILES[0], cc.REPORT_NAME)
    t14 = parse_report(text14, "p14")
    tcc = parse_report(textc, "cold")
    gap14 = p14_check(d, doc14, t14)
    cold_table_check(C, tcc)
    groups = [
        (t14[0]["report"], [thesis_table(t) for t in t14]),
        (tcc[0]["report"], [thesis_table(t) for t in tcc]),
        ("Example explanations (ListenBrainz demo)", [explanation_table(E)]),
    ]
    intro = [
        "The tables of reports/p14_report.md and "
        "reports/cold_start_curve.md, cell for cell, with the thesis "
        "labels, then the example explanations of the demo. Each "
        "table has its own CSV file (tab_<id>.csv). The thesis "
        "numbering and captions go into the document."
    ]
    shas = {}
    html_text = tables_html(groups, intro)
    atomic_text_write(html_text, out / "tables.html")
    md = (
        "# Thesis tables\n\n"
        + "\n".join(intro)
        + "\n\n"
        + "\n".join(
            f"## {title}\n\n" + "\n".join(table_md(t) for t in tables)
            for title, tables in groups
        )
    )
    atomic_text_write(md, out / "tables.md")
    all_t = [t for _, ts in groups for t in ts]
    ids = [t["id"] for t in all_t]
    if len(set(ids)) != len(ids):
        raise ValueError("two tables share an id")
    for t in all_t:
        shas[f"tab_{t['id']}.csv"] = write_csv(
            out / f"tab_{t['id']}.csv", t["header"], t["rows"]
        )
    rd = _TableReader()
    rd.feed((out / "tables.html").read_text(encoding="utf-8"))
    for t in all_t:
        back = rd.tables.get(t["id"])
        if back is None or back["header"] != t["header"] or back["rows"] != t["rows"]:
            raise ValueError(f"tables.html: the table {t['id']} does not read " f"back")
    if len(rd.tables) != len(all_t):
        raise ValueError("tables.html holds other tables than written")
    if (out / "tables.md").read_text(encoding="utf-8") != md:
        raise ValueError("tables.md does not read back")
    shas["tables.html"] = file_sha256(out / "tables.html")
    shas["tables.md"] = file_sha256(out / "tables.md")
    return shas, {
        "tables": len(all_t),
        "p14_gap": gap14,
        "reports_follow_from_json": True,
    }


def captions(S, C, E, P, m):
    rec = m["rec"]
    n_terms = len(S["terms"])
    unused = ", ".join(STAGE_TEXT[n][0] for n in P["unused"]) or "none"
    ex = E["examples"]
    L = [
        "# Draft captions",
        "",
        "Drafts for the thesis document. The document gives the numbers "
        "(Fig. 2.1 and so on) and the final words.",
        "",
        "## fig_shapes",
        "",
        f"The {n_terms} terms of the final model ({rec['fit_name']}). Each "
        f"panel shows the centered effect of a term on the log-odds of a "
        f"track against the value of its concept. The blue line is the "
        f"fit, and the blue band is the pointwise 95 percent band over "
        f"{S['n_listener']} listener bootstrap replicates."
        + (
            " The gray line is the free fit without monotone bounds."
            if S["free_name"]
            else ""
        )
        + ' The points at the left are the "no match" and "no data" '
        "states with their bands. The marks under each curve show the "
        "median, the 25th to 75th and the 5th to 95th percentiles of "
        "the selection picks. Each panel has its own y scale. "
        "Importance: the mean absolute contribution of the term.",
        "",
        f"Sources: the P13c results of the final model "
        f"({m['G']['ptr']['files']['results']}), its "
        f"{S['n_listener']} listener replicates, P13a's shape quantiles and "
        f"p13d_model_set.npz.",
        "",
        "## fig_cold_start",
        "",
        f"NDCG@100 and R@100 of {C['listeners']:,} eval listeners when "
        f"each history keeps only its first N training pairs. The point "
        f"at the right uses the full history (P14). Error bars: 95 percent "
        f"intervals over the listeners. Popularity ignores the history. "
        f"Both axes are logarithmic. Descriptive, no test.",
        "",
        "Sources: cold_start_curve.json and cold_start_curve.npz.",
        "",
        "## fig_explanations",
        "",
        f"The shown reasons of {len(ex)} tracks of the GAM's top 25 for "
        f"the author's ListenBrainz history: rank "
        + ", ".join(str(e["rank"]) for e in ex)
        + ". Each bar is the contribution of a reason to the log-odds of "
        "the track against the mean candidate, with its odds factor. "
        "Only the reasons that the stability rules D1 and S4 show "
        "appear.",
        "",
        "Source: lb_demo.json of the demo. The evidence text is in "
        "tab_demo_01_explanations.csv.",
        "",
        "## fig_pipeline",
        "",
        f"The stages of the pipeline runner in five phases, between the "
        f"data and the plain scripts on the stored outputs. Gray stages "
        f"({unused}) are not used by the final model.",
        "",
        "Source: pipeline/runner.py (the stages and their upstream " "lists).",
        "",
        "## Tables",
        "",
        "tables.html holds every table of the P14 and cold-start reports "
        "with the thesis labels, and the example explanations. Its "
        "captions are the report headings. Each table also has a CSV "
        "file.",
        "",
    ]
    return "\n".join(L)


# Run


def _group_key(inputs, extra):
    doc = {
        "version": VERSION,
        "code": file_sha256(Path(__file__)),
        "inputs": inputs,
        "extra": extra,
    }
    return _sha_text(json.dumps(doc, sort_keys=True, default=str))


def _complete(rec, key, out):
    if not rec or rec.get("key") != key:
        return False
    for name, sha in (rec.get("outputs") or {}).items():
        p = Path(out) / name
        if not p.is_file() or file_sha256(p) != sha:
            return False
    return True


def run(st, model_dir, scratch_dir, out_dir, demo_path, say=print, force=False):
    """Every output of the module docstring. st: a StageP13b2 (its base
    context and, when P13c's work holds no means, its loaders). Returns the
    manifest."""
    t0 = time.time()
    d, scratch, out = Path(model_dir), Path(scratch_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    font, font_file = resolved_font()
    say(
        f"{WORK_NAME}: {VERSION}, rules declared {DECLARED}. Font {font} "
        f"({font_file}), matplotlib {matplotlib.__version__}. Output: {out}"
    )
    src = {}
    for stage in (
        pd13.StageP13d(d, scratch),
        pc.StageP13c("p13d", d, scratch),
        p14.StageP14(d, scratch),
        cc.StageColdCurve(d, scratch),
    ):
        src.update(check_marker(d, stage))
    say(
        f"{WORK_NAME}: the outputs of p13d_bounds, p13c_final, p14_eval and "
        f"cold_start_curve equal their markers ({len(src)} files)"
    )
    m = model_part(st, d, scratch, say)
    base_files = ["p13a_model.json", "p13a_arrays.npz", "p13b_model.json"]
    for name in base_files:
        src[name] = file_sha256(d / name)
    fits = m["work"] / "fits.json"
    src[_rel(scratch, fits)] = file_sha256(fits) if fits.is_file() else None
    src[_rel(d, demo_path)] = file_sha256(demo_path)
    S = shapes_data(m)
    say(
        f"{WORK_NAME}: the panel bands equal P13c's C1 bands at every shape "
        f"quantile and state (curve gap {S['curve_gap']:.1e})"
    )
    C = cold_data(d)
    E = demo_data(m, demo_path)
    P = pipeline_data()
    say(
        f"{WORK_NAME}: the cold-start curve from its ranks equals its JSON "
        f"file (gap {C['gap']:.1e}), the demo names the final model, and "
        f"the runner's {len(P['order'])} stages fit the phases "
        f"({len(P['unused'])} not used by the final model)"
    )
    font_extra = {
        "font": font,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    }
    graph = _sha_text(json.dumps(P["upstream"], sort_keys=True))
    model_inputs = {
        k: v
        for k, v in src.items()
        if k.startswith(("p13", "p14_cold"))
        or k in base_files
        or k.endswith("fits.json")
    }
    groups = [
        (
            "pipeline",
            {"runner_graph": graph},
            lambda: _figure(
                "fig_pipeline", draw_pipeline(P, out), out, pipeline_rows(P)
            ),
        ),
        (
            "shapes",
            model_inputs,
            lambda: _figure("fig_shapes", draw_shapes(S, out), out, shapes_rows(S)),
        ),
        (
            "cold_start",
            {k: v for k, v in src.items() if k.startswith("cold_start_curve")},
            lambda: _figure("fig_cold_start", draw_cold(C, out), out, cold_rows(C)),
        ),
        (
            "explanations",
            {
                _rel(d, demo_path): src[_rel(d, demo_path)],
                "p13c_final.json": src["p13c_final.json"],
            },
            lambda: _figure(
                "fig_explanations", draw_explanations(E, out), out, explanation_rows(E)
            ),
        ),
        (
            "tables",
            {
                k: v
                for k, v in src.items()
                if k.startswith(
                    ("p14.", "cold_start_curve", "reports/p14", "reports/cold")
                )
                or k == _rel(d, demo_path)
            },
            lambda: tables_part(d, C, E, out),
        ),
    ]
    mpath = out / "manifest.json"
    old = load_json(mpath) or {}
    manifest = {
        "version": VERSION,
        "declared": DECLARED,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "font": {"name": font, "file": font_file},
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "sources": src,
        "groups": dict(old.get("groups") or {}),
        "checks": {
            "centering_gap": m["centering_gap"],
            "curve_gap": S["curve_gap"],
            "cold_gap": C["gap"],
            "unused_stages": P["unused"],
        },
    }
    prog = Progress(f"{WORK_NAME}: outputs", len(groups), say, unit=" groups")
    for name, inputs, build in groups:
        key = _group_key(inputs, font_extra)
        if not force and _complete(old.get("groups", {}).get(name), key, out):
            say(f"{WORK_NAME}: {name}: complete from an earlier run, kept")
            prog.advance(1)
            continue
        t1 = time.time()
        with matplotlib.rc_context(style(font)):
            shas, checks = build()
        manifest["groups"][name] = {"key": key, "outputs": shas, "checks": checks}
        atomic_json_write(manifest, mpath)
        say(
            f"{WORK_NAME}: {name}: {len(shas)} files written and read back "
            f"in {_clock(time.time() - t1)}"
        )
        prog.advance(1)
    prog.finish()
    text = captions(S, C, E, P, m)
    atomic_text_write(text, out / "captions.md")
    manifest["captions"] = file_sha256(out / "captions.md")
    atomic_json_write(manifest, mpath)
    say(
        f"{WORK_NAME}: done in {_clock(time.time() - t0)}. Figures, tables "
        f"and captions in {out}"
    )
    return manifest


def _figure(stem, fig, out, data):
    """Save a figure and its data file, and read both back."""
    shas = save_figure(fig, out, stem)
    checks = check_figure_files(out, stem)
    header, rows = data
    shas[f"{stem}.csv"] = write_csv(Path(out) / f"{stem}.csv", header, rows)
    checks["rows"] = len(rows)
    return shas, checks


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="The thesis figures " "and tables (module docstring)"
    )
    ap.add_argument(
        "--demo-user",
        help="the ListenBrainz user of the demo "
        "page (a folder under MODEL_DIR/"
        "demo). Default: the only one there",
    )
    ap.add_argument("--force", action="store_true", help="draw every output again")
    args = ap.parse_args(argv)
    fmt = logging.Formatter(LOG_FORMAT)
    fmt.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    # the PDF backend's font subsetting logs every table it prunes
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        with log_file(MODEL_DIR / "logs" / LOG_NAME):
            demo = demo_file(MODEL_DIR, args.demo_user)
            run(
                s13b2.StageP13b2(),
                MODEL_DIR,
                SCRATCH_DIR,
                MODEL_DIR / OUT_NAME,
                demo,
                say=log.info,
                force=args.force,
            )
    except KeyboardInterrupt:
        log.warning(
            f"{WORK_NAME}: interrupt. The complete groups stay, and "
            f"the same command draws the rest."
        )
        return 130
    except ValueError as exc:
        log.error(f"{WORK_NAME}: stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

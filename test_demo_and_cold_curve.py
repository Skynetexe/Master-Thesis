"""Smoke test of pipeline/lb_demo.py, pipeline/lb_outside.py and the
runner stage cold_start_curve
(pipeline/cold_curve.py), on the P13 chain of p13_chain.py. Every path
points into a temporary folder, and nothing touches the real model folder.

1. Units without the chain: lb_outside's P4 rules (ISRC and title keys,
   duplicates, the alias fold, the term and credit rows, the audio
   z-scores), the listen parser, the export reader (zip, JSON Lines, JSON),
   the API fetch on a fake server (pages that split a second, a rate limit,
   Ctrl+C and the resume, the saved fetch, no answer for large pages, a
   cut or broken answer, a full page inside one second, the stop after
   the last tries and its resume), the chain's tiers and the map check.
2. The chain with its markers, p14_cold_gam and p14_eval through the
   runner on synthetic inputs, as test_p14.py makes them.
3. cold_start_curve through the runner with cut points for the small world
   (2, 5, 10, 20 pairs, listeners with 20 or more): every rank of every
   listener at every point against a dense full sort written here, ItemKNN
   against a loop written here, popularity and the full-history point
   against P14, the metrics from the stored ranks, a second call that
   skips, a refused change, a stop inside the pass and its resume with
   another worker count, and the report.
4. The demo on a synthetic ListenBrainz export: the match report by tier,
   MBID source and miss, the history weights, the stored top 500 against a
   dense full sort, the reasons (D1 and S4) against a loop, the evidence,
   ItemKNN's lists against a loop, the outside list (tracks that MLHD
   lacks, built from raw links) against the catalog's own concepts of the
   tracks they copy, the audit, the data files and their read-back. A
   changed audit value and a changed map stop the run.
5. The demo without a match: seed artists, then the popularity list with
   the note "No matched history".
6. Parts 2 and 3 of the demo on the chain (pipeline/lb_demo.py, module
   docstring):
   the known artists from the history and from raw listens (artist MBID,
   and exact name only for names without an artist MBID), the filter keys,
   the fresh list against a loop with its caps, the co-listening reasons of
   ItemKNN and fresh against loops over P9b's lists, the strengths of the
   model and outside reasons against a dense group sum, the words, the
   match summary, and the refusals of the document checks. Part 4 checks
   the new-artist tracks of both model lists (the union with exact P5
   ranks, the new artists of the outside list) and the song keys of the
   known-song drop and the duplicates.
The page itself (demo/index.html) is tested by test_demo_page.py.

Usage, from the repository root:
    python3 test_demo_and_cold_curve.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import dataclasses
import http.client
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
import zipfile
import zlib
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import als
from pipeline import common
from pipeline import concepts as cn
from pipeline import config
from pipeline import evaluation as ev
from pipeline import gam
from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
from pipeline import itemknn_select as IK
from pipeline import cold_curve as CC
from pipeline import lb_demo as LB
from pipeline import lb_outside as LO
from pipeline import p13c as PC
from pipeline import p13d as D
from pipeline import p14 as P
from pipeline import p14_cold as PK
from pipeline import runner as RN
from pipeline import stage_p13b2 as s13b2

failures = []
T0 = time.time()
PARAMS = dict(config.DEFAULT_PARAMS)
FAKED = ("p6_als", "p9b_colisten", "p13b2_loss", "p5_eval_baseline", "als_phase_d")


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(msg):
    pass


def clock():
    return f"{time.time() - T0:.0f} s"


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


CAP = Capture()
common.log.addHandler(CAP)
common.log.setLevel(logging.INFO)


def raises(fn, text):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return text in str(exc)
    return False


# 1: units without the chain
check(
    LO.norm_isrc("us-ab1-23-45678") == "USAB12345678"
    and LO.norm_isrc("US1234") is None
    and LO.norm_isrc(None) is None
    and LO.title_key("My  Song - Remastered 2011") == "my song"
    and LO.title_key("My Song") == "my song",
    "1: ISRC and title keys",
)
dup = [
    {"sp_id": "b", "isrc": "X1", "name": "A", "popularity": 50, "sp_artists": ("s1",)},
    {
        "sp_id": "a",
        "isrc": "X1",
        "name": "A (other)",
        "popularity": 20,
        "sp_artists": ("s1",),
    },
    {
        "sp_id": "c",
        "isrc": "X2",
        "name": "A - Live",
        "popularity": 50,
        "sp_artists": ("s1",),
    },
    {
        "sp_id": "d",
        "isrc": None,
        "name": "B",
        "popularity": None,
        "sp_artists": ("s2",),
    },
    {"sp_id": "e", "isrc": "X3", "name": "B", "popularity": 1, "sp_artists": ("s2",)},
]
kept = [t["sp_id"] for t in LO.dedupe_tracks(dup)]
check(
    kept == ["b", "e"],
    f"1: one track per ISRC, then per title key and artists, the most "
    f"popular first, ties by id ({kept})",
)
al = LO.alias_map([("hiphop", "hip hop"), ("hiphop", "hip-hop"), ("rnb", "r&b")])
check(
    LO.fold(["hiphop", "rock", None], al) == {"hip hop", "hip-hop", "rock"}
    and LO.fold(["rnb", "r&b"], al) == {"r&b"},
    "1: the alias fold gives each canonical of a variant, as P4's LEFT " "JOIN",
)
ip, cols = LO.term_csr([{"b", "a", "zz"}, set(), {"c", "a"}], {"a": 0, "b": 1, "c": 2})
ipc, colc = LO.credit_csr([[7, 3, 3, 99], [], [5]], [3, 5, 7])
check(
    ip.tolist() == [0, 2, 2, 4]
    and cols.tolist() == [0, 1, 0, 2]
    and ipc.tolist() == [0, 2, 2, 3]
    and colc.tolist() == [0, 2, 1],
    "1: the term rows keep the vocabulary columns, the credit rows the "
    "catalog columns",
)
st_ = {
    c: {"mean": 2.0 + k, "sd": 0.5 + k}
    for k, c in enumerate(
        [
            "duration_ms",
            "tempo",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
        ]
    )
}
raw_ = {c: np.array([3.0, 0.0, np.nan]) for c in st_}
zz = LO.audio_z(raw_, st_)
want = np.array(
    [
        [(3.0 - st_[c]["mean"]) / st_[c]["sd"] for c in st_],
        [(0.0 - st_[c]["mean"]) / st_[c]["sd"] for c in st_],
        [np.nan] * 10,
    ]
)
want[1, 1] = np.nan  # tempo 0 is missing
check(
    np.array_equal(zz, want, equal_nan=True),
    "1: the audio z-scores follow P4's formula, tempo 0 and NULL as " "missing",
)

U1 = "0a0a0a0a-0000-4000-8000-000000000001"
U2 = "0a0a0a0a-0000-4000-8000-000000000002"
r1 = LB.parse_listen(
    {
        "listened_at": 100,
        "recording_msid": "m",
        "track_metadata": {
            "artist_name": "A",
            "track_name": "T",
            "additional_info": {"recording_mbid": U1.upper()},
            "mbid_mapping": {"recording_mbid": U2},
        },
    }
)
r2 = LB.parse_listen(
    {
        "listened_at": "2024-01-02T03:04:05Z",
        "track_metadata": {
            "additional_info": {"recording_mbid": "nope"},
            "mbid_mapping": {"recording_mbid": U2},
        },
    }
)
check(
    r1["mbids"] == [("submitted", U1), ("mapped", U2)]
    and r2["mbids"] == [("mapped", U2)]
    and r2["ts"] == 1704164645
    and LB.parse_listen({"track_metadata": {}}) is None,
    "1: a listen's MBIDs: the submitted one first, then the mapping, "
    "a non-UUID dropped, ISO times read",
)


def listen(ts, mbid=None, mapped=None, track="t", msid=None):
    info = {} if mbid is None else {"recording_mbid": mbid}
    mm = None if mapped is None else {"recording_mbid": mapped}
    return {
        "listened_at": int(ts),
        "recording_msid": msid or f"msid{ts}",
        "user_name": "tester",
        "track_metadata": {
            "artist_name": "Artist",
            "track_name": track,
            "release_name": "R",
            "additional_info": info,
            "mbid_mapping": mm,
        },
    }


TMP = Path(tempfile.mkdtemp(prefix="test_demo_and_cold_curve_"))
ex_listens = [listen(1000 + k, U1 if k % 2 else None) for k in range(9)]
zpath = TMP / "export.zip"
with zipfile.ZipFile(zpath, "w") as zf:
    zf.writestr("listenbrainz_tester/user.json", json.dumps({"user_name": "tester"}))
    zf.writestr(
        "listenbrainz_tester/feedback.jsonl",
        json.dumps({"recording_msid": "x", "score": 1}) + "\n",
    )
    zf.writestr(
        "listenbrainz_tester/listens/2024/1.jsonl",
        "\n".join(json.dumps(x) for x in ex_listens[:5]) + "\n",
    )
    zf.writestr(
        "listenbrainz_tester/listens/2024/2.jsonl",
        "\n".join(json.dumps(x) for x in ex_listens[4:]) + "\n",
    )
(TMP / "export.json").write_text(json.dumps(ex_listens))
(TMP / "export.jsonl").write_text("\n".join(json.dumps(x) for x in ex_listens))
(TMP / "page.json").write_text(
    json.dumps({"payload": {"count": 9, "listens": ex_listens}})
)
outs_ = [
    LB.unique_listens(LB.read_export(TMP / n))
    for n in ("export.zip", "export.json", "export.jsonl", "page.json")
]
check(
    all(len(o) == 9 and o == outs_[0] for o in outs_) and outs_[0][0]["ts"] == 1008,
    "1: the export reader gives the same 9 listens from a zip (with "
    "user.json, feedback and a repeated line), JSON, JSON Lines and an "
    "API page, newest first",
)

# the API on a fake server: equal seconds across page ends
rng0 = np.random.default_rng(21)
srv_ts = np.sort(rng0.integers(1_600_000_000, 1_600_000_400, 160))[::-1]
SRV = [listen(t, U1 if k % 3 else U2, msid=f"s{k}") for k, t in enumerate(srv_ts)]
calls = {"n": 0, "list": 0, "stop": None, "rate": 2}


def server(url):
    calls["n"] += 1
    u = urllib.parse.urlparse(url)
    if u.path.endswith("/listen-count"):
        return 200, {}, json.dumps({"payload": {"count": len(SRV)}}).encode()
    calls["list"] += 1
    if calls["stop"] is not None and calls["list"] == calls["stop"]:
        raise KeyboardInterrupt
    if calls["list"] == calls["rate"]:
        return 429, {"X-RateLimit-Reset-In": "3"}, b"slow down"
    q = urllib.parse.parse_qs(u.query)
    mx = int(q["max_ts"][0]) if "max_ts" in q else None
    page = [x for x in SRV if mx is None or x["listened_at"] < mx]
    page = page[: int(q["count"][0])]
    return (
        200,
        {"X-RateLimit-Remaining": "9"},
        json.dumps({"payload": {"listens": page, "count": len(page)}}).encode(),
    )


slept = []
cache_f = TMP / "fetch"
calls["stop"] = 4
try:
    LB.fetch_listens(
        "tester", cache_f, quiet, opener=server, sleep=slept.append, page_size=7
    )
    stopped = False
except KeyboardInterrupt:
    stopped = True
calls["stop"] = None
got_f = LB.fetch_listens(
    "tester", cache_f, quiet, opener=server, sleep=slept.append, page_size=7
)


def no_net(url):
    raise AssertionError("no request expected")


again_f = LB.fetch_listens("tester", cache_f, quiet, opener=no_net, page_size=7)
want_f = LB.unique_listens(SRV)
check(
    stopped
    and LB.unique_listens(got_f) == want_f
    and len(LB.unique_listens(again_f)) == len(SRV)
    and 4.0 in slept
    and len(got_f) == len(want_f),
    f"1: the fetch walks back through max_ts with pages that split a "
    f"second, waits out a rate limit, resumes after Ctrl+C with every "
    f"listen once ({len(want_f)}), and reuses a complete fetch",
)
calls.update(n=0, list=0, rate=-1)
fresh = LB.fetch_listens(
    "tester", cache_f, quiet, opener=server, refresh=True, page_size=50
)
check(
    LB.unique_listens(fresh) == want_f and calls["list"] > 0,
    "1: --refresh fetches the listens again",
)


# the fetch when ListenBrainz gives no answer: a fake server that closes
# the connection for pages larger than `limit` or after `die_after` pages,
# cuts the body (IncompleteRead) or breaks the JSON of chosen calls, and
# orders the listens of one second in another way at each call
def lb_fake(listens, limit=None, die_after=None, cut=(), broken=()):
    log = {"counts": [], "max_ts": []}

    def opener(url):
        u = urllib.parse.urlparse(url)
        if u.path.endswith("/listen-count"):
            return 200, {}, json.dumps({"payload": {"count": len(listens)}}).encode()
        q = urllib.parse.parse_qs(u.query)
        n, k = int(q["count"][0]), len(log["counts"]) + 1
        mx = int(q["max_ts"][0]) if "max_ts" in q else None
        log["counts"].append(n)
        log["max_ts"].append(mx)
        if k in cut:
            raise http.client.IncompleteRead(b'{"pay', 100)
        if (limit is not None and n > limit) or (
            die_after is not None and k > die_after
        ):
            raise http.client.RemoteDisconnected(
                "Remote end closed connection without response"
            )
        pool = [x for x in listens if mx is None or x["listened_at"] < mx]
        pool.sort(
            key=lambda x: (
                -x["listened_at"],
                zlib.crc32(f"{k}:{x['recording_msid']}".encode()),
            )
        )
        body = json.dumps(
            {"payload": {"listens": pool[:n], "count": len(pool[:n])}}
        ).encode()
        return 200, {}, (body[: len(body) // 2] if k in broken else body)

    return opener, log


SAVED_LIMITS = (LB.MIN_PAGE, LB.API_MAX)
LB.MIN_PAGE, LB.API_MAX = 10, 80
op_a, log_a = lb_fake(SRV, limit=20, cut={6}, broken={9})
slept_a = []
got_a = LB.fetch_listens(
    "tester", TMP / "fetch_a", quiet, opener=op_a, sleep=slept_a.append, page_size=80
)
ca = log_a["counts"]
check(
    LB.unique_listens(got_a) == want_f
    and len(got_a) == len(want_f)
    and ca[:4] == [80, 80, 40, 40]
    and set(ca[4:]) == {20}
    and slept_a == [2.0] * 4,
    f"1: a page without an answer is asked again with half the listens "
    f"and the smaller size stays, a cut or broken answer is one failed "
    f"try ({len(ca)} calls)",
)

slept_e = []
op_e, log_e = lb_fake(SRV, die_after=3)
stop_e = raises(
    lambda: LB.fetch_listens(
        "tester",
        TMP / "fetch_e",
        quiet,
        opener=op_e,
        sleep=slept_e.append,
        page_size=20,
    ),
    "The 3 saved pages stay",
)
st_e = common.load_json(TMP / "fetch_e" / "state.json")
op_e2, log_e2 = lb_fake(SRV)
got_e = LB.fetch_listens(
    "tester", TMP / "fetch_e", quiet, opener=op_e2, sleep=quiet, page_size=20
)
check(
    stop_e
    and st_e["pages"] == 3
    and not st_e["done"]
    and log_e["counts"][3:] == [20, 20] + [10] * LB.RETRIES
    and slept_e == [2.0] + [2.0**k for k in range(1, LB.RETRIES)]
    and log_e2["max_ts"][0] == st_e["max_ts"]
    and LB.unique_listens(got_e) == want_f
    and len(got_e) == len(want_f),
    "1: no answer at MIN_PAGE stops after all RETRIES tries with the "
    "pages saved, and the same call continues from them",
)

LB.MIN_PAGE, LB.API_MAX = 4, 64
CL = (
    [listen(1_700_000_100 + k, U1, msid=f"n{k}") for k in range(4, -1, -1)]
    + [listen(1_700_000_000, U1, msid=f"c{k}") for k in range(30)]
    + [listen(1_699_999_000 + k, U2, msid=f"o{k}") for k in range(19, -1, -1)]
)
op_b, log_b = lb_fake(CL)
got_b = LB.fetch_listens(
    "tester", TMP / "fetch_b", quiet, opener=op_b, sleep=quiet, page_size=8
)
op_c, log_c = lb_fake([listen(1_700_000_000, U1, msid=f"b{k}") for k in range(70)])
stop_c = raises(
    lambda: LB.fetch_listens(
        "tester", TMP / "fetch_c", quiet, opener=op_c, sleep=quiet, page_size=16
    ),
    "more than 64 listens share the second",
)
op_d, log_d = lb_fake(
    [listen(1_700_000_000, U1, msid=f"t{k}") for k in range(12)], limit=10
)
stop_d = raises(
    lambda: LB.fetch_listens(
        "tester", TMP / "fetch_d", quiet, opener=op_d, sleep=quiet, page_size=8
    ),
    "gave no answer",
)
check(
    LB.unique_listens(got_b) == LB.unique_listens(CL)
    and len(got_b) == len(CL) == 55
    and log_b["counts"][:4] == [8, 8, 16, 32]
    and set(log_b["counts"][3:]) == {32}
    and stop_c
    and log_c["counts"] == [16, 32, 64]
    and stop_d
    and log_d["counts"] == [8] + [16] * LB.RETRIES,
    "1: a full page inside one second is asked again with twice the "
    "listens (no listen of that second lost), API_MAX listens in one "
    "second stop the fetch, and a page that grew gets all RETRIES tries "
    "and no smaller size",
)
LB.MIN_PAGE, LB.API_MAX = SAVED_LIMITS

rows_c = [
    ("c1", 1, None, None, 0, None, None, 0),
    ("c1", 2, "x", None, 0, None, None, 0),
    (None, 0, "c2", None, 0, None, None, 0),
    (None, 0, None, "g", 1, "c3", "c4", 1),
    (None, 0, None, "g", 1, None, "c4", 2),
    (None, 0, None, "g", 1, None, None, 0),
    (None, 0, None, None, 0, None, None, 0),
]
tiers_c = [LB.chain_canonical(r) for r in rows_c]
check(
    tiers_c
    == [
        ("c1", "redirect", False),
        ("c1", "redirect", True),
        ("c2", "canonical", False),
        ("c3", "merged", False),
        ("c4", "merged", True),
        (None, None, False),
        (None, None, False),
    ]
    and raises(
        lambda: LB.check_map({"m": {"canonical": "a"}}, {"m": "b"}),
        "recording_canonical_map",
    )
    and LB.check_map(
        {"m": {"canonical": "a"}, "n": {"canonical": None}}, {"m": "a", "n": None}
    )
    == 2,
    "1: the chain's tiers follow BuildFeatures.py's order, two targets "
    "count as ambiguous, and the map check refuses another canonical",
)

# 2: the chain and P14
print(W.summary())
ROOT = TMP / "chain"
CHAIN = C.chain_into(ROOT)
SCR = ROOT / "scratch"
common.MODEL_DIR = CHAIN
NBOOT = {"listener": 6, "twoway": 3}


def stage():
    return C.p13_loaders(s13b2.StageP13b2())


def fake_marker(name):
    st = RN.STAGE_BY_NAME[name]
    common.write_marker(
        name,
        RN.stage_params_digest(st, PARAMS),
        {"test": name},
        {},
        source_sha=RN.stage_source_sha(st),
    )


def stages(model, scratch):
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
    out[3].n_boot, out[5].n_boot = NBOOT, {"listener": 4, "twoway": 2}
    out[3].n_verify = out[5].n_verify = 6
    return out


def execute(sts, **kw):
    by = dict(RN.STAGE_BY_NAME)
    by.update({s.name: s for s in sts})
    return RN.execute(sts, PARAMS, [], by_name=by, **kw)


for name in FAKED:
    fake_marker(name)
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
STS = stages(CHAIN, SCR)
real_decide = D.decide


def forced(scores, names_, model_arm):
    res = real_decide(scores, names_, model_arm)
    res["s2"] = dict(res["s2"], chosen=D.PILOT)
    return res


D.decide = forced
try:
    D.run(
        stage(), CHAIN, SCR / "p13d", say=quiet, n_verify=6, inputs=D.provenance(CHAIN)
    )
    rc_a = execute(STS, adopt=True)
finally:
    D.decide = real_decide
COLD = PK.StageP14Cold(CHAIN, SCR)
COLD._stage = stage

users_all = np.arange(1, len(W.u_indptr) - 1)
rest = np.setdiff1d(users_all, W.SEL)
EU = np.sort(np.random.default_rng(1404).choice(rest, size=60, replace=False))
NE = len(EU)
r4 = np.random.default_rng(1405)
y2011, y2014 = 1293840000, 1388534400
a_last = 120 * r4.integers((y2011 + 3600 * 24 * 30) // 120, y2014 // 120 - 10, NE) + 75
a_first = a_last - 120 * r4.integers(0, 40 * 24 * 30, NE)
FRAME = {
    "user": EU,
    "decile": EU % 10,
    "picks": np.array([[W.held[u][0], W.held[u][2]] for u in EU], dtype=np.int64),
    "ts_first": np.stack([a_last - 120 * r4.integers(10, 99999, NE), a_first], axis=1),
    "ts_last": np.stack([a_last, a_first + (a_last - a_first) // 2], axis=1),
    "dropped": 0,
}
tr_u, tr_i, tr_f, tr_l, tr_w = [], [], [], [], []
pos_e = {int(u): i for i, u in enumerate(EU)}
for u in users_all:
    s, e = int(W.u_indptr[u]), int(W.u_indptr[u + 1])
    n = e - s
    lis = np.maximum(1, np.round(np.expm1(W.u_data[s:e] / 1000.0))).astype(np.int64)
    if int(u) in pos_e:
        i = pos_e[int(u)]
        lo_t = a_first[i] - 120 * 3000 * 24 * 30
        f_ = r4.integers(lo_t, a_first[i] - 120, n)
        l_ = np.minimum(f_ + r4.integers(0, 120 * 24 * 900, n), a_last[i] - 120)
    else:
        f_ = r4.integers(1104537600, y2014 - 200 * 24 * 3600, n)
        l_ = f_ + r4.integers(0, 120 * 24 * 900, n)
    f_ = (f_ // 120) * 120 + 30
    if n > 3:  # equal first listens: the tie rule
        f_[1] = f_[0]
        f_[3] = f_[2]
    l_ = np.maximum((l_ // 120) * 120 + 30, f_)
    tr_u.append(np.full(n, u))
    tr_i.append(np.asarray(W.u_indices[s:e], dtype=np.int64))
    tr_f.append(f_)
    tr_l.append(l_)
    tr_w.append(lis)
TR = [np.concatenate(x) for x in (tr_u, tr_f, tr_l, tr_w)]
TR_ITEM = np.concatenate(tr_i)
TRAIN_PAIRS = np.bincount(W.u_indices, minlength=W.NI).astype(np.int64)
TRAIN = {int(u): W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]] for u in EU}
r5 = np.random.default_rng(1406)
XD = r5.normal(size=(len(W.u_indptr) - 1, 8)).astype(np.float32)
YD = r5.normal(size=(W.NI, 8)).astype(np.float32)
X6 = r5.normal(size=(len(W.u_indptr) - 1, 4)).astype(np.float32)
Y6 = r5.normal(size=(W.NI, 4)).astype(np.float32)
rows_ = P.TrainRows(W.u_indptr, W.u_indices)
g_pop = ev.global_ranks(C.POP)
STORED = {
    "p5": (
        EU.copy(),
        np.stack(
            [ev.user_ranks(g_pop, FRAME["picks"][:, p], EU, TRAIN) for p in range(2)],
            axis=1,
        ),
    ),
    "p6": (EU.copy(), als.rank_items(X6, Y6, rows_, EU, FRAME["picks"])),
}
ALS_D_SCORE = float(
    np.where(
        (rk := als.rank_items(XD, YD, rows_, W.SEL, W.SEL_PICKS)) <= 100,
        1 / np.log2(rk + 1.0),
        0,
    )
    .mean(axis=1)
    .mean()
)


class FakeInputs:
    def __init__(self):
        self.d = CHAIN

    def key(self):
        return {"fake": "test_demo_and_cold_curve"}

    def frame(self):
        return dict(FRAME), (
            FRAME["user"],
            FRAME["decile"],
            FRAME["picks"][:, 0],
            FRAME["picks"][:, 1],
            0,
        )

    def covariates(self, n_slots):
        return C.pop_centiles(), TRAIN_PAIRS.copy(), W.tier.copy()

    def popularity(self, n_slots, id_min):
        return C.POP.copy()

    def leakage(self, frame, say):
        return P.leakage_counts(*TR, frame["user"], P.anchor_of(frame))

    def lists(self, n_slots):
        return C.LISTS

    def als(self, which):
        return (XD, YD, {"d": "x"}) if which == "d" else (X6, Y6, {"p6": "y"})

    def als_reference(self, which):
        return ALS_D_SCORE if which == "d" else None

    def stored_ranks(self, which):
        return STORED[which]


P.EVAL_CHUNK, P.EVAL_BLOCK, P.PROBE_USERS = 64, 7, 3
P.N_PERM = 500
EVS = P.StageP14(CHAIN, SCR)
EVS._stage = stage
EVS._io = lambda: FakeInputs()
EVS.workers = 1
rc = execute(STS + [COLD, EVS])
check(
    rc_a == 0
    and rc == 0
    and common.marker_path("p14_eval").exists()
    and common.marker_path("p14_cold_gam").exists(),
    f"2: the chain, p14_cold_gam and p14_eval run through the runner " f"({clock()})",
)
base = stage()._base_context()
G14 = P.load_models(CHAIN, base)
terms = G14["terms"]
tables = G14["tables"]
offs = P.offsets_of(terms)
Z14 = dict(np.load(CHAIN / P.OUT_FILES[1]))


# Dense helpers written here


def term_value(t, tab, v, h):
    K = t.K
    kn = t.knot_array
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t.zero_state:
        val = np.where(h & (v == 0.0), tab[K + gam.ZERO], val)
    return np.where(h, val, tab[K + gam.MISS] if t.missing_state else 0.0)


def dense_terms(full, T):
    return np.stack(
        [
            term_value(t, T[o : o + t.size], full[t.source][0], full[t.source][1])
            for t, o in zip(terms, offs)
        ]
    )


def dense_score(full, T):
    s = np.zeros(len(full[terms[0].source][0]))
    for v in dense_terms(full, T):
        s += v
    return s


def p5_rank(s, t):
    return 1 + int(np.count_nonzero(s > s[t])) + int(np.count_nonzero(s[:t] == s[t]))


def knn_score(items, weights, kinds):
    w_ = np.zeros(W.NI)
    w_[items] = weights
    s = np.zeros(W.NI)
    for kk in kinds:
        ip_, nb, cs = C.LISTS[kk]
        for i in range(W.NI):
            a, b = ip_[i], ip_[i + 1]
            s[i] += float(np.dot(cs[a:b], w_[nb[a:b]]))
    return s


# 3: the cold-start curve
CC.POINTS, CC.MIN_PAIRS = (2, 5, 10, 20), 20
CC.CHUNK, CC.BLOCK, CC.PROBE_USERS = 64, 5, 3
hist_len = np.diff(W.u_indptr)[EU]
CU = EU[hist_len >= CC.MIN_PAIRS]


class FakeCurveInputs:
    """Two train files: each listener's pairs split between them, each
    file giving its own first pairs and count, as the real query does."""

    def key(self):
        return {"fake": "curve"}

    def first_pairs(self, users, keep, say):
        split = np.random.default_rng(7).random(len(TR[0])) < 0.5
        out_u, out_i, out_t, counts = [], [], [], {}
        for part in (split, ~split):
            m = part & np.isin(TR[0], users)
            u, i, t = TR[0][m], TR_ITEM[m], TR[1][m]
            o = np.lexsort((i, t, u))
            u, i, t = u[o], i[o], t[o]
            for x in np.unique(u):
                a, b = np.searchsorted(u, x), np.searchsorted(u, x, "right")
                counts[int(x)] = counts.get(int(x), 0) + int(b - a)
                out_u.append(u[a : min(b, a + keep)])
                out_i.append(i[a : min(b, a + keep)])
                out_t.append(t[a : min(b, a + keep)])
        return (
            (np.concatenate(out_u), np.concatenate(out_i), np.concatenate(out_t)),
            counts,
        )


CURVE = CC.StageColdCurve(CHAIN, SCR)
CURVE._stage = stage
CURVE._io = lambda: FakeCurveInputs()
CURVE.workers = 2
CAP.lines.clear()
rc = execute(STS + [COLD, EVS, CURVE])
cdoc = json.loads((CHAIN / CC.OUT_FILES[0]).read_text())
cz = dict(np.load(CHAIN / CC.OUT_FILES[1]))
check(
    rc == 0
    and common.marker_path("cold_start_curve").exists()
    and cdoc["data"]["listeners"] == len(CU) > 0
    and np.array_equal(cz["user_id"], CU)
    and cdoc["check"]["listeners"] == min(6, len(CU))
    and cdoc["check"]["full_history_equals_p14"]
    and any("probe listener" in ln for ln in CAP.lines)
    and any("2 workers" in ln for ln in CAP.lines),
    f"3: cold_start_curve runs through the runner with 2 workers on "
    f"{len(CU)} listeners, its dense and full-history checks pass "
    f"({clock()})",
)

# every rank by a dense full sort written here
first_true = {}
for u in CU:
    m = TR[0] == u
    it, ts = TR_ITEM[m], TR[1][m]
    first_true[int(u)] = it[np.lexsort((it, ts))]
ok_g = ok_k = True
picks_c = cz["picks"]
for k, u in enumerate(CU):
    items_all = W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]]
    cand = np.ones(W.NI, dtype=bool)
    cand[0] = False
    cand[items_all] = False
    ok_g &= np.array_equal(
        cz["first_items"][cz["first_indptr"][k] : cz["first_indptr"][k + 1]],
        first_true[int(u)][: max(CC.POINTS)],
    )
    for j, N in enumerate(CC.POINTS):
        its = np.sort(first_true[int(u)][:N])
        w_ = np.full(len(its), CC.W_ONE)
        state = cn.build_user_state(its, w_, W.SPACE)
        full = cn.full_catalog(state, W.SPACE)
        for arm, row in (("gam", 0), ("cold_gam", 2)):
            sc = np.where(cand, dense_score(full, tables[row]), -np.inf)
            for p in range(2):
                want_r = min(p5_rank(sc, int(picks_c[k, p])), 101)
                ok_g &= int(cz[f"rank_{arm}"][k, j, p]) == want_r
        sk = knn_score(its, w_, ("same_artist", "other_artists"))
        sk[~cand] = -np.inf
        for p in range(2):
            t = int(picks_c[k, p])
            lo = 1 + int(np.count_nonzero(sk > sk[t] + 1e-12))
            hi = int(np.count_nonzero(sk >= sk[t] - 1e-12))
            ok_k &= lo <= int(cz["rank_knn_both"][k, j, p]) <= hi
check(
    ok_g and ok_k,
    "3: the first pairs follow (first_ts, item) over two files, and every "
    "rank of both GAM tables at every point equals a dense full sort "
    "(cut at 101), ItemKNN a loop written here",
)
pos14 = np.searchsorted(Z14["user_id"], CU)
ok_m = True
for arm in CC.ARMS:
    full_r = Z14[f"rank_{arm}"][pos14].astype(float)
    nd_full = np.where(full_r <= 100, 1 / np.log2(full_r + 1), 0).mean(1)
    rows_a = cdoc["results"]["arms"][arm]
    ok_m &= abs(rows_a[-1]["ndcg_at_100"]["mean"] - nd_full.mean()) < 1e-15
    for j in range(len(CC.POINTS)):
        r = full_r if arm == "popularity" else cz[f"rank_{arm}"][:, j, :].astype(float)
        nd = np.where(r <= 100, 1 / np.log2(r + 1), 0).mean(1)
        rc_ = (r <= 100).mean(1)
        se = nd.std(ddof=1) / math.sqrt(len(nd))
        ok_m &= (
            abs(rows_a[j]["ndcg_at_100"]["mean"] - nd.mean()) < 1e-15
            and abs(rows_a[j]["recall_at_100"]["mean"] - rc_.mean()) < 1e-15
            and abs(
                rows_a[j]["ndcg_at_100"]["lo"] - (nd.mean() - 1.959963984540054 * se)
            )
            < 1e-12
        )
check(
    ok_m and [r["n"] for r in cdoc["results"]["arms"]["gam"]] == [2, 5, 10, 20, "full"],
    "3: the metrics follow from the stored ranks with normal intervals, "
    "popularity is P14's at every point, and the last point is P14's",
)
CAP.lines.clear()
check(
    execute(STS + [COLD, EVS, CURVE]) == 0
    and any("skip cold_start_curve" in ln for ln in CAP.lines),
    "3: a second runner call skips cold_start_curve",
)
saved = (CHAIN / CC.OUT_FILES[0]).read_bytes()
bad = json.loads(saved)
bad["results"]["arms"]["gam"][0]["ndcg_at_100"]["mean"] += 1e-6
(CHAIN / CC.OUT_FILES[0]).write_text(json.dumps(bad))
refused = raises(CURVE.check, "results")
(CHAIN / CC.OUT_FILES[0]).write_bytes(saved)
CURVE.check()
check(refused, "3: the read-back refuses a changed result")
first_c = {k: v.copy() for k, v in cz.items()}
real_one = CC._curve_one
cnt = {"n": 0}


def stopping(c, k):
    cnt["n"] += 1
    if cnt["n"] == 9:
        raise KeyboardInterrupt
    return real_one(c, k)


CC._curve_one = stopping
try:
    CC.run(
        stage(),
        FakeCurveInputs(),
        CHAIN,
        SCR / "curve_resume",
        say=quiet,
        n_verify=3,
        inputs=CURVE.provenance(),
        workers=1,
    )
    stopped = False
except KeyboardInterrupt:
    stopped = True
finally:
    CC._curve_one = real_one
CAP.lines.clear()
CC.run(
    stage(),
    FakeCurveInputs(),
    CHAIN,
    SCR / "curve_resume",
    say=common.log.info,
    n_verify=3,
    inputs=CURVE.provenance(),
    workers=None,
)
cz2 = dict(np.load(CHAIN / CC.OUT_FILES[1]))
check(
    stopped
    and all(np.array_equal(first_c[k], cz2[k]) for k in first_c)
    and any("pass resume" in ln for ln in CAP.lines)
    and any("first pairs are saved" in ln for ln in CAP.lines),
    "3: a run stopped inside the pass resumes with 1 worker and gives "
    "the arrays of the run with 2 workers",
)
rep = (CHAIN / "reports" / CC.REPORT_NAME).read_text()
check(
    ";" not in rep
    and "## NDCG@100" in rep
    and "## R@100" in rep
    and "full history (P14)" in rep,
    "3: the report has both tables and no semicolon",
)

# 4: the demo
TAGS = [f"t{c}" for c in range(20)]
GENRES = [f"g{c}" for c in range(8)]
SRC2 = dataclasses.replace(
    W.SRC, term_names={"tags_direct": TAGS, "tags_artist": TAGS, "genres": GENRES}
)
ID_OF_COL = np.asarray(W.SRC.credit_ids["artists"], dtype=np.int64)


def demo_stage():
    st = stage()
    st._sources = lambda n: SRC2
    return st


def mb(kind, i):
    return f"{kind:08x}-0000-4000-8000-{int(i):012x}"


CAN = {i: mb(1, i) for i in range(1, W.NI)}  # canonical MBIDs
RED = {mb(2, i): CAN[i] for i in range(1, W.NI)}  # LB redirects
MERGE = {mb(3, i): mb(4, i) for i in range(1, W.NI)}  # old gid -> new gid
MERGE_RED = {mb(4, i): CAN[i] for i in range(1, W.NI) if i % 2}
NEW_CANON = {mb(4, i): i for i in range(1, W.NI) if i % 2 == 0}
OUTSIDE_CANON = mb(9, 1)  # canonical, no item
NO_CANON = mb(8, 1)
REC_GID_MLHD = mb(5, 5)  # an ISRC recording in MLHD
U0 = int(users_all[np.argmax(np.diff(W.u_indptr)[1:])])  # longest history
H0 = np.asarray(W.u_indices[W.u_indptr[U0] : W.u_indptr[U0 + 1]], dtype=np.int64)
HIST = np.sort(H0[:30])


def resolve_fake(mbids, mapped_bad=False):
    out, mapped = {}, {}
    item_of = {c: i for i, c in CAN.items()}
    item_of_new = NEW_CANON
    for m in mbids:
        c1 = RED.get(m)
        c2 = m if (m in item_of or m == OUTSIDE_CANON) else None
        g = MERGE.get(m)
        if m == REC_GID_MLHD:
            c2 = CAN[5]
        c3a = g if (g is not None and g in item_of_new) else None
        c3b = MERGE_RED.get(g) if g is not None else None
        row = (
            c1,
            int(c1 is not None),
            c2,
            g,
            int(g is not None),
            c3a,
            c3b,
            int(c3b is not None),
        )
        canon, tier, amb = LB.chain_canonical(row)
        it = item_of.get(canon, 0) if canon else 0
        if canon in item_of_new:
            it = item_of_new[canon]
        out[m] = {"canonical": canon, "tier": tier, "item": int(it), "ambiguous": amb}
        if m in item_of or m in RED:
            mapped[m] = canon
    if mapped_bad and mapped:
        k = sorted(mapped)[0]
        mapped[k] = "ffffffff-0000-4000-8000-000000000000"
    return out, mapped


# synthetic outside tracks: copies of world items that share an artist
# with the listener's history
ind_a, col_a, _ = W.SRC.credits["artists"]
hist_cols = set(
    col_a[np.concatenate([np.arange(ind_a[i], ind_a[i + 1]) for i in HIST])].tolist()
)
OUT_ITEMS = [
    i
    for i in range(1, W.NI)
    if i not in set(H0.tolist())
    and set(col_a[ind_a[i] : ind_a[i + 1]].tolist()) & hist_cols
][:18]


def tdir(i):
    m = W.SPACE.term_mats["tags_direct"]
    return [TAGS[c] for c in m.indices[m.indptr[i] : m.indptr[i + 1]]]


def trows(name, i):
    m = W.SPACE.term_mats[name]
    return [int(c) for c in m.indices[m.indptr[i] : m.indptr[i + 1]]]


SP = []  # [rowid, id, name, isrc, popularity, duration, album, date]
for k, i in enumerate(OUT_ITEMS):
    SP.append(
        [
            1000 + i,
            f"sp{i:05d}",
            f"Song {i}",
            f"ZZ{i:010d}",
            50,
            1,
            "Album",
            "2016-05-0" + str(1 + k % 9),
            i,
        ]
    )
dup_i = OUT_ITEMS[0]
SP.append(
    [
        1900,
        "spdup1",
        f"Song {dup_i} (single)",
        f"ZZ{dup_i:010d}",
        10,
        1,
        "Single",
        "2017",
        dup_i,
    ]
)  # same ISRC
SP.append(
    [
        1901,
        "spdup2",
        f"Song {dup_i} - Remastered",
        "ZZ9999999999",
        5,
        1,
        "Best of",
        "2019",
        dup_i,
    ]
)  # same title
SP.append(
    [1902, "spbridge", "Bridged", "ZZ8888888888", 90, 1, "A", "2015", OUT_ITEMS[1]]
)  # a bridge
SP.append(
    [
        1903,
        "spmlhd",
        "In MLHD by ISRC",
        "ZZ7777777777",
        90,
        1,
        "A",
        "2015",
        OUT_ITEMS[2],
    ]
)  # ISRC hit
K0, K1, K2 = OUT_ITEMS[3], OUT_ITEMS[4], OUT_ITEMS[5]
SP.append(
    [
        1904,
        "spknown1",
        f"Track {K0} \u2013 x - Remastered 2011",
        "ZZ6666666661",
        95,
        1,
        "Hits",
        "2011",
        K0,
    ]
)  # not an MLHD title
SP.append(
    [
        1905,
        "spknown2",
        f"Track {K0} - Remastered 2011",
        "ZZ6666666662",
        95,
        1,
        "Hits",
        "2011",
        K0,
    ]
)  # another version of an MLHD song
SP.append(
    [1906, "spknown3", "Raw Song! - Live", "ZZ6666666663", 95, 1, "Live", "2012", K1]
)  # a raw listen's song, by name
SP.append(
    [1907, "spknown4", "Mbid\u2019s Song", "ZZ6666666664", 95, 1, "Single", "2013", K2]
)  # a raw listen's song, by MBID
SP_BY_ROW = {r[0]: r for r in SP}
SP_ARTIST_ROW = {r[0]: 5000 + r[8] for r in SP}  # one Spotify artist
EXTRA_MB = {i: 100000 + i for i in OUT_ITEMS}  # holds artist tags


class FakeDemoInputs:
    def __init__(self, bad_audit=False, bad_map=False):
        self.bad_audit, self.bad_map = bad_audit, bad_map
        self.calls = []

    def resolve(self, mbids):
        self.calls.append(("resolve", len(mbids)))
        return resolve_fake(mbids, self.bad_map)

    def track_names(self, items):
        return {
            int(i): (f"Artist of {int(i)}", f"Track {int(i)}", f"Album {int(i)}")
            for i in items
        }

    def artist_names(self, ids):
        return {int(a): f"Artist {int(a)}" for a in ids}

    def popularity(self, n_slots, id_min):
        return C.POP.copy()

    def seed_artists(self, specs):
        # the SQL's rule: an artist MBID, or the exact name, case ignored
        self.calls.append(("seed_artists", len(specs)))
        ms = {s for s in specs if LB.as_mbid(s)}
        ns = {s.strip().lower() for s in specs if not LB.as_mbid(s)}
        out = []
        for a in ID_OF_COL:
            if mb(6, a) in ms or f"artist {int(a)}" in ns:
                out.append((int(a), mb(6, a), f"Artist {int(a)}", None))
        return out

    def spotify_tracks_of(self, mb_ids):
        mbs = set(int(x) for x in mb_ids)
        rows = []
        for r in SP:
            i = r[8]
            cr = set(ID_OF_COL[col_a[ind_a[i] : ind_a[i + 1]]].tolist())
            hit = sorted(cr & mbs)
            if hit:
                rows.append(r[:8] + [hit[0]])
        return rows

    def spotify_raw(self, rowids, ids=()):
        out = {
            k: []
            for k in (
                "tracks",
                "track_artists",
                "artists",
                "sp2mb",
                "mb2sp",
                "genres",
                "artist_tags",
                "isrc_rec",
                "rec_gid",
                "rec_tags",
                "audio",
                "bridged",
            )
        }
        self.calls.append(("raw", len(rowids), len(ids)))
        rowids = list(rowids) + [20000 + int(x[3:]) for x in ids]
        for rid in rowids:
            rid = int(rid)
            if rid >= 20000:  # the audit's catalog tracks
                i = rid - 20000
                out["tracks"].append([rid, f"spc{i}", None])
                a = W.audio[i].copy()
                if self.bad_audit and i == AUDIT_FIRST:
                    a[0] += 1e-9
                out["audio"].append(
                    [f"spc{i}"] + [None if np.isnan(x) else float(x) for x in a]
                )
                continue
            r = SP_BY_ROW[rid]
            i = r[8]
            arow = SP_ARTIST_ROW[rid]
            out["tracks"].append([rid, r[1], r[3]])
            out["track_artists"].append([rid, arow])
            out["artists"].append([arow, f"spa{arow}", f"Spotify artist {i}"])
            cr = ID_OF_COL[col_a[ind_a[i] : ind_a[i + 1]]].tolist()
            for m in cr + [EXTRA_MB[i]]:
                out["sp2mb"].append([f"spa{arow}", int(m)])
            out["mb2sp"].append([EXTRA_MB[i], f"spa{arow}"])
            gs = [GENRES[c] for c in trows("genres", i)]
            out["genres"] += [[arow, g] for g in gs]
            at = [TAGS[c] for c in trows("tags_artist", i)]
            at = ["t3_variant" if t == "t3" else t for t in at]
            out["artist_tags"] += [[EXTRA_MB[i], t] for t in at]
            isrc = LO.norm_isrc(r[3])
            rec = f"rec{rid}"
            out["isrc_rec"].append([isrc, rec])
            out["rec_gid"].append([rec, REC_GID_MLHD if rid == 1903 else mb(7, rid)])
            dt_ = ["t5_variant" if t == "t5" else t for t in tdir(i)]
            out["rec_tags"] += [[rec, t] for t in dt_]
            a = W.audio[i]
            out["audio"].append([r[1]] + [None if np.isnan(x) else float(x) for x in a])
            if rid == 1902:
                out["bridged"].append(r[1])
        return out

    def catalog_tracks(self, items):
        return {
            int(i): f"spc{int(i)}" for i in items if W.SPACE.audio_mask[int(i)].any()
        }

    def audio_stats(self):
        return {c: {"mean": 0.0, "sd": 1.0} for c in LB.s4.CONT_COLS}

    def tag_aliases(self):
        return [("t3_variant", "t3"), ("t5_variant", "t5")]


AUDIT_FIRST = int(HIST[W.SPACE.audio_mask[HIST].any(axis=1)][0])

# the export: plays per item, MBIDs of every tier and source, and misses
plays_want = {int(i): 1 + int(i) % 4 for i in HIST}
LS, ts_ = [], 1_700_000_000
tier_want = {"redirect": 0, "canonical": 0, "merged": 0}
src_want = {"submitted": 0, "mapped": 0}
for i in HIST:
    i = int(i)
    for p in range(plays_want[i]):
        ts_ += 37
        v = i % 5
        if v == 0:
            LS.append(listen(ts_, CAN[i]))
            tier_want["canonical"] += 1
            src_want["submitted"] += 1
        elif v == 1:
            LS.append(listen(ts_, mb(2, i)))
            tier_want["redirect"] += 1
            src_want["submitted"] += 1
        elif v == 2:
            LS.append(listen(ts_, mb(3, i)))
            tier_want["merged"] += 1
            src_want["submitted"] += 1
        elif v == 3:
            LS.append(listen(ts_, "not-a-uuid", CAN[i]))
            tier_want["canonical"] += 1
            src_want["mapped"] += 1
        else:
            LS.append(listen(ts_, OUTSIDE_CANON, mb(2, i)))
            tier_want["redirect"] += 1
            src_want["mapped"] += 1
for k in range(3):
    LS.append(listen(ts_ + 100 + k, None))
for k in range(2):
    LS.append(listen(ts_ + 200 + k, NO_CANON))
    LS.append(listen(ts_ + 300 + k, OUTSIDE_CANON))
LS.append(dict(LS[0]))  # a repeated line

# part 2: raw listens that name artists outside the history (one by an
# artist MBID, one by its exact name) and the songs of two outside tracks
items_h0 = np.asarray(sorted(plays_want), dtype=np.int64)
w_h0 = (
    np.round(1000 * np.log1p(np.array([plays_want[int(i)] for i in items_h0], float)))
    / 1000
)
hist_cols0 = set(
    col_a[
        np.concatenate([np.arange(ind_a[i], ind_a[i + 1]) for i in items_h0])
    ].tolist()
)
so0 = knn_score(items_h0, w_h0, ("other_artists",))
fresh0 = [
    int(i)
    for i in np.lexsort((np.arange(W.NI), -so0))
    if so0[i] > 0
    and i != 0
    and int(i) not in set(items_h0.tolist())
    and ind_a[i + 1] > ind_a[i]
    and not set(col_a[ind_a[i] : ind_a[i + 1]].tolist()) & hist_cols0
]
KB = int(ID_OF_COL[col_a[ind_a[fresh0[0]]]])  # known by artist MBID
KC = next(
    int(ID_OF_COL[c])
    for i in fresh0
    for c in col_a[ind_a[i] : ind_a[i + 1]]
    if int(ID_OF_COL[c]) != KB
)  # known by its name
A2 = int(ID_OF_COL[col_a[ind_a[K2]]])  # an artist of K2


def raw_listen(ts, artist, track, artist_mbids=()):
    info = {"artist_mbids": list(artist_mbids)} if artist_mbids else {}
    return {
        "listened_at": int(ts),
        "recording_msid": f"raw{ts}",
        "user_name": "tester",
        "track_metadata": {
            "artist_name": artist,
            "track_name": track,
            "release_name": "R",
            "additional_info": info,
            "mbid_mapping": None,
        },
    }


LS.append(raw_listen(ts_ + 400, f"Spotify artist {K1}", "Raw Song"))
LS.append(raw_listen(ts_ + 401, "Someone", "Mbid's Song", [mb(6, A2)]))
LS.append(raw_listen(ts_ + 402, "Someone else", "x", [mb(6, KB)]))
LS.append(raw_listen(ts_ + 403, f"Artist {KC}", "y"))
# part 3, 3: a raw artist named once with an artist MBID (one that names
# no artist) and once without one: its name must not resolve by name
KD = next(
    int(ID_OF_COL[c])
    for i in fresh0
    for c in col_a[ind_a[i] : ind_a[i + 1]]
    if int(ID_OF_COL[c]) not in (KB, KC)
)
LS.append(raw_listen(ts_ + 404, f"Artist {KD}", "z1", [mb(6, 999999)]))
LS.append(raw_listen(ts_ + 405, f"Artist {KD}", "z2"))
N_RAW_PLANTED = 6
HIST_IDS = {int(ID_OF_COL[c]) for c in hist_cols0}
KNOWN = HIST_IDS | {KB, KC, A2}  # not KD (part 3, 3)


def ids_of(slot):
    return sorted({int(ID_OF_COL[c]) for c in col_a[ind_a[slot] : ind_a[slot + 1]]})


# the dense routes of the folded-in listener, before the run, because the
# new artists of the outside list (part 3, 1) come from them
items_h = np.asarray(sorted(plays_want), dtype=np.int64)
w_h = (
    np.round(1000 * np.log1p(np.array([plays_want[int(i)] for i in items_h], float)))
    / 1000
)
state_h = cn.build_user_state(items_h, w_h, W.SPACE)
full_h = cn.full_catalog(state_h, W.SPACE)
cand_h = np.ones(W.NI, dtype=bool)
cand_h[0] = False
cand_h[items_h] = False
sc_h = np.where(cand_h, dense_score(full_h, tables[0]), -np.inf)
scc_h = np.where(cand_h, dense_score(full_h, tables[2]), -np.inf)
NEW_CAND = [
    i
    for i in range(1, W.NI)
    if cand_h[i] and ind_a[i + 1] > ind_a[i] and not set(ids_of(i)) & KNOWN
]
order_c = sorted(NEW_CAND, key=lambda i: (-scc_h[i], i))
NEW_IDS = []
for i in order_c:
    for a in ids_of(i):
        if a not in NEW_IDS and len(NEW_IDS) < LB.NEW_ARTISTS:
            NEW_IDS.append(a)
# Spotify tracks of new artists (copies of catalog tracks by new artists),
# a new song in two versions, and versions of an MLHD title
NEW_OUT_ITEMS = [i for i in order_c if set(ids_of(i)) & set(NEW_IDS)][:12]
for i in NEW_OUT_ITEMS:
    SP.append(
        [
            3000 + i,
            f"spn{i:05d}",
            f"Fresh Song {i}",
            f"NN{i:010d}",
            40,
            1,
            "New",
            "2015",
            i,
        ]
    )
SP.append([1908, "spver1", "New Song", "ZZ6666666665", 40, 1, "S", "2016", K1])
SP.append(
    [1909, "spver2", "New Song (Live)", "ZZ6666666666", 30, 1, "S", "2016", K1]
)  # one per song key: dropped
SP.append(
    [
        1910,
        "spver3",
        f"Track {K0} (2011 Remaster)",
        "ZZ6666666667",
        95,
        1,
        "Hits",
        "2011",
        K0,
    ]
)  # a version of an MLHD song
SP.append(
    [
        1911,
        "spver4",
        f'Wip ("Track {K0}" Rough Mix)',
        "ZZ6666666668",
        95,
        1,
        "Box",
        "2012",
        K0,
    ]
)  # a working title quotes it
SP.append(
    [
        1912,
        "spver5",
        f"Track {K0} (Take Me Home)",
        "ZZ6666666669",
        95,
        1,
        "Hits",
        "2011",
        K0,
    ]
)  # no version word: kept
SP_BY_ROW = {r[0]: r for r in SP}
SP_ARTIST_ROW = {r[0]: 5000 + r[8] for r in SP}
EXTRA_MB.update({i: 100000 + i for i in NEW_OUT_ITEMS})
SPID_ITEM = {r[1]: r[8] for r in SP}
NEW_SP = {f"spn{i:05d}" for i in NEW_OUT_ITEMS}
(TMP / "lb_export.jsonl").write_text("\n".join(json.dumps(x) for x in LS))
RAW = list(LB.read_export(TMP / "lb_export.jsonl"))

demo_out = TMP / "demo"
CAP.lines.clear()
FDI = FakeDemoInputs()
doc = LB.run(
    demo_stage(),
    FDI,
    CHAIN,
    demo_out,
    TMP / "demo_cache",
    raw_listens=RAW,
    source={"kind": "export", "export": "lb_export.jsonl"},
    say=common.log.info,
)
m = doc["match"]
n_list = len(LS) - 1
matched_want = sum(plays_want.values())
check(
    m["listens"] == n_list
    and m["matched"] == matched_want
    and m["with_mbid"] == n_list - 3 - N_RAW_PLANTED
    and m["items"] == len(HIST)
    and m["tiers"] == tier_want
    and m["sources"] == src_want
    and m["misses"]
    == {
        "no MBID": 3 + N_RAW_PLANTED,
        "no canonical recording": 2,
        "canonical recording outside MLHD": 2,
    }
    and m["recordings"]
    == len(
        {LB.parse_listen(x)["mbids"][0][1] for x in LS if LB.parse_listen(x)["mbids"]}
    )
    and doc["checks"]["map_mbids_equal"] > 0,
    f"4: the match report: {m['matched']} of {m['listens']} listens, the "
    f"tiers, the MBID sources (a second MBID matches when the first "
    f"fails) and the misses follow the export",
)
top_h = np.lexsort((np.arange(W.NI), -sc_h))[: LB.STORE_N]
top_new_h = sorted(NEW_CAND, key=lambda i: (-sc_h[i], i))[: LB.STORE_N]
UNION = np.array(
    sorted(set(top_h.tolist()) | set(top_new_h), key=lambda i: (-sc_h[i], i)),
    dtype=np.int64,
)
ng = doc["new_artists"]["gam"]
check(
    [r["item"] for r in doc["gam"]] == UNION.tolist()
    and [r["rank"] for r in doc["gam"]] == [p5_rank(sc_h, int(t)) for t in UNION]
    and all(r["score"] == float(sc_h[t]) for r, t in zip(doc["gam"], UNION))
    and len(UNION) > LB.STORE_N
    and UNION[: LB.STORE_N].tolist() == top_h.tolist()
    and ng
    == {
        "candidates": len(NEW_CAND),
        "stored": len(top_new_h),
        "first_rank": p5_rank(sc_h, int(top_new_h[0])),
    }
    and doc["history"]["items"] == len(HIST)
    and doc["history"]["plays"] == matched_want
    and doc["checks"]["dense_slots_equal"] > 0,
    f"4: the model list stores the top {LB.STORE_N} of a dense full sort "
    f"and the best {len(top_new_h)} of {len(NEW_CAND)} new-artist "
    f"candidates ({len(UNION)} tracks), each with its P5 rank (the first "
    f"new artist at rank {ng['first_rank']}), exactly (history weights "
    f"round(1000 log1p(plays)) / 1000)",
)

# the reasons by a loop written here
groups = PC.reason_groups(G14["M"]["kept"])
jx = {t.name: j for j, t in enumerate(terms)}
TOPR = UNION


def reason_vals(T):
    vt = dense_terms(full_h, T)
    val = np.zeros((len(groups), len(TOPR)))
    cnd = np.zeros((len(groups), len(TOPR)), dtype=bool)
    for g, (_, names) in enumerate(groups):
        for nme in names:
            j = jx[nme]
            t = terms[j]
            v, h = full_h[t.source]
            val[g] += vt[j][TOPR] - vt[j][cand_h].mean()
            cnd[g] |= h[TOPR] & ~(t.zero_state & (v[TOPR] == 0.0))
    return val, cnd


def top3(val, cnd):
    r = np.zeros(val.shape, dtype=bool)
    for i in range(val.shape[1]):
        idx = [g for g in range(val.shape[0]) if cnd[g, i]]
        idx.sort(key=lambda g: (-val[g, i], g))
        for g in idx[:3]:
            r[g, i] = True
    return r


M14 = G14["M"]
v0, c0 = reason_vals(M14["tables"])
pos_ = np.zeros(v0.shape, dtype=np.int64)
tp_ = np.zeros(v0.shape, dtype=np.int64)
for th in G14["boot"]["listener"]:
    vb, _ = reason_vals(M14["lay"].tables(th))
    pos_ += c0 & (vb > 0)
    tp_ += c0 & top3(vb, c0)
B = len(G14["boot"]["listener"])
d1 = (
    c0
    & (v0 > 0)
    & (pos_ >= math.ceil(0.95 * B - 1e-9))
    & (tp_ >= math.ceil(0.8 * B - 1e-9))
)
allpos = c0.copy()
tm_ = np.zeros(v0.shape, dtype=np.int64)
for T_ in G14["member_tables"]:
    vm, _ = reason_vals(T_)
    allpos &= vm > 0
    tm_ += c0 & top3(vm, c0)
s4 = allpos & (tm_ >= math.ceil(0.8 * len(G14["member_tables"]) - 1e-9))
shown_w = d1 & s4
near = np.abs(v0) < 1e-9
ok_r = True
n_shown = 0
for r, row in enumerate(doc["gam"]):
    got = {x["reason"]: x["value"] for x in row["reasons"]}
    want_s = {groups[g][0] for g in range(len(groups)) if shown_w[g, r]}
    amb = {groups[g][0] for g in range(len(groups)) if near[g, r]}
    ok_r &= (set(got) | amb) == (want_s | amb)
    for g, (gname, _) in enumerate(groups):
        if gname in got:
            n_shown += 1
            ok_r &= abs(got[gname] - v0[g, r]) < 1e-9
    for x in row["reasons"]:
        ok_r &= x["odds"] is None or abs(
            x["odds"] - math.exp(x["value"])
        ) <= 1e-12 * math.exp(x["value"])
check(
    ok_r and n_shown > 0,
    f"4: the shown reasons (D1 over {B} replicates, S4 over "
    f"{len(G14['member_tables'])} members) and their values equal a loop "
    f"written here ({n_shown} shown on the top {len(TOPR)})",
)
ok_e = True
for r, row in enumerate(doc["gam"][:10]):
    evd = cn.evidence(state_h, W.SPACE, int(row["item"]))
    for x in row["reasons"]:
        if x["reason"] in ("artist", "audio_fit", "popularity"):
            continue
        e = evd[x["reason"]]
        names_w = [
            TAGS[t["id"]] if e["unit"] != "genres" else GENRES[t["id"]]
            for t in e["terms"]
        ]
        ok_e &= [u["name"] for u in x["evidence"]["units"]] == names_w[
            : LB.EVIDENCE_K
        ] and x["evidence"]["count"] == len(names_w)
    art = [x for x in row["reasons"] if x["reason"] == "artist"]
    if art:
        e = evd["artist_affinity"]
        ok_e &= (
            art[0]["credit"] == int(ID_OF_COL[e["credit"]])
            and art[0]["credit_tracks"] == len(e["terms"])
            and art[0]["credit_name"] == f"Artist {art[0]['credit']}"
        )
check(
    ok_e,
    "4: the evidence names the matched tags and genres from "
    "concepts.evidence, and the artist reason its credit",
)
ok_k = True
for kind, kinds in (
    ("same_artist", ("same_artist",)),
    ("both", ("same_artist", "other_artists")),
):
    sk = knn_score(items_h, w_h, kinds)
    sk[~cand_h] = -np.inf
    pos_c = np.flatnonzero(np.isfinite(sk) & (sk > 0))
    o = pos_c[np.lexsort((pos_c, -sk[pos_c]))][: LB.STORE_N]
    got = doc["itemknn"][kind]
    ok_k &= [r["item"] for r in got] == o.tolist()
    ok_k &= all(
        abs(r["score"] - sk[r["item"]]) <= 1e-12 * max(1, sk[r["item"]]) for r in got
    )
check(
    ok_k,
    "4: both ItemKNN lists equal a loop written here (positive "
    "scores, ties by item id)",
)

# the outside list against the catalog's concepts of the copied items
o = doc["outside"]
kept_items = list(OUT_ITEMS)  # the extra rows copy kept items
cold_T = tables[2]
exp_scores = {}
for i in kept_items:
    full_i = {}
    for src_ in {t.source for t in terms}:
        v, h = full_h[src_]
        full_i[src_] = (np.array([v[i]]), np.array([h[i]]))
    full_i["colisten_same_artist"] = (np.array([np.nan]), np.array([False]))
    full_i["colisten_other_artists"] = (np.array([np.nan]), np.array([False]))
    full_i["popularity"] = (np.array([0.0]), np.array([True]))
    exp_scores[f"sp{i:05d}"] = float(dense_score(full_i, cold_T)[0])
for sid in ["spknown1", "spver1", "spver5"] + sorted(NEW_SP):
    i = SPID_ITEM[sid]  # the features copy item i
    full_i = {}
    for src_ in {t.source for t in terms}:
        v, h = full_h[src_]
        full_i[src_] = (np.array([v[i]]), np.array([h[i]]))
    full_i["colisten_same_artist"] = (np.array([np.nan]), np.array([False]))
    full_i["colisten_other_artists"] = (np.array([np.nan]), np.array([False]))
    full_i["popularity"] = (np.array([0.0]), np.array([True]))
    exp_scores[sid] = float(dense_score(full_i, cold_T)[0])
order_e = sorted(exp_scores, key=lambda s: (-exp_scores[s], s))
got_o = [(t["sp_id"], t["score"]) for t in o["tracks"]]
new_rank = [k + 1 for k, sid in enumerate(order_e) if sid in NEW_SP]
check(
    o["counts"]["tracks"] == len(SP)
    and o["counts"]["in_mlhd"] == 2
    and o["counts"]["kept"] == len(kept_items) + 3 + len(NEW_SP)
    and o["counts"]["known_song_mlhd"] == 3
    and o["counts"]["known_song_listens"] == 2
    and [s for s, _ in got_o] == order_e[: LB.OUTSIDE_N]
    and [t["rank"] for t in o["tracks"]] == list(range(1, len(got_o) + 1))
    and all(sc == exp_scores[s] for s, sc in got_o)
    and o["new_artists"] == NEW_IDS
    and len(NEW_SP) > 0
    and all(t["known"] == (t["sp_id"] not in NEW_SP) for t in o["tracks"])
    and doc["new_artists"]["outside"]
    == {
        "kept": len(NEW_SP),
        "stored": len(NEW_SP),
        "first_rank": new_rank[0],
        "artists": len(NEW_IDS),
    }
    and o["counts"]["kept_new"] == len(NEW_SP)
    and o["checks"]["reference"],
    f"4: the outside list leaves out the bridged track, the track whose "
    f"ISRC reaches MLHD, three duplicates (one a live version of a new "
    f'song), three versions of MLHD songs (after " - ", in brackets, '
    f"under a working title) and two songs of the raw listens (by name, "
    f"quotes and punctuation folded, and by artist MBID), keeps a title "
    f"that only shares a prefix and one with a bracket but no version "
    f"word, adds the {len(NEW_SP)} Spotify tracks of the first "
    f"{len(NEW_IDS)} new artists of the cold-track GAM (first at rank "
    f"{new_rank[0]}), and its {len(got_o)} scores "
    f"equal the cold-track GAM on the catalog's own concepts of the "
    f"copied tracks exactly (tags by ISRC, artist tags, two-path genres, "
    f"credits and audio rebuilt from raw links, aliases folded)",
)
ok_or = (
    all(
        x["value"] > 0 and x["rule"].startswith("none")
        for t in o["tracks"]
        for x in t["reasons"]
    )
    and all(len(t["reasons"]) <= 3 for t in o["tracks"])
    and any(t["reasons"] for t in o["tracks"])
)
check(
    ok_or
    and o["audit"]["audio_equal"]
    and o["audit"]["tracks"] > 0
    and o["audit"]["agreement"]["tags_direct"]["mean"] is not None
    and sum(1 for c in FDI.calls if c[0] == "raw") == 1,
    f"4: the outside reasons are positive, at most 3, marked without a "
    f"stability rule, and the audit rebuilds {o['audit']['tracks']} "
    f"history tracks' audio from Spotify exactly, in the same raw query "
    f"as the outside tracks",
)
back = json.loads((demo_out / "lb_demo.json").read_text())
js_back = LB.read_data_js(demo_out / LB.DATA_FILE)
users_js = (TMP / LB.USERS_FILE).read_text()
check(
    back == js_back
    and back["version"] == "lb-demo-v2.1"
    and (demo_out / LB.DATA_FILE).read_text().count("\n") == 1
    and (TMP / LB.PAGE_FILE).read_bytes() == LB.PAGE_SRC.read_bytes()
    and users_js == 'window.DEMO_USERS = ["demo"];\n'
    and not (demo_out / "lb_demo.html").exists()
    and [x["id"] for x in back["lists"]]
    == ["gam", "itemknn_same", "itemknn_both", "fresh", "outside"]
    and any("wrote lb_demo.json and demo_data.js" in ln for ln in CAP.lines),
    "4: lb_demo.json and the one line of demo_data.js hold the same "
    "document, the static page is copied byte for byte, users.js names "
    "the folder, no HTML is written",
)
check(
    raises(
        lambda: LB.run(
            demo_stage(),
            FakeDemoInputs(bad_audit=True),
            CHAIN,
            TMP / "demo_bad",
            TMP / "demo_bad_cache",
            raw_listens=RAW,
            source={},
            say=quiet,
        ),
        "audit",
    )
    and raises(
        lambda: LB.run(
            demo_stage(),
            FakeDemoInputs(bad_map=True),
            CHAIN,
            TMP / "demo_bad2",
            TMP / "demo_bad_cache2",
            raw_listens=RAW,
            source={},
            say=quiet,
        ),
        "recording_canonical_map",
    ),
    "4: a changed audit value and a changed map entry stop the run",
)


# 6: part 2 on the chain
# Every check below has a route written here, apart from lb_demo's code:
# loops over P9b's lists, a dense sum per reason group, a loop for the main
# genre and the caps, and the raw listens of the export.
def mid_rank(pool_sorted, x):
    a = np.searchsorted(pool_sorted, x, "left")
    b = np.searchsorted(pool_sorted, x, "right")
    return 100.0 * (a + 0.5 * (b - a)) / len(pool_sorted)


def word_of(p):
    return "strong" if p >= 99 else "moderate" if p >= 90 else "slight"


def akeys_of(slot):
    return sorted(f"a{i}" for i in ids_of(slot)) or [f"ki{slot}"]


SHARE_G = state_h.term_share["genres"]


def main_genre_loop(slot, share):
    m_ = W.SPACE.term_mats["genres"]
    idf_ = W.SPACE.idf["genres"]
    best = None
    for c, x in zip(
        m_.indices[m_.indptr[slot] : m_.indptr[slot + 1]],
        m_.data[m_.indptr[slot] : m_.indptr[slot + 1]],
    ):
        w_ = idf_[c] ** 2 * x * (share[c] if share is not None else 0.0)
        key = (-w_, -idf_[c], int(c))
        if best is None or key < best:
            best = key
    return None if best is None else GENRES[best[2]]


ka = doc["known_artists"]
ok_k = (
    ka["known"] == len(KNOWN)
    and ka["by_mbid"] == len({KB, A2})
    and ka["by_name"] == 1
    and ka["history"] == len(HIST_IDS)
)
n_rows = n_none = 0
for lid, path, _ in LB.LISTS:
    for r in LB._get(doc, path) or []:
        if "item" not in r:
            continue
        n_rows += 1
        i = int(r["item"])
        ok_k &= (
            r["artist_keys"] == akeys_of(i)
            and r["genre"] == main_genre_loop(i, SHARE_G)
            and r["known"] == (bool(set(ids_of(i)) & KNOWN) if ids_of(i) else None)
            and r["key"] == f"i{i}"
        )
        n_none += r["known"] is None
for t in doc["outside"]["tracks"]:
    ok_k &= t["known"] == (t["sp_id"] not in NEW_SP) and t["artist_keys"] == [
        f"a{x}" for x in t["mb_artists"]
    ]
ok_k &= KD not in KNOWN and any(KD in ids_of(i) for i in NEW_CAND)
check(
    ok_k and n_rows > 4 * LB.TOP_N and n_none > 0,
    f"6: the known artists ({len(KNOWN)}: the history's, one more by a raw "
    f"listen's artist MBID, one by its exact name, none by a name that a "
    f"listen also gives with an artist MBID) and the filter keys of "
    f"{n_rows} catalog rows (credited artists, main genre by a loop, "
    f"known, and None for {n_none} rows without a credited artist) follow "
    f"the rules",
)

# the fresh list by a loop over P9b's other-artist lists
so = knn_score(items_h, w_h, ("other_artists",))
elig = [
    i
    for i in range(1, W.NI)
    if so[i] > 0
    and cand_h[i]
    and ind_a[i + 1] > ind_a[i]
    and not set(ids_of(i)) & KNOWN
]
elig.sort(key=lambda i: (-so[i], i))
na_, ng_, want_f = {}, {}, []
for i in elig:
    ids, g = ids_of(i), main_genre_loop(i, SHARE_G)
    if any(na_.get(a, 0) >= LB.ARTIST_CAP for a in ids):
        continue
    if g is not None and ng_.get(g, 0) >= LB.GENRE_CAP:
        continue
    want_f.append(i)
    for a in ids:
        na_[a] = na_.get(a, 0) + 1
    if g is not None:
        ng_[g] = ng_.get(g, 0) + 1
    if len(want_f) == LB.STORE_N:
        break
fr = doc["fresh"]["tracks"]
pool_f = np.sort(np.array([so[i] for i in elig]))
check(
    [r["item"] for r in fr] == want_f
    and len(fr) > 0
    and not any(set(ids_of(r["item"])) & KNOWN for r in fr)
    and not any(r["known"] for r in fr)
    and doc["fresh"]["counts"]["candidates"] == len(elig)
    and all(
        abs(r["score"] - so[r["item"]]) <= 1e-12 * max(1.0, so[r["item"]]) for r in fr
    )
    and all(
        abs(r["reasons"][0]["percentile"] - mid_rank(pool_f, so[r["item"]])) < 1e-9
        for r in fr
    )
    and len(LB.apply_filters(fr)) == len(fr)
    and any(set(ids_of(i)) & {KB, KC} for i in fresh0[:3]),
    f"6: the fresh list ({len(fr)} of {len(elig)} candidates) equals a "
    f"loop: other-artist co-listening, no artist of the history or of the "
    f"raw listens, at most {LB.ARTIST_CAP} per artist and {LB.GENRE_CAP} "
    f"per main genre, scores and percentiles exact",
)


# the caps of the fresh list bind: a run with caps 1 and 1 against the loop
saved_caps = (LB.ARTIST_CAP, LB.GENRE_CAP)
LB.ARTIST_CAP, LB.GENRE_CAP = 1, 1
try:
    doc_c = LB.run(
        demo_stage(),
        FakeDemoInputs(),
        CHAIN,
        TMP / "demo_caps",
        TMP / "demo_caps_cache",
        raw_listens=RAW,
        source={"kind": "export"},
        outside=False,
        say=quiet,
    )
finally:
    LB.ARTIST_CAP, LB.GENRE_CAP = saved_caps
na_, ng_, want_c = {}, {}, []
for i in elig:
    ids, g = ids_of(i), main_genre_loop(i, SHARE_G)
    if any(na_.get(a, 0) >= 1 for a in ids) or (g is not None and ng_.get(g, 0) >= 1):
        continue
    want_c.append(i)
    for a in ids:
        na_[a] = na_.get(a, 0) + 1
    if g is not None:
        ng_[g] = ng_.get(g, 0) + 1
got_c = [r["item"] for r in doc_c["fresh"]["tracks"]]
check(
    got_c == want_c
    and len(want_c) < len(elig)
    and doc_c["fresh"]["caps"] == {"artist": 1, "genre": 1}
    and len(LB.apply_filters(doc_c["fresh"]["tracks"], 1, 1)) == len(got_c)
    and "outside" not in doc_c,
    f"6: with caps of 1 the fresh list keeps {len(got_c)} of {len(elig)} "
    f"candidates, as the loop does (the caps bind)",
)


# the co-listening reasons of both ItemKNN lists and the fresh list
def units_loop(i, kinds):
    w_ = dict(zip(items_h.tolist(), w_h.tolist()))
    out = []
    for kk in kinds:
        ip_, nb, cs = C.LISTS[kk]
        for a in range(ip_[i], ip_[i + 1]):
            if int(nb[a]) in w_:
                out.append((int(nb[a]), float(cs[a] * w_[int(nb[a])])))
    out.sort(key=lambda r: (-r[1], r[0]))
    return out


def title_of(h, row_artist):
    a = f"Artist of {h}"
    return f"“Track {h}”" + ("" if a == row_artist else f" ({a})")


def join_(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


ok_u, n_u = True, 0
for name, kinds, rows in (
    ("same_artist", ("same_artist",), doc["itemknn"]["same_artist"]),
    ("both", ("same_artist", "other_artists"), doc["itemknn"]["both"]),
    ("fresh", ("other_artists",), fr),
):
    sk = knn_score(items_h, w_h, kinds)
    pool_k = np.sort(sk[cand_h & (sk > 0)]) if name != "fresh" else pool_f
    for r in rows:
        u = units_loop(r["item"], kinds)
        x = r["reasons"][0]
        shown_u = u[: LB.UNITS_SHOWN]
        more = len(u) - len(shown_u)
        words = (
            "Listeners of "
            + join_(
                [title_of(h, r["artist"]) for h, _ in shown_u]
                + ([f"{more} more of your tracks"] if more else [])
            )
            + " also play this."
        )
        ok_u &= (
            x["reason"] == "colisten"
            and x["evidence"]["count"] == len(u)
            and [(v["id"], v["term"]) for v in x["evidence"]["units"]] == shown_u
            and abs(math.fsum(t for _, t in u) - r["score"])
            <= 1e-12 * max(1.0, r["score"])
            and abs(x["percentile"] - mid_rank(pool_k, sk[r["item"]])) < 1e-9
            and x["strength"] == word_of(x["percentile"])
            and x["words"] == words
            and x["label"] == "Co-listening"
            and "cosine × log(1 + plays)" in x["details"]
        )
        n_u += 1
check(
    ok_u and n_u > 2 * LB.TOP_N,
    f"6: the co-listening reasons of {n_u} rows (ItemKNN same-artist, both "
    f"lists, fresh): the history tracks of a loop over the lists, terms "
    f"that sum to the score, percentiles among the list's positive "
    f"scores, and the words",
)

# the strengths of the model's reasons and the outside reasons
n_cand_h = int(cand_h.sum())
vt0 = dense_terms(full_h, tables[0])
vtc = dense_terms(full_h, tables[2])
cold_groups = PC.reason_groups(G14["cold"]["kept"])
ok_s, n_s = True, 0
for tabv, grp, rows, copy_of in (
    (vt0, groups, doc["gam"], None),
    (vtc, cold_groups, doc["outside"]["tracks"], True),
):
    for gname, names_ in grp:
        gv_all = sum(tabv[jx[n_]] for n_ in names_)
        pool_g = np.sort(gv_all[cand_h])
        for row in rows:
            slot_ = row["item"] if copy_of is None else SPID_ITEM[row["sp_id"]]
            for x in row["reasons"]:
                if x["reason"] != gname:
                    continue
                n_s += 1
                one = (
                    abs(x["percentile"] - mid_rank(pool_g, gv_all[slot_])) < 1e-9
                    and x["pool"] == n_cand_h
                    and x["strength"] == word_of(x["percentile"])
                    and "odds" not in x["words"]
                    and "×" not in x["words"]
                    and f"odds factor {LB._odds(x['odds'])}" in x["details"]
                )
                if not one and ok_s:
                    print(
                        "      first difference:",
                        gname,
                        slot_,
                        x["percentile"],
                        mid_rank(pool_g, gv_all[slot_]),
                        x["pool"],
                        n_cand_h,
                        x["strength"],
                        x["words"],
                        x["details"],
                    )
                ok_s &= one
check(
    ok_s and n_s > LB.TOP_N,
    f"6: the strengths of {n_s} shown reasons (model list, and outside "
    f"list under the cold-track table) equal the mid-rank percentile of a "
    f"dense group sum over the {n_cand_h} candidates, and only the details "
    f"line holds the odds factor",
)

ok_w = True
for row in doc["gam"][:40]:
    for x in row["reasons"]:
        units_ = x["evidence"]["units"][: LB.UNITS_SHOWN]
        more = x["evidence"]["count"] - len(units_)
        if x["reason"] == "artist":
            n_ = x["credit_tracks"]
            lead = (
                f"You play {n_} track{'' if n_ == 1 else 's'} by "
                f"Artist {x['credit']}"
            )
            want = lead + "."
            if units_:
                want = (
                    lead
                    + ". Listeners of "
                    + join_(
                        [title_of(u["id"], row["artist"]) for u in units_]
                        + ([f"{more} more of your tracks"] if more else [])
                    )
                    + " also play it."
                )
        elif x["reason"] == "audio_fit":
            want = (
                "Close to your usual "
                + join_(
                    [
                        {"duration_ms": "duration"}.get(u["name"], u["name"])
                        for u in units_
                    ]
                )
                + "."
            )
        elif x["reason"] == "popularity":
            want = "Played often in the training data (2005 to 2013)."
        else:
            lead = {
                "tag_affinity": "Track tags you play often",
                "artist_tag_affinity": "Artist tags you play often",
                "genre_affinity": "Genres you play often",
            }[x["reason"]]
            want = (
                lead
                + ": "
                + join_(
                    [u["name"] for u in units_] + ([f"{more} more"] if more else [])
                )
                + "."
            )
        ok_w &= x["words"] == want and x["label"] == LB.REASON_LABELS[x["reason"]]
check(
    ok_w,
    "6: the words of the model's reasons: the top 3 evidence units "
    "and 'N more', the artist reason with its credit and co-listened "
    "tracks",
)

# the match summary from the raw listens of the export
lsn = LB.unique_listens(LS)
raw_c = {}
for rec in lsn:
    raw_c[rec["artist"]] = raw_c.get(rec["artist"], 0) + 1
top_raw = sorted(raw_c.items(), key=lambda x: (-x[1], x[0]))[: LB.RAW_ARTISTS]
cs_cols = np.asarray(state_h.credit_cols["artists"])
cs_share = np.asarray(state_h.credit_share["artists"])
o_m = np.lexsort((ID_OF_COL[cs_cols], -cs_share))[: LB.RAW_ARTISTS]
ma = doc["match_artists"]
check(
    [(r["name"], r["listens"]) for r in ma["raw"]] == top_raw
    and all(abs(r["share"] - r["listens"] / len(lsn)) < 1e-15 for r in ma["raw"])
    and ma["raw"][0]["name"] == "Artist"
    and ma["raw"][0]["history_share"] == 0.0
    and [r["artist"] for r in ma["matched"]] == ID_OF_COL[cs_cols[o_m]].tolist()
    and all(abs(r["share"] - cs_share[k]) < 1e-15 for r, k in zip(ma["matched"], o_m))
    and all(r["name"] == f"Artist {r['artist']}" for r in ma["matched"]),
    "6: the match summary: the most played raw artist names against the "
    "largest credited artists of the matched history",
)

# refusals of the document checks
import copy as _copy

bad1 = _copy.deepcopy(back)
bad1["fresh"]["tracks"][0]["known"] = True
bad2 = _copy.deepcopy(back)
bad2["gam"][1]["key"] = bad2["gam"][0]["key"]
bad3 = _copy.deepcopy(back)
for r in bad3["fresh"]["tracks"][: LB.ARTIST_CAP + 1]:
    r["artist_keys"] = ["a999999"]
bad4 = _copy.deepcopy(back)
bad4["gam"][0]["reasons"][0]["percentile"] = 100.5
bad5 = _copy.deepcopy(back)
bad5["gam"][3]["rank"], bad5["gam"][4]["rank"] = (
    bad5["gam"][4]["rank"],
    bad5["gam"][3]["rank"],
)
bad6 = _copy.deepcopy(back)
bad6["outside"]["tracks"][2]["score"] = bad6["outside"]["tracks"][1]["score"] + 1.0
bad7 = _copy.deepcopy(back)
first_new = next(r for r in bad7["gam"] if r["known"] is False)
first_new["known"] = True
bad8 = _copy.deepcopy(back)
no_credit = next(r for r in bad8["gam"] if r["known"] is None)
no_credit["known"] = False
check(
    raises(lambda: LB.check_v2(bad1, KNOWN), "known artists")
    and raises(lambda: LB.check_v2(bad2, KNOWN), "twice")
    and raises(lambda: LB.check_v2(bad3, KNOWN), "caps")
    and raises(lambda: LB.check_v2(bad4, KNOWN), "percentile")
    and raises(lambda: LB.check_v2(bad5, KNOWN), "rank order")
    and raises(lambda: LB.check_v2(bad6, KNOWN), "score rises")
    and raises(lambda: LB.check_v2(bad7, KNOWN), "known flag")
    and raises(lambda: LB.check_v2(bad8, KNOWN), "known flag")
    and LB.check_v2(_copy.deepcopy(back), KNOWN) is None,
    "6: the document checks refuse a known artist in the fresh list, a "
    "track twice, broken caps, a percentile outside [0, 100], ranks out "
    "of order, a score that rises with the rank and a wrong known flag "
    "(also False for a track without a credited artist)",
)
bad_js = TMP / "bad_data.js"
bad_js.write_text(LB.JS_PREFIX + "{}\n;\n")
check(
    raises(lambda: LB.read_data_js(bad_js), "one line"),
    "6: a data file of another form is refused",
)

# 5: no match
no_match = [listen(5000 + k, NO_CANON) for k in range(4)]
seed_col = int(np.argmax(np.bincount(col_a, minlength=len(ID_OF_COL))))
seed_id = int(ID_OF_COL[seed_col])
doc_s = LB.run(
    demo_stage(),
    FakeDemoInputs(),
    CHAIN,
    TMP / "demo_seed",
    TMP / "demo_seed_cache",
    raw_listens=no_match,
    source={"kind": "export"},
    seeds=[f"artist {seed_id}"],
    outside=False,
    say=quiet,
)
rows_of = np.repeat(np.arange(len(ind_a) - 1), np.diff(ind_a))
tr_s = rows_of[col_a == seed_col]
want_tr = sorted(tr_s[np.lexsort((tr_s, -C.POP[tr_s]))][: LB.SEED_TRACKS].tolist())
cand_s = np.ones(W.NI, dtype=bool)
cand_s[0] = False
cand_s[want_tr] = False
st_s = cn.build_user_state(np.array(want_tr), np.full(len(want_tr), LB.W_ONE), W.SPACE)
sc_s = np.where(cand_s, dense_score(cn.full_catalog(st_s, W.SPACE), tables[0]), -np.inf)
known_s = {int(ID_OF_COL[c]) for i in want_tr for c in col_a[ind_a[i] : ind_a[i + 1]]}
new_s = sorted(
    (
        i
        for i in range(1, W.NI)
        if cand_s[i] and ind_a[i + 1] > ind_a[i] and not set(ids_of(i)) & known_s
    ),
    key=lambda i: (-sc_s[i], i),
)[: LB.STORE_N]
top_s = np.lexsort((np.arange(W.NI), -sc_s))[: LB.STORE_N]
union_s = sorted(set(top_s.tolist()) | set(new_s), key=lambda i: (-sc_s[i], i))
check(
    doc_s["match"]["matched"] == 0
    and doc_s["seeds"][0]["artist"] == seed_id
    and doc_s["seeds"][0]["tracks"] == want_tr
    and doc_s["history"]["kind"] == "seed artists"
    and [r["item"] for r in doc_s["gam"]] == union_s
    and [r["item"] for r in doc_s["gam"]][: LB.STORE_N] == top_s.tolist()
    and [r["rank"] for r in doc_s["gam"]] == [p5_rank(sc_s, i) for i in union_s]
    and "outside" not in doc_s,
    f"5: zero matches with a seed artist: its {LB.SEED_TRACKS} most "
    f"listened tracks, one play each, form the history, and the stored "
    f"top {LB.STORE_N} and best new-artist tracks equal a dense full "
    f"sort, with their P5 ranks",
)
doc_p = LB.run(
    demo_stage(),
    FakeDemoInputs(),
    CHAIN,
    TMP / "demo_pop",
    TMP / "demo_pop_cache",
    raw_listens=no_match,
    source={"kind": "export"},
    say=quiet,
)
pop_order = np.lexsort((np.arange(W.NI), -C.POP))[: LB.STORE_N]
pop_pool = np.sort(C.POP[C.POP >= 0].astype(float))
check(
    [r["item"] for r in doc_p["popularity"]] == pop_order.tolist()
    and "gam" not in doc_p
    and "fresh" not in doc_p
    and [x["id"] for x in doc_p["lists"]] == ["popularity"]
    and doc_p["lists"][0]["note"].startswith("No matched history")
    and doc_p["history"]["kind"] == "no history"
    and all(
        r["reasons"][0]["words"].startswith(f"{r['train_listens']:,} " f"listens")
        and abs(
            r["reasons"][0]["percentile"]
            - 100.0
            * (
                np.searchsorted(pop_pool, r["train_listens"], "left")
                + 0.5
                * (
                    np.searchsorted(pop_pool, r["train_listens"], "right")
                    - np.searchsorted(pop_pool, r["train_listens"], "left")
                )
            )
            / len(pop_pool)
        )
        < 1e-9
        for r in doc_p["popularity"]
    )
    and LB.read_data_js(TMP / "demo_pop" / LB.DATA_FILE) == doc_p,
    "5: zero matches without seeds: the popularity list in P5's order with "
    "the note 'No matched history', its reason the train listens, and no "
    "model list",
)

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)

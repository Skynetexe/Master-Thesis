"""Smoke test of the demo page (demo/index.html, rules in the module
docstring of pipeline/lb_demo.py, part 2) and of the parts of
lb_demo.py that the page reads. Every path points into a temporary
folder, and nothing touches the real model folder.

1. Units without a browser: apply_filters() (caps, a cap that is off, only
   new artists, the shown count, a track with two artists, a track without
   a genre), percentile_ranks() with ties, strength_of() at the thresholds,
   lb_outside.norm_name(), song_key() and song_keys() (part 3), the
   duplicates rule by song key, main_genres() on a hand-built
   matrix against a loop, the data file round trip and its refusals,
   users.js, the page copy (an edited copy is kept as index.old.html),
   list_meta() and finish_rows().
2. The page in headless Chromium (Playwright) on small synthetic data files
   that lb_demo.write_outputs() writes: no script error, the match summary,
   the default view (caps 5 and 5, 25 tracks), many filter settings set
   through the page's own boxes, every list, the rows against
   apply_filters() in Python, the caps and the unique keys in the rendered
   table, the odds factor only inside the details, the new-artist tracks of
   the model list with their true ranks (part 3), an empty list, a list
   that the filters empty (and why), a page without lists, a listener without
   history (the popularity list and "No matched history"), a missing data
   file, the listener menu, and no horizontal scroll at 360 pixels. The
   page's filter function also meets apply_filters() on 300 random
   settings per list.
3. The page on the chain's demo of test_demo_and_cold_curve.py (a child
   process with KEEP_TMP=1, or a copy of DEMO_CURVE_TMP): the model,
   ItemKNN, fresh and outside lists under the default filters against
   apply_filters(), the popularity page and the seed page.

Without Playwright, parts 2 and 3 print SKIP.

Usage, from the repository root:
    python3 test_demo_page.py
With DEMO_CURVE_TMP=<folder> (kept by test_demo_and_cold_curve.py with
KEEP_TMP=1), part 3 copies it. P13B2_BASE_CACHE and P13_CHAIN_CACHE pass to
test_demo_and_cold_curve.py. With KEEP_TMP=1 the temporary folder stays.
"""

import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from pipeline import lb_demo as LB
from pipeline import lb_outside as LO

HERE = Path(__file__).resolve().parent
T0 = time.time()
failures, skipped = [], []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def skip(msg):
    print("SKIP  " + msg, flush=True)
    skipped.append(msg)


def clock():
    return f"{time.time() - T0:.0f} s"


def raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        return text in str(exc)
    return False


def quiet(msg):
    pass


TMP = Path(tempfile.mkdtemp(prefix="test_demo_page_"))

# 1: units
R_ = [
    {"key": "k1", "artist_keys": ["a1"], "genre": "rock", "known": True},
    {"key": "k2", "artist_keys": ["a1", "a2"], "genre": "rock", "known": False},
    {"key": "k3", "artist_keys": ["a1"], "genre": None, "known": False},
    {"key": "k4", "artist_keys": ["a2"], "genre": "pop", "known": True},
    {"key": "k5", "artist_keys": ["a3"], "genre": "rock", "known": False},
    {"key": "k6", "artist_keys": ["a3", "a3"], "genre": None, "known": False},
    {"key": "k7", "artist_keys": ["ki7"], "genre": None, "known": None},
]


def keys(rows):
    return [r["key"] for r in rows]


check(
    keys(LB.apply_filters(R_, 2, 2)) == ["k1", "k2", "k4", "k6", "k7"]
    and keys(LB.apply_filters(R_, 1, None)) == ["k1", "k4", "k5", "k7"]
    and keys(LB.apply_filters(R_, None, 1)) == ["k1", "k3", "k4", "k6", "k7"]
    and keys(LB.apply_filters(R_, 0, 0, only_new=True)) == ["k2", "k3", "k5", "k6"]
    and keys(LB.apply_filters(R_, None, None, show=3)) == ["k1", "k2", "k3"]
    and keys(LB.apply_filters(R_, 2, 1, only_new=True, show=2)) == ["k2", "k3"]
    and LB.apply_filters([], 5, 5) == [],
    "1: apply_filters: a track counts once for each of its artists, a "
    "track without genre escapes the genre cap, a cap of 0 or None is off, "
    "only new artists (known False, so not a track without a credited "
    "artist), the shown count",
)
pool = np.array([1.0, 2.0, 2.0, 2.0, 5.0])
pr = LB.percentile_ranks(pool, [2.0, 5.0, 0.5, 9.0])
check(
    np.allclose(pr, [50.0, 90.0, 0.0, 100.0], atol=0, rtol=0)
    and np.isnan(LB.percentile_ranks(np.array([]), [1.0])[0])
    and [LB.strength_of(x) for x in (99.0, 98.999, 90.0, 89.99, None)]
    == ["strong", "moderate", "moderate", "slight", None],
    "1: the percentile rank counts equal values at half weight, and the "
    "strength words start at 99 and 90",
)
check(
    LO.norm_name("  Sigur Rós ") == "sigur rós"
    and LO.norm_name("Guns N’ Roses") == "guns n' roses"
    and LO.song_key("Don’t Stop Me Now - 2011 Mix")
    == LO.song_key("Don't Stop Me Now")
    == "dont stop me now"
    and LO.song_key("Song! (Live)") == "song"
    and LO.song_key("Song! (Part 2)") == "song part 2"
    and LO.song_key("Ça va - Remastered") == "ça va"
    and LO.song_key("!!!") == ""
    and LO.song_key(None) == "",
    "1: norm_name folds case, width and quotes, and song_key cuts the "
    'version text after " - " and in brackets, '
    "version text, folds quotes and drops punctuation",
)
SK = {
    t: LO.song_keys(t)
    for t in (
        "Free As A Bird (2025 Mix)",
        'Dusty ("Poor Twisted Me" Take 12)',
        "F.O.B.D. (\u201cUntil It Sleeps\u201d Rough Chorus Vocal Idea Mix)",
        "Country Roads (Take Me Home)",
        "That Means A Lot (Take 1 - Remastered)",
        "Song (with Ed Sheeran)",
        "Song [Live",
        "Song (Alt Take) [Mono]",
        "(Live)",
        None,
    )
}
check(
    SK["Free As A Bird (2025 Mix)"] == ["free as a bird"]
    and SK['Dusty ("Poor Twisted Me" Take 12)'] == ["dusty", "poor twisted me"]
    and SK["F.O.B.D. (\u201cUntil It Sleeps\u201d Rough Chorus Vocal Idea " "Mix)"]
    == ["fobd", "until it sleeps"]
    and SK["Country Roads (Take Me Home)"] == ["country roads take me home"]
    and SK["That Means A Lot (Take 1 - Remastered)"] == ["that means a lot"]
    and SK["Song (with Ed Sheeran)"]
    == SK["Song [Live"]
    == SK["Song (Alt Take) [Mono]"]
    == ["song"]
    and SK["(Live)"] == []
    and SK[None] == [],
    '1: song_keys cuts bracketed version parts (a version word, "take" '
    'with a number, "with ...", an open bracket), keeps other brackets, '
    "and adds a quoted working title as a second key",
)
dd = [
    {
        "sp_id": "x2",
        "isrc": None,
        "name": "New Song (Live)",
        "popularity": 30,
        "sp_artists": ("s1",),
    },
    {
        "sp_id": "x1",
        "isrc": None,
        "name": "New Song",
        "popularity": 40,
        "sp_artists": ("s1",),
    },
    {
        "sp_id": "x3",
        "isrc": None,
        "name": "New Song (Part 2)",
        "popularity": 20,
        "sp_artists": ("s1",),
    },
]
check(
    [t["sp_id"] for t in LO.dedupe_tracks(dd)] == ["x1", "x3"],
    "1: the duplicates rule keeps one track per song key and artists",
)
rng = np.random.default_rng(5)
n_s, n_g = 40, 9
m = sp.random(n_s, n_g, density=0.3, random_state=7, format="lil")
m[0, :] = 0
m = m.tocsr()
m.eliminate_zeros()
m.data[:] = 1.0
idf = rng.uniform(1.0, 3.0, n_g)
idf[3] = idf[5]  # a tie in idf
share = rng.uniform(0.0, 0.2, n_g) * (rng.random(n_g) < 0.6)


def mg_loop(slot, share_):
    best = None
    for c in m.indices[m.indptr[slot] : m.indptr[slot + 1]]:
        w = idf[c] ** 2 * (share_[c] if share_ is not None else 0.0)
        k = (-w, -idf[c], int(c))
        best = k if best is None or k < best else best
    return -1 if best is None else best[2]


slots_ = rng.permutation(np.arange(n_s))
check(
    LB.main_genres(m, idf, share, slots_).tolist()
    == [mg_loop(s, share) for s in slots_]
    and LB.main_genres(m, idf, None, slots_).tolist()
    == [mg_loop(s, None) for s in slots_]
    and LB.main_genres(m, idf, share, []).tolist() == [],
    "1: main_genres equals a loop: idf^2 x share, then idf, then the "
    "column, and idf alone without a history",
)


# A synthetic document, as lb_demo.run() builds one


def synth_rows(n, rnd, prefix, known_share=0.5, with_item=True):
    rows = []
    for k in range(n):
        arts = sorted(
            {f"a{rnd.randint(1, 6)}" for _ in range(1 if rnd.random() < 0.8 else 2)}
        )
        g = rnd.choice(["rock", "pop", "jazz", "metal", None])
        key = f"{prefix}{k}"
        r = {
            "rank": k + 1,
            "key": key,
            "title": f"Title {key} <b>x</b>",
            "artist": "Artist " + " & ".join(arts),
            "artist_keys": arts,
            "genre": g,
            "known": rnd.random() < known_share,
            "score": float(n - k),
            "details": f"Score {n - k}.",
        }
        if with_item:
            r["item"] = 1000 + k
        rows.append(r)
    return rows


def model_reason(rnd, kind):
    v = rnd.uniform(0.1, 9.0)
    units = [
        {
            "id": rnd.randint(1, 50),
            "name": f"tag{u}",
            "term": 0.1,
            "title": f"Hist {u}",
            "artist": f"Other {u}",
        }
        for u in range(rnd.randint(0, 6))
    ]
    x = {
        "reason": kind,
        "value": v,
        "odds": float(np.exp(v)),
        "evidence": {"count": len(units) + rnd.randint(0, 9), "units": units[:5]},
        "percentile": rnd.uniform(50.0, 100.0),
        "pool": 1234,
    }
    if kind == "artist":
        x.update(credit=7, credit_tracks=12, credit_name="Seven")
    return x


def colisten_reason(rnd, score):
    units = [
        {"id": u, "term": 0.5 / (u + 1), "title": f"Hist {u}", "artist": f"Other {u}"}
        for u in range(rnd.randint(1, 3))
    ]
    return {
        "reason": "colisten",
        "value": score,
        "percentile": rnd.uniform(80.0, 100.0),
        "pool": 99,
        "evidence": {"count": len(units) + rnd.randint(0, 4), "units": units},
    }


def synth_doc(seed=11, empty_both=True):
    rnd = random.Random(seed)
    gam_rows = synth_rows(80, rnd, "i", known_share=1.0)
    for k, r in enumerate(gam_rows[60:]):  # part 3: new artists
        r.update(
            rank=5001 + 3 * k,
            known=False,
            artist_keys=[f"a9{k:02d}"],
            artist=f"New artist {k}",
        )
    for r in gam_rows[50:53]:  # no credited artist
        r.update(known=None, artist_keys=[f"k{r['key']}"])
    for r in gam_rows:
        r["reasons"] = [
            model_reason(rnd, k)
            for k in rnd.sample(
                ["artist", "tag_affinity", "genre_affinity", "audio_fit"],
                rnd.randint(0, 3),
            )
        ]
        r["cold_score"] = 0.0
    same = synth_rows(40, rnd, "i", known_share=0.9)  # shares keys
    for r in same:
        r["reasons"] = [colisten_reason(rnd, r["score"])]
    fresh = synth_rows(30, rnd, "f", known_share=0.0)
    for r in fresh:
        r["reasons"] = [colisten_reason(rnd, r["score"])]
    outside = synth_rows(20, rnd, "s", known_share=1.0, with_item=False)
    for r in outside:
        r["reasons"] = [model_reason(rnd, "artist")]
        r["sp_id"] = r["key"][1:]
        r["sub"] = "Album, 2016"
    doc = {
        "version": LB.VERSION,
        "created_utc": "2024-01-02 12:00 UTC",
        "source": {"kind": "api", "user": "synth"},
        "match": {
            "listens": 1200,
            "matched": 400,
            "with_mbid": 1100,
            "share_of_listens": 400 / 1200,
            "share_of_listens_with_mbid": 400 / 1100,
            "recordings": 500,
            "recordings_matched": 150,
            "items": 140,
            "misses": {m_: 3 for m_ in LB.MISSES},
            "first_listen_utc": "2022-01-01",
            "last_listen_utc": "2023-12-31",
        },
        "match_artists": {
            "listens": 1200,
            "raw_names": 77,
            "raw": [
                {
                    "name": f"Raw {k}",
                    "listens": 90 - k,
                    "share": (90 - k) / 1200,
                    "history_share": 0.01 * k,
                }
                for k in range(10)
            ],
            "matched": [
                {"artist": k, "name": f"Hist {k}", "share": 0.03, "listen_share": 0.0}
                for k in range(10)
            ],
        },
        "history": {
            "items": 140,
            "plays": 400,
            "kind": "matched listens",
            "top_tracks": [{"item": 5, "plays": 9, "title": "T", "artist": "A"}],
            "top_artists": [{"artist": 1, "share": 0.2, "name": "A"}],
        },
        "model": {
            "final": "synthetic",
            "terms": ["tag_affinity"],
            "candidates": 99999,
            "replicates": 100,
            "members": 12,
            "s4_need": 10,
        },
        "new_artists": {"gam": {"candidates": 4321, "stored": 20, "first_rank": 5001}},
        "gam": gam_rows,
        "itemknn": {
            "same_artist": same,
            "both": [] if empty_both else copy.deepcopy(same),
        },
        "fresh": {"tracks": fresh, "counts": {"candidates": 30}},
        "outside": {
            "tracks": outside,
            "artists": [1, 2],
            "counts": {
                "kept": 20,
                "tracks": 90,
                "in_mlhd": 50,
                "known_song_mlhd": 7,
                "known_song_listens": 3,
            },
        },
    }
    LB.finish_rows(doc)
    doc["lists"] = LB.list_meta(doc)
    doc["page"] = {
        "show": LB.TOP_N,
        "stored": LB.STORE_N,
        "artist_cap": LB.ARTIST_CAP,
        "genre_cap": LB.GENRE_CAP,
        "units": LB.UNITS_SHOWN,
        "strong": LB.STRONG_PCT,
        "moderate": LB.MODERATE_PCT,
    }
    return doc


SITE = TMP / "site"
DOC = synth_doc()
back = LB.write_outputs(DOC, SITE / "synth", quiet)
doc_pop = synth_doc(seed=3)
for k in ("gam", "itemknn", "fresh", "outside", "match_artists", "model"):
    doc_pop.pop(k)
doc_pop["history"] = {"items": 0, "plays": 0, "kind": "no history"}
doc_pop["match"].update(
    matched=0, items=0, share_of_listens=0.0, share_of_listens_with_mbid=0.0
)
pop_rows = synth_rows(60, random.Random(4), "i", known_share=0.3)
for r in pop_rows:
    r["train_listens"] = int(r["score"] * 100)
    r["reasons"] = [
        {
            "reason": "train_listens",
            "value": r["train_listens"],
            "percentile": 99.5,
            "pool": 600,
        }
    ]
doc_pop["popularity"] = pop_rows
LB.finish_rows(doc_pop)
doc_pop["lists"] = LB.list_meta(doc_pop)
LB.write_outputs(doc_pop, SITE / "nohistory", quiet)
doc_empty = synth_doc(seed=5)
for k in ("gam", "fresh"):
    doc_empty[k] = [] if k == "gam" else {"tracks": [], "counts": {}}
doc_empty["itemknn"] = {"same_artist": [], "both": []}
doc_empty["outside"]["tracks"] = []
doc_empty["lists"] = LB.list_meta(doc_empty)
LB.write_outputs(doc_empty, SITE / "empty", quiet)
doc_none = {
    "version": LB.VERSION,
    "source": {"kind": "export", "export": "x.zip"},
    "history": {"items": 0, "kind": "no history"},
    "lists": [],
}
LB.write_outputs(doc_none, SITE / "nolists", quiet)
users = (SITE / LB.USERS_FILE).read_text()
check(
    back == LB.read_data_js(SITE / "synth" / LB.DATA_FILE)
    and back == json.loads((SITE / "synth" / "lb_demo.json").read_text())
    and (SITE / "synth" / LB.DATA_FILE).read_text().count("\n") == 1
    and users == 'window.DEMO_USERS = ["empty", "nohistory", "nolists", ' '"synth"];\n'
    and (SITE / LB.PAGE_FILE).read_bytes() == LB.PAGE_SRC.read_bytes()
    and [x["id"] for x in back["lists"]]
    == ["gam", "itemknn_same", "itemknn_both", "fresh", "outside"]
    and back["lists"][-1]["note"].startswith(
        "Spotify tracks that the training data never saw, by your 2 "
    )
    and [x["id"] for x in LB.read_data_js(SITE / "nohistory" / LB.DATA_FILE)["lists"]]
    == ["popularity"],
    "1: write_outputs: lb_demo.json and the one line of demo_data.js hold "
    "the same document, the page is copied byte for byte, users.js names "
    "the four folders, list_meta keeps the lists in page order",
)
bad = SITE / "bad.js"
bad.write_text("var DEMO = {};\n")
two = SITE / "two.js"
two.write_text(LB.JS_PREFIX + "{}\n" + LB.JS_PREFIX + "{};\n")
check(
    raises(lambda: LB.read_data_js(bad), "one line")
    and raises(lambda: LB.read_data_js(two), "one line"),
    "1: read_data_js refuses another prefix and a second line",
)
ok_w = all(
    x["words"]
    and x["details"]
    and x["label"]
    and x["strength"] in ("strong", "moderate", "slight")
    and "odds factor" not in x["words"]
    for lid, path, _ in LB.LISTS
    for r in LB._get(back, path) or []
    for x in r["reasons"]
)
check(
    ok_w and all(r.get("no_reason") for r in back["gam"] if not r["reasons"]),
    "1: finish_rows gives every reason a label, a strength, words without "
    "the odds factor and a details line, and a model row without a "
    "reason its note",
)
saved_src = LB.PAGE_SRC
LB.PAGE_SRC = TMP / "missing.html"
check(
    raises(lambda: LB.copy_page(TMP / "x"), "is missing"),
    "1: a missing static page stops the run",
)
LB.PAGE_SRC = saved_src
EDIT = TMP / "edit"
EDIT.mkdir()
(EDIT / LB.PAGE_FILE).write_text("<p>an edit</p>\n")
said = []
LB.copy_page(EDIT, said.append)
LB.copy_page(EDIT, said.append)
check(
    (EDIT / LB.OLD_PAGE_FILE).read_text() == "<p>an edit</p>\n"
    and (EDIT / LB.PAGE_FILE).read_bytes() == LB.PAGE_SRC.read_bytes()
    and len(said) == 1
    and "index.old.html" in said[0],
    "1: a copy of the page that differs (an edit) is kept as "
    "index.old.html, and the next call changes nothing",
)

# 2 and 3: the page
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


def page_view(page):
    """(row keys, {key: (artist_keys, genre)} lookups are done by the
    caller), the empty message, the count line."""
    keys_ = page.eval_on_selector_all(
        "#tracks tbody tr", "rs => rs.map(r => r.dataset.key)"
    )
    empty = page.eval_on_selector_all("#empty", "e => e.map(x => x.textContent)")
    return keys_, empty


def set_filters(page, artist_on, artist_cap, genre_on, genre_cap, only_new, show):
    for box, on in (
        ("#f-artistOn", artist_on),
        ("#f-genreOn", genre_on),
        ("#f-onlyNew", only_new),
    ):
        if page.is_checked(box) != on:
            page.click(box)
    page.fill("#f-artistCap", str(artist_cap))
    page.fill("#f-genreCap", str(genre_cap))
    page.fill("#f-show", str(show))


def py_view(rows, artist_on, artist_cap, genre_on, genre_cap, only_new, show):
    return keys(
        LB.apply_filters(
            rows,
            artist_cap if artist_on else 0,
            genre_cap if genre_on else 0,
            only_new,
            show,
        )
    )


def caps_hold(rows_by_key, shown, artist_cap, genre_cap):
    na, ng = {}, {}
    for k in shown:
        r = rows_by_key[k]
        for a in set(r["artist_keys"]):
            na[a] = na.get(a, 0) + 1
        if r["genre"] is not None:
            ng[r["genre"]] = ng.get(r["genre"], 0) + 1
    return (not artist_cap or max(na.values(), default=0) <= artist_cap) and (
        not genre_cap or max(ng.values(), default=0) <= genre_cap
    )


def open_page(browser, folder, query, width=1200):
    errors = []
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.on("pageerror", lambda e: errors.append(f"pageerror {e}"))
    page.on(
        "console",
        lambda m_: errors.append(f"console {m_.text}") if m_.type == "error" else None,
    )
    page.goto((Path(folder) / LB.PAGE_FILE).as_uri() + query)
    page.wait_for_function("document.querySelector('#app h1') !== null")
    return page, errors


def list_ids(doc):
    return [x["id"] for x in doc["lists"]]


def rows_of(doc, lid):
    meta = next(x for x in doc["lists"] if x["id"] == lid)
    return LB._get(doc, meta["path"]) or []


if sync_playwright is None:
    skip(
        "2 and 3: the page checks need Playwright (python3 -m pip install "
        "playwright, then python3 -m playwright install chromium)"
    )
else:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page, errs = open_page(browser, SITE, "?user=synth")
        page.wait_for_selector("#tracks tbody tr")
        k_, _ = page_view(page)
        gam = back["gam"]
        by_key = {r["key"]: r for r in gam}
        want = py_view(gam, True, 5, True, 5, False, 25)
        txt = page.inner_text("#match")
        check(
            not errs
            and k_ == want
            and len(k_) == len(set(k_))
            and caps_hold(by_key, k_, 5, 5)
            and len(want) == 25
            and "1,200" in txt
            and "33.3% of all listens" in txt
            and page.eval_on_selector_all("#raw-artists tbody tr", "r => r.length")
            == 10
            and page.eval_on_selector_all("#matched-artists tbody tr", "r => r.length")
            == 10
            and page.title() == "Recommendations for synth",
            f"2: the page opens without a script error, shows the match "
            f"summary and the model list under the default filters (25 "
            f"rows, at most 5 per artist and per genre), as "
            f"apply_filters() ({clock()})",
        )
        # the odds factor only inside the details, the words outside
        first = gam[[r["key"] for r in gam].index(k_[0])]
        row_txt = page.inner_text(f"#tracks tr[data-key='{k_[0]}'] td.why")
        vis = page.eval_on_selector(
            f"#tracks tr[data-key='{k_[0]}'] td.why",
            "td => Array.from(td.querySelectorAll('p.reason'))"
            ".map(p => p.textContent)",
        )
        dets = page.eval_on_selector(
            f"#tracks tr[data-key='{k_[0]}'] td.why details", "d => d.textContent"
        )
        ok_r = (
            all(
                f"{x['label']} ({x['strength']}): {x['words']}" in vis
                for x in first["reasons"]
            )
            and all(x["details"] in dets for x in first["reasons"])
            and not any("odds factor" in v for v in vis)
            and (not first["reasons"] or "odds factor" in dets)
        )
        check(
            ok_r
            and "<b>"
            not in page.inner_html(
                f"#tracks tr[data-key='{k_[0]}'] td:nth-child(2)"
            ).replace("&lt;b&gt;", ""),
            "2: each reason shows its label, strength and words, the odds "
            "factor stays inside the details, and a title shows as text",
        )
        # many settings through the page's boxes
        rnd = random.Random(2026)
        ok_s, n_set = True, 0
        for lid in list_ids(back):
            page.check(f"input[name='list'][value='{lid}']")
            rows = rows_of(back, lid)
            rk = {r["key"]: r for r in rows}
            for _ in range(6):
                st_ = (
                    rnd.random() < 0.7,
                    rnd.randint(1, 6),
                    rnd.random() < 0.7,
                    rnd.randint(1, 6),
                    rnd.random() < 0.4,
                    rnd.randint(1, 40),
                )
                set_filters(page, *st_)
                k_, empty = page_view(page)
                w_ = py_view(rows, *st_)
                ok_s &= (
                    k_ == w_
                    and len(k_) == len(set(k_))
                    and caps_hold(
                        rk, k_, st_[1] if st_[0] else 0, st_[3] if st_[2] else 0
                    )
                    and (bool(empty) == (not w_))
                )
                n_set += 1
        check(
            ok_s and not errs,
            f"2: {n_set} filter settings set through the page's boxes on "
            f"every list give the rows of apply_filters(), unique, with "
            f"the caps held, and a message when no row passes",
        )
        page.check("input[name='list'][value='itemknn_both']")
        set_filters(page, True, 5, True, 5, False, 25)
        _, empty_both = page_view(page)
        page.check("input[name='list'][value='outside']")
        set_filters(page, True, 5, True, 5, True, 25)
        _, empty_new = page_view(page)
        check(
            empty_both == ["This list is empty."]
            and empty_new
            == [
                "All 20 stored tracks of this list are by "
                "artists you know, so this filter leaves the "
                "list empty."
            ],
            "2: an empty list and a list that the filters empty show a "
            "message, which says when every stored track is by a known "
            "artist",
        )
        page.check("input[name='list'][value='gam']")
        set_filters(page, True, 5, True, 5, True, 25)
        k_new, _ = page_view(page)
        note_g = page.inner_text("#list > p.muted")
        det_g = page.eval_on_selector(
            f"#tracks tr[data-key='{k_new[0]}'] details", "d => d.textContent"
        )
        new_keys = [r["key"] for r in back["gam"][60:]]
        none_key = back["gam"][50]["key"]
        set_filters(page, False, 5, False, 5, False, 80)
        none_det = page.eval_on_selector(
            f"#tracks tr[data-key='{none_key}'] details", "d => d.textContent"
        )
        set_filters(page, True, 5, True, 5, True, 25)
        check(
            k_new == py_view(back["gam"], True, 5, True, 5, True, 25)
            and k_new
            and set(k_new) <= set(new_keys)
            and k_new[0] == new_keys[0]
            and "No credited artist in the catalog" in none_det
            and "the first sits at rank 5,001 of 99,999 catalog " "candidates" in note_g
            and "Rank before the filters: 5,001." in det_g,
            "2: with only new artists the model list shows its stored "
            "new-artist tracks, with their true ranks (5,001 and on) and "
            "the note on the first one",
        )
        # the page's filter function on random settings against Python
        ok_f, n_f = True, 0
        for lid in list_ids(back):
            rows = rows_of(back, lid)
            for _ in range(300):
                opt = {
                    "artistCap": rnd.choice([0, 1, 2, 3, 5, 9]),
                    "genreCap": rnd.choice([0, 1, 2, 4, 5]),
                    "onlyNew": rnd.random() < 0.3,
                    "show": rnd.choice([None, 1, 3, 10, 25, 500]),
                }
                got = page.evaluate(
                    "([rows, opt]) => window.demoApplyFilters(rows, opt)"
                    ".map(r => r.key)",
                    [rows, opt],
                )
                ok_f &= got == keys(
                    LB.apply_filters(
                        rows,
                        opt["artistCap"],
                        opt["genreCap"],
                        opt["onlyNew"],
                        opt["show"],
                    )
                )
                n_f += 1
        check(
            ok_f,
            f"2: the page's filter function equals apply_filters() "
            f"on {n_f} random settings",
        )
        page.close()
        # also-in, a narrow screen
        page, errs = open_page(browser, SITE, "?user=synth", width=360)
        page.wait_for_selector("#tracks tbody tr")
        sw = page.evaluate("document.documentElement.scrollWidth")
        iw = page.evaluate("window.innerWidth")
        also = page.eval_on_selector_all(
            "#tracks details", "ds => ds.map(d => d.textContent)"
        )
        check(
            sw <= iw
            and not errs
            and any("Also shown in: ItemKNN, same artist" in a for a in also),
            f"2: at 360 pixels the page does not scroll sideways ({sw} of "
            f"{iw}), and a track that another list shows says so",
        )
        page.close()
        # without history, without lists, without data, and the menu
        page, errs = open_page(browser, SITE, "?user=nohistory")
        page.wait_for_selector("#tracks tbody tr")
        k_, _ = page_view(page)
        dp = LB.read_data_js(SITE / "nohistory" / LB.DATA_FILE)
        note = page.inner_text("#lists")
        check(
            not errs
            and k_ == py_view(dp["popularity"], True, 5, True, 5, False, 25)
            and page.is_visible("#no-history")
            and "No matched history" in note
            and page.inner_text("#tracks tbody tr td.why").startswith(
                "Popularity (strong): "
            ),
            "2: a listener without history gets the popularity list with "
            "the note 'No matched history'",
        )
        page.close()
        page, errs = open_page(browser, SITE, "?user=empty")
        page.wait_for_selector("#empty")
        e1 = page.inner_text("#empty")
        page.close()
        page, errs2 = open_page(browser, SITE, "?user=nolists")
        page.wait_for_selector("#empty")
        e2 = page.inner_text("#empty")
        page.close()
        page, errs3 = open_page(browser, SITE, "?user=nobody")
        page.wait_for_selector("#message")
        e3 = page.inner_text("#message")
        menu = page.eval_on_selector_all("#user option", "o => o.length")
        page.close()
        page, errs4 = open_page(browser, SITE, "")
        page.wait_for_selector("#user")
        first_user = page.input_value("#user")
        page.select_option("#user", "synth")
        page.wait_for_url("**?user=synth")
        page.wait_for_selector("#tracks tbody tr")
        page.close()
        check(
            not errs
            and not errs2
            and e1 == "This list is empty."
            and e2 == "This page has no list."
            and "cannot read nobody/demo_data.js" in e3
            and first_user == "empty"
            and not errs4,
            "2: a page whose lists are all empty and a page without lists "
            "render without a script error, a missing data file gives a "
            "message, and the listener menu opens another folder",
        )

        # 3: the chain's demo
        if os.environ.get("DEMO_CURVE_TMP"):
            B2 = TMP / "demo_curve"
            shutil.copytree(os.environ["DEMO_CURVE_TMP"], B2)
        else:
            print(
                "      running test_demo_and_cold_curve.py " "(about 2 minutes)",
                flush=True,
            )
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
                f"3: test_demo_and_cold_curve.py passes and keeps its "
                f"folder ({clock()})",
            )
            if r.returncode != 0 or len(kept) != 1:
                print(r.stdout[-3000:], r.stderr[-3000:])
                sys.exit(1)
            B2 = kept[0]
        dd = LB.read_data_js(B2 / "demo" / LB.DATA_FILE)
        page, errs = open_page(browser, B2, "?user=demo")
        page.wait_for_selector("#tracks tbody tr")
        ok_c, n_c = True, 0
        for lid in list_ids(dd):
            page.check(f"input[name='list'][value='{lid}']")
            k_, empty = page_view(page)
            w_ = py_view(rows_of(dd, lid), True, 5, True, 5, False, 25)
            ok_c &= k_ == w_ and (bool(empty) == (not w_))
            n_c += len(k_)
        page.check("input[name='list'][value='fresh']")
        fresh_note = page.inner_text("#list")
        page.close()
        check(
            ok_c
            and not errs
            and n_c > 2 * LB.TOP_N
            and list_ids(dd)
            == ["gam", "itemknn_same", "itemknn_both", "fresh", "outside"]
            and "declared ranking policy, not a model result" in fresh_note,
            f"3: the chain's demo page shows every list ({n_c} rows) as "
            f"apply_filters() under the default filters, and labels the "
            f"fresh list as a declared ranking policy",
        )
        ok_p = True
        for user, lid in (("demo_pop", "popularity"), ("demo_seed", "gam")):
            dd2 = LB.read_data_js(B2 / user / LB.DATA_FILE)
            page, errs = open_page(browser, B2, f"?user={user}")
            page.wait_for_selector("#tracks tbody tr")
            k_, _ = page_view(page)
            ok_p &= not errs and k_ == py_view(
                rows_of(dd2, lid), True, 5, True, 5, False, 25
            )
            if user == "demo_pop":
                ok_p &= page.is_visible("#no-history")
            page.close()
        check(
            ok_p,
            "3: the chain's popularity page (no matched history) and "
            "seed page render as apply_filters()",
        )
        browser.close()

print(f"\n{len(failures)} failures, {len(skipped)} skipped, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)

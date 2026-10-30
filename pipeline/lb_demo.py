"""The ListenBrainz demo: one listener's
ListenBrainz listens, folded into the frozen model, as the data of a static
page with the model's list and its reasons, ItemKNN's lists beside it, a
list of new artists, and tracks that MLHD lacks ranked by the cold-track GAM.

Part 2 below changes the outputs, stores 500 tracks per list and adds the
fresh list, and part 3 refines the lists after the first runs. Where the
parts differ, the later part holds. The model, its scores and the reason
rules D1 and S4 do not change.

RULES, written before any run on real data. The demo is
descriptive. No number in it tests, selects or changes a model.

Listens.
- Source: the ListenBrainz API (GET /1/user/<user>/listens, at most
  PAGE_SIZE per page, walked back in time through max_ts), or a
  ListenBrainz JSON export (a .zip of JSON Lines files, a .jsonl file or a
  .json file). The API pages are saved as they arrive, so a stopped fetch
  continues, and a complete fetch is reused until --refresh.
- A listen counts once per (listened_at, recording_msid, artist name,
  track name). A listen without listened_at is skipped.
- Its recording MBIDs, in this order: the MBID the client submitted
  (track_metadata.additional_info.recording_mbid), then ListenBrainz's own
  mapping (track_metadata.mbid_mapping.recording_mbid). A string that is
  not a UUID is no MBID.

Change to the fetch only, after the first fetch stopped
at 75 percent and before any listen was matched: ListenBrainz closed many
page requests after about 40 seconds without an answer, and one page on
all six tries. Now a page without an answer in TRIES_PER_SIZE tries is
asked again with half the listens, down to MIN_PAGE, and the fetch keeps
the smaller size. A page at MIN_PAGE gets all RETRIES tries. A cut or
broken answer is a failed try. A rate-limit wait is no failed try. A full
page inside one second is asked again with twice the listens (up to
API_MAX), so max_ts never skips listens of that second. The saved pages
stay valid (FETCH_VERSION). The rules of the demo did not change.

Mapping: the canonical chain of BuildFeatures.py, redirects included.
1. ListenBrainz's canonical redirect (canonical_recording_redirect),
2. else the MBID itself when it is canonical (canonical_musicbrainz_data),
3. else MusicBrainz's merge redirect (recording_gid_redirect to the
   current recording): the current recording when it is canonical, else
   its canonical redirect.
The canonical MBID gives the item through item_ids. A listen matches
through its first MBID that reaches an item. Report: matched listens over
all listens and over the listens with an MBID, the same for distinct
recordings (a listen's first MBID), the matched items, the tier of each
match, the MBID source of each match, and the reason of each miss (no
MBID, no canonical recording, a canonical recording outside MLHD). An MBID
with two canonical targets takes the smaller one and is counted.

Fold-in: the history is the matched items, plays = the matched listens of
an item, weight round(1000 log1p(plays)) / 1000 (P6's train weights).
concepts.build_user_state() gives the listener's state. Nothing is fitted.

Ranking: P5's candidates and ties: every item slot except slot 0 and the
history items, equal scores by item id. The score of a slot is
p14.term_sums() with the final model's table (the term values added in
term order). TOP_N = 25 tracks.

Reasons, P14's rules on the top 25:
- contribution: the term value minus its mean over the listener's
  candidate set. artist_affinity and colisten_same_artist form the artist
  reason (their sum), every other term is a reason of its own. A reason is
  a candidate when one of its terms is on its shape at the track.
- shown: D1 (the 100 listener replicates of the final model: positive in
  at least 95 percent, in the top 3 in at least 80 percent) and S4 (the S3
  set: positive under every member, in the top 3 in at least 80 percent).
- strength: the odds factor exp(contribution).
- evidence (concepts.evidence): the matched tags or genres, the audio
  dimensions from the best fit down, and for the artist reason the
  credited artist, the history tracks by that artist and the co-listened
  history tracks. Up to EVIDENCE_K units each, with the count of all.

ItemKNN beside it (itemknn.py): the same-artist list and both lists, the
top 25 candidates with a positive score, ties by item id.

Zero matched plays, or no listens given: with --seed-artist (an artist
MBID, merge redirects included, or an exact artist name, case ignored,
then the artist with the most train listens over its tracks), a history
of each seed artist's SEED_TRACKS most-listened train tracks, one play
each, ranked and explained as above. Without seed artists, the popularity
list (train listens, ties by item id) and no reasons.

Outside list, only when the history has credited artists:
- artists: the listener's credited artists (pipeline/artists.py's rule)
  with the OUTSIDE_ARTISTS largest listening shares, ties by artist id.
- tracks: their Spotify tracks through artist_spotify_map
  (sp_artist_resolved) and track_artists, without the tracks that MLHD
  has: a row in item_spotify_map or item_spotify_fuzzy, or an ISRC whose
  MusicBrainz recording reaches an item through the canonical chain.
- one track per ISRC, then one per (title up to " - ", Spotify artists):
  the highest Spotify popularity, ties by Spotify id.
- features, rebuilt to match P4 (pipeline/lb_outside.py): binary tags of
  the MusicBrainz recordings with the track's ISRC (tags_direct) and of
  its credited MusicBrainz artists (tags_artist), alias fold before the
  vocabulary; binary genres of the track's Spotify artists and of the
  Spotify artists of its credited artists (SpotifyBridge.py's two paths);
  credits: the MusicBrainz artists that artist_spotify_map links to the
  track's Spotify artists, in the catalog's credit columns; audio z-scores
  with P4's unweighted means and SDs, tempo 0 as missing. No co-listening
  (the cold-track GAM has no such term). Tier "isrc" (no term of the final
  model reads the tier).
- score: p14.term_sums() with the cold-track GAM's table. OUTSIDE_N = 25
  tracks, ties by Spotify id.
- reasons: contributions against the same background (the catalog
  candidates under the cold-track table). The cold-track GAM has no
  replicates, so D1 and S4 cannot apply: a reason is shown when it is a
  candidate, positive and in its track's top 3. The page marks these
  reasons "no stability rule".
- audit: up to AUDIT_TRACKS history tracks with Spotify audio get their
  features again through the outside route. Their audio z-scores must
  equal the catalog's exactly. The agreement of the term sets and credits
  is descriptive, because the two routes read other links by design.

Checks while it runs, each one stops the run:
- every MBID that recording_canonical_map holds reaches the same canonical
  MBID through the chain (the map was built by this chain).
- the term sums over the whole catalog equal catalog.reference_scores() on
  concepts.full_catalog() exactly (an independent concept route), for the
  final model and the cold-track GAM.
- per term, the background from the basis sums equals the dense mean over
  the candidates within CHECK_ATOL.
- the reason checks of P14: contributions sum to the score minus the mean
  candidate score (CHECK_ATOL), term values sum to the score (SUM_ATOL),
  raw and centered tables give the same contributions, every shown reason
  is positive, every shown content reason sits on a non-decreasing part
  of its shape. The same sums for the outside list.
- ItemKNN: the scores equal a second route (itemknn.brute_scores) within
  itemknn_select.SCORE_RTOL.
- the outside scores equal catalog.reference_scores() on their concepts.
- the audit's audio z-scores (above).
- the page's numbers are read back from the JSON file.

Outputs: MODEL_DIR/demo/<name>/lb_demo.html (the page) and lb_demo.json
(every number on it), in the data folder, never in the repository: they
hold Spotify-derived values (genres, audio), which never leave this
machine. Lines go to the console and to MODEL_DIR/logs/lb_demo.log. The
fetch prints its share done with the time left.

Resume: the API pages and the outside tracks' raw data are saved under
SCRATCH_DIR/lb_demo/<name>. After Ctrl+C the same command continues.

Expected run on the real data: the fetch about 1 second per 1,000 listens,
loading the item side and the models about a minute, the ranking and its
dense check under a minute, the outside list a few minutes of DuckDB scans
over the Spotify and MusicBrainz tables (saved for the next run).

PART 2, rules written before any run on real data. A presentation change
only.

Outputs.
- One document, written twice in MODEL_DIR/demo/<name>/: lb_demo.json and
  demo_data.js, one line "window.DEMO = <the same JSON>;". A page opened
  from disk (file://) cannot fetch JSON, so the page reads the data through
  a script tag.
- The page is the static file demo/index.html of the repository. The run
  copies it to MODEL_DIR/demo/index.html byte for byte (a copy there
  that differs, an edit or an older page, is first kept as
  index.old.html) and writes
  MODEL_DIR/demo/users.js with the names of the folders there that hold a
  demo_data.js. index.html?user=<name> opens one of them (default: the
  first name). The code writes no HTML. The page shows only what the code
  wrote, and its filters (below).

Lists. Each list stores its STORE_N = 500 best tracks (fewer when it has
fewer candidates). The page shows TOP_N = 25 after its filters.
- gam: part 1's ranking and reasons.
- itemknn, same_artist and both: part 1's ranking, now with a reason.
- fresh: new, below.
- outside: part 1's list (OUTSIDE_N = STORE_N), with the drop of known songs.
- popularity: only without a history (no match and no seed artist), with
  the note "No matched history". Its reason: the train listens.

Known artists. An artist is known when it is a credited artist of the
history, or when a raw listen names it: through an artist MBID of the
listen (track_metadata.additional_info.artist_mbids, then
mbid_mapping.artist_mbids, merge redirects included), or through the
listen's artist name, equal to a MusicBrainz artist name with case ignored.
Inputs.seed_artists() resolves both. A catalog track is known when one of
its credited artists is known. An outside track is known when one of its
MusicBrainz artists is known, or when one of its Spotify artist names equals
a raw artist name after lb_outside.norm_name() (NFKC, case folded,
typographic quotes folded, single spaces).

Fresh list: a declared ranking policy, not a model result. It is
descriptive, and P14 does not evaluate it.
- candidates: P5's candidates whose own other-artist list (P9b) holds a
  history track, with at least one credited artist and no known artist.
- score: ItemKNN's rule on the other-artist list alone, the sum of
  cosine x w_h over the history tracks h in the track's list. Equal scores
  rank by item id.
- caps: in rank order a track is kept unless one of its credited artists
  already has ARTIST_CAP = 5 kept tracks, or its main genre already has
  GENRE_CAP = 5 (apply_filters()). The stored list holds the caps.
- reason: co-listening (below).

Co-listening reason (both ItemKNN lists and the fresh list): the history
tracks h in the track's own lists of that kind ("both": either kind), each
with its term cosine x w_h, ordered by the term, then by item id. The terms
sum to the score. Words: "Listeners of <the top UNITS_SHOWN = 3 tracks>
and N more of your tracks also play this".

Outside list, known songs. Before the duplicates, a track is dropped when
its song key and one of its artists match an MLHD item or a raw listen.
- song key: lb_outside.song_key(), part 1's title key (the title up to " - ",
  lower case, single spaces) with typographic quotes folded and every
  character that is no letter, digit or space removed. An empty key
  matches nothing.
- MLHD item: a catalog track credited to one of the track's MusicBrainz
  artists, whose item_catalog title has the same song key.
- raw listen: a listen with the same song key (track_name) whose artist
  MBIDs resolve to one of the track's MusicBrainz artists, or whose
  normalized artist name equals one of the track's Spotify artist names.
- The counts give both kinds, an MLHD match first.

Per track the document holds rank, key ("i<item>", outside "s<Spotify
id>"), title, artist, score, details and reasons, and three filter keys:
- artist_keys: "a<MusicBrainz artist id>" of each credited artist (outside:
  of each MusicBrainz artist, else "s<Spotify artist id>"). A track
  without one gets "k<its key>", so no other track counts against its cap.
- genre: the main genre, shown on the page. Of the track's genres, the one
  with the largest idf^2 x listening share (the order of the genre
  evidence), then the larger idf, then the smaller column. Without a
  history every share is 0, so the larger idf decides. A track without
  genres has none, and no genre cap applies to it.
- known: above. part 3: None for a catalog track without a credited
  artist, whose artists nobody can tell. "Only artists you do not know"
  keeps a track only when known is False.

Per reason the document holds label, strength, words, details and the
numbers.
- words: the top UNITS_SHOWN = 3 evidence units, and "N more" for the rest.
- strength: from the percentile rank of the reason's value among the
  list's candidates, 100 (below + equal / 2) / n. strong from
  STRONG_PCT = 99, moderate from MODERATE_PCT = 90, else slight. A shown
  model reason lies above the candidate mean, so slight still means above
  the average candidate.
  - gam: the reason's group value (the sum of its terms' values) under the
    final model's table, among P5's candidates.
  - outside: the group value under the cold-track table, among P5's
    candidates of the catalog (the background of these reasons).
  - itemknn: the score among the list's candidates with a positive score
    (ItemKNN lists only those). fresh: the score among its candidates.
  - popularity: the train listens among every catalog track.
- details: the numbers behind the words. The odds factor exp(value) of a
  model reason shows only there.

Match summary: listens, matched share, and the RAW_ARTISTS = 10 most played
artist names of the raw listens (share of all listens, and the share of
the history weight that matched artists of the same normalized name hold)
against the RAW_ARTISTS largest credited artists of the matched history
(share of the history weight, and the share of all listens with the same
normalized artist name).

Filters, on the page only (client side). They walk a list in its stored
order and keep a track unless "only artists you do not know" is on and the
track is known, or one of its artist keys already has the per-artist cap
of kept tracks, or its genre already has the per-genre cap. They stop at
the shown count. Defaults: 5 per artist, 5 per genre, all artists, 25
shown. A cap whose box is not ticked is off. They never change a score or
the stored order. apply_filters() is the same rule in Python, for the
tests.

Checks while it runs, in addition to part 1's (each one stops the run):
- the co-listening terms of every listed track sum to its score within
  itemknn_select.SCORE_RTOL.
- every list holds each track key once.
- the fresh list holds no known artist and keeps its caps.
- no outside track matches a known song.
- every percentile lies in [0, 100].
- lb_demo.json and demo_data.js read back to the same document, and the
  copied page equals the repository's page.

PART 3, rules written after the first runs of part 2 and before the next
one. Those runs showed that "only artists you do not know" empties both
model lists, because all 500 stored tracks of each are by known artists.
Where part 3 and part 2 differ, part 3 holds.
1. New-artist tracks in both model lists. A new-artist track: at least one
   credited artist and no known one (the fresh list's rule, an outside
   track by the outside rule above). Every row's rank is now its place in
   the list's full ranking, so the stored ranks can jump. The note of each
   list gives the rank of its best new-artist track.
   - gam: the list also stores the best STORE_N new-artist candidates under
     the final model (P5's candidates, ties by item id), with their reasons
     under the same rules.
   - outside: the candidate artists are the OUTSIDE_ARTISTS largest artists
     of the history (part 1) and NEW_ARTISTS = 100 new artists: the first
     distinct credited artists of the new-artist catalog candidates in the
     cold-track GAM's ranking (cold-track table, ties by item id, the
     artists of one track by id). One pass builds the tracks of both sets.
     The cold-track GAM ranks every kept track (ties by Spotify id), and
     the list stores the first STORE_N tracks and the first STORE_N tracks
     that are not known.
2. Song keys (lb_outside.song_key and song_keys). Before part 2's song key, a
   bracketed part ("(...)", "[...]", or an open bracket to the end) is cut
   when it holds a version word (lb_outside.VERSION_RE: remaster, mix,
   live, demo, "take" with a number, version, edit, mono, acoustic and
   more) or starts with "with ". So "Free As A Bird (2025 Mix)" meets "Free
   As a Bird", and "Song (Part 2)" keeps its part. A quoted title inside a
   cut part gives an outside track a second key, so a working title such as
   "Dusty ("Poor Twisted Me" Take 12)" also meets "Poor Twisted Me". A
   track is a known song when one of its keys and one of its artists match
   (MLHD titles and raw listens give their first key). The duplicates rule
   keeps one track per (song key, Spotify artists), in place of part 1's title
   key.
3. Known artists by name: a raw artist name resolves by exact name only
   when no listen with that name carries an artist MBID. A name with MBIDs
   already names its artists, and its namesakes are other artists (the
   first run: 3,094 names gave 4,753 artists). The MBID route and the
   Spotify-name check of outside tracks do not change.
More checks: each list runs in rank order with unique ranks, the model
lists' scores do not rise with the rank, and the stored new-artist tracks
of each model list equal its first STORE_N tracks that are not known.

Usage, from the repository root:
    python3 -m pipeline.lb_demo --user NAME
    python3 -m pipeline.lb_demo --export ~/Downloads/listenbrainz_export.zip
    python3 -m pipeline.lb_demo --seed-artist Radiohead --seed-artist Bjork
Then open parquet/model/demo/index.html of the data folder in a browser.
"""

import argparse
import datetime as dt
import hashlib
import html
import http.client
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import catalog as ct
from pipeline import concepts as cn
from pipeline import evaluation as ev
from pipeline import gam
from pipeline import itemknn as knn
from pipeline import itemknn_select as iks
from pipeline import lb_outside as lo
from pipeline import p13c as pc
from pipeline import p14
from pipeline import stage_p13b2 as s13b2
from pipeline import stage_p4 as s4
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    connect,
    features_table,
    load_json,
    log,
)
from pipeline.config import CURATED, MODEL_DIR, ROOT, SCRATCH_DIR
from pipeline.script_stage import LOG_FORMAT, log_file

VERSION = "lb-demo-v2.1"
FETCH_VERSION = "lb-demo-v1"  # the saved pages' format, kept apart
CACHE_VERSION = "lb-demo-v1"  # the saved outside queries' format, kept apart
DECLARED = "before any run on real data"
DECLARED_V2 = "before any run on real data (part 2)"
DECLARED_V21 = "before the next run on real data (part 3)"
WORK_NAME = "lb_demo"
API_ROOT = "https://api.listenbrainz.org/1"
USER_AGENT = "thesis-hybrid-recommender-demo/1.0 (+https://listenbrainz.org)"
PAGE_SIZE = 1000  # the first page size
API_MAX = 1000  # the largest page the API gives
MIN_PAGE = 100
TRIES_PER_SIZE = 2
RATE_WAITS = 30
HTTP_TIMEOUT = 60
RETRIES = 6
TOP_N = 25  # shown on the page after its filters
STORE_N = 500  # stored per list
EVIDENCE_K = p14.EVIDENCE_K
SEED_TRACKS = 10
OUTSIDE_ARTISTS = 100
OUTSIDE_N = STORE_N
NEW_ARTISTS = 100  # part 3, 1
AUDIT_TRACKS = 200
ARTIST_CAP = 5  # the page's default and the fresh list's cap
GENRE_CAP = 5
UNITS_SHOWN = 3  # evidence units in the words of a reason
STRONG_PCT = 99.0
MODERATE_PCT = 90.0
RAW_ARTISTS = 10
DATA_FILE = "demo_data.js"
USERS_FILE = "users.js"
PAGE_FILE = "index.html"
OLD_PAGE_FILE = "index.old.html"
JS_PREFIX = "window.DEMO = "
USERS_PREFIX = "window.DEMO_USERS = "
PAGE_SRC = Path(__file__).resolve().parent.parent / "demo" / PAGE_FILE
CHECK_ATOL = p14.CHECK_ATOL
SUM_ATOL = p14.SUM_ATOL
W_ONE = round(1000.0 * math.log1p(1.0)) / 1000.0
GAM, COLD = 0, 2  # rows of p14.load_models()["tables"]
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-" r"[0-9a-f]{12}$"
)
TIERS = ("redirect", "canonical", "merged")
MISSES = ("no MBID", "no canonical recording", "canonical recording outside MLHD")

CANON_DIR = ROOT / "parquet" / "canonical"
MB_DIR = ROOT / "parquet" / "musicbrainz" / "fullexport" / "mbdump"
MBD_DIR = ROOT / "parquet" / "musicbrainz" / "fullexport" / "mbdump-derived"
SP_DIR = ROOT / "kaggle" / "spotify-catalogue"
SP_CLEAN = SP_DIR / "spotify_clean_parquet"
SP_AUDIO = (
    SP_DIR / "spotify_clean_audio_features_parquet" / "track_audio_features.parquet"
)
AUDIO_STATS = MODEL_DIR / "audio_std_stats.json"
REASON_LABELS = {
    "artist": "Artist",
    "tag_affinity": "Track tags",
    "artist_tag_affinity": "Artist tags",
    "genre_affinity": "Genres",
    "audio_fit": "Sound",
    "popularity": "Popularity",
}
V2_LABELS = {"colisten": "Co-listening", "train_listens": "Popularity"}
# the lists of the page: (id, path in the document, label)
LISTS = (
    ("gam", ("gam",), "Explainable model (GAM)"),
    ("itemknn_same", ("itemknn", "same_artist"), "ItemKNN, same artist"),
    ("itemknn_both", ("itemknn", "both"), "ItemKNN, both lists"),
    ("fresh", ("fresh", "tracks"), "Fresh: new artists (declared policy)"),
    ("outside", ("outside", "tracks"), "Not in MLHD (cold-track GAM)"),
    ("popularity", ("popularity",), "Popular in the training data"),
)


# Listens


def as_mbid(x):
    """x as a lower-case MBID, or None when it is no UUID."""
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    return s if UUID_RE.match(s) else None


def _ts(x):
    """listened_at as Unix seconds: a number, a string of digits or an ISO
    8601 time. None when it is none of them."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x) if math.isfinite(x) else None
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
        try:
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return int(t.timestamp())
    return None


def parse_listen(obj):
    """One listen of an API page or an export line as a dict, or None
    without listened_at (module docstring, listens)."""
    if not isinstance(obj, dict):
        return None
    ts = _ts(obj.get("listened_at"))
    if ts is None:
        return None
    tm = obj.get("track_metadata") or {}
    info = tm.get("additional_info") or {}
    mm = tm.get("mbid_mapping") or {}
    mbids = []
    for src, x in (
        ("submitted", info.get("recording_mbid")),
        ("mapped", mm.get("recording_mbid")),
    ):
        m = as_mbid(x)
        if m and m not in [y for _, y in mbids]:
            mbids.append((src, m))
    artist_mbids = []  # part 2, known artists
    for x in (info.get("artist_mbids"), mm.get("artist_mbids")):
        for y in (x if isinstance(x, list) else []):
            m = as_mbid(y)
            if m and m not in artist_mbids:
                artist_mbids.append(m)
    return {
        "ts": ts,
        "msid": str(obj.get("recording_msid") or ""),
        "artist": str(tm.get("artist_name") or ""),
        "track": str(tm.get("track_name") or ""),
        "release": str(tm.get("release_name") or ""),
        "mbids": mbids,
        "artist_mbids": artist_mbids,
    }


def listen_key(rec):
    return (rec["ts"], rec["msid"], rec["artist"], rec["track"])


def unique_listens(objs):
    """The parsed listens, each once, newest first."""
    out, seen = [], set()
    for obj in objs:
        rec = parse_listen(obj)
        if rec is None:
            continue
        k = listen_key(rec)
        if k not in seen:
            seen.add(k)
            out.append(rec)
    out.sort(key=lambda r: (-r["ts"], r["msid"], r["artist"], r["track"]))
    return out


def _listens_in(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from (
                _listens_in(x)
                if isinstance(x, list)
                else ([x] if isinstance(x, dict) else [])
            )
    elif isinstance(obj, dict):
        if isinstance(obj.get("payload"), dict):
            yield from _listens_in(obj["payload"].get("listens") or [])
        elif isinstance(obj.get("listens"), list):
            yield from _listens_in(obj["listens"])
        elif "track_metadata" in obj or "listened_at" in obj:
            yield obj


def _parse_blob(text, name):
    s = text.lstrip()
    if not name.lower().endswith(".jsonl") and s[:1] in ("[", "{"):
        try:
            yield from _listens_in(json.loads(text))
            return
        except json.JSONDecodeError:
            pass  # JSON Lines under a .json name
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield from _listens_in(json.loads(line))


def read_export(path):
    """Listen objects of a ListenBrainz export: a .zip (its .jsonl and
    .json members under a path part that starts with "listens", or every
    member when none does), a .jsonl file or a .json file (a list of
    listens, an API page or one listen per line). Other members yield no
    listen, because a listen needs listened_at."""
    p = Path(path).expanduser()
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            names = sorted(
                n
                for n in z.namelist()
                if not n.endswith("/") and n.lower().endswith((".jsonl", ".json"))
            )
            use = [
                n
                for n in names
                if any(part.lower().startswith("listens") for part in n.split("/"))
            ]
            use = use or names
            for n in use:
                yield from _parse_blob(z.read(n).decode("utf-8"), n)
        return
    yield from _parse_blob(p.read_text(encoding="utf-8"), p.name)


# The API


def _urlopen(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp is not None else b""
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, body


class NoAnswer(RuntimeError):
    """The server gave no answer to a request in its tries."""


def get_json(url, say, opener=None, sleep=time.sleep, tries=RETRIES):
    """The JSON of a GET. A network error, a cut or broken answer and a 5xx
    answer are failed tries and wait 2, 4, 8 ... seconds. NoAnswer after
    `tries` failed tries. HTTP 429 waits X-RateLimit-Reset-In seconds and
    is no failed try (at most RATE_WAITS waits). Any other answer than 200
    stops the run."""
    failed = waits = 0
    while True:
        why = None
        try:
            status, headers, body = (opener or _urlopen)(url)
        except (http.client.HTTPException, OSError) as exc:
            # OSError holds URLError, timeouts and closed connections,
            # HTTPException a body cut short (IncompleteRead)
            status, headers, body = None, {}, b""
            why = f"network error ({str(exc) or type(exc).__name__})"
        h = {str(k).lower(): v for k, v in (headers or {}).items()}
        if status == 429:
            waits += 1
            if waits > RATE_WAITS:
                raise RuntimeError(f"ListenBrainz kept its rate limit for " f"{url}")
            wait = float(h.get("x-ratelimit-reset-in") or 5) + 1.0
            say(f"{WORK_NAME}: ListenBrainz rate limit, next try in " f"{wait:.0f} s")
            sleep(wait)
            continue
        if status == 200:
            try:
                data = json.loads(body.decode("utf-8"))
            except ValueError:
                why = f"a broken answer ({len(body):,} bytes)"
            else:
                if str(h.get("x-ratelimit-remaining", "1")) == "0":
                    sleep(float(h.get("x-ratelimit-reset-in") or 1) + 0.5)
                return data
        elif status is not None and status >= 500:
            why = f"ListenBrainz answered {status}"
        elif status is not None:
            raise RuntimeError(
                f"ListenBrainz answered {status} for {url}: " f"{body[:300]!r}"
            )
        failed += 1
        if failed >= tries:
            raise NoAnswer(f"no answer to {url} in {failed} tries, last: " f"{why}")
        wait = 2.0**failed
        say(f"{WORK_NAME}: {why}, next try in {wait:.0f} s")
        sleep(wait)


def fetch_listens(
    user,
    cache_dir,
    say,
    opener=None,
    refresh=False,
    sleep=time.sleep,
    page_size=PAGE_SIZE,
):
    """Every listen of user from the API, as raw objects, walked back in
    time through max_ts. Each page with new listens is saved as
    page_<n>.json, and state.json holds the next max_ts, so a stopped
    fetch continues. A page ends at a second that later listens may share,
    so the next request repeats that second (max_ts = oldest + 1) and the
    keys drop the repeats. A page of known listens steps past its oldest
    second. An empty page ends the walk. A page without an answer is asked
    again with half the listens (all RETRIES tries at MIN_PAGE), a full
    page inside one second with twice the listens (module docstring,
    change to the fetch)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "state.json"
    state = load_json(state_path) or {}
    if refresh or state.get("user") != user or state.get("version") != FETCH_VERSION:
        for f in cache_dir.glob("page_*.json"):
            f.unlink()
        state = {
            "user": user,
            "version": FETCH_VERSION,
            "pages": 0,
            "max_ts": None,
            "done": False,
        }
        atomic_json_write(state, state_path)
    raw, seen = [], set()
    for i in range(1, int(state["pages"]) + 1):
        page = load_json(cache_dir / f"page_{i:05d}.json")
        if page is None:
            raise RuntimeError(
                f"{cache_dir / f'page_{i:05d}.json'} is "
                f"missing or broken: run again with "
                f"--refresh"
            )
        for obj in page["listens"]:
            rec = parse_listen(obj)
            if rec is not None:
                seen.add(listen_key(rec))
        raw.extend(page["listens"])
    if state["done"]:
        say(
            f"{WORK_NAME}: {len(seen):,} listens of {user} are saved in "
            f"{cache_dir} (--refresh fetches them again)"
        )
        return raw
    quoted = urllib.parse.quote(user, safe="")
    try:
        total = int(
            get_json(f"{API_ROOT}/user/{quoted}/listen-count", say, opener, sleep)[
                "payload"
            ]["count"]
        )
    except (KeyError, TypeError, ValueError):
        total = 0
    say(
        f"{WORK_NAME}: fetching the listens of {user} from ListenBrainz "
        f"({total:,} listens by its count, {len(seen):,} saved before)"
    )
    prog = Progress(
        f"{WORK_NAME}: fetch", max(total, len(seen), 1), say, unit=" listens"
    )
    prog.advance(len(seen))
    max_ts = state["max_ts"]
    size = int(min(max(int(page_size), 1), API_MAX))
    grown_at = ()  # (max_ts,) of the last page that grew
    hint = (
        f"The {{n}} saved pages stay, and the same command continues "
        f"later. An export needs no API: python3 -m pipeline.lb_demo "
        f"--export <file>"
    )
    while True:
        url = f"{API_ROOT}/user/{quoted}/listens?count={size}"
        if max_ts is not None:
            url += f"&max_ts={int(max_ts)}"
        last = size <= MIN_PAGE or grown_at == (max_ts,)
        try:
            payload = (
                get_json(
                    url, say, opener, sleep, tries=RETRIES if last else TRIES_PER_SIZE
                ).get("payload")
                or {}
            )
        except NoAnswer as exc:
            if last:
                raise RuntimeError(
                    f"ListenBrainz gave no answer for the listens before "
                    f"max_ts {max_ts}, {size} at a time ({exc}). "
                    + hint.format(n=state["pages"])
                ) from exc
            size = max(MIN_PAGE, size // 2)
            say(
                f"{WORK_NAME}: no answer for this page, so the fetch asks "
                f"for {size} listens per page from here on"
            )
            continue
        listed = payload.get("listens") or []
        page = [obj for obj in listed if parse_listen(obj) is not None]
        if not page:
            state["done"] = True
            atomic_json_write(state, state_path)
            break
        recs = [parse_listen(obj) for obj in page]
        if len(listed) >= size and len({r["ts"] for r in recs}) == 1:
            if size >= API_MAX:
                raise RuntimeError(
                    f"more than {API_MAX} listens share the second "
                    f"{recs[0]['ts']}, so max_ts cannot page through them. "
                    + hint.format(n=state["pages"])
                )
            size = min(API_MAX, 2 * size)
            grown_at = (max_ts,)
            say(
                f"{WORK_NAME}: a full page lies inside one second, so the "
                f"fetch asks for {size} listens per page"
            )
            continue
        new = [obj for obj, r in zip(page, recs) if listen_key(r) not in seen]
        oldest = min(r["ts"] for r in recs)
        if new and (max_ts is None or oldest + 1 < max_ts):
            nxt = oldest + 1
        else:
            nxt = oldest
        if max_ts is not None and nxt >= max_ts:
            raise RuntimeError(
                "the fetch made no progress: ListenBrainz "
                "returned listens at or after max_ts"
            )
        if new:
            state["pages"] = int(state["pages"]) + 1
            atomic_json_write(
                {"max_ts": max_ts, "listens": new},
                cache_dir / f"page_{state['pages']:05d}.json",
            )
            raw.extend(new)
            for obj in new:
                seen.add(listen_key(parse_listen(obj)))
            prog.advance(len(new))
        state["max_ts"] = nxt
        atomic_json_write(state, state_path)
        max_ts = nxt
    prog.finish(f", {len(seen):,} listens")
    return raw


# Matching


def chain_canonical(row):
    """(canonical, tier, ambiguous) of one MBID from the chain's columns:
    (c_redirect, n_redirect, c_canonical, merged_gid, n_merged,
    merged_canonical, merged_redirect, n_merged_redirect)."""
    c1, n1, c2, g, ng, c3a, c3b, n3 = row
    if c1 is not None:
        return c1, "redirect", int(n1) > 1
    if c2 is not None:
        return c2, "canonical", False
    if g is not None:
        canon = c3a if c3a is not None else c3b
        if canon is not None:
            return canon, "merged", int(ng) > 1 or (c3a is None and int(n3) > 1)
    return None, None, False


def check_map(resolved, mapped):
    """Raise ValueError unless every MBID of recording_canonical_map
    reaches the map's canonical MBID through the chain. mapped: {mbid:
    canonical or None}."""
    bad = [
        m for m, c in mapped.items() if (resolved.get(m) or {}).get("canonical") != c
    ]
    if bad:
        m = bad[0]
        raise ValueError(
            f"{len(bad)} MBIDs of recording_canonical_map reach "
            f"another canonical MBID through the chain, for "
            f"example {m}: map {mapped[m]}, chain "
            f"{(resolved.get(m) or {}).get('canonical')}. The "
            f"canonical tables changed after BuildFeatures.py"
        )
    return len(mapped)


def match_listens(listens, resolved):
    """(item per listen, 0 for a miss, the match report). resolved:
    {mbid: {"canonical", "tier", "item", "ambiguous"}}."""
    item = np.zeros(len(listens), dtype=np.int64)
    tiers, sources, misses = Counter(), Counter(), Counter()
    rec_first, rec_hit = set(), set()
    for i, rec in enumerate(listens):
        if not rec["mbids"]:
            misses[MISSES[0]] += 1
            continue
        first = rec["mbids"][0][1]
        rec_first.add(first)
        for src, m in rec["mbids"]:
            r = resolved.get(m) or {}
            if int(r.get("item") or 0) > 0:
                item[i] = int(r["item"])
                tiers[r["tier"]] += 1
                sources[src] += 1
                rec_hit.add(first)
                break
        else:
            canon = any(
                (resolved.get(m) or {}).get("canonical") for _, m in rec["mbids"]
            )
            misses[MISSES[2] if canon else MISSES[1]] += 1
    n = len(listens)
    with_mbid = n - misses[MISSES[0]]
    matched = int(np.count_nonzero(item))
    report = {
        "listens": n,
        "with_mbid": int(with_mbid),
        "matched": matched,
        "share_of_listens": matched / n if n else None,
        "share_of_listens_with_mbid": matched / with_mbid if with_mbid else None,
        "recordings": len(rec_first),
        "recordings_matched": len(rec_hit),
        "share_of_recordings": (len(rec_hit) / len(rec_first) if rec_first else None),
        "items": int(len(np.unique(item[item > 0]))),
        "tiers": {t: int(tiers[t]) for t in TIERS},
        "sources": {s: int(sources[s]) for s in ("submitted", "mapped")},
        "misses": {m: int(misses[m]) for m in MISSES},
        "ambiguous_mbids": int(sum(1 for r in resolved.values() if r.get("ambiguous"))),
        "first_listen_utc": _utc(min(r["ts"] for r in listens)) if n else None,
        "last_listen_utc": _utc(max(r["ts"] for r in listens)) if n else None,
    }
    return item, report


def _utc(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def history_of(items):
    """(items sorted, plays, weights) from the matched item per listen."""
    hit = items[items > 0]
    if not len(hit):
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64))
    ids, plays = np.unique(hit, return_counts=True)
    w = np.round(1000.0 * np.log1p(plays.astype(np.float64))) / 1000.0
    return ids.astype(np.int64), plays.astype(np.int64), w


class Inputs:
    """The demo's DuckDB queries on the real data. The smoke test
    replaces this class with one that answers from the synthetic world."""

    def _rows(self, sql, frames=None, threads=8):
        import pandas as pd

        con = connect(threads)
        try:
            for name, cols in (frames or {}).items():
                con.register(name, pd.DataFrame(cols))
            return con.execute(sql).fetchall()
        finally:
            con.close()

    # The chain

    def resolve(self, mbids):
        """{mbid: {"canonical", "tier", "item", "ambiguous"}} through the
        canonical chain (module docstring, mapping), and {mbid: canonical}
        of the MBIDs that recording_canonical_map holds."""
        mbids = sorted(set(mbids))
        if not mbids:
            return {}, {}
        redir = CANON_DIR / "canonical_recording_redirect.parquet"
        data = CANON_DIR / "canonical_musicbrainz_data.parquet"
        gid_redirect = MB_DIR / "recording_gid_redirect.parquet"
        recording = MB_DIR / "recording.parquet"
        q = {"lb_q": {"mbid": mbids}}
        rows = self._rows(
            f"""
            WITH q AS (SELECT DISTINCT mbid FROM lb_q),
            redir AS (
                SELECT recording_mbid AS mbid,
                       min(canonical_recording_mbid) AS canon,
                       count(DISTINCT canonical_recording_mbid) AS n
                FROM read_parquet('{redir}')
                WHERE recording_mbid IN (SELECT mbid FROM q)
                GROUP BY recording_mbid),
            canon AS (
                SELECT DISTINCT recording_mbid AS mbid
                FROM read_parquet('{data}')
                WHERE recording_mbid IN (SELECT mbid FROM q)),
            merged AS (
                SELECT g.gid AS mbid, min(r.gid) AS new_gid,
                       count(DISTINCT r.gid) AS n
                FROM read_parquet('{gid_redirect}') g
                JOIN read_parquet('{recording}') r
                  ON r.id = g.new_id
                WHERE g.gid IN (SELECT mbid FROM q)
                GROUP BY g.gid),
            m_canon AS (
                SELECT DISTINCT recording_mbid AS new_gid
                FROM read_parquet('{data}')
                WHERE recording_mbid IN (SELECT new_gid FROM merged)),
            m_redir AS (
                SELECT recording_mbid AS new_gid,
                       min(canonical_recording_mbid) AS canon,
                       count(DISTINCT canonical_recording_mbid) AS n
                FROM read_parquet('{redir}')
                WHERE recording_mbid IN (SELECT new_gid FROM merged)
                GROUP BY recording_mbid)
            SELECT q.mbid, redir.canon, coalesce(redir.n, 0), canon.mbid,
                   merged.new_gid, coalesce(merged.n, 0), m_canon.new_gid,
                   m_redir.canon, coalesce(m_redir.n, 0)
            FROM q
            LEFT JOIN redir ON redir.mbid = q.mbid
            LEFT JOIN canon ON canon.mbid = q.mbid
            LEFT JOIN merged ON merged.mbid = q.mbid
            LEFT JOIN m_canon ON m_canon.new_gid = merged.new_gid
            LEFT JOIN m_redir ON m_redir.new_gid = merged.new_gid
            ORDER BY q.mbid""",
            q,
        )
        out = {}
        for r in rows:
            canon, tier, amb = chain_canonical(r[1:])
            out[str(r[0])] = {
                "canonical": canon,
                "tier": tier,
                "item": 0,
                "ambiguous": bool(amb),
            }
        canons = sorted({v["canonical"] for v in out.values() if v["canonical"]})
        if canons:
            item_of = dict(
                self._rows(
                    f"""
                SELECT canonical_recording_mbid, item_id
                FROM read_parquet('{features_table("item_ids")}')
                WHERE canonical_recording_mbid IN (SELECT c FROM lb_c)""",
                    {"lb_c": {"c": canons}},
                )
            )
            for v in out.values():
                v["item"] = int(item_of.get(v["canonical"], 0) or 0)
        mapped = {
            str(m): (None if c is None else str(c))
            for m, c in self._rows(
                f"""
            SELECT recording_mbid, canonical_recording_mbid
            FROM read_parquet('{features_table("recording_canonical_map")}')
            WHERE recording_mbid IN (SELECT mbid FROM lb_q)""",
                q,
            )
        }
        return out, mapped

    # Names

    def track_names(self, item_ids):
        """{item: (artist, title, release)}: one item_catalog row per item,
        names present first, then the most listens, then the smallest MBID
        (stage_p12's rule)."""
        ids = sorted({int(i) for i in item_ids if int(i) > 0})
        if not ids:
            return {}
        rows = self._rows(
            f"""
            SELECT i.item_id, c.artist_name, c.recording_name, c.release_name
            FROM read_parquet('{features_table("item_ids")}') AS i
            JOIN read_parquet('{features_table("item_catalog")}') AS c
              ON c.canonical_recording_mbid = i.canonical_recording_mbid
            WHERE i.item_id IN (SELECT item_id FROM lb_items)
            QUALIFY row_number() OVER (
                PARTITION BY i.item_id
                ORDER BY c.recording_name IS NOT NULL DESC,
                         c.artist_name IS NOT NULL DESC,
                         c.listens DESC, c.recording_mbid) = 1""",
            {"lb_items": {"item_id": np.asarray(ids, dtype=np.int64)}},
        )
        return {int(r[0]): (r[1], r[2], r[3]) for r in rows}

    def artist_names(self, artist_ids):
        """{MusicBrainz artist row id: name}."""
        ids = sorted({int(i) for i in artist_ids})
        if not ids:
            return {}
        rows = self._rows(
            f"""
            SELECT CAST(a.id AS BIGINT), a.name
            FROM read_parquet('{MB_DIR / "artist.parquet"}') a
            WHERE CAST(a.id AS BIGINT) IN (SELECT id FROM lb_ids)""",
            {"lb_ids": {"id": np.asarray(ids, dtype=np.int64)}},
        )
        return {int(r[0]): str(r[1]) for r in rows}

    def popularity(self, n_slots, id_min):
        """P5's popularity: train listens per slot, -1 on phantom slots."""
        con = connect(4)
        try:
            return np.asarray(
                ev.popularity_scores(con, n_slots, id_min), dtype=np.int64
            )
        finally:
            con.close()

    def seed_artists(self, specs):
        """[(artist id, MBID, name, asked MBID or None)] of the MusicBrainz
        artists that a spec names: its MBID, an old MBID merged into it,
        or its exact name (case ignored)."""
        mbids = sorted({as_mbid(s) for s in specs if as_mbid(s)}) or [""]
        names = sorted({s.strip().lower() for s in specs if not as_mbid(s)}) or [""]
        art = MB_DIR / "artist.parquet"
        rows = self._rows(
            f"""
            SELECT CAST(a.id AS BIGINT), a.gid, a.name, NULL
            FROM read_parquet('{art}') a
            WHERE a.gid IN (SELECT m FROM lb_m)
               OR lower(a.name) IN (SELECT n FROM lb_n)
            UNION ALL
            SELECT CAST(a.id AS BIGINT), a.gid, a.name, r.gid
            FROM read_parquet('{MB_DIR / "artist_gid_redirect.parquet"}') r
            JOIN read_parquet('{art}') a ON a.id = r.new_id
            WHERE r.gid IN (SELECT m FROM lb_m)""",
            {"lb_m": {"m": mbids}, "lb_n": {"n": names}},
        )
        return [
            (int(r[0]), str(r[1]), str(r[2]), None if r[3] is None else str(r[3]))
            for r in rows
        ]

    # Outside list

    def spotify_tracks_of(self, mb_ids):
        """The Spotify tracks of MusicBrainz artists: artist_spotify_map
        (sp_artist_resolved) to the artists table, track_artists, tracks,
        and the album's name and release date. Rows: [track rowid,
        Spotify id, name, raw ISRC, popularity, duration_ms, album name,
        release date, the smallest MusicBrainz artist id that led there]."""
        ids = sorted({int(i) for i in mb_ids})
        if not ids:
            return []
        rows = self._rows(
            f"""
            WITH m AS (
                SELECT DISTINCT CAST(mb_artist_id AS BIGINT) AS mb,
                       sp_artist_resolved AS sp
                FROM read_parquet('{features_table("artist_spotify_map")}')
                WHERE CAST(mb_artist_id AS BIGINT) IN (SELECT id FROM lb_mb)),
            a AS (
                SELECT ar.rowid AS arow, min(m.mb) AS mb
                FROM read_parquet('{SP_CLEAN / "artists.parquet"}') ar
                JOIN m ON ar.id = m.sp
                GROUP BY ar.rowid),
            t AS (
                SELECT ta.track_rowid, min(a.mb) AS mb
                FROM read_parquet('{SP_CLEAN / "track_artists.parquet"}') ta
                JOIN a ON ta.artist_rowid = a.arow
                GROUP BY ta.track_rowid)
            SELECT tr.rowid, tr.id, tr.name, tr.external_id_isrc,
                   tr.popularity, tr.duration_ms, al.name, al.release_date,
                   t.mb
            FROM read_parquet('{SP_CLEAN / "tracks.parquet"}') tr
            JOIN t ON tr.rowid = t.track_rowid
            LEFT JOIN read_parquet('{SP_CLEAN / "albums.parquet"}') al
              ON al.rowid = tr.album_rowid
            ORDER BY tr.rowid""",
            {"lb_mb": {"id": np.asarray(ids, dtype=np.int64)}},
        )
        return [
            [
                int(r[0]),
                str(r[1]),
                r[2],
                r[3],
                None if r[4] is None else int(r[4]),
                None if r[5] is None else int(r[5]),
                r[6],
                r[7],
                int(r[8]),
            ]
            for r in rows
        ]

    def spotify_raw(self, track_rowids, track_ids=()):
        """The raw links of Spotify tracks (by rowid or by Spotify id) for
        lb_outside's rules, as plain lists (see build_outside): their
        artists, the MusicBrainz artists that artist_spotify_map links both
        ways, the Spotify genres, the MusicBrainz artist tags, the
        MusicBrainz recordings with their ISRC and those recordings' tags,
        the audio features (typed as SpotifyBridge.py casts them, null
        responses left out) and the tracks that a bridge to MLHD holds.
        One scan per table serves both kinds of key."""
        rid = sorted({int(r) for r in track_rowids})
        tid = sorted({str(t) for t in track_ids})
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
        if not rid and not tid:
            return out
        out["tracks"] = [
            [int(r[0]), str(r[1]), r[2]]
            for r in self._rows(
                f"""
            SELECT rowid, id, external_id_isrc
            FROM read_parquet('{SP_CLEAN / "tracks.parquet"}')
            WHERE rowid IN (SELECT r FROM lb_t)
               OR id IN (SELECT i FROM lb_i)""",
                {
                    "lb_t": {"r": np.asarray(rid or [-1], dtype=np.int64)},
                    "lb_i": {"i": tid or [""]},
                },
            )
        ]
        rid = sorted({t[0] for t in out["tracks"]})
        if not rid:
            return out
        T = {"lb_t": {"r": np.asarray(rid, dtype=np.int64)}}
        out["track_artists"] = [
            [int(a), int(b)]
            for a, b in self._rows(
                f"""
            SELECT DISTINCT track_rowid, artist_rowid
            FROM read_parquet('{SP_CLEAN / "track_artists.parquet"}')
            WHERE track_rowid IN (SELECT r FROM lb_t)""",
                T,
            )
        ]
        arows = sorted({b for _, b in out["track_artists"]})
        art = SP_CLEAN / "artists.parquet"
        amap = features_table("artist_spotify_map")
        if arows:
            out["artists"] = [
                [int(a), str(b), c]
                for a, b, c in self._rows(
                    f"""
                SELECT rowid, id, name FROM read_parquet('{art}')
                WHERE rowid IN (SELECT r FROM lb_a)""",
                    {"lb_a": {"r": np.asarray(arows, dtype=np.int64)}},
                )
            ]
        sp_ids = sorted({a[1] for a in out["artists"]})
        if sp_ids:
            out["sp2mb"] = [
                [str(a), int(b)]
                for a, b in self._rows(
                    f"""
                SELECT DISTINCT sp_artist_resolved,
                       CAST(mb_artist_id AS BIGINT)
                FROM read_parquet('{amap}')
                WHERE sp_artist_resolved IN (SELECT s FROM lb_s)""",
                    {"lb_s": {"s": sp_ids}},
                )
            ]
        mbs = sorted({b for _, b in out["sp2mb"]})
        if mbs:
            M = {"lb_m": {"id": np.asarray(mbs, dtype=np.int64)}}
            out["mb2sp"] = [
                [int(a), str(b)]
                for a, b in self._rows(
                    f"""
                SELECT DISTINCT CAST(mb_artist_id AS BIGINT),
                       sp_artist_resolved
                FROM read_parquet('{amap}')
                WHERE CAST(mb_artist_id AS BIGINT) IN (SELECT id FROM lb_m)""",
                    M,
                )
            ]
            out["artist_tags"] = [
                [int(a), str(b)]
                for a, b in self._rows(
                    f"""
                SELECT DISTINCT CAST(t.artist AS BIGINT), g.name
                FROM read_parquet('{MBD_DIR / "artist_tag.parquet"}') t
                JOIN read_parquet('{MBD_DIR / "tag.parquet"}') g
                  ON g.id = t.tag
                WHERE CAST(t.artist AS BIGINT) IN (SELECT id FROM lb_m)""",
                    M,
                )
            ]
        extra = sorted({b for _, b in out["mb2sp"]} - set(sp_ids))
        if extra:
            out["artists"] += [
                [int(a), str(b), c]
                for a, b, c in self._rows(
                    f"""
                SELECT rowid, id, name FROM read_parquet('{art}')
                WHERE id IN (SELECT s FROM lb_s)""",
                    {"lb_s": {"s": extra}},
                )
            ]
        all_rows = sorted({a[0] for a in out["artists"]})
        if all_rows:
            out["genres"] = [
                [int(a), str(b)]
                for a, b in self._rows(
                    f"""
                SELECT DISTINCT artist_rowid, genre
                FROM read_parquet('{SP_CLEAN / "artist_genres.parquet"}')
                WHERE artist_rowid IN (SELECT r FROM lb_a)""",
                    {"lb_a": {"r": np.asarray(all_rows, dtype=np.int64)}},
                )
            ]
        isrcs = sorted({lo.norm_isrc(t[2]) for t in out["tracks"]} - {None})
        if isrcs:
            norm = "upper(regexp_replace(isrc, '[^A-Za-z0-9]', '', 'g'))"
            out["isrc_rec"] = [
                [str(a), str(b)]
                for a, b in self._rows(
                    f"""
                SELECT DISTINCT {norm}, recording
                FROM read_parquet('{MB_DIR / "isrc.parquet"}')
                WHERE regexp_replace(isrc, '[^A-Za-z0-9]', '', 'g')
                      ~ '^[A-Za-z0-9]{{12}}$'
                  AND {norm} IN (SELECT i FROM lb_i)""",
                    {"lb_i": {"i": isrcs}},
                )
            ]
        recs = sorted({b for _, b in out["isrc_rec"]})
        if recs:
            R = {"lb_r": {"r": recs}}
            out["rec_gid"] = [
                [str(a), str(b)]
                for a, b in self._rows(
                    f"""
                SELECT id, gid
                FROM read_parquet('{MB_DIR / "recording.parquet"}')
                WHERE id IN (SELECT r FROM lb_r)""",
                    R,
                )
            ]
            out["rec_tags"] = [
                [str(a), str(b)]
                for a, b in self._rows(
                    f"""
                SELECT DISTINCT t.recording, g.name
                FROM read_parquet('{MBD_DIR / "recording_tag.parquet"}') t
                JOIN read_parquet('{MBD_DIR / "tag.parquet"}') g
                  ON g.id = t.tag
                WHERE t.recording IN (SELECT r FROM lb_r)""",
                    R,
                )
            ]
        ids = sorted({t[1] for t in out["tracks"]})
        if ids:
            S = {"lb_s": {"s": ids}}
            cols = ", ".join(
                f"CAST({c} AS {'BIGINT' if c == 'duration_ms' else 'DOUBLE'})"
                for c in s4.CONT_COLS
            )
            out["audio"] = [
                [str(r[0])] + [None if x is None else float(x) for x in r[1:]]
                for r in self._rows(
                    f"""
                SELECT track_id, {cols}
                FROM read_parquet('{SP_AUDIO}')
                WHERE track_id IN (SELECT s FROM lb_s)
                  AND NOT COALESCE(CAST(null_response AS BOOLEAN), FALSE)""",
                    S,
                )
            ]
            bmap = features_table("item_spotify_map")
            bfuzzy = features_table("item_spotify_fuzzy")
            out["bridged"] = sorted(
                {
                    str(r[0])
                    for r in self._rows(
                        f"""
                SELECT sp_track FROM read_parquet('{bmap}')
                WHERE sp_track IN (SELECT s FROM lb_s)
                UNION
                SELECT sp_track FROM read_parquet('{bfuzzy}')
                WHERE sp_track IN (SELECT s FROM lb_s)""",
                        S,
                    )
                }
            )
        return out

    def catalog_tracks(self, item_ids):
        """{item: Spotify track id} of catalog items: the track that
        item_audio_features_ext holds, the source of their audio."""
        ids = sorted({int(i) for i in item_ids})
        if not ids:
            return {}
        rows = self._rows(
            f"""
            SELECT item_id, sp_track
            FROM read_parquet('{features_table("item_audio_features_ext")}')
            WHERE item_id IN (SELECT i FROM lb_i)""",
            {"lb_i": {"i": np.asarray(ids, dtype=np.int64)}},
        )
        return {int(a): str(b) for a, b in rows}

    def audio_stats(self):
        doc = load_json(AUDIO_STATS)
        if doc is None:
            raise ValueError(f"{AUDIO_STATS} is missing")
        return doc["columns"]

    def tag_aliases(self):
        return s4._read_tag_aliases(CURATED / "tag_aliases.txt")


# Model


def load_model_side(st, model_dir, say):
    """(base, models of p14.load_models, item side, sources, slots)."""
    t0 = time.time()
    base = st._base_context()
    G = p14.load_models(Path(model_dir), base)
    n_slots, id_min = st._id_slots()
    say(
        f"{WORK_NAME}: final model {G['rec']['fit_name']} ({len(G['terms'])} "
        f"terms), {len(G['boot']['listener'])} listener replicates, S3 set "
        f"of {len(G['members'])}, cold-track GAM ({len(G['cold']['kept'])} "
        f"terms). Loading the item side (no progress lines in this step)"
    )
    src = st._sources(n_slots)
    space = st._space(src)
    say(f"{WORK_NAME}: item side ready in {_clock(time.time() - t0)}")
    return base, G, space, src, n_slots, id_min


def term_sum_tables(G):
    return np.asarray(G["tables"][[GAM, COLD]], dtype=np.float64)


def score_catalog(terms, T, space, state):
    """The concepts of every slot (catalog.concept_catalog, one source at
    a time) and their term sums for the tables T (p14.term_sums)."""
    rows = np.flatnonzero(space.audio_mask.any(axis=1))
    conc = {}
    for t in terms:
        if t.source not in conc:
            conc[t.source] = ct.concept_catalog(state, space, t.source, audio_rows=rows)
    return p14.term_sums(terms, T, conc, space.tier), conc


def top_slots(s, cand, k):
    """The k best candidates of s, equal scores by slot (P5's rule)."""
    idx = np.flatnonzero(cand)
    if len(idx) > 4 * k:
        thr = np.partition(s[idx], len(idx) - k)[len(idx) - k]
        idx = idx[s[idx] >= thr]
    return idx[np.lexsort((idx, -s[idx]))[:k]]


def dense_check(terms, T, space, state, S):
    """Raise ValueError unless the term sums equal catalog.reference_scores()
    on concepts.full_catalog() for every slot, exactly."""
    full = cn.full_catalog(state, space)
    ref = ct.reference_scores(full, terms, T, 0.0, space.tier)
    if not np.array_equal(S, ref):
        gap = float(np.max(np.abs(S - ref)))
        raise ValueError(
            f"the term sums differ from the dense reference " f"(largest gap {gap:.1e})"
        )
    return int(S.size)


def basis_sum(terms, conc, tier, cand):
    """The sum of the basis vectors over the candidates: per term, (1 - w)
    at the slot and w at the next (gam.encode), as P14's backgrounds."""
    offs = p14.offsets_of(terms)
    L = int(sum(t.size for t in terms))
    out = np.zeros(L)
    for t, off in zip(terms, offs):
        v, h = conc[t.source]
        slot, w = gam.encode(t, v, h, tier)
        s_ = off + slot[cand].astype(np.int64)
        w_ = w[cand]
        out += np.bincount(s_, weights=1.0 - w_, minlength=L) + np.bincount(
            s_ + 1, weights=w_, minlength=L
        )
    return out


def check_backgrounds(terms, T, conc, tier, cand, bsum):
    """Raise ValueError unless each term's background from the basis sums
    equals its dense mean over the candidates within CHECK_ATOL. Returns
    the largest gap."""
    n_cand = float(cand.sum())
    worst = 0.0
    for j, (t, off) in enumerate(zip(terms, p14.offsets_of(terms))):
        for c in range(len(T)):
            Tj = np.zeros(T.shape[1])
            Tj[off : off + t.size] = T[c, off : off + t.size]
            dense = float(p14.term_sums(terms, Tj, conc, tier)[0][cand].mean())
            bg = float(bsum[off : off + t.size] @ Tj[off : off + t.size]) / n_cand
            worst = max(worst, abs(dense - bg))
            if abs(dense - bg) > CHECK_ATOL:
                raise ValueError(
                    f"term {t.name}, table {c}: the basis "
                    f"background differs from the dense mean "
                    f"({abs(dense - bg):.1e})"
                )
    return worst


# Reasons


def top_encoding(terms, conc, tier, slots):
    """(1, R, J) slots and positions of the term sources at the top
    slots (gam.encode), the input of p14.TopReasons."""
    J = len(terms)
    slot = np.empty((1, len(slots), J), dtype=np.int16)
    w = np.empty((1, len(slots), J))
    for j, t in enumerate(terms):
        v, h = conc[t.source]
        slot[0, :, j], w[0, :, j] = gam.encode(t, v[slots], h[slots], tier[slots])
    return slot, w


def check_top_reasons(R, G, top_score, mean_score):
    """P14's sum and shift checks (p14.check_reasons) for any list
    length. top_score: (R,). mean_score: the mean candidate score."""
    M = G["M"]
    T0 = np.asarray(M["tables"], dtype=np.float64)
    phi = R.contributions(T0)
    s = np.asarray(top_score, dtype=np.float64)[None, :]
    gap_sum = float(np.max(np.abs(phi.sum(axis=0) - (s - mean_score))))
    if gap_sum > CHECK_ATOL:
        raise ValueError(
            f"the contributions do not sum to the score minus "
            f"the mean candidate score ({gap_sum:.1e})"
        )
    v, _ = R.parts(T0)
    gap_v = float(np.max(np.abs(v.sum(axis=0) - s) / np.maximum(1.0, np.abs(s))))
    if gap_v > SUM_ATOL:
        raise ValueError(f"the term values do not sum to the score " f"({gap_v:.1e})")
    raw = R.contributions(M["lay"].tables(M["theta"]))
    gap_shift = float(np.max(np.abs(raw - phi)))
    if gap_shift > CHECK_ATOL:
        raise ValueError(
            f"the raw and the centered tables give other "
            f"contributions ({gap_shift:.1e})"
        )
    return {"sum": gap_sum, "term_values": gap_v, "shift": gap_shift}


def check_direction(R, T, shown, val):
    """Every shown reason is positive and every shown content reason (all
    terms bounded) sits on a non-decreasing part of its shape."""
    if np.any(val[shown] <= 0):
        raise ValueError("a shown reason is not positive")
    for g, js in enumerate(R.gix):
        if all(R.terms[j].monotone for j in js) and shown[g].any():
            cls = p14.local_slope(R, T, js[0])[shown[g]]
            if np.any(cls == -1):
                raise ValueError(
                    f"a shown {R.groups[g][0]} reason sits on "
                    f"a falling part of a bounded shape"
                )


# Evidence


def _term_name(src, source, i):
    names = (src.term_names or {}).get(source) or []
    return names[int(i)] if 0 <= int(i) < len(names) else f"{source} #{i}"


def evidence_units(ev_row, name, src, terms_g):
    """The evidence of one reason as display units. ev_row: (count, ids,
    vals) of p14._evidence for the reason's group."""
    count, ids, vals = ev_row
    units = []
    source = None
    for t in terms_g:
        if t in cn.TERM_CONCEPTS:
            source = cn.TERM_CONCEPTS[t]
    for i, v in zip(ids, vals):
        if int(i) < 0:
            continue
        if name == "audio_fit":
            units.append(
                {"id": int(i), "name": cn.AUDIO_DIMS[int(i)][2:], "term": float(v)}
            )
        elif name == "artist":
            units.append({"id": int(i), "term": float(v)})
        else:
            units.append(
                {"id": int(i), "name": _term_name(src, source, i), "term": float(v)}
            )
    return {"count": int(count), "units": units}


# The lists


def knn_lists(space, items, weights, k):
    """({"same_artist", "both": [(slot, score)]}, the largest relative gap
    to the second route, {kind and "both": the score of every slot}): the
    k best candidates with a positive score, ties by slot. The transposed
    lists are the item side's (itemknn.transposed builds the same
    matrices)."""
    t_mats = {kind: space.colisten_t[kind] for kind in knn.KINDS}
    sc = knn.user_scores(t_mats, items, weights)
    lists = {
        kind: (
            np.asarray(space.colisten[kind].indptr, np.int64),
            np.asarray(space.colisten[kind].indices, np.int32),
            np.asarray(space.colisten[kind].data, np.float64),
        )
        for kind in knn.KINDS
    }
    slow = knn.brute_scores(lists, items, weights, space.n_slots)
    worst = 0.0
    out = {}
    for name in ("same_artist", "both"):
        scale = max(1.0, float(np.max(np.abs(slow[name]))))
        gap = float(np.max(np.abs(sc[name] - slow[name]))) / scale
        worst = max(worst, gap)
        if gap > iks.SCORE_RTOL:
            raise ValueError(
                f"ItemKNN {name}: the scores differ from the "
                f"second route ({gap:.1e})"
            )
        s = knn.candidate_scores(sc[name], items)
        cand = np.isfinite(s) & (s > 0.0)
        top = top_slots(s, cand, k)
        out[name] = [(int(t), float(s[t])) for t in top]
    return out, worst, sc


def popularity_list(pop, k):
    """P5's popularity order (train listens, ties by slot), top k."""
    order = ev.global_ranks(np.asarray(pop))
    top = np.argsort(order, kind="stable")[:k]
    return [(int(t), int(pop[t])) for t in top if pop[t] >= 0]


def seed_history(io, specs, src, pop, say):
    """(items, weights, report) of the seed artists (module docstring)."""
    ind, cols, n_cols = src.credits["artists"]
    id_of_col = np.asarray(src.credit_ids["artists"], dtype=np.int64)
    rows_of = np.repeat(np.arange(len(ind) - 1), np.diff(ind))
    found = io.seed_artists(specs)
    pop = np.asarray(pop, dtype=np.int64)
    picked, items = [], set()
    for spec in specs:
        m = as_mbid(spec)
        cands = [
            r
            for r in found
            if (m and (r[1] == m or r[3] == m))
            or (not m and r[2].strip().lower() == spec.strip().lower())
        ]
        best = None
        for a_id, gid, name, _ in sorted(set(cands)):
            pos = int(np.searchsorted(id_of_col, a_id))
            if pos >= len(id_of_col) or id_of_col[pos] != a_id:
                continue
            tracks = rows_of[cols == pos]
            mass = int(np.maximum(pop[tracks], 0).sum())
            if best is None or mass > best[0] or (mass == best[0] and a_id < best[1]):
                best = (mass, a_id, name, gid, tracks)
        if best is None:
            say(f"{WORK_NAME}: no catalog artist for the seed {spec!r}")
            picked.append({"seed": spec, "artist": None})
            continue
        mass, a_id, name, gid, tracks = best
        order = np.lexsort((tracks, -pop[tracks]))[:SEED_TRACKS]
        chosen = sorted(int(t) for t in tracks[order])
        items.update(chosen)
        picked.append(
            {
                "seed": spec,
                "artist": int(a_id),
                "name": name,
                "mbid": gid,
                "train_listens": mass,
                "tracks": chosen,
            }
        )
        say(f"{WORK_NAME}: seed {spec!r}: {name} ({gid}), {len(chosen)} " f"tracks")
    it = np.asarray(sorted(items), dtype=np.int64)
    return it, np.full(len(it), W_ONE), picked


# Outside list


def outside_artists(state, src):
    """The listener's credited artists with the OUTSIDE_ARTISTS largest
    shares, ties by artist id: (MusicBrainz ids, shares)."""
    cols = np.asarray(state.credit_cols["artists"], dtype=np.int64)
    share = np.asarray(state.credit_share["artists"], dtype=np.float64)
    ids = np.asarray(src.credit_ids["artists"], dtype=np.int64)[cols]
    order = np.lexsort((ids, -share))[:OUTSIDE_ARTISTS]
    return ids[order], share[order]


def build_outside(
    tracks, raw, in_mlhd, aliases, src, stats, dedupe=True, song_drop=None
):
    """The outside tracks and their P4 rows from the raw links
    (lb_outside's rules). tracks: dicts of spotify_tracks_of rows.
    in_mlhd: Spotify ids that MLHD has. dedupe=False keeps every track in
    input order (the audit). song_drop(track): "mlhd", "listens" or None,
    the known-song rule of part 2, applied before the duplicates. Returns
    (kept tracks, term CSRs, credit CSR, audio (n, D), counts)."""
    t_art = {}
    for t, a in raw["track_artists"]:
        t_art.setdefault(int(t), set()).add(int(a))
    a_id = {int(a[0]): str(a[1]) for a in raw["artists"]}
    a_name = {int(a[0]): (a[2] or "") for a in raw["artists"]}
    a_row = {str(a[1]): int(a[0]) for a in raw["artists"]}
    sp2mb, mb2sp = {}, {}
    for s, m in raw["sp2mb"]:
        sp2mb.setdefault(str(s), set()).add(int(m))
    for m, s in raw["mb2sp"]:
        mb2sp.setdefault(int(m), set()).add(str(s))
    genres, atags, rtags, recs = {}, {}, {}, {}
    for r, g in raw["genres"]:
        genres.setdefault(int(r), set()).add(str(g))
    for m, g in raw["artist_tags"]:
        atags.setdefault(int(m), set()).add(str(g))
    for r, g in raw["rec_tags"]:
        rtags.setdefault(str(r), set()).add(str(g))
    for i, r in raw["isrc_rec"]:
        recs.setdefault(str(i), set()).add(str(r))
    audio = {str(r[0]): r[1:] for r in raw["audio"]}
    rows = []
    n_in = 0
    known = Counter()
    for t in tracks:
        if t["sp_id"] in in_mlhd:
            n_in += 1
            continue
        arts = sorted(t_art.get(t["rowid"], ()))
        t = dict(
            t,
            sp_artists=tuple(a_id.get(a, str(a)) for a in arts),
            sp_artist_names=[a_name.get(a, "") for a in arts],
            artist_rowids=arts,
        )
        mbs = set()
        for s in t["sp_artists"]:
            mbs |= sp2mb.get(s, set())
        t["mb_artists"] = sorted(mbs)
        why = song_drop(t) if song_drop is not None else None
        if why:
            known[why] += 1
            continue
        rows.append(t)
    kept = lo.dedupe_tracks(rows) if dedupe else rows
    td, ta, gn, cr, au = [], [], [], [], []
    for t in kept:
        mbs = set(t["mb_artists"])
        rec_ids = sorted(recs.get(t["isrc"], ())) if t["isrc"] else []
        td.append(
            lo.fold(set().union(*[rtags.get(r, set()) for r in rec_ids]), aliases)
        )
        ta.append(lo.fold(set().union(*[atags.get(m, set()) for m in mbs]), aliases))
        g_rows = set(t["artist_rowids"])
        for m in mbs:
            g_rows |= {a_row[s] for s in mb2sp.get(m, ()) if s in a_row}
        gn.append(set().union(*[genres.get(r, set()) for r in g_rows]))
        cr.append(mbs)
        a = audio.get(t["sp_id"])
        au.append(
            [
                math.nan if a is None or x is None else float(x)
                for x in (a if a is not None else [None] * len(s4.CONT_COLS))
            ]
        )
    tag_index = {
        n: i for i, n in enumerate((src.term_names or {}).get("tags_direct") or [])
    }
    art_index = {
        n: i for i, n in enumerate((src.term_names or {}).get("tags_artist") or [])
    }
    gen_index = {n: i for i, n in enumerate((src.term_names or {}).get("genres") or [])}
    terms = {
        "tags_direct": lo.term_csr(td, tag_index),
        "tags_artist": lo.term_csr(ta, art_index),
        "genres": lo.term_csr(gn, gen_index),
    }
    credits = {"artists": lo.credit_csr(cr, src.credit_ids["artists"])}
    raw_a = {
        c: np.array([r[k] for r in au], dtype=np.float64).reshape(-1)
        for k, c in enumerate(s4.CONT_COLS)
    }
    if not kept:
        raw_a = {c: np.zeros(0) for c in s4.CONT_COLS}
    z = lo.audio_z(raw_a, stats)
    counts = {
        "tracks": len(tracks),
        "in_mlhd": n_in,
        "outside": len(tracks) - n_in,
        "known_song_mlhd": int(known["mlhd"]),
        "known_song_listens": int(known["listens"]),
        "kept": len(kept),
        "with_audio": int(np.count_nonzero(~np.isnan(z).all(axis=1))),
        "with_tags_direct": int(np.count_nonzero(np.diff(terms["tags_direct"][0]))),
        "with_tags_artist": int(np.count_nonzero(np.diff(terms["tags_artist"][0]))),
        "with_genres": int(np.count_nonzero(np.diff(terms["genres"][0]))),
    }
    return kept, terms, credits, z, counts


def in_mlhd_of(tracks, raw, resolved):
    """The Spotify ids that MLHD has: a bridge row, or an ISRC whose
    MusicBrainz recording reaches an item through the chain."""
    gid = {str(a): str(b) for a, b in raw["rec_gid"]}
    rec_items = {
        r: int((resolved.get(g) or {}).get("item") or 0) for r, g in gid.items()
    }
    isrc_hit = set()
    for i, r in raw["isrc_rec"]:
        if rec_items.get(str(r), 0) > 0:
            isrc_hit.add(str(i))
    bridged = set(raw["bridged"])
    return {
        t["sp_id"]
        for t in tracks
        if t["sp_id"] in bridged or (t["isrc"] and t["isrc"] in isrc_hit)
    }


def tracks_from_rows(rows):
    return [
        {
            "rowid": int(r[0]),
            "sp_id": str(r[1]),
            "name": r[2],
            "isrc": lo.norm_isrc(r[3]),
            "popularity": r[4],
            "duration_ms": r[5],
            "album": r[6],
            "release_date": r[7],
            "from_mb": int(r[8]),
        }
        for r in rows
    ]


def _cached(path, key, fn, say, label):
    """fn()'s JSON-able result, saved at path and keyed by key."""
    doc = load_json(path)
    k = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    if doc is not None and doc.get("k") == k:
        say(f"{WORK_NAME}: {label}: saved, so no query runs")
        return doc["v"]
    t0 = time.time()
    say(f"{WORK_NAME}: {label} with DuckDB (no progress lines in this step)")
    v = fn()
    atomic_json_write({"k": k, "v": v}, path)
    say(f"{WORK_NAME}: {label}: done in {_clock(time.time() - t0)}")
    return v


def artist_tracks(space, src, mb_ids):
    """(slots, artist ids): every catalog track credited to one of the
    artists mb_ids, once per (track, artist), from the item side's credit
    rows (pipeline/artists.py's rule)."""
    ids = np.asarray(src.credit_ids["artists"], dtype=np.int64)
    want = np.unique(np.asarray(list(mb_ids), dtype=np.int64))
    m = space.credit_mats["artists"]
    if not len(want) or not len(ids) or not m.nnz:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pos = np.minimum(np.searchsorted(ids, want), len(ids) - 1)
    cols = pos[ids[pos] == want]
    hit = np.isin(m.indices, cols)
    rows_of = np.repeat(np.arange(m.shape[0], dtype=np.int64), np.diff(m.indptr))
    return rows_of[hit], ids[m.indices[hit].astype(np.int64)]


def known_song_rule(titles, slots, arts, listens, known):
    """song_drop(track) of build_outside (module docstring, part 2,
    outside list, known songs). titles: {slot: item_catalog title} of the
    catalog tracks (slots, arts) of artist_tracks(). known: the dict of
    known_artists()."""
    mlhd = set()
    for s, a in zip(slots.tolist(), arts.tolist()):
        k = lo.song_key(titles.get(int(s)))
        if k:
            mlhd.add((k, int(a)))
    raw = set()
    by_mbid = known.get("by_mbid") or {}
    for rec in listens:
        k = lo.song_key(rec.get("track"))
        if not k:
            continue
        for m in rec.get("artist_mbids") or ():
            for a in by_mbid.get(m, ()):
                raw.add((k, "a", int(a)))
        n = lo.norm_name(rec.get("artist"))
        if n:
            raw.add((k, "n", n))

    def drop(t):
        ks = lo.song_keys(t.get("name"))  # part 3, 2
        if not ks:
            return None
        mbs = [int(a) for a in t.get("mb_artists") or ()]
        names = [lo.norm_name(n) for n in t.get("sp_artist_names") or ()]
        names = [n for n in names if n]
        if any((k, a) in mlhd for k in ks for a in mbs):
            return "mlhd"
        if any((k, "a", a) in raw for k in ks for a in mbs) or any(
            (k, "n", n) in raw for k in ks for n in names
        ):
            return "listens"
        return None

    drop.sizes = {"mlhd_songs": len(mlhd), "listen_songs": len(raw)}
    return drop


def outside_known(t, known):
    """The known flag of an outside track (module docstring, part 2)."""
    return bool(
        any(int(m) in known["ids"] for m in t["mb_artists"])
        or any(
            lo.norm_name(n) in known["names"]
            for n in t["sp_artist_names"]
            if lo.norm_name(n)
        )
    )


def outside_part(
    io,
    state,
    space,
    src,
    G,
    bsum,
    n_cand,
    items,
    cache,
    say,
    listens=(),
    known=None,
    new_ids=(),
):
    """The outside list and the audit (module docstring). Returns (their
    JSON part, {reason group: values of the listed tracks under the
    cold-track table}, for the strengths of run()). One spotify_raw() call
    serves both, so each large table is scanned once. new_ids: the new
    artists of part 3, 1, whose tracks join the listener's artists'
    tracks."""
    terms = G["terms"]
    T = np.asarray(G["tables"][COLD], dtype=np.float64)
    mb_ids, shares = outside_artists(state, src)
    if not len(mb_ids):
        return {"skipped": "the history has no credited artist"}, {}
    known = known or {"ids": set(), "names": set(), "by_mbid": {}}
    old_ids = {int(x) for x in mb_ids}
    new_ids = [int(x) for x in new_ids if int(x) not in old_ids]
    all_ids = [int(x) for x in mb_ids] + new_ids
    items = np.asarray(items, dtype=np.int64)
    take = items[space.audio_mask[items].any(axis=1)][:AUDIT_TRACKS]
    key = {
        "version": CACHE_VERSION,
        "artists": [int(x) for x in mb_ids],
        "audit": [int(x) for x in take],
    }
    if new_ids:
        key["new_artists"] = new_ids
    rows = _cached(
        cache / "outside_tracks.json",
        key,
        lambda: {
            "tracks": io.spotify_tracks_of(all_ids),
            "audit": {str(k): v for k, v in io.catalog_tracks(take).items()},
        },
        say,
        f"outside list, step 1 of 4: the Spotify tracks of "
        f"{len(mb_ids)} of your artists and {len(new_ids)} "
        f"new artists, and of {len(take)} history tracks",
    )
    tracks = tracks_from_rows(rows["tracks"])
    audit_of = {int(k): str(v) for k, v in rows["audit"].items()}
    raw = _cached(
        cache / "outside_raw.json",
        dict(key, step="raw"),
        lambda: io.spotify_raw([t["rowid"] for t in tracks], sorted(audit_of.values())),
        say,
        f"outside list, step 2 of 4: the links of "
        f"{len(tracks) + len(audit_of):,} tracks",
    )
    gids = sorted({str(b) for _, b in raw["rec_gid"]})
    resolved, _ = io.resolve(gids)
    in_mlhd = in_mlhd_of(tracks, raw, resolved)
    a_slots, a_ids = artist_tracks(space, src, all_ids)
    t_slots = sorted({int(x) for x in a_slots})
    tkey = {
        "version": CACHE_VERSION,
        "step": "titles",
        "slots": hashlib.sha256(
            np.asarray(t_slots, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    titles = _cached(
        cache / "outside_titles.json",
        tkey,
        lambda: {str(i): v[1] for i, v in io.track_names(t_slots).items()},
        say,
        f"outside list, step 3 of 4: the titles of the "
        f"{len(t_slots):,} catalog tracks of these "
        f"artists",
    )
    titles = {int(k): v for k, v in titles.items()}
    drop = known_song_rule(titles, a_slots, a_ids, listens, known)
    aliases = lo.alias_map(io.tag_aliases())
    stats = io.audio_stats()
    kept, tcsr, ccsr, z, counts = build_outside(
        tracks, raw, in_mlhd, aliases, src, stats, song_drop=drop
    )
    counts.update(drop.sizes)
    if any(drop(t) for t in kept):
        raise ValueError(
            "an outside track that matches a known song is " "still in the list"
        )
    say(
        f"{WORK_NAME}: outside list, step 4 of 4: {counts['tracks']:,} "
        f"Spotify tracks, {counts['in_mlhd']:,} that MLHD has, "
        f"{counts['known_song_mlhd']:,} other versions of MLHD songs, "
        f"{counts['known_song_listens']:,} songs of the raw listens, "
        f"{counts['kept']:,} kept after the duplicates"
    )
    part = {
        "artists": [int(x) for x in mb_ids],
        "new_artists": new_ids,
        "counts": counts,
        "tracks": [],
        "audit": audit_part(audit_of, raw, space, src, aliases, stats, say),
    }
    if not kept:
        return part, {}
    no_credit = int(np.count_nonzero(np.diff(ccsr["artists"][0]) == 0))
    if no_credit:
        raise ValueError(
            f"{no_credit} outside tracks have no credit column, "
            f"but every track comes from an artist of the list"
        )
    space_out = lo.outside_space(space, tcsr, ccsr, z, cn.MATCH_TIERS.index("isrc"))
    conc = lo.concepts_of(state, space_out)
    tier = space_out.tier[1:]
    s = p14.term_sums(terms, T, conc, tier)[0]
    ref = ct.reference_scores(conc, terms, T, 0.0, tier)[0]
    if not np.array_equal(s, ref):
        raise ValueError(
            "the outside scores differ from "
            "catalog.reference_scores on their concepts"
        )
    # the ranking of every kept track, and the stored tracks: the first
    # OUTSIDE_N and the first OUTSIDE_N that are not known
    order_all = np.lexsort((np.arange(len(s)), -s))
    rank_all = np.empty(len(s), dtype=np.int64)
    rank_all[order_all] = np.arange(1, len(s) + 1)
    kn_all = np.array([outside_known(t, known) for t in kept], dtype=bool)
    new_idx = order_all[~kn_all[order_all]][:OUTSIDE_N]
    top = np.union1d(order_all[:OUTSIDE_N], new_idx)
    top = top[np.argsort(rank_all[top], kind="stable")]
    k = len(top)
    counts["new_artists"] = len(new_ids)
    counts["kept_new"] = int((~kn_all).sum())
    part["new_summary"] = {
        "kept": int((~kn_all).sum()),
        "stored": int(len(new_idx)),
        "first_rank": int(rank_all[new_idx[0]]) if len(new_idx) else None,
    }
    groups = pc.reason_groups(G["cold"]["kept"])
    names = [t.name for t in terms]
    offs = p14.offsets_of(terms)
    vals = np.zeros((len(terms), k))
    bg = np.zeros(len(terms))
    on = np.zeros((len(terms), k), dtype=bool)
    sub = {t.source: (conc[t.source][0][top], conc[t.source][1][top]) for t in terms}
    for j, (t, off) in enumerate(zip(terms, offs)):
        Tj = np.zeros_like(T)
        Tj[off : off + t.size] = T[off : off + t.size]
        vals[j] = p14.term_sums(terms, Tj, sub, tier[top])[0]
        bg[j] = float(bsum[off : off + t.size] @ Tj[off : off + t.size]) / n_cand
        slot, _ = gam.encode(t, *sub[t.source], tier[top])
        on[j] = slot < t.K
    gap_v = float(
        np.max(np.abs(vals.sum(axis=0) - s[top]) / np.maximum(1.0, np.abs(s[top])))
    )
    phi = vals - bg[:, None]
    gap_s = float(np.max(np.abs(phi.sum(axis=0) - (s[top] - bg.sum()))))
    if gap_v > SUM_ATOL or gap_s > CHECK_ATOL:
        raise ValueError(
            f"the outside term values do not sum to the score "
            f"({gap_v:.1e}) or the contributions to the margin "
            f"({gap_s:.1e})"
        )
    gv = np.stack([phi[[names.index(n) for n in g]].sum(axis=0) for _, g in groups])
    gc = np.stack([on[[names.index(n) for n in g]].any(axis=0) for _, g in groups])
    shown = lo.shown_top(gv, gc)
    if np.any(gv[shown] <= 0):
        raise ValueError("a shown outside reason is not positive")
    # the group values for the strengths, by the expression that run()
    # applies to the catalog's candidates (group_sums)
    gvals = {
        gname: group_sums(
            terms, [names.index(n) for n in g], T[None, :], sub, tier[top]
        )[0]
        for gname, g in groups
    }
    part["background"] = {t.name: float(b) for t, b in zip(terms, bg)}
    part["checks"] = {"term_values": gap_v, "sum": gap_s, "reference": True}
    gcol = main_genres(
        space_out.term_mats["genres"],
        space_out.idf["genres"],
        state.term_share["genres"],
        top + 1,
    )
    gnames = (src.term_names or {}).get("genres") or []
    prog = Progress(f"{WORK_NAME}: outside list, evidence", k, say, unit=" tracks")
    for r, i in enumerate(top):
        t = kept[int(i)]
        ev_ = lo.evidence(state, space, space_out, int(i) + 1)
        reasons = []
        for g, (gname, gterms) in enumerate(groups):
            if not shown[g, r]:
                continue
            x = outside_reason(gname, gterms, ev_, float(gv[g, r]), src)
            x["group_value"] = float(gvals[gname][r])
            reasons.append(x)
        reasons.sort(key=lambda x: -x["value"])
        mbs = [int(m) for m in t["mb_artists"]]
        sp_names = [a or "" for a in t["sp_artist_names"]]
        akeys = (
            [f"a{m}" for m in mbs]
            or sorted({f"s{a}" for a in t["sp_artists"]})
            or [f"ks{t['sp_id']}"]
        )
        year = str(t.get("release_date") or "")[:4]
        part["tracks"].append(
            {
                "rank": int(rank_all[i]),
                "key": f"s{t['sp_id']}",
                "sp_id": t["sp_id"],
                "title": t["name"],
                "artists": sp_names,
                "artist": ", ".join(x for x in sp_names if x) or None,
                "album": t["album"],
                "release_date": t["release_date"],
                "popularity": t["popularity"],
                "isrc": t["isrc"],
                "score": float(s[i]),
                "reasons": reasons,
                "mb_artists": mbs,
                "artist_keys": akeys,
                "genre": _genre_name(gnames, gcol[r]),
                "known": bool(kn_all[i]),
                "sub": ", ".join(x for x in (t["album"], year) if x) or None,
                "details": (
                    f"Cold-track GAM score {float(s[i]):.3f} (the sum of "
                    f"its term values). Spotify popularity "
                    f"{_or_na(t['popularity'])}, ISRC "
                    f"{t['isrc'] or 'none'}."
                ),
            }
        )
        prog.advance(1)
    prog.finish()
    got_new = sorted(int(i) for i, row in zip(top, part["tracks"]) if not row["known"])
    if got_new != sorted(int(x) for x in new_idx):
        raise ValueError(
            "outside list: the stored tracks by new artists "
            "differ from the best tracks that are not known"
        )
    years = [
        int(str(t["release_date"])[:4])
        for t in kept
        if str(t.get("release_date") or "")[:4].isdigit()
    ]
    part["counts"]["released_after_2013"] = int(sum(y > 2013 for y in years))
    part["counts"]["with_release_year"] = len(years)
    return part, gvals


def _or_na(x):
    return "n/a" if x is None else x


def outside_reason(gname, gterms, ev_, value, src):
    """One shown reason of the outside list with its evidence."""
    out = {
        "reason": gname,
        "value": value,
        "odds": _exp(value),
        "rule": "none (no replicates for the cold-track GAM)",
    }
    if gname == "artist":
        e = ev_["artist_affinity"]
        out["credit"] = int(np.asarray(src.credit_ids["artists"])[int(e["credit"])])
        out["credit_tracks"] = len(e["terms"])
        out["history_tracks"] = [int(x["id"]) for x in e["terms"]][:EVIDENCE_K]
        return out
    t = gterms[0]
    e = ev_[t]
    rows = e["terms"] if e["has_data"] else []
    if t == "audio_fit":
        rows = sorted(rows, key=lambda r: (-r["term"], r["id"]))
        units = [{"name": r["id"][2:], "term": float(r["term"])} for r in rows]
    else:
        source = cn.TERM_CONCEPTS[t]
        units = [
            {"name": _term_name(src, source, r["id"]), "term": float(r["term"])}
            for r in rows
        ]
    out["evidence"] = {"count": len(units), "units": units[:EVIDENCE_K]}
    return out


def _exp(v):
    try:
        return math.exp(v)
    except OverflowError:
        return math.inf


def audit_part(audit_of, raw, space, src, aliases, stats, say):
    """The audit of the outside route on history tracks (module
    docstring): audit_of maps a history item to the Spotify track of its
    audio. Raises ValueError when an audio z-score differs."""
    by_id = {str(r[1]): r for r in raw["tracks"]}
    tracks = []
    for i in sorted(audit_of):
        r = by_id.get(audit_of[i])
        if r is None:
            continue
        tracks.append(
            {
                "rowid": int(r[0]),
                "sp_id": str(r[1]),
                "name": f"item {int(i)}",
                "isrc": lo.norm_isrc(r[2]),
                "popularity": 0,
                "duration_ms": None,
                "album": None,
                "release_date": None,
                "from_mb": 0,
                "item": int(i),
            }
        )
    # the audit keeps every track: no MLHD filter and no duplicate rule
    kept, tcsr, ccsr, z, _ = build_outside(
        tracks, raw, set(), aliases, src, stats, dedupe=False
    )
    if not kept:
        return {"tracks": 0}
    it = np.asarray([t["item"] for t in kept], dtype=np.int64)
    cat = space.audio[it]
    same = np.array_equal(z, cat, equal_nan=True)
    if not same:
        bad = int(np.count_nonzero(~((z == cat) | (np.isnan(z) & np.isnan(cat)))))
        raise ValueError(
            f"the audit: {bad} audio z-scores of history tracks "
            f"rebuilt from Spotify differ from the catalog's"
        )
    agree = {}
    for name in ("tags_direct", "tags_artist", "genres"):
        ip, cols = tcsr[name]
        m = space.term_mats[name]
        agree[name] = _jaccards(ip, cols, m, it)
    ip, cols = ccsr["artists"]
    agree["credits"] = _jaccards(ip, cols, space.credit_mats["artists"], it)
    say(
        f"{WORK_NAME}: audit: {len(it)} history tracks rebuilt from Spotify, "
        f"audio z-scores equal the catalog's. Mean Jaccard agreement: "
        + ", ".join(
            f"{k} {v['mean']:.2f}" for k, v in agree.items() if v["mean"] is not None
        )
    )
    return {"tracks": int(len(it)), "audio_equal": True, "agreement": agree}


def _jaccards(ip, cols, m, items):
    """Mean Jaccard of the rebuilt column sets against the catalog rows,
    over the tracks where either side has a column."""
    vals = []
    for k, i in enumerate(items):
        a = set(int(x) for x in cols[ip[k] : ip[k + 1]])
        b = set(int(x) for x in m.indices[m.indptr[i] : m.indptr[i + 1]])
        if a or b:
            vals.append(len(a & b) / len(a | b))
    return {
        "tracks": len(vals),
        "mean": float(np.mean(vals)) if vals else None,
        "equal": int(sum(v == 1.0 for v in vals)),
    }


# Part 2


def apply_filters(
    rows, artist_cap=ARTIST_CAP, genre_cap=GENRE_CAP, only_new=False, show=None
):
    """The page's filters in Python (module docstring, part 2, Filters):
    the rows kept, in their stored order. A cap of None or 0 is off. show:
    the most rows to keep, None for all. only_new keeps a row only when its
    known flag is False (None: no credited artist, part 3). demo/index.html applies the same
    rule (applyFilters), and the tests compare the two."""
    out, n_art, n_gen = [], Counter(), Counter()
    for r in rows:
        if show is not None and len(out) >= show:
            break
        if only_new and r.get("known") is not False:
            continue
        arts = sorted(set(r.get("artist_keys") or ()))
        if artist_cap and any(n_art[a] >= artist_cap for a in arts):
            continue
        g = r.get("genre")
        if genre_cap and g is not None and n_gen[g] >= genre_cap:
            continue
        out.append(r)
        for a in arts:
            n_art[a] += 1
        if g is not None:
            n_gen[g] += 1
    return out


def percentile_ranks(pool_sorted, x):
    """100 (below + equal / 2) / n of each value of x among the sorted pool
    (module docstring, part 2, strength). NaN for an empty pool."""
    v = np.asarray(pool_sorted, dtype=np.float64)
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if not len(v):
        return np.full(len(x), np.nan)
    lo_ = np.searchsorted(v, x, "left")
    hi_ = np.searchsorted(v, x, "right")
    return 100.0 * (lo_ + 0.5 * (hi_ - lo_)) / len(v)


def strength_of(pct):
    """strong, moderate or slight (module docstring), None without a
    percentile."""
    if pct is None or not math.isfinite(float(pct)):
        return None
    return (
        "strong"
        if pct >= STRONG_PCT
        else "moderate" if pct >= MODERATE_PCT else "slight"
    )


def main_genres(m, idf, share, slots):
    """The main genre column of each slot (module docstring, part 2, filter
    keys), -1 for a slot without genres. m: a genres CSR over slots. share:
    the listener's genre shares, or None without a history."""
    slots = np.asarray(slots, dtype=np.int64)
    out = np.full(len(slots), -1, dtype=np.int64)
    if not len(slots):
        return out
    sub = m[slots]
    if not sub.nnz:
        return out
    row = np.repeat(np.arange(len(slots)), np.diff(sub.indptr))
    cols = sub.indices.astype(np.int64)
    idf = np.asarray(idf, dtype=np.float64)
    p = (
        np.asarray(share, dtype=np.float64)[cols]
        if share is not None
        else np.zeros(len(cols))
    )
    w = idf[cols] ** 2 * np.asarray(sub.data, dtype=np.float64) * p
    o = np.lexsort((cols, -idf[cols], -w, row))
    first = o[np.r_[0, np.flatnonzero(np.diff(row[o])) + 1]]
    out[row[first]] = cols[first]
    return out


def _genre_name(names, col):
    col = int(col)
    return names[col] if 0 <= col < len(names) else None


class CatalogKeys:
    """The filter keys of catalog tracks (module docstring, part 2):
    artist_keys, genre and known, for a list of slots."""

    def __init__(self, space, src, share, known_track):
        self.m = space.credit_mats["artists"]
        self.ids = np.asarray(src.credit_ids["artists"], dtype=np.int64)
        self.gm = space.term_mats["genres"]
        self.idf = space.idf["genres"]
        self.share = share
        self.names = (src.term_names or {}).get("genres") or []
        self.known = known_track

    def __call__(self, slots):
        slots = [int(s) for s in slots]
        g = main_genres(self.gm, self.idf, self.share, slots)
        out = []
        for s, c in zip(slots, g):
            cols = self.m.indices[self.m.indptr[s] : self.m.indptr[s + 1]]
            ak = sorted({f"a{int(self.ids[x])}" for x in cols}) or [f"ki{s}"]
            out.append(
                {
                    "artist_keys": ak,
                    "genre": _genre_name(self.names, c),
                    "known": bool(self.known[s]) if len(cols) else None,
                }
            )
        return out


def known_artists(io, listens, hist_ids):
    """The known artists (module docstring, part 2, and part 3, 3):
    {"ids": set of MusicBrainz artist ids, "names": normalized raw artist
    names (of every listen, for the outside rule), "by_mbid": {artist MBID
    of a listen: its artist ids}, "counts"}. A raw name resolves to
    MusicBrainz artists by name only when no listen with that name carries
    an artist MBID."""
    all_names = {r["artist"].strip() for r in listens if r["artist"].strip()}
    with_mbid = {
        r["artist"].strip()
        for r in listens
        if r["artist"].strip() and r.get("artist_mbids")
    }
    names_raw = sorted(all_names - with_mbid)
    mbids = sorted({m for r in listens for m in r.get("artist_mbids") or ()})
    found = io.seed_artists(mbids + names_raw) if (mbids or names_raw) else []
    mset, nset = set(mbids), {n.lower() for n in names_raw}
    by_mbid, by_name = {}, set()
    for a_id, gid, name, asked in found:
        for m in {gid, asked} & mset:
            by_mbid.setdefault(m, set()).add(int(a_id))
        if str(name).strip().lower() in nset:
            by_name.add(int(a_id))
    from_mbid = set().union(*by_mbid.values()) if by_mbid else set()
    hist = {int(a) for a in hist_ids}
    ids = hist | from_mbid | by_name
    return {
        "ids": ids,
        "names": {lo.norm_name(n) for n in all_names} - {""},
        "by_mbid": {k: sorted(v) for k, v in by_mbid.items()},
        "counts": {
            "history": len(hist),
            "raw_artist_mbids": len(mbids),
            "raw_artist_names": len(all_names),
            "names_without_mbid": len(names_raw),
            "by_mbid": len(from_mbid),
            "by_name": len(by_name),
            "known": len(ids),
        },
    }


def known_tracks(space, src, ids):
    """True for every slot with a known credited artist."""
    cid = np.asarray(src.credit_ids["artists"], dtype=np.int64)
    v = np.zeros(len(cid))
    a = np.asarray(sorted(ids), dtype=np.int64)
    if len(a) and len(cid):
        pos = np.minimum(np.searchsorted(cid, a), len(cid) - 1)
        v[pos[cid[pos] == a]] = 1.0
    return np.asarray(space.credit_mats["artists"] @ v).ravel() > 0.0


def group_sums(terms, js, T, conc, tier):
    """(C, n) values of one reason group, the terms js summed in term
    order, under the tables T (C, L): p14.term_sums on the group's terms
    alone. The same expression serves the pool and the listed tracks."""
    offs = p14.offsets_of(terms)
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    Tg = np.concatenate([T[:, offs[j] : offs[j] + terms[j].size] for j in js], axis=1)
    return p14.term_sums([terms[j] for j in js], Tg, conc, tier)


def reason_percentiles(terms, T, conc, tier, cand, groups, slots=None, values=None):
    """{group: percentile ranks} of each reason group among the candidates
    under the table T (L,) (module docstring, part 2, strength): of the
    group values at slots, or of the given values {group: (k,)}."""
    names = [t.name for t in terms]
    out = {}
    for gname, gterms in groups:
        if values is not None and gname not in values:
            continue
        js = [names.index(x) for x in gterms]
        S = group_sums(terms, js, T[None, :], conc, tier)[0]
        pool = np.sort(S[cand])
        x = S[np.asarray(slots, dtype=np.int64)] if values is None else values[gname]
        out[gname] = percentile_ranks(pool, x)
        del S, pool
    return out


def colisten_terms(space, items, weights, kinds, slot):
    """[(h, cosine x w_h)] of the history tracks h in the slot's own lists
    of these kinds, by the term descending, then by item (module
    docstring, co-listening reason). items: sorted."""
    rows = []
    for kind in kinds:
        m = space.colisten[kind]
        a, b = int(m.indptr[slot]), int(m.indptr[slot + 1])
        if a == b or not len(items):
            continue
        nb = m.indices[a:b].astype(np.int64)
        cs = m.data[a:b]
        pos = np.minimum(np.searchsorted(items, nb), len(items) - 1)
        hit = items[pos] == nb
        rows += [
            (int(h), float(c * weights[p]))
            for h, c, p in zip(nb[hit], cs[hit], pos[hit])
        ]
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def colisten_rows(space, state, ranked, kinds, pool_sorted, keys, label, say):
    """The rows of a co-listening list: ranked [(slot, score)], each with
    its co-listening reason. Raises ValueError when the terms of a track do
    not sum to its score."""
    pct = percentile_ranks(pool_sorted, [sc for _, sc in ranked])
    kk = keys([t for t, _ in ranked])
    rows = []
    prog = Progress(f"{WORK_NAME}: {label}, reasons", len(ranked), say, unit=" tracks")
    for r, ((t, sc), p, k) in enumerate(zip(ranked, pct, kk)):
        terms_ = colisten_terms(space, state.items, state.weights, kinds, t)
        tot = math.fsum(v for _, v in terms_)
        if abs(tot - sc) > iks.SCORE_RTOL * max(1.0, abs(sc)):
            raise ValueError(
                f"{label}: the co-listening terms of item {t} "
                f"sum to {tot!r}, its score is {sc!r}"
            )
        x = {
            "reason": "colisten",
            "value": float(sc),
            "percentile": float(p),
            "pool": int(len(pool_sorted)),
            "evidence": {
                "count": len(terms_),
                "units": [{"id": h, "term": v} for h, v in terms_[:UNITS_SHOWN]],
            },
        }
        rows.append(
            dict(
                {
                    "rank": r + 1,
                    "item": int(t),
                    "key": f"i{t}",
                    "score": float(sc),
                    "reasons": [x],
                    "details": "",
                },
                **k,
            )
        )
        prog.advance(1)
    prog.finish()
    return rows


def fresh_part(space, state, s_other, known_track, keys, say):
    """The fresh list (module docstring, part 2): its JSON part."""
    t0 = time.time()
    s = knn.candidate_scores(s_other, state.items)
    pos = np.isfinite(s) & (s > 0.0)
    credit = space.has_credit["artists"]
    elig = pos & credit & ~known_track
    idx = np.flatnonzero(elig)
    order = idx[np.lexsort((idx, -s[idx]))]
    pool = np.sort(s[idx])
    gcol = main_genres(
        space.term_mats["genres"],
        space.idf["genres"],
        state.term_share["genres"],
        order,
    )
    m = space.credit_mats["artists"]
    n_art, n_gen = Counter(), Counter()
    kept, visited = [], 0
    for k, i in enumerate(order):
        visited = k + 1
        cols = {int(c) for c in m.indices[m.indptr[i] : m.indptr[i + 1]]}
        if any(n_art[c] >= ARTIST_CAP for c in cols):
            continue
        g = int(gcol[k])
        if g >= 0 and n_gen[g] >= GENRE_CAP:
            continue
        kept.append(int(i))
        for c in cols:
            n_art[c] += 1
        if g >= 0:
            n_gen[g] += 1
        if len(kept) == STORE_N:
            break
    counts = {
        "positive": int(pos.sum()),
        "without_credit": int((pos & ~credit).sum()),
        "known_artist": int((pos & credit & known_track).sum()),
        "candidates": int(len(idx)),
        "visited": int(visited),
        "kept": len(kept),
    }
    say(
        f"{WORK_NAME}: fresh list: {counts['positive']:,} tracks with "
        f"other-artist co-listening, {counts['known_artist']:,} by known "
        f"artists, {counts['candidates']:,} candidates, {len(kept)} kept "
        f"under the caps after {visited:,} in "
        f"{_clock(time.time() - t0)}"
    )
    rows = colisten_rows(
        space,
        state,
        [(i, float(s[i])) for i in kept],
        ("other_artists",),
        pool,
        keys,
        "fresh list",
        say,
    )
    return {
        "tracks": rows,
        "counts": counts,
        "caps": {"artist": ARTIST_CAP, "genre": GENRE_CAP},
        "policy": "declared ranking policy, not a model result",
    }


def popularity_rows(pop, keys, k):
    """The popularity list (no history) with its reason."""
    pop = np.asarray(pop, dtype=np.int64)
    lst = popularity_list(pop, k)
    pool = np.sort(pop[pop >= 0].astype(np.float64))
    pct = percentile_ranks(pool, [c for _, c in lst])
    kk = keys([t for t, _ in lst])
    rows = []
    for r, ((t, c), p, kd) in enumerate(zip(lst, pct, kk)):
        x = {
            "reason": "train_listens",
            "value": int(c),
            "percentile": float(p),
            "pool": int(len(pool)),
        }
        rows.append(
            dict(
                {
                    "rank": r + 1,
                    "item": int(t),
                    "key": f"i{t}",
                    "train_listens": int(c),
                    "score": int(c),
                    "reasons": [x],
                    "details": f"{int(c):,} train listens.",
                },
                **kd,
            )
        )
    return rows


def artist_summary(listens, hist_ids, hist_share, names):
    """The artists of the match summary (module docstring, part 2)."""
    n = len(listens)
    raw = Counter(r["artist"].strip() for r in listens if r["artist"].strip())
    raw_norm = Counter()
    for a, c in raw.items():
        raw_norm[lo.norm_name(a)] += c
    hist_norm = {}
    for a, s in zip(hist_ids, hist_share):
        k = lo.norm_name(names.get(int(a)))
        if k:
            hist_norm[k] = hist_norm.get(k, 0.0) + float(s)
    top = sorted(raw.items(), key=lambda x: (-x[1], x[0]))[:RAW_ARTISTS]
    hist_ids = np.asarray(hist_ids, dtype=np.int64)
    hist_share = np.asarray(hist_share, dtype=np.float64)
    o = np.lexsort((hist_ids, -hist_share))[:RAW_ARTISTS]
    return {
        "listens": n,
        "raw_names": len(raw),
        "raw": [
            {
                "name": a,
                "listens": int(c),
                "share": c / n if n else None,
                "history_share": hist_norm.get(lo.norm_name(a), 0.0),
            }
            for a, c in top
        ],
        "matched": [
            {
                "artist": int(hist_ids[k]),
                "name": names.get(int(hist_ids[k])),
                "share": float(hist_share[k]),
                "listen_share": (
                    (raw_norm.get(lo.norm_name(names.get(int(hist_ids[k]))), 0) / n)
                    if n
                    else None
                ),
            }
            for k in o
        ],
    }


# Words
AUDIO_WORDS = {"duration_ms": "duration"}
LEADS = {
    "tag_affinity": "Track tags you play often",
    "artist_tag_affinity": "Artist tags you play often",
    "genre_affinity": "Genres you play often",
    "popularity": "Played often in the training data (2005 to 2013)",
}
POOL_WORDS = {
    "gam": "catalog candidates",
    "outside": "catalog candidates",
    "itemknn_same": "tracks with a score in this list",
    "itemknn_both": "tracks with a score in this list",
    "fresh": "fresh candidates",
    "popularity": "catalog tracks",
}


def _join(xs):
    xs = [x for x in xs if x]
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _titles(units, row):
    """The "title" of each unit, with its artist when that differs from the
    row's artist."""
    out = []
    for u in units:
        t = u.get("title") or f"item {u.get('id')}"
        a = u.get("artist")
        out.append(f"“{t}”" + (f" ({a})" if a and a != row.get("artist") else ""))
    return out


def _pct_text(p):
    return f"{math.floor(float(p) * 100.0) / 100.0:.2f}"


def reason_words(x, row):
    """The words of a reason (module docstring, part 2)."""
    ev_ = x.get("evidence") or {}
    units = ev_.get("units") or []
    shown = units[:UNITS_SHOWN]
    more = max(0, int(ev_.get("count") or 0) - len(shown))
    kind = x["reason"]
    rest = [f"{more} more of your tracks"] if more else []
    if kind == "artist":
        who = x.get("credit_name") or "this artist"
        n = int(x.get("credit_tracks") or 0)
        lead = f"You play {n} track{'' if n == 1 else 's'} by {who}"
        if shown:
            return (
                f"{lead}. Listeners of {_join(_titles(shown, row) + rest)}"
                f" also play it."
            )
        hist = (x.get("history_labels") or [])[:UNITS_SHOWN]
        if hist:
            return f"{lead}, among them {_join(_titles(hist, row))}."
        return f"{lead}."
    if kind == "colisten":
        return f"Listeners of {_join(_titles(shown, row) + rest)} also play " f"this."
    if kind == "audio_fit":
        dims = [
            AUDIO_WORDS.get(u.get("name"), str(u.get("name") or "").replace("_", " "))
            for u in shown
        ]
        return (
            f"Close to your usual {_join(dims)}."
            if dims
            else "Close to your usual sound."
        )
    if kind == "train_listens":
        return f"{int(x['value']):,} listens in the training data (2005 to " f"2013)."
    names = [str(u.get("name") or "") for u in shown]
    lead = LEADS.get(kind, REASON_LABELS.get(kind, kind))
    if not names:
        return f"{lead}."
    return f"{lead}: {_join(names + ([f'{more} more'] if more else []))}."


def reason_details(x, list_id):
    """The details line of a reason: the numbers behind its words."""
    p, pool = x.get("percentile"), x.get("pool")
    ptxt = (
        f" Percentile {_pct_text(p)} of {int(pool):,} "
        f"{POOL_WORDS.get(list_id, 'candidates')}."
        if p is not None and pool
        else ""
    )
    if x["reason"] == "colisten":
        n = int((x.get("evidence") or {}).get("count") or 0)
        return (
            f"Score {x['value']:.4f}: the sum of cosine × log(1 + plays) "
            f"over {n} of your tracks." + ptxt
        )
    if x["reason"] == "train_listens":
        return f"{int(x['value']):,} train listens." + ptxt
    s = (
        f"Contribution {x['value']:+.3f} log-odds against the average "
        f"candidate, odds factor {_odds(x.get('odds'))}." + ptxt
    )
    if list_id == "outside":
        s += " No stability rule: the cold-track GAM has no replicates."
    return s


def _get(doc, path):
    x = doc
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def finish_rows(doc):
    """Labels, strengths, words and details of every reason, and the
    note for a model row without a shown reason."""
    for lid, path, _ in LISTS:
        for r in _get(doc, path) or []:
            for x in r.get("reasons") or []:
                x["label"] = V2_LABELS.get(x["reason"]) or REASON_LABELS.get(
                    x["reason"], x["reason"]
                )
                x["strength"] = strength_of(x.get("percentile"))
                x["words"] = reason_words(x, r)
                x["details"] = reason_details(x, lid)
            if not r.get("reasons"):
                r["no_reason"] = "No reason passed the stability rules " "(D1 and S4)."


def list_meta(doc):
    """The lists of the page (module docstring, part 2): id, path, label,
    note and size, in page order."""
    md = doc.get("model") or {}
    na = doc.get("new_artists") or {}
    notes = {
        "gam": (
            f"The frozen model of the thesis ({len(md.get('terms') or [])}"
            f" terms), ranked over {int(md.get('candidates') or 0):,} "
            f"catalog candidates. Each reason is a term of the model that "
            f"passes the stability rules D1 and S4. Its strength compares "
            f"it with the same reason of every candidate."
            + _new_note(na.get("gam"), md.get("candidates") or 0, "catalog candidates")
        ),
        "itemknn_same": (
            "The co-listening baseline (item-based kNN) on the "
            "same-artist neighbour lists: a track scores the sum "
            "of cosine × log(1 + plays) over your tracks in its "
            "list."
        ),
        "itemknn_both": (
            "The co-listening baseline on both neighbour lists, "
            "same artist and other artists."
        ),
        "fresh": (
            "A declared ranking policy, not a model result: tracks "
            "that listeners of your tracks also play, by artists you "
            "do not know (other-artist co-listening, at most "
            f"{ARTIST_CAP} tracks per artist and {GENRE_CAP} per "
            "genre). It is descriptive, and P14 does not evaluate it."
        ),
        "outside": "",
        "popularity": (
            "No matched history: no listen matched the catalog "
            "and no seed artist was given, so this is the "
            "popularity list of the training data (2005 to "
            "2013)."
        ),
    }
    o = doc.get("outside") or {}
    c = o.get("counts") or {}
    if c:
        n_new = len(o.get("new_artists") or [])
        notes["outside"] = (
            f"Spotify tracks that the training data never saw, by your "
            f"{len(o.get('artists') or [])} largest artists"
            + (
                f" and by {n_new} artists you do not know (the new artists "
                f"that the cold-track GAM ranks highest in the catalog)"
                if n_new
                else ""
            )
            + f": {int(c.get('kept') or 0):,} kept of "
            f"{int(c.get('tracks') or 0):,} found. Left out: "
            f"{int(c.get('in_mlhd') or 0):,} that MLHD has, "
            f"{int(c.get('known_song_mlhd') or 0):,} other versions of songs "
            f"that MLHD has, {int(c.get('known_song_listens') or 0):,} songs "
            f"you played, and duplicates. The cold-track GAM ranks them "
            f"without co-listening. These reasons have no stability rule, "
            f"because the cold-track GAM has no replicates."
            + _new_note(na.get("outside"), c.get("kept") or 0, "kept tracks")
        )
    out = []
    for lid, path, label in LISTS:
        rows = _get(doc, path)
        if rows is None:
            continue
        out.append(
            {
                "id": lid,
                "path": list(path),
                "label": label,
                "note": notes[lid],
                "stored": len(rows),
            }
        )
    return out


def _new_note(x, n, pool):
    """The sentence of a list note on its best new-artist track."""
    if not x:
        return ""
    if not x.get("stored"):
        return " No track of this list is by an artist you do not know."
    return (
        f" It also stores its best {int(x['stored']):,} tracks by artists "
        f'you do not know (tick "Only artists you do not know"): the '
        f"first sits at rank {int(x['first_rank']):,} of {int(n):,} "
        f"{pool}."
    )


def check_v2(doc, known_ids):
    """The part 2 checks on the document (module docstring). Raises
    ValueError."""
    for lid, path, _ in LISTS:
        rows = _get(doc, path) or []
        keys = [r["key"] for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"the {lid} list holds a track twice")
        ranks = [int(r["rank"]) for r in rows]
        if ranks and (ranks[0] < 1 or any(b <= a for a, b in zip(ranks, ranks[1:]))):
            raise ValueError(f"the {lid} list is not in rank order")
        if lid in ("gam", "outside") and any(
            b["score"] > a["score"] for a, b in zip(rows, rows[1:])
        ):
            raise ValueError(f"the {lid} list: a score rises with the rank")
        for r in rows:
            for x in r.get("reasons") or []:
                p = x.get("percentile")
                if p is not None and not 0.0 <= float(p) <= 100.0:
                    raise ValueError(
                        f"the {lid} list: a percentile outside " f"[0, 100]"
                    )
    for r in _get(doc, ("gam",)) or []:
        ids = {int(k[1:]) for k in r["artist_keys"] if k.startswith("a")}
        if r["known"] != (bool(ids & known_ids) if ids else None):
            raise ValueError(
                f"the gam list: the known flag of item "
                f"{r['item']} differs from its artists"
            )
    fresh = _get(doc, ("fresh", "tracks")) or []
    bad = [
        r
        for r in fresh
        if r.get("known")
        or any(k.startswith("a") and int(k[1:]) in known_ids for k in r["artist_keys"])
    ]
    if bad:
        raise ValueError(f"the fresh list holds {len(bad)} tracks by known " f"artists")
    if len(apply_filters(fresh, ARTIST_CAP, GENRE_CAP)) != len(fresh):
        raise ValueError("the fresh list breaks its caps")


# Output


def data_js(doc):
    """demo_data.js: one line, the document as JSON in a script."""
    return (
        JS_PREFIX
        + json.dumps(
            doc,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + ";\n"
    )


def read_data_js(path):
    """The document of a demo_data.js file. Raises ValueError for another
    form."""
    text = Path(path).read_text(encoding="utf-8")
    if not (
        text.startswith(JS_PREFIX) and text.endswith(";\n") and text.count("\n") == 1
    ):
        raise ValueError(f"{path} is not one line of the form " f"{JS_PREFIX}<JSON>;")
    return json.loads(text[len(JS_PREFIX) : -2])


def write_users(root):
    """users.js: the folders of root that hold a demo_data.js."""
    root = Path(root)
    names = sorted(p.parent.name for p in root.glob(f"*/{DATA_FILE}"))
    atomic_text_write(
        USERS_PREFIX + json.dumps(names, ensure_ascii=True) + ";\n", root / USERS_FILE
    )
    return names


def copy_page(root, say=print):
    """The static page next to the user folders, byte for byte. A copy
    there that differs (an older page or an edit) is kept as
    index.old.html first."""
    if not PAGE_SRC.is_file():
        raise ValueError(
            f"{PAGE_SRC} is missing: the static page of the demo "
            f"belongs in the repository"
        )
    want = PAGE_SRC.read_bytes()
    dest = Path(root) / PAGE_FILE
    if dest.is_file() and dest.read_bytes() != want:
        old = dest.with_name(OLD_PAGE_FILE)
        dest.replace(old)
        say(
            f"{WORK_NAME}: {dest} differed from {PAGE_SRC}, so it is now "
            f"{old.name}. Edit the page in the repository"
        )
    if not dest.is_file():
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(want)
        tmp.replace(dest)
    if dest.read_bytes() != want:
        raise ValueError(f"{dest} differs from {PAGE_SRC}")
    return dest


def write_outputs(doc, out_dir, say):
    """lb_demo.json, demo_data.js, the page and users.js, read back
    (module docstring, part 2). Returns the document as read back."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _clean(doc)
    jpath, dpath = out_dir / "lb_demo.json", out_dir / DATA_FILE
    atomic_json_write(clean, jpath)
    atomic_text_write(data_js(clean), dpath)
    back = load_json(jpath)
    if back != p14._norm(clean) or read_data_js(dpath) != back:
        raise ValueError(
            "lb_demo.json and demo_data.js do not read back to " "the same document"
        )
    page = copy_page(out_dir.parent, say)
    users = write_users(out_dir.parent)
    say(
        f"{WORK_NAME}: wrote {jpath.name} and {dpath.name} in {out_dir} "
        f"({dpath.stat().st_size / 1e6:.1f} MB), the page {page} and "
        f"{USERS_FILE} ({len(users)} demo folders)"
    )
    return back


# Run


def gam_rows(G, R, A, top, ranks, S, terms, ctx, state, src, keys, say):
    """The rows of the model list: part 1's reasons on the stored slots top, in
    rank order. ranks: their places in the full ranking."""
    credit_ids = np.asarray(src.credit_ids["artists"], dtype=np.int64)
    kk = keys(top)
    rows = []
    prog = Progress(f"{WORK_NAME}: model list, evidence", len(top), say, unit=" tracks")
    for r, i in enumerate(top):
        count, ids, vals, credit, credit_n = p14._evidence(ctx, state, i)
        reasons = []
        for g, (gname, js) in enumerate(R.groups):
            if not A["shown"][g, 0, r]:
                continue
            v = float(A["reason_value"][g, 0, r])
            ev_ = evidence_units((count[g], ids[g], vals[g]), gname, src, list(js))
            rr = {"reason": gname, "value": v, "odds": _exp(v), "evidence": ev_}
            if gname == "artist" and credit >= 0:
                rr["credit"] = int(credit_ids[credit])
                rr["credit_tracks"] = int(credit_n)
            reasons.append(rr)
        reasons.sort(key=lambda x: -x["value"])
        rows.append(
            dict(
                {
                    "rank": int(ranks[r]),
                    "item": int(i),
                    "key": f"i{int(i)}",
                    "score": float(S[0][i]),
                    "cold_score": float(S[1][i]),
                    "reasons": reasons,
                    "candidates": [
                        R.groups[g][0]
                        for g in range(len(R.groups))
                        if A["candidate"][g, 0, r]
                    ],
                    "details": (
                        f"Model score {float(S[0][i]):.3f} (the sum of the "
                        f"term values, log-odds). Cold-track GAM score "
                        f"{float(S[1][i]):.3f}."
                    ),
                },
                **kk[r],
            )
        )
        prog.advance(1)
    prog.finish()
    return rows


def stored_slots(s, cand, top, top_new):
    """(slots, ranks): the union of top and top_new in rank order, and their
    places in the full ranking of the candidates by s (descending, ties by
    slot, P5's rule, from 1). Raises ValueError unless top holds ranks 1 to
    len(top)."""
    top = np.asarray(top, dtype=np.int64)
    u = np.union1d(top, np.asarray(top_new, dtype=np.int64))
    u = u[np.lexsort((u, -s[u]))]
    c = np.flatnonzero(cand)
    order = np.lexsort((c, -s[c]))
    pos = np.empty(len(c), dtype=np.int64)
    pos[order] = np.arange(1, len(c) + 1)
    k = np.minimum(np.searchsorted(c, u), max(len(c) - 1, 0))
    if len(u) and not np.array_equal(c[k], u):
        raise ValueError("a stored slot is no candidate")
    ranks = pos[k] if len(u) else np.zeros(0, dtype=np.int64)
    if not (
        np.array_equal(u[: len(top)], top)
        and np.array_equal(ranks[: len(top)], np.arange(1, len(top) + 1))
    ):
        raise ValueError("the stored top differs from the full ranking")
    return u, ranks


def check_new_rows(rows, top_new, label):
    """The stored rows by new artists (not known, with a credited artist)
    are exactly the best new-artist tracks top_new."""
    got = sorted(int(r["item"]) for r in rows if r["known"] is False)
    if got != sorted(int(x) for x in top_new):
        raise ValueError(
            f"{label}: the stored tracks by new artists differ "
            f"from the best new-artist tracks"
        )


def new_artists_by(s, new_cand, space, src, n):
    """The first n distinct credited artists of the new-artist candidates in
    the ranking by s (descending, ties by slot), the artists of one track by
    id (module docstring, part 3, 1)."""
    ids = np.asarray(src.credit_ids["artists"], dtype=np.int64)
    m = space.credit_mats["artists"]
    total = int(new_cand.sum())
    k = min(total, max(20 * n, 1000))
    while True:
        out, seen = [], set()
        for i in top_slots(s, new_cand, k):
            for c in np.sort(m.indices[m.indptr[i] : m.indptr[i + 1]]):
                a = int(ids[c])
                if a not in seen:
                    seen.add(a)
                    out.append(a)
                    if len(out) == n:
                        return out
        if k >= total:
            return out
        k = min(total, 4 * k)


def _set_percentiles(rows, pct, pool):
    """Each shown reason's percentile from {group: (rows,) percentiles}."""
    for r, row in enumerate(rows):
        for x in row["reasons"]:
            if x["reason"] in pct:
                x["percentile"] = float(pct[x["reason"]][r])
                x["pool"] = int(pool)


def run(
    st,
    io,
    model_dir,
    out_dir,
    cache_dir,
    raw_listens=None,
    source=None,
    seeds=(),
    outside=True,
    say=print,
):
    """The demo (module docstring). raw_listens: listen objects of the API
    or an export, or None. Returns the JSON document as read back."""
    t0 = time.time()
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": VERSION,
        "declared": DECLARED,
        "declared_v2": DECLARED_V2,
        "declared_v21": DECLARED_V21,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": dict(source or {}),
        "checks": {},
        "seconds": {},
    }
    listens = unique_listens(raw_listens or [])
    if raw_listens and not doc["source"].get("user"):
        names = Counter(
            str(o.get("user_name"))
            for o in raw_listens
            if isinstance(o, dict) and o.get("user_name")
        )
        if names:
            doc["source"]["user"] = names.most_common(1)[0][0]
    match, item_of = None, np.zeros(0, dtype=np.int64)
    if raw_listens is not None:
        mbids = sorted({m for rec in listens for _, m in rec["mbids"]})
        say(
            f"{WORK_NAME}: {len(listens):,} listens, {len(mbids):,} distinct "
            f"recording MBIDs. Mapping them through the canonical chain "
            f"with DuckDB (no progress lines in this step)"
        )
        t1 = time.time()
        resolved, mapped = io.resolve(mbids)
        doc["checks"]["map_mbids_equal"] = check_map(resolved, mapped)
        item_of, match = match_listens(listens, resolved)
        doc["match"] = match
        doc["seconds"]["mapping"] = time.time() - t1
        say(
            f"{WORK_NAME}: {match['matched']:,} of {match['listens']:,} "
            f"listens matched ({_pct(match['share_of_listens'])}, "
            f"{_pct(match['share_of_listens_with_mbid'])} of the listens "
            f"with an MBID), {match['items']:,} items"
        )
    items, plays, weights = history_of(item_of)
    base, G, space, src, n_slots, id_min = load_model_side(st, model_dir, say)
    terms = G["terms"]
    seed_info = None
    if not len(items) and seeds:
        pop = io.popularity(n_slots, id_min)
        items, weights, seed_info = seed_history(io, list(seeds), src, pop, say)
        plays = np.ones(len(items), dtype=np.int64)
        doc["seeds"] = seed_info
    doc["history"] = {
        "items": int(len(items)),
        "plays": int(plays.sum()),
        "kind": (
            "seed artists"
            if seed_info is not None
            else "matched listens" if len(items) else "no history"
        ),
    }
    credit_ids = np.asarray(src.credit_ids["artists"], dtype=np.int64)
    state = cn.build_user_state(items, weights, space) if len(items) else None
    if state is not None:
        art_cols = np.asarray(state.credit_cols["artists"], np.int64)
        art_share = np.asarray(state.credit_share["artists"], np.float64)
    else:
        art_cols, art_share = np.zeros(0, np.int64), np.zeros(0)
    hist_ids = credit_ids[art_cols]
    t1 = time.time()
    known = known_artists(io, listens, hist_ids.tolist())
    known_track = known_tracks(space, src, known["ids"])
    doc["known_artists"] = known["counts"]
    keys = CatalogKeys(
        space, src, None if state is None else state.term_share["genres"], known_track
    )
    say(
        f"{WORK_NAME}: {len(known['ids']):,} known artists (history "
        f"{known['counts']['history']:,}, raw listens by artist MBID "
        f"{known['counts']['by_mbid']:,}, by name "
        f"{known['counts']['by_name']:,}), "
        f"{int(known_track.sum()):,} catalog tracks by them, in "
        f"{_clock(time.time() - t1)}"
    )
    if state is None:
        pop = io.popularity(n_slots, id_min)
        doc["popularity"] = popularity_rows(pop, keys, STORE_N)
        say(f"{WORK_NAME}: no history, so the page shows the popularity " f"list")
    else:
        say(
            f"{WORK_NAME}: history of {len(items):,} items, "
            f"{int(plays.sum()):,} plays. Scoring the catalog with the final "
            f"model and the cold-track GAM"
        )
        t1 = time.time()
        T = term_sum_tables(G)
        S, conc = score_catalog(terms, T, space, state)
        doc["checks"]["dense_slots_equal"] = dense_check(terms, T, space, state, S)
        cand = np.ones(n_slots, dtype=bool)
        cand[0] = False
        cand[items] = False
        n_cand = int(cand.sum())
        top = top_slots(S[0], cand, STORE_N)
        new_cand = cand & space.has_credit["artists"] & ~known_track
        top_new = top_slots(S[0], new_cand, STORE_N)
        top, g_ranks = stored_slots(S[0], cand, top, top_new)
        doc["new_artists"] = {
            "gam": {
                "candidates": int(new_cand.sum()),
                "stored": int(len(top_new)),
                "first_rank": (
                    int(g_ranks[list(top).index(int(top_new[0]))])
                    if len(top_new)
                    else None
                ),
            }
        }
        bsum = basis_sum(terms, conc, space.tier, cand)
        doc["checks"]["background_gap"] = check_backgrounds(
            terms, T, conc, space.tier, cand, bsum
        )
        slot, w = top_encoding(terms, conc, space.tier, top)
        R = p14.TopReasons(
            terms, G["M"]["kept"], slot, w, bsum[None, :], np.array([float(n_cand)])
        )
        mean_score = float(S[0][cand].mean())
        doc["checks"]["reasons"] = check_top_reasons(R, G, S[0][top], mean_score)
        A, need = p14.reason_arrays(R, G, terms)
        check_direction(R, np.asarray(G["M"]["tables"]), A["shown"], A["reason_value"])
        groups = [
            (n, [[t.name for t in terms].index(x) for x in g])
            for n, g in pc.reason_groups(G["M"]["kept"])
        ]
        ctx = {"space": space, "groups": groups, "sources": [t.source for t in terms]}
        doc["gam"] = gam_rows(
            G, R, A, top, g_ranks, S, terms, ctx, state, src, keys, say
        )
        check_new_rows(doc["gam"], top_new, "model list")
        doc["seconds"]["ranking"] = time.time() - t1
        say(
            f"{WORK_NAME}: {len(top)} tracks ranked and explained over "
            f"{n_cand:,} candidates (the top {STORE_N} and the best "
            f"{len(top_new)} of {int(new_cand.sum()):,} by new artists, the "
            f"first at rank {doc['new_artists']['gam']['first_rank']}), the "
            f"dense check passed, in {_clock(time.time() - t1)}"
        )
        t1 = time.time()
        lists, gap, sc = knn_lists(space, items, weights, STORE_N)
        doc["checks"]["itemknn_gap"] = gap
        doc["itemknn"] = {}
        for name, kinds in (("same_artist", ("same_artist",)), ("both", knn.KINDS)):
            s = knn.candidate_scores(sc[name], items)
            pool = np.sort(s[np.isfinite(s) & (s > 0.0)])
            doc["itemknn"][name] = colisten_rows(
                space,
                state,
                lists[name],
                kinds,
                pool,
                keys,
                f"ItemKNN {name.replace('_', '-')}",
                say,
            )
        doc["fresh"] = fresh_part(
            space, state, sc["other_artists"], known_track, keys, say
        )
        doc["seconds"]["itemknn_fresh"] = time.time() - t1
        doc["model"] = {
            "final": G["rec"]["fit_name"],
            "terms": [t.name for t in terms],
            "cold_terms": list(G["cold"]["kept"]),
            "replicates": int(len(G["boot"]["listener"])),
            "members": int(len(G["members"])),
            "s4_need": int(need),
            "candidates": n_cand,
            "mean_score": mean_score,
        }
        o = np.lexsort((credit_ids[art_cols], -art_share))[:15]
        doc["history"]["top_artists"] = [
            {"artist": int(credit_ids[art_cols[k]]), "share": float(art_share[k])}
            for k in o
        ]
        ot = np.lexsort((items, -plays))[:15]
        doc["history"]["top_tracks"] = [
            {"item": int(items[k]), "plays": int(plays[k])} for k in ot
        ]
        gvals = {}
        if outside:
            t1 = time.time()
            new_ids = new_artists_by(S[1], new_cand, space, src, NEW_ARTISTS)
            doc["outside"], gvals = outside_part(
                io,
                state,
                space,
                src,
                G,
                bsum,
                float(n_cand),
                items,
                cache_dir,
                say,
                listens=listens,
                known=known,
                new_ids=new_ids,
            )
            doc["new_artists"]["outside"] = dict(
                doc["outside"].pop("new_summary", {}), artists=len(new_ids)
            )
            doc["seconds"]["outside"] = time.time() - t1
        t1 = time.time()
        say(
            f"{WORK_NAME}: strengths: each reason among the {n_cand:,} "
            f"candidates (no progress lines in this step)"
        )
        gam_groups = pc.reason_groups(G["M"]["kept"])
        shown_groups = {x["reason"] for r in doc["gam"] for x in r["reasons"]}
        pct = reason_percentiles(
            terms,
            T[0],
            conc,
            space.tier,
            cand,
            [g for g in gam_groups if g[0] in shown_groups],
            slots=top,
        )
        _set_percentiles(doc["gam"], pct, n_cand)
        if gvals:
            pct_o = reason_percentiles(
                terms,
                T[1],
                conc,
                space.tier,
                cand,
                pc.reason_groups(G["cold"]["kept"]),
                values=gvals,
            )
            _set_percentiles(doc["outside"]["tracks"], pct_o, n_cand)
        doc["seconds"]["strengths"] = time.time() - t1
        say(f"{WORK_NAME}: strengths done in {_clock(time.time() - t1)}")
        del S, conc
    t1 = time.time()
    names = _attach_names(doc, io, extra_artists=hist_ids.tolist())
    if match:
        doc["match_artists"] = artist_summary(listens, hist_ids, art_share, names)
    finish_rows(doc)
    doc["lists"] = list_meta(doc)
    doc["page"] = {
        "show": TOP_N,
        "stored": STORE_N,
        "artist_cap": ARTIST_CAP,
        "genre_cap": GENRE_CAP,
        "units": UNITS_SHOWN,
        "strong": STRONG_PCT,
        "moderate": MODERATE_PCT,
    }
    check_v2(doc, known["ids"])
    doc["seconds"]["names_words"] = time.time() - t1
    doc["seconds"]["total"] = time.time() - t0
    back = write_outputs(doc, out_dir, say)
    say(f"{WORK_NAME}: done in {_clock(time.time() - t0)}")
    return back


def _catalog_rows(doc):
    """Every list of catalog tracks in the document."""
    for key in ("gam", "popularity"):
        yield doc.get(key) or []
    for v in (doc.get("itemknn") or {}).values():
        yield v
    yield (doc.get("fresh") or {}).get("tracks") or []


def _needed_items(doc):
    """Every catalog item that the page names."""
    out = set()
    for rows in _catalog_rows(doc):
        for r in rows:
            out.add(int(r["item"]))
            for x in r.get("reasons") or []:
                if x["reason"] in ("artist", "colisten"):
                    out |= {
                        int(u["id"]) for u in (x.get("evidence") or {}).get("units", [])
                    }
    out |= {int(r["item"]) for r in (doc.get("history") or {}).get("top_tracks", [])}
    for t in (doc.get("outside") or {}).get("tracks", []):
        for x in t["reasons"]:
            out |= {int(h) for h in x.get("history_tracks", [])}
    return out


def _attach_names(doc, io, extra_artists=()):
    """Track and artist names for every list on the page. Returns {artist
    id: name} of the artists it looked up."""
    tn = io.track_names(_needed_items(doc))
    artists = {int(a) for a in extra_artists}
    for rows in _catalog_rows(doc):
        for r in rows:
            for x in r.get("reasons") or []:
                if x.get("credit") is not None:
                    artists.add(int(x["credit"]))
    artists |= {a["artist"] for a in (doc.get("history") or {}).get("top_artists", [])}
    for t in (doc.get("outside") or {}).get("tracks", []):
        for x in t["reasons"]:
            if x.get("credit") is not None:
                artists.add(x["credit"])
    an = io.artist_names(artists)

    def label(i):
        a, t, rel = tn.get(int(i), (None, None, None))
        return {"artist": a, "title": t, "release": rel}

    for rows in _catalog_rows(doc):
        for r in rows:
            r.update(label(r["item"]))
            for x in r.get("reasons") or []:
                if x.get("credit") is not None:
                    x["credit_name"] = an.get(x["credit"])
                if x["reason"] in ("artist", "colisten"):
                    for u in (x.get("evidence") or {}).get("units", []):
                        u.update(label(u["id"]))
    for r in (doc.get("history") or {}).get("top_tracks", []):
        r.update(label(r["item"]))
    for a in (doc.get("history") or {}).get("top_artists", []):
        a["name"] = an.get(a["artist"])
    for t in (doc.get("outside") or {}).get("tracks", []):
        for x in t["reasons"]:
            if x.get("credit") is not None:
                x["credit_name"] = an.get(x["credit"])
            if "history_tracks" in x:
                x["history_labels"] = [label(h) for h in x["history_tracks"]]
    return an


def _pct(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _clean(x):
    return p14._clean(x)


# Part 1 reason text


def _e(x):
    return html.escape("" if x is None else str(x))


def _odds(x):
    if x is None or not math.isfinite(x):
        return "very large"
    return f"×{x:,.0f}" if x >= 100 else f"×{x:,.2f}" if x < 10 else f"×{x:,.1f}"


def _reason_html(x):
    """part 1's reason chip, kept for pipeline/figures.py, whose explanation
    table quotes its evidence text. The page of part 2 shows x["words"]."""
    label = REASON_LABELS.get(x["reason"], x["reason"])
    ev_ = x.get("evidence") or {}
    units = ev_.get("units") or []
    more = max(0, int(ev_.get("count") or 0) - len(units))
    if x["reason"] == "artist":
        who = x.get("credit_name") or "this artist"
        n = int(x.get("credit_tracks") or 0)
        parts = [f"You play {n} track{'' if n == 1 else 's'} by {_e(who)}"]
        co = [u.get("title") or f"item {u['id']}" for u in units]
        if co:
            parts.append(
                "Listeners of "
                + ", ".join(f"“{_e(c)}”" for c in co[:3])
                + " also play it"
            )
        hist = x.get("history_labels")
        if hist and not co:
            parts.append(
                "For example "
                + ", ".join(f"“{_e(h.get('title') or '')}”" for h in hist[:2])
            )
        body = ". ".join(parts) + "."
    elif x["reason"] == "audio_fit":
        body = (
            "Close to your usual " + ", ".join(_e(u["name"]) for u in units[:4]) + "."
            if units
            else ""
        )
    else:
        body = ", ".join(_e(u["name"]) for u in units)
        if more:
            body += f" and {more} more"
    return (
        f'<div class="chip"><b>{_e(label)}</b>'
        f'<span class="odds">{_e(_odds(x.get("odds")))}</span>'
        + (f": {body}" if body else "")
        + "</div>"
    )


# Main


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "listener"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="The ListenBrainz demo " "(module docstring)"
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--user", help="a ListenBrainz user name")
    grp.add_argument("--export", help="a ListenBrainz JSON export file")
    ap.add_argument(
        "--seed-artist",
        action="append",
        default=[],
        help="an artist name or MBID (repeat for more)",
    )
    ap.add_argument("--refresh", action="store_true", help="fetch the listens again")
    ap.add_argument(
        "--page-size",
        type=int,
        default=PAGE_SIZE,
        help=f"listens per API page at the start "
        f"({MIN_PAGE} to {API_MAX}, default {PAGE_SIZE})",
    )
    ap.add_argument(
        "--no-outside",
        action="store_true",
        help="leave out the list of tracks that MLHD lacks",
    )
    args = ap.parse_args(argv)
    if not (args.user or args.export or args.seed_artist):
        ap.error("give --user, --export or --seed-artist")
    if not MIN_PAGE <= args.page_size <= API_MAX:
        ap.error(f"--page-size must lie between {MIN_PAGE} and {API_MAX}")
    name = _slug(args.user or (Path(args.export).stem if args.export else "seeds"))
    fmt = logging.Formatter(LOG_FORMAT)
    fmt.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    cache = SCRATCH_DIR / WORK_NAME / name
    try:
        with log_file(MODEL_DIR / "logs" / "lb_demo.log"):
            raw, source = None, {}
            if args.user:
                raw = fetch_listens(
                    args.user,
                    cache / "pages",
                    log.info,
                    refresh=args.refresh,
                    page_size=args.page_size,
                )
                source = {"kind": "api", "user": args.user}
            elif args.export:
                raw = list(read_export(args.export))
                source = {"kind": "export", "export": Path(args.export).name}
            if args.seed_artist:
                source["seeds"] = list(args.seed_artist)
            run(
                s13b2.StageP13b2(),
                Inputs(),
                MODEL_DIR,
                MODEL_DIR / "demo" / name,
                cache,
                raw_listens=raw,
                source=source,
                seeds=args.seed_artist,
                outside=not args.no_outside,
                say=log.info,
            )
            log.info(
                f"{WORK_NAME}: open {MODEL_DIR / 'demo' / PAGE_FILE} "
                f"in a browser (add ?user={name} when the folder holds "
                f"more than one listener)"
            )
    except KeyboardInterrupt:
        log.warning(
            "lb_demo: interrupt. The saved pages and queries stay, "
            "and the same command continues."
        )
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

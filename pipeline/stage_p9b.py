"""Stage P9b: exact co-listening neighbours for the two co-listening
concepts, colisten_same_artist and colisten_other_artists, and their
"because you played X" evidence.

P9 stays as it was. P9b replaces its numbers in the concept layer.
A coverage diagnosis measured P9 on the real data: its
co_count was a median 21 percent of the true number of listeners who
played both tracks, its score ordered edges with a Spearman correlation
of 0.36 against the exact cosine, and 12.6 percent of the popular test
picks had any P9 neighbour.

v1 stored one list per track with exact counts. Its report and a recall
diagnosis found two problems. Mid and rare tracks got
almost no neighbours. And for 91 percent of tracks the closest
co-listened track is by the same artist, so one co-listening concept
mostly repeated artist_affinity, and the model could move credit between
the two from run to run.

v2 (this version) stores two lists per track.
pipeline/colisten.py holds the method and its reasons. In short: the
same-artist lists are exact for every track (one sparse product per
artist), and the other-artist lists come from two samples of 100 tracks
per listener with the same-artist pairs left out, recounted exactly on
all train listeners. pipeline/artists.py holds the artist rule, which
P12's artist_affinity uses too.

Outputs:
- colisten_same_artist.npz: CSR arrays over item slots. Row i lists
  track i's same-artist neighbours: indptr, neighbor, cosine, co (train
  listeners who played both), and params (a JSON string).
- colisten_other_artists.npz: the same for other-artist neighbours, plus
  support_most and support_low (listeners who hold both tracks in that
  sample, 0 when that sample did not propose the pair).
- reports/p9b_report.md: coverage of the eval picks for both concepts,
  validation against the exact whole-catalog lists, and named examples.

The three long steps save their results in SCRATCH_DIR, keyed by the
parameters, the user-major SHA-256, the SHA-256 of the artist credit
arrays and the SHA-256 of colisten.py. A stopped run resumes after the
last finished step. A change in colisten.py or in the credits computes
the steps again. The steps take nothing from artists.py except the
credit arrays, so a change in artists.py that leaves the credits as
they were resumes. The marker still holds the artists.py SHA-256, so
the runner runs the stage again after any change there.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import artists as ar
from . import colisten as cl
from . import evaluation as ev
from .als import user_major_fingerprint
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    connect,
    features_table,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from .config import MODEL_DIR, SCRATCH_DIR

SAMPLE_K = 100  # tracks per listener in each sample
MIN_SUPPORT = 5  # co-listeners a pair needs, in a sample and in train
CAND_PER_ITEM = 200  # other-artist candidates each track keeps per sample
TOP_N = 50  # neighbours each track keeps, per list
VERIFY_EDGES = 2000  # per list, recounted independently
VERIFY_EXACT_TRACKS = 20  # same-artist lists recomputed in full
VERIFY_EXACT_MAX_POP = 5000  # listener cap for those tracks
VALIDATION_ITEMS = 100  # per validation set
SEED = 20260923
VERSION = "colisten-v2"
PICKS = ("test_last", "test_first", "val_last", "val_first")
BANDS = ("01-10", "11-50", "51-90", "91-100")
KINDS = ("same", "other")
LABELS = {"same": "same-artist", "other": "other-artist"}
SPECIAL_TABLE_MAX = 50  # special-purpose artists the report lists


def _atomic_savez(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez keeps .npz
    np.savez(tmp, **arrays)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _csr():
    mm = {"mmap_mode": "r"}
    return (
        np.load(MODEL_DIR / "csr_user_indptr.npy"),
        np.load(MODEL_DIR / "csr_user_indices.npy", **mm),
        np.load(MODEL_DIR / "csr_user_data.npy", **mm),
        np.load(MODEL_DIR / "csr_item_indptr.npy"),
        np.load(MODEL_DIR / "csr_item_indices.npy", **mm),
    )


def _params():
    return {
        "version": VERSION,
        "sample_k": SAMPLE_K,
        "min_support": MIN_SUPPORT,
        "cand_per_item": CAND_PER_ITEM,
        "top_n": TOP_N,
    }


def _band_codes(pop_centile):
    c = np.asarray(pop_centile)
    return np.where(c <= 10, 0, np.where(c <= 50, 1, np.where(c <= 90, 2, 3)))


def _md(text):
    """text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _special_table(special):
    """Report lines: each special-purpose artist, the part of the rule
    that matched it, its credit rows and its most printed names."""
    lines = [
        "## Special-purpose artists",
        "",
        "pipeline/artists.py matched these artist ids by name or by "
        "MBID. A credit prints the name on the release. So the "
        "printed names of one id can differ from the name that the "
        "rule matched.",
        "",
        "| artist id | matched by | credit rows | printed names | most "
        "printed (rows) |",
        "|---:|---|---:|---:|---|",
    ]
    for e in special[:SPECIAL_TABLE_MAX]:
        lines.append(
            f"| {e['artist_id']} | {_md(', '.join(e['matched_by']))} | "
            f"{e['rows']:,} | {e['distinct_names']:,} | "
            + ", ".join(f"{_md(n)} ({c:,})" for n, c in e["names"])
            + " |"
        )
    if not special:
        lines.append("| none | | | | |")
    if len(special) > SPECIAL_TABLE_MAX:
        lines += ["", f"{len(special) - SPECIAL_TABLE_MAX:,} more are not " f"shown."]
    return lines + [""]


def _pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def _path_sha(path):
    """SHA-256 of a file, or of a directory's files in name order."""
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(f.relative_to(path)).encode("utf-8"))
            digest.update(file_sha256(f).encode("utf-8"))
        return digest.hexdigest()
    return file_sha256(path)


def _arrays_sha(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def _rows(graph):
    return np.repeat(
        np.arange(len(graph["indptr"]) - 1, dtype=np.int64), np.diff(graph["indptr"])
    )


class StageP9b:
    name = "p9b_colisten"
    upstream = ["p25_train", "p6_als"]

    def outputs(self):
        return {
            "same": MODEL_DIR / "colisten_same_artist.npz",
            "other": MODEL_DIR / "colisten_other_artists.npz",
            "report": MODEL_DIR / "reports" / "p9b_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        inputs["colisten_module"] = file_sha256(Path(cl.__file__))
        inputs["artists_module"] = file_sha256(Path(ar.__file__))
        try:
            inputs["item_artists"] = _path_sha(features_table("item_artists"))
        except FileNotFoundError:
            inputs["item_artists"] = None
        return inputs

    # The five loaders below read DuckDB tables. test_p9b.py replaces them.
    def _credits(self, n_items):
        return ar.load_credits(n_items)

    def _key(self, u_arrays, art):
        """The data, parameters and code that the cached steps depend on.
        P6's csr state holds the user-major SHA-256. Without it, compute
        it. The colisten.py hash stops a resume into results that older
        code computed. artists.py reaches the steps only through the
        credit arrays, so their hash stands for it."""
        fp = (load_json(SCRATCH_DIR / "p6_csr_state.json") or {}).get(
            "user_major_sha256"
        )
        if not fp:
            fp = user_major_fingerprint(u_arrays, log=log.info)
        return dict(
            _params(),
            user_major_sha256=fp,
            colisten_sha256=file_sha256(Path(cl.__file__)),
            credits_sha256=_arrays_sha(*art),
        )

    def run(self, params, files):
        outputs = self.outputs()
        u_indptr, u_indices, u_data, i_indptr, i_indices = _csr()
        check(
            int(u_indptr[-1]) == int(i_indptr[-1]),
            "the user-major and item-major CSR disagree on nnz",
        )
        n_users, n_items = len(u_indptr) - 1, len(i_indptr) - 1
        pop = np.diff(i_indptr).astype(np.int64)
        eligible = pop >= MIN_SUPPORT
        eligible[0] = False
        art_indptr, art_ids, credit_stats = self._credits(n_items)
        art = (art_indptr, art_ids)
        log.info(
            "p9b: %d credit rows, %d special-purpose artists left out, "
            "%d tracks with a credit",
            credit_stats["credit_rows"],
            credit_stats["special_artists"],
            credit_stats["tracks_with_credit"],
        )
        same_path = SCRATCH_DIR / "p9b_same.npz"
        cand_path = SCRATCH_DIR / "p9b_candidates.npz"
        count_path = SCRATCH_DIR / "p9b_counts.npz"
        state_path = SCRATCH_DIR / "p9b_state.json"
        key = self._key((u_indptr, u_indices, u_data), art)
        state = load_json(state_path) or {}
        if state.get("key") != key:
            state = {"key": key}
        state["credit_stats"] = credit_stats
        atomic_json_write(state, state_path)

        def cached(name, path):
            return (
                state.get(name) and path.exists() and file_sha256(path) == state[name]
            )

        if cached("same_sha", same_path):
            log.info("p9b: same-artist lists resume")
            z = np.load(same_path)
            same = {k: z[k] for k in ("indptr", "neighbor", "cosine", "co")}
        else:
            same = cl.same_artist_lists(
                art_indptr,
                art_ids,
                i_indptr,
                i_indices,
                pop,
                MIN_SUPPORT,
                TOP_N,
                log=log.info,
            )
            _atomic_savez(same_path, **same)
            state["same_sha"] = file_sha256(same_path)
            atomic_json_write(state, state_path)

        if cached("candidates_sha", cand_path):
            log.info("p9b: other-artist candidates resume")
            z = np.load(cand_path)
            a, b = z["a"], z["b"]
            sm, sl = z["support_most"], z["support_low"]
        else:
            s_indptr, s_most, s_low = cl.sample_users(
                u_indptr,
                u_indices,
                u_data,
                pop,
                SAMPLE_K,
                log=log.info,
                eligible=eligible,
            )
            most = cl.candidate_pairs(
                s_indptr,
                s_most,
                n_items,
                MIN_SUPPORT,
                CAND_PER_ITEM,
                log=log.info,
                art=art,
                label="p9b: other-artist candidates from the most-played " "sample",
            )
            low = cl.candidate_pairs(
                s_indptr,
                s_low,
                n_items,
                MIN_SUPPORT,
                CAND_PER_ITEM,
                log=log.info,
                art=art,
                label="p9b: other-artist candidates from the least-popular " "sample",
            )
            del s_indptr, s_most, s_low
            a, b, sm, sl = cl.union_candidates(most, low, n_items)
            state["candidate_counts"] = {
                "most": int(len(most[0])),
                "low": int(len(low[0])),
                "union": int(len(a)),
            }
            del most, low
            _atomic_savez(cand_path, a=a, b=b, support_most=sm, support_low=sl)
            state["candidates_sha"] = file_sha256(cand_path)
            state.pop("counts_sha", None)
            atomic_json_write(state, state_path)
        log.info("p9b: %d other-artist candidate pairs", len(a))

        if cached("counts_sha", count_path):
            log.info("p9b: exact counts resume")
            co = np.load(count_path)["co"]
        else:
            co = cl.exact_co_listeners(a, b, i_indptr, i_indices, n_users, log=log.info)
            _atomic_savez(count_path, co=co)
            state["counts_sha"] = file_sha256(count_path)
            atomic_json_write(state, state_path)

        other = cl.directed_topn(
            a, b, co, sm, sl, pop, n_items, TOP_N, MIN_SUPPORT, log=log.info
        )
        stamp = np.array(json.dumps(_params()))
        _atomic_savez(outputs["same"], params=stamp, **same)
        _atomic_savez(outputs["other"], params=stamp, **other)
        result = {}
        for kind, graph in (("same", same), ("other", other)):
            log.info(
                "p9b: %s lists: %d tracks hold %d neighbours",
                LABELS[kind],
                int(np.count_nonzero(np.diff(graph["indptr"]))),
                len(graph["neighbor"]),
            )
            result[kind] = {
                "path": str(outputs[kind]),
                "rows": int(len(graph["neighbor"])),
                "sha256": file_sha256(outputs[kind]),
            }
        return result

    def _load(self, kind):
        z = np.load(self.outputs()[kind])
        cols = ["indptr", "neighbor", "cosine", "co"]
        if kind == "other":
            cols += ["support_most", "support_low"]
        return {k: z[k] for k in cols}, json.loads(str(z["params"]))

    def verify(self, params):
        u_indptr, u_indices, u_data, i_indptr, i_indices = _csr()
        pop = np.diff(i_indptr).astype(np.int64)
        n_items = len(pop)
        art = self._credits(n_items)[:2]
        graphs = {}
        for kind in KINDS:
            graph, stored = self._load(kind)
            graphs[kind] = graph
            label = LABELS[kind]
            check(
                stored == _params(),
                f"the {label} file was built with {stored}, the stage now "
                f"says {_params()}",
            )
            problems = cl.knn_problems(graph, pop, TOP_N, MIN_SUPPORT)
            check(not problems, f"{label} lists: " + " ".join(problems))
            nnz = len(graph["neighbor"])
            check(nnz > 0, f"the {label} lists hold no neighbours")
            rows = _rows(graph)
            shared = cl.share_artist(rows, graph["neighbor"], *art)
            if kind == "same":
                check(
                    bool(shared.all()),
                    f"{int((~shared).sum())} " f"same-artist pairs share no artist",
                )
            else:
                check(
                    not shared.any(),
                    f"{int(shared.sum())} other-artist " f"pairs share an artist",
                )
            rng = np.random.default_rng(SEED)
            k = np.sort(rng.choice(nnz, size=min(VERIFY_EDGES, nnz), replace=False))
            bad = 0
            for r, n, c in zip(
                rows[k], graph["neighbor"][k].astype(np.int64), graph["co"][k]
            ):
                ur = np.asarray(i_indices[i_indptr[r] : i_indptr[r + 1]])
                un = np.asarray(i_indices[i_indptr[n] : i_indptr[n + 1]])
                bad += int(np.intersect1d(ur, un, assume_unique=True).size != c)
            check(
                bad == 0,
                f"{bad} of {len(k)} sampled {label} pairs have a "
                f"co count that differs from an independent "
                f"recount",
            )
            log.info(
                "p9b: %d sampled %s pairs match an independent recount", len(k), label
            )

        # The same-artist lists claim to be exact: recompute some in full,
        # half of them from tracks that hold a list.
        same = graphs["same"]
        deg = np.diff(same["indptr"])
        ok_pop = (pop >= MIN_SUPPORT) & (pop <= VERIFY_EXACT_MAX_POP)
        ok_pop[0] = False
        rng = np.random.default_rng(SEED + 1)
        half = VERIFY_EXACT_TRACKS // 2
        tracks = []
        for pool in (np.flatnonzero(ok_pop & (deg > 0)), np.flatnonzero(ok_pop)):
            if len(pool):
                tracks += rng.choice(
                    pool, size=min(half, len(pool)), replace=False
                ).tolist()
        tracks = np.unique(tracks)
        exact = cl.exact_topn(
            tracks,
            u_indptr,
            u_indices,
            i_indptr,
            i_indices,
            TOP_N,
            MIN_SUPPORT,
            art=art,
        )
        wrong = []
        for x in tracks:
            s, e = same["indptr"][x], same["indptr"][x + 1]
            nb, co, _ = exact[int(x)]["same"]
            if not (
                np.array_equal(same["neighbor"][s:e], nb)
                and np.array_equal(same["co"][s:e], co)
            ):
                wrong.append(int(x))
        check(
            not wrong,
            f"{len(wrong)} of {len(tracks)} same-artist lists "
            f"differ from a full recomputation: {wrong[:5]}",
        )
        log.info("p9b: %d same-artist lists equal a full recomputation", len(tracks))

    def _eval_picks(self):
        con = connect(4)
        try:
            cols = con.execute(f"""
                WITH picks AS (
                    SELECT user_id,
                           max(CASE WHEN is_test_last THEN item_id END) AS test_last,
                           max(CASE WHEN is_test_first THEN item_id END) AS test_first,
                           max(CASE WHEN is_val_last THEN item_id END) AS val_last,
                           max(CASE WHEN is_val_first THEN item_id END) AS val_first
                    FROM read_parquet('{ev.SPLITS}')
                    GROUP BY user_id)
                SELECT e.user_id,
                       coalesce(p.test_last, 0) AS test_last,
                       coalesce(p.test_first, 0) AS test_first,
                       coalesce(p.val_last, 0) AS val_last,
                       coalesce(p.val_first, 0) AS val_first
                FROM read_parquet('{ev.EVAL_USERS}') e
                LEFT JOIN picks p ON e.user_id = p.user_id
                ORDER BY e.user_id
            """).fetchnumpy()
        finally:
            con.close()
        return (
            np.asarray(cols["user_id"], dtype=np.int64),
            np.stack([np.asarray(cols[n], dtype=np.int64) for n in PICKS], axis=1),
        )

    def _pop_centile(self, n_slots):
        con = connect(4)
        try:
            return ev.covariate_arrays(con, n_slots)[0]
        finally:
            con.close()

    def _names(self, ids):
        """{item_id: "artist - title"}, best effort."""
        if len(ids) == 0:
            return {}
        con = connect(4)
        try:
            id_list = ", ".join(str(int(i)) for i in ids)
            rows = con.execute(f"""
                SELECT i.item_id,
                       arg_max(c.artist_name, (c.recording_mbid
                               = i.canonical_recording_mbid)::INT),
                       arg_max(c.recording_name, (c.recording_mbid
                               = i.canonical_recording_mbid)::INT)
                FROM read_parquet('{features_table("item_ids")}') i
                JOIN read_parquet('{features_table("item_catalog")}') c
                  ON c.canonical_recording_mbid = i.canonical_recording_mbid
                WHERE i.item_id IN ({id_list})
                GROUP BY i.item_id
            """).fetchall()
        except Exception as exc:  # names are optional
            log.warning("p9b: no names for the report examples: %s", exc)
            return {}
        finally:
            con.close()
        return {int(r[0]): f"{r[1]} - {r[2]}" for r in rows}

    def _validation_sets(self, picks, bands, pop):
        rng = np.random.default_rng(SEED)
        test_picks = picks[:, 0][picks[:, 0] > 0]
        sets = [
            (
                "random last_ts test picks",
                rng.choice(
                    test_picks,
                    size=min(VALIDATION_ITEMS, len(test_picks)),
                    replace=False,
                ),
            )
        ]
        for code in (3, 2, 1):
            pool = np.flatnonzero((bands == code) & (pop >= MIN_SUPPORT))
            if len(pool):
                sets.append(
                    (
                        f"random tracks in pop band {BANDS[code]}",
                        rng.choice(
                            pool, size=min(VALIDATION_ITEMS, len(pool)), replace=False
                        ),
                    )
                )
        return sets, rng

    def report(self, params):
        outputs = self.outputs()
        graphs = {kind: self._load(kind)[0] for kind in KINDS}
        u_indptr, u_indices, u_data, i_indptr, i_indices = _csr()
        pop = np.diff(i_indptr).astype(np.int64)
        n_items = len(pop)
        art_indptr, art_ids, stats = self._credits(n_items)
        art = (art_indptr, art_ids)
        state = load_json(SCRATCH_DIR / "p9b_state.json") or {}
        counts = state.get("candidate_counts", {})
        deg = {kind: np.diff(g["indptr"]) for kind, g in graphs.items()}
        q = [0, 10, 50, 90, 100]
        lines = [
            "# P9b report: exact co-listening neighbours, same artist "
            "and other artists",
            "",
            f"- parameters: {json.dumps(_params())}",
            f"- artist credits: {stats['credit_rows']:,} rows. "
            f"Special-purpose artists, which do not count as an "
            f"artist: {stats['special_artists']:,}, with "
            f"{stats['special_rows']:,} credit rows. The section "
            f'"Special-purpose artists" names each one and the '
            f"part of the rule that matched it. Tracks that keep a "
            f"credit: {stats['tracks_with_credit']:,}.",
            f"- tracks with {MIN_SUPPORT} or more train listeners: "
            f"{int((pop[1:] >= MIN_SUPPORT).sum()):,} of {n_items - 1:,}",
        ]
        if counts:
            lines.append(
                f"- other-artist candidate pairs: most-played sample "
                f"{counts['most']:,}, least-popular sample "
                f"{counts['low']:,}, union {counts['union']:,}"
            )
        for kind in KINDS:
            g, d = graphs[kind], deg[kind]
            lines.append(
                f"- {LABELS[kind]} lists: {int(np.count_nonzero(d)):,} tracks "
                f"hold {len(g['neighbor']):,} neighbours. Per track with "
                f"any: median {np.median(d[d > 0]) if d.any() else 0:.0f}. "
                f"Train listeners who played both, per stored pair: "
                + ", ".join(
                    f"p{p} {v:,.0f}"
                    for p, v in zip(
                        q, np.percentile(g["co"], q) if len(g["co"]) else [0] * 5
                    )
                )
            )
        lines.append("")

        users, picks = self._eval_picks()
        bands = _band_codes(self._pop_centile(n_items))
        a_hit = cl.artist_in_history(users, picks, u_indptr, u_indices, *art)
        hits = {
            kind: cl.pick_hits(
                g["indptr"], g["neighbor"], u_indptr, u_indices, users, picks
            )
            for kind, g in graphs.items()
        }
        lines += [
            "## Coverage of the eval users' held-out picks",
            "",
            "Artist: the pick shares an artist with a track in the "
            "user's train history (artist_affinity above zero). "
            '"Neighbour": the pick has at least one of that kind. '
            '"In history": at least one of them is a track the user '
            "played in train, so the concept is above zero and an "
            "explanation can name the track.",
            "",
            "| pick | band | picks | artist | same-artist neighbour | "
            "same-artist in history | other-artist neighbour | "
            "other-artist in history |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for j, name in enumerate(PICKS):
            p = picks[:, j]
            ok = p > 0
            for code, label in [(None, "all")] + list(enumerate(BANDS)):
                m = ok if code is None else ok & (bands[p] == code)
                n = int(m.sum())
                if n == 0:
                    continue
                qq = p[m]
                cells = [_pct(int(a_hit[m, j].sum()), n)]
                for kind in KINDS:
                    cells += [
                        _pct(int((deg[kind][qq] > 0).sum()), n),
                        _pct(int((hits[kind][m, j] > 0).sum()), n),
                    ]
                lines.append(
                    f"| {name} | {label} | {n:,} | " + " | ".join(cells) + " |"
                )

        sets, rng = self._validation_sets(picks, bands, pop)
        all_items = np.unique(np.concatenate([s for _, s in sets]))
        exact = cl.exact_topn(
            all_items,
            u_indptr,
            u_indices,
            i_indptr,
            i_indices,
            TOP_N,
            MIN_SUPPORT,
            log=log.info,
            art=art,
        )
        ks = sorted({5, 10, TOP_N})
        lines += [
            "",
            f"## Validation against the exact whole-catalog top " f"{TOP_N}",
            "",
            "The exact lists count every listener of the track and "
            "every other track, with no sampling, and split them by "
            "artist. Recall at k: the share of the exact top k that "
            "the stored list holds. The same-artist lists are built "
            "exactly, so their recall must be 1.000.",
            "",
            "| set | kind | tracks with an exact list | exact, median | "
            "stored, median | "
            + " | ".join(f"recall@{k}" for k in ks)
            + " | identical lists |",
            "|---|---|---|---|---|" + "---|" * len(ks) + "---|",
        ]
        for label, items in sets:
            for kind in KINDS:
                g = graphs[kind]
                rec = {k: [] for k in ks}
                same_n = n_ex = 0
                ex_len, st_len = [], []
                for x in items:
                    ex_nb = exact[int(x)][kind][0]
                    st_nb = g["neighbor"][g["indptr"][x] : g["indptr"][x + 1]].astype(
                        np.int64
                    )
                    ex_len.append(len(ex_nb))
                    st_len.append(len(st_nb))
                    if len(ex_nb) == 0:
                        continue
                    n_ex += 1
                    held = np.isin(ex_nb, st_nb)
                    for k in ks:
                        rec[k].append(float(held[:k].mean()))
                    same_n += int(np.array_equal(ex_nb, st_nb))
                lines.append(
                    f"| {label} | {LABELS[kind]} | {n_ex} | "
                    f"{np.median(ex_len):.0f} | {np.median(st_len):.0f} | "
                    + " | ".join(
                        f"{np.mean(rec[k]):.3f}" if rec[k] else "n/a" for k in ks
                    )
                    + f" | {_pct(same_n, n_ex)} |"
                )

        lines += ["", "## Examples", ""]
        picks_ex = []
        both = (deg["same"] >= min(3, TOP_N)) & (deg["other"] >= min(3, TOP_N))
        for code in (3, 2):
            pool = np.flatnonzero((bands == code) & both)
            if len(pool):
                picks_ex += rng.choice(
                    pool, size=min(2, len(pool)), replace=False
                ).tolist()
        shown = set(picks_ex)
        for x in picks_ex:
            for g in graphs.values():
                shown.update(
                    g["neighbor"][g["indptr"][x] : g["indptr"][x] + 5].tolist()
                )
        names = self._names(sorted(shown))
        for x in picks_ex:
            lines += [
                f"### {names.get(x, f'item {x}')} (item {x}, "
                f"{int(pop[x]):,} train listeners)",
                "",
            ]
            for kind in KINDS:
                g = graphs[kind]
                lines += [
                    f"{LABELS[kind]} neighbours:",
                    "",
                    "| neighbour | its listeners | played both | " "cosine |",
                    "|---|---|---|---|",
                ]
                s = g["indptr"][x]
                for k in range(s, min(s + 5, g["indptr"][x + 1])):
                    y = int(g["neighbor"][k])
                    lines.append(
                        f"| {names.get(y, f'item {y}')} | "
                        f"{int(pop[y]):,} | {int(g['co'][k]):,} | "
                        f"{g['cosine'][k]:.4f} |"
                    )
                lines.append("")
        lines += _special_table(stats.get("special", []))
        atomic_text_write("\n".join(lines) + "\n", outputs["report"])
        log.info("p9b report written")

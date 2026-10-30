"""Stage P4: the feature matrices.

Outputs:
- id_spaces.json: the measured id convention, the contract for all matrices
- item_audio_std.parquet: ten z-scored columns, one-hot enums, NULL tempo mask
- item_audio_std_maponly.parquet: the ablation copy, same statistics
- audio_std_stats.json: means and sds, unweighted and train-listen-weighted
- vocab_tags.parquet + item_tags_direct.npz + item_tags_artist.npz:
  the vocabulary after the alias merge, the filter file, the DF floor,
  and the debris rule
- vocab_genres.parquet + item_genres.npz
- item_work_flags.parquet: attribute-token flags, relation path removed
- SPOTIFY_DERIVED.txt: the no-copy list for the license boundary

Merge semantics: the alias map folds variant spellings onto canonicals
before the DISTINCT steps, so one item tagged with two spellings of one
tag counts once. The merge covers the DF-floor vocabulary; sub-floor
variants stay out. Matrix rows use item_id directly. Slots below the
id minimum stay phantom. Train statistics only.
"""

import hashlib
import re
from pathlib import Path

import numpy as np

from .common import (
    atomic_json_write,
    atomic_parquet_write,
    atomic_text_write,
    check,
    connect,
    content_hash,
    features_table,
    file_sha256,
    load_json,
    log,
    marker_path,
    with_thread_ladder,
)
from .config import CURATED, FROZEN, MODEL_DIR

CONT_COLS = [
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


def _npz_hash(path: Path) -> str:
    data = np.load(path)
    digest = hashlib.sha256()
    for name in sorted(data.files):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    return digest.hexdigest()


def _read_tag_filter(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# One tag per line. A line that starts with '#' is "
            "a comment.\nspecial purpose artist\n",
            encoding="utf-8",
        )
    tags = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tags.append(line)
    return tags


def _load_filter_table(con, path: Path):
    tags = _read_tag_filter(path)
    con.execute("CREATE OR REPLACE TABLE tag_filter (tag_name VARCHAR)")
    if tags:
        con.executemany("INSERT INTO tag_filter VALUES (?)", [(t,) for t in tags])
    return tags


def _read_tag_aliases(path: Path):
    if not path.exists():
        return []
    aliases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("=", 1)
        if len(parts) == 2:
            variant = parts[0].strip()
            canonical = parts[1].strip()
            if variant and canonical and variant != canonical:
                aliases.append((variant, canonical))
    return aliases


def _load_alias_table(con, path: Path):
    aliases = _read_tag_aliases(path)
    con.execute(
        "CREATE OR REPLACE TABLE tag_alias " "(variant VARCHAR, canonical VARCHAR)"
    )
    if aliases:
        con.executemany("INSERT INTO tag_alias VALUES (?, ?)", aliases)
    return aliases


def _aliased_pairs_sql(table: str) -> str:
    """Distinct (item, tag) pairs with variants folded onto canonicals."""
    return f"""
(SELECT DISTINCT item_id, COALESCE(al.canonical, q.tag_name) AS tag_name
 FROM (SELECT DISTINCT item_id, tag_name FROM read_parquet('{table}')) q
 LEFT JOIN tag_alias al ON q.tag_name = al.variant)
"""


class StageP4:
    name = "p4_features"
    upstream = ["p3_sampling"]

    def outputs(self):
        return {
            "id_spaces": MODEL_DIR / "id_spaces.json",
            "vocab_tags": MODEL_DIR / "vocab_tags.parquet",
            "tags_direct": MODEL_DIR / "item_tags_direct.npz",
            "tags_artist": MODEL_DIR / "item_tags_artist.npz",
            "vocab_genres": MODEL_DIR / "vocab_genres.parquet",
            "genres_csr": MODEL_DIR / "item_genres.npz",
            "audio_std": MODEL_DIR / "item_audio_std.parquet",
            "audio_std_maponly": MODEL_DIR / "item_audio_std_maponly.parquet",
            "audio_stats": MODEL_DIR / "audio_std_stats.json",
            "work_flags": MODEL_DIR / "item_work_flags.parquet",
            "spotify_list": MODEL_DIR / "SPOTIFY_DERIVED.txt",
            "report": MODEL_DIR / "reports" / "p4_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        try:
            inputs["marker:p3_sampling"] = file_sha256(marker_path("p3_sampling"))
        except FileNotFoundError:
            inputs["marker:p3_sampling"] = None
        fpath = CURATED / "tag_filter.txt"
        inputs["tag_filter"] = file_sha256(fpath) if fpath.exists() else None
        apath = CURATED / "tag_aliases.txt"
        inputs["tag_aliases"] = file_sha256(apath) if apath.exists() else None
        return inputs

    def run(self, params, files):
        outputs = self.outputs()
        filter_path = CURATED / "tag_filter.txt"
        alias_path = CURATED / "tag_aliases.txt"

        def work(con):
            id_doc = self._id_spaces(con)
            atomic_json_write(id_doc, outputs["id_spaces"])
            i_slots = id_doc["items"]["slots"]
            i_min = id_doc["items"]["min"]
            filter_tags = _load_filter_table(con, filter_path)
            aliases = _load_alias_table(con, alias_path)
            raw_tags, kept_tags = self._write_vocab_tags(
                con, outputs["vocab_tags"], int(params["tag_min_df"]), filter_tags
            )
            log.info(
                "p4: %d alias lines active, %d tags before the " "filters, %d kept",
                len(aliases),
                raw_tags,
                kept_tags,
            )
            self._write_tag_csr(
                con,
                features_table("item_tags"),
                str(outputs["vocab_tags"]),
                outputs["tags_direct"],
                i_slots,
                i_min,
            )
            self._write_tag_csr(
                con,
                features_table("item_tags_artist"),
                str(outputs["vocab_tags"]),
                outputs["tags_artist"],
                i_slots,
                i_min,
            )
            self._write_vocab_genres(con, outputs["vocab_genres"])
            self._write_genre_csr(con, outputs, i_slots, i_min)
            self._write_audio(con, outputs)
            self._write_work_flags(con, outputs["work_flags"])
            self._write_license_list(outputs["spotify_list"])

        with_thread_ladder(work, params["ladder"])

        info = {}
        con = connect(params["ladder"][0])
        try:
            for key, order in [
                ("vocab_tags", "tag_name"),
                ("vocab_genres", "genre"),
                ("audio_std", "item_id"),
                ("audio_std_maponly", "item_id"),
                ("work_flags", "item_id"),
            ]:
                path = outputs[key]
                rows = con.execute(
                    f"SELECT count(*) FROM read_parquet('{path}')"
                ).fetchone()[0]
                digest = content_hash(con, f"read_parquet('{path}')", order)
                info[key] = {"path": str(path), "rows": int(rows), "sha256": digest}
            for key in ("tags_direct", "tags_artist", "genres_csr"):
                info[key] = {
                    "path": str(outputs[key]),
                    "rows": 0,
                    "sha256": _npz_hash(outputs[key]),
                }
            for key in ("audio_stats", "id_spaces", "spotify_list"):
                info[key] = {
                    "path": str(outputs[key]),
                    "rows": 0,
                    "sha256": file_sha256(outputs[key]),
                }
        finally:
            con.close()
        return info

    def _id_spaces(self, con):
        ids_table = features_table("item_ids")
        users_table = features_table("user_ids")
        i_lo, i_hi, i_d = con.execute(
            f"SELECT min(item_id), max(item_id), count(DISTINCT item_id) "
            f"FROM read_parquet('{ids_table}')"
        ).fetchone()
        check(
            i_d == FROZEN["items"], f"item_ids distinct {i_d}, frozen {FROZEN['items']}"
        )
        check(
            i_hi - i_lo + 1 == i_d,
            f"item_id has gaps: min {i_lo}, max {i_hi}, distinct {i_d}",
        )
        u_lo, u_hi, u_d = con.execute(
            f"SELECT min(user_id), max(user_id), count(DISTINCT user_id) "
            f"FROM read_parquet('{users_table}')"
        ).fetchone()
        check(
            u_d == FROZEN["users"], f"user_ids distinct {u_d}, frozen {FROZEN['users']}"
        )
        check(
            u_hi - u_lo + 1 == u_d,
            f"user_id has gaps: min {u_lo}, max {u_hi}, distinct {u_d}",
        )
        doc = {
            "items": {
                "min": int(i_lo),
                "max": int(i_hi),
                "distinct": int(i_d),
                "slots": int(i_hi) + 1,
            },
            "users": {
                "min": int(u_lo),
                "max": int(u_hi),
                "distinct": int(u_d),
                "slots": int(u_hi) + 1,
            },
            "rule": "Every matrix uses item_id or user_id directly as the "
            "row or column index. Slots below the id minimum are "
            "phantom and must stay empty. Assert this in every "
            "consumer.",
        }
        log.info(
            "p4: item_id %d to %d, user_id %d to %d, phantom slots " "stay empty",
            i_lo,
            i_hi,
            u_lo,
            u_hi,
        )
        return doc

    def _write_vocab_tags(self, con, dest, min_df, filter_tags):
        direct = features_table("item_tags")
        artist = features_table("item_tags_artist")
        sql = f"""
WITH d AS {_aliased_pairs_sql(direct)},
     a AS {_aliased_pairs_sql(artist)},
     u AS (SELECT tag_name, item_id FROM d
           UNION SELECT tag_name, item_id FROM a),
     dfu AS (SELECT tag_name, count(*) AS df_union FROM u GROUP BY tag_name),
     dfd AS (SELECT tag_name, count(*) AS df_direct FROM d GROUP BY tag_name),
     dfa AS (SELECT tag_name, count(*) AS df_artist FROM a GROUP BY tag_name)
SELECT dfu.tag_name, dfu.df_union,
       COALESCE(dfd.df_direct, 0) AS df_direct,
       COALESCE(dfa.df_artist, 0) AS df_artist
FROM dfu
LEFT JOIN dfd ON dfu.tag_name = dfd.tag_name
LEFT JOIN dfa ON dfu.tag_name = dfa.tag_name
ORDER BY dfu.tag_name
"""
        rows = con.execute(sql).fetchall()
        filt = set(filter_tags)
        kept = [
            r
            for r in rows
            if r[0] not in filt and r[1] >= min_df and _is_tag_name(r[0])
        ]
        kept.sort(key=lambda r: r[0])
        con.execute(
            "CREATE OR REPLACE TABLE vocab_build ("
            "tag_name VARCHAR, tag_idx BIGINT, df_union BIGINT, "
            "df_direct BIGINT, df_artist BIGINT)"
        )
        con.executemany(
            "INSERT INTO vocab_build VALUES (?, ?, ?, ?, ?)",
            [
                (name, idx, dfu, dfd, dfa)
                for idx, (name, dfu, dfd, dfa) in enumerate(kept)
            ],
        )
        atomic_parquet_write(
            con, "SELECT * FROM vocab_build ORDER BY tag_name", dest, expected_mb=32
        )
        log.info(
            "p4: tag vocabulary written, %d tags after the merge, "
            "the filter, the floor, and the debris rule",
            len(kept),
        )
        return len(rows), len(kept)

    def _write_tag_csr(self, con, table, vocab, dest, n_slots, id_min):
        sql = f"""
SELECT p.item_id, v.tag_idx AS col_idx
FROM {_aliased_pairs_sql(table)} p
JOIN read_parquet('{vocab}') v ON p.tag_name = v.tag_name
ORDER BY p.item_id, v.tag_idx
"""
        self._csr_from_sql(con, sql, vocab, dest, n_slots, id_min)

    def _write_vocab_genres(self, con, dest):
        genres = features_table("item_spotify_genres")
        sql = f"""
WITH g AS (SELECT DISTINCT item_id, genre FROM read_parquet('{genres}'))
SELECT genre, count(*) AS df,
       row_number() OVER (ORDER BY genre) - 1 AS genre_idx
FROM g GROUP BY genre ORDER BY genre
"""
        atomic_parquet_write(con, sql, dest, expected_mb=8)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[0]
        log.info("p4: genre vocabulary written, %d genres", n)

    def _write_genre_csr(self, con, outputs, n_slots, id_min):
        genres = features_table("item_spotify_genres")
        vocab = str(outputs["vocab_genres"])
        sql = f"""
SELECT p.item_id, v.genre_idx AS col_idx
FROM (SELECT DISTINCT item_id, genre FROM read_parquet('{genres}')) p
JOIN read_parquet('{vocab}') v ON p.genre = v.genre
ORDER BY p.item_id, v.genre_idx
"""
        self._csr_from_sql(con, sql, vocab, outputs["genres_csr"], n_slots, id_min)

    def _csr_from_sql(self, con, sql, vocab, dest, n_slots, id_min):
        cols = con.execute(sql).fetchnumpy()
        rows = np.asarray(cols["item_id"], dtype=np.int64)
        indices = np.asarray(cols["col_idx"], dtype=np.int32)
        if len(rows):
            check(int(rows.min()) >= id_min, "a row index falls below the id minimum")
            check(int(rows.max()) < n_slots, "a row index exceeds the id maximum")
        counts = np.bincount(rows, minlength=n_slots)
        check(int(counts[:id_min].sum()) == 0, "phantom rows are not empty")
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        n_cols = con.execute(
            f"SELECT count(*) FROM read_parquet('{vocab}')"
        ).fetchone()[0]
        np.savez_compressed(
            dest,
            indices=indices,
            indptr=indptr,
            shape=np.array([n_slots, n_cols], dtype=np.int64),
        )
        log.info(
            "p4: csr %s written, %d nonzeros, %d rows, %d columns",
            dest.name,
            len(indices),
            n_slots,
            n_cols,
        )

    def _write_audio(self, con, outputs):
        ext = features_table("item_audio_features_ext")
        mapt = features_table("item_audio_features")
        stats_train = str(MODEL_DIR / "stats_item_train.parquet")

        cols = con.execute(self._stats_sql(ext)).fetchnumpy()
        means = {c: float(cols[f"{c}_mean"][0]) for c in CONT_COLS}
        sds = {c: float(cols[f"{c}_sd"][0]) for c in CONT_COLS}
        for col in CONT_COLS:
            check(sds[col] > 0, f"zero standard deviation for {col}")

        wmean_cols = con.execute(self._weighted_mean_sql(ext, stats_train)).fetchnumpy()
        wmeans = {c: float(wmean_cols[f"{c}_mean"][0]) for c in CONT_COLS}
        wsd_cols = con.execute(
            self._weighted_sd_sql(ext, stats_train, wmeans)
        ).fetchnumpy()
        wsds = {c: float(wsd_cols[f"{c}_sd"][0]) for c in CONT_COLS}

        tempo_zero = con.execute(
            f"SELECT count(*) FROM read_parquet('{ext}') " f"WHERE tempo = 0"
        ).fetchone()[0]
        doc = {
            "n_rows": FROZEN["audio_ext_items"],
            "tempo_zero_masked": int(tempo_zero),
            "note": "Unweighted statistics are the default. Weighted "
            "statistics use train listens only. The map-only "
            "ablation copy reuses the unweighted ext statistics.",
            "columns": {
                c: {
                    "mean": means[c],
                    "sd": sds[c],
                    "weighted_mean": wmeans[c],
                    "weighted_sd": wsds[c],
                }
                for c in CONT_COLS
            },
        }
        atomic_json_write(doc, outputs["audio_stats"])
        atomic_parquet_write(
            con, self._std_sql(ext, means, sds), outputs["audio_std"], expected_mb=512
        )
        atomic_parquet_write(
            con,
            self._std_sql(mapt, means, sds),
            outputs["audio_std_maponly"],
            expected_mb=192,
        )
        log.info("p4: audio matrices written, %d tempo rows masked", tempo_zero)

    def _stats_sql(self, table):
        parts = []
        for col in CONT_COLS:
            expr = f"CAST({col} AS DOUBLE)"
            if col == "tempo":
                parts.append(f"avg({expr}) FILTER (WHERE tempo != 0) " f"AS {col}_mean")
                parts.append(
                    f"stddev_samp({expr}) FILTER " f"(WHERE tempo != 0) AS {col}_sd"
                )
            else:
                parts.append(f"avg({expr}) AS {col}_mean")
                parts.append(f"stddev_samp({expr}) AS {col}_sd")
        select_list = ", ".join(parts)
        return f"SELECT {select_list} FROM read_parquet('{table}')"

    def _weighted_mean_sql(self, table, stats_train):
        parts = []
        for col in CONT_COLS:
            expr = f"CAST(e.{col} AS DOUBLE)"
            if col == "tempo":
                parts.append(
                    f"sum(s.listens * {expr}) FILTER "
                    f"(WHERE e.tempo != 0) / "
                    f"sum(s.listens) FILTER "
                    f"(WHERE e.tempo != 0) AS {col}_mean"
                )
            else:
                parts.append(
                    f"sum(s.listens * {expr}) / sum(s.listens) " f"AS {col}_mean"
                )
        select_list = ", ".join(parts)
        return (
            f"SELECT {select_list} FROM read_parquet('{table}') e "
            f"JOIN read_parquet('{stats_train}') s "
            f"ON e.item_id = s.item_id"
        )

    def _weighted_sd_sql(self, table, stats_train, means):
        parts = []
        for col in CONT_COLS:
            expr = f"CAST(e.{col} AS DOUBLE) - {means[col]!r}"
            if col == "tempo":
                parts.append(
                    f"sqrt(sum(s.listens * power({expr}, 2)) FILTER "
                    f"(WHERE e.tempo != 0) / "
                    f"sum(s.listens) FILTER "
                    f"(WHERE e.tempo != 0)) AS {col}_sd"
                )
            else:
                parts.append(
                    f"sqrt(sum(s.listens * power({expr}, 2)) "
                    f"/ sum(s.listens)) AS {col}_sd"
                )
        select_list = ", ".join(parts)
        return (
            f"SELECT {select_list} FROM read_parquet('{table}') e "
            f"JOIN read_parquet('{stats_train}') s "
            f"ON e.item_id = s.item_id"
        )

    def _std_sql(self, table, means, sds):
        zcols = []
        for col in CONT_COLS:
            cast = f"CAST({col} AS DOUBLE)"
            if col == "tempo":
                zcols.append(
                    f"CASE WHEN tempo = 0 THEN NULL ELSE "
                    f"({cast} - {means[col]!r}) / {sds[col]!r} END "
                    f"AS z_tempo"
                )
            else:
                zcols.append(f"({cast} - {means[col]!r}) / {sds[col]!r} " f"AS z_{col}")
        onehots = (
            [f'("key" = {k}) AS key_{k}' for k in range(12)]
            + [f'("mode" = {m}) AS mode_{m}' for m in (0, 1)]
            + [f"(time_signature = {t}) AS ts_{t}" for t in range(3, 8)]
            + ["(time_signature NOT BETWEEN 3 AND 7) AS ts_other"]
        )
        select_list = ", ".join(["item_id"] + zcols + onehots)
        return f"SELECT {select_list} FROM read_parquet('{table}') " f"ORDER BY item_id"

    def _write_work_flags(self, con, dest):
        ids_table = features_table("item_ids")
        works = features_table("item_works")
        sql = f"""
WITH w AS (SELECT item_id, attributes FROM read_parquet('{works}')),
     exploded AS (
         SELECT item_id,
                unnest(string_split(coalesce(attributes, ''), ',')) AS raw_token
         FROM w
         WHERE coalesce(attributes, '') <> ''
     ),
     toks AS (SELECT DISTINCT item_id, lower(trim(raw_token)) AS token
              FROM exploded),
     tflags AS (
         SELECT item_id,
                max(CASE WHEN token = 'live' THEN 1 ELSE 0 END) AS f_live,
                max(CASE WHEN token = 'cover' THEN 1 ELSE 0 END) AS f_cover,
                max(CASE WHEN token = 'partial' THEN 1 ELSE 0 END) AS f_partial,
                max(CASE WHEN token = 'instrumental' THEN 1 ELSE 0 END) AS f_instr,
                max(CASE WHEN token = 'medley' THEN 1 ELSE 0 END) AS f_medley
         FROM toks GROUP BY item_id
     ),
     hasw AS (SELECT DISTINCT item_id FROM w)
SELECT i.item_id,
       (hf.item_id IS NOT NULL) AS has_work,
       (COALESCE(t.f_live, 0) = 1) AS is_live,
       (COALESCE(t.f_cover, 0) = 1) AS is_cover,
       (COALESCE(t.f_partial, 0) = 1) AS is_partial,
       (COALESCE(t.f_instr, 0) = 1) AS is_instrumental,
       (COALESCE(t.f_medley, 0) = 1) AS is_medley
FROM read_parquet('{ids_table}') i
LEFT JOIN hasw hf ON i.item_id = hf.item_id
LEFT JOIN tflags t ON i.item_id = t.item_id
ORDER BY i.item_id
"""
        atomic_parquet_write(con, sql, dest, expected_mb=64)
        log.info(
            "p4: work flags written, attribute path only: the "
            "relation column holds a single value"
        )

    def _write_license_list(self, dest):
        lines = [
            "# Files that derive from the Spotify catalogue.",
            "# They must not leave this machine.",
            "# item_covariates.parquet is MIXED: the sp_popularity and tier",
            "# columns derive from the bridge. Strip those columns before",
            "# any copy leaves this machine.",
            "",
            "item_audio_std.parquet",
            "item_audio_std_maponly.parquet",
            "audio_std_stats.json",
            "item_genres.npz",
            "vocab_genres.parquet",
        ]
        atomic_text_write("\n".join(lines) + "\n", dest)
        log.info("p4: license list written")

    def verify(self, params):
        con = connect(4)
        outputs = self.outputs()
        filter_path = CURATED / "tag_filter.txt"
        alias_path = CURATED / "tag_aliases.txt"
        try:
            id_doc = self._id_spaces(con)
            stored = load_json(outputs["id_spaces"])
            check(stored == id_doc, "id_spaces.json does not match the ids tables")
            i_slots = id_doc["items"]["slots"]
            i_min = id_doc["items"]["min"]

            _load_filter_table(con, filter_path)
            _load_alias_table(con, alias_path)

            amap = features_table("artist_spotify_map")
            n_am, n_amb, n_sp = con.execute(
                f"SELECT count(*), count(DISTINCT mb_artist_id), "
                f"count(DISTINCT sp_artist) FROM read_parquet('{amap}')"
            ).fetchone()
            check(
                n_am == FROZEN["artist_map_rows"],
                f"artist_spotify_map rows {n_am}, frozen "
                f"{FROZEN['artist_map_rows']}",
            )
            check(
                n_amb == FROZEN["artist_map_artists"],
                f"artist_spotify_map artists {n_amb}, frozen "
                f"{FROZEN['artist_map_artists']}",
            )
            check(
                n_sp == FROZEN["artist_map_sp"],
                f"artist_spotify_map spotify ids {n_sp}, frozen "
                f"{FROZEN['artist_map_sp']}",
            )
            am_nulls = con.execute(
                f"SELECT count(*) FROM read_parquet('{amap}') "
                f"WHERE sp_artist IS NULL OR sp_artist_resolved IS NULL"
            ).fetchone()[0]
            check(am_nulls == 0, "artist_spotify_map carries NULL ids")

            self_maps = con.execute(
                "SELECT count(*) FROM tag_alias " "WHERE variant = canonical"
            ).fetchone()[0]
            check(self_maps == 0, "an alias line maps a tag to itself")
            chains = con.execute(
                "SELECT count(*) FROM tag_alias a JOIN tag_alias b "
                "ON a.canonical = b.variant"
            ).fetchone()[0]
            check(
                chains == 0,
                "alias chains exist: a canonical is itself a variant. "
                "The merge applies one step only.",
            )
            crossed = con.execute(
                "SELECT count(*) FROM tag_alias a JOIN tag_filter f "
                "ON a.variant = f.tag_name "
                "OR a.canonical = f.tag_name"
            ).fetchone()[0]
            check(crossed == 0, "an alias line touches a filtered tag")

            ext = features_table("item_audio_features_ext")
            mapt = features_table("item_audio_features")
            n_ext = con.execute(
                f"SELECT count(*) FROM read_parquet('{ext}')"
            ).fetchone()[0]
            check(
                n_ext == FROZEN["audio_ext_items"],
                f"ext rows {n_ext}, frozen {FROZEN['audio_ext_items']}",
            )
            n_map = con.execute(
                f"SELECT count(*) FROM read_parquet('{mapt}')"
            ).fetchone()[0]
            check(
                n_map == FROZEN["audio_map_items"],
                f"map-only rows {n_map}, frozen " f"{FROZEN['audio_map_items']}",
            )
            for table, label in [(ext, "ext"), (mapt, "map-only")]:
                kmin, kmax = con.execute(
                    f'SELECT min("key"), max("key") ' f"FROM read_parquet('{table}')"
                ).fetchone()
                check(0 <= kmin and kmax <= 11, f"{label}: key range {kmin} to {kmax}")
                bad_mode = con.execute(
                    f"SELECT count(*) FROM read_parquet('{table}') "
                    f'WHERE "mode" NOT IN (0, 1)'
                ).fetchone()[0]
                check(
                    bad_mode == 0,
                    f"{label}: {bad_mode} rows carry a mode outside 0 and 1",
                )
            tz_map = con.execute(
                f"SELECT count(*) FROM read_parquet('{mapt}') " f"WHERE tempo = 0"
            ).fetchone()[0]
            check(
                tz_map == FROZEN["tempo_zero_map"],
                f"map-only tempo zero {tz_map}, frozen " f"{FROZEN['tempo_zero_map']}",
            )
            ts_map = con.execute(
                f"SELECT count(*) FROM read_parquet('{mapt}') "
                f"WHERE time_signature NOT BETWEEN 3 AND 7"
            ).fetchone()[0]
            check(
                ts_map == FROZEN["ts_outside_map"],
                f"map-only time signatures outside 3 to 7: {ts_map}, "
                f"frozen {FROZEN['ts_outside_map']}",
            )

            std = str(outputs["audio_std"])
            n_std = con.execute(
                f"SELECT count(*) FROM read_parquet('{std}')"
            ).fetchone()[0]
            check(n_std == FROZEN["audio_ext_items"], f"audio std rows {n_std}")
            for col in CONT_COLS:
                m, s = con.execute(
                    f"SELECT avg(z_{col}), stddev_samp(z_{col}) "
                    f"FROM read_parquet('{std}')"
                ).fetchone()
                check(m is not None and abs(m) < 1e-6, f"z_{col} mean is {m}")
                check(s is not None and abs(s - 1.0) < 1e-6, f"z_{col} sd is {s}")
            tempo_zero = con.execute(
                f"SELECT count(*) FROM read_parquet('{ext}') " f"WHERE tempo = 0"
            ).fetchone()[0]
            null_tempo = con.execute(
                f"SELECT count(*) FROM read_parquet('{std}') " f"WHERE z_tempo IS NULL"
            ).fetchone()[0]
            check(
                null_tempo == tempo_zero,
                f"masked tempo rows {null_tempo}, source holds " f"{tempo_zero}",
            )
            key_chain = " + ".join(f"CAST(key_{k} AS INTEGER)" for k in range(12))
            kmin, kmax = con.execute(
                f"SELECT min({key_chain}), max({key_chain}) "
                f"FROM read_parquet('{std}')"
            ).fetchone()
            check(kmin == 1 and kmax == 1, "a row carries zero or several key one-hots")
            mode_chain = "CAST(mode_0 AS INTEGER) + CAST(mode_1 AS INTEGER)"
            mmin, mmax = con.execute(
                f"SELECT min({mode_chain}), max({mode_chain}) "
                f"FROM read_parquet('{std}')"
            ).fetchone()
            check(
                mmin == 1 and mmax == 1, "a row carries zero or several mode one-hots"
            )
            ts_chain = (
                " + ".join(f"CAST(ts_{t} AS INTEGER)" for t in range(3, 8))
                + " + CAST(ts_other AS INTEGER)"
            )
            tmin, tmax = con.execute(
                f"SELECT min({ts_chain}), max({ts_chain}) "
                f"FROM read_parquet('{std}')"
            ).fetchone()
            check(
                tmin == 1 and tmax == 1,
                "a row carries zero or several time-signature one-hots",
            )

            stdm = str(outputs["audio_std_maponly"])
            n_stdm = con.execute(
                f"SELECT count(*) FROM read_parquet('{stdm}')"
            ).fetchone()[0]
            check(n_stdm == FROZEN["audio_map_items"], f"map-only std rows {n_stdm}")
            null_tm = con.execute(
                f"SELECT count(*) FROM read_parquet('{stdm}') " f"WHERE z_tempo IS NULL"
            ).fetchone()[0]
            check(
                null_tm == FROZEN["tempo_zero_map"],
                f"map-only masked tempo rows {null_tm}",
            )
            ts_other_m = con.execute(
                f"SELECT count(*) FROM read_parquet('{stdm}') " f"WHERE ts_other"
            ).fetchone()[0]
            check(
                ts_other_m == FROZEN["ts_outside_map"],
                f"map-only ts_other rows {ts_other_m}",
            )

            vocab = str(outputs["vocab_tags"])
            min_df = con.execute(
                f"SELECT min(df_union) FROM read_parquet('{vocab}')"
            ).fetchone()[0]
            check(
                min_df is None or min_df >= int(params["tag_min_df"]),
                f"a vocabulary tag falls below the DF floor: {min_df}",
            )
            leaks = con.execute(f"""
                SELECT count(*) FROM read_parquet('{vocab}') v
                JOIN tag_filter f ON v.tag_name = f.tag_name
            """).fetchone()[0]
            check(leaks == 0, f"{leaks} filtered tags entered the vocabulary")
            names = [
                r[0]
                for r in con.execute(
                    f"SELECT tag_name FROM read_parquet('{vocab}')"
                ).fetchall()
            ]
            noletter = [n for n in names if not _is_tag_name(n)]
            check(
                not noletter,
                f"tags without letters entered the vocabulary: " f"{noletter[:5]}",
            )

            for key, table in [
                ("tags_direct", "item_tags"),
                ("tags_artist", "item_tags_artist"),
            ]:
                src = features_table(table)
                data = np.load(outputs[key])
                indices = np.asarray(data["indices"])
                indptr = np.asarray(data["indptr"])
                n_rows, n_cols = data["shape"]
                check(
                    int(n_rows) == i_slots,
                    f"{key}: CSR rows {n_rows}, expected {i_slots}",
                )
                check(int(n_cols) > 0, f"{key}: empty vocabulary")
                check(len(indptr) == i_slots + 1, f"{key}: indptr length {len(indptr)}")
                check(int(indptr[i_min]) == 0, f"{key}: phantom rows are not empty")
                check(
                    int(indptr[-1]) == len(indices),
                    f"{key}: indptr end {indptr[-1]} != nnz {len(indices)}",
                )
                check(
                    bool(np.all(np.diff(indptr) >= 0)),
                    f"{key}: indptr decreases somewhere",
                )
                if len(indices):
                    check(
                        int(indices.max()) < int(n_cols),
                        f"{key}: an index exceeds the vocabulary",
                    )
                nnz_sql = con.execute(f"""
                    SELECT count(*) FROM {_aliased_pairs_sql(src)} p
                    JOIN read_parquet('{vocab}') v
                      ON p.tag_name = v.tag_name
                """).fetchone()[0]
                check(
                    len(indices) == nnz_sql,
                    f"{key}: nnz {len(indices)} != expected {nnz_sql}",
                )
            n_dp = con.execute(
                f"SELECT count(*) FROM read_parquet("
                f"'{features_table('item_tags')}')"
            ).fetchone()[0]
            check(
                n_dp == FROZEN["tags_direct_pairs"],
                f"direct tag rows {n_dp}, frozen " f"{FROZEN['tags_direct_pairs']}",
            )
            artist_src = features_table("item_tags_artist")
            n_ap, n_ad = con.execute(
                f"SELECT count(*), count(DISTINCT item_id) "
                f"FROM read_parquet('{artist_src}')"
            ).fetchone()
            check(
                n_ap == FROZEN["tags_artist_rows"],
                f"artist tag rows {n_ap}, frozen " f"{FROZEN['tags_artist_rows']}",
            )
            check(
                n_ad == FROZEN["tags_artist_items"],
                f"artist tag items {n_ad}, frozen " f"{FROZEN['tags_artist_items']}",
            )

            n_gv = con.execute(
                f"SELECT count(*) FROM read_parquet(" f"'{outputs['vocab_genres']}')"
            ).fetchone()[0]
            check(
                n_gv == FROZEN["genres_vocab"],
                f"genre vocabulary {n_gv}, frozen {FROZEN['genres_vocab']}",
            )
            n_gr = con.execute(
                f"SELECT count(*) FROM read_parquet("
                f"'{features_table('item_spotify_genres')}')"
            ).fetchone()[0]
            check(
                n_gr == FROZEN["genres_rows"],
                f"genre rows {n_gr}, frozen {FROZEN['genres_rows']}",
            )
            gdata = np.load(outputs["genres_csr"])
            gindices = np.asarray(gdata["indices"])
            gindptr = np.asarray(gdata["indptr"])
            gn, gc = gdata["shape"]
            check(
                int(gn) == i_slots and int(gc) == FROZEN["genres_vocab"],
                f"genre CSR shape {gn} x {gc}",
            )
            check(len(gindptr) == i_slots + 1, "genre indptr length is wrong")
            check(int(gindptr[i_min]) == 0, "genre phantom rows are not empty")
            check(
                int(gindptr[-1]) == len(gindices), "genre indptr end does not match nnz"
            )
            gnnz = con.execute(f"""
                SELECT count(*) FROM
                (SELECT DISTINCT item_id, genre
                 FROM read_parquet(
                   '{features_table('item_spotify_genres')}'))
            """).fetchone()[0]
            check(
                len(gindices) == gnnz,
                f"genre nnz {len(gindices)} != distinct pairs {gnnz}",
            )

            flags = str(outputs["work_flags"])
            n_wf = con.execute(
                f"SELECT count(*), count(DISTINCT item_id) "
                f"FROM read_parquet('{flags}')"
            ).fetchone()
            check(
                n_wf[0] == FROZEN["items"] and n_wf[1] == FROZEN["items"],
                f"work flag rows {n_wf[0]}, distinct {n_wf[1]}",
            )
            n_hw = con.execute(
                f"SELECT count(*) FROM read_parquet('{flags}') " f"WHERE has_work"
            ).fetchone()[0]
            check(
                n_hw == FROZEN["works_items"],
                f"items with works {n_hw}, frozen {FROZEN['works_items']}",
            )
        finally:
            con.close()

    def report(self, params):
        con = connect(4)
        outputs = self.outputs()
        ext = features_table("item_audio_features_ext")
        std = str(outputs["audio_std"])
        vocab = str(outputs["vocab_tags"])
        genres_src = features_table("item_spotify_genres")
        works_src = features_table("item_works")
        alias_path = CURATED / "tag_aliases.txt"
        lines = ["# P4 feature report", ""]
        add = lines.append
        try:
            _load_alias_table(con, alias_path)
            id_doc = load_json(outputs["id_spaces"]) or {}
            add("Id spaces (the contract for every matrix):")
            for kind in ("items", "users"):
                entry = id_doc.get(kind, {})
                add(
                    f"- {kind}: {entry.get('min')} to {entry.get('max')}, "
                    f"{entry.get('distinct')} distinct, "
                    f"{entry.get('slots')} matrix slots. Slots below the "
                    f"minimum stay phantom."
                )
            add("")
            stats = load_json(outputs["audio_stats"]) or {}
            add("Audio matrix (item-uniform mean 0, sd 1):")
            add(f"- ext rows: {stats.get('n_rows')}")
            add(f"- tempo rows masked to NULL: " f"{stats.get('tempo_zero_masked')}")
            for col in CONT_COLS:
                entry = (stats.get("columns") or {}).get(col, {})
                add(
                    f"- {col}: mean {entry.get('mean', 0):.4f}, "
                    f"sd {entry.get('sd', 0):.4f}, "
                    f"weighted mean {entry.get('weighted_mean', 0):.4f}, "
                    f"weighted sd {entry.get('weighted_sd', 0):.4f}"
                )
            add("- source min/max per column (sanity):")
            for col in CONT_COLS:
                lo, hi = con.execute(
                    f"SELECT min(CAST({col} AS DOUBLE)), "
                    f"max(CAST({col} AS DOUBLE)) "
                    f"FROM read_parquet('{ext}')"
                ).fetchone()
                add(f"  - {col}: {lo} to {hi}")
            add("")
            n_vocab = con.execute(
                f"SELECT count(*) FROM read_parquet('{vocab}')"
            ).fetchone()[0]
            n_alias = len(_read_tag_aliases(alias_path))
            add(
                f"Tag vocabulary: {n_vocab} tags after the alias merge "
                f"({n_alias} alias lines), the filter file, the DF floor "
                f"at {params['tag_min_df']}, and the debris rule"
            )
            for label, table_name in [
                ("item_tags_direct.npz", "item_tags"),
                ("item_tags_artist.npz", "item_tags_artist"),
            ]:
                table = features_table(table_name)
                raw, merged_pairs, covered, kept = con.execute(f"""
                    WITH src AS (SELECT DISTINCT item_id, tag_name
                                 FROM read_parquet('{table}')),
                         merged AS {_aliased_pairs_sql(table)}
                    SELECT (SELECT count(*) FROM src),
                           (SELECT count(*) FROM merged),
                           (SELECT count(DISTINCT item_id) FROM src),
                           (SELECT count(DISTINCT m.item_id) FROM merged m
                            JOIN read_parquet('{vocab}') v
                              ON m.tag_name = v.tag_name)
                """).fetchone()
                add(
                    f"- {label}: raw distinct pairs {raw}, pairs after the "
                    f"merge {merged_pairs}, items {covered}, items kept "
                    f"after all filters {kept}"
                )
            add("")
            add("Top 30 tags by item coverage, after the merge:")
            for name, dfu, dfd, dfa in con.execute(
                f"SELECT tag_name, df_union, df_direct, df_artist "
                f"FROM read_parquet('{vocab}') "
                f"ORDER BY df_union DESC LIMIT 30"
            ).fetchall():
                add(f"- {name}: {dfu} items (direct {dfd}, artist {dfa})")
            add("")
            graw, gdistinct = con.execute(f"""
                SELECT count(*),
                       (SELECT count(*) FROM (SELECT DISTINCT item_id, genre
                        FROM read_parquet('{genres_src}')))
                FROM read_parquet('{genres_src}')
            """).fetchone()
            gdata = np.load(outputs["genres_csr"])
            add(
                f"Genre CSR: {FROZEN['genres_vocab']} genres, "
                f"{len(gdata['indices'])} nonzeros, raw rows {graw}, "
                f"distinct pairs {gdistinct}"
            )
            add("")
            flags = str(outputs["work_flags"])
            add(
                "Work flags (attribute tokens only; the relation column "
                "holds a single value, performance):"
            )
            for label, column in [
                ("items with a work link", "has_work"),
                ("live", "is_live"),
                ("cover", "is_cover"),
                ("partial", "is_partial"),
                ("instrumental", "is_instrumental"),
                ("medley", "is_medley"),
            ]:
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{flags}') " f"WHERE {column}"
                ).fetchone()[0]
                add(f"- {label}: {n}")
            add("")
            add("Top 15 attribute tokens:")
            for token, n in con.execute(f"""
                    WITH w AS (SELECT attributes FROM
                               read_parquet('{works_src}')),
                         e AS (SELECT unnest(string_split(
                               coalesce(attributes, ''), ',')) AS t FROM w
                               WHERE coalesce(attributes, '') <> '')
                    SELECT lower(trim(t)), count(*) FROM e
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall():
                add(f"- {token}: {n}")
        finally:
            con.close()
        atomic_text_write("\n".join(lines) + "\n", outputs["report"])
        log.info("p4 report written")


def _is_tag_name(name):
    """A tag carries a letter, or is a year or a year range.
    Barcodes, disc ids, symbols, and emoji fragments fail this test.
    Measured on the real data: the 68 letterless tags at DF >= 50 held about
    24 years, 30 catalog numbers, and 10 symbol fragments."""
    if any(ch.isalpha() for ch in name):
        return True
    return re.fullmatch(r"[0-9]{4}(-[0-9]{4})?", name) is not None

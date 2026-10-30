"""Stage P3: sampling tables for training and evaluation.

Outputs:
- neg_proposals.parquet and neg_proposals.npz: negative sampling over the core
- eval_users.parquet: the stratified, seeded eval sample
- item_covariates.parquet: tier, availability, train popularity bands

Every probability and statistic derives from train-only tables.
The leakage wall holds. sp_popularity is a 2025 snapshot, a covariate only.

Audio semantics, corrected after the first run: presence in
item_audio_features_ext means the features exist. The ext table holds only
feature-bearing rows. The AF-less items sit outside the table: 3,092 in the
map tiers and 6,116 in fuzzy_dur, from the counts of the bridge.
"""

import hashlib
from pathlib import Path

import numpy as np

from .common import (
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
    year_expr,
    with_thread_ladder,
)
from .config import FROZEN, MODEL_DIR


def _npz_hash(path: Path) -> str:
    data = np.load(path)
    digest = hashlib.sha256()
    for name in sorted(data.files):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    return digest.hexdigest()


class StageP3:
    name = "p3_sampling"
    upstream = ["p25_train"]

    def outputs(self):
        return {
            "proposals": MODEL_DIR / "neg_proposals.parquet",
            "proposals_npz": MODEL_DIR / "neg_proposals.npz",
            "eval_users": MODEL_DIR / "eval_users.parquet",
            "covariates": MODEL_DIR / "item_covariates.parquet",
            "report": MODEL_DIR / "reports" / "p3_report.md",
        }

    def current_inputs(self, files):
        try:
            return {"marker:p25_train": file_sha256(marker_path("p25_train"))}
        except FileNotFoundError:
            return {"marker:p25_train": None}

    def run(self, params, files):
        outputs = self.outputs()
        schema = load_json(MODEL_DIR / "schema_interactions.json") or {}
        ts_mode = schema.get("ts_mode")

        def work(con):
            self._write_proposals(
                con,
                outputs["proposals"],
                int(params["core_item_floor"]),
                float(params["neg_proposal_power"]),
            )
            self._write_npz(con, outputs)
            self._write_eval_users(
                con,
                outputs["eval_users"],
                ts_mode,
                int(params["eval_user_sample"]),
                int(params["seed"]),
            )
            self._write_covariates(con, outputs["covariates"])

        with_thread_ladder(work, params["ladder"])

        info = {}
        con = connect(params["ladder"][0])
        try:
            for key, order in [
                ("proposals", "item_id"),
                ("eval_users", "user_id"),
                ("covariates", "item_id"),
            ]:
                path = outputs[key]
                rows = con.execute(
                    f"SELECT count(*) FROM read_parquet('{path}')"
                ).fetchone()[0]
                digest = content_hash(con, f"read_parquet('{path}')", order)
                info[key] = {"path": str(path), "rows": int(rows), "sha256": digest}
            info["proposals_npz"] = {
                "path": str(outputs["proposals_npz"]),
                "rows": 0,
                "sha256": _npz_hash(outputs["proposals_npz"]),
            }
        finally:
            con.close()
        return info

    def _write_proposals(self, con, dest, floor, power):
        stats = str(MODEL_DIR / "stats_item_train.parquet")
        sql = f"""
WITH core AS (
    SELECT item_id, CAST(listens AS DOUBLE) AS listens
    FROM read_parquet('{stats}')
    WHERE pairs >= {floor}
),
w AS (
    SELECT item_id, power(listens, {power}) AS w FROM core
)
SELECT item_id,
       1.0 / count(*) OVER () AS p_uniform,
       w / sum(w) OVER () AS p_pop
FROM w
ORDER BY item_id
"""
        atomic_parquet_write(con, sql, dest, expected_mb=128)
        log.info("p3: negative proposals written")

    def _write_npz(self, con, outputs):
        prop = str(outputs["proposals"])
        cols = con.execute(
            f"SELECT item_id, p_uniform, p_pop FROM read_parquet('{prop}') "
            f"ORDER BY item_id"
        ).fetchnumpy()
        item_ids = np.asarray(cols["item_id"])
        p_uniform = np.asarray(cols["p_uniform"], dtype=np.float64)
        p_pop = np.asarray(cols["p_pop"], dtype=np.float64)
        np.savez_compressed(
            outputs["proposals_npz"],
            item_ids=item_ids,
            p_uniform=p_uniform,
            cum_uniform=np.cumsum(p_uniform),
            p_pop=p_pop,
            cum_pop=np.cumsum(p_pop),
        )
        log.info("p3: proposal npz written, %d items", len(item_ids))

    def _write_eval_users(self, con, dest, ts_mode, n_target, seed):
        stats = str(MODEL_DIR / "stats_user_train.parquet")
        strata_sql = f"""
SELECT user_id, pairs,
       CAST({year_expr('min_first_ts', ts_mode)} AS INTEGER) AS first_year,
       ntile(10) OVER (ORDER BY pairs, user_id) AS activity_decile
FROM read_parquet('{stats}')
ORDER BY user_id
"""
        cols = con.execute(strata_sql).fetchnumpy()
        user_ids = np.asarray(cols["user_id"], dtype=np.int64)
        pairs = np.asarray(cols["pairs"], dtype=np.int64)
        first_year = np.asarray(cols["first_year"], dtype=np.int64)
        decile = np.asarray(cols["activity_decile"], dtype=np.int64)

        rng = np.random.default_rng(seed)
        draws = rng.random(len(user_ids))

        key = decile * 10000 + first_year
        uniq, counts = np.unique(key, return_counts=True)
        exact = counts * (n_target / counts.sum())
        quota = np.floor(exact).astype(np.int64)
        short = n_target - int(quota.sum())
        frac = exact - quota
        order_by_frac = np.lexsort((uniq, -frac))
        quota[order_by_frac[:short]] += 1

        order = np.lexsort((draws, key))
        sorted_key = key[order]
        quota_map = dict(zip(uniq.tolist(), quota.tolist()))
        selected = np.zeros(len(user_ids), dtype=bool)
        starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_key)) + 1))
        ends = np.concatenate((starts[1:], [len(sorted_key)]))
        for start, end in zip(starts.tolist(), ends.tolist()):
            q = quota_map[int(sorted_key[start])]
            if q > 0:
                selected[order[start : start + q]] = True

        con.execute(
            "CREATE OR REPLACE TABLE eval_sel ("
            "user_id BIGINT, train_pairs BIGINT, first_year INTEGER, "
            "activity_decile INTEGER, draw DOUBLE)"
        )
        rows = list(
            zip(
                user_ids[selected].tolist(),
                pairs[selected].tolist(),
                first_year[selected].tolist(),
                decile[selected].tolist(),
                draws[selected].tolist(),
            )
        )
        con.executemany("INSERT INTO eval_sel VALUES (?, ?, ?, ?, ?)", rows)
        atomic_parquet_write(
            con, "SELECT * FROM eval_sel ORDER BY user_id", dest, expected_mb=8
        )
        log.info("p3: eval sample written, %d users", len(rows))

    def _write_covariates(self, con, dest):
        ids_table = features_table("item_ids")
        stats = str(MODEL_DIR / "stats_item_train.parquet")
        map_table = features_table("item_spotify_map")
        fuzzy_table = features_table("item_spotify_fuzzy")
        ext_table = features_table("item_audio_features_ext")
        tags_table = features_table("item_tags")
        tagsa_table = features_table("item_tags_artist")
        genres_table = features_table("item_spotify_genres")
        works_table = features_table("item_works")
        sql = f"""
WITH ids AS (SELECT item_id FROM read_parquet('{ids_table}')),
     core AS (SELECT item_id, pairs, listens FROM read_parquet('{stats}')),
     mapx AS (SELECT item_id, match_method FROM read_parquet('{map_table}')),
     fuzx AS (SELECT item_id, basis FROM read_parquet('{fuzzy_table}')),
     audx AS (SELECT item_id, sp_popularity
              FROM read_parquet('{ext_table}')),
     tagsd AS (SELECT DISTINCT item_id FROM read_parquet('{tags_table}')),
     tagsa AS (SELECT DISTINCT item_id FROM read_parquet('{tagsa_table}')),
     genr AS (SELECT DISTINCT item_id FROM read_parquet('{genres_table}')),
     wrks AS (SELECT DISTINCT item_id FROM read_parquet('{works_table}')),
     joined AS (
         SELECT i.item_id,
                COALESCE(c.pairs, 0) AS train_pairs,
                COALESCE(c.listens, 0) AS train_listens,
                COALESCE(m.match_method,
                         CASE WHEN f.basis = 'name+duration' THEN 'fuzzy_dur'
                              WHEN f.basis = 'name_only' THEN 'fuzzy_name'
                              ELSE 'unmatched' END) AS tier,
                (a.item_id IS NOT NULL) AS has_audio,
                (td.item_id IS NOT NULL) AS has_tags_direct,
                (ta.item_id IS NOT NULL) AS has_tags_artist,
                (g.item_id IS NOT NULL) AS has_genres,
                (w.item_id IS NOT NULL) AS has_work,
                a.sp_popularity
         FROM ids i
         LEFT JOIN core c ON i.item_id = c.item_id
         LEFT JOIN mapx m ON i.item_id = m.item_id
         LEFT JOIN fuzx f ON i.item_id = f.item_id
         LEFT JOIN audx a ON i.item_id = a.item_id
         LEFT JOIN tagsd td ON i.item_id = td.item_id
         LEFT JOIN tagsa ta ON i.item_id = ta.item_id
         LEFT JOIN genr g ON i.item_id = g.item_id
         LEFT JOIN wrks w ON i.item_id = w.item_id
     )
SELECT item_id, train_pairs, train_listens, tier, has_audio,
       has_tags_direct, has_tags_artist, has_genres, has_work,
       sp_popularity,
       ntile(100) OVER (ORDER BY train_pairs, item_id) AS pop_centile
FROM joined
ORDER BY item_id
"""
        atomic_parquet_write(con, sql, dest, expected_mb=512)
        log.info("p3: item covariates written")

    def verify(self, params):
        con = connect(4)
        outputs = self.outputs()
        floor = int(params["core_item_floor"])
        n_target = int(params["eval_user_sample"])
        stats_items = str(MODEL_DIR / "stats_item_train.parquet")
        stats_users = str(MODEL_DIR / "stats_user_train.parquet")
        try:
            map_table = features_table("item_spotify_map")
            ext_table = features_table("item_audio_features_ext")
            fuzzy_table = features_table("item_spotify_fuzzy")
            genres_table = features_table("item_spotify_genres")
            tags_table = features_table("item_tags")
            works_table = features_table("item_works")
            for table, key in [
                (map_table, "map_items"),
                (ext_table, "audio_ext_items"),
                (fuzzy_table, "fuzzy_rows"),
            ]:
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{table}')"
                ).fetchone()[0]
                check(
                    n == FROZEN[key],
                    f"{Path(table).name} holds {n} rows, "
                    f"frozen expects {FROZEN[key]}",
                )
            for table, key in [
                (genres_table, "genres_items"),
                (tags_table, "tags_direct_items"),
                (works_table, "works_items"),
            ]:
                n = con.execute(
                    f"SELECT count(DISTINCT item_id) FROM read_parquet(" f"'{table}')"
                ).fetchone()[0]
                check(
                    n == FROZEN[key],
                    f"{Path(table).name} covers {n} items, "
                    f"frozen expects {FROZEN[key]}",
                )

            prop = str(outputs["proposals"])
            n_prop, sum_u, sum_p, min_p = con.execute(
                f"SELECT count(*), sum(p_uniform), sum(p_pop), min(p_pop) "
                f"FROM read_parquet('{prop}')"
            ).fetchone()
            n_core = con.execute(
                f"SELECT count(*) FROM read_parquet('{stats_items}') "
                f"WHERE pairs >= {floor}"
            ).fetchone()[0]
            check(
                n_prop == n_core,
                f"proposal count {n_prop} does not match core {n_core}",
            )
            check(abs(sum_u - 1.0) < 1e-6, f"p_uniform sums to {sum_u}")
            check(abs(sum_p - 1.0) < 1e-6, f"p_pop sums to {sum_p}")
            check(
                min_p is not None and min_p > 0,
                "a proposal probability is not positive",
            )
            worst = con.execute(f"""
                SELECT min(s.pairs) FROM read_parquet('{prop}') p
                JOIN read_parquet('{stats_items}') s ON p.item_id = s.item_id
            """).fetchone()[0]
            check(worst >= floor, f"a proposal item carries only {worst} train pairs")

            data = np.load(outputs["proposals_npz"])
            cols = con.execute(
                f"SELECT item_id, p_pop FROM read_parquet('{prop}') "
                f"ORDER BY item_id"
            ).fetchnumpy()
            check(
                np.array_equal(
                    np.asarray(data["item_ids"]), np.asarray(cols["item_id"])
                ),
                "npz item ids differ from the parquet",
            )
            check(
                np.array_equal(np.asarray(data["p_pop"]), np.asarray(cols["p_pop"])),
                "npz p_pop differs from the parquet",
            )

            ev = str(outputs["eval_users"])
            n_ev, n_distinct = con.execute(
                f"SELECT count(*), count(DISTINCT user_id) "
                f"FROM read_parquet('{ev}')"
            ).fetchone()
            check(
                n_ev == n_target, f"eval sample holds {n_ev} users, expected {n_target}"
            )
            check(n_distinct == n_ev, "duplicate users in the eval sample")
            missing = con.execute(f"""
                SELECT count(*) FROM read_parquet('{ev}') e
                LEFT JOIN read_parquet('{stats_users}') s
                       ON e.user_id = s.user_id
                WHERE s.user_id IS NULL
            """).fetchone()[0]
            check(missing == 0, f"{missing} eval users are absent from train stats")
            schema = load_json(MODEL_DIR / "schema_interactions.json") or {}
            ts_mode = schema.get("ts_mode")
            mism = con.execute(f"""
                WITH b AS (
                    SELECT user_id,
                           ntile(10) OVER (ORDER BY pairs, user_id) AS d,
                           CAST({year_expr('min_first_ts', ts_mode)} AS INTEGER) AS y
                    FROM read_parquet('{stats_users}'))
                SELECT count(*) FROM read_parquet('{ev}') e
                JOIN b ON e.user_id = b.user_id
                WHERE e.activity_decile != b.d OR e.first_year != b.y
            """).fetchone()[0]
            check(mism == 0, f"{mism} eval users carry a stale stratum")

            cov = str(outputs["covariates"])
            n_cov, n_d = con.execute(
                f"SELECT count(*), count(DISTINCT item_id) "
                f"FROM read_parquet('{cov}')"
            ).fetchone()
            check(
                n_cov == FROZEN["items"],
                f"covariate rows {n_cov}, catalog holds {FROZEN['items']}",
            )
            check(n_d == n_cov, "duplicate items in the covariates")
            bridged = con.execute(
                f"SELECT count(*) FROM read_parquet('{cov}') "
                f"WHERE tier != 'unmatched'"
            ).fetchone()[0]
            check(
                bridged == FROZEN["bridge_items"],
                f"bridged items {bridged}, frozen expects " f"{FROZEN['bridge_items']}",
            )

            has_audio_n = con.execute(
                f"SELECT count(*) FROM read_parquet('{cov}') " f"WHERE has_audio"
            ).fetchone()[0]
            check(
                has_audio_n == FROZEN["audio_ext_items"],
                f"items with audio features {has_audio_n}, expected "
                f"{FROZEN['audio_ext_items']}",
            )
            fuzzy_afless = con.execute(
                f"SELECT count(*) FROM read_parquet('{cov}') "
                f"WHERE tier = 'fuzzy_dur' AND NOT has_audio"
            ).fetchone()[0]
            check(
                fuzzy_afless == FROZEN["audio_ext_afless"],
                f"fuzzy_dur items without audio {fuzzy_afless}, frozen "
                f"expects {FROZEN['audio_ext_afless']}",
            )
            map_afless = con.execute(
                f"SELECT count(*) FROM read_parquet('{cov}') "
                f"WHERE tier IN ('isrc', 'isrc+url', 'url') "
                f"AND NOT has_audio"
            ).fetchone()[0]
            check(
                map_afless == FROZEN["map_afless"],
                f"map-tier items without audio {map_afless}, frozen "
                f"expects {FROZEN['map_afless']}",
            )

            band_min, band_max = con.execute(
                f"SELECT min(pop_centile), max(pop_centile) "
                f"FROM read_parquet('{cov}')"
            ).fetchone()
            check(
                band_min == 1 and band_max == 100,
                "popularity bands fall outside 1 to 100",
            )
            null_tier = con.execute(
                f"SELECT count(*) FROM read_parquet('{cov}') " f"WHERE tier IS NULL"
            ).fetchone()[0]
            check(null_tier == 0, "NULL tier in the covariates")
        finally:
            con.close()

    def report(self, params):
        con = connect(4)
        outputs = self.outputs()
        prop = str(outputs["proposals"])
        ev = str(outputs["eval_users"])
        cov = str(outputs["covariates"])
        ext_table = features_table("item_audio_features_ext")
        lines = ["# P3 sampling report", ""]
        add = lines.append
        try:
            n_core = con.execute(
                f"SELECT count(*) FROM read_parquet('{prop}')"
            ).fetchone()[0]
            sum_u, sum_p = con.execute(
                f"SELECT sum(p_uniform), sum(p_pop) " f"FROM read_parquet('{prop}')"
            ).fetchone()
            add(
                f"- core items (train pairs >= "
                f"{params['core_item_floor']}): {n_core}"
            )
            add(f"- p_uniform sum: {sum_u:.9f}")
            add(f"- p_pop sum: {sum_p:.9f}")
            add("")
            n_ev = con.execute(f"SELECT count(*) FROM read_parquet('{ev}')").fetchone()[
                0
            ]
            cells = con.execute(
                f"SELECT count(*) FROM (SELECT activity_decile, first_year "
                f"FROM read_parquet('{ev}') GROUP BY 1, 2)"
            ).fetchone()[0]
            add(f"- eval users: {n_ev}, stratum cells covered: {cells}")
            add("")
            add("Eval sample by activity decile:")
            for decile, n in con.execute(
                f"SELECT activity_decile, count(*) "
                f"FROM read_parquet('{ev}') GROUP BY 1 ORDER BY 1"
            ).fetchall():
                add(f"- decile {decile}: {n}")
            add("")
            add("Eval sample by first-year cohort:")
            for year, n in con.execute(
                f"SELECT first_year, count(*) "
                f"FROM read_parquet('{ev}') GROUP BY 1 ORDER BY 1"
            ).fetchall():
                add(f"- {year}: {n}")
            add("")
            add("Item covariates, tier counts:")
            for tier, n in con.execute(
                f"SELECT tier, count(*) FROM read_parquet('{cov}') "
                f"GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall():
                add(f"- {tier}: {n}")
            add("")
            add(
                "Bridged items without audio features "
                "(expected: 3,092 map, 6,116 fuzzy, name_only by design):"
            )
            af_map, af_fuzzy, af_name = con.execute(f"""
                SELECT count(*) FILTER (WHERE tier IN ('isrc', 'isrc+url', 'url') AND NOT has_audio),
                       count(*) FILTER (WHERE tier = 'fuzzy_dur' AND NOT has_audio),
                       count(*) FILTER (WHERE tier = 'fuzzy_name' AND NOT has_audio)
                FROM read_parquet('{cov}')
            """).fetchone()
            add(f"- map tiers: {af_map}")
            add(f"- fuzzy_dur: {af_fuzzy}")
            add(f"- fuzzy_name: {af_name} (excluded by design)")
            add("")
            add("Availability (of 6,637,382 items):")
            for label, column in [
                ("audio features", "has_audio"),
                ("direct MB tags", "has_tags_direct"),
                ("artist MB tags", "has_tags_artist"),
                ("Spotify genres", "has_genres"),
                ("work links", "has_work"),
            ]:
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{cov}') " f"WHERE {column}"
                ).fetchone()[0]
                add(f"- {label}: {n}")
            cov_a, cov_v, cov_b, item_v = con.execute(f"""
                SELECT CAST(sum(train_listens) FILTER (WHERE has_audio) AS DOUBLE) / sum(train_listens),
                       CAST(sum(train_listens) FILTER (WHERE has_tags_direct OR has_tags_artist OR has_genres) AS DOUBLE) / sum(train_listens),
                       CAST(sum(train_listens) FILTER (WHERE tier != 'unmatched') AS DOUBLE) / sum(train_listens),
                       CAST(count(*) FILTER (WHERE has_tags_direct OR has_tags_artist OR has_genres) AS DOUBLE) / count(*)
                FROM read_parquet('{cov}')
            """).fetchone()
            add("")
            add("Train-listen-weighted coverage, expected value in brackets:")
            add(f"- audio features: {cov_a:.4f} [expected 0.8747]")
            add(f"- explanation vocabulary: {cov_v:.4f} [expected 0.9593]")
            add(f"- bridged: {cov_b:.4f} [expected 0.9452]")
            add(f"- vocabulary item share: {item_v:.4f} [expected 0.673]")
            sp = con.execute(
                f"SELECT count(*) FROM read_parquet('{cov}') "
                f"WHERE sp_popularity IS NOT NULL"
            ).fetchone()[0]
            add(
                f"- sp_popularity present: {sp} items "
                f"(2025 snapshot, covariate only)"
            )
            bmin, bmax = con.execute(
                f"SELECT min(n), max(n) FROM (SELECT pop_centile, count(*) "
                f"AS n FROM read_parquet('{cov}') GROUP BY 1)"
            ).fetchone()
            add(f"- popularity bands: 100 bands, {bmin} to {bmax} items each")
            add("")
            add(
                "Audio feature NULL diagnostics over the ext table "
                "(informative, no assert. tempo = 0 is a missing marker):"
            )
            diag = con.execute(f"""
                SELECT count(*) FILTER (WHERE tempo IS NULL),
                       count(*) FILTER (WHERE "key" IS NULL),
                       count(*) FILTER (WHERE "mode" IS NULL),
                       count(*) FILTER (WHERE time_signature IS NULL),
                       count(*) FILTER (WHERE duration_ms IS NULL),
                       count(*) FILTER (WHERE danceability IS NULL),
                       count(*) FILTER (WHERE energy IS NULL),
                       count(*) FILTER (WHERE loudness IS NULL),
                       count(*) FILTER (WHERE speechiness IS NULL),
                       count(*) FILTER (WHERE acousticness IS NULL),
                       count(*) FILTER (WHERE instrumentalness IS NULL),
                       count(*) FILTER (WHERE liveness IS NULL),
                       count(*) FILTER (WHERE valence IS NULL),
                       count(*) FILTER (WHERE sp_popularity IS NULL),
                       count(*) FILTER (WHERE tempo = 0)
                FROM read_parquet('{ext_table}')
            """).fetchone()
            labels = [
                "tempo",
                "key",
                "mode",
                "time_signature",
                "duration_ms",
                "danceability",
                "energy",
                "loudness",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "sp_popularity",
                "tempo = 0 (missing marker)",
            ]
            for label, value in zip(labels, diag):
                add(f"- {label}: {value}")
        finally:
            con.close()
        atomic_text_write("\n".join(lines) + "\n", outputs["report"])
        log.info("p3 report written")

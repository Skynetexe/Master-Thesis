"""Stage P8: the parts of the fusion-layer feature set that P4 does not
produce, and how they combine with P4's tables.

P4 already builds item_audio_std.parquet (ten z-scored columns,
NULL-masked tempo, one-hot key, mode and time signature),
item_tags_direct.npz, item_tags_artist.npz, item_genres.npz (sparse CSR,
item rows), and item_work_flags.parquet (full item coverage, has_work
plus five boolean flags). This stage builds none of them again.

What is missing:
- match_method as an explicit category. It lives in
  item_spotify_map.parquet (isrc / isrc+url / url) and, for the fuzzy
  tier, in the basis column of item_spotify_fuzzy.parquet (name+duration
  / name_only). P4 reads neither file.
- Explicit flags for whether an item has audio, tags or genres at all. A
  missing row in a sparse CSR or a missing join match is not a stated
  false. The fusion layer must tell "z_danceability is 0 because that is
  this item's value" apart from "z_danceability is 0 because no audio
  bridge exists for this item".

Output: item_coverage_flags.parquet, one row per item_id (every item_id in
item_ids.parquet, not only the items with a content signal):
    item_id
    has_audio, has_tempo
    has_tags_direct, has_tags_artist, has_genres
    mm_isrc, mm_isrc_url, mm_url, mm_fuzzy_dur, mm_fuzzy_name, mm_none
        (one-hot, exactly one true per row, in the form of P4's key_,
        mode_ and ts_ one-hot columns, and checked as P4 checks them:
        each row sums to exactly 1)

To build a training matrix from these tables, join this table's item_id
to item_audio_std.parquet, the tag and genre CSRs (row index = item_id,
as in P4) and item_work_flags.parquet. This table supplies the flags and
the category. It repeats no value of those tables.

No later stage reads this table. The final model does not use it.
"""

import numpy as np
import pandas as pd

from .common import (
    atomic_parquet_write,
    check,
    connect,
    features_table,
    file_sha256,
    log,
    marker_path,
    with_thread_ladder,
)
from .config import FROZEN_MODEL_ROWS, MODEL_DIR


def _has_rows_by_item_id(npz_path):
    """From a P4-built CSR (rows = item_id directly), which row indices
    have at least one nonzero. Returns a DataFrame with item_id, has_flag,
    built from indptr alone. The sparse arrays themselves never get
    touched, so this is cheap regardless of how many nonzeros exist."""
    z = np.load(npz_path)
    indptr = z["indptr"]
    nnz_per_row = np.diff(indptr)
    return pd.DataFrame(
        {
            "item_id": np.arange(len(nnz_per_row), dtype=np.int64),
            "has": nnz_per_row > 0,
        }
    )


class StageP8:
    name = "p8_fusion_features"
    upstream = ["p4_features"]

    def outputs(self):
        return {
            "coverage": MODEL_DIR / "item_coverage_flags.parquet",
            "report": MODEL_DIR / "reports" / "p8_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        return inputs

    def run(self, params, files):
        outputs = self.outputs()

        def work(con):
            item_ids = features_table("item_ids")
            audio_std = MODEL_DIR / "item_audio_std.parquet"
            sp_map = features_table("item_spotify_map")
            sp_fuzzy = features_table("item_spotify_fuzzy")

            tags_direct = _has_rows_by_item_id(MODEL_DIR / "item_tags_direct.npz")
            tags_artist = _has_rows_by_item_id(MODEL_DIR / "item_tags_artist.npz")
            genres = _has_rows_by_item_id(MODEL_DIR / "item_genres.npz")
            con.register("tags_direct_has", tags_direct)
            con.register("tags_artist_has", tags_artist)
            con.register("genres_has", genres)

            atomic_parquet_write(
                con,
                f"""
                WITH mm AS (
                    SELECT i.item_id,
                           COALESCE(
                               m.match_method,
                               CASE f.basis WHEN 'name+duration' THEN 'fuzzy_dur'
                                            WHEN 'name_only' THEN 'fuzzy_name' END,
                               'none') AS match_method
                    FROM read_parquet('{item_ids}') i
                    LEFT JOIN read_parquet('{sp_map}') m ON m.item_id = i.item_id
                    LEFT JOIN read_parquet('{sp_fuzzy}') f ON f.item_id = i.item_id
                )
                SELECT i.item_id,
                       (a.item_id IS NOT NULL) AS has_audio,
                       (a.item_id IS NOT NULL AND a.z_tempo IS NOT NULL) AS has_tempo,
                       COALESCE(td.has, false) AS has_tags_direct,
                       COALESCE(ta.has, false) AS has_tags_artist,
                       COALESCE(g.has, false) AS has_genres,
                       (mm.match_method = 'isrc') AS mm_isrc,
                       (mm.match_method = 'isrc+url') AS mm_isrc_url,
                       (mm.match_method = 'url') AS mm_url,
                       (mm.match_method = 'fuzzy_dur') AS mm_fuzzy_dur,
                       (mm.match_method = 'fuzzy_name') AS mm_fuzzy_name,
                       (mm.match_method = 'none') AS mm_none
                FROM read_parquet('{item_ids}') i
                JOIN mm ON mm.item_id = i.item_id
                LEFT JOIN read_parquet('{audio_std}') a ON a.item_id = i.item_id
                LEFT JOIN tags_direct_has td ON td.item_id = i.item_id
                LEFT JOIN tags_artist_has ta ON ta.item_id = i.item_id
                LEFT JOIN genres_has g ON g.item_id = i.item_id
                ORDER BY i.item_id
            """,
                outputs["coverage"],
                expected_mb=256,
            )

        with_thread_ladder(work, params["ladder"])

    def verify(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            path = str(outputs["coverage"])
            item_ids = features_table("item_ids")

            n_cov, n_ids = con.execute(f"""
                SELECT (SELECT count(*) FROM read_parquet('{path}')),
                       (SELECT count(*) FROM read_parquet('{item_ids}'))
            """).fetchone()
            check(
                n_cov == n_ids,
                f"item_coverage_flags has {n_cov:,} rows, "
                f"item_ids has {n_ids:,}. They must match exactly.",
            )

            n_bad_onehot = con.execute(f"""
                SELECT count(*) FROM read_parquet('{path}')
                WHERE (mm_isrc::INT + mm_isrc_url::INT + mm_url::INT
                       + mm_fuzzy_dur::INT + mm_fuzzy_name::INT
                       + mm_none::INT) <> 1
            """).fetchone()[0]
            check(
                n_bad_onehot == 0,
                f"{n_bad_onehot} rows do not sum to exactly 1 across "
                f"the match_method one-hot columns",
            )

            n_audio_true, n_audio_std = con.execute(f"""
                SELECT (SELECT count(*) FROM read_parquet('{path}') WHERE has_audio),
                       (SELECT count(*) FROM read_parquet('{MODEL_DIR / "item_audio_std.parquet"}'))
            """).fetchone()
            check(
                n_audio_true == n_audio_std,
                f"has_audio is true for {n_audio_true:,} items, but "
                f"item_audio_std.parquet has {n_audio_std:,} rows. "
                f"These must match exactly, because the flag comes from "
                f"exactly this join.",
            )

            expected = FROZEN_MODEL_ROWS["item_coverage_flags.parquet"]
            check(
                n_cov == expected,
                f"item_coverage_flags.parquet: measured {n_cov} rows, "
                f"frozen expects {expected}",
            )
        finally:
            con.close()

    def report(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            path = str(outputs["coverage"])
            lines = ["# P8 report", ""]

            for flag in [
                "has_audio",
                "has_tempo",
                "has_tags_direct",
                "has_tags_artist",
                "has_genres",
            ]:
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{path}') " f"WHERE {flag}"
                ).fetchone()[0]
                lines.append(f"- {flag}: {n:,}")

            lines.append("")
            lines.append("match_method distribution:")
            mm_cols = [
                "mm_isrc",
                "mm_isrc_url",
                "mm_url",
                "mm_fuzzy_dur",
                "mm_fuzzy_name",
                "mm_none",
            ]
            for col in mm_cols:
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{path}') " f"WHERE {col}"
                ).fetchone()[0]
                lines.append(f"- {col}: {n:,}")

            outputs["report"].parent.mkdir(parents=True, exist_ok=True)
            outputs["report"].write_text("\n".join(lines) + "\n")
            log.info("\n".join(lines))
        finally:
            con.close()

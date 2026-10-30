"""Paths, frozen contracts, and default parameters.

The data folder comes from the environment variable THESIS_DATA_ROOT. The
default is the folder data/ next to the pipeline package. That folder
holds no data in a new copy of the code, so the runner stops there at "no
interaction files" and computes nothing.
"""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("THESIS_DATA_ROOT", CODE_DIR / "data"))
CURATED = CODE_DIR / "curated"
PARQUET = ROOT / "parquet"
INTERACTIONS_DIR = PARQUET / "interactions"
FEATURES_DIR = PARQUET / "features"
MODEL_DIR = PARQUET / "model"
SCRATCH_DIR = PARQUET / "scratch"
DUCKDB_TMP = ROOT / "duckdb_tmp"

# Frozen numbers of the prepared data. These values are contracts.
# A failed check means: stop and investigate. Never adjust a number.
FROZEN = {
    "pairs": 3_687_114_567,
    "users": 594_415,
    "items": 6_637_382,
    "matrix_listens": 21_071_051_915,
    "fuzzy_name_duration": 2_393_709,
    "map_items": 1_020_242,
    "fuzzy_rows": 3_031_107,
    "bridge_items": 4_051_349,
    "audio_ext_items": 3_404_743,
    "audio_ext_afless": 6_116,
    "map_afless": 3_092,
    "audio_map_items": 1_017_150,
    "tempo_zero_map": 954,
    "ts_outside_map": 13_047,
    "tags_direct_pairs": 4_196_587,
    "tags_artist_rows": 22_262_351,
    "tags_artist_items": 3_847_858,
    "genres_vocab": 684,
    "genres_rows": 7_459_187,
    "artist_map_rows": 458_488,
    "artist_map_artists": 444_356,
    "artist_map_sp": 455_408,
    "genres_items": 2_814_416,
    "tags_direct_items": 1_133_043,
    "works_items": 1_614_856,
}

DUCKDB_SETTINGS = {
    "memory_limit": "24GB",
    "temp_directory": str(DUCKDB_TMP),
    "preserve_insertion_order": "false",
    "max_temp_directory_size": "150GiB",
    "TimeZone": "UTC",  # year math must never see the local clock
}

DEFAULT_PARAMS = {
    "seed": 20260907,
    "eligibility_min_train_pairs": 5,
    "item_pair_floors": [1, 3, 5, 10, 20],
    "user_pair_floors": [1, 5, 20, 100],
    "primary_holdout": "last_ts",
    "double_check_split": True,
    "ladder": [8, 4, 2, 1],
    "core_item_floor": 5,
    "eval_user_sample": 20000,
    "neg_proposal_power": 0.75,
    "tag_min_df": 50,
}

# Documented or measured-frozen row counts. P1 asserts every one.
FROZEN_TABLE_ROWS = {
    "artist_spotify_map.parquet": 458_488,
    "complete_recording_popularity.parquet": 6_685_542,
    "gid_redirect_resolution.parquet": 93_753,
    "item_artists.parquet": 7_317_477,
    "item_audio_features.parquet": 1_017_150,
    "item_audio_features_ext.parquet": 3_404_743,
    "item_catalog.parquet": 6_685_542,
    "item_ids.parquet": 6_637_382,
    "item_listens.parquet": 6_637_382,
    "item_spotify_candidates.parquet": 127_031,
    "item_spotify_fuzzy.parquet": 3_031_107,
    "item_spotify_genres.parquet": 7_459_187,
    "item_spotify_isrc_candidates.parquet": 4_907_685,
    "item_spotify_map.parquet": 1_020_242,
    "item_tags.parquet": 4_196_587,
    "item_tags_artist.parquet": 22_262_351,
    "item_works.parquet": 1_638_583,
    "recording_canonical_map.parquet": 6_685_542,
    "user_ids.parquet": 594_415,
}

# Just under the measured p999 (515): the extreme end of the real
# distribution is a small, clearly separated cluster of outliers.
ARTIST_DEGREE_CAP = 500
# Deliberately generous: a tight cap here would drop "Yesterday" along
# with "Intro" (see stage_p7.py's docstring).
WORK_DEGREE_CAP = 700
GENERIC_WORK_TITLES_PATH = MODEL_DIR / "generic_work_titles.txt"

FROZEN_MODEL_ROWS = {
    "item_edges.parquet": 295_059_237,
    "item_coverage_flags.parquet": 6_637_382,
    "item_cooccurrence.parquet": 12_725_887,
}

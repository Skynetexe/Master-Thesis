"""Stage P10: per-user profile vectors for the fusion model.

Builds, for every user in the train table, a weighted audio-feature mean
and weighted tag and genre profile vectors: the user side of per-pair
features. The item side is in P4's outputs (item_audio_std.parquet,
item_tags_direct.npz, item_tags_artist.npz, item_genres.npz), and this
stage reuses them rather than deriving anything again from the raw tag
and genre tables of the features/ layer.

The users run in batches (BATCH_SIZE at a time), not all 594K at once.
Each batch's train slice is materialized first (filter before join, not
after), then the batch's item history is aggregated against P4's
item-side matrices by sparse matrix multiplication. No join against the
raw item_tags.parquet, item_tags_artist.parquet or
item_spotify_genres.parquet tables happens in this stage. That join gave
tens of billions of naive pairs in the diagnostics. P4's aggregated and
verified per-item matrices avoid it.

No separate "history item-id set" output: the train table already holds
one row per (user, item) pair, queryable by user_id.

Outputs:
- user_audio_mean.parquet: one row per user_id, weighted mean of the
  ten z-scored audio columns over the user's history, plus has_tempo
  (false where the user's entire history lacks a tempo value, the same
  NULL-tempo case that P4 masks at the item level, carried through rather
  than averaged as if it were zero) and n_items (history size).
- user_tags_direct.npz, user_tags_artist.npz, user_genres.npz: sparse,
  user rows, in the on-disk form of P4's item-side CSRs
  (indices/indptr/shape), plus data, because these are weighted sums, not
  binary presence.

Row alignment, checked against the real data: P4's item-side CSRs have
exactly n_items + 1 rows (item_ids.parquet: 6,637,382 distinct items,
item_id 1 to 6,637,382, so each CSR has 6,637,383 rows). Row 0 is an
unused slot for the item_id that does not exist. Row i is item_id i, with
no shift.

The .npz files use a local atomic write (_atomic_savez). The report is
written directly.

No later stage reads these profiles. The final model does not use them.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse

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
from .config import MODEL_DIR

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
AUDIO_ZCOLS = [f"z_{c}" for c in CONT_COLS]
TEMPO_COL_IDX = AUDIO_ZCOLS.index("z_tempo")

# Users per batch: twice the size measured at about 20 s per batch for the
# materialize step. Make it smaller if a batch is slow.
BATCH_SIZE = 10_000


def _atomic_savez(path: Path, **arrays):
    """The atomic write of common.py (a temporary file, fsync, then
    replace), for the .npz format.

    The temp name has to end in .npz itself: np.savez_compressed
    silently appends .npz to whatever name it's given if that name
    doesn't already end with it, so a ".npz.tmp" name comes out on
    disk as ".npz.tmp.npz", and the fsync/replace below then goes
    looking for a file that was never written under that name."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_item_csr(npz_path: Path) -> scipy.sparse.csr_matrix:
    """Loads one of P4's item-side CSRs. data is optional in P4's own
    on-disk format (tag presence may be implicitly binary), so this falls
    back to an all-ones data array rather than assuming either way."""
    z = np.load(npz_path)
    indices = z["indices"]
    indptr = z["indptr"]
    shape = tuple(int(x) for x in z["shape"])
    data = z["data"] if "data" in z.files else np.ones(len(indices), dtype=np.float32)
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def _load_dense_audio(con, audio_std_path: str, n_rows: int) -> np.ndarray:
    """Dense (n_rows x 10) array indexed by item_id directly, with the
    same row-0 slot as the CSRs (module docstring). NaN for any item
    missing audio entirely, or missing tempo specifically."""
    cols = ", ".join(AUDIO_ZCOLS)
    arrs = con.execute(f"""
        SELECT item_id, {cols} FROM read_parquet('{audio_std_path}')
    """).fetchnumpy()
    out = np.full((n_rows, len(AUDIO_ZCOLS)), np.nan, dtype=np.float32)
    idx = arrs["item_id"]
    for j, name in enumerate(AUDIO_ZCOLS):
        out[idx, j] = arrs[name]
    return out


def _process_batch(
    con,
    train_glob: str,
    user_batch: list,
    tag_direct,
    tag_artist,
    genre,
    audio_vals,
    audio_mask,
):
    """One batch of users, start to finish: materialize their train
    slice first (filter before join), build a small sparse interaction
    matrix, and multiply it against P4's item-side matrices."""
    n_items = audio_vals.shape[0]
    n_batch = len(user_batch)

    con.execute("CREATE OR REPLACE TEMP TABLE batch_users (user_id BIGINT)")
    con.executemany("INSERT INTO batch_users VALUES (?)", [(u,) for u in user_batch])

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE batch_train AS
        SELECT t.user_id, t.item_id, t.w_log
        FROM read_parquet('{train_glob}') t
        JOIN batch_users b ON b.user_id = t.user_id
    """)
    arrs = con.execute("SELECT user_id, item_id, w_log FROM batch_train").fetchnumpy()

    user_batch_arr = np.asarray(user_batch, dtype=np.int64)  # already sorted
    user_pos = np.searchsorted(user_batch_arr, arrs["user_id"])
    item_ids = arrs["item_id"].astype(np.int64)
    weights = arrs["w_log"].astype(np.float32)

    interaction = scipy.sparse.csr_matrix(
        (weights, (user_pos, item_ids)), shape=(n_batch, n_items)
    )

    batch_tags_direct = (interaction @ tag_direct).tocsr()
    batch_tags_artist = (interaction @ tag_artist).tocsr()
    batch_genres = (interaction @ genre).tocsr()

    weighted_sum = interaction @ audio_vals  # (n_batch, 10), dense
    weighted_denom = interaction @ audio_mask  # (n_batch, 10), dense
    audio_mean = weighted_sum / np.maximum(weighted_denom, 1e-9)
    has_tempo = weighted_denom[:, TEMPO_COL_IDX] > 0

    # indptr diff, not (interaction != 0).sum(axis=1): != against a
    # sparse matrix densifies the whole thing first, which at n_items =
    # 6.6M would try to allocate on the order of tens of GB per batch.
    n_items_per_user = np.diff(interaction.indptr)

    return {
        "tags_direct": batch_tags_direct,
        "tags_artist": batch_tags_artist,
        "genres": batch_genres,
        "audio_mean": audio_mean,
        "has_tempo": has_tempo,
        "n_items": n_items_per_user,
    }


class StageP10:
    name = "p10_profiles"
    upstream = ["p4_features", "p25_train"]

    def outputs(self):
        return {
            "audio": MODEL_DIR / "user_audio_mean.parquet",
            "tags_direct": MODEL_DIR / "user_tags_direct.npz",
            "tags_artist": MODEL_DIR / "user_tags_artist.npz",
            "genres": MODEL_DIR / "user_genres.npz",
            "report": MODEL_DIR / "reports" / "p10_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage_id in self.upstream:
            try:
                inputs[f"marker:{stage_id}"] = file_sha256(marker_path(stage_id))
            except FileNotFoundError:
                inputs[f"marker:{stage_id}"] = None
        return inputs

    def run(self, params, files):
        outputs = self.outputs()
        train_glob = str(MODEL_DIR / "interactions_train_*.parquet")
        audio_std_path = str(MODEL_DIR / "item_audio_std.parquet")

        def work(con):
            item_ids_path = features_table("item_ids")
            n_items = (
                con.execute(
                    f"SELECT count(*) FROM read_parquet('{item_ids_path}')"
                ).fetchone()[0]
                + 1
            )  # +1 for the phantom row-0 slot

            log.info(
                f"p10: loading item-side matrices ({n_items:,} rows incl. phantom)"
            )
            tag_direct = _load_item_csr(MODEL_DIR / "item_tags_direct.npz")
            tag_artist = _load_item_csr(MODEL_DIR / "item_tags_artist.npz")
            genre = _load_item_csr(MODEL_DIR / "item_genres.npz")
            check(
                tag_direct.shape[0] == n_items,
                f"item_tags_direct.npz has {tag_direct.shape[0]:,} rows, "
                f"expected {n_items:,}",
            )
            check(
                tag_artist.shape[0] == n_items,
                f"item_tags_artist.npz has {tag_artist.shape[0]:,} rows, "
                f"expected {n_items:,}",
            )
            check(
                genre.shape[0] == n_items,
                f"item_genres.npz has {genre.shape[0]:,} rows, "
                f"expected {n_items:,}",
            )

            audio_raw = _load_dense_audio(con, audio_std_path, n_items)
            audio_mask = (~np.isnan(audio_raw)).astype(np.float32)
            audio_vals = np.nan_to_num(audio_raw, nan=0.0)

            user_ids = sorted(r[0] for r in con.execute(f"""
                SELECT DISTINCT user_id FROM read_parquet('{train_glob}')
            """).fetchall())
            n_batches = -(-len(user_ids) // BATCH_SIZE)
            log.info(
                f"p10: {len(user_ids):,} distinct users, "
                f"batching {BATCH_SIZE:,} at a time ({n_batches} batches)"
            )

            tags_direct_parts, tags_artist_parts, genres_parts = [], [], []
            audio_rows = []
            for i in range(0, len(user_ids), BATCH_SIZE):
                batch = user_ids[i : i + BATCH_SIZE]
                result = _process_batch(
                    con,
                    train_glob,
                    batch,
                    tag_direct,
                    tag_artist,
                    genre,
                    audio_vals,
                    audio_mask,
                )
                tags_direct_parts.append(result["tags_direct"])
                tags_artist_parts.append(result["tags_artist"])
                genres_parts.append(result["genres"])
                for j, u in enumerate(batch):
                    audio_rows.append(
                        (
                            u,
                            *[float(x) for x in result["audio_mean"][j]],
                            bool(result["has_tempo"][j]),
                            int(result["n_items"][j]),
                        )
                    )
                log.info(
                    f"p10: batch {i // BATCH_SIZE + 1}/{n_batches} done "
                    f"({i + len(batch):,}/{len(user_ids):,} users)"
                )

            final_tags_direct = scipy.sparse.vstack(tags_direct_parts).tocsr()
            final_tags_artist = scipy.sparse.vstack(tags_artist_parts).tocsr()
            final_genres = scipy.sparse.vstack(genres_parts).tocsr()

            _atomic_savez(
                outputs["tags_direct"],
                indices=final_tags_direct.indices,
                indptr=final_tags_direct.indptr,
                data=final_tags_direct.data,
                shape=np.array(final_tags_direct.shape, dtype=np.int64),
            )
            _atomic_savez(
                outputs["tags_artist"],
                indices=final_tags_artist.indices,
                indptr=final_tags_artist.indptr,
                data=final_tags_artist.data,
                shape=np.array(final_tags_artist.shape, dtype=np.int64),
            )
            _atomic_savez(
                outputs["genres"],
                indices=final_genres.indices,
                indptr=final_genres.indptr,
                data=final_genres.data,
                shape=np.array(final_genres.shape, dtype=np.int64),
            )

            cols = ["user_id"] + AUDIO_ZCOLS + ["has_tempo", "n_items"]
            audio_df = pd.DataFrame(audio_rows, columns=cols)
            con.register("audio_profile", audio_df)
            atomic_parquet_write(
                con,
                "SELECT * FROM audio_profile ORDER BY user_id",
                outputs["audio"],
                expected_mb=64,
            )

        with_thread_ladder(work, params["ladder"])

    def verify(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            train_glob = str(MODEL_DIR / "interactions_train_*.parquet")
            n_users = con.execute(f"""
                SELECT count(DISTINCT user_id) FROM read_parquet('{train_glob}')
            """).fetchone()[0]

            n_audio = con.execute(f"""
                SELECT count(*) FROM read_parquet('{outputs["audio"]}')
            """).fetchone()[0]
            check(
                n_audio == n_users,
                f"user_audio_mean.parquet has {n_audio:,} rows, "
                f"expected {n_users:,} distinct users",
            )

            n_dup = con.execute(f"""
                SELECT count(*) - count(DISTINCT user_id)
                FROM read_parquet('{outputs["audio"]}')
            """).fetchone()[0]
            check(n_dup == 0, f"{n_dup} duplicate user_id rows in audio output")

            n_zero = con.execute(f"""
                SELECT count(*) FROM read_parquet('{outputs["audio"]}')
                WHERE n_items <= 0
            """).fetchone()[0]
            check(n_zero == 0, f"{n_zero} users have n_items <= 0")

            for key in ("tags_direct", "tags_artist", "genres"):
                z = np.load(outputs[key])
                shape = tuple(int(x) for x in z["shape"])
                check(
                    shape[0] == n_users,
                    f"user_{key}.npz has {shape[0]:,} rows, "
                    f"expected {n_users:,} distinct users",
                )
                check(
                    len(z["indptr"]) == shape[0] + 1,
                    f"user_{key}.npz indptr length {len(z['indptr'])} "
                    f"!= n_rows + 1 ({shape[0] + 1})",
                )
        finally:
            con.close()

    def report(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            row = con.execute(f"""
                SELECT count(*), avg(n_items), median(n_items),
                       sum(CASE WHEN has_tempo THEN 1 ELSE 0 END)
                FROM read_parquet('{outputs["audio"]}')
            """).fetchone()
            n_users, avg_items, median_items, n_has_tempo = row
            lines = [
                "# P10 report\n",
                f"- users profiled: {n_users:,}",
                f"- history size per user: mean {avg_items:,.1f}, "
                f"median {median_items:,.0f}",
                f"- users with at least one tempo-bearing item: "
                f"{n_has_tempo:,} ({100 * n_has_tempo / n_users:.1f}%)",
            ]
            for key in ("tags_direct", "tags_artist", "genres"):
                z = np.load(outputs[key])
                lines.append(
                    f"- {key}: {len(z['data']):,} nonzeros, "
                    f"shape {tuple(int(x) for x in z['shape'])}"
                )
            # The report is written directly (module docstring).
            outputs["report"].parent.mkdir(parents=True, exist_ok=True)
            outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
            log.info(
                f"p10: {n_users:,} users profiled, "
                f"mean history {avg_items:,.1f} items"
            )
        finally:
            con.close()

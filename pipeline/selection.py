"""The model-selection user sample, disjoint from the eval sample.

Every choice of hyperparameters, early-stopping point or experiment arm
is made on these users' validation picks. The eval sample's test picks
are read once, for the final numbers. Fusion training uses the remaining
users. This keeps the three roles apart.

The rule is deterministic and stores no random state. Rank every
eligible user outside the eval sample by SHA-256 of "<SEED>:<user_id>",
then take the first SIZE. Eligible means the user has split picks (P2).
The same call returns the same users on any machine.
"""

import hashlib

import numpy as np
import pandas as pd

from . import evaluation as ev

SEED = "selection-20260922"
SIZE = 20_000


def select(candidate_ids, excluded_ids, seed=SEED, size=SIZE):
    """Pure function behind selection_user_ids(). Returns sorted ids."""
    cand = np.unique(np.asarray(candidate_ids, dtype=np.int64))
    excl = np.unique(np.asarray(excluded_ids, dtype=np.int64))
    pool = np.setdiff1d(cand, excl, assume_unique=True)
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} eligible users, need {size}")
    keys = np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}:{int(u)}".encode()).digest()[:8], "big"
            )
            for u in pool
        ],
        dtype=np.uint64,
    )
    order = np.lexsort((pool, keys))  # hash first, user_id breaks ties
    return np.sort(pool[order[:size]])


def selection_user_ids(con):
    eligible = con.execute(
        f"SELECT DISTINCT user_id FROM read_parquet('{ev.SPLITS}')"
    ).fetchnumpy()["user_id"]
    evals = con.execute(
        f"SELECT user_id FROM read_parquet('{ev.EVAL_USERS}')"
    ).fetchnumpy()["user_id"]
    return select(eligible, evals)


def val_picks(con, users):
    """(n, 2) int64: each user's val_last and val_first item, in the order
    of users. P2 guarantees one of each for every eligible user."""
    users = np.asarray(users, dtype=np.int64)
    con.register("sel_users_df", pd.DataFrame({"user_id": users}))
    try:
        cols = con.execute(f"""
            SELECT s.user_id,
                   max(CASE WHEN s.is_val_last THEN s.item_id END) AS val_last,
                   max(CASE WHEN s.is_val_first THEN s.item_id END) AS val_first
            FROM read_parquet('{ev.SPLITS}') s
            JOIN sel_users_df u ON u.user_id = s.user_id
            GROUP BY s.user_id
            ORDER BY s.user_id
        """).fetchnumpy()
    finally:
        con.unregister("sel_users_df")
    for name in ("val_last", "val_first"):
        if np.ma.is_masked(cols[name]):
            raise ValueError(f"a selection user lacks a {name} pick")
    got = np.asarray(cols["user_id"], dtype=np.int64)
    if not np.array_equal(got, np.sort(users)):
        raise ValueError("some selection users have no split rows")
    picks = np.stack(
        [
            np.asarray(cols["val_last"], dtype=np.int64),
            np.asarray(cols["val_first"], dtype=np.int64),
        ],
        axis=1,
    )
    order = np.searchsorted(got, users)
    return picks[order]

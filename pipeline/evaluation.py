"""Full-catalog ranking evaluation on the stratified eval sample.

Candidate rule: all item slots except the phantom slots and the user's
train items. Held-out picks of the same user stay in the candidate set.
Tie rule: equal scores rank by item_id ascending. Both rules belong to
the pre-registered protocol.
"""

import glob as globlib
import hashlib

import numpy as np

from .common import load_json
from .config import FROZEN, MODEL_DIR

EVAL_USERS = str(MODEL_DIR / "eval_users.parquet")
SPLITS = str(MODEL_DIR / "splits.parquet")
COVARIATES = str(MODEL_DIR / "item_covariates.parquet")
ITEM_STATS = str(MODEL_DIR / "stats_item_train.parquet")
TRAIN_GLOB = str(MODEL_DIR / "interactions_train_*.parquet")

TIER_NAMES = {
    1: "isrc",
    2: "isrc+url",
    3: "url",
    4: "fuzzy_dur",
    5: "fuzzy_name",
    6: "unmatched",
}


def npz_hash(path):
    data = np.load(path)
    digest = hashlib.sha256()
    for name in sorted(data.files):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    return digest.hexdigest()


def load_id_slots():
    doc = load_json(MODEL_DIR / "id_spaces.json") or {}
    items = doc.get("items", {})
    return (int(items.get("slots", FROZEN["items"] + 1)), int(items.get("min", 1)))


def load_eval_frame(con):
    rows = con.execute(f"""
        WITH picks AS (
            SELECT user_id,
                   max(CASE WHEN is_test_last THEN item_id END) AS test_last,
                   max(CASE WHEN is_test_first THEN item_id END) AS test_first
            FROM read_parquet('{SPLITS}')
            GROUP BY user_id)
        SELECT e.user_id, e.activity_decile, p.test_last, p.test_first
        FROM read_parquet('{EVAL_USERS}') e
        LEFT JOIN picks p ON e.user_id = p.user_id
        ORDER BY e.user_id
    """).fetchall()
    dropped = sum(1 for r in rows if r[2] is None or r[3] is None)
    rows = [r for r in rows if r[2] is not None and r[3] is not None]
    user = np.array([r[0] for r in rows], dtype=np.int64)
    decile = np.array([r[1] for r in rows], dtype=np.int64)
    test_last = np.array([r[2] for r in rows], dtype=np.int64)
    test_first = np.array([r[3] for r in rows], dtype=np.int64)
    return user, decile, test_last, test_first, dropped


def load_train_items(con, users_sql):
    user_parts, item_parts = [], []
    for path in sorted(globlib.glob(TRAIN_GLOB)):
        cols = con.execute(f"""
            SELECT user_id, item_id FROM read_parquet('{path}')
            WHERE user_id IN ({users_sql})
        """).fetchnumpy()
        user_parts.append(np.asarray(cols["user_id"], dtype=np.int64))
        item_parts.append(np.asarray(cols["item_id"], dtype=np.int32))
    u = np.concatenate(user_parts)
    i = np.concatenate(item_parts)
    order = np.argsort(u, kind="stable")
    u, i = u[order], i[order]
    starts = np.flatnonzero(np.concatenate(([True], u[1:] != u[:-1])))
    return dict(zip(u[starts].tolist(), np.split(i, starts[1:]))), len(u)


def popularity_scores(con, n_slots, id_min):
    scores = np.zeros(n_slots, dtype=np.int64)
    cols = con.execute(
        f"SELECT item_id, listens FROM read_parquet('{ITEM_STATS}')"
    ).fetchnumpy()
    scores[np.asarray(cols["item_id"], dtype=np.int64)] = np.asarray(
        cols["listens"], dtype=np.int64
    )
    scores[:id_min] = -1
    return scores


def global_ranks(scores):
    """Deterministic total order: score desc, then slot index asc."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def user_ranks(ranks, test_items, user_ids, train_items):
    out = np.empty(len(test_items), dtype=np.int64)
    for k in range(len(test_items)):
        r = int(ranks[int(test_items[k])])
        tr = train_items.get(int(user_ids[k]))
        if tr is not None and len(tr):
            out[k] = r - int(np.count_nonzero(ranks[tr] < r))
        else:
            out[k] = r
    return out


def brute_rank(scores, test_item, train_ids):
    """Independent rank for one user. No global ranks, no sort."""
    s = scores.copy()
    if train_ids is not None and len(train_ids):
        s[train_ids] = -2
    t = int(test_item)
    st = int(s[t])
    return 1 + int(np.count_nonzero(s > st)) + int(np.count_nonzero(s[:t] == st))


def metric_block(ranks):
    n = int(len(ranks))
    rf = ranks.astype(np.float64)
    dcg = np.where(ranks <= 100, 1.0 / np.log2(rf + 1.0), 0.0)
    return {
        "n_users": n,
        "recall_at_10": float(np.count_nonzero(ranks <= 10) / n),
        "recall_at_100": float(np.count_nonzero(ranks <= 100) / n),
        "ndcg_at_100": float(dcg.mean()),
        "mrr": float((1.0 / rf).mean()),
        "median_rank": float(np.median(rf)),
        "mean_rank": float(rf.mean()),
    }


def slice_blocks(ranks, labels):
    out = {}
    for value in sorted(set(labels.tolist())):
        mask = labels == value
        out[str(value)] = metric_block(ranks[mask])
    return out


def covariate_arrays(con, n_slots):
    cols = con.execute(f"""
        SELECT item_id, pop_centile, train_pairs,
               CASE tier WHEN 'isrc' THEN 1 WHEN 'isrc+url' THEN 2
                         WHEN 'url' THEN 3 WHEN 'fuzzy_dur' THEN 4
                         WHEN 'fuzzy_name' THEN 5 ELSE 6 END AS tier_code
        FROM read_parquet('{COVARIATES}')
    """).fetchnumpy()
    ids = np.asarray(cols["item_id"], dtype=np.int64)
    pop = np.zeros(n_slots, dtype=np.int64)
    pairs = np.full(n_slots, -1, dtype=np.int64)
    tier = np.zeros(n_slots, dtype=np.int64)
    pop[ids] = np.asarray(cols["pop_centile"], dtype=np.int64)
    pairs[ids] = np.asarray(cols["train_pairs"], dtype=np.int64)
    tier[ids] = np.asarray(cols["tier_code"], dtype=np.int64)
    return pop, pairs, tier

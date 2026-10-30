"""Training pairs for the fusion model: the users, their positive picks and
their sampled negatives. stage_p11.py runs these functions on the real
data. test_p11.py checks them against slow reference versions.

Why val picks. Every concept feature comes from the train pairs only (P4,
P6, P9b, P10). A model trained on train pairs would learn from features
that already contain its own positives. P2 holds out four picks per user
outside train, so a (user, val pick) pair has out-of-sample features,
exactly like a (user, test pick) pair at evaluation time. This is
stacked generalization (Wolpert 1992, Breiman 1996): the combiner learns
from inputs that were computed without the rows it learns from. The
Netflix Prize teams did the same with their probe set (Töscher, Jahrer &
Bell 2009, section 3.1). Kaufman et al. (2012) call the rule
learn-predict separation.

Roles:
1. The eval sample never appears. Its test picks are read once, in P14.
2. The selection sample (pipeline/selection.py) gets pairs with split
   SELECTION. P13 may monitor a loss on them, never train on them.
3. A val pick that equals the user's test pick in the other variant is
   not a positive. No pick serves as a positive in one role and as a
   test pick in another.
4. When val_last and val_first are the same track, it is one positive
   with both flags, so it does not count twice.

Negatives. Each user gets NEG_PER_KIND independent draws from each of two
proposals, restricted to the tracks that the user has neither played in
train nor holds as a held-out pick:
- KIND_UNIFORM: uniform over every item slot that the evaluation ranks
  (pipeline/evaluation.py). BPR's criterion is a smoothed AUC (Rendle et
  al. 2009, section 4.1.1), and with these negatives it is the AUC over
  the evaluation's own candidate set. P3's p_uniform covers only its
  5,201,011 core tracks, so 21.6 percent of the candidates would never
  appear as a negative, and every shape function would be extrapolated
  there. This changes rule 4 of the roles above.
- KIND_POP: P3's p_pop, proportional to train listens^0.75. The 0.75
  power comes from word2vec (Mikolov et al. 2013, section 2.2), where it
  beat the unigram and uniform noise distributions. For recommendation
  it is a heuristic, not a proven gain. Rendle & Freudenthaler (2014)
  found uniform sampling slow when popularity is skewed. Their static
  popularity oversampling still did worse than uniform sampling, as He
  et al. (2016) summarise. The experiment arms settle it on this data.

Draws are independent and with replacement, so the accepted draws of one
user and proposal are an exact sample from the proposal restricted to
that user's allowed tracks. Each user and proposal has its own random
stream, seeded by (SEED, user_id, kind). So a user's negatives do not
depend on the other users, the block size or the number of workers, and
any user's rows can be recomputed alone (stage_p11.verify() does).

Exact sampling probabilities. For a negative j of user u:
- KIND_UNIFORM: q = 1 / n_allowed[u].
- KIND_POP: q = p_pop[j] / pop_allowed[u].
n_allowed and pop_allowed are stored per user. With them, P13 can
correct a loss for the proposal, for example the log-Q correction of Yi
et al. (2019), or its mixture form Q* for a mixed arm (Yang et al.
2020). The explanation side needs this: a sampler shifts the learned
popularity shape, so without the correction, the popularity contribution
in an explanation depends on a training choice that the listener never
sees.

Unobserved is not negative. A sampled track may be one the user would
like (Pan et al. 2008, Marlin & Zemel 2009, Ding et al. 2020). The
rejection removes the only such tracks we know of: the user's own
held-out picks.
"""

import numpy as np

KIND_POSITIVE, KIND_UNIFORM, KIND_POP = 0, 1, 2
KIND_NAMES = {KIND_POSITIVE: "positive", KIND_UNIFORM: "uniform", KIND_POP: "p_pop"}
PICK_VAL_LAST, PICK_VAL_FIRST = 1, 2
SPLIT_TRAIN, SPLIT_SELECTION = 0, 1
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_SELECTION: "selection"}
MAX_BATCH = 1 << 20  # raw draws per batch, at most
MAX_DRAWS = 1 << 27  # raw draws per user and proposal, at most


def pick_table(user, item, test_last, val_last, test_first, val_first):
    """Each user's four held-out picks, from the rows of splits.parquet.
    Returns (users, picks): users sorted, picks (n, 4) int64 in the order
    test_last, val_last, test_first, val_first. Raises when a user lacks
    a role or holds one twice (P2's verify() forbids both)."""
    user = np.asarray(user, dtype=np.int64)
    item = np.asarray(item, dtype=np.int64)
    users, inv = np.unique(user, return_inverse=True)
    picks = np.full((len(users), 4), -1, dtype=np.int64)
    for col, flag in enumerate((test_last, val_last, test_first, val_first)):
        flag = np.asarray(flag, dtype=bool)
        rows = inv[flag]
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"a user holds role {col} twice")
        picks[rows, col] = item[flag]
    if (picks < 0).any():
        raise ValueError("a user lacks a held-out role")
    return users, picks


def positives(picks):
    """The positives of each user under rules 3 and 4 of the module
    docstring. picks: (n, 4) from pick_table(). Returns (indptr, items,
    flags, counts): CSR arrays over the users, rows sorted by item, flags
    PICK_VAL_LAST | PICK_VAL_FIRST, and counts of the rule outcomes."""
    tl, vl, tf, vf = (picks[:, k] for k in range(4))
    keep_l = vl != tf
    keep_f = vf != tl
    merged = keep_l & keep_f & (vl == vf)
    n = len(picks)
    rows, items, flags = [], [], []
    for k in range(n):
        if merged[k]:
            rows.append(k)
            items.append(vl[k])
            flags.append(PICK_VAL_LAST | PICK_VAL_FIRST)
            continue
        cand = []
        if keep_l[k]:
            cand.append((vl[k], PICK_VAL_LAST))
        if keep_f[k]:
            cand.append((vf[k], PICK_VAL_FIRST))
        for x, f in sorted(cand):
            rows.append(k)
            items.append(x)
            flags.append(f)
    rows = np.asarray(rows, dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n))))
    counts = {
        "val_last_is_test_first": int((~keep_l).sum()),
        "val_first_is_test_last": int((~keep_f).sum()),
        "val_last_is_val_first": int(merged.sum()),
        "no_positive": int((~keep_l & ~keep_f).sum()),
    }
    return (
        indptr.astype(np.int64),
        np.asarray(items, dtype=np.int64),
        np.asarray(flags, dtype=np.int8),
        counts,
    )


def user_rng(seed, user, kind):
    """The random stream of one user and one proposal."""
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(seed), int(user), int(kind)]))
    )


def uniform_sampler(lo, hi):
    """Map uniform doubles in [0, 1) to item slots lo..hi-1, each with
    probability 1 / (hi - lo)."""
    n = int(hi) - int(lo)

    def sample(u):
        return int(lo) + np.minimum((u * n).astype(np.int64), n - 1)

    return sample


def pop_sampler(items, cum):
    """Map uniform doubles in [0, 1) to items[k] with probability
    (cum[k] - cum[k-1]) / cum[-1] (inverse CDF, half-open intervals)."""
    items = np.asarray(items, dtype=np.int64)
    cum = np.asarray(cum, dtype=np.float64)
    last = len(items) - 1

    def sample(u):
        k = np.searchsorted(cum, u * cum[-1], side="right")
        return items[np.minimum(k, last)]

    return sample


def member(values, sorted_row):
    """Boolean mask: values found in sorted_row."""
    values = np.asarray(values)
    if len(sorted_row) == 0:
        return np.zeros(len(values), dtype=bool)
    pos = np.searchsorted(sorted_row, values)
    pos = np.minimum(pos, len(sorted_row) - 1)
    return np.asarray(sorted_row)[pos] == values


def draw_allowed(rng, k, sample, excluded_sorted):
    """The first k draws of rng, mapped by sample(), that are not in
    excluded_sorted. Returns (items, consumed): consumed is the number of
    raw draws up to and including the k-th accepted one. The batch size
    does not change the result, because the stream is read in order and
    one double is one draw."""
    items = np.empty(k, dtype=np.int64)
    got = consumed = 0
    batch = max(16, 2 * k)
    while got < k:
        cand = sample(rng.random(batch))
        idx = np.flatnonzero(~member(cand, excluded_sorted))
        need = k - got
        if len(idx) >= need:
            items[got:] = cand[idx[:need]]
            consumed += int(idx[need - 1]) + 1
            got = k
        else:
            items[got : got + len(idx)] = cand[idx]
            got += len(idx)
            consumed += batch
            batch = min(2 * batch, MAX_BATCH)
        if consumed > MAX_DRAWS:
            raise ValueError(
                f"only {got} of {k} allowed draws after " f"{consumed:,} raw draws"
            )
    return items, consumed


def reference_draws(seed, user, kind, k, sample, excluded):
    """Slow reference for draw_allowed(): one double at a time from the
    user's own stream, checked against a Python set."""
    rng = user_rng(seed, user, kind)
    excluded = set(int(x) for x in excluded)
    out, consumed = [], 0
    while len(out) < k:
        x = int(sample(np.array([rng.random()]))[0])
        consumed += 1
        if x not in excluded:
            out.append(x)
    return np.asarray(out, dtype=np.int64), consumed


def excluded_items(train_row, held):
    """Sorted union of a user's train items (a sorted CSR row) and
    held-out picks, in time linear in the row length."""
    row = np.asarray(train_row, dtype=np.int64)
    held = np.unique(np.asarray(held, dtype=np.int64))
    new = held[~member(held, row)]
    if len(new) == 0:
        return row
    return np.insert(row, np.searchsorted(row, new), new)


def user_rows(
    seed,
    user,
    pos_items,
    pos_flags,
    train_row,
    held,
    k,
    samplers,
    n_candidates,
    p_pop_slot,
    pop_total,
):
    """All rows of one user: positives first, then k draws per proposal.
    samplers: {kind: sample function}. p_pop_slot holds p_pop per item
    slot (0 outside P3's core) and pop_total its sum. Returns (items,
    kinds, picks, draws, n_allowed, pop_allowed, consumed): n_allowed is
    the number of candidate slots left after the exclusions, pop_allowed
    the p_pop mass left, and consumed maps each kind to its raw draw
    count."""
    excl = excluded_items(train_row, held)
    n_allowed = int(n_candidates) - len(excl)
    pop_allowed = float(pop_total) - float(p_pop_slot[excl].sum())
    parts = [
        (
            np.asarray(pos_items, dtype=np.int64),
            np.full(len(pos_items), KIND_POSITIVE, dtype=np.int8),
            np.asarray(pos_flags, dtype=np.int8),
            np.zeros(len(pos_items), dtype=np.int16),
        )
    ]
    consumed = {}
    for kind in (KIND_UNIFORM, KIND_POP):
        got, used = draw_allowed(user_rng(seed, user, kind), k, samplers[kind], excl)
        consumed[kind] = used
        parts.append(
            (
                got,
                np.full(k, kind, dtype=np.int8),
                np.zeros(k, dtype=np.int8),
                np.arange(k, dtype=np.int16),
            )
        )
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
        np.concatenate([p[3] for p in parts]),
        n_allowed,
        pop_allowed,
        consumed,
    )

"""Independent check of pipeline/concepts.py against a brute-force
implementation built from plain dicts and loops. The brute force shares
no code with the module under test.

Checks, on a synthetic catalog with deliberate gaps:
1. smooth_idf matches a counted IDF.
2. full_catalog() matches brute force for every concept, every slot and
   every user, for both value and has_data.
3. pairs() matches full_catalog() at the same slots.
4. Every evidence list sums to its value.
5. Zero and missing stay distinct. Both cases occur and are told apart.
6. Scaling a user's weights leaves every concept unchanged.
7. Slot 0 is missing for every concept.
8. The two co-listening graphs (same artist, other artists) are directed,
   as P9b stores them: row i holds track i's own neighbours, and a pair
   can carry a different value in each direction. Each graph feeds its
   own concept, and ItemSide refuses a missing graph.
9. An ItemSide built with transposed=False (the form P12 uses) gives the
   same pairs() and evidence(), and full_catalog() refuses it.

Usage (no data needed), from the repository root:
    python3 test_concepts.py
"""

import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from pipeline.concepts import (
    AUDIO_DIMS,
    COLISTEN_CONCEPTS,
    CONTINUOUS,
    CREDIT_CONCEPTS,
    NEVER_MISSING,
    SIGMA_FLOOR,
    TERM_CONCEPTS,
    ItemSide,
    build_user_state,
    evidence,
    full_catalog,
    pairs,
    smooth_idf,
)

TOL = 1e-10
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# Catalog
rng = np.random.default_rng(20260922)
N = 300  # item slots, slot 0 is the phantom
D = len(AUDIO_DIMS)


def rand_rows(n_cols, p_empty, max_per_row, integer_values=False):
    rows = [dict()]  # slot 0 stays empty
    for _ in range(1, N):
        row = {}
        if rng.random() >= p_empty:
            k = int(rng.integers(1, max_per_row + 1))
            for c in rng.choice(n_cols, size=k, replace=False):
                row[int(c)] = float(rng.integers(1, 6)) if integer_values else 1.0
        rows.append(row)
    return rows


def to_csr(rows, n_cols):
    r, c, v = [], [], []
    for i, row in enumerate(rows):
        for col, val in row.items():
            r.append(i)
            c.append(col)
            v.append(val)
    return sp.csr_matrix((v, (r, c)), shape=(len(rows), n_cols))


term_rows = {
    "tags_direct": rand_rows(25, 0.45, 3),
    "tags_artist": rand_rows(25, 0.25, 5, integer_values=True),
    "genres": rand_rows(12, 0.35, 2),
}
term_cols = {"tags_direct": 25, "tags_artist": 25, "genres": 12}
credit_rows = {
    "artists": rand_rows(60, 0.03, 2),
    "works": rand_rows(40, 0.7, 1),
}
credit_cols = {"artists": 60, "works": 40}

# Two directed graphs, as P9b stores them: row a lists a's own neighbours.
# The same pair can appear in both rows with different values, so
# full_catalog() must read the rows of the candidate items, not of the
# history items.
colisten_rows = {}
for name, n_edges in (("same_artist", 500), ("other_artists", 700)):
    rows = [dict() for _ in range(N)]
    for _ in range(n_edges):
        a, b = (int(x) for x in rng.integers(1, N, size=2))
        if a == b:
            continue
        rows[a][b] = float(rng.uniform(0.01, 1.0))
    colisten_rows[name] = rows

audio = rng.normal(size=(N, D))
audio[0] = np.nan
no_audio = rng.choice(np.arange(1, N), size=40, replace=False)
audio[no_audio] = np.nan
no_tempo = rng.choice(np.arange(1, N), size=25, replace=False)
audio[no_tempo, AUDIO_DIMS.index("z_tempo")] = np.nan

popularity = np.log1p(rng.integers(0, 5000, size=N).astype(np.float64))
tier = rng.integers(0, 6, size=N)

space = ItemSide(
    term_mats={k: to_csr(v, term_cols[k]) for k, v in term_rows.items()},
    credit_mats={k: to_csr(v, credit_cols[k]) for k, v in credit_rows.items()},
    colisten={k: to_csr(v, N) for k, v in colisten_rows.items()},
    audio=audio.copy(),
    popularity=popularity.copy(),
    tier=tier.copy(),
)

# Brute force


def brute_idf(rows, n_cols):
    n_docs = sum(1 for r in rows if r)
    df = [0] * n_cols
    for r in rows:
        for c in r:
            df[c] += 1
    return [math.log((1 + n_docs) / (1 + d)) + 1 for d in df]


brute_idfs = {k: brute_idf(v, term_cols[k]) for k, v in term_rows.items()}


def brute_concepts(items, weights):
    W = sum(weights)
    hist = dict(zip(items, weights))
    out = {c: ([math.nan] * N, [False] * N) for c in CONTINUOUS}

    for concept, name in TERM_CONCEPTS.items():
        idf = brute_idfs[name]
        p = defaultdict(float)
        for h, w in hist.items():
            for c, x in term_rows[name][h].items():
                p[c] += w * x / W
        u_norm = math.sqrt(sum((idf[c] * v) ** 2 for c, v in p.items()))
        for i in range(1, N):
            row = term_rows[name][i]
            if not row or u_norm == 0:
                continue
            i_norm = math.sqrt(sum((x * idf[c]) ** 2 for c, x in row.items()))
            numer = sum(x * idf[c] ** 2 * p.get(c, 0.0) for c, x in row.items())
            out[concept][0][i] = numer / (i_norm * u_norm)
            out[concept][1][i] = True

    for concept, name in CREDIT_CONCEPTS.items():
        s = defaultdict(float)
        for h, w in hist.items():
            for a in credit_rows[name][h]:
                s[a] += w / W
        for i in range(1, N):
            row = credit_rows[name][i]
            if not row:
                continue
            out[concept][0][i] = max(s.get(a, 0.0) for a in row)
            out[concept][1][i] = True

    for concept, name in COLISTEN_CONCEPTS.items():
        for i in range(1, N):
            row = colisten_rows[name][i]
            if not row:
                continue
            out[concept][0][i] = sum(sc * hist.get(h, 0.0) for h, sc in row.items()) / W
            out[concept][1][i] = True

    mu, sigma, ok = [0.0] * D, [1.0] * D, [False] * D
    for d in range(D):
        wd = sum(w for h, w in hist.items() if not math.isnan(audio[h, d]))
        if wd == 0:
            continue
        m1 = (
            sum(w * audio[h, d] for h, w in hist.items() if not math.isnan(audio[h, d]))
            / wd
        )
        m2 = (
            sum(
                w * audio[h, d] ** 2
                for h, w in hist.items()
                if not math.isnan(audio[h, d])
            )
            / wd
        )
        mu[d] = m1
        sigma[d] = max(math.sqrt(max(m2 - m1 * m1, 0.0)), SIGMA_FLOOR)
        ok[d] = True
    for i in range(1, N):
        dims = [d for d in range(D) if ok[d] and not math.isnan(audio[i, d])]
        if not dims:
            continue
        out["audio_fit"][0][i] = -sum(
            ((audio[i, d] - mu[d]) / sigma[d]) ** 2 for d in dims
        ) / len(dims)
        out["audio_fit"][1][i] = True

    for i in range(1, N):
        out["popularity"][0][i] = popularity[i]
        out["popularity"][1][i] = True
    return out


def same(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.abs(a - b) <= TOL * np.maximum(1.0, np.abs(b))
    return bool(np.all(both_nan | close))


# Users
all_items = np.arange(1, N)
untagged = [i for i in all_items if not term_rows["tags_direct"][i]]
users = {
    "broad": rng.choice(all_items, size=60, replace=False),
    "large": rng.choice(all_items, size=180, replace=False),
    "single": rng.choice(all_items, size=1, replace=False),
    "no_direct_tags": rng.choice(untagged, size=15, replace=False),
    "no_audio": rng.choice(no_audio, size=10, replace=False),
}

# Checks
for name in term_rows:
    check(
        np.allclose(space.idf[name], brute_idfs[name], rtol=0, atol=1e-12),
        f"idf mismatch for {name}",
    )

seen_zero = defaultdict(bool)
seen_missing = defaultdict(bool)
for label, hist_items in users.items():
    weights = np.log1p(rng.integers(1, 800, size=len(hist_items)).astype(np.float64))
    state = build_user_state(hist_items, weights, space)
    full = full_catalog(state, space)
    brute = brute_concepts([int(x) for x in hist_items], [float(x) for x in weights])

    for c in CONTINUOUS:
        v, h = full[c]
        bv, bh = brute[c]
        check(
            np.array_equal(h, np.array(bh)),
            f"{label}/{c}: has_data differs from brute force",
        )
        check(same(v, bv), f"{label}/{c}: value differs from brute force")
        check(not h[0], f"{label}/{c}: slot 0 is not missing")
        if c not in NEVER_MISSING:
            seen_missing[c] |= bool(np.any(~h[1:]))
        seen_zero[c] |= bool(np.any(h & (v == 0.0)))

    cand = np.concatenate(([0], rng.choice(all_items, size=50, replace=False)))
    pr = pairs(state, space, cand)
    for c in CONTINUOUS:
        check(
            np.array_equal(pr[c][1], full[c][1][cand]),
            f"{label}/{c}: pairs() has_data differs from full_catalog()",
        )
        check(
            same(pr[c][0], full[c][0][cand]),
            f"{label}/{c}: pairs() value differs from full_catalog()",
        )
    check(
        np.array_equal(pr["match_tier"], full["match_tier"][cand]),
        f"{label}: pairs() tier differs",
    )

    for item in rng.choice(all_items, size=60, replace=False):
        ev = evidence(state, space, int(item))
        for c in CONTINUOUS:
            e = ev[c]
            check(
                bool(e["has_data"]) == bool(full[c][1][item]),
                f"{label}/{c}/{item}: evidence has_data differs",
            )
            if e["has_data"] and c != "popularity":
                total = sum(t["term"] for t in e["terms"])
                check(
                    abs(total - e["value"]) <= 1e-12 * max(1.0, abs(e["value"])),
                    f"{label}/{c}/{item}: evidence terms sum to {total}, "
                    f"value is {e['value']}",
                )
        for c, name in CREDIT_CONCEPTS.items():
            e = ev[c]
            if e["has_data"]:
                check(
                    e["credit"] in credit_rows[name][int(item)],
                    f"{label}/{c}/{item}: cited credit is not the item's",
                )

    scaled = build_user_state(hist_items, weights * 7.3, space)
    full_s = full_catalog(scaled, space)
    for c in CONTINUOUS:
        check(
            same(full_s[c][0], full[c][0]),
            f"{label}/{c}: scaling the weights changed the value",
        )

    if label == "no_direct_tags":
        check(
            not full["tag_affinity"][1].any(),
            "a user with no tagged history still got tag_affinity values",
        )
    if label == "no_audio":
        check(
            not full["audio_fit"][1].any(),
            "a user with no audio history still got audio_fit values",
        )

for c in ("tag_affinity", "artist_affinity") + tuple(COLISTEN_CONCEPTS):
    check(seen_zero[c], f"{c}: no zero-valued case with data was exercised")
    check(seen_missing[c], f"{c}: no missing case was exercised")
check(
    len(set(COLISTEN_CONCEPTS)) == 2
    and all(c in CONTINUOUS for c in COLISTEN_CONCEPTS),
    "both co-listening concepts are model inputs",
)
try:
    ItemSide(
        term_mats={k: to_csr(v, term_cols[k]) for k, v in term_rows.items()},
        credit_mats={k: to_csr(v, credit_cols[k]) for k, v in credit_rows.items()},
        colisten={"same_artist": to_csr(colisten_rows["same_artist"], N)},
        audio=audio.copy(),
        popularity=popularity.copy(),
        tier=tier.copy(),
    )
    check(False, "ItemSide accepted a missing co-listening graph")
except ValueError:
    pass

# 9. transposed=False, the form P12 builds: pairs() and evidence() read
# only the forward lists, so they give the same results, and
# full_catalog() refuses the ItemSide instead of reading an empty
# transpose.
lean = ItemSide(
    term_mats={k: to_csr(v, term_cols[k]) for k, v in term_rows.items()},
    credit_mats={k: to_csr(v, credit_cols[k]) for k, v in credit_rows.items()},
    colisten={k: to_csr(v, N) for k, v in colisten_rows.items()},
    audio=audio.copy(),
    popularity=popularity.copy(),
    tier=tier.copy(),
    transposed=False,
)
check(
    not lean.colisten_t and len(space.colisten_t) == 2,
    "transposed=False still built the transposed lists",
)
for label, hist_items in users.items():
    w = np.log1p(np.arange(1, len(hist_items) + 1, dtype=np.float64))
    s_full = build_user_state(hist_items, w, space)
    s_lean = build_user_state(hist_items, w, lean)
    cand = np.arange(N)
    a, b = pairs(s_full, space, cand), pairs(s_lean, lean, cand)
    for c in CONTINUOUS:
        check(
            np.array_equal(a[c][1], b[c][1]) and same(b[c][0], a[c][0]),
            f"{label}/{c}: transposed=False changed pairs()",
        )
    for item in rng.choice(all_items, size=20, replace=False):
        e1 = evidence(s_full, space, int(item))
        e2 = evidence(s_lean, lean, int(item))
        for c in CONTINUOUS:
            check(
                e1[c]["has_data"] == e2[c]["has_data"]
                and same([e2[c]["value"]], [e1[c]["value"]])
                and e1[c]["terms"] == e2[c]["terms"],
                f"{label}/{c}/{item}: transposed=False changed evidence()",
            )
    try:
        full_catalog(s_lean, lean)
        check(False, "full_catalog() accepted an ItemSide without " "transposed lists")
    except ValueError:
        pass

if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures[:40]:
        print("  -", f)
    raise SystemExit(1)
print(
    f"PASSED: concepts.py matches brute force for {len(users)} users x "
    f"{N - 1} items x {len(CONTINUOUS)} concepts, pairs() == "
    f"full_catalog(), every evidence list sums to its value, zero and "
    f"missing stay distinct, scaling the weights changes nothing, and "
    f"transposed=False gives the same pairs() and evidence()."
)

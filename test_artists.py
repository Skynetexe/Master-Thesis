"""Checks for pipeline/artists.py. No real data and no DuckDB: a stand-in
connection returns the rows.

1. special_artist_ids() matches the listed names without regard to case
   or outer spaces, and the two MBIDs without regard to case. It leaves
   real bracketed names such as "[dunkelbunt]" alone, and None never
   matches.
2. An artist id counts as special when any of its credited names matches.
3. credit_csr() drops special credits, rows outside the item slots and
   negative artist ids, removes repeats, and sorts each row.
4. load_credits() reads the table through connect(), turns masked
   (NULL) strings into None, returns the same CSR as credit_csr(), and
   counts rows, special artists and tracks with a credit.
5. special_summary() keeps the matched part of the rule apart from the
   printed names: it names the rule part, counts the rows and the
   distinct printed names, ranks the printed names, and orders the
   artists by rows.

Usage, from the repository root:
    python3 test_artists.py
"""

import numpy as np

from pipeline import artists as ar

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# item_id, artist_id, credited name, artist MBID
ROWS = [
    (1, 10, "Stacey Kent", "m-10"),
    (1, 11, " [Unknown] ", "m-11"),  # listed name, case, spaces
    (1, 14, None, ar.UNKNOWN_MBID.upper()),  # MBID in upper case, no name
    (2, 10, "Stacey Kent", "m-10"),
    (2, 12, "Various artists", "m-12"),
    (2, 15, "Third Name", "m-15"),
    (3, 13, "[dunkelbunt]", "m-13"),  # a real artist
    (3, 14, "Some Name", ar.UNKNOWN_MBID),  # special by MBID
    (3, 15, "Fourth Name", "m-15"),
    (4, 15, "Zed Alias", "m-15"),  # special through another row
    (5, 15, "[anonymous]", "m-15"),
    (5, 16, None, None),  # no name and no MBID
    (6, 10, "Stacey Kent", "m-10"),
    (6, 10, "Stacey Kent", "m-10"),  # a repeated row
    (6, 20, "[unknown]", ar.UNKNOWN_MBID),  # special by name and MBID
    (7, 15, "Zed Alias", "m-15"),
    (9, 17, "Out Of Range", "m-17"),  # item 9 is outside 0..7
    (0, 18, "Phantom Slot", "m-18"),  # slot 0 is the phantom
    (7, -1, "Bad Id", "m-19"),
    (7, 19, "B Artist", "m-20"),
    (7, 16, "Another", "m-21"),
]
N_ITEMS = 8
item = np.array([r[0] for r in ROWS])
aid = np.array([r[1] for r in ROWS])
names = [r[2] for r in ROWS]
mbids = [r[3] for r in ROWS]

# 1 and 2.
special = ar.special_artist_ids(aid, names, mbids)
check(
    special.tolist() == [11, 12, 14, 15, 20],
    f"special_artist_ids() finds names, case, spaces, both MBID cases and "
    f"a special name on another row ({special.tolist()})",
)

# 3.
indptr, ids = ar.credit_csr(item, aid, special, N_ITEMS)
got = {x: ids[indptr[x] : indptr[x + 1]].tolist() for x in range(N_ITEMS)}
want = {0: [], 1: [10], 2: [10], 3: [13], 4: [], 5: [16], 6: [10], 7: [16, 19]}
check(
    got == want,
    f"credit_csr() keeps the right credits, sorted and " f"without repeats ({got})",
)


# 4. DuckDB returns a masked array for a column with NULLs. The data
# under the mask is garbage that must never be read.
def masked(values):
    return np.ma.array(
        np.array(["GARBAGE" if v is None else v for v in values], dtype=object),
        mask=[v is None for v in values],
    )


class FakeResult:
    def fetchnumpy(self):
        return {
            "item_id": item,
            "artist_id": aid,
            "artist_name": masked(names),
            "artist_mbid": masked(mbids),
        }


class FakeCon:
    closed = False

    def execute(self, sql):
        self.sql = sql
        return FakeResult()

    def close(self):
        FakeCon.closed = True


con = FakeCon()
ar.connect = lambda threads: con
ar.features_table = lambda name: f"/nowhere/{name}.parquet"
l_indptr, l_ids, stats = ar.load_credits(N_ITEMS)
check(
    np.array_equal(l_indptr, indptr)
    and np.array_equal(l_ids, ids)
    and "item_artists" in con.sql
    and FakeCon.closed,
    "load_credits() reads item_artists and returns credit_csr()'s arrays",
)
want_special = [
    {
        "artist_id": 15,
        "matched_by": ['name "[anonymous]"'],
        "rows": 5,
        "distinct_names": 4,
        "names": [["Zed Alias", 2], ["Fourth Name", 1], ["Third Name", 1]],
    },
    {
        "artist_id": 14,
        "matched_by": ["MBID of [unknown]"],
        "rows": 2,
        "distinct_names": 2,
        "names": [["(no name)", 1], ["Some Name", 1]],
    },
    {
        "artist_id": 11,
        "matched_by": ['name "[unknown]"'],
        "rows": 1,
        "distinct_names": 1,
        "names": [[" [Unknown] ", 1]],
    },
    {
        "artist_id": 12,
        "matched_by": ['name "various artists"'],
        "rows": 1,
        "distinct_names": 1,
        "names": [["Various artists", 1]],
    },
    {
        "artist_id": 20,
        "matched_by": ["MBID of [unknown]", 'name "[unknown]"'],
        "rows": 1,
        "distinct_names": 1,
        "names": [["[unknown]", 1]],
    },
]
check(
    stats
    == {
        "credit_rows": len(ROWS),
        "special_artists": 5,
        "special_rows": 10,
        "special": want_special,
        "tracks_with_credit": 6,
    },
    f"load_credits() counts rows, special artists and tracks, and reads "
    f"masked strings as None ({stats})",
)

# 5. special_summary() alone, with the printed names capped at one.
one = ar.special_summary(aid, names, mbids, special, top_names=1)
check(
    [e["artist_id"] for e in one] == [15, 14, 11, 12, 20]
    and one[0]["names"] == [["Zed Alias", 2]]
    and one[0]["distinct_names"] == 4
    and one[0]["matched_by"] == ['name "[anonymous]"'],
    "special_summary() ranks artists by rows, caps the printed names and "
    "keeps the matched rule part apart from them",
)
check(
    ar.special_summary(aid, names, mbids, []) == [],
    "special_summary() of no special artists is empty",
)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL ARTIST CHECKS PASSED")

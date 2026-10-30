"""The title normalizer of FuzzyBridge.py beside a candidate fix, on a few
names. It needs DuckDB only, no data. Run: python3 normalizer_check.py

The feat/ft branch of the normalizer removes only the marker word ('feat.',
'featuring', 'ft.'), with no '.*$' to remove what follows, as the
hyphen-strip branch does. So 'Artist feat. Someone' normalizes to
'artistsomeone'. The same branch has no word boundary, so "feat" also
matches inside ordinary words: 'Defeat' becomes 'de', and 'Feature Artist'
becomes 'ureartist'. This is the "feat." bug that the thesis lists among
its limitations. It was found too late to fix before the evaluation, so
the bridge keeps the normalizer as it ran.

The script prints the CURRENT regex (copied from FuzzyBridge.py) beside a
CANDIDATE fix on the same cases. The candidate adds \\b (a word boundary,
which DuckDB's RE2 engine supports) before the marker and '.*$' after it,
as the hyphen-strip branch does. The last case shows that the candidate is
not a complete fix: 'feat' at the start of a word still follows a word
boundary.
"""

import duckdb

CURRENT = """regexp_replace(regexp_replace(regexp_replace(regexp_replace(
         lower(%s),
         '\\s+-\\s+.*$', '', 'g'),
         '\\([^)]*\\)|\\[[^]]*\\]', '', 'g'),
         '(feat\\.?|featuring|ft\\.)\\s*', '', 'g'),
         '[^a-z0-9]+', '', 'g')"""

# Candidate fix: \b before the marker group (blocks mid-word matches like
# "defeat"), and .*$ after it (drops the featured artist's name too,
# mirroring the hyphen-strip branch's own shape).
CANDIDATE = """regexp_replace(regexp_replace(regexp_replace(regexp_replace(
         lower(%s),
         '\\s+-\\s+.*$', '', 'g'),
         '\\([^)]*\\)|\\[[^]]*\\]', '', 'g'),
         '\\b(feat\\.?|featuring|ft\\.).*$', '', 'g'),
         '[^a-z0-9]+', '', 'g')"""

cases = [
    (
        "夜に駆ける",
        "",
        "",
        "non-Latin title: empty-key guard territory, unaffected by this fix",
    ),
    (
        "Song Title (Remastered 2011)",
        "songtitle",
        "songtitle",
        "parenthetical: unaffected, both strip it the same way",
    ),
    (
        "Artist feat. Someone",
        "artistsomeone",
        "artist",
        "The finding: current only removes the marker word, not the name "
        "after it. Candidate drops the whole featured-credit clause.",
    ),
    (
        "Somebody - Someone Else Duet",
        "somebody",
        "somebody",
        "hyphen mid-title: unrelated known issue, unaffected by this fix",
    ),
    (
        "Bring It On - Live at Wembley",
        "bringiton",
        "bringiton",
        "hyphen as a real suffix: unaffected, both correct",
    ),
    (
        "Defeat",
        "de",
        "defeat",
        "The bug: 'feat' matches as a bare substring with no word boundary. "
        "Candidate's \\b fixes this specific shape (marker preceded by "
        "another word character).",
    ),
    (
        "Feature Artist",
        "ureartist",
        "",
        "Candidate does not fully fix this one: 'feat' at the very start "
        "of a word still has a word boundary in front of it (start-of-string "
        "counts as one), so \\b alone can't tell 'Feature' apart from a bare "
        "'feat' marker. RE2 has no lookahead, so there's no clean way to also "
        "require a boundary-or-nothing immediately after the marker. Current "
        "produces a wrong but partially-intact string; candidate produces "
        "empty. Neither is right for this case.",
    ),
]

con = duckdb.connect()
print(f"{'input':<32} {'current':<16} {'candidate':<16}")
for text, exp_current, exp_candidate, note in cases:
    got_current = con.execute(f"SELECT {CURRENT % '?'}", [text]).fetchone()[0]
    got_candidate = con.execute(f"SELECT {CANDIDATE % '?'}", [text]).fetchone()[0]
    c_ok = "OK" if got_current == exp_current else f"expected {exp_current!r}"
    d_ok = "OK" if got_candidate == exp_candidate else f"expected {exp_candidate!r}"
    print(f"{text:<32} {got_current!r:<16} {got_candidate!r:<16}")
    print(f"    current: {c_ok}   candidate: {d_ok}")
    print(f"    {note}")
    print()

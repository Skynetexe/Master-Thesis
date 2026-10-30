"""Stage P9: item-item co-occurrence, the collaborative half of the
explanation-centric hybrid: "N listeners who played this also played
that". In the user study of Herlocker, Konstan and Riedl, neighbour
evidence of this kind was the most convincing explanation.

The naive version, every user's full item set joined with itself, was
measured to be infeasible: 20,126,047,540,713 naive pairs over the real
train table, and no per-user breadth cap both keeps the computation
tractable and keeps most of the data (even a cap of 5,000 excludes 76% of
all rows). The median user has 4,784 distinct items, unlike P7's artist
and work fan-out, where a cap could separate a few outliers from a
normal bulk.

The method follows the core idea of DIMSUM (Zadeh & Carlsson, 2014,
all-pairs cosine similarity at scale), moved from MapReduce to a single
DuckDB query: sample toward each user's least popular items before any
pair exists, rather than capping or excluding whole users. This keeps
each user's most distinctive listening, the signal that makes a
co-occurrence edge specific, and it bounds the pairs per user whatever
the size of the history. Unlike DIMSUM's estimator, every count here is
exact over the sampled items, because the output must stand behind a
sentence such as "214 listeners who played both of these".

Steps:
  1. Item popularity (distinct users per item) from the train table.
  2. Per user, keep only the SAMPLE_K least-popular items (QUALIFY on a
     popularity-ascending rank). This bounds the pairs per user to
     C(SAMPLE_K, 2).
  3. Self-join within each user's sampled set, item_id_a < item_id_b,
     aggregated to the raw co_count = count(DISTINCT user_id), the number
     that the explanation sentence quotes.
  4. MIN_SUPPORT filter (co_count >= MIN_SUPPORT), the minimum-support
     threshold of market-basket analysis (Agrawal & Srikant): a pair that
     two users share by coincidence is not a claim worth making.
  5. A cosine-style score, co_count / sqrt(pop_a * pop_b) (Sarwar,
     Karypis, Konstan & Riedl, 2001), kept beside co_count. The score
     decides which edges stay, and co_count decides what the sentence
     says.
  6. Each item keeps its top TOP_N neighbours by score, as in item-based
     collaborative filtering, so an item with hundreds of neighbours above
     the support floor keeps its best few.

SAMPLE_K=100, MIN_SUPPORT=5 and TOP_N=50 are starting points, not
measured optima. SAMPLE_K keeps the worst case (594,415 x C(100,2), about
2.94 billion pairs) within the scale of the other stages.

P9b (stage_p9b.py) replaced these sampled counts with exact ones. No later
stage reads P9's output.
"""

from .common import (
    atomic_parquet_write,
    check,
    connect,
    log,
    marker_path,
    file_sha256,
    with_thread_ladder,
)
from .config import MODEL_DIR

SAMPLE_K = 100
MIN_SUPPORT = 5
TOP_N = 50

TRAIN_GLOB = str(MODEL_DIR / "interactions_train_*.parquet")


class StageP9:
    name = "p9_cooccurrence"
    upstream = ["p25_train"]

    def outputs(self):
        return {
            "edges": MODEL_DIR / "item_cooccurrence.parquet",
            "report": MODEL_DIR / "reports" / "p9_report.md",
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
        sample_k = params.get("cooc_sample_k", SAMPLE_K)
        min_support = params.get("cooc_min_support", MIN_SUPPORT)
        top_n = params.get("cooc_top_n", TOP_N)

        def work(con):
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE item_pop AS
                SELECT item_id, count(DISTINCT user_id) AS pop
                FROM read_parquet('{TRAIN_GLOB}')
                GROUP BY item_id
            """)

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE user_sample AS
                SELECT t.user_id, t.item_id
                FROM read_parquet('{TRAIN_GLOB}') t
                JOIN item_pop p ON p.item_id = t.item_id
                QUALIFY row_number() OVER (
                    PARTITION BY t.user_id ORDER BY p.pop ASC, t.item_id
                ) <= {sample_k}
            """)

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE pairs AS
                SELECT a.item_id AS item_id_a, b.item_id AS item_id_b,
                       count(DISTINCT a.user_id) AS co_count
                FROM user_sample a
                JOIN user_sample b
                  ON a.user_id = b.user_id AND a.item_id < b.item_id
                GROUP BY 1, 2
                HAVING count(DISTINCT a.user_id) >= {min_support}
            """)

            atomic_parquet_write(
                con,
                f"""
                WITH scored AS (
                    SELECT p.item_id_a, p.item_id_b, p.co_count,
                           p.co_count / sqrt(pa.pop * pb.pop) AS score
                    FROM pairs p
                    JOIN item_pop pa ON pa.item_id = p.item_id_a
                    JOIN item_pop pb ON pb.item_id = p.item_id_b
                ),
                ranked AS (
                    SELECT *,
                           row_number() OVER (
                               PARTITION BY item_id_a ORDER BY score DESC, item_id_b
                           ) AS rank_a,
                           row_number() OVER (
                               PARTITION BY item_id_b ORDER BY score DESC, item_id_a
                           ) AS rank_b
                    FROM scored
                )
                SELECT item_id_a, item_id_b, co_count, score
                FROM ranked
                WHERE rank_a <= {top_n} AND rank_b <= {top_n}
                ORDER BY item_id_a, item_id_b
            """,
                outputs["edges"],
                expected_mb=512,
            )

        with_thread_ladder(work, params["ladder"])

    def verify(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            path = str(outputs["edges"])

            rows, ordered, self_loops, bad_support = con.execute(f"""
                SELECT count(*),
                       sum(CASE WHEN item_id_a < item_id_b THEN 1 ELSE 0 END),
                       sum(CASE WHEN item_id_a = item_id_b THEN 1 ELSE 0 END),
                       sum(CASE WHEN co_count < {MIN_SUPPORT} THEN 1 ELSE 0 END)
                FROM read_parquet('{path}')
            """).fetchone()
            check(
                rows == ordered,
                f"{rows - ordered} rows do not satisfy item_id_a < item_id_b",
            )
            check(self_loops == 0, f"{self_loops} self-loop rows")
            check(
                bad_support == 0,
                f"{bad_support} rows fall below the minimum support "
                f"floor of {MIN_SUPPORT}, which the HAVING clause should "
                f"make impossible",
            )

            n_dup = con.execute(f"""
                SELECT count(*) - count(DISTINCT (item_id_a, item_id_b))
                FROM read_parquet('{path}')
            """).fetchone()[0]
            check(n_dup == 0, f"{n_dup} duplicate (item_id_a, item_id_b) rows")

            over_cap = con.execute(f"""
                WITH deg AS (
                    SELECT item_id, count(*) AS n FROM (
                        SELECT item_id_a AS item_id FROM read_parquet('{path}')
                        UNION ALL
                        SELECT item_id_b AS item_id FROM read_parquet('{path}')
                    ) GROUP BY item_id
                )
                SELECT count(*) FROM deg WHERE n > 2 * {TOP_N}
            """).fetchone()[0]
            check(
                over_cap == 0,
                f"{over_cap} items exceed 2x top_n total neighbors, which "
                f"the symmetric top-N filter should prevent",
            )
        finally:
            con.close()

    def report(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            path = str(outputs["edges"])
            total, n_items, avg_cc, med_cc = con.execute(f"""
                SELECT count(*), count(DISTINCT item_id_a) + count(DISTINCT item_id_b),
                       avg(co_count), quantile_cont(co_count, 0.5)
                FROM read_parquet('{path}')
            """).fetchone()
            lines = [
                "# P9 report",
                "",
                f"Total edges: {total:,}",
                f"co_count: mean={avg_cc:.1f}, median={med_cc:.0f}",
                f"SAMPLE_K={SAMPLE_K}  MIN_SUPPORT={MIN_SUPPORT}  TOP_N={TOP_N}",
            ]
            outputs["report"].parent.mkdir(parents=True, exist_ok=True)
            outputs["report"].write_text("\n".join(lines) + "\n")
            log.info("\n".join(lines))
        finally:
            con.close()

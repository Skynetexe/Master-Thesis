"""Stage P5: the evaluation harness and the popularity baseline.

The first thesis numbers. The baseline scores every item by its train
listen count and ranks the full catalog for every eval user. Later
models import pipeline.evaluation and reuse the same rules.
"""

import numpy as np

from . import evaluation as ev
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
    with_thread_ladder,
)
from .config import FROZEN, MODEL_DIR


class StageP5:
    name = "p5_eval_baseline"
    upstream = ["p4_features"]

    def outputs(self):
        return {
            "metrics": MODEL_DIR / "eval_baseline.json",
            "ranks": MODEL_DIR / "eval_baseline_ranks.npz",
            "report": MODEL_DIR / "reports" / "p5_eval_baseline.md",
        }

    def current_inputs(self, files):
        try:
            return {"marker:p4_features": file_sha256(marker_path("p4_features"))}
        except FileNotFoundError:
            return {"marker:p4_features": None}

    def run(self, params, files):
        outputs = self.outputs()

        def work(con):
            n_slots, id_min = ev.load_id_slots()
            user, decile, test_last, test_first, dropped = ev.load_eval_frame(con)
            train_items, n_pairs = ev.load_train_items(
                con, f"SELECT user_id FROM read_parquet('{ev.EVAL_USERS}')"
            )
            scores = ev.popularity_scores(con, n_slots, id_min)
            ranks = ev.global_ranks(scores)
            rank_last = ev.user_ranks(ranks, test_last, user, train_items)
            rank_first = ev.user_ranks(ranks, test_first, user, train_items)
            log.info("p5: %d users ranked, %d train pairs scanned", len(user), n_pairs)
            self._result = (
                n_slots,
                id_min,
                user,
                decile,
                test_last,
                test_first,
                dropped,
                rank_last,
                rank_first,
                n_pairs,
            )

        with_thread_ladder(work, params["ladder"])
        (
            n_slots,
            id_min,
            user,
            decile,
            test_last,
            test_first,
            dropped,
            rank_last,
            rank_first,
            n_pairs,
        ) = self._result

        con = connect(params["ladder"][0])
        try:
            pop, pairs, tier = ev.covariate_arrays(con, n_slots)
        finally:
            con.close()

        decile_labels = np.array([f"d{int(d):02d}" for d in decile])

        def variant_block(test_items, rk):
            tp = pairs[test_items]
            cold = np.where(
                tp <= 0,
                "0",
                np.where(tp <= 5, "01-05", np.where(tp <= 99, "06-99", "100+")),
            )
            pc = pop[test_items]
            band = np.where(
                pc <= 10,
                "01-10",
                np.where(pc <= 50, "11-50", np.where(pc <= 90, "51-90", "91-100")),
            )
            tier_labels = np.array([ev.TIER_NAMES[int(c)] for c in tier[test_items]])
            return {
                "overall": ev.metric_block(rk),
                "slices": {
                    "tier": ev.slice_blocks(rk, tier_labels),
                    "pop_band": ev.slice_blocks(rk, band),
                    "cold_test_item": ev.slice_blocks(rk, cold),
                    "user_decile": ev.slice_blocks(rk, decile_labels),
                },
            }

        doc = {
            "model": "popularity: train listens per item",
            "candidate_rule": "full catalog minus the user's train items, "
            "held-out picks stay candidates",
            "tie_rule": "score desc, then item_id asc",
            "n_eval_users": int(len(user)),
            "dropped_no_test_pick": int(dropped),
            "n_train_pairs_scanned": int(n_pairs),
            "variants": {
                "last_ts": variant_block(test_last, rank_last),
                "first_ts": variant_block(test_first, rank_first),
            },
        }
        atomic_json_write(doc, outputs["metrics"])
        np.savez_compressed(
            outputs["ranks"], user_id=user, rank_last=rank_last, rank_first=rank_first
        )
        return {
            "metrics": {
                "path": str(outputs["metrics"]),
                "rows": 0,
                "sha256": file_sha256(outputs["metrics"]),
            },
            "ranks": {
                "path": str(outputs["ranks"]),
                "rows": int(len(user)),
                "sha256": ev.npz_hash(outputs["ranks"]),
            },
        }

    def verify(self, params):
        con = connect(4)
        outputs = self.outputs()
        try:
            doc = load_json(outputs["metrics"])
            check(doc is not None, "eval_baseline.json is missing")
            data = np.load(outputs["ranks"])
            user = np.asarray(data["user_id"])
            rl = np.asarray(data["rank_last"])
            rf = np.asarray(data["rank_first"])
            n = len(user)
            check(
                n == doc["n_eval_users"] and n > 0,
                "the json and the npz disagree on the user count",
            )
            check(
                doc["dropped_no_test_pick"] <= 25,
                f"{doc['dropped_no_test_pick']} users lack a pick",
            )
            check(
                int(rl.min()) >= 1
                and int(rl.max()) <= FROZEN["items"]
                and int(rf.min()) >= 1
                and int(rf.max()) <= FROZEN["items"],
                "a rank falls outside the candidate range",
            )
            for variant in ("last_ts", "first_ts"):
                ov = doc["variants"][variant]["overall"]
                check(
                    0.0 <= ov["recall_at_10"] <= ov["recall_at_100"] <= 1.0,
                    f"{variant}: the recall ordering is wrong",
                )
                check(
                    ov["ndcg_at_100"] <= ov["recall_at_100"] + 1e-12,
                    f"{variant}: ndcg@100 exceeds recall@100",
                )
                n_slice = sum(
                    sb["n_users"]
                    for sb in doc["variants"][variant]["slices"]["user_decile"].values()
                )
                check(n_slice == n, f"{variant}: slice counts do not sum")

            n_slots, id_min = ev.load_id_slots()
            scores = ev.popularity_scores(con, n_slots, id_min)
            con.execute("CREATE OR REPLACE TABLE check_users (user_id BIGINT)")
            con.executemany(
                "INSERT INTO check_users VALUES (?)", [(int(u),) for u in user[:50]]
            )
            picks = con.execute(f"""
                WITH p AS (SELECT user_id,
                                  max(CASE WHEN is_test_last
                                      THEN item_id END) AS test_last,
                                  max(CASE WHEN is_test_first
                                      THEN item_id END) AS test_first
                           FROM read_parquet('{ev.SPLITS}')
                           GROUP BY user_id)
                SELECT p.test_last, p.test_first
                FROM check_users c JOIN p ON c.user_id = p.user_id
                ORDER BY c.user_id
            """).fetchall()
            train_items, _ = ev.load_train_items(con, "SELECT user_id FROM check_users")
            for k in range(50):
                tr = train_items.get(int(user[k]))
                bl = ev.brute_rank(scores, picks[k][0], tr)
                bf = ev.brute_rank(scores, picks[k][1], tr)
                check(
                    bl == int(rl[k]) and bf == int(rf[k]),
                    f"user {int(user[k])}: rank mismatch, global "
                    f"{int(rl[k])}/{int(rf[k])}, brute {bl}/{bf}",
                )
            log.info("p5: brute-force check passed on 50 users")
        finally:
            con.close()

    def report(self, params):
        doc = load_json(self.outputs()["metrics"]) or {}
        lines = ["# P5 popularity baseline", ""]
        add = lines.append
        add(f"- model: {doc.get('model')}")
        add(
            f"- eval users: {doc.get('n_eval_users')} "
            f"(dropped {doc.get('dropped_no_test_pick')} without a pick)"
        )
        add(
            f"- train pairs scanned for masking: " f"{doc.get('n_train_pairs_scanned')}"
        )
        add(f"- candidate rule: {doc.get('candidate_rule')}")
        add(f"- tie rule: {doc.get('tie_rule')}")
        add("")
        for variant in ("last_ts", "first_ts"):
            block = (doc.get("variants") or {}).get(variant) or {}
            ov = block.get("overall") or {}
            add(f"## Variant {variant}")
            add("")
            add(f"- users: {ov.get('n_users')}")
            add(f"- Recall@10: {ov.get('recall_at_10', 0):.4f}")
            add(f"- Recall@100: {ov.get('recall_at_100', 0):.4f}")
            add(f"- NDCG@100: {ov.get('ndcg_at_100', 0):.4f}")
            add(f"- MRR: {ov.get('mrr', 0):.4f}")
            add(
                f"- median rank: {ov.get('median_rank', 0):.0f}, "
                f"mean rank: {ov.get('mean_rank', 0):.0f}"
            )
            add("")
            slices = block.get("slices") or {}
            for slice_name in ("tier", "pop_band", "cold_test_item", "user_decile"):
                add(f"### {variant} by {slice_name}")
                add("")
                add("| value | users | R@10 | R@100 | NDCG@100 |")
                add("|---|---|---|---|---|")
                for value, sb in sorted((slices.get(slice_name) or {}).items()):
                    add(
                        f"| {value} | {sb['n_users']} | "
                        f"{sb['recall_at_10']:.4f} | "
                        f"{sb['recall_at_100']:.4f} | "
                        f"{sb['ndcg_at_100']:.4f} |"
                    )
                add("")
        atomic_text_write("\n".join(lines) + "\n", self.outputs()["report"])
        log.info("p5 report written")

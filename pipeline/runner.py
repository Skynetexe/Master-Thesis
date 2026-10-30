"""Pipeline runner.

Start P1 alone first:  python -m pipeline.runner --only p1_audit
Then run everything:   python -m pipeline.runner

Marker keys:
- "params"     holds the scoped digest: only the keys the stage reads.
- "source_sha" holds the hashes of the stage module, common.py, and
                config.py. A code edit invalidates its own marker.
A stage re-runs when its scope changes, its source changes, its inputs
change, or --force is set. Execution-only keys (ladder,
double_check_split) are out of every scope: determinism across worker
counts is proven, and the double-check is a verification pass, not an
output.

The scripts after P13b part 2 are stages too (SCRIPT_STAGES, built on
script_stage.py): ALS phase D, ItemKNN, the GAM diagnostic and refit,
P13c, P13d, P14, the cold-start curve, the beyond-accuracy measures and
the evidence-removal test. Two rules protect their outputs:
- --adopt gives a marker to outputs that exist, without a run, when the
  freshness rule and the read-back pass. Nothing runs, and no output
  changes.
- The guard: such a stage that finds its outputs without a current marker
  stops (exit 2) and names --adopt and --force. --force runs it again.
P1 to P13b2 have no guard: a stale marker makes them run again.
"""

import argparse
import hashlib
import inspect
import logging
import sys
import time
from pathlib import Path

from .common import (
    LockHeld,
    ResourceExhausted,
    VerificationError,
    atomic_json_write,
    file_sha256,
    load_json,
    log,
    marker_path,
    params_digest,
    pipeline_lock,
    read_marker,
    write_marker,
)
from .config import DEFAULT_PARAMS, INTERACTIONS_DIR, MODEL_DIR
from .stage_p1 import StageP1
from .stage_p2 import StageP2
from .stage_p25 import StageP25
from .stage_p3 import StageP3
from .stage_p4 import StageP4
from .stage_p5 import StageP5
from .stage_p6 import StageP6
from .stage_p7 import StageP7
from .stage_p8 import StageP8
from .stage_p9 import StageP9
from .stage_p9b import StageP9b
from .stage_p10 import StageP10
from .stage_p11 import StageP11
from .stage_p12 import StageP12
from .stage_p13a import StageP13a
from .stage_p13b import StageP13b
from .stage_p11b import StageP11b
from .stage_p12b import StageP12b
from .stage_p13b2 import StageP13b2

from .als_tune_d import StageALSD
from .beyond import StageBeyond
from .cold_curve import StageColdCurve
from .evidence_removal import StageEvidenceRemoval
from .gam_knn_diag import StageGamKnnSlices
from .gam_refit import StageGamRefit
from .itemknn_select import StageItemKNN
from .p13c import StageP13c
from .p13d import StageP13d
from .p14 import StageP14
from .p14_cold import StageP14Cold

# The scripts after P13b part 2, in order, then P14, the cold-start
# curve, the beyond-accuracy measures of the recommended lists (beyond.py)
# and the evidence-removal test of the explanations (evidence_removal.py).
SCRIPT_STAGES = [
    StageALSD(),
    StageItemKNN(),
    StageGamKnnSlices(),
    StageGamRefit(),
    StageP13c(),
    StageP13d(),
    StageP13c(model="p13d"),
    StageP14Cold(),
    StageP14(),
    StageColdCurve(),
    StageBeyond(),
    StageEvidenceRemoval(),
]
STAGES = [
    StageP1(),
    StageP2(),
    StageP25(),
    StageP3(),
    StageP4(),
    StageP5(),
    StageP6(),
    StageP7(),
    StageP8(),
    StageP9(),
    StageP9b(),
    StageP10(),
    StageP11(),
    StageP12(),
    StageP13a(),
    StageP13b(),
    StageP11b(),
    StageP12b(),
    StageP13b2(),
] + SCRIPT_STAGES
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}

# The params.json keys each stage actually reads.
PARAMS_SCOPE = {
    "p1_audit": ("item_pair_floors", "user_pair_floors"),
    "p2_split": ("eligibility_min_train_pairs",),
    "p25_train": (),
    "p3_sampling": (
        "core_item_floor",
        "eval_user_sample",
        "neg_proposal_power",
        "seed",
    ),
    "p4_features": ("tag_min_df",),
    "p5_eval_baseline": (),
    "p6_als": (),
}


def setup_logging():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fmt.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    filelog = logging.FileHandler(MODEL_DIR / "logs" / "pipeline.log")
    filelog.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(filelog)


def load_params() -> dict:
    path = MODEL_DIR / "params.json"
    if not path.exists():
        atomic_json_write(DEFAULT_PARAMS, path)
        log.info("params.json created with defaults")
        return dict(DEFAULT_PARAMS)
    return load_json(path)


def stage_params_digest(stage, params):
    scope = PARAMS_SCOPE.get(stage.name, ())
    scoped = {key: params[key] for key in scope}
    return params_digest(scoped)


def stage_source_sha(stage):
    """Hash of the code that computes the output: stage module, common,
    config. A code edit invalidates its own marker."""
    stage_file = Path(inspect.getfile(type(stage)))
    digest = hashlib.sha256()
    for path in (
        stage_file,
        stage_file.parent / "common.py",
        stage_file.parent / "config.py",
    ):
        digest.update(file_sha256(path).encode("utf-8"))
    return digest.hexdigest()


def _stage_current(stage, scoped, source_sha, files):
    marker = read_marker(stage.name)
    if not marker or marker.get("params") != scoped:
        return None
    if marker.get("source_sha") != source_sha:
        return None
    if marker.get("inputs") != stage.current_inputs(files):
        return None
    for path in stage.outputs().values():
        if not Path(path).exists():
            return None
    return marker


def execute(stages, params, files, only=None, force=False, adopt=False, by_name=None):
    """Skip, adopt or run each stage in order (module docstring). Returns
    0, or 2 when a stage cannot go on. A failed read-back raises
    VerificationError."""
    by_name = STAGE_BY_NAME if by_name is None else by_name
    for stage in stages:
        if only and stage.name != only:
            continue
        scoped = stage_params_digest(stage, params)
        source_sha = stage_source_sha(stage)
        for dep in stage.upstream:
            dep_stage = by_name[dep]
            dep_marker = read_marker(dep)
            if (
                not dep_marker
                or dep_marker.get("params") != stage_params_digest(dep_stage, params)
                or dep_marker.get("source_sha") != stage_source_sha(dep_stage)
            ):
                log.error(
                    "stage %s needs a current marker for %s. " "Run that stage first.",
                    stage.name,
                    dep,
                )
                return 2
        if not force and _stage_current(stage, scoped, source_sha, files):
            log.info("skip %s (marker valid)", stage.name)
            continue
        guarded = getattr(stage, "guarded", False)
        if adopt:
            if not guarded:
                log.error(
                    "stage %s has no current marker, and --adopt is "
                    "only for the stages after P13b part 2. Run it: "
                    "python3 -m pipeline.runner --only %s",
                    stage.name,
                    stage.name,
                )
                return 2
            if not stage.outputs_exist():
                text = (
                    "nothing to adopt for %s: its outputs do not exist. "
                    "Run it: python3 -m pipeline.runner --only %s"
                )
                if only:
                    log.error(text, stage.name, stage.name)
                    return 2
                log.info(text, stage.name, stage.name)
                continue
            log.info(
                "adopt %s: freshness rule and read-back of its outputs, " "no run",
                stage.name,
            )
            stage.adopt(params)
            write_marker(
                stage.name,
                scoped,
                stage.current_inputs(files),
                stage.output_info(),
                source_sha=source_sha,
            )
            log.info("adopted %s", stage.name)
            continue
        if guarded and not force and stage.outputs_exist():
            log.error(
                "stage %s found its outputs without a current marker. "
                "To keep them: python3 -m pipeline.runner --only %s "
                "--adopt. To compute them again: python3 -m "
                "pipeline.runner --only %s --force",
                stage.name,
                stage.name,
                stage.name,
            )
            return 2
        try:
            marker_path(stage.name).unlink()
        except FileNotFoundError:
            pass
        log.info("run %s", stage.name)
        outputs = stage.run(params, files)
        stage.verify(params)
        stage.report(params)
        write_marker(
            stage.name,
            scoped,
            stage.current_inputs(files),
            outputs,
            source_sha=source_sha,
        )
        log.info("done %s", stage.name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stages of the pipeline")
    parser.add_argument(
        "--only", choices=[s.name for s in STAGES], help="run one stage"
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore markers and run again"
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="give a marker to the existing outputs of the "
        "stages after P13b part 2, without a run, when "
        "their freshness rule and read-back pass",
    )
    args = parser.parse_args()
    if args.adopt and args.force:
        parser.error("--adopt and --force exclude each other")
    setup_logging()
    log.info("pipeline start%s", " (adopt)" if args.adopt else "")
    files = sorted(str(p) for p in INTERACTIONS_DIR.glob("interactions_*.parquet"))
    if not files:
        log.error("no interaction files at %s", INTERACTIONS_DIR)
        return 2
    params = load_params() or dict(DEFAULT_PARAMS)
    log.info(
        "%d interaction files, full params digest %s", len(files), params_digest(params)
    )
    try:
        with pipeline_lock(MODEL_DIR / ".lock"):
            rc = execute(
                STAGES,
                params,
                files,
                only=args.only,
                force=args.force,
                adopt=args.adopt,
            )
            if rc:
                return rc
    except VerificationError as exc:
        log.error("VERIFICATION FAILED: %s", exc)
        log.error("stop and investigate. Do not adjust a number to make it fit.")
        return 2
    except ResourceExhausted as exc:
        log.error("resource exhausted: %s", exc)
        return 3
    except LockHeld as exc:
        log.error("%s", exc)
        return 4
    except KeyboardInterrupt:
        log.warning("interrupt. Partials stay on disk. The next run resumes.")
        return 130
    log.info("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

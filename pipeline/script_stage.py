"""Runner stages for the scripts after P13b part 2.

A ScriptStage runs a module's run() for pipeline.runner. Its marker inputs
hold the SHA-256 of the upstream markers, of the input files that the
module's provenance lists, and of every pipeline module whose code the
stage can run (as stage_p13b.current_inputs does). A change in a shared
module or in an input file then makes the marker stale.

Two rules protect outputs that exist:
- adopt: the runner gives a marker to outputs that exist, with no run,
  only when the inputs that the outputs recorded at their run equal the
  inputs now (stale_input) and the module's read-back passes (check).
- guard (guarded = True): a stage that finds its outputs without a
  current marker stops, so a plain runner call never rewrites them.
  --force runs it again.
"""

import contextlib
import inspect
import logging
import sys
import time
import types
from pathlib import Path

from . import config
from .common import VerificationError, file_sha256, marker_path

PACKAGE_DIR = Path(__file__).resolve().parent
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"  # as in runner.py


def code_files(module):
    """The pipeline modules whose code a module can run: the module itself
    and every pipeline module that one of them imports, as a module or
    through an imported name. Sorted by file name. runner.py is never
    one of them."""
    seen = {}
    stack = [module]
    while stack:
        m = stack.pop()
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.parent != PACKAGE_DIR or p.name in seen or p.name == "runner.py":
            continue
        seen[p.name] = p
        for v in list(vars(m).values()):
            if isinstance(v, types.ModuleType):
                stack.append(v)
                continue
            try:
                owner = sys.modules.get(getattr(v, "__module__", None) or "")
            except Exception:
                owner = None
            if owner is not None:
                stack.append(owner)
    return [seen[k] for k in sorted(seen)]


@contextlib.contextmanager
def log_file(path):
    """The lines of this block also go to path: the module's own log file,
    as its own entry writes it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    fmt = logging.Formatter(LOG_FORMAT)
    fmt.converter = time.gmtime
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


class ScriptStage:
    """The common part of the stages in this module's docstring. A stage
    defines name, upstream, outputs(), provenance(), recorded_inputs(),
    run() and check(). model_dir and scratch_dir default to config's, and
    the smoke test points them to a temporary folder."""

    name = None
    upstream = []
    guarded = True

    def __init__(self, model_dir=None, scratch_dir=None):
        self._model_dir = model_dir
        self._scratch_dir = scratch_dir

    @property
    def model_dir(self):
        return Path(self._model_dir or config.MODEL_DIR)

    @property
    def scratch_dir(self):
        return Path(self._scratch_dir or config.SCRATCH_DIR)

    # Per stage

    def outputs(self):
        raise NotImplementedError

    def provenance(self):
        """The input files as the module's provenance records them."""
        return {}

    def recorded_inputs(self):
        """The provenance that the outputs recorded at their run."""
        return None

    def run(self, params, files):
        raise NotImplementedError

    def check(self):
        """The module's read-back of the stored outputs. Raises
        ValueError."""
        raise NotImplementedError

    # Common part

    def code_files(self):
        return code_files(inspect.getmodule(type(self)))

    def current_inputs(self, files):
        out = {}
        for stage in self.upstream:
            try:
                out[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                out[f"marker:{stage}"] = None
        for key, value in self.provenance().items():
            out[f"input:{key}"] = value
        for p in self.code_files():
            out[f"code:{p.name}"] = file_sha256(p)
        return out

    def outputs_exist(self):
        return all(Path(p).exists() for p in self.outputs().values())

    def output_info(self):
        return {
            k: {"path": str(p), "sha256": file_sha256(p) if Path(p).is_file() else None}
            for k, p in sorted(self.outputs().items())
        }

    def stale_input(self):
        """None when the inputs that the outputs recorded at their run
        equal the inputs now, else the name of the first that differs."""
        rec = self.recorded_inputs()
        if rec is None:
            return "(the outputs record no inputs)"
        now = self.provenance()
        for key in sorted(set(rec) | set(now)):
            if rec.get(key) != now.get(key):
                return key
        return None

    def verify(self, params):
        try:
            self.check()
        except ValueError as exc:
            raise VerificationError(f"{self.name}: {exc}") from exc

    def adopt(self, params):
        """The freshness rule, then the read-back (module docstring)."""
        stale = self.stale_input()
        if stale is not None:
            raise VerificationError(
                f"{self.name}: its outputs cannot be adopted: the input "
                f"{stale} changed after they were made. Compute them again: "
                f"python3 -m pipeline.runner --only {self.name} --force"
            )
        self.verify(params)

    def report(self, params):
        """run() writes the report."""

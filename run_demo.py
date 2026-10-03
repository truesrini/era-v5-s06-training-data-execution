#!/usr/bin/env python
"""One command for the whole demonstration:

    python run_demo.py

documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches -> training
-> consumption ledger -> learning ledger -> checkpoint -> crash -> resume -> replay -> fork
-> audit -> performance -> evidence.  Regenerates submission_artifacts/ from scratch.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import tdes  # noqa: E402  (pins BLAS threads before numpy loads)
import numpy as np  # noqa: E402

from tdes.audit import audit  # noqa: E402
from tdes.build import build  # noqa: E402
from tdes.config import default_config, small_config  # noqa: E402
from tdes.evidence import build_evidence  # noqa: E402
from tdes.performance import build_performance  # noqa: E402
from tdes.trainer import CRASH_EXIT_CODE  # noqa: E402
from tdes.util import RunLog, rmtree_force, write_json  # noqa: E402


def child_env():
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONPATH": ROOT})
    return env


def trainer(art, log, *args):
    cmd = [sys.executable, "-m", "tdes.trainer", "--art", art, *[str(a) for a in args]]
    log.info("spawn: python -m tdes.trainer " + " ".join(str(a) for a in args))
    p = subprocess.run(cmd, cwd=ROOT, env=child_env())
    return p.returncode


def run_tests(art, log):
    log.section("automated invariant tests")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
                       cwd=ROOT, env=child_env(), capture_output=True, text=True, encoding="utf-8")
    out = p.stdout + p.stderr
    with open(os.path.join(art, "reports", "tests.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    ran = failures = errors = 0
    for line in out.splitlines():
        if line.startswith("Ran "):
            ran = int(line.split()[1])
        if line.startswith("FAILED"):
            for part in line[line.find("(") + 1:line.rfind(")")].split(","):
                k, _, v = part.strip().partition("=")
                if k == "failures":
                    failures = int(v)
                if k == "errors":
                    errors = int(v)
    res = {"ran": ran, "failures": failures, "errors": errors, "passed": p.returncode == 0 and ran > 0,
           "seconds": round(time.perf_counter() - t0, 2), "command": "python -m unittest discover -s tests -t . -v"}
    write_json(os.path.join(art, "reports", "tests_summary.json"), res)
    log.check("unit_tests_passed", res["passed"], f"{ran} tests in {res['seconds']}s (failures={failures}, errors={errors})")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--art", default=os.path.join(ROOT, "submission_artifacts"))
    ap.add_argument("--quick", action="store_true", help="tiny configuration (used by the tests)")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    art = os.path.abspath(a.art)
    t0 = time.perf_counter()
    rmtree_force(art)
    os.makedirs(art)
    log = RunLog(os.path.join(art, "run.log"), "demo")
    cfg = small_config() if a.quick else default_config()
    log.section("TDES: Training Data Execution System demo")
    import torch
    log.info(f"python {platform.python_version()} numpy {np.__version__} torch {torch.__version__} on {platform.system()} {platform.machine()}; "
             f"artifacts -> {art}")
    demo, T = cfg["demo"], cfg["train"]["total_steps"]

    build(art, cfg, log)

    log.section("uninterrupted reference run (expected stream)")
    rc = trainer(art, log, "--mode", "fresh", "--branch", "reference", "--until", T, "--checkpoint-every", 0)
    if rc != 0:
        raise SystemExit(f"reference run failed ({rc})")

    log.section("main run: train until the deliberate crash")
    rc = trainer(art, log, "--mode", "fresh", "--branch", "main", "--until", T,
                 "--crash-at", demo["crash_at_step"], "--crash-after", demo["crash_after_microbatches"])
    log.check("crash_was_real_process_death", rc == CRASH_EXIT_CODE,
              f"training process exited with code {rc} mid-step (expected {CRASH_EXIT_CODE})")

    log.section("resume from the latest checkpoint")
    rc = trainer(art, log, "--mode", "resume", "--branch", "main", "--until", T, "--reference-branch", "reference")
    if rc != 0:
        raise SystemExit(f"resume failed ({rc})")

    log.section("replay an earlier interval from the ledger")
    rc = trainer(art, log, "--mode", "replay", "--source-branch", "main",
                 "--from-step", demo["replay_from_step"], "--to-step", demo["replay_to_step"])
    if rc != 0:
        raise SystemExit(f"replay failed ({rc})")

    log.section("fork a new data branch from an earlier checkpoint")
    rc = trainer(art, log, "--mode", "fork", "--source-branch", "main", "--from-step", demo["fork_from_step"],
                 "--steps", demo["fork_steps"], "--overrides-json", json.dumps(demo["fork_overrides"]))
    if rc != 0:
        raise SystemExit(f"fork failed ({rc})")

    log.section("audit")
    rep = audit(art, log)
    log.section("performance")
    build_performance(art, rep, log)
    tests = None if a.skip_tests else run_tests(art, log)
    log.section("evidence bundle")
    ev = build_evidence(art, log, tests)
    log.info(f"demo finished in {time.perf_counter() - t0:.1f}s; overall {ev['overall']}")
    for r in ev["requirements"]:
        log.info(f"  {r['result']:<4}  {r['requirement']}")
    return 0 if ev["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

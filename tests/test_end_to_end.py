"""Runs the complete demonstration on the small configuration (real subprocess crash included)
and checks the properties the assignment grades: exact resume, exact replay, explicit fork,
clean firewall, and evidence that is derived from the artifacts."""
import os
import subprocess
import sys
import unittest

from tests.helpers import ROOT, tmpdir
from tdes.util import Ledger, read_json


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = os.path.join(tmpdir(), "art")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
        cls.proc = subprocess.run([sys.executable, "run_demo.py", "--quick", "--skip-tests", "--art", cls.art],
                                  cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")

    def test_demo_succeeds_and_evidence_passes(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout[-3000:] + self.proc.stderr[-3000:])
        ev = read_json(os.path.join(self.art, "evidence.json"))
        self.assertEqual(ev["overall"], "PASS", [r["id"] for r in ev["requirements"] if r["result"] != "PASS"])
        for name in ("run.log", "evidence.json", "evidence.md", "performance.json"):
            self.assertTrue(os.path.exists(os.path.join(self.art, name)))
        for d in ("manifests", "ledgers", "checkpoints"):
            self.assertTrue(os.listdir(os.path.join(self.art, d)))

    def test_resume_neither_skips_nor_repeats(self):
        recs = Ledger.read(os.path.join(self.art, "ledgers", "main", "consumption.jsonl"))
        eff = Ledger.effective(recs)
        steps = [r["global_step"] for r in eff if r["type"] == "step_committed"]
        cfg = read_json(os.path.join(self.art, "run_config.json"))
        self.assertEqual(steps, list(range(1, cfg["train"]["total_steps"] + 1)))
        self.assertTrue(any(r["type"] == "rollback" for r in recs), "the crash left uncommitted records")
        rep = read_json(os.path.join(self.art, "reports", "resume_report.json"))
        self.assertTrue(all(rep["checks"].values()), rep["checks"])
        self.assertEqual(rep["resumed_next_batch"]["batch_hash"], rep["expected_from_crashed_process"]["batch_hash"])
        ref = Ledger.effective(Ledger.read(os.path.join(self.art, "ledgers", "reference", "consumption.jsonl")))
        self.assertEqual([r["batch_hash"] for r in eff if r["type"] == "batch_planned"],
                         [r["batch_hash"] for r in ref if r["type"] == "batch_planned"])

    def test_replay_reproduces_ids_spans_hashes_and_weights(self):
        rep = read_json(os.path.join(self.art, "reports", "replay_report.json"))
        self.assertTrue(rep["all_match"])
        for r in rep["steps"]:
            self.assertEqual(r["orig_batch_hash"], r["replay_batch_hash"])
            self.assertEqual(r["orig_weights_hash"], r["replay_weights_hash"])

    def test_fork_is_explicit(self):
        rep = read_json(os.path.join(self.art, "reports", "fork_report.json"))
        first = Ledger.read(os.path.join(self.art, "ledgers", rep["branch"], "consumption.jsonl"))[0]
        self.assertEqual(first["type"], "branch_forked")
        self.assertTrue(rep["steps"][0]["differs"])

    def test_run_log_has_required_lines(self):
        with open(os.path.join(self.art, "run.log"), encoding="utf-8") as f:
            log = f.read()
        for line in ("[PASS] tokenizer_hash_verified", "[PASS] eval_shard_blocked", "[PASS] checkpoint_saved",
                     "[PASS] resume_next_batch_matched", "[PASS] replay_hash_matched"):
            self.assertIn(line, log)
        self.assertNotIn("[FAIL]", log)

    def test_evidence_fails_when_an_artifact_is_tampered(self):
        """Evidence is derived, not asserted: corrupting a ledger record must flip the audit."""
        import shutil
        from tdes.audit import audit
        from tdes.util import RunLog
        art = os.path.join(tmpdir(), "copy")
        shutil.copytree(self.art, art)
        p = os.path.join(art, "ledgers", "main", "consumption.jsonl")
        with open(p, encoding="utf-8") as f:
            lines = f.read().splitlines()
        i = next(i for i, l in enumerate(lines) if '"type":"microbatch_consumed"' in l)
        lines[i] = lines[i].replace('"rank":0', '"rank":1', 1)       # silently re-attribute a microbatch
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        rep = audit(art, RunLog(os.path.join(tmpdir(), "a.log"), "t", echo=False))
        chk = {c["name"]: c["passed"] for c in rep["checks"]}
        self.assertFalse(chk["ledger_hash_chains_valid"])


if __name__ == "__main__":
    unittest.main()

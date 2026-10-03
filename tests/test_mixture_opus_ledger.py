import copy
import os
import unittest

from tests.helpers import env, tmpdir
from tdes.config import LANES, PROTECTED_LANES, default_config, global_batch
from tdes.loader import DataLoader
from tdes.mixture import MixtureError, compile_schedule
from tdes.opus import threshold, triage
from tdes.util import Ledger

SUPPLY = {l: 10 ** 6 for l in LANES}


class MixtureTests(unittest.TestCase):
    def test_quotas_floors_and_shares(self):
        cfg = default_config()
        s = compile_schedule(cfg, SUPPLY)
        B = global_batch(cfg)
        for e in s["steps"]:
            self.assertEqual(sum(e["quotas"].values()), B)
            for lane, c in e["floor_counts"].items():
                self.assertGreaterEqual(e["quotas"][lane], c)
            if e["stage"] != "anneal":
                self.assertEqual(e["quotas"]["anneal_reserve"], 0)
            else:
                self.assertEqual(e["quotas"]["general_web"], 0, "web is cut immediately at the anneal boundary")
        for st in s["stages"]:
            self.assertLessEqual(st["max_quota_deviation"], 0.02)

    def test_invalid_plans_are_rejected(self):
        cfg = default_config()
        bad = copy.deepcopy(cfg)
        bad["mixture"]["stages"][0]["protected_floors"]["indic"] = 0.5      # floor above weight
        with self.assertRaises(MixtureError):
            compile_schedule(bad, SUPPLY)
        bad = copy.deepcopy(cfg)
        bad["mixture"]["stages"][0]["mixture"]["anneal_reserve"] = 0.05
        bad["mixture"]["stages"][0]["mixture"]["general_web"] -= 0.05
        with self.assertRaises(MixtureError):
            compile_schedule(bad, SUPPLY)
        with self.assertRaises(MixtureError):                                 # supply cannot cover the plan
            compile_schedule(cfg, {**SUPPLY, "agentic": 10})


class OpusTests(unittest.TestCase):
    def test_triage(self):
        cands = [{"candidate_id": f"c{i}", "sample_id": f"s{i}", "score": sc} for i, sc in enumerate([0.9, 0.5, 0.1, -0.2, 0.7])]
        cands.append({"candidate_id": "c9", "sample_id": "s0", "score": 0.95})
        acc, low, beyond, dups = triage(cands, 2, threshold([c["score"] for c in cands], 0.4), {"s4"}, set())
        self.assertEqual([c["candidate_id"] for c in acc], ["c9", "c1"])
        self.assertEqual([c["candidate_id"] for c in dups], ["c0", "c4"])
        self.assertEqual([c["candidate_id"] for c in low], ["c2", "c3"])
        self.assertEqual(beyond, [])

    def _loader(self, scorer_bias):
        e = env()
        sched = compile_schedule(e.cfg, SUPPLY)
        dl = DataLoader(e.cfg, sched, e.store, e.train_manifests, e.tokenizer)

        def scorer(seqs):
            return [(scorer_bias.get(s.lane, 0.5) + int(s.sample_id[3:9], 16) % 7 / 100.0, 1.0, s.n_loss) for s in seqs]
        return dl, sched, scorer

    def test_protected_floor_override_and_deferral(self):
        dl, sched, scorer = self._loader({"indic": -5.0})      # a proxy that undervalues Indic
        _, decs = dl.build_batch(1, scorer, {"branch": "t"})
        over = [d for d in decs if d["protected_floor_override"]]
        self.assertTrue(over and all(d["lane"] == "indic" and d["status"] == "accepted" for d in over))
        self.assertTrue(any(d["status"] == "deferred" for d in decs))
        self.assertTrue(any(d["status"] == "rejected" and d["reason"] == "low_proxy_utility" for d in decs))
        deferred = {d["candidate_id"] for d in decs if d["status"] == "deferred"}
        _, decs2 = dl.build_batch(2, scorer, {"branch": "t"})
        self.assertTrue(deferred & {d["candidate_id"] for d in decs2}, "deferred candidates are re-offered")
        e1 = sched["steps"][0]
        for lane in PROTECTED_LANES:
            got = sum(1 for d in decs if d["lane"] == lane and d["status"] == "accepted")
            self.assertEqual(got, e1["quotas"][lane])

    def test_stage_mismatch_rejection(self):
        dl, sched, scorer = self._loader({})
        e = sched["steps"][0]
        dl.deferred = [{"candidate_id": "cand-x", "lane": "general_web", "spans": dl.streams["general_web"].next_spans(),
                        "defer_count": 1, "first_step": 0, "last_score": 0.9}]
        anneal_step = next(s["step"] for s in sched["steps"] if s["quotas"]["general_web"] == 0)
        _, decs = dl.build_batch(anneal_step, scorer, {"branch": "t"})
        self.assertTrue(any(d["candidate_id"] == "cand-x" and d["reason"] == "stage_mismatch" for d in decs))
        self.assertTrue(e)

    def test_loader_state_roundtrip(self):
        dl, _, scorer = self._loader({"indic": -1.0})
        dl.build_batch(1, scorer, {"branch": "t"})
        st = dl.state_dict()
        b2, d2 = dl.build_batch(2, scorer, {"branch": "t"})
        dl2, _, _ = self._loader({"indic": -1.0})
        dl2.load_state_dict(st)
        b2b, d2b = dl2.build_batch(2, scorer, {"branch": "t"})
        self.assertEqual(b2.batch_hash, b2b.batch_hash)
        self.assertEqual([d["decision_id"] for d in d2], [d["decision_id"] for d in d2b])


class LedgerTests(unittest.TestCase):
    def test_hash_chain_detects_tampering(self):
        p = os.path.join(tmpdir(), "l.jsonl")
        lg = Ledger(p)
        for i in range(5):
            lg.append("e", {"i": i})
        self.assertEqual(Ledger.verify_chain(Ledger.read(p)), (True, None))
        recs = Ledger.read(p)
        recs[2]["i"] = 99
        self.assertEqual(Ledger.verify_chain(recs)[1], 2)
        recs = Ledger.read(p)
        del recs[1]
        self.assertFalse(Ledger.verify_chain(recs)[0])
        lg2 = Ledger(p)                       # reopening continues the chain
        lg2.append("e", {"i": 5})
        self.assertTrue(Ledger.verify_chain(Ledger.read(p))[0])

    def test_rollback_is_appended_not_deleted(self):
        p = os.path.join(tmpdir(), "l.jsonl")
        lg = Ledger(p)
        for i in range(4):
            lg.append("e", {"i": i})
        lg.append("rollback", {"rolled_back_seqs": [2, 3]})
        self.assertEqual(len(Ledger.read(p)), 5)
        self.assertEqual([r["i"] for r in Ledger.effective(Ledger.read(p))], [0, 1])


if __name__ == "__main__":
    unittest.main()

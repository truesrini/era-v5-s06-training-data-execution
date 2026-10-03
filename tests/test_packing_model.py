import copy
import unittest

import numpy as np

from tests.helpers import env
from tdes.config import LANE_POLICY
import torch

from tdes.model import Optimizer, TinyLM, eval_ce, loss_and_backward, to_tensors
from tdes.packing import LaneStream, build_sequence, lane_units, simulate_policy, verify_sequence


def _stream(e, lane, buffer=8):
    ms = {s: e.manifests[s] for s in e.catalog["admitted_train"] if e.manifests[s]["capability_lane"] == lane}
    nl = {i for i in range(e.tokenizer.vocab_size) if b"\n" in e.tokenizer.token_bytes(i)}
    units, _ = lane_units(lane, LANE_POLICY[lane], ms, e.store, e.cfg["train"]["seq_len"], nl)
    return LaneStream(lane, LANE_POLICY[lane], units, e.cfg["train"]["seq_len"], e.cfg["seed"], buffer)


class PackingTests(unittest.TestCase):
    def test_masks_positions_attention_for_every_policy(self):
        e = env()
        L = e.cfg["train"]["seq_len"]
        for lane in ("general_web", "code", "indic", "reasoning", "agentic", "anneal_reserve"):
            st = _stream(e, lane)
            for _ in range(12):
                seq = build_sequence(st.next_spans(), e.store, L, lane, LANE_POLICY[lane])
                self.assertEqual(verify_sequence(seq, e.store), [], (lane, seq.sample_id))
                pad = seq.segment_ids == 0
                self.assertTrue((seq.loss_mask[pad] == 0).all())
                for k in range(1, len(seq.spans) + 1):
                    idx = np.nonzero(seq.segment_ids == k)[0]
                    self.assertEqual(seq.position_ids[idx].tolist(), list(range(len(idx))))
                    self.assertEqual(seq.labels[idx[-1]], -1, "no target across a segment boundary")
                att = seq.attention_mask()
                for i in range(L):
                    allowed = np.nonzero(att[i])[0]
                    if seq.segment_ids[i] > 0:
                        self.assertTrue((seq.segment_ids[allowed] == seq.segment_ids[i]).all())
                        self.assertTrue((allowed <= i).all())

    def test_structured_samples_never_split_and_context_has_no_loss(self):
        e = env()
        L = e.cfg["train"]["seq_len"]
        st = _stream(e, "agentic")
        for _ in range(10):
            seq = build_sequence(st.next_spans(), e.store, L, "agentic", "structure_preserving")
            for k, sp in enumerate(seq.spans, start=1):
                doc = e.store.get(sp["shard_id"]).docs[sp["doc_index"]]
                self.assertEqual((sp["start"], sp["end"]), (0, doc["length"]))
                idx = np.nonzero(seq.segment_ids == k)[0]
                targets_context = seq.train_on[idx[1:]] == 0
                self.assertTrue((seq.loss_mask[idx[:-1]][targets_context] == 0).all())
            self.assertLess(seq.n_loss, seq.n_tokens, "user/tool_obs tokens are context only")

    def test_concat_chop_windows_continue_the_stream_exactly(self):
        e = env()
        st = _stream(e, "general_web")
        prev = st.next_spans()
        for _ in range(10):
            cur = st.next_spans()
            last, first = prev[-1], cur[0]
            doc_len = e.store.get(last["shard_id"]).docs[last["doc_index"]]["length"]
            if last["end"] < doc_len:
                self.assertEqual((first["shard_id"], first["doc_index"], first["start"]),
                                 (last["shard_id"], last["doc_index"], last["end"]))
            else:
                self.assertEqual(first["start"], 0)
            self.assertEqual(sum(s["end"] - s["start"] for s in cur), e.cfg["train"]["seq_len"])
            prev = cur

    def test_stream_state_roundtrip_reproduces_sequences(self):
        e = env()
        for lane in ("general_web", "code", "agentic"):
            a = _stream(e, lane)
            for _ in range(7):
                a.next_spans()
            saved = a.state_dict()
            want = [a.next_spans() for _ in range(9)]
            b = _stream(e, lane)
            b.load_state_dict(saved)
            self.assertEqual([b.next_spans() for _ in range(9)], want, lane)

    def test_packing_lab_ordering(self):
        lengths = [30, 90, 20, 60, 100, 10, 40, 70]
        res = {p: simulate_policy(lengths, p, 128, structured=True) for p in
               ("pad_only", "greedy_first_fit", "best_fit_decreasing")}
        self.assertLessEqual(res["pad_only"]["utilization"], res["greedy_first_fit"]["utilization"])
        self.assertLessEqual(res["greedy_first_fit"]["sequences"], res["pad_only"]["sequences"])
        self.assertFalse(simulate_policy(lengths, "concat_chop", 128, structured=True)["structure_safe"])


class ModelTests(unittest.TestCase):
    def _batch(self, rng, B=2, L=10):
        seg = np.array([[1, 1, 1, 2, 2, 2, 2, 0, 0, 0], [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]])
        pos = np.zeros_like(seg)
        for b in range(B):
            for s in set(seg[b].tolist()) - {0}:
                idx = np.nonzero(seg[b] == s)[0]
                pos[b, idx] = np.arange(len(idx))
        tok = rng.integers(0, 20, (B, L))
        lab = rng.integers(0, 20, (B, L))
        lm = ((rng.random((B, L)) > 0.3) & (seg > 0)).astype(np.float32)
        return tok, pos, seg, lab, lm

    def test_autograd_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        m = TinyLM(20, 8, 12, 10, 0, 0.3, "float64")
        arr = self._batch(rng)
        m.zero_grad()
        loss_and_backward(m, arr, 0.1)
        t = to_tensors(arr)

        def f():
            with torch.no_grad():
                return float((m.token_ce(*t) * t[4]).sum() * 0.1)
        worst = 0.0
        for name, p in m.named_parameters():
            flat = p.data.view(-1)
            for _ in range(4):
                i = int(rng.integers(0, flat.numel()))
                old = float(flat[i])
                flat[i] = old + 1e-6
                a = f()
                flat[i] = old - 1e-6
                b = f()
                flat[i] = old
                num, ana = (a - b) / 2e-6, float(p.grad.view(-1)[i])
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-9))
        self.assertLess(worst, 1e-4)

    def test_attention_is_isolated_per_segment(self):
        rng = np.random.default_rng(1)
        m = TinyLM(20, 8, 12, 10, 0, 0.3, "float64")
        tok, pos, seg, lab, lm = self._batch(rng)
        full = (seg > 0).astype(np.float32)
        ce1 = eval_ce(m, (tok, pos, seg, lab, full))
        tok2 = tok.copy()
        tok2[1, :6] = (tok2[1, :6] + 7) % 20          # change only segment 1 of row 1
        ce2 = eval_ce(m, (tok2, pos, seg, lab, full))
        np.testing.assert_allclose(ce1[1, 6:], ce2[1, 6:], rtol=0, atol=1e-12)

    def test_training_is_bit_reproducible_and_optimizer_state_roundtrips(self):
        from tdes.config import small_config
        cfg = small_config()
        arr = self._batch(np.random.default_rng(2))

        def run(m, opt, steps):
            for s in steps:
                opt.zero_grad()
                loss_and_backward(m, arr, 0.05)
                opt.step(s)
        m1 = TinyLM(20, 8, 12, 10, 0, 0.3)
        o1 = Optimizer(m1, cfg)
        run(m1, o1, [1, 2])
        m2 = TinyLM(20, 8, 12, 10, 0, 0.3)
        o2 = Optimizer(m2, cfg)
        m2.load_state_dict(m1.state_dict())
        o2.opt.load_state_dict(copy.deepcopy(o1.opt.state_dict()))   # as torch.load would give
        self.assertEqual(o1.state_hash(), o2.state_hash())
        run(m1, o1, [3])
        run(m2, o2, [3])
        self.assertEqual(m1.weights_hash(), m2.weights_hash())

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA build of torch and a GPU")
    def test_cuda_training_is_bit_reproducible(self):
        from tdes.config import small_config
        cfg = small_config()
        arr = self._batch(np.random.default_rng(3))
        hashes = []
        for _ in range(2):
            m = TinyLM(20, 8, 12, 10, 0, 0.3).to("cuda")
            opt = Optimizer(m, cfg)
            for s in (1, 2, 3):
                opt.zero_grad()
                loss_and_backward(m, arr, 0.05)
                opt.step(s)
            hashes.append(m.weights_hash())
        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()

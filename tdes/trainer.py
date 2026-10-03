"""Training loop bound to the data ledgers: fresh run, crash, resume, replay and fork.

Ledgers per branch (ledgers/<branch>/):
  consumption.jsonl  batch_planned -> microbatch_consumed x (ranks*accum) -> step_committed
                     -> checkpoint_saved; plus rollback / branch_forked / replay_started
  opus.jsonl         one record per OPUS candidate decision
  learning.jsonl     per-sample loss before/after, per-step grad norm, validation evals
  token_trace/       per-token cross-entropy for every loss-bearing token of every step

A checkpoint stores model + optimizer + scheduler + loader state AND the offset/hash of every
ledger, so model state and data state travel together.

Run as a module:  python -m tdes.trainer --art submission_artifacts --branch main --mode fresh ...
"""
import argparse
import os
import sys
import time

import numpy as np

from .config import ATTENTION_POLICY, LANE_POLICY, LANES, POSITION_POLICY, apply_overrides, global_batch
from .firewall import EvalRegistry, Firewall
from .loader import LOADER_VERSION, Batch, DataLoader
from .mixture import compile_schedule
from .model import Adam, PARAM_NAMES, flat_grad, forward_backward, init_params
from .opus import cosine
from .packing import build_sequence, lane_units, verify_sequence
from .shards import ShardStore, load_manifests
from .tokenizer import Tokenizer
from .util import (Ledger, RunLog, Stopwatch, array_hash, params_hash, read_json, rel, sha256_json,
                   write_json)

CRASH_EXIT_CODE = 86


class Env:
    """Everything a training process needs, loaded from disk and verified."""

    def __init__(self, art):
        self.art = art
        self.cfg = read_json(os.path.join(art, "run_config.json"))
        man = os.path.join(art, "manifests")
        self.lock = read_json(os.path.join(man, "tokenizer.lock.json"))
        self.tokenizer = Tokenizer.load(os.path.join(man, "tokenizer.json"), expected_hash=self.lock["tokenizer_hash"])
        self.manifests = load_manifests(man)
        self.catalog = read_json(os.path.join(man, "catalog.json"))
        bad = [s for s in self.catalog["admitted_train"] if self.manifests[s]["tokenizer_hash"] != self.tokenizer.hash]
        if bad:
            raise RuntimeError(f"shards tokenized with a different tokenizer: {bad}")
        self.registry = EvalRegistry.load(man)
        self.store = ShardStore(art, self.manifests, self.tokenizer.hash, capacity=12)
        self.firewall = Firewall(self.registry, os.path.join(art, "ledgers", "firewall.jsonl"),
                                 self.catalog["admitted_train"])
        self.train_manifests = {s: self.manifests[s] for s in self.catalog["admitted_train"]}
        self.run_id = "v5s06-" + sha256_json(self.cfg)[:10]


def _seqs_to_arrays(seqs):
    return (np.stack([s.tokens for s in seqs]), np.stack([s.position_ids for s in seqs]),
            np.stack([s.segment_ids for s in seqs]), np.stack([s.labels for s in seqs]),
            np.stack([s.loss_mask for s in seqs]))


class Trainer:
    def __init__(self, env, branch, log, cfg=None, schedule=None, checkpoint_every=None):
        self.env, self.branch, self.log = env, branch, log
        self.cfg = cfg or env.cfg
        self.art = env.art
        self.schedule = schedule or read_json(os.path.join(self.art, "manifests", "mixture_schedule.json"))
        t, m = self.cfg["train"], self.cfg["model"]
        self.B, self.L = global_batch(self.cfg), t["seq_len"]
        self.W, self.mb, self.accum = t["world_size"], t["micro_batch"], t["grad_accum"]
        self.ckpt_every = t["checkpoint_every"] if checkpoint_every is None else checkpoint_every
        self.params = init_params(env.tokenizer.vocab_size, m["d_model"], m["d_ff"], self.L,
                                  self.cfg["seed"], m["init_std"], m["dtype"])
        self.opt = Adam(self.params, self.cfg)
        self.loader = DataLoader(self.cfg, self.schedule, env.store, env.train_manifests, env.tokenizer)
        ldir = os.path.join(self.art, "ledgers", branch)
        self.ldir = ldir
        self.cons = Ledger(os.path.join(ldir, "consumption.jsonl"))
        self.opus = Ledger(os.path.join(ldir, "opus.jsonl"))
        self.learn = Ledger(os.path.join(ldir, "learning.jsonl"))
        self.trace_dir = os.path.join(ldir, "token_trace")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.ckpt_root = os.path.join(self.art, "checkpoints", branch)
        self.step = 0
        self.last_ckpt_id = None
        self.counters = {"positions": 0, "nonpad_tokens": 0, "loss_tokens": 0}
        self.sw = Stopwatch()
        self.perf = {"steps": 0, "positions": 0, "nonpad_tokens": 0, "loss_tokens": 0,
                     "candidates_scored": 0, "candidate_tokens": 0, "accepted_tokens": 0,
                     "rejected_tokens": 0, "decisions": {}, "rejections_by_lane": {}, "candidates_by_lane": {},
                     "packing_checks": 0, "packing_failures": 0, "firewall_sequences_checked": 0}
        self.mode = "fresh"
        self.perf_tag = "fresh"
        self.t_start = time.perf_counter()
        self._proxy = self._build_proxy()

    # ------------------------------------------------------------------ OPUS proxy
    def _build_proxy(self):
        ms = {s: m for s, m in self.env.manifests.items() if m["split"] == "proxy"}
        units, _ = lane_units("opus_proxy", "concat_chop", ms, self.env.store, self.L, set())
        from .packing import LaneStream
        st = LaneStream("opus_proxy", "concat_chop", units, self.L, self.cfg["seed"], 1)
        seqs = [build_sequence(st.next_spans(), self.env.store, self.L, "opus_proxy", "concat_chop")
                for _ in range(self.cfg["opus"]["proxy_sequences"])]
        self.env.firewall.log_proxy_read(sorted(ms), self.branch, step=None)
        return {"seqs": seqs, "shard_ids": sorted(ms),
                "version": "proxy-v1-" + sha256_json([s.seq_hash for s in seqs])[:10]}

    def _proxy_grad(self):
        arr = _seqs_to_arrays(self._proxy["seqs"])
        n = float(arr[4].sum())
        _, _, g = forward_backward(self.params, *arr, loss_scale=1.0 / n)
        return flat_grad(g)

    def _make_scorer(self, gp):
        def scorer(seqs):
            out = []
            with self.sw.time("opus_scoring"):
                for s in seqs:
                    n = max(1, s.n_loss)
                    arr = _seqs_to_arrays([s])
                    tot, _ce, g = forward_backward(self.params, *arr, loss_scale=1.0 / n)
                    out.append((cosine(flat_grad(g), gp), tot, s.n_loss))
                    self.perf["candidates_scored"] += 1
                    self.perf["candidate_tokens"] += s.n_tokens
            return out
        return scorer

    # ------------------------------------------------------------------ batches
    def build_batch(self, step):
        with self.sw.time("loader_total"):
            gp = None
            if self.cfg["opus"]["enabled"]:
                with self.sw.time("opus_scoring"):
                    gp = self._proxy_grad()
            ctx = {"branch": self.branch, "scoring_checkpoint": self.last_ckpt_id,
                   "scoring_model_hash": params_hash(self.params)[:16], "proxy_version": self._proxy["version"]}
            batch, decisions = self.loader.build_batch(step, self._make_scorer(gp), ctx)
        return batch, decisions

    def batch_from_refs(self, step, stage, refs, decision_ids):
        seqs = [build_sequence(r["spans"], self.env.store, self.L, r["lane"], r["policy"]) for r in refs]
        return Batch(step, stage, seqs, decision_ids, None)

    # ------------------------------------------------------------------ one optimizer step
    def train_step(self, batch, decisions=None, crash_after=None, extra=None, on_planned=None):
        step = batch.step
        extra = extra or {}
        with self.sw.time("firewall"):
            for s in batch.seqs:
                self.env.firewall.check_sequence(s, self.env.store)
                self.perf["firewall_sequences_checked"] += 1
        with self.sw.time("packing_verify"):
            fails = {}
            for s in batch.seqs:
                bad = verify_sequence(s, self.env.store)
                self.perf["packing_checks"] += 1
                if bad:
                    fails[s.sample_id] = bad
            if fails:
                self.perf["packing_failures"] += len(fails)
                raise RuntimeError(f"packing invariants violated at step {step}: {fails}")
        with self.sw.time("ledger_io"):
            if decisions:
                for d in decisions:
                    self.opus.append("opus_decision", d)
                    self.perf["decisions"][d["status"]] = self.perf["decisions"].get(d["status"], 0) + 1
                    key = d["lane"]
                    self.perf["candidates_by_lane"][key] = self.perf["candidates_by_lane"].get(key, 0) + 1
                    if d["status"] == "rejected":
                        self.perf["rejections_by_lane"][key] = self.perf["rejections_by_lane"].get(key, 0) + 1
                        self.perf["rejected_tokens"] += d["n_loss_tokens"]
            planned = self.cons.append("batch_planned", {
                "run_id": self.env.run_id, "branch": self.branch, "mode": self.mode, "global_step": step,
                "stage": batch.stage, "batch_id": batch.batch_id, "batch_hash": batch.batch_hash,
                "sample_ids": [s.sample_id for s in batch.seqs], "lanes": [s.lane for s in batch.seqs],
                "quotas": batch.quotas, "opus_decision_ids": batch.decision_ids,
                "checkpoint_id": self.last_ckpt_id, "tokenizer_hash": self.env.tokenizer.hash,
                "tokenizer_version": self.env.tokenizer.version, "dataloader_version": LOADER_VERSION,
                "schedule_hash": self.schedule["schedule_hash"], **extra})
        if on_planned:
            on_planned(batch, planned)

        arr_all = _seqs_to_arrays(batch.seqs)
        n_loss_total = float(arr_all[4].sum())
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        ce_before = np.zeros((self.B, self.L), dtype=np.float32)
        loss_sum = 0.0
        consumed = 0
        for mb in range(self.accum):
            for r in range(self.W):
                gidx = [g for g in range(self.B) if g % self.W == r][mb * self.mb:(mb + 1) * self.mb]
                seqs = [batch.seqs[g] for g in gidx]
                arr = _seqs_to_arrays(seqs)
                with self.sw.time("train_compute"):
                    tot, ce, g = forward_backward(self.params, *arr, loss_scale=1.0 / n_loss_total)
                    for k in PARAM_NAMES:
                        grads[k] += g[k]
                loss_sum += tot
                ce_before[gidx] = ce
                with self.sw.time("ledger_io"):
                    self.cons.append("microbatch_consumed", {
                        "run_id": self.env.run_id, "branch": self.branch, "mode": self.mode, "global_step": step,
                        "batch_id": batch.batch_id, "checkpoint_id": self.last_ckpt_id, "rank": r,
                        "microbatch_id": f"{batch.batch_id}/r{r}/m{mb}", "microbatch_index": mb,
                        "global_indices": gidx,
                        "samples": [{"sample_id": s.sample_id, "lane": s.lane, "policy": s.policy,
                                     "spans": s.spans, "seq_hash": s.seq_hash,
                                     "loss_mask_hash": s.loss_mask_hash(), "n_tokens": s.n_tokens,
                                     "n_loss_tokens": s.n_loss, "opus_decision_id": batch.decision_ids[g]}
                                    for g, s in zip(gidx, seqs)],
                        "shard_ids": sorted({sp["shard_id"] for s in seqs for sp in s.spans}),
                        "token_span_ids": [sp["span_id"] for s in seqs for sp in s.spans],
                        "loss_mask_hash": array_hash(arr[4]), "microbatch_hash": array_hash(*arr),
                        "attention_policy": ATTENTION_POLICY, "position_policy": POSITION_POLICY,
                        "stage": batch.stage, "tokenizer_version": self.env.tokenizer.version,
                        "dataloader_version": LOADER_VERSION,
                        "positions": int(arr[0].size), "nonpad_tokens": int((arr[2] > 0).sum()),
                        "loss_tokens": int(arr[4].sum())})
                consumed += 1
                if crash_after is not None and consumed >= crash_after:
                    self.log.event("crash simulated", f"process killed inside step {step} after {consumed} of "
                                   f"{self.accum * self.W} microbatches (exit code {CRASH_EXIT_CODE}); "
                                   f"last checkpoint {self.last_ckpt_id}")
                    sys.stdout.flush()
                    os._exit(CRASH_EXIT_CODE)
        with self.sw.time("train_compute"):
            gnorm, lr = self.opt.step(self.params, grads, step)
            _, ce_after, _ = forward_backward(self.params, *arr_all, loss_scale=1.0, need_grad=False)
        whash = params_hash(self.params)
        positions = self.B * self.L
        nonpad = int((arr_all[2] > 0).sum())
        nloss = int(n_loss_total)
        self.counters["positions"] += positions
        self.counters["nonpad_tokens"] += nonpad
        self.counters["loss_tokens"] += nloss
        for k, v in (("positions", positions), ("nonpad_tokens", nonpad), ("loss_tokens", nloss),
                     ("accepted_tokens", nonpad)):
            self.perf[k] += v
        self.perf["steps"] += 1
        with self.sw.time("ledger_io"):
            self._learning(batch, ce_before, ce_after, arr_all, gnorm, lr, step)
            self.cons.append("step_committed", {
                "run_id": self.env.run_id, "branch": self.branch, "mode": self.mode, "global_step": step,
                "batch_id": batch.batch_id, "batch_hash": batch.batch_hash, "loss": loss_sum,
                "grad_norm": gnorm, "lr": lr, "weights_hash": whash,
                "cum_positions": self.counters["positions"], "cum_nonpad_tokens": self.counters["nonpad_tokens"],
                "cum_loss_tokens": self.counters["loss_tokens"], "microbatches": consumed})
        self.step = step
        dec = {}
        if decisions:
            for d in decisions:
                k = "override" if d["protected_floor_override"] else d["status"]
                dec[k] = dec.get(k, 0) + 1
        lanes = {}
        for s in batch.seqs:
            lanes[s.lane] = lanes.get(s.lane, 0) + 1
        self.log.event("batches packed", f"[{self.branch}] step {step:>3} {batch.stage:<18} {batch.batch_id} "
                       f"util={nonpad / positions:.3f} loss_tok={nloss} loss={loss_sum:.4f} gnorm={gnorm:.3f} "
                       f"lanes={lanes}" + (f" opus={dec}" if dec else ""))
        if decisions:
            self.log.event("OPUS decisions recorded", f"[{self.branch}] step {step}: {len(decisions)} candidate records")
        return {"loss": loss_sum, "grad_norm": gnorm, "weights_hash": whash, "batch": batch}

    # ------------------------------------------------------------------ learning ledger
    def _learning(self, batch, ce_b, ce_a, arr, gnorm, lr, step):
        mask = arr[4] > 0
        seq_i, pos_i = np.nonzero(mask)
        seg = arr[2][seq_i, pos_i]
        np.savez_compressed(
            os.path.join(self.trace_dir, f"step_{step:05d}.npz"),
            seq_index=seq_i.astype(np.int16), position=pos_i.astype(np.int16),
            target_id=arr[3][seq_i, pos_i].astype(np.int32), segment=seg.astype(np.int16),
            loss_before=ce_b[seq_i, pos_i].astype(np.float32), loss_after=ce_a[seq_i, pos_i].astype(np.float32),
            sample_ids=np.array([s.sample_id for s in batch.seqs]),
            span_ids=np.array(["|".join(sp["span_id"] for sp in s.spans) for s in batch.seqs]))
        trace_rel = rel(os.path.join(self.trace_dir, f"step_{step:05d}.npz"), self.art)
        stage = batch.stage
        for gi, s in enumerate(batch.seqs):
            m = s.loss_mask > 0
            n = int(m.sum())
            lb = float(ce_b[gi][m].mean()) if n else 0.0
            la = float(ce_a[gi][m].mean()) if n else 0.0
            per_span = []
            for k, sp in enumerate(s.spans, start=1):
                mk = m & (s.segment_ids == k)
                if mk.any():
                    per_span.append({"span_id": sp["span_id"], "shard_id": sp["shard_id"], "doc_id": sp["doc_id"],
                                     "repeated_pass": sp["epoch"] + 1, "n_loss_tokens": int(mk.sum()),
                                     "loss_before": round(float(ce_b[gi][mk].mean()), 6),
                                     "loss_after": round(float(ce_a[gi][mk].mean()), 6)})
            top = np.argsort(-ce_b[gi] * m)[:3]
            top = [{"position": int(p), "token_id": int(s.labels[p]), "loss": round(float(ce_b[gi][p]), 4),
                    "ppl": round(float(np.exp(ce_b[gi][p])), 2),
                    "preview": self.env.tokenizer.decode([int(s.labels[p])]),
                    "is_eos": int(s.labels[p]) == 1} for p in top if m[p]]
            self.learn.append("sample_learning", {
                "branch": self.branch, "global_step": step, "stage": stage, "model_phase": stage,
                "batch_id": batch.batch_id, "sample_id": s.sample_id, "lane": s.lane,
                "opus_decision_id": batch.decision_ids[gi] if batch.decision_ids else None,
                "n_loss_tokens": n, "loss_before": round(lb, 6), "loss_after": round(la, 6),
                "loss_delta": round(la - lb, 6), "ppl_before": round(float(np.exp(lb)), 4),
                "spans": per_span, "top_surprise": top, "grad_norm_step": round(gnorm, 6),
                "tokens_seen_before": self.counters["positions"] - self.B * self.L,
                "checkpoint_before": self.last_ckpt_id, "token_trace": trace_rel, "trace_seq_index": gi})
        self.learn.append("step_learning", {"branch": self.branch, "global_step": step, "stage": stage,
                                            "grad_norm": gnorm, "lr": lr,
                                            "loss_before": float((ce_b * (arr[4] > 0)).sum() / max(1, (arr[4] > 0).sum())),
                                            "loss_after": float((ce_a * (arr[4] > 0)).sum() / max(1, (arr[4] > 0).sum()))})

    def validation_eval(self, step):
        """Eval-only read of validation shards: forward pass, no gradient, access logged."""
        val = {s: m for s, m in self.env.manifests.items() if m["split"] == "validation"}
        res = {}
        for lane in LANES:
            ms = {s: m for s, m in val.items() if m["capability_lane"] == lane}
            if not ms:
                continue
            units, _ = lane_units(lane, LANE_POLICY[lane], ms, self.env.store, self.L, set())
            from .packing import LaneStream
            st = LaneStream(lane, LANE_POLICY[lane], units, self.L, self.cfg["seed"], 4)
            seqs = [build_sequence(st.next_spans(), self.env.store, self.L, lane, LANE_POLICY[lane]) for _ in range(2)]
            arr = _seqs_to_arrays(seqs)
            n = float(arr[4].sum())
            tot, _, g = forward_backward(self.params, *arr, loss_scale=1.0 / max(n, 1), need_grad=False)
            assert g is None
            res[lane] = round(tot, 5)
        self.env.firewall.log_validation_read(sorted(val), self.branch, step)
        self.learn.append("validation_eval", {"branch": self.branch, "global_step": step, "loss_by_lane": res,
                                              "gradient": False})
        return res

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self):
        t0 = time.perf_counter()
        whash = params_hash(self.params)
        ckpt_id = f"ckpt-{self.branch}-s{self.step:05d}-{whash[:10]}"
        val = self.validation_eval(self.step)
        self.cons.append("checkpoint_saved", {"run_id": self.env.run_id, "branch": self.branch,
                                              "global_step": self.step, "checkpoint_id": ckpt_id,
                                              "weights_hash": whash, "validation_loss": val})
        state = {
            "checkpoint_id": ckpt_id, "run_id": self.env.run_id, "branch": self.branch, "global_step": self.step,
            "weights_hash": whash, "optimizer_hash": array_hash(*[v for _, v in sorted(self.opt.state_arrays().items())]),
            "optimizer_t": self.opt.t, "scheduler": {"step": self.step, "next_lr": float(self.opt.lr_at(self.step + 1))},
            "rng_state": {"seed": self.cfg["seed"], "note": "all data randomness is derived from (seed, lane, epoch); "
                                                            "model init from seed; no other RNG is consumed"},
            "loader_state": self.loader.state_dict(), "dataloader_version": LOADER_VERSION,
            "counters": dict(self.counters), "previous_checkpoint": self.last_ckpt_id,
            "ledger_offsets": {name: {"offset": lg.count, "last_hash": lg.last_hash, "path": rel(lg.path, self.art)}
                               for name, lg in (("consumption", self.cons), ("opus", self.opus), ("learning", self.learn))},
            "tokenizer_hash": self.env.tokenizer.hash, "schedule_hash": self.schedule["schedule_hash"],
            "config_hash": sha256_json(self.cfg), "mode": self.mode,
        }
        final = os.path.join(self.ckpt_root, f"step_{self.step:05d}")
        tmp = final + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        np.savez(os.path.join(tmp, "model.npz"), **self.params)
        np.savez(os.path.join(tmp, "optimizer.npz"), **self.opt.state_arrays())
        write_json(os.path.join(tmp, "state.json"), state)
        with open(os.path.join(tmp, "COMPLETE"), "w") as f:
            f.write(ckpt_id + "\n")
        os.replace(tmp, final)
        self.last_ckpt_id = ckpt_id
        # verify by reloading
        with np.load(os.path.join(final, "model.npz")) as z:
            ok = params_hash({k: z[k] for k in z.files}) == whash
        self.sw.add("checkpoint", time.perf_counter() - t0)
        write_perf(self, self.perf_tag, {"wall_s": round(time.perf_counter() - self.t_start, 6),
                                         "accounted_through_step": self.step})
        self.log.event("checkpoint saved", f"[{self.branch}] {ckpt_id} at step {self.step}; consumption ledger offset "
                       f"{state['ledger_offsets']['consumption']['offset']} hash {state['ledger_offsets']['consumption']['last_hash'][:12]}")
        self.log.check("checkpoint_saved", ok, f"{rel(final, self.art)} reloads to weights_hash {whash[:12]}; "
                       f"bound to ledger offset {state['ledger_offsets']['consumption']['offset']}")
        return state

    def load_checkpoint(self, path):
        state = read_json(os.path.join(path, "state.json"))
        with np.load(os.path.join(path, "model.npz")) as z:
            params = {k: z[k] for k in z.files}
        if params_hash(params) != state["weights_hash"]:
            raise RuntimeError(f"checkpoint {path} weights do not match its state.json")
        with np.load(os.path.join(path, "optimizer.npz")) as z:
            arrs = {k: z[k] for k in z.files}
        self.params = params
        self.opt.load_arrays(arrs, state["optimizer_t"])
        self.loader.load_state_dict(state["loader_state"])
        self.step = state["global_step"]
        self.counters = dict(state["counters"])
        self.last_ckpt_id = state["checkpoint_id"]
        return state

    # ------------------------------------------------------------------ loops
    def run(self, until, crash_at=None, crash_after=None, on_planned=None):
        while self.step < until:
            step = self.step + 1
            batch, decisions = self.build_batch(step)
            res = self.train_step(batch, decisions, crash_after=crash_after if step == crash_at else None,
                                  on_planned=on_planned)
            if (self.ckpt_every and step % self.ckpt_every == 0) or step == until:
                self.save_checkpoint()
        return self


def latest_checkpoint(root):
    if not os.path.isdir(root):
        return None
    done = [d for d in sorted(os.listdir(root)) if d.startswith("step_") and not d.endswith(".tmp")
            and os.path.exists(os.path.join(root, d, "COMPLETE"))]
    return os.path.join(root, done[-1]) if done else None


def rollback_ledgers(tr, state, reason):
    """Supersede (never delete) every ledger record written after the checkpoint's offset."""
    out = {}
    for name, lg in (("consumption", tr.cons), ("opus", tr.opus), ("learning", tr.learn)):
        off = state["ledger_offsets"][name]
        recs = Ledger.read(lg.path)
        ok, bad = Ledger.verify_chain(recs)
        if not ok:
            raise RuntimeError(f"{name} ledger hash chain broken at seq {bad}")
        if off["offset"] > 0 and recs[off["offset"] - 1]["event_hash"] != off["last_hash"]:
            raise RuntimeError(f"{name} ledger does not match checkpoint offset")
        already = set()
        for r in recs:
            if r["type"] == "rollback":
                already.update(r["rolled_back_seqs"])
        rolled = [r["seq"] for r in recs[off["offset"]:] if r["type"] != "rollback" and r["seq"] not in already]
        lg.append("rollback", {"branch": tr.branch, "resume_from_checkpoint": state["checkpoint_id"],
                               "ledger_offset": off["offset"], "rolled_back_seqs": rolled, "reason": reason})
        out[name] = {"offset": off["offset"], "rolled_back": len(rolled),
                     "rolled_back_records": [recs[s] for s in rolled]}
    return out


def write_perf(tr, tag, extra=None):
    d = os.path.join(tr.art, "perf")
    os.makedirs(d, exist_ok=True)
    out = {"branch": tr.branch, "mode": tr.mode, "tag": tag, "counts": tr.perf,
           "timings_s": {k: round(v, 6) for k, v in tr.sw.totals.items()},
           "store": tr.env.store.stats(), "firewall_ngrams_checked": tr.env.firewall.ngrams_checked}
    out.update(extra or {})
    write_json(os.path.join(d, f"{tr.branch}__{tag}.json"), out)
    return out


# ============================================================================ entry points
def mode_fresh(art, branch, until, crash_at=None, crash_after=None, checkpoint_every=None):
    env = Env(art)
    log = RunLog(os.path.join(art, "run.log"), f"train:{branch}")
    tr = Trainer(env, branch, log, checkpoint_every=checkpoint_every)
    tr.mode = tr.perf_tag = "fresh"
    log.info(f"[{branch}] fresh run {env.run_id}: B={tr.B} seqs x L={tr.L}, ranks={tr.W}, micro_batch={tr.mb}, "
             f"grad_accum={tr.accum}, steps 1..{until}, loader {LOADER_VERSION}")
    tr.run(until, crash_at=crash_at, crash_after=crash_after)
    return tr


def mode_resume(art, branch, until, reference_branch=None):
    t_start = time.perf_counter()
    env = Env(art)
    log = RunLog(os.path.join(art, "run.log"), f"train:{branch}")
    tr = Trainer(env, branch, log)
    tr.mode = tr.perf_tag = "resume"
    tr.t_start = t_start
    ck = latest_checkpoint(tr.ckpt_root)
    if ck is None:
        raise RuntimeError("no checkpoint to resume from")
    state = tr.load_checkpoint(ck)
    rb = rollback_ledgers(tr, state, "crash_recovery")
    log.event("run resumed", f"[{branch}] restored {state['checkpoint_id']} (step {state['global_step']}); consumption "
              f"ledger offset {state['ledger_offsets']['consumption']['offset']}; superseded "
              f"{rb['consumption']['rolled_back']} uncommitted consumption records, {rb['opus']['rolled_back']} OPUS records")
    crashed_planned = [r for r in rb["consumption"]["rolled_back_records"] if r["type"] == "batch_planned"]
    crashed_mbs = [r for r in rb["consumption"]["rolled_back_records"] if r["type"] == "microbatch_consumed"]
    ref_planned = {}
    if reference_branch:
        for r in Ledger.effective(Ledger.read(os.path.join(art, "ledgers", reference_branch, "consumption.jsonl"))):
            if r["type"] == "batch_planned":
                ref_planned[r["global_step"]] = r
    report = {"branch": branch, "resumed_from": state["checkpoint_id"], "checkpoint_step": state["global_step"],
              "ledger_offset": state["ledger_offsets"]["consumption"],
              "rolled_back": {k: v["rolled_back"] for k, v in rb.items()}}
    first = {}

    def on_planned(batch, planned):
        if first:
            return
        first["latency_s"] = time.perf_counter() - t_start
        step = batch.step
        exp_crash = next((r for r in crashed_planned if r["global_step"] == step), None)
        exp_ref = ref_planned.get(step)
        resumed = {"global_step": step, "batch_id": batch.batch_id, "batch_hash": batch.batch_hash,
                   "sample_ids": [s.sample_id for s in batch.seqs]}
        first.update(resumed)
        report["expected_next_step"] = state["global_step"] + 1
        report["resumed_next_batch"] = resumed
        report["expected_from_crashed_process"] = None if exp_crash is None else {
            k: exp_crash[k] for k in ("global_step", "batch_id", "batch_hash", "sample_ids", "seq")}
        report["expected_from_uninterrupted_reference"] = None if exp_ref is None else {
            k: exp_ref[k] for k in ("global_step", "batch_id", "batch_hash", "sample_ids", "seq")}
        same_step = step == state["global_step"] + 1
        m_crash = exp_crash is not None and exp_crash["batch_hash"] == batch.batch_hash and exp_crash["batch_id"] == batch.batch_id
        m_ref = exp_ref is not None and exp_ref["batch_hash"] == batch.batch_hash and exp_ref["sample_ids"] == resumed["sample_ids"]
        report["checks"] = {"next_step_is_checkpoint_plus_one": same_step,
                            "matches_batch_planned_by_crashed_process": m_crash,
                            "matches_uninterrupted_reference_run": m_ref}
        log.check("resume_next_batch_matched", same_step and m_crash and m_ref,
                  f"step {step}: resumed {batch.batch_id} == crashed-process plan "
                  f"{exp_crash['batch_id'] if exp_crash else None} == reference {exp_ref['batch_id'] if exp_ref else None}")

    t0 = time.perf_counter()
    step = tr.step + 1
    batch, decisions = tr.build_batch(step)
    # microbatches re-consumed after the crash must equal the ones the crashed process consumed
    res = tr.train_step(batch, decisions, on_planned=on_planned)
    new_mbs = [r for r in Ledger.effective(Ledger.read(tr.cons.path))
               if r["type"] == "microbatch_consumed" and r["global_step"] == step]
    mb_cmp = []
    for old in crashed_mbs:
        new = next((n for n in new_mbs if n["microbatch_id"] == old["microbatch_id"]), None)
        mb_cmp.append({"microbatch_id": old["microbatch_id"], "crashed_hash": old["microbatch_hash"],
                       "resumed_hash": new["microbatch_hash"] if new else None,
                       "match": bool(new and new["microbatch_hash"] == old["microbatch_hash"])})
    report["rolled_back_microbatches_reconsumed_identically"] = mb_cmp
    if (tr.ckpt_every and step % tr.ckpt_every == 0) or step == until:
        tr.save_checkpoint()
    tr.run(until)
    report["resume_latency_s"] = round(first.get("latency_s", 0.0), 6)
    report["final_step"] = tr.step
    report["final_weights_hash"] = params_hash(tr.params)
    write_json(os.path.join(art, "reports", "resume_report.json"), report)
    write_perf(tr, "resume", {"wall_s": round(time.perf_counter() - t_start, 6), "accounted_through_step": tr.step,
                              "resume_latency_s": report["resume_latency_s"]})
    return report


def mode_replay(art, source_branch, from_step, to_step):
    """Restore an older checkpoint and re-feed the historical stream recorded in the source
    ledger (no sampling, no OPUS): rebuild every packed sample from its span refs, prove ids,
    spans and hashes match, retrain and prove the weights match bit for bit."""
    t_start = time.perf_counter()
    env = Env(art)
    branch = f"replay-{source_branch}-s{from_step:05d}"
    log = RunLog(os.path.join(art, "run.log"), f"replay:{branch}")
    tr = Trainer(env, branch, log, checkpoint_every=env.cfg["train"]["checkpoint_every"])
    tr.mode = tr.perf_tag = "replay"
    tr.t_start = t_start
    src_ck = os.path.join(art, "checkpoints", source_branch, f"step_{from_step:05d}")
    state = tr.load_checkpoint(src_ck)
    tr.ckpt_root = os.path.join(art, "checkpoints", branch)
    src_recs = Ledger.effective(Ledger.read(os.path.join(art, "ledgers", source_branch, "consumption.jsonl")))
    tr.cons.append("replay_started", {"source_branch": source_branch, "source_checkpoint": state["checkpoint_id"],
                                      "from_step": from_step + 1, "to_step": to_step,
                                      "source_ledger_offset": state["ledger_offsets"]["consumption"]})
    latency = None
    steps = []
    all_ok = True
    for step in range(from_step + 1, to_step + 1):
        planned = next(r for r in src_recs if r["type"] == "batch_planned" and r["global_step"] == step)
        mbs = [r for r in src_recs if r["type"] == "microbatch_consumed" and r["global_step"] == step]
        committed = next(r for r in src_recs if r["type"] == "step_committed" and r["global_step"] == step)
        refs = [None] * len(planned["sample_ids"])
        orig = [None] * len(planned["sample_ids"])
        for mb in mbs:
            for g, smp in zip(mb["global_indices"], mb["samples"]):
                refs[g] = smp
                orig[g] = smp
        batch = tr.batch_from_refs(step, planned["stage"], refs, planned["opus_decision_ids"])
        batch.quotas = planned["quotas"]
        if latency is None:
            latency = time.perf_counter() - t_start
        sample_cmp = [{"sample_id": s.sample_id, "orig_sample_id": o["sample_id"],
                       "span_ids_match": [sp["span_id"] for sp in s.spans] == [sp["span_id"] for sp in o["spans"]],
                       "seq_hash_match": s.seq_hash == o["seq_hash"],
                       "loss_mask_hash_match": s.loss_mask_hash() == o["loss_mask_hash"]}
                      for s, o in zip(batch.seqs, orig)]
        res = tr.train_step(batch, None, extra={"source_branch": source_branch, "source_seq": planned["seq"]})
        row = {"global_step": step, "orig_batch_id": planned["batch_id"], "replay_batch_id": batch.batch_id,
               "orig_batch_hash": planned["batch_hash"], "replay_batch_hash": batch.batch_hash,
               "batch_ids_match": planned["batch_id"] == batch.batch_id,
               "batch_hash_match": planned["batch_hash"] == batch.batch_hash,
               "sample_ids_match": all(c["sample_id"] == c["orig_sample_id"] for c in sample_cmp),
               "token_spans_match": all(c["span_ids_match"] for c in sample_cmp),
               "seq_hashes_match": all(c["seq_hash_match"] for c in sample_cmp),
               "loss_mask_hashes_match": all(c["loss_mask_hash_match"] for c in sample_cmp),
               "orig_loss": committed["loss"], "replay_loss": res["loss"],
               "orig_weights_hash": committed["weights_hash"], "replay_weights_hash": res["weights_hash"],
               "weights_match": committed["weights_hash"] == res["weights_hash"],
               "n_samples": len(sample_cmp), "n_spans": sum(len(s.spans) for s in batch.seqs)}
        ok = all(row[k] for k in ("batch_ids_match", "batch_hash_match", "sample_ids_match", "token_spans_match",
                                  "seq_hashes_match", "loss_mask_hashes_match", "weights_match"))
        all_ok &= ok
        steps.append(row)
        log.info(f"replay step {step}: {batch.batch_id} vs original {planned['batch_id']} -> "
                 f"{'identical' if ok else 'MISMATCH'} (weights {res['weights_hash'][:12]})")
        if tr.ckpt_every and step % tr.ckpt_every == 0:
            tr.save_checkpoint()
    log.event("historical stream replayed", f"steps {from_step + 1}-{to_step} of '{source_branch}' rebuilt from ledger "
              f"span refs and retrained from {state['checkpoint_id']}")
    log.check("replay_hash_matched", all_ok and all(r["batch_hash_match"] for r in steps),
              f"{len(steps)} batches: batch ids, token spans, sequence and loss-mask hashes identical")
    log.check("replay_weights_matched", all(r["weights_match"] for r in steps),
              f"weights after step {to_step} {steps[-1]['replay_weights_hash'][:12]} == original "
              f"{steps[-1]['orig_weights_hash'][:12]}")
    report = {"branch": branch, "source_branch": source_branch, "source_checkpoint": state["checkpoint_id"],
              "interval": [from_step + 1, to_step], "steps": steps, "all_match": all_ok,
              "replay_latency_s": round(latency or 0.0, 6)}
    write_json(os.path.join(art, "reports", "replay_report.json"), report)
    write_perf(tr, "replay", {"wall_s": round(time.perf_counter() - t_start, 6), "accounted_through_step": tr.step,
                              "replay_latency_s": report["replay_latency_s"]})
    return report


def mode_fork(art, parent_branch, from_step, n_steps, overrides):
    env = Env(art)
    parent_ck = os.path.join(art, "checkpoints", parent_branch, f"step_{from_step:05d}")
    pstate = read_json(os.path.join(parent_ck, "state.json"))
    cfg = apply_overrides(env.cfg, overrides)
    branch = "fork-" + sha256_json([pstate["checkpoint_id"], overrides])[:8]
    log = RunLog(os.path.join(art, "run.log"), f"fork:{branch}")
    supply = {}
    for sid in env.catalog["admitted_train"]:
        lane = env.manifests[sid]["capability_lane"]
        supply[lane] = supply.get(lane, 0) + env.manifests[sid]["token_count"]
    sched = compile_schedule(cfg, supply)
    write_json(os.path.join(art, "manifests", f"mixture_schedule.{branch}.json"), sched)
    tr = Trainer(env, branch, log, cfg=cfg, schedule=sched)
    tr.mode = tr.perf_tag = "fork"
    state = tr.load_checkpoint(parent_ck)
    tr.ckpt_root = os.path.join(art, "checkpoints", branch)
    tr.cons.append("branch_forked", {"parent_branch": parent_branch, "parent_checkpoint": state["checkpoint_id"],
                                     "parent_ledger_offset": state["ledger_offsets"]["consumption"],
                                     "divergence_step": from_step + 1, "config_overrides": overrides,
                                     "parent_config_hash": sha256_json(env.cfg), "fork_config_hash": sha256_json(cfg),
                                     "parent_schedule_hash": state["schedule_hash"], "fork_schedule_hash": sched["schedule_hash"]})
    log.event("branch forked", f"{branch} from {state['checkpoint_id']} (ledger offset "
              f"{state['ledger_offsets']['consumption']['offset']}); overrides {sorted(overrides)}")
    t0 = time.perf_counter()
    tr.run(from_step + n_steps)
    parent = {r["global_step"]: r for r in Ledger.effective(Ledger.read(
        os.path.join(art, "ledgers", parent_branch, "consumption.jsonl"))) if r["type"] == "batch_planned"}
    mine = {r["global_step"]: r for r in Ledger.read(tr.cons.path) if r["type"] == "batch_planned"}
    first = from_step + 1
    rows = [{"global_step": s, "parent_batch_id": parent.get(s, {}).get("batch_id"), "fork_batch_id": mine[s]["batch_id"],
             "differs": parent.get(s, {}).get("batch_hash") != mine[s]["batch_hash"]} for s in sorted(mine)]
    report = {"branch": branch, "parent_branch": parent_branch, "parent_checkpoint": state["checkpoint_id"],
              "parent_weights_hash": state["weights_hash"], "divergence_step": first, "overrides": overrides,
              "fork_schedule_hash": sched["schedule_hash"], "parent_schedule_hash": state["schedule_hash"],
              "steps": rows, "final_step": tr.step, "final_weights_hash": params_hash(tr.params)}
    fork_rec = Ledger.read(tr.cons.path)[0]
    ok = rows and rows[0]["differs"] and min(mine) == first
    report["checks"] = {"first_fork_batch_differs_from_parent": bool(rows and rows[0]["differs"]),
                        "fork_starts_at_divergence_step": min(mine) == first,
                        "fork_ledger_starts_with_branch_forked": fork_rec["type"] == "branch_forked"
                        and fork_rec["parent_checkpoint"] == state["checkpoint_id"]}
    ok = ok and report["checks"]["fork_ledger_starts_with_branch_forked"]
    log.check("fork_diverged_explicitly", bool(ok), f"step {first}: fork {rows[0]['fork_batch_id']} vs parent "
              f"{rows[0]['parent_batch_id']}; divergence recorded in ledgers/{branch}/consumption.jsonl")
    write_json(os.path.join(art, "reports", "fork_report.json"), report)
    write_perf(tr, "fork", {"wall_s": round(time.perf_counter() - t0, 6), "accounted_through_step": tr.step})
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--mode", choices=["fresh", "resume", "replay", "fork"], required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--until", type=int)
    ap.add_argument("--crash-at", type=int)
    ap.add_argument("--crash-after", type=int)
    ap.add_argument("--checkpoint-every", type=int)
    ap.add_argument("--reference-branch")
    ap.add_argument("--source-branch")
    ap.add_argument("--from-step", type=int)
    ap.add_argument("--to-step", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--overrides-json")
    a = ap.parse_args(argv)
    if a.mode == "fresh":
        mode_fresh(a.art, a.branch, a.until, a.crash_at, a.crash_after, a.checkpoint_every)
    elif a.mode == "resume":
        mode_resume(a.art, a.branch, a.until, a.reference_branch)
    elif a.mode == "replay":
        mode_replay(a.art, a.source_branch, a.from_step, a.to_step)
    else:
        import json
        mode_fork(a.art, a.source_branch, a.from_step, a.steps, json.loads(a.overrides_json))


if __name__ == "__main__":
    main()

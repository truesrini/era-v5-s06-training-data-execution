"""The data loader: lane streams + compiled mixture quotas + OPUS -> one global batch per step.

All of its state (stream cursors, deferred queue, duplicate window, candidate counter) is a
small JSON object saved in every checkpoint, so the stream after a resume is exactly the stream
an uninterrupted run would have produced.
"""
import inspect
import json
import math

from . import mixture as _mixture
from . import opus as _opus
from . import packing as _packing
from .config import LANE_POLICY, LANES, PROTECTED_LANES
from .opus import effective_tokens, threshold, triage
from .packing import LaneStream, batch_hash, build_sequence, lane_units
from .util import sha256_json

LOADER_VERSION = "tdes-loader/1.0+" + sha256_json([inspect.getsource(m) for m in (_packing, _opus, _mixture)])[:10] + \
    "." + sha256_json(open(__file__, encoding="utf-8").read())[:6]


class Batch:
    def __init__(self, step, stage, seqs, decision_ids, quotas):
        self.step, self.stage, self.seqs, self.decision_ids, self.quotas = step, stage, seqs, decision_ids, quotas
        self.batch_hash = batch_hash(seqs)
        self.batch_id = f"b{step:05d}-{self.batch_hash[:12]}"


class DataLoader:
    def __init__(self, cfg, schedule, store, train_manifests, tokenizer):
        self.cfg, self.schedule, self.store = cfg, schedule, store
        self.L = cfg["train"]["seq_len"]
        self.ocfg = cfg["opus"]
        newline_ids = {i for i in range(tokenizer.vocab_size) if b"\n" in tokenizer.token_bytes(i)}
        self.streams, self.dropped = {}, {}
        for lane in LANES:
            ms = {sid: m for sid, m in train_manifests.items() if m["capability_lane"] == lane}
            if not ms:
                continue
            units, dropped = lane_units(lane, LANE_POLICY[lane], ms, store, self.L, newline_ids)
            self.dropped[lane] = dropped
            self.streams[lane] = LaneStream(lane, LANE_POLICY[lane], units, self.L,
                                            cfg["seed"], cfg["train"]["packing_buffer"])
        self.deferred = []
        self.recent = []
        self.cand_counter = 0

    # ------------------------------------------------------------------ state
    def state_dict(self):
        """A detached, JSON-serialisable snapshot (later batches never mutate it)."""
        return json.loads(json.dumps({"streams": {l: s.state_dict() for l, s in self.streams.items()},
                                      "deferred": self.deferred, "recent": self.recent,
                                      "cand_counter": self.cand_counter, "loader_version": LOADER_VERSION}))

    def load_state_dict(self, st):
        for l, s in st["streams"].items():
            self.streams[l].load_state_dict(s)
        self.deferred = [dict(d) for d in st["deferred"]]
        self.recent = [list(r) for r in st["recent"]]
        self.cand_counter = st["cand_counter"]

    # ------------------------------------------------------------------ candidates
    def _draw(self, lane, step):
        spans = self.streams[lane].next_spans()
        seq = build_sequence(spans, self.store, self.L, lane, LANE_POLICY[lane])
        c = {"candidate_id": f"cand-{self.cand_counter:06d}", "lane": lane, "seq": seq,
             "sample_id": seq.sample_id, "defer_count": 0, "first_step": step}
        self.cand_counter += 1
        return c

    def _rebuild(self, d):
        seq = build_sequence(d["spans"], self.store, self.L, d["lane"], LANE_POLICY[d["lane"]])
        return {"candidate_id": d["candidate_id"], "lane": d["lane"], "seq": seq, "sample_id": seq.sample_id,
                "defer_count": d["defer_count"], "first_step": d["first_step"]}

    # ------------------------------------------------------------------ batch
    def build_batch(self, step, scorer, ctx):
        """scorer(list[PackedSequence]) -> list[(score, candidate_loss, n_loss)].
        Returns (Batch, decision records)."""
        entry = _mixture.step_entry(self.schedule, step)
        quotas, stage, fcount = entry["quotas"], entry["stage"], entry["floor_counts"]
        decisions = []
        dec_idx = [0]
        tau_box = [0.0]

        def record(c, status, reason, override=False, quota_fill=False, rnd=0):
            s = c["seq"]
            rec = {"decision_id": f"opd-{step:05d}-{dec_idx[0]:03d}", "candidate_id": c["candidate_id"],
                   "step": step, "stage": stage, "lane": c["lane"], "protected_lane": c["lane"] in PROTECTED_LANES,
                   "sample_id": c["sample_id"], "shard_ids": sorted({sp["shard_id"] for sp in s.spans}),
                   "span_ids": [sp["span_id"] for sp in s.spans], "status": status, "reason": reason,
                   "protected_floor_override": override, "quota_fill": quota_fill,
                   "score": c.get("score"), "tau": tau_box[0], "candidate_loss": c.get("loss"),
                   "n_loss_tokens": s.n_loss, "effective_token_estimate":
                       effective_tokens(s.n_loss, c["score"], tau_box[0]) if c.get("score") is not None else s.n_loss,
                   "defer_count": c["defer_count"], "refill_round": rnd, **ctx}
            dec_idx[0] += 1
            decisions.append(rec)
            return rec

        def score(cands):
            if not cands:
                return
            res = scorer([c["seq"] for c in cands])
            for c, (sc, loss, _n) in zip(cands, res):
                c["score"], c["loss"] = float(sc), float(loss)

        accepted = {l: [] for l in LANES}
        if not self.ocfg["enabled"]:
            for lane in LANES:
                for _ in range(quotas[lane]):
                    c = self._draw(lane, step)
                    accepted[lane].append((c, record(c, "accepted", "opus_disabled")))
        else:
            pool = {l: [] for l in LANES}
            for d in self.deferred:
                c = self._rebuild(d)
                if quotas[c["lane"]] == 0:
                    c["score"] = d.get("last_score")
                    record(c, "rejected", "stage_mismatch")
                else:
                    pool[c["lane"]].append(c)
            self.deferred = []
            for lane in LANES:
                q = quotas[lane]
                if q == 0:
                    continue
                n_fresh = max(1, q + int(math.ceil(q * self.ocfg["extra_candidates_frac"])) - len(pool[lane]))
                pool[lane].extend(self._draw(lane, step) for _ in range(n_fresh))
            everything = [c for l in LANES for c in pool[l]]
            score(everything)
            tau = threshold([c["score"] for c in everything], self.ocfg["reject_quantile"])
            tau_box[0] = tau
            recent_ids = {sid for _st, ids in self.recent for sid in ids}
            seen = set()
            for lane in LANES:
                q = quotas[lane]
                if q == 0:
                    continue
                acc, low, beyond, dups = triage(pool[lane], q, tau, recent_ids, seen)
                for c in dups:
                    record(c, "rejected", "duplicate")
                acc = [(c, None) for c in acc]
                # protected floor: rescue the best rejected candidates of an under-floor lane
                if lane in PROTECTED_LANES:
                    while len(acc) < fcount.get(lane, 0) and low:
                        best = low.pop(0)
                        acc.append((best, "override"))
                rnd = 0
                while len(acc) < q and rnd < self.ocfg["refill_rounds"]:
                    rnd += 1
                    extra = [self._draw(lane, step) for _ in range(q - len(acc) + 1)]
                    score(extra)
                    a2, l2, b2, d2 = triage(extra, q - len(acc), tau, recent_ids, seen)
                    for c in d2:
                        record(c, "rejected", "duplicate", rnd=rnd)
                    acc.extend((c, None) for c in a2)
                    low = sorted(low + l2, key=lambda c: (-c["score"], c["candidate_id"]))
                    beyond.extend(b2)
                while len(acc) < q and low:
                    acc.append((low.pop(0), "fill"))
                while len(acc) < q:   # stream exhausted of acceptable candidates: take fresh ones
                    c = self._draw(lane, step)
                    score([c])
                    acc.append((c, "fill"))
                for c, how in acc:
                    if how == "override":
                        rec = record(c, "accepted", "protected_floor_override", override=True)
                    elif how == "fill":
                        rec = record(c, "accepted", "quota_fill_below_threshold", quota_fill=True)
                    else:
                        rec = record(c, "accepted", "proxy_utility_above_threshold")
                    accepted[lane].append((c, rec))
                for c in low:
                    record(c, "rejected", "low_proxy_utility")
                for c in beyond:
                    if c["defer_count"] < self.ocfg["max_defer"]:
                        record(c, "deferred", "quota_full_above_threshold")
                        self.deferred.append({"candidate_id": c["candidate_id"], "lane": c["lane"],
                                              "spans": c["seq"].spans, "defer_count": c["defer_count"] + 1,
                                              "first_step": c["first_step"], "last_score": c["score"]})
                    else:
                        record(c, "rejected", "quota_pressure")
        seqs, dids = [], []
        for lane in LANES:
            for c, rec in sorted(accepted[lane], key=lambda x: x[0]["candidate_id"]):
                seqs.append(c["seq"])
                dids.append(rec["decision_id"])
        self.recent.append([step, [s.sample_id for s in seqs]])
        self.recent = self.recent[-self.ocfg["dup_window_steps"]:]
        return Batch(step, stage, seqs, dids, quotas), decisions

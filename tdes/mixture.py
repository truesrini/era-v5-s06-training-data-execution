"""Mixture timeline compiler: curriculum stages -> executable per-step lane quotas.

Every step's global batch has B sequence slots. Protected floors are reserved first, the
remaining slots go to the lanes with the largest accumulated credit (credit += weight * B each
step), so cumulative actual shares track the planned weights to within one slot. Lanes whose
weight drops to zero in a new stage are cut immediately (e.g. no general web in anneal); other
lanes warm up linearly across `warmup_steps`.
"""
import math

from .config import LANES, PROTECTED_LANES, global_batch
from .util import sha256_json


class MixtureError(Exception):
    pass


def _stage_ranges(cfg):
    T = cfg["train"]["total_steps"]
    out, prev = [], 0
    stages = cfg["mixture"]["stages"]
    for i, st in enumerate(stages):
        end = T if i == len(stages) - 1 else int(round(st["end_frac"] * T))
        if end <= prev:
            raise MixtureError(f"stage {st['stage']} is empty")
        out.append((st, prev + 1, end))
        prev = end
    return out


def floor_counts(floors: dict, B: int) -> dict:
    return {lane: int(math.ceil(share * B - 1e-9)) for lane, share in floors.items()}


def compile_schedule(cfg: dict, lane_supply_tokens: dict) -> dict:
    B, L = global_batch(cfg), cfg["train"]["seq_len"]
    W = cfg["mixture"]["warmup_steps"]
    ranges = _stage_ranges(cfg)
    problems = []

    for st, _, _ in ranges:
        mix = st["mixture"]
        if set(mix) != set(LANES):
            problems.append(f"{st['stage']}: mixture must name every lane")
        if abs(sum(mix.values()) - 1.0) > 1e-6:
            problems.append(f"{st['stage']}: weights sum to {sum(mix.values()):.4f}")
        for lane, f in st["protected_floors"].items():
            if lane not in PROTECTED_LANES:
                problems.append(f"{st['stage']}: {lane} is not a protected lane")
            if mix[lane] + 1e-9 < f:
                problems.append(f"{st['stage']}: {lane} weight {mix[lane]} below its floor {f}")
        if sum(floor_counts(st["protected_floors"], B).values()) > B:
            problems.append(f"{st['stage']}: floors exceed the global batch")
        if st["stage"] != "anneal" and mix.get("anneal_reserve", 0) > 0:
            problems.append(f"{st['stage']}: anneal reserve may only be drawn in the anneal stage")
    if problems:
        raise MixtureError("; ".join(problems))

    credit = {lane: 0.0 for lane in LANES}
    steps, prev_mix = [], None
    for st, s0, s1 in ranges:
        target = st["mixture"]
        fcount = floor_counts(st["protected_floors"], B)
        for step in range(s0, s1 + 1):
            k = step - s0 + 1
            if prev_mix is not None and k <= W:
                a = k / (W + 1)
                w = {l: (0.0 if target[l] == 0 else prev_mix[l] + (target[l] - prev_mix[l]) * a) for l in LANES}
                z = sum(w.values())
                w = {l: v / z for l, v in w.items()}
            else:
                w = dict(target)
            for l in LANES:
                credit[l] += w[l] * B
            alloc = {l: 0 for l in LANES}
            for l, c in fcount.items():
                alloc[l] = c
            for _ in range(B - sum(alloc.values())):
                cands = [l for l in LANES if w[l] > 0]
                best = max(cands, key=lambda l: (credit[l] - alloc[l], -LANES.index(l)))
                alloc[best] += 1
            for l in LANES:
                credit[l] -= alloc[l]
            steps.append({"step": step, "stage": st["stage"], "weights": {l: round(w[l], 6) for l in LANES},
                          "quotas": alloc, "floor_counts": fcount, "warmup": prev_mix is not None and k <= W})
        prev_mix = target

    stages_out = []
    for st, s0, s1 in ranges:
        rows = [s for s in steps if s0 <= s["step"] <= s1]
        n = len(rows)
        planned = {l: round(sum(r["weights"][l] for r in rows) / n, 6) for l in LANES}
        quota_share = {l: round(sum(r["quotas"][l] for r in rows) / (n * B), 6) for l in LANES}
        stages_out.append({"stage": st["stage"], "step_start": s0, "step_end": s1,
                           "token_start": (s0 - 1) * B * L, "token_end": s1 * B * L,
                           "sequence_length": L, "mixture": st["mixture"],
                           "protected_floors": st["protected_floors"],
                           "planned_share": planned, "quota_share": quota_share,
                           "max_quota_deviation": round(max(abs(planned[l] - quota_share[l]) for l in LANES), 6),
                           "warmup_steps": W})

    overhead = 1.0 + cfg["opus"]["extra_candidates_frac"] if cfg["opus"]["enabled"] else 1.0
    scarcity = {}
    for l in LANES:
        req = sum(s["quotas"][l] for s in steps) * L
        sup = lane_supply_tokens.get(l, 0)
        epochs = (req * overhead / sup) if sup else float("inf") if req else 0.0
        status = "ok" if epochs <= 1.0 else ("repeat" if epochs <= cfg["mixture"]["max_epochs"] else "infeasible")
        scarcity[l] = {"requested_positions": req, "requested_with_opus_overhead": int(req * overhead),
                       "available_unique_tokens": sup, "planned_epochs": round(epochs, 3), "status": status,
                       "mitigation": {"ok": "none", "repeat": "repeat existing data (repeated-pass number is ledgered)",
                                      "infeasible": "reduce lane share or move share to a later stage"}[status]}
    infeasible = [l for l, v in scarcity.items() if v["status"] == "infeasible"]
    if infeasible:
        raise MixtureError(f"lanes cannot be satisfied from available shards: {infeasible}")

    sched = {"format": "tdes-mixture-schedule-v1", "global_batch_sequences": B, "sequence_length": L,
             "total_steps": cfg["train"]["total_steps"], "lanes": LANES, "protected_lanes": PROTECTED_LANES,
             "stages": stages_out, "steps": steps, "scarcity": scarcity}
    sched["schedule_hash"] = sha256_json(sched)
    return sched


def step_entry(schedule, step):
    return schedule["steps"][step - 1]

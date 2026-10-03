"""Independent audit. Reads only what is on disk (manifests, shards, ledgers, checkpoints) and
re-derives every claim: ledger integrity, checkpoint/ledger binding, a gap-free and
duplicate-free committed stream, firewall cleanliness of every loss-bearing token, packing
invariants, mixture compliance, the OPUS trail and the learning trace."""
import glob
import os
from collections import defaultdict

import numpy as np

from .config import LANE_POLICY, LANES, global_batch
from .firewall import EvalRegistry, ngram_hashes
from .packing import build_sequence, verify_sequence
from .shards import ShardStore, load_manifests
from .tokenizer import Tokenizer
from .model import checkpoint_weights_hash
from .util import Ledger, read_json, rel, write_json


def _committed_steps(recs):
    return [r["global_step"] for r in recs if r["type"] == "step_committed"]


def _samples_by_step(recs):
    out = defaultdict(dict)
    for r in recs:
        if r["type"] == "microbatch_consumed":
            for g, s in zip(r["global_indices"], r["samples"]):
                out[r["global_step"]][g] = s
    return out


def audit(art, log, main="main", reference="reference"):
    cfg = read_json(os.path.join(art, "run_config.json"))
    man = os.path.join(art, "manifests")
    lock = read_json(os.path.join(man, "tokenizer.lock.json"))
    tok = Tokenizer.load(os.path.join(man, "tokenizer.json"), expected_hash=lock["tokenizer_hash"])
    manifests = load_manifests(man)
    catalog = read_json(os.path.join(man, "catalog.json"))
    registry = EvalRegistry.load(man)
    store = ShardStore(art, manifests, tok.hash, capacity=64)
    sched = read_json(os.path.join(man, "mixture_schedule.json"))
    admitted = set(catalog["admitted_train"])
    B, L = global_batch(cfg), cfg["train"]["seq_len"]
    per_step_mb = cfg["train"]["world_size"] * cfg["train"]["grad_accum"]
    checks = []
    sections = {}

    def check(section, name, ok, detail="", evidence=None):
        checks.append({"section": section, "name": name, "passed": bool(ok), "detail": detail, "evidence": evidence})
        log.check(f"audit.{name}", ok, detail)
        return ok

    # ------------------------------------------------------------ A. ledger integrity
    ledgers = sorted(glob.glob(os.path.join(art, "ledgers", "**", "*.jsonl"), recursive=True))
    chain = {}
    for p in ledgers:
        recs = Ledger.read(p)
        ok, bad = Ledger.verify_chain(recs)
        chain[rel(p, art)] = {"records": len(recs), "chain_ok": ok, "first_bad": bad}
    check("ledgers", "ledger_hash_chains_valid", all(v["chain_ok"] for v in chain.values()),
          f"{len(chain)} ledgers, {sum(v['records'] for v in chain.values())} records, every prev_hash/event_hash verified")
    sections["ledger_chains"] = chain

    raw = {b: Ledger.read(os.path.join(art, "ledgers", b, "consumption.jsonl"))
           for b in os.listdir(os.path.join(art, "ledgers")) if os.path.isdir(os.path.join(art, "ledgers", b))}
    eff = {b: Ledger.effective(r) for b, r in raw.items()}

    # ------------------------------------------------------------ B. checkpoint binding
    ck_rows = []
    for st_path in sorted(glob.glob(os.path.join(art, "checkpoints", "*", "step_*", "state.json"))):
        st = read_json(st_path)
        d = os.path.dirname(st_path)
        w_ok = checkpoint_weights_hash(os.path.join(d, "model.pt")) == st["weights_hash"]
        row = {"checkpoint_id": st["checkpoint_id"], "path": rel(d, art), "step": st["global_step"], "weights_ok": w_ok}
        for name, off in st["ledger_offsets"].items():
            recs = Ledger.read(os.path.join(art, off["path"]))
            row[f"{name}_offset"] = off["offset"]
            row[f"{name}_bound"] = (off["offset"] == 0 and off["last_hash"] == Ledger.GENESIS) or                 (0 < off["offset"] <= len(recs) and recs[off["offset"] - 1]["event_hash"] == off["last_hash"])
        crec = Ledger.read(os.path.join(art, st["ledger_offsets"]["consumption"]["path"]))[st["ledger_offsets"]["consumption"]["offset"] - 1]
        row["offset_points_at_own_checkpoint_event"] = crec["type"] == "checkpoint_saved" and crec["checkpoint_id"] == st["checkpoint_id"]
        prefix = Ledger.effective(Ledger.read(os.path.join(art, st["ledger_offsets"]["consumption"]["path"]))[:st["ledger_offsets"]["consumption"]["offset"]])
        commits = [r for r in prefix if r["type"] == "step_committed"]
        row["last_committed_step_matches"] = bool(commits) and commits[-1]["global_step"] == st["global_step"] and \
            commits[-1]["weights_hash"] == st["weights_hash"]
        row["ok"] = all(v for k, v in row.items() if k.endswith("_ok") or k.endswith("_bound") or k in (
            "offset_points_at_own_checkpoint_event", "last_committed_step_matches"))
        ck_rows.append(row)
    check("checkpoints", "checkpoints_bound_to_ledger_offsets", ck_rows and all(r["ok"] for r in ck_rows),
          f"{len(ck_rows)} checkpoints: weights hash, ledger offset+hash, offset == own checkpoint_saved event, "
          f"last committed step and weights agree", "reports/audit_report.json#checkpoints")
    sections["checkpoints"] = ck_rows

    # ------------------------------------------------------------ C. committed stream of main
    m_eff = eff[main]
    steps = _committed_steps(m_eff)
    T = cfg["train"]["total_steps"]
    dup = sorted({s for s in steps if steps.count(s) > 1})
    gaps = sorted(set(range(1, T + 1)) - set(steps))
    rolled = sum(len(r["rolled_back_seqs"]) for r in raw[main] if r["type"] == "rollback")
    mb_counts = defaultdict(int)
    for r in m_eff:
        if r["type"] == "microbatch_consumed":
            mb_counts[r["global_step"]] += 1
    planned = {r["global_step"]: r for r in m_eff if r["type"] == "batch_planned"}
    by_step = _samples_by_step(m_eff)
    complete = all(mb_counts[s] == per_step_mb for s in steps) and all(
        [by_step[s][g]["sample_id"] for g in range(B)] == planned[s]["sample_ids"] for s in steps)
    check("stream", "committed_stream_has_no_gaps_or_repeats", steps == list(range(1, T + 1)) and not dup and not gaps,
          f"steps 1..{T} each committed exactly once (duplicates={dup}, gaps={gaps}); {rolled} uncommitted records "
          f"superseded by an explicit rollback record", f"ledgers/{main}/consumption.jsonl")
    check("stream", "every_step_complete", complete,
          f"each step has {per_step_mb} microbatches whose samples equal the batch_planned sample list")
    sections["stream"] = {"committed_steps": len(steps), "duplicates": dup, "gaps": gaps, "rolled_back_records": rolled}

    # resume equivalence: the crashed+resumed run must equal the uninterrupted reference run
    if reference in eff:
        rp = {r["global_step"]: r for r in eff[reference] if r["type"] == "batch_planned"}
        rc = {r["global_step"]: r for r in eff[reference] if r["type"] == "step_committed"}
        mc = {r["global_step"]: r for r in m_eff if r["type"] == "step_committed"}
        same = [s for s in steps if rp.get(s, {}).get("batch_hash") == planned[s]["batch_hash"]]
        w_same = mc[T]["weights_hash"] == rc.get(T, {}).get("weights_hash")
        check("resume", "resumed_stream_equals_uninterrupted_run", len(same) == T and w_same,
              f"{len(same)}/{T} batch hashes equal the reference run; final weights {mc[T]['weights_hash'][:12]} "
              f"== reference {rc.get(T, {}).get('weights_hash', '')[:12]}")
        sections["resume_equivalence"] = {"matching_steps": len(same), "total_steps": T, "final_weights_equal": w_same}

    # ------------------------------------------------------------ D. firewall over every consumed token
    never = registry.never_train_ids()
    scanned_tokens = 0
    violations = []
    for b, recs in eff.items():
        for r in recs:
            if r["type"] != "microbatch_consumed":
                continue
            for smp in r["samples"]:
                for sp in smp["spans"]:
                    sid = sp["shard_id"]
                    if sid in never or sid not in admitted or manifests[sid]["split"] != "train":
                        violations.append({"branch": b, "step": r["global_step"], "span": sp["span_id"], "why": "shard"})
                        continue
                    t = store.get(sid).doc_slice(sp["doc_index"], sp["start"], sp["end"])[0]
                    scanned_tokens += len(t)
                    ng = ngram_hashes(t)
                    if len(ng) and np.isin(ng, registry.fingerprints).any():
                        violations.append({"branch": b, "step": r["global_step"], "span": sp["span_id"], "why": "fingerprint"})
    fw = Ledger.read(os.path.join(art, "ledgers", "firewall.jsonl"))
    vreads = [r for r in fw if r["type"] == "validation_read"]
    blocked = [r for r in fw if r["type"] in ("shard_blocked", "batch_blocked", "shard_quarantined")]
    check("firewall", "no_eval_or_validation_token_in_any_trained_batch", not violations,
          f"{scanned_tokens} consumed tokens across {len(eff)} branches re-scanned against "
          f"{len(registry.fingerprints)} eval 12-gram fingerprints and shard permissions: {len(violations)} violations")
    check("firewall", "validation_reads_are_gradient_free", vreads and all(r["gradient"] is False for r in vreads),
          f"{len(vreads)} validation reads logged, all eval-only (gradient=false)", "ledgers/firewall.jsonl")
    check("firewall", "blocked_events_recorded", any(r["type"] == "shard_blocked" and r["split"] == "test" for r in blocked),
          f"{len(blocked)} block/quarantine events in the firewall ledger", "ledgers/firewall.jsonl")
    sections["firewall"] = {"scanned_tokens": scanned_tokens, "violations": violations, "validation_reads": len(vreads),
                            "block_events": [{k: r.get(k) for k in ("seq", "type", "shard_id", "split", "actor", "reason")}
                                             for r in blocked]}

    # ------------------------------------------------------------ E. packing, rebuilt from span refs
    rebuilt, mismatch, failures = 0, [], []
    lane_pos, lane_nonpad, lane_loss = defaultdict(int), defaultdict(int), defaultdict(int)
    policy_ok = True
    totals = {"positions": 0, "nonpad_tokens": 0, "loss_tokens": 0}
    for s in steps:
        for g in range(B):
            smp = by_step[s][g]
            seq = build_sequence(smp["spans"], store, L, smp["lane"], smp["policy"])
            rebuilt += 1
            if seq.seq_hash != smp["seq_hash"] or seq.sample_id != smp["sample_id"] or seq.loss_mask_hash() != smp["loss_mask_hash"]:
                mismatch.append(smp["sample_id"])
            bad = verify_sequence(seq, store)
            if bad:
                failures.append({"sample_id": smp["sample_id"], "failed": bad})
            policy_ok &= smp["policy"] == LANE_POLICY[smp["lane"]]
            lane_pos[smp["lane"]] += L
            lane_nonpad[smp["lane"]] += seq.n_tokens
            lane_loss[smp["lane"]] += seq.n_loss
            totals["positions"] += L
            totals["nonpad_tokens"] += seq.n_tokens
            totals["loss_tokens"] += seq.n_loss
    ledger_tot = {"positions": 0, "nonpad_tokens": 0, "loss_tokens": 0}
    for r in m_eff:
        if r["type"] == "microbatch_consumed":
            for k in ledger_tot:
                ledger_tot[k] += r[k]
    check("packing", "packed_samples_rebuild_bit_exact", not mismatch,
          f"{rebuilt} consumed samples rebuilt from span refs: sample ids, sequence hashes and loss-mask hashes identical")
    check("packing", "mask_and_position_invariants", not failures,
          f"{rebuilt} samples: no loss on pads, no loss across segment ends, position ids reset per segment, "
          f"block-causal attention, structured samples never split")
    check("packing", "lane_packing_policy_applied", policy_ok, "every sample used its lane's packing policy")
    check("packing", "ledger_token_counts_reconstruct", ledger_tot == totals,
          f"ledger counts {ledger_tot} == recount from shards {totals}")
    util = {l: round(lane_nonpad[l] / lane_pos[l], 4) for l in lane_pos}
    sections["packing"] = {"rebuilt_samples": rebuilt, "mismatches": mismatch, "invariant_failures": failures,
                           "totals": totals, "utilization": round(totals["nonpad_tokens"] / totals["positions"], 6),
                           "loss_bearing_fraction": round(totals["loss_tokens"] / totals["positions"], 6),
                           "utilization_by_lane": util,
                           "loss_tokens_by_lane": dict(lane_loss), "positions_by_lane": dict(lane_pos)}

    # ------------------------------------------------------------ F. mixture compliance
    mix_rows, floor_viol = [], []
    floor_by_step = {e["step"]: e["floor_counts"] for e in sched["steps"]}
    for st in sched["stages"]:
        cnt = defaultdict(int)
        tok_cnt = defaultdict(int)
        n = 0
        for s in range(st["step_start"], st["step_end"] + 1):
            per = defaultdict(int)
            for g in range(B):
                smp = by_step[s][g]
                cnt[smp["lane"]] += 1
                per[smp["lane"]] += 1
                tok_cnt[smp["lane"]] += smp["n_loss_tokens"]
                n += 1
            for lane, c in floor_by_step[s].items():
                if per[lane] < c:
                    floor_viol.append({"step": s, "lane": lane, "got": per[lane], "floor": c})
        tot_loss = sum(tok_cnt.values())
        actual = {l: round(cnt[l] / n, 6) for l in LANES}
        dev = {l: round(abs(actual[l] - st["planned_share"][l]), 6) for l in LANES}
        # integer slots: a lane can be off by its carried credit (<1 slot) plus floor rounding (<1 slot)
        tol = max(0.02, 2.0 / n)
        mix_rows.append({"stage": st["stage"], "steps": [st["step_start"], st["step_end"]],
                         "planned_share": st["planned_share"], "compiled_quota_share": st["quota_share"],
                         "actual_share_sequences": actual, "tolerance": round(tol, 6),
                         "actual_equals_compiled_quota": all(abs(actual[l] - st["quota_share"][l]) < 1e-6 for l in LANES),
                         "actual_share_loss_tokens": {l: round(tok_cnt[l] / tot_loss, 6) for l in LANES},
                         "abs_deviation": dev, "max_abs_deviation": max(dev.values()),
                         "protected_floors": st["protected_floors"]})
    check("mixture", "actual_shares_equal_compiled_quotas", all(r["actual_equals_compiled_quota"] for r in mix_rows),
          "per stage, consumed sequences per lane equal the compiled per-step quotas exactly")
    check("mixture", "planned_vs_actual_shares_within_tolerance", all(r["max_abs_deviation"] <= r["tolerance"] for r in mix_rows),
          "; ".join(f"{r['stage']} max|dev|={r['max_abs_deviation']:.4f} (tol {r['tolerance']:.4f})" for r in mix_rows),
          "reports/audit_report.json#mixture")
    check("mixture", "protected_floors_held_every_step", not floor_viol,
          f"indic/agentic/reasoning floors met in all {len(steps)} committed steps ({len(floor_viol)} violations)")
    anneal_leak = [s for s in steps for g in range(B)
                   if by_step[s][g]["lane"] == "anneal_reserve" and planned[s]["stage"] != "anneal"]
    check("mixture", "anneal_reserve_fenced", not anneal_leak, "anneal-reserve shards consumed only in the anneal stage")
    sections["mixture"] = {"stages": mix_rows, "floor_violations": floor_viol,
                           "tolerance_rule": "max(0.02, 2 / sequences_in_stage)"}

    # ------------------------------------------------------------ G. OPUS trail
    o_eff = Ledger.effective(Ledger.read(os.path.join(art, "ledgers", main, "opus.jsonl")))
    decs = {r["decision_id"]: r for r in o_eff}
    linked = 0
    bad_links = []
    for s in steps:
        for g in range(B):
            smp = by_step[s][g]
            d = decs.get(smp["opus_decision_id"])
            if d and d["status"] == "accepted" and d["sample_id"] == smp["sample_id"] and d["step"] == s:
                linked += 1
            else:
                bad_links.append(smp["opus_decision_id"])
    status = defaultdict(int)
    reasons = defaultdict(int)
    by_lane = defaultdict(lambda: defaultdict(int))
    for d in o_eff:
        status[d["status"]] += 1
        reasons[d["reason"]] += 1
        by_lane[d["lane"]][d["status"]] += 1
    overrides = sum(1 for d in o_eff if d["protected_floor_override"])
    req = ("decision_id", "candidate_id", "shard_ids", "lane", "stage", "scoring_checkpoint", "proxy_version", "score",
           "status", "reason", "protected_floor_override", "effective_token_estimate")
    complete_fields = all(all(k in d for k in req) for d in o_eff)
    consumed_decisions = {by_step[s][g]["opus_decision_id"] for s in steps for g in range(B)}
    rejected_consumed = [d["decision_id"] for d in o_eff if d["status"] != "accepted" and d["decision_id"] in consumed_decisions]
    check("opus", "every_consumed_sample_has_accepted_decision", not bad_links,
          f"{linked}/{len(steps) * B} consumed samples link to an accepted OPUS record")
    check("opus", "all_decision_kinds_exercised", all(status[k] > 0 for k in ("accepted", "rejected", "deferred")) and overrides > 0,
          f"accepted={status['accepted']} rejected={status['rejected']} deferred={status['deferred']} "
          f"protected_floor_override={overrides}; reasons={dict(reasons)}", f"ledgers/{main}/opus.jsonl")
    check("opus", "rejected_or_deferred_candidates_never_consumed", not rejected_consumed,
          f"{len(rejected_consumed)} rejected/deferred candidates were consumed")
    check("opus", "decision_records_complete", complete_fields, f"{len(o_eff)} records carry {len(req)} required fields")
    sections["opus"] = {"records": len(o_eff), "status": dict(status), "reasons": dict(reasons),
                        "protected_floor_overrides": overrides, "by_lane": {k: dict(v) for k, v in by_lane.items()},
                        "rejection_rate_by_lane": {k: round(v.get("rejected", 0) / sum(v.values()), 4) for k, v in by_lane.items()}}

    # ------------------------------------------------------------ H. learning trace
    l_eff = Ledger.effective(Ledger.read(os.path.join(art, "ledgers", main, "learning.jsonl")))
    srec = {(r["global_step"], r["sample_id"], r["trace_seq_index"]): r for r in l_eff if r["type"] == "sample_learning"}
    missing = [(s, g) for s in steps for g in range(B) if (s, by_step[s][g]["sample_id"], g) not in srec]
    trace_bad, trace_rows = [], 0
    for s in steps:
        z = np.load(os.path.join(art, "ledgers", main, "token_trace", f"step_{s:05d}.npz"))
        trace_rows += len(z["loss_before"])
        for g in range(B):
            r = srec.get((s, by_step[s][g]["sample_id"], g))
            sel = z["seq_index"] == g
            if r is None:
                continue
            m = float(z["loss_before"][sel].mean()) if sel.any() else 0.0
            if abs(m - r["loss_before"]) > 1e-4 or int(sel.sum()) != r["n_loss_tokens"] or str(z["sample_ids"][g]) != r["sample_id"]:
                trace_bad.append((s, g))
            if [sp["shard_id"] for sp in r["spans"]] and not set(sp["shard_id"] for sp in r["spans"]) <= set(
                    sp["shard_id"] for sp in by_step[s][g]["spans"]):
                trace_bad.append((s, g, "shard"))
    check("learning", "every_consumed_sample_has_learning_record", not missing,
          f"{len(srec)} sample learning records; {len(missing)} consumed samples without one")
    check("learning", "token_trace_reconciles_with_ledger", not trace_bad,
          f"{trace_rows} token-level losses; per-sample means equal ledger loss_before and link to span/shard/doc")
    report_card = _learning_report(l_eff, o_eff, manifests, tok, art)
    write_json(os.path.join(art, "reports", "learning_report.json"), report_card)
    check("learning", "shard_report_cards_written", len(report_card["shards"]) > 0,
          f"{len(report_card['shards'])} shard report cards (loss, ppl, delta, grad norm, OPUS score, class)",
          "reports/learning_report.json")
    sections["learning"] = {"sample_records": len(srec), "token_trace_rows": trace_rows,
                            "classification_counts": report_card["classification_counts"]}

    # ------------------------------------------------------------ I. replay + fork (from their ledgers)
    rep = [b for b in eff if b.startswith("replay-")]
    if rep:
        rb = rep[0]
        rplan = {r["global_step"]: r for r in eff[rb] if r["type"] == "batch_planned"}
        rcom = {r["global_step"]: r for r in eff[rb] if r["type"] == "step_committed"}
        mcom = {r["global_step"]: r for r in m_eff if r["type"] == "step_committed"}
        rows = [{"step": s, "original": planned[s]["batch_hash"], "replay": rplan[s]["batch_hash"],
                 "weights_equal": rcom[s]["weights_hash"] == mcom[s]["weights_hash"]} for s in sorted(rplan)]
        rspans = _samples_by_step(eff[rb])
        span_eq = all([x["span_id"] for g in range(B) for x in rspans[s][g]["spans"]] ==
                      [x["span_id"] for g in range(B) for x in by_step[s][g]["spans"]] for s in rplan)
        check("replay", "replay_ledger_matches_original", rows and all(r["original"] == r["replay"] and r["weights_equal"] for r in rows) and span_eq,
              f"{rb}: {len(rows)} steps, batch hashes + token spans + post-step weights equal to '{main}'")
        sections["replay"] = {"branch": rb, "rows": rows}
    fk = [b for b in eff if b.startswith("fork-")]
    if fk:
        fb = fk[0]
        frecs = raw[fb]
        fev = frecs[0]
        pst = read_json(os.path.join(art, "checkpoints", fev["parent_branch"], f"step_{fev['divergence_step'] - 1:05d}", "state.json"))
        lineage = [s for s in steps if s < fev["divergence_step"]] + _committed_steps(eff[fb])
        check("fork", "fork_lineage_explicit", fev["type"] == "branch_forked" and fev["parent_ledger_offset"] == pst["ledger_offsets"]["consumption"]
              and lineage == list(range(1, max(lineage) + 1)),
              f"{fb}: first record is branch_forked at step {fev['divergence_step']} with the parent's ledger offset "
              f"{fev['parent_ledger_offset']['offset']}; lineage = {main}[1..{fev['divergence_step'] - 1}] + {fb}"
              f"[{fev['divergence_step']}..{max(lineage)}]")
        sections["fork"] = {"branch": fb, "divergence_step": fev["divergence_step"], "lineage_steps": lineage}

    # ------------------------------------------------------------ J. audit queries
    sections["queries"] = _queries(m_eff, o_eff, by_step, planned, steps, B, L, cfg)
    log.event("audit completed", f"{sum(c['passed'] for c in checks)}/{len(checks)} audit checks passed")
    out = {"checks": checks, **sections}
    write_json(os.path.join(art, "reports", "audit_report.json"), out)
    return out


def _queries(m_eff, o_eff, by_step, planned, steps, B, L, cfg):
    """Examples of questions the ledger answers (Session 6 sec. 14)."""
    commits = {r["global_step"]: r for r in m_eff if r["type"] == "step_committed"}
    lo, hi = (cfg["train"]["total_steps"] // 4) * B * L, (cfg["train"]["total_steps"] // 2) * B * L
    infl = defaultdict(lambda: {"loss_tokens": 0, "samples": 0, "lanes": set()})
    for s in steps:
        start = commits[s]["cum_positions"] - B * L
        if start < lo or start >= hi:
            continue
        for g in range(B):
            smp = by_step[s][g]
            for sp in smp["spans"]:
                infl[sp["shard_id"]]["samples"] += 1
                infl[sp["shard_id"]]["lanes"].add(smp["lane"])
            infl[smp["spans"][0]["shard_id"]]["loss_tokens"] += smp["n_loss_tokens"]
    q1 = sorted(({"shard_id": k, "spans": v["samples"], "loss_tokens": v["loss_tokens"], "lanes": sorted(v["lanes"])}
                 for k, v in infl.items()), key=lambda r: -r["spans"])
    deltas = [(s, commits[s]["loss"] - commits[s - 1]["loss"]) for s in steps if s > 1]
    spike_step, spike = max(deltas, key=lambda x: x[1])
    before = [d for d in o_eff if d["step"] in (spike_step - 1, spike_step) and d["status"] == "accepted"]
    q2 = {"spike_step": spike_step, "loss_increase": round(spike, 6), "stage": planned[spike_step]["stage"],
          "opus_accepted_before_spike": [{k: d[k] for k in ("decision_id", "step", "lane", "score", "reason", "shard_ids")}
                                         for d in before]}
    return {"which_shards_influenced_tokens": {"token_range": [lo, hi], "shards": q1},
            "what_preceded_the_largest_loss_spike": q2}


def _learning_report(l_eff, o_eff, manifests, tok, art):
    steps = {r["global_step"]: r for r in l_eff if r["type"] == "step_learning"}
    score = {d["decision_id"]: d["score"] for d in o_eff}
    agg = defaultdict(lambda: {"n_spans": 0, "loss_tokens": 0, "loss_before_sum": 0.0, "loss_after_sum": 0.0,
                               "grad_norms": [], "opus_scores": [], "phases": set(), "max_pass": 0, "lane": None})
    tok_loss = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for r in l_eff:
        if r["type"] != "sample_learning":
            continue
        for sp in r["spans"]:
            a = agg[sp["shard_id"]]
            a["lane"] = r["lane"]
            a["n_spans"] += 1
            a["loss_tokens"] += sp["n_loss_tokens"]
            a["loss_before_sum"] += sp["loss_before"] * sp["n_loss_tokens"]
            a["loss_after_sum"] += sp["loss_after"] * sp["n_loss_tokens"]
            a["grad_norms"].append(steps[r["global_step"]]["grad_norm"])
            if r["opus_decision_id"] in score and score[r["opus_decision_id"]] is not None:
                a["opus_scores"].append(score[r["opus_decision_id"]])
            a["phases"].add(r["model_phase"])
            a["max_pass"] = max(a["max_pass"], sp["repeated_pass"])
        for t in r["top_surprise"]:
            sid = r["spans"][0]["shard_id"] if r["spans"] else None
            if sid:
                tl = tok_loss[sid][t["token_id"]]
                tl[0] += t["loss"]
                tl[1] += 1
    shards, counts = [], defaultdict(int)
    for sid, a in sorted(agg.items()):
        n = max(1, a["loss_tokens"])
        lb, la = a["loss_before_sum"] / n, a["loss_after_sum"] / n
        delta = la - lb
        cls = "useful" if delta < -0.005 else ("harmful" if delta > 0.005 else "neutral")
        counts[cls] += 1
        hard = sorted(((tid, v[0] / v[1], v[1]) for tid, v in tok_loss[sid].items()), key=lambda x: -x[1])[:5]
        shards.append({"shard_id": sid, "lane": a["lane"], "spans_trained": a["n_spans"], "loss_tokens": a["loss_tokens"],
                       "avg_token_loss": round(lb, 5), "avg_token_ppl": round(float(np.exp(lb)), 3),
                       "loss_delta_after_exposure": round(delta, 5),
                       "mean_grad_norm": round(float(np.mean(a["grad_norms"])), 5),
                       "mean_opus_score": round(float(np.mean(a["opus_scores"])), 5) if a["opus_scores"] else None,
                       "model_phases": sorted(a["phases"]), "max_repeated_pass": a["max_pass"],
                       "high_perplexity_tokens": [{"token_id": int(t), "preview": tok.decode([t]), "mean_loss": round(l, 3), "count": c}
                                                  for t, l, c in hard],
                       "classification": cls})
    # rejected-but-hard lanes: OPUS rejected candidates whose own loss was high (proxy may miss a capability)
    lane_rej = defaultdict(list)
    lane_acc = defaultdict(list)
    for d in o_eff:
        if d.get("candidate_loss") is None:
            continue
        (lane_rej if d["status"] == "rejected" else lane_acc)[d["lane"]].append(d["candidate_loss"])
    proxy_gaps = {l: {"rejected_mean_candidate_loss": round(float(np.mean(v)), 4),
                      "accepted_mean_candidate_loss": round(float(np.mean(lane_acc[l])), 4) if lane_acc[l] else None,
                      "flag": "proxy may undervalue this lane" if lane_acc[l] and np.mean(v) > np.mean(lane_acc[l]) else ""}
                  for l, v in lane_rej.items()}
    return {"shards": shards, "classification_counts": dict(counts), "opus_proxy_gap_by_lane": proxy_gaps,
            "classification_rule": "useful if mean token loss after the update is >0.005 nats lower than before, "
                                   "harmful if >0.005 higher, else neutral"}

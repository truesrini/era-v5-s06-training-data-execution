"""Evidence bundle: evidence.json + evidence.md, assembled only from checks the pipeline and the
audit computed and wrote to disk. Nothing here decides a result on its own: a requirement
passes iff every underlying check passed."""
import os

from .packing import build_sequence
from .shards import ShardStore, load_manifests
from .tokenizer import Tokenizer
from .util import Ledger, read_json, sha256_json, write_json

REQUIRED_LOG_EVENTS = ["shards created", "manifests validated", "evaluation data blocked", "mixture compiled",
                       "batches packed", "OPUS decisions recorded", "checkpoint saved", "crash simulated",
                       "run resumed", "historical stream replayed", "branch forked", "audit completed",
                       "performance measured"]
REQUIRED_PASS_LINES = ["tokenizer_hash_verified", "eval_shard_blocked", "checkpoint_saved",
                       "resume_next_batch_matched", "replay_hash_matched"]


def _first_seq(recs, pred):
    r = next((r for r in recs if pred(r)), None)
    return None if r is None else r["seq"]


def packed_batch_report(art, step=1, branch="main"):
    """Human-inspectable view of one consumed batch: per sample spans, segments, masks, positions."""
    lock = read_json(os.path.join(art, "manifests", "tokenizer.lock.json"))
    tok = Tokenizer.load(os.path.join(art, "manifests", "tokenizer.json"), expected_hash=lock["tokenizer_hash"])
    store = ShardStore(art, load_manifests(os.path.join(art, "manifests")), tok.hash)
    recs = Ledger.effective(Ledger.read(os.path.join(art, "ledgers", branch, "consumption.jsonl")))
    L = read_json(os.path.join(art, "run_config.json"))["train"]["seq_len"]
    mbs = [r for r in recs if r["type"] == "microbatch_consumed" and r["global_step"] == step]
    samples = []
    for mb in mbs:
        for g, smp in zip(mb["global_indices"], mb["samples"]):
            seq = build_sequence(smp["spans"], store, L, smp["lane"], smp["policy"])
            segs = []
            for k, sp in enumerate(seq.spans, start=1):
                idx = (seq.segment_ids == k).nonzero()[0]
                segs.append({"segment": k, "span_id": sp["span_id"], "doc_id": sp["doc_id"], "positions": [int(idx[0]), int(idx[-1])],
                             "position_ids_start_end": [int(seq.position_ids[idx[0]]), int(seq.position_ids[idx[-1]])],
                             "loss_tokens": int(seq.loss_mask[idx].sum()), "context_tokens": int(len(idx) - seq.loss_mask[idx].sum()),
                             "text_preview": tok.decode(seq.tokens[idx][:24])})
            show = min(L, 48)
            samples.append({"global_index": g, "rank": mb["rank"], "microbatch_id": mb["microbatch_id"],
                            "sample_id": smp["sample_id"], "lane": smp["lane"], "policy": smp["policy"],
                            "opus_decision_id": smp["opus_decision_id"], "seq_hash": seq.seq_hash,
                            "seq_hash_matches_ledger": seq.seq_hash == smp["seq_hash"],
                            "loss_mask_hash": seq.loss_mask_hash(), "real_tokens": seq.n_tokens, "padding": L - seq.n_tokens,
                            "loss_tokens": seq.n_loss, "segments": segs,
                            "first_positions": {"token_ids": seq.tokens[:show].tolist(),
                                                "segment_ids": seq.segment_ids[:show].tolist(),
                                                "position_ids": seq.position_ids[:show].tolist(),
                                                "loss_mask": seq.loss_mask[:show].astype(int).tolist()}})
    samples.sort(key=lambda s: s["global_index"])
    planned = next(r for r in recs if r["type"] == "batch_planned" and r["global_step"] == step)
    out = {"step": step, "batch_id": planned["batch_id"], "batch_hash": planned["batch_hash"], "stage": planned["stage"],
           "attention_policy": mbs[0]["attention_policy"], "position_policy": mbs[0]["position_policy"],
           "mask_conventions": "segment_ids 1..k per packed span (0=pad); position_ids restart at 0 per segment; "
                               "token i attends to j iff same segment and j<=i; loss_mask[i]=1 iff token i+1 is in the "
                               "same segment and is a loss-bearing target",
           "samples": samples}
    write_json(os.path.join(art, "reports", "packed_batch_report.json"), out)
    return out


def build_evidence(art, log, tests=None):
    cfg = read_json(os.path.join(art, "run_config.json"))
    build = read_json(os.path.join(art, "reports", "build_report.json"))
    audit = read_json(os.path.join(art, "reports", "audit_report.json"))
    resume = read_json(os.path.join(art, "reports", "resume_report.json"))
    replay = read_json(os.path.join(art, "reports", "replay_report.json"))
    fork = read_json(os.path.join(art, "reports", "fork_report.json"))
    perf = read_json(os.path.join(art, "performance.json"))
    lock = read_json(os.path.join(art, "manifests", "tokenizer.lock.json"))
    pbr = packed_batch_report(art)
    with open(os.path.join(art, "run.log"), encoding="utf-8") as f:
        runlog = f.read()
    fw = Ledger.read(os.path.join(art, "ledgers", "firewall.jsonl"))
    opus = Ledger.effective(Ledger.read(os.path.join(art, "ledgers", "main", "opus.jsonl")))
    learn = Ledger.effective(Ledger.read(os.path.join(art, "ledgers", "main", "learning.jsonl")))

    bchk = {c["name"]: c for c in build["checks"]}
    achk = {c["name"]: c for c in audit["checks"]}

    def B(name):
        c = bchk[name]
        return {"check": f"build.{name}", "passed": c["passed"], "detail": c["detail"]}

    def A(name):
        c = achk[name]
        return {"check": f"audit.{name}", "passed": c["passed"], "detail": c["detail"]}

    def C(name, ok, detail):
        return {"check": name, "passed": bool(ok), "detail": detail}

    m = perf["main_run"]
    a_manifest = sorted(os.listdir(os.path.join(art, "manifests", "shards")))[0]
    test_block = _first_seq(fw, lambda r: r["type"] == "shard_blocked" and r["split"] == "test")
    val_block = _first_seq(fw, lambda r: r["type"] == "shard_blocked" and r["split"] == "validation")
    batch_block = _first_seq(fw, lambda r: r["type"] == "batch_blocked")
    quarantine = _first_seq(fw, lambda r: r["type"] == "shard_quarantined")
    ex = {}
    for kind, pred in (("accepted", lambda d: d["status"] == "accepted" and not d["protected_floor_override"]),
                       ("rejected", lambda d: d["status"] == "rejected" and d["reason"] == "low_proxy_utility"),
                       ("deferred", lambda d: d["status"] == "deferred"),
                       ("protected_floor_override", lambda d: d["protected_floor_override"]),
                       ("stage_mismatch", lambda d: d["reason"] == "stage_mismatch")):
        d = next((d for d in opus if pred(d)), None)
        if d:
            ex[kind] = {"seq": d["seq"], "decision_id": d["decision_id"], "lane": d["lane"], "score": d["score"],
                        "tau": d["tau"], "reason": d["reason"]}
    lrec = next(r for r in learn if r["type"] == "sample_learning")
    resumed = resume["resumed_next_batch"]
    exp_c = resume["expected_from_crashed_process"] or {}
    exp_r = resume["expected_from_uninterrupted_reference"] or {}
    events_present = {e: (f"EVENT  " in runlog and f"] {e}" in runlog) for e in REQUIRED_LOG_EVENTS}
    pass_present = {p: f"[PASS] {p}" in runlog and f"[FAIL] {p}" not in runlog for p in REQUIRED_PASS_LINES}

    reqs = [
        ("end_to_end", "End-to-end execution", "run.log event sequence",
         [C("run_log_has_every_required_event", all(events_present.values()),
            "missing: " + ", ".join(k for k, v in events_present.items() if not v) if not all(events_present.values())
            else f"all {len(events_present)} required events present: " + ", ".join(events_present)),
          C("run_log_has_required_pass_lines", all(pass_present.values()), ", ".join(f"[PASS] {k}" for k in pass_present))],
         ["run.log"], {"events": events_present}),
        ("tokenizer_integrity", "Tokenizer integrity", "Manifest record",
         [B("tokenizer_hash_verified"), B("tokenizer_deterministic"), B("tokenizer_nfc_stable"),
          B("special_tokens_not_injectable"), B("all_manifests_bind_frozen_tokenizer")],
         ["manifests/tokenizer.lock.json", "manifests/tokenizer.json", f"manifests/shards/{a_manifest}#tokenizer_hash"],
         {"tokenizer_hash": lock["tokenizer_hash"], "vocab_size": lock["vocab_size"]}),
        ("shards_manifests", "Immutable shards and manifests", "Manifest validation report",
         [B("manifests_validated"), B("shard_files_immutable"), B("license_gate_blocked_unknown_source"),
          B("contaminated_shard_quarantined")],
         ["manifests/shards/", "manifests/catalog.json", "reports/manifest_validation.json", "reports/admission_report.json"],
         {"n_shards": build["n_shards"], "admitted_train": len(build["admitted_train"])}),
        ("evaluation_firewall", "Evaluation firewall", "Blocked-shard event",
         [B("eval_shard_blocked"), B("validation_shard_blocked"), B("quarantined_shard_blocked"),
          B("eval_batch_blocked_at_serve_time"), B("admitted_train_shards_clean"),
          A("no_eval_or_validation_token_in_any_trained_batch"), A("validation_reads_are_gradient_free"),
          A("blocked_events_recorded")],
         [f"ledgers/firewall.jsonl#seq={test_block} (test shard blocked)", f"ledgers/firewall.jsonl#seq={val_block} (validation shard blocked)",
          f"ledgers/firewall.jsonl#seq={batch_block} (eval batch blocked)", f"ledgers/firewall.jsonl#seq={quarantine} (shard quarantined)",
          "manifests/eval_registry.json"],
         {"scanned_consumed_tokens": audit["firewall"]["scanned_tokens"], "violations": len(audit["firewall"]["violations"])}),
        ("packing_correctness", "Packing correctness", "Packed-batch report",
         [A("packed_samples_rebuild_bit_exact"), A("mask_and_position_invariants"), A("lane_packing_policy_applied"),
          A("ledger_token_counts_reconstruct"),
          C("packed_batch_report_rebuilds", all(s["seq_hash_matches_ledger"] for s in pbr["samples"]),
            f"step {pbr['step']} batch {pbr['batch_id']} rebuilt sample by sample")],
         ["reports/packed_batch_report.json", "reports/audit_report.json#packing", "reports/packing_policy_lab.json"],
         {"utilization": audit["packing"]["utilization"], "loss_bearing_fraction": audit["packing"]["loss_bearing_fraction"]}),
        ("mixture_compliance", "Mixture compliance", "Planned versus actual shares",
         [B("mixture_compiled"), B("protected_floors_in_every_step_quota"), A("actual_shares_equal_compiled_quotas"),
          A("planned_vs_actual_shares_within_tolerance"),
          A("protected_floors_held_every_step"), A("anneal_reserve_fenced")],
         ["manifests/mixture_schedule.json", "reports/audit_report.json#mixture"],
         {r["stage"]: {"max_abs_deviation": r["max_abs_deviation"]} for r in audit["mixture"]["stages"]}),
        ("opus_audit_trail", "OPUS audit trail", "Candidate decision records",
         [A("every_consumed_sample_has_accepted_decision"), A("all_decision_kinds_exercised"),
          A("rejected_or_deferred_candidates_never_consumed"), A("decision_records_complete")],
         [f"ledgers/main/opus.jsonl#seq={v['seq']} ({k})" for k, v in ex.items()],
         {"status": audit["opus"]["status"], "reasons": audit["opus"]["reasons"],
          "protected_floor_overrides": audit["opus"]["protected_floor_overrides"]}),
        ("consumption_ledger", "Consumption ledger and checkpoints", "Hash-chained ledger + checkpoint offsets",
         [A("ledger_hash_chains_valid"), A("committed_stream_has_no_gaps_or_repeats"), A("every_step_complete"),
          A("checkpoints_bound_to_ledger_offsets")],
         ["ledgers/main/consumption.jsonl", "checkpoints/main/", "reports/audit_report.json#checkpoints"],
         audit["stream"]),
        ("crash_recovery", "Crash recovery", "Expected and resumed batch ids",
         [C("resume_next_batch_matched", all(resume["checks"].values()),
            f"expected step {resume['expected_next_step']}: crashed-process plan {exp_c.get('batch_id')}, reference "
            f"{exp_r.get('batch_id')}, resumed {resumed['batch_id']}"),
          C("rolled_back_microbatches_reconsumed_identically",
            resume["rolled_back_microbatches_reconsumed_identically"] and all(x["match"] for x in resume["rolled_back_microbatches_reconsumed_identically"]),
            f"{len(resume['rolled_back_microbatches_reconsumed_identically'])} microbatches consumed before the crash were re-served bit-identically"),
          A("resumed_stream_equals_uninterrupted_run")],
         ["reports/resume_report.json", f"ledgers/main/consumption.jsonl#seq={exp_c.get('seq')} (plan written before the crash)",
          "ledgers/reference/consumption.jsonl"],
         {"resumed_from": resume["resumed_from"], "expected_batch_id": exp_c.get("batch_id"),
          "reference_batch_id": exp_r.get("batch_id"), "resumed_batch_id": resumed["batch_id"],
          "expected_batch_hash": exp_c.get("batch_hash"), "resumed_batch_hash": resumed["batch_hash"]}),
        ("replay", "Replay", "Original and replay hashes",
         [C("replay_hash_matched", replay["all_match"], f"{len(replay['steps'])} steps {replay['interval']} rebuilt from ledger span refs"),
          A("replay_ledger_matches_original")],
         ["reports/replay_report.json", f"ledgers/{replay['branch']}/consumption.jsonl"],
         {"interval": replay["interval"], "steps": [{k: r[k] for k in ("global_step", "orig_batch_hash", "replay_batch_hash",
                                                                      "orig_weights_hash", "replay_weights_hash")}
                                                   for r in replay["steps"]]}),
        ("fork", "Fork from an earlier checkpoint", "Branch-forked record",
         [C("fork_diverged_explicitly", all(fork["checks"].values()), f"{fork['branch']} from {fork['parent_checkpoint']}"),
          A("fork_lineage_explicit")],
         ["reports/fork_report.json", f"ledgers/{fork['branch']}/consumption.jsonl#seq=0",
          f"manifests/mixture_schedule.{fork['branch']}.json"],
         {"branch": fork["branch"], "divergence_step": fork["divergence_step"], "overrides": fork["overrides"]}),
        ("learning_trace", "Learning trace", "Loss linked to source data",
         [A("every_consumed_sample_has_learning_record"), A("token_trace_reconciles_with_ledger"),
          A("shard_report_cards_written")],
         [f"ledgers/main/learning.jsonl#seq={lrec['seq']}", lrec["token_trace"], "reports/learning_report.json"],
         {"example": {"sample_id": lrec["sample_id"], "loss_before": lrec["loss_before"], "loss_after": lrec["loss_after"],
                      "spans": lrec["spans"][:2]}, "classification_counts": audit["learning"]["classification_counts"]}),
        ("throughput", "Throughput", "Performance report",
         [C("throughput_reconstructs_from_ledger", perf["reconciliation"]["all_match"],
            f"positions/non-pad/loss-bearing counts equal the audit recount {perf['reconciliation']['ledger_recount']}"),
          C("packing_utilization_at_least_0.90", m["packing_utilization"] >= 0.90, f"utilization {m['packing_utilization']}"),
          C("beats_pad_only_baseline", m["packing_utilization"] > (m["pad_only_baseline_utilization"] or 1),
            f"{m['packing_utilization']} vs pad-only {m['pad_only_baseline_utilization']} (x{m['utilization_gain_vs_pad_only']})"),
          C("useful_tokens_per_second_measured", m["useful_loss_bearing_tokens_per_s"] > 0,
            f"{m['useful_loss_bearing_tokens_per_s']} loss-bearing tok/s = {m['loss_bearing_tokens']} / {m['wall_s']} s")],
         ["performance.json", "perf/"],
         {k: m[k] for k in ("raw_tokens_per_s", "useful_loss_bearing_tokens_per_s", "accepted_tokens_per_s_after_opus",
                            "packing_utilization", "loader_wait_fraction", "resume_latency_s", "replay_latency_s")}),
    ]
    if tests is not None:
        reqs.append(("tests", "Automated invariant tests", "unittest run",
                     [C("unit_tests_passed", tests["passed"], f"{tests['ran']} tests, failures={tests['failures']}, errors={tests['errors']}")],
                     ["reports/tests.log"], tests))

    out_reqs = []
    for rid, title, ev_label, checks, paths, values in reqs:
        out_reqs.append({"id": rid, "requirement": title, "result": "PASS" if all(c["passed"] for c in checks) else "FAIL",
                         "evidence_label": ev_label, "evidence": paths, "checks": checks, "key_values": values})
    overall = "PASS" if all(r["result"] == "PASS" for r in out_reqs) else "FAIL"
    ev = {"format": "tdes-evidence-v1", "generated_by": "tdes.evidence.build_evidence (from reports written by the run)",
          "run_id": "v5s06-" + sha256_json(cfg)[:10], "config_hash": sha256_json(cfg),
          "tokenizer_hash": lock["tokenizer_hash"], "overall": overall, "requirements": out_reqs,
          "command": "python run_demo.py"}
    write_json(os.path.join(art, "evidence.json"), ev)

    md = ["# Evidence summary", "", f"Run `{ev['run_id']}` | tokenizer `{lock['tokenizer_hash'][:16]}` | overall **{overall}**",
          "", "Generated by `tdes/evidence.py` from the reports, ledgers and manifests in this folder.", "",
          "| Requirement | Result | Evidence |", "|---|---|---|"]
    for r in out_reqs:
        md.append(f"| {r['requirement']} | {r['result']} | {r['evidence_label']}: `{r['evidence'][0]}` |")
    md += ["", "## Key facts", "",
           f"- **Crash recovery:** crashed inside step {resume['expected_next_step']} after checkpoint `{resume['resumed_from']}`. "
           f"Expected next batch `{exp_c.get('batch_id')}` (planned by the crashed process) and `{exp_r.get('batch_id')}` "
           f"(uninterrupted reference run); resumed batch `{resumed['batch_id']}`.",
           f"- **Replay:** steps {replay['interval'][0]}-{replay['interval'][1]} rebuilt from ledger span refs. "
           f"Batch hashes, token spans, sequence and loss-mask hashes and post-step weights all identical: {replay['all_match']}. "
           f"First step original `{replay['steps'][0]['orig_batch_hash'][:16]}` vs replay `{replay['steps'][0]['replay_batch_hash'][:16]}`.",
           f"- **Fork:** `{fork['branch']}` from `{fork['parent_checkpoint']}`, diverging at step {fork['divergence_step']} "
           f"(overrides: {', '.join(fork['overrides'])}).",
           f"- **Firewall:** {audit['firewall']['scanned_tokens']} consumed tokens re-scanned, {len(audit['firewall']['violations'])} violations; "
           f"test/validation shards blocked at pool entry, a benchmark-bearing batch blocked at serve time, one leaked web shard quarantined.",
           f"- **OPUS:** {audit['opus']['status']} with {audit['opus']['protected_floor_overrides']} protected-floor overrides.",
           f"- **Mixture:** " + "; ".join(f"{r['stage']} max deviation {r['max_abs_deviation']:.4f}" for r in audit["mixture"]["stages"]) + ".",
           f"- **Throughput:** utilization {m['packing_utilization']:.4f} (pad-only {m['pad_only_baseline_utilization']:.4f}); "
           f"{m['useful_loss_bearing_tokens_per_s']:.0f} useful loss-bearing tok/s, {m['raw_tokens_per_s']:.0f} raw tok/s on CPU.",
           "", "## Checks", ""]
    for r in out_reqs:
        md.append(f"**{r['requirement']}** ({r['result']})")
        md.append("")
        for c in r["checks"]:
            md.append(f"- {'PASS' if c['passed'] else 'FAIL'} `{c['check']}`: {c['detail']}")
        md.append("")
    with open(os.path.join(art, "evidence.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    log.event("evidence bundle written", f"overall {overall}: " + ", ".join(f"{r['id']}={r['result']}" for r in out_reqs))
    return ev

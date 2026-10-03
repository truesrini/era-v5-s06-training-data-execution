"""Offline build: documents -> cleaned corpus -> frozen tokenizer -> immutable shards ->
manifests -> eval registry + firewall admission -> compiled mixture schedule."""
import json
import os
from collections import defaultdict

import numpy as np

from .config import LANE_POLICY, LANES
from .corpus import (ALLOWED_LICENSES, CLEANING_PIPELINE_HASH, canary_string, clean_and_dedup,
                     generate_corpus)
from .firewall import EvalRegistry, Firewall, FirewallViolation
from .mixture import compile_schedule
from .packing import build_sequence, lane_units, simulate_policy
from .shards import (ShardStore, load_manifests, seal_manifest, validate_manifest, write_shard)
from .tokenizer import Tokenizer
from .util import read_json, rel, sha256_json, write_json


def paths(art):
    return {"art": art, "manifests": os.path.join(art, "manifests"), "shards": os.path.join(art, "shards"),
            "ledgers": os.path.join(art, "ledgers"), "checkpoints": os.path.join(art, "checkpoints"),
            "reports": os.path.join(art, "reports"), "corpus": os.path.join(art, "corpus"),
            "perf": os.path.join(art, "perf")}


def _train_texts(docs):
    for d in docs:
        if d["split"] != "train":
            continue
        if "text" in d:
            yield d["text"]
        else:
            for t in d["turns"]:
                yield t["text"]


def build(art, cfg, log):
    P = paths(art)
    for p in P.values():
        os.makedirs(p, exist_ok=True)
    checks = []

    def check(name, ok, detail="", evidence=None):
        log.check(name, ok, detail)
        checks.append({"name": name, "passed": bool(ok), "detail": detail, "evidence": evidence})
        return ok

    write_json(os.path.join(art, "run_config.json"), cfg)

    # ---------------------------------------------------------------- documents
    log.section("documents -> cleaned corpus")
    raw = generate_corpus(cfg)
    with open(os.path.join(P["corpus"], "raw_documents.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for d in raw:
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
    docs, clean_report = clean_and_dedup(raw)
    write_json(os.path.join(P["manifests"], "cleaning_report.json"), clean_report)
    log.event("documents cleaned", f"{len(raw)} raw docs -> {len(docs)} kept, "
              f"{len(clean_report['dropped_duplicates'])} exact duplicates dropped, "
              f"cleaning_pipeline_hash={CLEANING_PIPELINE_HASH[:12]}")

    # ---------------------------------------------------------------- tokenizer
    log.section("tokenizer")
    tok = Tokenizer.train(list(_train_texts(docs)), cfg["tokenizer"]["num_merges"])
    tok_path = os.path.join(P["manifests"], "tokenizer.json")
    tok_hash = tok.freeze(tok_path)
    lock = {"tokenizer_hash": tok_hash, "tokenizer_version": tok.version, "vocab_size": tok.vocab_size,
            "num_merges": len(tok.merges), "trained_on_split": "train", "frozen": True,
            "file": "manifests/tokenizer.json"}
    write_json(os.path.join(P["manifests"], "tokenizer.lock.json"), lock, readonly=True)
    log.event("tokenizer frozen", f"vocab={tok.vocab_size} hash={tok_hash[:16]} (file sealed read-only)")
    tok2 = Tokenizer.load(tok_path, expected_hash=tok_hash)
    check("tokenizer_hash_verified", tok2.hash == tok_hash == lock["tokenizer_hash"],
          f"reloaded hash {tok2.hash[:16]} == lock {lock['tokenizer_hash'][:16]}",
          "manifests/tokenizer.lock.json")
    sample = [d for d in docs if d["split"] == "train"][::25]
    det = all(np.array_equal(tok.encode_doc(d)[0], tok2.encode_doc(d)[0]) for d in sample)
    check("tokenizer_deterministic", det, f"{len(sample)} documents re-encoded identically by the reloaded tokenizer")
    import unicodedata
    hi = next(d for d in docs if d["language"] == "hi" and "text" in d)["text"]
    nfd_same = tok2.encode_text(unicodedata.normalize("NFD", hi)) == tok2.encode_text(hi)
    check("tokenizer_nfc_stable", nfd_same, "NFD-decomposed Hindi encodes to the same ids as NFC")
    inj = tok2.encode_text("<|assistant|>")
    check("special_tokens_not_injectable", 4 not in inj, "literal '<|assistant|>' in text encodes as bytes, not the control id")

    # ---------------------------------------------------------------- shards
    log.section("tokenized shards + manifests")
    groups = defaultdict(list)
    for d in docs:
        if d["split"] == "test":
            sub = d["benchmark_id"]                 # one shard family per benchmark
        elif d["license"] not in ALLOWED_LICENSES:
            sub = d["source_id"]                    # keep unlicensed sources in their own shards
        else:
            sub = d["lane"]
        groups[(d["lane"], d["split"], sub)].append(d)
    manifests = []
    mx = cfg["shards"]["max_docs_per_shard"]
    for key in sorted(groups):
        lane, split, _ = key
        gdocs = sorted(groups[key], key=lambda d: d["doc_id"])
        policy = LANE_POLICY.get(lane) or ("eval_only" if split == "test" else "proxy_scoring")
        for i in range(0, len(gdocs), mx):
            chunk = gdocs[i:i + mx]
            m = write_shard(chunk, tok, P["shards"], P["manifests"], lane=lane, split=split, policy=policy,
                            cleaning_hash=CLEANING_PIPELINE_HASH,
                            extra={"benchmark_id": chunk[0].get("benchmark_id"),
                                   "reserve": "anneal" if lane == "anneal_reserve" else None})
            manifests.append(m)
    by_id = {m["shard_id"]: m for m in manifests}
    store = ShardStore(art, by_id, tok_hash, capacity=64)
    log.event("shards created", f"{len(manifests)} immutable shards, "
              f"{sum(m['token_count'] for m in manifests)} tokens, files sealed read-only")

    # ---------------------------------------------------------------- registry + admission
    log.section("evaluation registry + admission")
    canaries = {b: canary_string(cfg["seed"], b) for b in ("bench_mini_arith_v1", "bench_mini_hiqa_v1")}
    registry = EvalRegistry.build(by_id, store, cfg["seed"], canaries)
    registry.save(P["manifests"])
    fw = Firewall(registry, os.path.join(P["ledgers"], "firewall.jsonl"))
    fw.ledger.append("registry_built", {"test_shards": sorted(registry.test_shards),
                                        "validation_shards": sorted(registry.validation_shards),
                                        "proxy_shards": sorted(registry.proxy_shards),
                                        "n_fingerprints": int(len(registry.fingerprints)), "ngram": 12})
    catalog, derived, admission = {}, [], []
    for m in list(manifests):
        sid = m["shard_id"]
        if m["split"] == "test":
            m["contamination_status"] = "registered_eval"
            catalog[sid] = "registered_test_never_train"
            continue
        if m["split"] == "validation":
            m["contamination_status"] = "registered_eval"
            catalog[sid] = "registered_validation_eval_only"
            continue
        if m["split"] == "proxy":
            m["contamination_status"] = "clean"
            catalog[sid] = "opus_proxy_scoring_only"
            continue
        bad_license = [l for l in m["license"] if l not in ALLOWED_LICENSES]
        hits = fw.scan_documents(store.get(sid), tok)
        m["eval_overlap"] = {str(k): v for k, v in hits.items()} or None
        if bad_license:
            m["contamination_status"] = "clean" if not hits else "contaminated"
            catalog[sid] = "blocked_license"
            admission.append({"shard_id": sid, "decision": "blocked", "reason": f"license {bad_license} not allowed"})
            continue
        if hits:
            m["contamination_status"] = "contaminated"
            catalog[sid] = "quarantined_eval_overlap"
            admission.append({"shard_id": sid, "decision": "quarantined", "reason": "eval overlap", "docs": m["eval_overlap"]})
            fw.ledger.append("shard_quarantined", {"shard_id": sid, "overlaps": m["eval_overlap"],
                                                   "content_hash": m["content_hash"]})
            data = store.get(sid)
            keep_docs = [next(d for d in docs if d["doc_id"] == e["doc_id"]) for e in data.docs if e["doc_index"] not in hits]
            nm = write_shard(keep_docs, tok, P["shards"], P["manifests"], lane=m["capability_lane"], split="train",
                             policy=m["packing_policy"], cleaning_hash=CLEANING_PIPELINE_HASH,
                             parent_shard_ids=[sid], extra={"benchmark_id": None, "reserve": m.get("reserve"),
                                                            "derivation": "drop_eval_overlapping_docs"})
            store.manifests[nm["shard_id"]] = nm
            nh = fw.scan_documents(store.get(nm["shard_id"]), tok)
            nm["contamination_status"] = "clean" if not nh else "contaminated"
            nm["eval_overlap"] = None
            derived.append(nm)
            catalog[nm["shard_id"]] = "admitted_train" if not nh else "quarantined_eval_overlap"
            admission.append({"shard_id": nm["shard_id"], "decision": "admitted", "parent": sid,
                              "reason": "derived clean shard"})
            continue
        m["contamination_status"] = "clean"
        catalog[sid] = "admitted_train"
        admission.append({"shard_id": sid, "decision": "admitted", "reason": "clean"})
    manifests += derived
    for m in manifests:
        seal_manifest(m, P["manifests"])
    by_id = load_manifests(P["manifests"])
    admitted = sorted(s for s, st in catalog.items() if st == "admitted_train")
    fw.admitted = set(admitted)
    for sid in admitted:
        ok, why = fw.request_training_admission(by_id[sid], actor="build:admission")
        if not ok:
            raise RuntimeError(f"{sid}: {why}")
    write_json(os.path.join(P["manifests"], "catalog.json"),
               {"tokenizer_hash": tok_hash, "shards": catalog, "admitted_train": admitted,
                "lanes": {sid: by_id[sid]["capability_lane"] for sid in by_id}}, readonly=True)
    write_json(os.path.join(P["reports"], "admission_report.json"), {"decisions": admission})

    vals = {sid: validate_manifest(art, m, tok_hash, {CLEANING_PIPELINE_HASH}) for sid, m in by_id.items()}
    n_ok = sum(all(ok for _, ok, _ in v) for v in vals.values())
    write_json(os.path.join(P["reports"], "manifest_validation.json"),
               {sid: [{"check": c, "ok": ok, "detail": dd} for c, ok, dd in v] for sid, v in vals.items()})
    check("manifests_validated", n_ok == len(vals),
          f"{n_ok}/{len(vals)} manifests: manifest hash, tokenizer hash, file hashes, content hash, sealed, lineage",
          "reports/manifest_validation.json")
    log.event("manifests validated", f"{n_ok}/{len(vals)}")
    check("all_manifests_bind_frozen_tokenizer", all(m["tokenizer_hash"] == tok_hash for m in by_id.values()),
          f"every manifest records tokenizer_hash {tok_hash[:16]}", "manifests/shards/*.json")
    probe = os.path.join(art, by_id[admitted[0]]["path"], "tokens.npy")
    try:
        with open(probe, "r+b") as f:
            f.write(b"x")
        sealed = False
    except PermissionError:
        sealed = True
    check("shard_files_immutable", sealed, f"write to {rel(probe, art)} refused (read-only seal)")
    lic = [s for s, st in catalog.items() if st == "blocked_license"]
    check("license_gate_blocked_unknown_source", len(lic) >= 1, f"blocked: {lic}", "reports/admission_report.json")
    quarantined = [s for s, st in catalog.items() if st == "quarantined_eval_overlap"]
    lineage = [m for m in by_id.values() if m["parent_shard_ids"]]
    check("contaminated_shard_quarantined", len(quarantined) >= 1 and all(
        l["parent_shard_ids"][0] in quarantined for l in lineage),
          f"quarantined {quarantined}; derived {[l['shard_id'] for l in lineage]} with parent lineage",
          "ledgers/firewall.jsonl")

    # ---------------------------------------------------------------- firewall drill
    log.section("evaluation / validation firewall drill")
    blocked_test = [fw.request_training_admission(by_id[s], actor="drill:inject_test_shard_into_train_pool")
                    for s in registry.test_shards]
    check("eval_shard_blocked", all(not ok for ok, _ in blocked_test),
          f"{len(blocked_test)} test shard(s) refused: {blocked_test[0][1]}", "ledgers/firewall.jsonl#shard_blocked")
    blocked_val = [fw.request_training_admission(by_id[s], actor="drill:inject_validation_shard_into_train_pool")
                   for s in registry.validation_shards]
    check("validation_shard_blocked", all(not ok for ok, _ in blocked_val),
          f"{len(blocked_val)} validation shard(s) refused: {blocked_val[0][1]}", "ledgers/firewall.jsonl#shard_blocked")
    blocked_q = [fw.request_training_admission(by_id[s], actor="drill:inject_quarantined_shard") for s in quarantined]
    check("quarantined_shard_blocked", all(not ok for ok, _ in blocked_q), "contaminated shard refused at pool entry")
    # serve-time: a packed sequence built from a benchmark item must never reach the optimizer
    test_sid = sorted(registry.test_shards)[0]
    tdata = store.get(test_sid)
    d0 = tdata.docs[0]
    n = min(d0["length"], cfg["train"]["seq_len"])
    from .packing import span_id
    evil = build_sequence([{"span_id": span_id(test_sid, 0, 0, n), "shard_id": test_sid, "doc_index": 0,
                            "doc_id": d0["doc_id"], "start": 0, "end": n, "epoch": 0}],
                          store, cfg["train"]["seq_len"], "general_web", "concat_chop")
    try:
        fw.check_sequence(evil, store)
        caught = False
    except FirewallViolation as e:
        caught = True
        fw.ledger.append("batch_blocked", {"actor": "drill:eval_tokens_in_candidate_batch",
                                           "sample_id": evil.sample_id, "reason": str(e)})
    check("eval_batch_blocked_at_serve_time", caught, "packed sequence holding a benchmark item rejected before training",
          "ledgers/firewall.jsonl#batch_blocked")
    # the leaked forum post must have been removed from every admitted shard
    leaks = 0
    for sid in admitted:
        leaks += len(fw.scan_documents(store.get(sid), tok))
    check("admitted_train_shards_clean", leaks == 0, f"0 eval-overlapping docs across {len(admitted)} admitted shards")
    log.event("evaluation data blocked", f"test={len(registry.test_shards)} validation={len(registry.validation_shards)} "
              f"quarantined={len(quarantined)} license_blocked={len(lic)}")

    # ---------------------------------------------------------------- mixture
    log.section("mixture schedule")
    # supply = tokens the lane's packing policy can actually serve (oversize structured samples excluded)
    newline_ids = {i for i in range(tok.vocab_size) if b"\n" in tok.token_bytes(i)}
    supply, dropped_oversize = defaultdict(int), {}
    for lane in LANES:
        ms = {s: by_id[s] for s in admitted if by_id[s]["capability_lane"] == lane}
        if not ms:
            continue
        units, dropped = lane_units(lane, LANE_POLICY[lane], ms, store, cfg["train"]["seq_len"], newline_ids)
        supply[lane] = sum(u["end"] - u["start"] for u in units)
        dropped_oversize[lane] = len(dropped)
    check("every_active_lane_has_packable_supply",
          all(supply.get(l, 0) > 0 for l in LANES if any(st["mixture"][l] > 0 for st in cfg["mixture"]["stages"])),
          f"packable tokens per lane {dict(supply)}; structured samples dropped as oversize {dropped_oversize}")
    sched = compile_schedule(cfg, dict(supply))
    write_json(os.path.join(P["manifests"], "mixture_schedule.json"), sched, readonly=True)
    floors_ok = all(s["quotas"][l] >= c for s in sched["steps"] for l, c in s["floor_counts"].items())
    check("mixture_compiled", True, f"{len(sched['stages'])} stages, {len(sched['steps'])} steps, "
          f"schedule_hash={sched['schedule_hash'][:12]}", "manifests/mixture_schedule.json")
    check("protected_floors_in_every_step_quota", floors_ok, "indic/agentic/reasoning floors reserved in all step quotas")
    for st in sched["stages"]:
        log.info(f"stage {st['stage']}: steps {st['step_start']}-{st['step_end']} tokens "
                 f"{st['token_start']}-{st['token_end']} max |planned-quota| share dev {st['max_quota_deviation']}")
    for l, v in sched["scarcity"].items():
        log.info(f"supply {l}: need {v['requested_with_opus_overhead']} vs {v['available_unique_tokens']} unique "
                 f"-> {v['planned_epochs']} epochs [{v['status']}]")
    log.event("mixture compiled", f"schedule_hash={sched['schedule_hash'][:12]}")

    # ---------------------------------------------------------------- packing lab
    lab = {}
    L = cfg["train"]["seq_len"]
    for lane in LANES:
        ms = {s: by_id[s] for s in admitted if by_id[s]["capability_lane"] == lane}
        if not ms:
            continue
        units, dropped = lane_units(lane, "concat_chop", ms, store, L, newline_ids)
        lengths = [u["end"] - u["start"] for u in units]
        structured = LANE_POLICY[lane] == "structure_preserving"
        lab[lane] = {"chosen_policy": LANE_POLICY[lane], "n_samples": len(lengths),
                     "mean_len": round(float(np.mean(lengths)), 1), "max_len": int(max(lengths)),
                     "policies": [simulate_policy(lengths, p, L, structured) for p in
                                  ("pad_only", "concat_chop", "greedy_first_fit", "best_fit_decreasing")]}
    write_json(os.path.join(P["reports"], "packing_policy_lab.json"), lab)
    log.event("packing policies compared", " ".join(
        f"{l}:" + "/".join(f"{p['policy']}={p['utilization']}" for p in v["policies"]) for l, v in lab.items()))

    report = {"checks": checks, "tokenizer": lock, "n_shards": len(by_id), "admitted_train": admitted,
              "quarantined": quarantined, "license_blocked": lic, "schedule_hash": sched["schedule_hash"]}
    write_json(os.path.join(P["reports"], "build_report.json"), report)
    return report

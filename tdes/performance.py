"""performance.json: throughput and packing efficiency, with every rate traceable to counts in the
ledger and seconds measured by the training processes (rate = count / seconds, both reported)."""
import glob
import os

from .util import read_json, write_json


def _sum(dicts, key):
    return sum(d.get(key, 0) for d in dicts)


def build_performance(art, audit_report, log, main="main"):
    segs = [read_json(p) for p in sorted(glob.glob(os.path.join(art, "perf", "*.json")))]
    main_segs = [s for s in segs if s["branch"] == main]
    counts = [s["counts"] for s in main_segs]
    timings = {}
    for s in main_segs:
        for k, v in s["timings_s"].items():
            timings[k] = timings.get(k, 0.0) + v
    wall = sum(s["wall_s"] for s in main_segs)
    positions, nonpad, loss = _sum(counts, "positions"), _sum(counts, "nonpad_tokens"), _sum(counts, "loss_tokens")
    accepted = _sum(counts, "accepted_tokens")
    cand_tok = _sum(counts, "candidate_tokens")
    steps = _sum(counts, "steps")
    loader_wait = timings.get("loader_total", 0.0)
    pk = audit_report["packing"]
    lab = read_json(os.path.join(art, "reports", "packing_policy_lab.json"))
    pad_pos = sum(next(p for p in v["policies"] if p["policy"] == "pad_only")["positions"] for v in lab.values())
    pad_tok = sum(next(p for p in v["policies"] if p["policy"] == "pad_only")["real_tokens"] for v in lab.values())
    pad_only_util = pad_tok / pad_pos if pad_pos else None
    resume = read_json(os.path.join(art, "reports", "resume_report.json"))
    replay = read_json(os.path.join(art, "reports", "replay_report.json"))
    decisions, rej_lane, cand_lane = {}, {}, {}
    for c in counts:
        for k, v in c["decisions"].items():
            decisions[k] = decisions.get(k, 0) + v
        for k, v in c["rejections_by_lane"].items():
            rej_lane[k] = rej_lane.get(k, 0) + v
        for k, v in c["candidates_by_lane"].items():
            cand_lane[k] = cand_lane.get(k, 0) + v
    store = {"cache_hits": _sum([s["store"] for s in main_segs], "cache_hits"),
             "cache_misses": _sum([s["store"] for s in main_segs], "cache_misses")}
    store["cache_hit_rate"] = round(store["cache_hits"] / max(1, store["cache_hits"] + store["cache_misses"]), 4)
    reads = [s["store"]["mean_shard_read_ms"] for s in main_segs if s["store"]["mean_shard_read_ms"] is not None]
    recon = {"positions": positions == pk["totals"]["positions"], "nonpad_tokens": nonpad == pk["totals"]["nonpad_tokens"],
             "loss_tokens": loss == pk["totals"]["loss_tokens"]}
    perf = {
        "definitions": {
            "raw_tokens_per_s": "token positions processed by the optimizer (incl. padding) / train-loop wall seconds",
            "useful_loss_bearing_tokens_per_s": "positions with loss_mask=1 / train-loop wall seconds",
            "accepted_tokens_per_s_after_opus": "non-pad tokens of OPUS-accepted samples / wall seconds",
            "packing_utilization": "non-pad positions / all positions (recounted from shards by the audit)",
            "loader_wait_fraction": "share of wall time the trainer waited for the loader (packing + OPUS scoring); "
                                    "single-process analogue of GPU idle time",
        },
        "main_run": {
            "segments": [{"tag": s["tag"], "accounted_through_step": s.get("accounted_through_step"),
                          "steps": s["counts"]["steps"], "wall_s": s["wall_s"]} for s in main_segs],
            "steps": steps, "wall_s": round(wall, 4),
            "positions": positions, "nonpad_tokens": nonpad, "loss_bearing_tokens": loss,
            "raw_tokens_per_s": round(positions / wall, 2),
            "useful_loss_bearing_tokens_per_s": round(loss / wall, 2),
            "accepted_tokens_per_s_after_opus": round(accepted / wall, 2),
            "packing_utilization": round(nonpad / positions, 6),
            "loss_bearing_fraction": round(loss / positions, 6),
            "padding_fraction": round(1 - nonpad / positions, 6),
            "pad_only_baseline_utilization": round(pad_only_util, 6) if pad_only_util else None,
            "utilization_gain_vs_pad_only": round((nonpad / positions) / pad_only_util, 4) if pad_only_util else None,
            "loader_wait_s": round(loader_wait, 4), "loader_wait_fraction": round(loader_wait / wall, 4),
            "opus_scoring_s": round(timings.get("opus_scoring", 0.0), 4),
            "train_compute_s": round(timings.get("train_compute", 0.0), 4),
            "ledger_io_s": round(timings.get("ledger_io", 0.0), 4),
            "checkpoint_s": round(timings.get("checkpoint", 0.0), 4),
            "firewall_s": round(timings.get("firewall", 0.0), 4),
            "packing_verify_s": round(timings.get("packing_verify", 0.0), 4),
            "opus": {"candidate_tokens_scored": cand_tok, "decisions": decisions,
                     "acceptance_rate": round(decisions.get("accepted", 0) / max(1, sum(decisions.values())), 4),
                     "rejection_rate_by_lane": {k: round(rej_lane.get(k, 0) / v, 4) for k, v in sorted(cand_lane.items())}},
            "shard_cache": {**store, "mean_shard_read_ms": round(sum(reads) / len(reads), 3) if reads else None},
            "resume_latency_s": resume.get("resume_latency_s"),
            "replay_latency_s": replay.get("replay_latency_s"),
            "utilization_by_lane": pk["utilization_by_lane"],
        },
        "other_branches": [{"branch": s["branch"], "tag": s["tag"], "steps": s["counts"]["steps"], "wall_s": s["wall_s"],
                            "positions": s["counts"]["positions"], "loss_tokens": s["counts"]["loss_tokens"]}
                           for s in segs if s["branch"] != main],
        "reconciliation": {"perf_counts": {"positions": positions, "nonpad_tokens": nonpad, "loss_tokens": loss},
                           "ledger_recount": pk["totals"], "match": recon, "all_match": all(recon.values()),
                           "note": "perf counts come from the training processes; the audit recounts the same "
                                   "quantities by rebuilding every consumed sample from its span refs"},
    }
    write_json(os.path.join(art, "performance.json"), perf)
    m = perf["main_run"]
    log.event("performance measured", f"util={m['packing_utilization']:.4f} (pad-only {m['pad_only_baseline_utilization']:.4f}), "
              f"raw={m['raw_tokens_per_s']:.0f} tok/s, useful loss-bearing={m['useful_loss_bearing_tokens_per_s']:.0f} tok/s, "
              f"accepted={m['accepted_tokens_per_s_after_opus']:.0f} tok/s, loader wait {m['loader_wait_fraction']:.1%}, "
              f"resume latency {m['resume_latency_s']}s")
    log.check("throughput_reconstructs_from_ledger", perf["reconciliation"]["all_match"],
              f"perf counts {perf['reconciliation']['perf_counts']} == ledger recount {pk['totals']}")
    return perf

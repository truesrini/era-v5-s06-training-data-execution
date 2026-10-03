"""Run configuration. Lane names and protected floors follow the Session 5 (Ex05) V5 mixture
spec; the scale is shrunk so the whole system runs on a laptop CPU in about a minute."""
import copy

# Capability lanes (Ex05: web / code / indic / math / reasoning / agentic + anneal reserve).
LANES = ["general_web", "code", "math_science", "indic", "reasoning", "agentic", "anneal_reserve"]

# Lanes that OPUS may never starve: the floor is enforced above the selector (Ex05 sec. 6).
PROTECTED_LANES = ["indic", "agentic", "reasoning"]

# Packing policy per data type (Session 6 sec. 5).
LANE_POLICY = {
    "general_web": "concat_chop",           # plain pretraining: EOS-joined stream cut into windows
    "math_science": "concat_chop",
    "indic": "concat_chop",
    "code": "best_fit_chunked",             # files split only at line boundaries, best-fit bins
    "reasoning": "structure_preserving",    # whole samples, never split, isolated attention
    "agentic": "structure_preserving",
    "anneal_reserve": "structure_preserving",
}

ATTENTION_POLICY = "segment_block_causal"   # a token attends only to earlier tokens of its own segment
POSITION_POLICY = "reset_per_segment"       # position ids restart at 0 for every packed segment

DEFAULT_CONFIG = {
    "seed": 20260801,
    "corpus": {"scale": 1.0},
    "tokenizer": {"num_merges": 384},
    "shards": {"max_docs_per_shard": 24},
    "model": {"d_model": 48, "d_ff": 96, "init_std": 0.05, "dtype": "float32"},
    "train": {
        "seq_len": 128,
        "world_size": 2,          # simulated data-parallel ranks
        "micro_batch": 2,         # sequences per rank per microbatch
        "grad_accum": 4,          # microbatches per rank per optimizer step  -> 16 sequences / step
        "total_steps": 32,
        "checkpoint_every": 4,
        "lr": 1e-2,
        "warmup_steps": 4,
        "min_lr_frac": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "packing_buffer": 8,
        "device": "cpu",          # or "cuda" (python run_demo.py --device cuda)
    },
    "opus": {
        "enabled": True,
        "proxy_lanes": ["general_web", "code", "math_science"],   # deliberately English/code biased
        "proxy_sequences": 4,
        "reject_quantile": 0.30,     # tau = this quantile of the step's candidate scores
        "extra_candidates_frac": 0.5,
        "max_defer": 2,
        "refill_rounds": 2,
        "dup_window_steps": 4,
    },
    "mixture": {
        "warmup_steps": 2,
        "max_epochs": 10.0,
        "stages": [
            {"stage": "foundation", "end_frac": 0.375,
             "mixture": {"general_web": 0.32, "code": 0.20, "math_science": 0.16, "indic": 0.18,
                         "reasoning": 0.07, "agentic": 0.07, "anneal_reserve": 0.0},
             "protected_floors": {"indic": 0.125, "agentic": 0.0625, "reasoning": 0.0625}},
            {"stage": "reasoning_midtrain", "end_frac": 0.75,
             "mixture": {"general_web": 0.22, "code": 0.20, "math_science": 0.16, "indic": 0.18,
                         "reasoning": 0.14, "agentic": 0.10, "anneal_reserve": 0.0},
             "protected_floors": {"indic": 0.125, "agentic": 0.0625, "reasoning": 0.0625}},
            {"stage": "anneal", "end_frac": 1.0,
             "mixture": {"general_web": 0.0, "code": 0.12, "math_science": 0.14, "indic": 0.20,
                         "reasoning": 0.14, "agentic": 0.10, "anneal_reserve": 0.30},
             "protected_floors": {"indic": 0.125, "agentic": 0.0625, "reasoning": 0.0625}},
        ],
    },
    "demo": {
        "crash_at_step": 21,              # dies inside step 21, after the step-20 checkpoint
        "crash_after_microbatches": 3,
        "replay_from_step": 8,            # restore the step-8 checkpoint ...
        "replay_to_step": 16,             # ... and re-feed steps 9..16 from the ledger
        "fork_from_step": 12,
        "fork_steps": 8,
        "fork_overrides": {
            "mixture.stages.1.mixture": {"general_web": 0.12, "code": 0.20, "math_science": 0.16,
                                          "indic": 0.28, "reasoning": 0.14, "agentic": 0.10,
                                          "anneal_reserve": 0.0},
            "opus.reject_quantile": 0.40,
        },
    },
}


def default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def small_config() -> dict:
    """A shrunken configuration used by the integration tests."""
    cfg = default_config()
    cfg["corpus"]["scale"] = 0.35
    cfg["tokenizer"]["num_merges"] = 320
    cfg["model"].update({"d_model": 16, "d_ff": 32})
    cfg["train"].update({"micro_batch": 2, "grad_accum": 2, "total_steps": 10,
                         "checkpoint_every": 2, "warmup_steps": 2})
    cfg["demo"].update({"crash_at_step": 7, "crash_after_microbatches": 1, "replay_from_step": 2,
                        "replay_to_step": 6, "fork_from_step": 4, "fork_steps": 3})
    return cfg


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dotted-path overrides, e.g. {"opus.reject_quantile": 0.4}."""
    out = copy.deepcopy(cfg)
    for path, value in overrides.items():
        node = out
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = copy.deepcopy(value)
        else:
            node[last] = copy.deepcopy(value)
    return out


def global_batch(cfg: dict) -> int:
    t = cfg["train"]
    return t["world_size"] * t["micro_batch"] * t["grad_accum"]

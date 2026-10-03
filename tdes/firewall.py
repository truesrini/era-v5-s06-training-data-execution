"""Evaluation registry and the train/validation/test firewall (Session 6 sec. 13).

Test shards are *registered* (content hashes, benchmark ids, version tags, n-gram
contamination fingerprints, canary strings, never-train flag, access log) precisely so they can
be kept out of training. Validation shards may be read for evaluation but never become
gradient-bearing. The firewall is enforced three times:
  1. admission  - a training shard whose documents overlap the registry is quarantined;
  2. pool entry - a never-train shard can never be registered as a training stream;
  3. serve time - every packed sequence is re-checked (shard permission + fingerprints)
                  immediately before it reaches the optimizer.
"""
import os

import numpy as np

from .util import Ledger, array_hash, read_json, write_json

NGRAM = 12
_P = np.uint64(1099511628211)


def ngram_hashes(tokens: np.ndarray, n: int = NGRAM) -> np.ndarray:
    """Rolling polynomial hashes (mod 2^64) of every n-gram of a token array."""
    t = np.asarray(tokens, dtype=np.uint64)
    if len(t) < n:
        return np.empty(0, dtype=np.uint64)
    m = len(t) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _P + t[j:j + m] + np.uint64(1)
    return h


class EvalRegistry:
    def __init__(self, test_shards, validation_shards, fingerprints, canaries, validation_doc_hashes, proxy_shards):
        self.test_shards = test_shards                    # shard_id -> record
        self.validation_shards = validation_shards        # shard_id -> record
        self.fingerprints = np.unique(np.asarray(fingerprints, dtype=np.uint64))
        self.canaries = canaries                          # benchmark_id -> canary string
        self.validation_doc_hashes = set(validation_doc_hashes)
        self.proxy_shards = proxy_shards

    @classmethod
    def build(cls, manifests, store, seed, canaries):
        test, val, proxy, fps, vhashes = {}, {}, {}, [], []
        for sid, m in manifests.items():
            if m["split"] == "test":
                data = store.get(sid)
                for d in data.docs:
                    fps.append(ngram_hashes(data.doc_slice(d["doc_index"])[0]))
                test[sid] = {"shard_id": sid, "content_hash": m["content_hash"],
                             "benchmark_ids": sorted({d["benchmark_id"] for d in data.docs}),
                             "version_tags": ["v1"], "never_train": True, "n_items": len(data.docs)}
            elif m["split"] == "validation":
                data = store.get(sid)
                vhashes.extend(d["doc_hash"] for d in data.docs)
                val[sid] = {"shard_id": sid, "content_hash": m["content_hash"], "lane": m["capability_lane"],
                            "never_train": True, "permission": "eval_read_only"}
            elif m["split"] == "proxy":
                proxy[sid] = {"shard_id": sid, "content_hash": m["content_hash"], "permission": "opus_scoring_only"}
        fps = np.concatenate(fps) if fps else np.empty(0, dtype=np.uint64)
        return cls(test, val, fps, canaries, vhashes, proxy)

    def save(self, manifests_dir):
        np.save(os.path.join(manifests_dir, "eval_fingerprints.npy"), self.fingerprints)
        blob = {"ngram": NGRAM, "hash": "poly64(p=1099511628211)",
                "test_shards": self.test_shards, "validation_shards": self.validation_shards,
                "proxy_shards": self.proxy_shards, "canaries": self.canaries,
                "n_fingerprints": int(len(self.fingerprints)),
                "fingerprints_sha256": array_hash(self.fingerprints),
                "validation_doc_hashes": sorted(self.validation_doc_hashes)}
        write_json(os.path.join(manifests_dir, "eval_registry.json"), blob, readonly=True)

    @classmethod
    def load(cls, manifests_dir):
        blob = read_json(os.path.join(manifests_dir, "eval_registry.json"))
        fps = np.load(os.path.join(manifests_dir, "eval_fingerprints.npy"))
        if array_hash(fps) != blob["fingerprints_sha256"]:
            raise ValueError("eval fingerprint file does not match registry")
        return cls(blob["test_shards"], blob["validation_shards"], fps, blob["canaries"],
                   blob["validation_doc_hashes"], blob["proxy_shards"])

    def never_train_ids(self):
        return set(self.test_shards) | set(self.validation_shards) | set(self.proxy_shards)


class FirewallViolation(Exception):
    pass


class Firewall:
    def __init__(self, registry: EvalRegistry, ledger_path: str, admitted_train_ids=()):
        self.registry = registry
        self.ledger = Ledger(ledger_path)
        self.admitted = set(admitted_train_ids)
        self.ngrams_checked = 0
        self.sequences_checked = 0

    # ---------------------------------------------------------------- admission
    def scan_documents(self, data, tokenizer):
        """Return {doc_index: [reasons]} for documents overlapping the registry."""
        hits = {}
        fp = self.registry.fingerprints
        for d in data.docs:
            reasons = []
            toks = data.doc_slice(d["doc_index"])[0]
            ng = ngram_hashes(toks)
            n_hit = int(np.isin(ng, fp).sum()) if len(ng) and len(fp) else 0
            if n_hit:
                reasons.append(f"test_ngram_overlap:{n_hit}")
            text = tokenizer.decode(toks)
            for bench, canary in self.registry.canaries.items():
                if canary in text:
                    reasons.append(f"canary:{bench}")
            if d["doc_hash"] in self.registry.validation_doc_hashes:
                reasons.append("validation_exact_duplicate")
            if reasons:
                hits[d["doc_index"]] = reasons
        return hits

    # ---------------------------------------------------------------- pool entry
    def request_training_admission(self, manifest, actor: str):
        """Called whenever something tries to register a shard as a training stream."""
        sid = manifest["shard_id"]
        reason = None
        if sid in self.registry.test_shards or manifest["split"] == "test":
            reason = "test_shard_never_train"
        elif sid in self.registry.validation_shards or manifest["split"] == "validation":
            reason = "validation_shard_never_gradient_bearing"
        elif sid in self.registry.proxy_shards or manifest["split"] == "proxy":
            reason = "opus_proxy_scoring_only"
        elif manifest.get("contamination_status") != "clean":
            reason = f"contamination_status={manifest.get('contamination_status')}"
        if reason:
            self.ledger.append("shard_blocked", {"shard_id": sid, "split": manifest["split"], "actor": actor,
                                                 "reason": reason, "content_hash": manifest["content_hash"]})
            return False, reason
        return True, "admitted"

    # ---------------------------------------------------------------- serve time
    def check_sequence(self, seq, store):
        """Raise FirewallViolation unless every span of the packed sequence comes from an
        admitted training shard and no loss-bearing span carries an eval fingerprint."""
        self.sequences_checked += 1
        for sp in seq.spans:
            sid = sp["shard_id"]
            if sid in self.registry.never_train_ids() or sid not in self.admitted:
                raise FirewallViolation(f"shard {sid} is not an admitted training shard")
            toks = store.get(sid).doc_slice(sp["doc_index"], sp["start"], sp["end"])[0]
            ng = ngram_hashes(toks)
            self.ngrams_checked += len(ng)
            if len(ng) and np.isin(ng, self.registry.fingerprints).any():
                raise FirewallViolation(f"span {sp['span_id']} overlaps evaluation fingerprints")
        return True

    def log_validation_read(self, shard_ids, branch, step, purpose="eval_loss_no_grad"):
        self.ledger.append("validation_read", {"shard_ids": sorted(shard_ids), "branch": branch, "step": step,
                                               "purpose": purpose, "gradient": False})

    def log_proxy_read(self, shard_ids, branch, step):
        self.ledger.append("proxy_read", {"shard_ids": sorted(shard_ids), "branch": branch, "step": step,
                                          "purpose": "opus_proxy_gradient", "gradient_applied": False})

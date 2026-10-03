"""Immutable tokenized shards, their manifests, admission validation and a verifying,
caching shard store.

A shard is three files (tokens.npy, train_on.npy, index.json). Its content hash covers all
three; its id embeds the hash; every file is sealed read-only after writing. Changing a
shard therefore means writing a *new* shard with a new id and `parent_shard_ids` lineage.
"""
import os
import time
from collections import OrderedDict

import numpy as np

from .util import (canonical_json, is_readonly, make_readonly, read_json, sha256_file,
                   sha256_json, write_json)
import hashlib

SHARD_FORMAT = "tdes-shard-v1"


def content_hash(tokens: np.ndarray, train_on: np.ndarray, index: dict) -> str:
    h = hashlib.sha256()
    h.update(SHARD_FORMAT.encode())
    h.update(canonical_json(index).encode("utf-8"))
    h.update(np.ascontiguousarray(tokens, dtype="<u2").tobytes())
    h.update(np.ascontiguousarray(train_on, dtype=np.uint8).tobytes())
    return h.hexdigest()


def write_shard(docs, tokenizer, shards_root, manifests_dir, *, lane, split, policy,
                cleaning_hash, parent_shard_ids=(), extra=None, encoded=None):
    """Tokenize `docs`, write and seal the shard, write and seal its manifest. Returns manifest."""
    toks, flags, entries = [], [], []
    offset = 0
    for i, d in enumerate(docs):
        ids, tr = encoded[i] if encoded is not None else tokenizer.encode_doc(d)
        entries.append({"doc_index": i, "doc_id": d["doc_id"], "source_id": d["source_id"],
                        "kind": d["kind"], "language": d["language"], "script": d["script"],
                        "start": offset, "length": int(len(ids)), "doc_hash": d["doc_hash"],
                        "benchmark_id": d.get("benchmark_id")})
        toks.append(ids)
        flags.append(tr)
        offset += len(ids)
    tokens = np.concatenate(toks).astype(np.uint16)
    train_on = np.concatenate(flags).astype(np.uint8)
    index = {"format": SHARD_FORMAT, "tokenizer_hash": tokenizer.hash, "docs": entries}
    chash = content_hash(tokens, train_on, index)
    shard_id = f"shd-{lane}-{split}-{chash[:12]}"
    sdir = os.path.join(shards_root, shard_id)
    os.makedirs(sdir, exist_ok=True)
    files = {}
    for name, arr in (("tokens.npy", tokens), ("train_on.npy", train_on)):
        p = os.path.join(sdir, name)
        np.save(p, arr)
        make_readonly(p)
        files[name] = sha256_file(p)
    write_json(os.path.join(sdir, "index.json"), index, readonly=True)
    files["index.json"] = sha256_file(os.path.join(sdir, "index.json"))

    langs = sorted({d["language"] for d in docs})
    scripts = sorted({d["script"] for d in docs})
    licenses = sorted({d["license"] for d in docs})
    tiers = sorted({d["provenance_tier"] for d in docs})
    manifest = {
        "manifest_format": "tdes-manifest-v1",
        "shard_id": shard_id,
        "split": split,
        "capability_lane": lane,
        "packing_policy": policy,
        "source_ids": sorted({d["source_id"] for d in docs}),
        "doc_ids": [d["doc_id"] for d in docs],
        "n_docs": len(docs),
        "token_count": int(len(tokens)),
        "loss_bearing_token_count": int(train_on.sum()),
        "language": langs,
        "script": scripts,
        "license": licenses,
        "provenance_tier": tiers,
        "tokenizer_hash": tokenizer.hash,
        "tokenizer_version": tokenizer.version,
        "cleaning_pipeline_hash": cleaning_hash,
        "dedup_status": "exact_dedup_v1",
        "contamination_status": "pending_scan",
        "eval_overlap": None,
        "content_hash": chash,
        "parent_shard_ids": list(parent_shard_ids),
        "never_train": split in ("validation", "test"),
        "files": files,
        "path": f"shards/{shard_id}",
    }
    if extra:
        manifest.update(extra)
    return manifest


def manifest_path(manifests_dir, shard_id):
    return os.path.join(manifests_dir, "shards", f"{shard_id}.json")


def seal_manifest(manifest, manifests_dir):
    """Manifests are written once, after the contamination scan has filled its fields."""
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    manifest["manifest_hash"] = sha256_json(body)
    write_json(manifest_path(manifests_dir, manifest["shard_id"]), manifest, readonly=True)
    return manifest


def load_manifests(manifests_dir):
    d = os.path.join(manifests_dir, "shards")
    out = {}
    for name in sorted(os.listdir(d)):
        m = read_json(os.path.join(d, name))
        out[m["shard_id"]] = m
    return out


class ShardData:
    __slots__ = ("shard_id", "tokens", "train_on", "index", "docs")

    def __init__(self, shard_id, tokens, train_on, index):
        self.shard_id, self.tokens, self.train_on, self.index = shard_id, tokens, train_on, index
        self.docs = index["docs"]

    def doc_slice(self, doc_index, start=0, end=None):
        d = self.docs[doc_index]
        s = d["start"] + start
        e = d["start"] + (d["length"] if end is None else end)
        return self.tokens[s:e], self.train_on[s:e]


class ShardIntegrityError(Exception):
    pass


def read_shard(root, manifest, verify=True) -> ShardData:
    sdir = os.path.join(root, manifest["path"])
    tokens = np.load(os.path.join(sdir, "tokens.npy"))
    train_on = np.load(os.path.join(sdir, "train_on.npy"))
    index = read_json(os.path.join(sdir, "index.json"))
    if verify:
        actual = content_hash(tokens, train_on, index)
        if actual != manifest["content_hash"]:
            raise ShardIntegrityError(f"{manifest['shard_id']}: content hash {actual[:12]} != manifest {manifest['content_hash'][:12]}")
    return ShardData(manifest["shard_id"], tokens, train_on, index)


def validate_manifest(root, manifest, tokenizer_hash, known_cleaning_hashes):
    """Re-derive everything a manifest claims. Returns a list of (check, ok, detail)."""
    checks = []
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    checks.append(("manifest_hash", sha256_json(body) == manifest.get("manifest_hash"), ""))
    checks.append(("tokenizer_hash", manifest["tokenizer_hash"] == tokenizer_hash, manifest["tokenizer_hash"][:12]))
    sdir = os.path.join(root, manifest["path"])
    files_ok = all(sha256_file(os.path.join(sdir, f)) == h for f, h in manifest["files"].items())
    checks.append(("file_hashes", files_ok, ""))
    sealed = all(is_readonly(os.path.join(sdir, f)) for f in manifest["files"])
    checks.append(("sealed_read_only", sealed, ""))
    try:
        data = read_shard(root, manifest, verify=True)
        checks.append(("content_hash", True, manifest["content_hash"][:12]))
        checks.append(("id_embeds_hash", manifest["shard_id"].endswith(manifest["content_hash"][:12]), ""))
        checks.append(("token_count", int(len(data.tokens)) == manifest["token_count"], str(manifest["token_count"])))
        checks.append(("index_tokenizer_hash", data.index["tokenizer_hash"] == tokenizer_hash, ""))
        checks.append(("doc_ids", [d["doc_id"] for d in data.docs] == manifest["doc_ids"], ""))
    except ShardIntegrityError as e:
        checks.append(("content_hash", False, str(e)))
    checks.append(("cleaning_lineage", manifest["cleaning_pipeline_hash"] in known_cleaning_hashes, ""))
    return checks


class ShardStore:
    """Read-through LRU cache of verified shards. Every load from disk re-verifies the content
    hash against the manifest, so a modified shard can never be served."""

    def __init__(self, root, manifests: dict, tokenizer_hash: str, capacity: int = 6):
        self.root = root
        self.manifests = manifests
        self.tokenizer_hash = tokenizer_hash
        self.capacity = capacity
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.read_seconds = 0.0

    def get(self, shard_id) -> ShardData:
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            self.hits += 1
            return self._cache[shard_id]
        m = self.manifests[shard_id]
        if m["tokenizer_hash"] != self.tokenizer_hash:
            raise ShardIntegrityError(f"{shard_id} was tokenized with a different tokenizer")
        t0 = time.perf_counter()
        data = read_shard(self.root, m, verify=True)
        self.read_seconds += time.perf_counter() - t0
        self.misses += 1
        self._cache[shard_id] = data
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return data

    def stats(self):
        total = self.hits + self.misses
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "cache_hit_rate": round(self.hits / total, 4) if total else None,
                "shard_reads": self.misses,
                "mean_shard_read_ms": round(1000 * self.read_seconds / self.misses, 3) if self.misses else None}

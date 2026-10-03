"""Byte-level BPE tokenizer with canonical special tokens, Indic-safe pretokenization and a
frozen, hash-identified serialization.

Contract (Session 2): the same raw text always produces the same ids. The tokenizer hash is
the sha256 of its canonical JSON spec; every shard manifest records it and the loader refuses
any shard whose tokenizer hash differs from the frozen one.
"""
import re
import unicodedata
from collections import Counter

import numpy as np

from .util import read_json, sha256_json, write_json

SPECIALS = ["<|pad|>", "<|eos|>", "<|bos|>", "<|user|>", "<|assistant|>", "<|tool_call|>",
            "<|tool_obs|>", "<|think|>", "<|answer|>"]
PAD, EOS = 0, 1
ROLE_TOKEN = {"user": 3, "assistant": 4, "tool_call": 5, "tool_obs": 6, "think": 7, "answer": 8}
BYTE_OFFSET = len(SPECIALS)

# Devanagari (U+0900-097F) and Tamil (U+0B80-0BFF) ranges are part of the word class so that
# combining vowel signs and viramas never split a syllable from its consonant.
_WORD = r"\wऀ-ॿ஀-௿"
PRETOKENIZE = rf" ?[{_WORD}]+| ?[^\s{_WORD}]+|\s+"
_PAT = re.compile(PRETOKENIZE)

# Roles whose tokens carry next-token loss, per sample kind (Session 5 loss maps).
LOSS_ROLES = {
    "chat": {"assistant"},
    "reasoning": {"think", "answer"},
    "agentic": {"assistant", "tool_call"},
}


class TokenizerError(Exception):
    pass


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = BYTE_OFFSET + 256 + len(self.merges)
        self._bytes = [s.encode("utf-8") for s in SPECIALS] + [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache = {}
        self.frozen = False

    # ------------------------------------------------------------------ spec / hash
    def spec(self) -> dict:
        return {"format": "tdes-byte-bpe", "version": "1.0", "normalization": "NFC",
                "pretokenizer": PRETOKENIZE, "specials": SPECIALS, "byte_offset": BYTE_OFFSET,
                "loss_roles": {k: sorted(v) for k, v in LOSS_ROLES.items()},
                "merges": [list(m) for m in self.merges]}

    @property
    def hash(self) -> str:
        return sha256_json(self.spec())

    @property
    def version(self) -> str:
        return f"tdes-bpe-v1-{self.hash[:12]}"

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts, num_merges: int) -> "Tokenizer":
        words = Counter()
        for text in texts:
            for piece in _PAT.findall(unicodedata.normalize("NFC", text)):
                words[tuple(b + BYTE_OFFSET for b in piece.encode("utf-8"))] += 1
        words = dict(words)
        merges = []
        next_id = BYTE_OFFSET + 256
        for _ in range(num_merges):
            pairs = Counter()
            for w, c in words.items():
                for pair in zip(w, w[1:]):
                    pairs[pair] += c
            if not pairs:
                break
            # deterministic: highest count, ties broken by smallest pair ids
            (pair, count) = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < 2:
                break
            merges.append(pair)
            words = {_merge(w, pair, next_id): c for w, c in words.items()}
            next_id += 1
        return cls(merges)

    # ------------------------------------------------------------------ encode / decode
    def _encode_piece(self, piece: str):
        ids = self._cache.get(piece)
        if ids is not None:
            return ids
        ids = [b + BYTE_OFFSET for b in piece.encode("utf-8")]
        while len(ids) > 1:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            ids = list(_merge(tuple(ids), best, BYTE_OFFSET + 256 + best_rank))
        self._cache[piece] = ids
        return ids

    def encode_text(self, text: str):
        """Plain text -> ids. Special-token strings inside text are encoded as ordinary bytes,
        so data can never inject a control token."""
        out = []
        for piece in _PAT.findall(unicodedata.normalize("NFC", text)):
            out.extend(self._encode_piece(piece))
        return out

    def encode_doc(self, doc: dict):
        """Document -> (token ids uint16, train_on uint8). `train_on[i]` says whether token i is
        a loss-bearing target (Session 1/5 loss map). Plain text and code: every token incl. EOS.
        Structured samples: only tokens of the kind's loss roles (and the closing EOS)."""
        ids, train_on = [], []
        if "text" in doc:
            body = self.encode_text(doc["text"]) + [EOS]
            ids, train_on = body, [1] * len(body)
        else:
            loss_roles = LOSS_ROLES[doc["kind"]]
            for turn in doc["turns"]:
                flag = 1 if turn["role"] in loss_roles else 0
                ids.append(ROLE_TOKEN[turn["role"]])
                train_on.append(0)                       # the role marker itself is context
                body = self.encode_text(turn["text"])
                ids.extend(body)
                train_on.extend([flag] * len(body))
            ids.append(EOS)
            train_on.append(1 if doc["turns"][-1]["role"] in loss_roles else 0)
        return np.asarray(ids, dtype=np.uint16), np.asarray(train_on, dtype=np.uint8)

    def decode(self, ids) -> str:
        return b"".join(self._bytes[int(i)] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        return self._bytes[int(i)]

    # ------------------------------------------------------------------ freeze / load
    def freeze(self, path: str) -> str:
        """Serialize, seal the file read-only and return the hash."""
        spec = self.spec()
        spec_hash = sha256_json(spec)
        write_json(path, {"spec": spec, "tokenizer_hash": spec_hash}, readonly=True)
        self.frozen = True
        return spec_hash

    @classmethod
    def load(cls, path: str, expected_hash: str = None) -> "Tokenizer":
        blob = read_json(path)
        tok = cls(blob["spec"]["merges"])
        if tok.spec() != blob["spec"]:
            raise TokenizerError("tokenizer spec does not match this implementation")
        actual = tok.hash
        if actual != blob["tokenizer_hash"]:
            raise TokenizerError(f"tokenizer file corrupted: {actual} != {blob['tokenizer_hash']}")
        if expected_hash is not None and actual != expected_hash:
            raise TokenizerError(f"tokenizer hash mismatch: {actual} != expected {expected_hash}")
        tok.frozen = True
        return tok


def _merge(word, pair, new_id):
    out, i, n = [], 0, len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


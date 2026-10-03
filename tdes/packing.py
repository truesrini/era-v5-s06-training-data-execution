"""Packing: token spans -> fixed-length packed sequences with loss masks, attention segments
and position ids, plus the deterministic, resumable per-lane streams that produce them.

Mask conventions (identical for every policy, so they can be re-derived from span refs alone):
  segment_ids[i]   1..k for the k packed spans, 0 for padding
  position_ids[i]  offset of token i inside its own segment (reset_per_segment); 0 for pads
  attention        token i may attend to j  iff  seg[i]==seg[j]!=0 and j<=i  (block-causal);
                   a pad attends only to itself
  labels[i]        tokens[i+1] if i+1 is in the same segment, else -1
  loss_mask[i]     1 iff labels[i] != -1 and train_on[i+1] == 1  (the *target* is loss-bearing)
"""
import numpy as np

from .config import ATTENTION_POLICY, POSITION_POLICY
from .tokenizer import PAD
from .util import array_hash, seed_int, sha256_json

POLICIES = ("concat_chop", "best_fit_chunked", "structure_preserving")


def span_id(shard_id, doc_index, start, end):
    return f"{shard_id}:{doc_index}:{start}-{end}"


class PackedSequence:
    __slots__ = ("lane", "policy", "spans", "tokens", "train_on", "segment_ids", "position_ids",
                 "labels", "loss_mask", "sample_id", "seq_hash")

    def to_ref(self):
        return {"sample_id": self.sample_id, "lane": self.lane, "policy": self.policy,
                "spans": [dict(s) for s in self.spans]}

    @property
    def n_tokens(self):
        return int((self.segment_ids > 0).sum())

    @property
    def n_loss(self):
        return int(self.loss_mask.sum())

    def attention_mask(self):
        seg = self.segment_ids
        L = len(seg)
        causal = np.tril(np.ones((L, L), dtype=bool))
        same = (seg[:, None] == seg[None, :]) & (seg[:, None] > 0)
        return (causal & same) | np.eye(L, dtype=bool)

    def loss_mask_hash(self):
        return array_hash(self.loss_mask)


def build_sequence(spans, store, seq_len, lane, policy):
    """Materialize a packed sequence from span references. Pure function of the spans and the
    (hash-verified) shards, which is what makes replay reconstruction possible."""
    toks = np.full(seq_len, PAD, dtype=np.int64)
    tr = np.zeros(seq_len, dtype=np.uint8)
    seg = np.zeros(seq_len, dtype=np.int64)
    pos = np.zeros(seq_len, dtype=np.int64)
    cur = 0
    for k, sp in enumerate(spans, start=1):
        t, f = store.get(sp["shard_id"]).doc_slice(sp["doc_index"], sp["start"], sp["end"])
        n = len(t)
        if n != sp["end"] - sp["start"] or cur + n > seq_len:
            raise ValueError(f"span {sp['span_id']} does not fit")
        toks[cur:cur + n] = t
        tr[cur:cur + n] = f
        seg[cur:cur + n] = k
        pos[cur:cur + n] = np.arange(n)
        cur += n
    labels = np.full(seq_len, -1, dtype=np.int64)
    same_next = (seg[:-1] == seg[1:]) & (seg[:-1] > 0)
    labels[:-1] = np.where(same_next, toks[1:], -1)
    loss = np.zeros(seq_len, dtype=np.float32)
    loss[:-1] = (same_next & (tr[1:] == 1)).astype(np.float32)
    s = PackedSequence()
    s.lane, s.policy, s.spans = lane, policy, [dict(x) for x in spans]
    s.tokens, s.train_on, s.segment_ids, s.position_ids, s.labels, s.loss_mask = toks, tr, seg, pos, labels, loss
    s.sample_id = "ps-" + sha256_json({"policy": policy, "seq_len": seq_len,
                                        "spans": [x["span_id"] for x in spans]})[:16]
    s.seq_hash = array_hash(toks, labels, loss, pos, seg)
    return s


def batch_hash(seqs):
    return sha256_json({"samples": [s.sample_id for s in seqs], "seq_hashes": [s.seq_hash for s in seqs]})


def verify_sequence(s, store):
    """Independent invariant checks on a packed sequence; returns list of failed check names."""
    bad = []
    L = len(s.tokens)
    seg, pos = s.segment_ids, s.position_ids
    if (s.loss_mask[seg == 0] != 0).any():
        bad.append("loss_on_padding")
    if (s.tokens[seg == 0] != PAD).any():
        bad.append("non_pad_in_padding")
    for k, sp in enumerate(s.spans, start=1):
        idx = np.nonzero(seg == k)[0]
        if len(idx) != sp["end"] - sp["start"] or (len(idx) and (idx != np.arange(idx[0], idx[0] + len(idx))).any()):
            bad.append("segment_not_contiguous")
            continue
        if (pos[idx] != np.arange(len(idx))).any():
            bad.append("position_ids_not_reset")
        t, f = store.get(sp["shard_id"]).doc_slice(sp["doc_index"], sp["start"], sp["end"])
        if (s.tokens[idx] != t).any():
            bad.append("tokens_do_not_match_shard")
        last = idx[-1]
        if s.loss_mask[last] != 0 or s.labels[last] != -1:
            bad.append("loss_crosses_segment_boundary")
        # loss only where the target token is loss-bearing in the shard
        expect = np.zeros(len(idx), dtype=np.float32)
        expect[:-1] = f[1:].astype(np.float32)
        if (s.loss_mask[idx] != expect).any():
            bad.append("loss_mask_disagrees_with_loss_map")
    att = s.attention_mask()
    i, j = np.nonzero(att)
    off_diag = i != j
    if ((seg[i[off_diag]] != seg[j[off_diag]]) | (j[off_diag] > i[off_diag])).any():
        bad.append("attention_leaks_across_segments_or_future")
    if s.policy == "structure_preserving":
        for sp in s.spans:
            d = store.get(sp["shard_id"]).docs[sp["doc_index"]]
            if sp["start"] != 0 or sp["end"] != d["length"]:
                bad.append("structured_sample_split")
    if seg.max() > 0 and len(set(seg[seg > 0].tolist())) != len(s.spans):
        bad.append("segment_count_mismatch")
    if L != len(s.labels):
        bad.append("shape")
    return sorted(set(bad))


# ----------------------------------------------------------------------------- units
def lane_units(lane, policy, manifests, store, seq_len, newline_ids):
    """Sample units a lane stream draws from, plus a report of anything dropped.
    concat_chop: whole documents (cut freely into windows)
    best_fit_chunked: documents split at line boundaries into chunks <= seq_len
    structure_preserving: whole samples; samples longer than seq_len are dropped, never cut"""
    units, dropped = [], []
    for sid in sorted(manifests):
        data = store.get(sid)
        for d in data.docs:
            n = d["length"]
            base = {"shard_id": sid, "doc_index": d["doc_index"], "doc_id": d["doc_id"]}
            if policy == "concat_chop":
                units.append({**base, "start": 0, "end": n})
            elif policy == "structure_preserving":
                if n <= seq_len:
                    units.append({**base, "start": 0, "end": n})
                else:
                    dropped.append({"doc_id": d["doc_id"], "length": n, "reason": "structured_sample_exceeds_seq_len"})
            elif policy == "best_fit_chunked":
                toks = data.doc_slice(d["doc_index"])[0]
                start = 0
                while start < n:
                    end = min(n, start + seq_len)
                    if end < n:
                        cut = None
                        for i in range(end - 1, start, -1):
                            if int(toks[i]) in newline_ids:
                                cut = i + 1
                                break
                        end = cut if cut else end
                    units.append({**base, "start": start, "end": end})
                    start = end
            else:
                raise ValueError(policy)
    return units, dropped


# ----------------------------------------------------------------------------- streams
class LaneStream:
    """Deterministic, resumable producer of packed sequences for one lane.
    The order of units in epoch e is a permutation seeded by (seed, lane, e); the full state is
    (epoch, pos, offset, buffer) and is saved inside every checkpoint."""

    def __init__(self, lane, policy, units, seq_len, seed, buffer_size):
        if not units:
            raise ValueError(f"lane {lane} has no units")
        self.lane, self.policy, self.units = lane, policy, units
        self.seq_len, self.seed, self.buffer_size = seq_len, seed, buffer_size
        self.epoch, self.pos, self.offset, self.buffer = 0, 0, 0, []
        self._orders = {}

    def _order(self, epoch):
        if epoch not in self._orders:
            rng = np.random.default_rng(seed_int(self.seed, "stream", self.lane, epoch))
            self._orders = {epoch: rng.permutation(len(self.units))}
        return self._orders[epoch]

    def _advance(self):
        self.pos += 1
        self.offset = 0
        if self.pos >= len(self.units):
            self.epoch += 1
            self.pos = 0

    def _span(self, u_idx, start, end, epoch):
        u = self.units[u_idx]
        s, e = u["start"] + start, u["start"] + end
        return {"span_id": span_id(u["shard_id"], u["doc_index"], s, e), "shard_id": u["shard_id"],
                "doc_index": u["doc_index"], "doc_id": u["doc_id"], "start": s, "end": e, "epoch": epoch}

    def next_spans(self):
        L = self.seq_len
        if self.policy == "concat_chop":
            spans, room = [], L
            while room > 0:
                u_idx = int(self._order(self.epoch)[self.pos])
                u = self.units[u_idx]
                n = min(room, (u["end"] - u["start"]) - self.offset)
                spans.append(self._span(u_idx, self.offset, self.offset + n, self.epoch))
                self.offset += n
                room -= n
                if self.offset >= u["end"] - u["start"]:
                    self._advance()
            return spans
        # best-fit decreasing over a small look-ahead buffer (code chunks / structured samples)
        while len(self.buffer) < self.buffer_size:
            u_idx = int(self._order(self.epoch)[self.pos])
            self.buffer.append([u_idx, self.epoch])
            self._advance()
        length = lambda b: self.units[b[0]]["end"] - self.units[b[0]]["start"]
        order = sorted(range(len(self.buffer)), key=lambda i: (-length(self.buffer[i]), self.buffer[i][0], self.buffer[i][1]))
        room, take = L, []
        for i in order:
            if length(self.buffer[i]) <= room:
                take.append(i)
                room -= length(self.buffer[i])
        spans = [self._span(self.buffer[i][0], 0, length(self.buffer[i]), self.buffer[i][1]) for i in take]
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in set(take)]
        return spans

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "offset": self.offset, "buffer": [list(b) for b in self.buffer]}

    def load_state_dict(self, st):
        self.epoch, self.pos, self.offset = st["epoch"], st["pos"], st["offset"]
        self.buffer = [list(b) for b in st["buffer"]]


# ----------------------------------------------------------------------------- policy lab
def simulate_policy(lengths, policy, L, structured):
    """Packing lab: what each policy would do with a lane's samples (arrival order).
    Unstructured samples longer than L are cut into L-sized pieces first (except for
    concat_chop, which streams them); structured samples longer than L are dropped."""
    lengths = [int(x) for x in lengths]
    if structured:
        items = [n for n in lengths if n <= L]
        truncated = sum(n for n in lengths if n > L)
    else:
        items = []
        for n in lengths:
            while n > 0:
                items.append(min(n, L))
                n -= L
        truncated = 0
    total_tokens = sum(items)
    crossings = 0
    if policy == "pad_only":
        seqs = len(items)
    elif policy == "concat_chop":
        seqs = -(-total_tokens // L)
        crossings = max(0, seqs - 1)     # windows that cut a sample in two
    else:
        bins = []
        order = items if policy == "greedy_first_fit" else sorted(items, reverse=True)
        for n in order:
            fits = [i for i, room in enumerate(bins) if n <= room]
            if not fits:
                bins.append(L - n)
            elif policy == "greedy_first_fit":
                bins[fits[0]] -= n
            else:
                best = min(fits, key=lambda i: bins[i])
                bins[best] -= n
        seqs = len(bins)
    positions = seqs * L
    return {"policy": policy, "sequences": seqs, "positions": positions, "real_tokens": total_tokens,
            "padding": positions - total_tokens, "utilization": round(total_tokens / positions, 4) if positions else 0.0,
            "dropped_oversize_tokens": truncated, "samples_cut_across_windows": crossings,
            "structure_safe": not (structured and policy == "concat_chop")}


def policy_metadata():
    return {"attention_policy": ATTENTION_POLICY, "position_policy": POSITION_POLICY}

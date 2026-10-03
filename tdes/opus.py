"""OPUS-style candidate selection (Session 5/6).

Each candidate packed sequence is scored by the cosine between its own gradient and the
gradient of a trusted *proxy* set at the current weights: "would this update move the model
in the direction the proxy wants?". The acceptance threshold tau is a quantile of the scores of
the step's candidate pool. Per lane, with quota q:

  accepted   score >= tau, best first, up to q
  deferred   score >= tau but the quota is full -> re-offered next step (max_defer times)
  rejected   low_proxy_utility (score < tau), quota_pressure (deferred too often),
             duplicate (seen in the recent window), stage_mismatch (lane inactive now)
  protected-floor override
             a protected lane below its floor takes its best rejected candidates anyway
  quota_fill a non-protected lane still short after refill rounds takes its best rejected
             candidates so the compiled mixture is honoured (flagged in the record)

Every decision is a ledger record, so rejected clean data never disappears.
"""
import math

import numpy as np


def threshold(scores, quantile):
    if not scores:
        return 0.0
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile, method="lower"))


def effective_tokens(n_loss, score, tau):
    """Heuristic: loss-bearing tokens scaled by a logistic of the margin above tau."""
    return int(round(n_loss / (1.0 + math.exp(-8.0 * (score - tau)))))


def triage(cands, quota, tau, recent_ids, seen_ids):
    """First-pass decision for one lane. `cands` are dicts with score/sample_id/candidate_id.
    Returns (accepted, low, beyond_quota, duplicates)."""
    accepted, low, beyond, dups = [], [], [], []
    for c in sorted(cands, key=lambda c: (-c["score"], c["candidate_id"])):
        if c["sample_id"] in recent_ids or c["sample_id"] in seen_ids:
            dups.append(c)
            continue
        seen_ids.add(c["sample_id"])
        if c["score"] < tau:
            low.append(c)
        elif len(accepted) < quota:
            accepted.append(c)
        else:
            beyond.append(c)
    return accepted, low, beyond, dups

"""A tiny one-block causal transformer in numpy with a hand-written backward pass.

It consumes exactly what the data system produces: token ids, position ids (learned position
embeddings indexed by the packed position ids), segment ids (block-causal attention mask) and
a loss mask. Per-token cross-entropy is returned for the learning ledger.
"""
import numpy as np

from .util import params_hash

PARAM_NAMES = ("E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "Wout")
NEG = -1e9


def init_params(vocab, d, ff, max_len, seed, std, dtype="float32"):
    rng = np.random.default_rng(seed)
    p = {
        "E": rng.normal(0, std, (vocab, d)), "P": rng.normal(0, std, (max_len, d)),
        "Wq": rng.normal(0, std, (d, d)), "Wk": rng.normal(0, std, (d, d)),
        "Wv": rng.normal(0, std, (d, d)), "Wo": rng.normal(0, std, (d, d)),
        "W1": rng.normal(0, std, (d, ff)), "b1": np.zeros(ff), "W2": rng.normal(0, std, (ff, d)),
        "Wout": rng.normal(0, std, (d, vocab)),
    }
    return {k: v.astype(dtype) for k, v in p.items()}


def attention_allowed(seg):
    """(B, L) segment ids -> (B, L, L) boolean: same non-pad segment and causal; pads see self."""
    B, L = seg.shape
    causal = np.tril(np.ones((L, L), dtype=bool))[None]
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] > 0)
    return (causal & same) | np.eye(L, dtype=bool)[None]


def forward_backward(params, tokens, positions, seg, labels, loss_mask, loss_scale, need_grad=True):
    """Returns (sum of weighted loss, per-token CE (B,L) [0 where unmasked], grads or None).
    Weighted loss = sum_t loss_mask_t * CE_t * loss_scale."""
    dt = params["E"].dtype
    E, P = params["E"], params["P"]
    d = E.shape[1]
    x0 = E[tokens] + P[positions]
    q, k, v = x0 @ params["Wq"], x0 @ params["Wk"], x0 @ params["Wv"]
    allowed = attention_allowed(seg)
    s = (q @ np.swapaxes(k, 1, 2)) / np.sqrt(d).astype(dt)
    s = np.where(allowed, s, NEG).astype(dt)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s)
    a = a / a.sum(-1, keepdims=True)
    c = a @ v
    x1 = x0 + c @ params["Wo"]
    u = x1 @ params["W1"] + params["b1"]
    r = np.maximum(u, 0)
    x2 = x1 + r @ params["W2"]
    logits = x2 @ params["Wout"]
    logits = logits - logits.max(-1, keepdims=True)
    ex = np.exp(logits)
    z = ex.sum(-1, keepdims=True)
    lab = np.where(labels >= 0, labels, 0)
    logp_true = np.take_along_axis(logits, lab[..., None], -1)[..., 0] - np.log(z[..., 0])
    ce = (-logp_true) * (loss_mask > 0)
    w = (loss_mask * loss_scale).astype(dt)
    total = float((ce * w).sum())
    if not need_grad:
        return total, ce, None

    g = {}
    dlogits = ex / z
    np.put_along_axis(dlogits, lab[..., None], np.take_along_axis(dlogits, lab[..., None], -1) - 1, -1)
    dlogits *= w[..., None]
    flat = lambda t: t.reshape(-1, t.shape[-1])
    g["Wout"] = flat(x2).T @ flat(dlogits)
    dx2 = dlogits @ params["Wout"].T
    g["W2"] = flat(r).T @ flat(dx2)
    du = (dx2 @ params["W2"].T) * (u > 0)
    g["W1"] = flat(x1).T @ flat(du)
    g["b1"] = du.sum((0, 1))
    dx1 = dx2 + du @ params["W1"].T
    g["Wo"] = flat(c).T @ flat(dx1)
    dc = dx1 @ params["Wo"].T
    da = dc @ np.swapaxes(v, 1, 2)
    dv = np.swapaxes(a, 1, 2) @ dc
    ds = a * (da - (da * a).sum(-1, keepdims=True))
    ds = ds / np.sqrt(d).astype(dt)
    dq = ds @ k
    dk = np.swapaxes(ds, 1, 2) @ q
    g["Wq"] = flat(x0).T @ flat(dq)
    g["Wk"] = flat(x0).T @ flat(dk)
    g["Wv"] = flat(x0).T @ flat(dv)
    dx0 = dx1 + dq @ params["Wq"].T + dk @ params["Wk"].T + dv @ params["Wv"].T
    gE = np.zeros_like(E)
    np.add.at(gE, tokens.reshape(-1), flat(dx0))
    gP = np.zeros_like(P)
    np.add.at(gP, positions.reshape(-1), flat(dx0))
    g["E"], g["P"] = gE, gP
    g = {kk: vv.astype(dt) for kk, vv in g.items()}
    return total, ce, g


def flat_grad(g):
    return np.concatenate([g[k].ravel().astype(np.float64) for k in PARAM_NAMES])


class Adam:
    """AdamW with global-norm clipping and warmup + cosine LR. All state is checkpointed."""

    def __init__(self, params, cfg):
        t = cfg["train"]
        self.lr_peak, self.warmup, self.total = t["lr"], t["warmup_steps"], t["total_steps"]
        self.min_frac, self.b1, self.b2 = t["min_lr_frac"], t["betas"][0], t["betas"][1]
        self.eps, self.wd, self.clip = t["eps"], t["weight_decay"], t["grad_clip"]
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def lr_at(self, step):
        if step <= self.warmup:
            return self.lr_peak * step / self.warmup
        frac = (step - self.warmup) / max(1, self.total - self.warmup)
        frac = min(1.0, frac)
        return self.lr_peak * (self.min_frac + (1 - self.min_frac) * 0.5 * (1 + np.cos(np.pi * frac)))

    def step(self, params, grads, global_step):
        norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12))
        lr = float(self.lr_at(global_step))
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in PARAM_NAMES:
            g = grads[k] * np.asarray(scale, dtype=grads[k].dtype)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            upd = (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            decay = self.wd * params[k] if k not in ("b1",) else 0.0
            params[k] = (params[k] - lr * (upd + decay)).astype(params[k].dtype)
        return norm, lr

    def state_arrays(self):
        out = {f"m.{k}": v for k, v in self.m.items()}
        out.update({f"v.{k}": v for k, v in self.v.items()})
        return out

    def load_arrays(self, arrs, t):
        self.m = {k: arrs[f"m.{k}"] for k in PARAM_NAMES}
        self.v = {k: arrs[f"v.{k}"] for k in PARAM_NAMES}
        self.t = t


__all__ = ["init_params", "forward_backward", "flat_grad", "Adam", "params_hash", "attention_allowed", "PARAM_NAMES"]

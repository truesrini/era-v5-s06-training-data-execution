"""A tiny one-block causal transformer in PyTorch, trained with torch.optim.AdamW.

It consumes exactly what the data system produces: token ids, position ids (learned position
embeddings indexed by the packed position ids), segment ids (block-causal attention mask) and
a loss mask. Per-token cross-entropy is returned for the learning ledger.

Determinism: `torch.use_deterministic_algorithms(True)`, one intra-op CPU thread, TF32 off and a
fixed cuBLAS workspace (set in tdes/__init__.py), so on a given device a resumed or replayed run
reproduces the original weights bit for bit (checked via weight hashes). Runs on CPU by default
or on CUDA with `--device cuda`.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import array_hash

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False "
                           "(install a CUDA build of torch, or use --device cpu)")
    return torch.device(name)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def attention_allowed(seg):
    """(B, L) segment ids -> (B, L, L) bool: same non-pad segment and causal; pads see only self."""
    L, dev = seg.shape[1], seg.device
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))[None]
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] > 0)
    return (causal & same) | torch.eye(L, dtype=torch.bool, device=dev)[None]


class TinyLM(nn.Module):
    def __init__(self, vocab, d, ff, max_len, seed, std, dtype="float32"):
        # parameters are drawn on the CPU, so initial weights are identical on every device
        super().__init__()
        g = torch.Generator().manual_seed(int(seed) % (2 ** 63))
        dt = _DTYPES[dtype]

        def p(*shape):
            return nn.Parameter(torch.randn(*shape, generator=g, dtype=torch.float64).mul(std).to(dt))
        self.E, self.P = p(vocab, d), p(max_len, d)
        self.Wq, self.Wk, self.Wv, self.Wo = p(d, d), p(d, d), p(d, d), p(d, d)
        self.W1, self.b1, self.W2 = p(d, ff), nn.Parameter(torch.zeros(ff, dtype=dt)), p(ff, d)
        self.Wout = p(d, vocab)
        self.d = d

    def forward(self, tokens, positions, seg):
        x0 = self.E[tokens] + self.P[positions]
        q, k, v = x0 @ self.Wq, x0 @ self.Wk, x0 @ self.Wv
        s = (q @ k.transpose(1, 2)) / math.sqrt(self.d)
        s = s.masked_fill(~attention_allowed(seg), float("-inf"))
        x1 = x0 + torch.softmax(s, dim=-1) @ v @ self.Wo
        x2 = x1 + F.relu(x1 @ self.W1 + self.b1) @ self.W2
        return x2 @ self.Wout

    def token_ce(self, tokens, positions, seg, labels, loss_mask):
        """Per-token cross-entropy (B, L); zero where loss_mask is 0."""
        logits = self(tokens, positions, seg)
        lab = labels.clamp(min=0)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), reduction="none")
        return ce.reshape(labels.shape) * (loss_mask > 0)

    # ------------------------------------------------------------------ hashing / io
    def state_arrays(self):
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def weights_hash(self):
        a = self.state_arrays()
        return array_hash(*[a[k] for k in sorted(a)])


def _device(model):
    return next(model.parameters()).device


def to_tensors(arrays, device="cpu"):
    tokens, positions, seg, labels, loss_mask = arrays
    as_t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=device)
    return (as_t(tokens, torch.long), as_t(positions, torch.long), as_t(seg, torch.long),
            as_t(labels, torch.long), as_t(loss_mask))


def loss_and_backward(model, arrays, loss_scale):
    """Accumulate d(sum_t mask*CE*scale)/dparams into .grad. Returns (weighted loss, CE (B,L) numpy)."""
    t = to_tensors(arrays, _device(model))
    ce = model.token_ce(*t)
    loss = (ce * t[4].to(ce.dtype)).sum() * loss_scale
    loss.backward()
    return float(loss.detach()), ce.detach().cpu().numpy()


def eval_ce(model, arrays):
    with torch.no_grad():
        ce = model.token_ce(*to_tensors(arrays, _device(model)))
    return ce.cpu().numpy()


def grad_vector(model, arrays):
    """Flattened gradient of the mean loss over loss-bearing tokens (does not touch .grad)."""
    t = to_tensors(arrays, _device(model))
    ce = model.token_ce(*t)
    n = max(1.0, float(t[4].sum()))
    loss = (ce * t[4].to(ce.dtype)).sum() / n
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return torch.cat([g.reshape(-1) for g in grads]).double(), float(loss.detach())


class Optimizer:
    """torch.optim.AdamW (no decay on biases) + global-norm clipping + warmup/cosine LR."""

    def __init__(self, model, cfg):
        t = cfg["train"]
        self.lr_peak, self.warmup, self.total = t["lr"], t["warmup_steps"], t["total_steps"]
        self.min_frac, self.clip = t["min_lr_frac"], t["grad_clip"]
        decay = [p for n, p in model.named_parameters() if n != "b1"]
        no_decay = [p for n, p in model.named_parameters() if n == "b1"]
        self.model = model
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                                      {"params": no_decay, "weight_decay": 0.0}],
                                     lr=self.lr_peak, betas=tuple(t["betas"]), eps=t["eps"], foreach=False)

    def lr_at(self, step):
        if step <= self.warmup:
            return self.lr_peak * step / self.warmup
        frac = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        return self.lr_peak * (self.min_frac + (1 - self.min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=False)
        for p in self.model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    def step(self, global_step):
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=False))
        lr = float(self.lr_at(global_step))
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        return norm, lr

    @property
    def t(self):
        st = self.opt.state_dict()["state"]
        return int(next(iter(st.values()))["step"]) if st else 0

    def state_hash(self):
        st = self.opt.state_dict()["state"]
        arrs = []
        for i in sorted(st):
            arrs += [st[i]["exp_avg"].cpu().numpy(), st[i]["exp_avg_sq"].cpu().numpy(), np.asarray(float(st[i]["step"]))]
        return array_hash(*arrs)


def state_dict_hash(sd):
    return array_hash(*[sd[k].detach().cpu().numpy() for k in sorted(sd)])


def checkpoint_weights_hash(path):
    """Hash of the weights stored in a checkpoint's model.pt (same convention as weights_hash)."""
    return state_dict_hash(torch.load(path, map_location="cpu", weights_only=True))

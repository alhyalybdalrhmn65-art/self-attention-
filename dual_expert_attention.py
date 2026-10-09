"""
dual_expert_attention.py
========================
Dual-Expert Attention, v2.8.3 (corrected spec), Abdulrahman Al-Hayali.

Implements Eq. (1)-(15) of "Learning When to Look Like vs. Work Like: A
Dual-Expert Attention with a Graded Alpha Knob", with the review fixes:

  (1)  E0(q,k)  = q.k / sqrt(d_head)                      "looks-like"
  (2)  E1(q,k)  = -(1/d_head) ||W_rel q - k||^2           "works-like" (lock & key)
       W_rel is PER HEAD: shape [n_heads, d_head, d_head], orthogonal init.
  (2b) per-query LayerNorm ACROSS KEYS, INDEPENDENT stats for E0 and E1
       E~ = (E - mean_k E) / (std_k E + 1e-5)   (std with unbiased=False)
  (3)  S*(q,k;a)  = (1-a) E~0 + a E~1                       oracle DJ mixer
  (6)  a_hat      = sigmoid(clip(W_gate [q; k; E~0; E~1] + c_gate, -10, 10))
       then clamp to [0, 0.95]  (never 1.0: the March "queen -> queen bee" collapse)
  (7)  S~         = (1-a_hat) E~0 + a_hat E~1
  (10) beta_{t,p} = (E~0p - E~0t) / (den + eps*sign(den)),
       den = (E~1t - E~0t) - (E~1p - E~0p); |den|<eps -> None; beta outside [0,1] -> skip
  (11) d_b        = sqrt((a - sg[beta])^2 + 1e-4)
  (12) w          = sg[1 + lambda_s exp(-d_b / tau_s)]
  (13) s_beta     = tanh((a - sg[beta]) / 0.05)          SIGNED, no abs
  (14) L_decision = softplus(-s_beta (a_hat - sg[beta]) / tau_d)
  (15) L_v2.8     = w (a_hat - a)^2 + lambda_d L_decision
       S_mix      = (1-a_mix) E~0 + a_mix E~1,  a_mix = g a + (1-g) a_hat,
       g = gamma_mix annealed 1 -> 0 (done by the training loop).

GRADIENT ISOLATION (built into the graph, no second optimizer needed)
  * The gate only ever sees DETACHED q, k, E~0, E~1  -> L_v2.8 cannot reach
    W_rel / W_q / W_k / W_v.
  * Attention scores use a DETACHED a_hat             -> task CE cannot reach
    W_gate / c_gate.
  * beta and w are detached (sg[.]).
  So one backward() of (CE + L_gate) is exactly equivalent to two separate
  backward passes into two disjoint parameter sets. transformer.py still puts
  them in two AdamW param groups (lr 3e-4 vs 1e-3) and clips them separately.

Shapes: B batch, H heads, Tq queries, Tk keys, Dh = d_model // n_heads.
alpha_oracle: [B,Tq,Tk] or [B,H,Tq,Tk]; NaN = "no oracle label for this pair".
key_mask:     [B,Tk] bool, True = real token, False = padding.
target_idx:   [B,Tq] long, gold key per query, -1 = none. If None, the gold key
              is the oracle winner argmax_k S*(q,k;alpha) over labelled keys.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Hyper-parameters straight from the paper
# ----------------------------------------------------------------------------
ALPHA_MAX = 0.95      # never 1.0, keep a 5-10% similarity anchor
LOGIT_CLIP = 10.0     # clip the gate LOGIT (not the weights) before sigmoid
NORM_EPS = 1e-5       # per-query LayerNorm epsilon
BETA_EPS = 1e-6       # photo-finish epsilon
DB_EPS = 1e-4         # sqrt stability in d_b
LAMBDA_S = 2.0        # spotlight height
TAU_S = 0.1           # spotlight width
S_BETA_TEMP = 0.05    # tanh temperature in s_beta
TAU_D = 0.1           # decision softplus temperature
LAMBDA_D = 0.5        # weight of L_decision
C_GATE_INIT = -0.5    # sigma^-1(mean alpha)


# ----------------------------------------------------------------------------
# Functional pieces (exported so visualize.py / tests can reuse them)
# ----------------------------------------------------------------------------
def per_query_layernorm(E: torch.Tensor, key_mask: Optional[torch.Tensor] = None,
                        eps: float = NORM_EPS) -> torch.Tensor:
    """Eq (2b). Normalise each query row across the KEY axis (last dim).

    NOT nn.LayerNorm(d_model): statistics are mean/std over Tk for every
    (b, h, q). Padding keys are excluded from the statistics and set to 0.
    std uses the biased estimator (unbiased=False), as specified.
    """
    if key_mask is None:
        mask = torch.ones_like(E, dtype=torch.bool)
    else:
        mask = key_mask.expand_as(E)
    m = mask.to(E.dtype)
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    mu = (E * m).sum(-1, keepdim=True) / n
    var = (((E - mu) * m) ** 2).sum(-1, keepdim=True) / n          # unbiased=False
    sigma = var.clamp_min(1e-12).sqrt()                              # NaN-safe grad
    E_norm = (E - mu) / (sigma + eps)
    return E_norm.masked_fill(~mask, 0.0)


def decision_boundary(E0n: torch.Tensor, E1n: torch.Tensor, t_idx: torch.Tensor,
                      eps: float = BETA_EPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq (8)-(10): photo-finish alpha' where target t and distractor p tie.

    E0n, E1n: [..., Tk]; t_idx: [...] long (gold key per query).
    Returns beta [..., Tk] and a validity mask (|den|>=eps, beta in [0,1], p != t).
    beta depends ONLY on E~0, E~1 (never on the oracle alpha).
    """
    t = t_idx.unsqueeze(-1)
    E0t, E1t = E0n.gather(-1, t), E1n.gather(-1, t)
    num = E0n - E0t                                   # E~0,p - E~0,t
    den = (E1t - E0t) - (E1n - E0n)                   # D
    sgn = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
    beta = num / (den + eps * sgn)                    # CORRECTED: sign(den)
    not_target = torch.arange(E0n.size(-1), device=E0n.device) != t
    valid = (den.abs() >= eps) & (beta >= 0) & (beta <= 1) & not_target & torch.isfinite(beta)
    return beta, valid


def spotlight_decision_loss(alpha_hat_t, alpha_t, beta, lambda_s=LAMBDA_S, tau_s=TAU_S,
                            tau_d=TAU_D, lambda_d=LAMBDA_D):
    """Eq (11)-(15) for every (t, p) pair. beta must already be sanitised."""
    beta_sg = beta.detach()                                       # sg[beta]
    a = alpha_t.detach()
    d_b = torch.sqrt((a - beta_sg) ** 2 + DB_EPS)                 # (11)
    w = (1.0 + lambda_s * torch.exp(-d_b / tau_s)).detach()       # (12) sg[...]
    s_beta = torch.tanh((a - beta_sg) / S_BETA_TEMP)              # (13) signed
    l_dec = F.softplus(-s_beta * (alpha_hat_t - beta_sg) / tau_d)  # (14)
    l_pair = w * (alpha_hat_t - a) ** 2 + lambda_d * l_dec        # (15)
    return l_pair, d_b, w, s_beta, l_dec


# ----------------------------------------------------------------------------
# The module
# ----------------------------------------------------------------------------
class DualExpertAttention(nn.Module):
    """Multi-head Dual-Expert Attention (drop-in for self/cross attention).

    forward(q, k, v) takes HIDDEN STATES [B, T, d_model]; projections W_q, W_k,
    W_v (and W_o unless out_proj=False) live inside the module.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1,
                 bias: bool = True, out_proj: bool = True,
                 lambda_s: float = LAMBDA_S, tau_s: float = TAU_S,
                 tau_d: float = TAU_D, lambda_d: float = LAMBDA_D,
                 alpha_max: float = ALPHA_MAX, logit_clip: float = LOGIT_CLIP):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads                 # d_k used for scaling
        self.lambda_s, self.tau_s, self.tau_d, self.lambda_d = lambda_s, tau_s, tau_d, lambda_d
        self.alpha_max, self.logit_clip = alpha_max, logit_clip

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias) if out_proj else nn.Identity()

        Dh, H = self.d_head, n_heads
        # E1 lock-and-key, PER HEAD [H, Dh, Dh]
        self.W_rel = nn.Parameter(torch.empty(H, Dh, Dh))
        # Gate, PER HEAD W_gate in R^{1 x (2 Dh + 2)}, layout [w_q | w_k | w_E0 | w_E1]
        self.W_gate = nn.Parameter(torch.empty(H, 2 * Dh + 2))
        self.c_gate = nn.Parameter(torch.full((H,), C_GATE_INIT))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    # -- init -----------------------------------------------------------------
    def reset_parameters(self) -> None:
        for lin in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if isinstance(self.W_o, nn.Linear):
            nn.init.xavier_uniform_(self.W_o.weight)
            if self.W_o.bias is not None:
                nn.init.zeros_(self.W_o.bias)
        Dh, H = self.d_head, self.n_heads
        with torch.no_grad():
            self.W_rel.copy_(torch.stack([nn.init.orthogonal_(torch.empty(Dh, Dh)) for _ in range(H)]))
            fan_in, fan_out = 2 * Dh + 2, 1                       # Xavier per head
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            self.W_gate.uniform_(-bound, bound)
            self.c_gate.fill_(C_GATE_INIT)

    # -- helpers ----------------------------------------------------------------
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)    # [B,H,T,Dh]

    def experts(self, qh: torch.Tensor, kh: torch.Tensor):
        """Eq (1) and (2). Returns raw E0, E1 of shape [B,H,Tq,Tk]."""
        Dh = self.d_head
        E0 = qh @ kh.transpose(-1, -2) / math.sqrt(Dh)
        Wq = torch.einsum("hij,bhtj->bhti", self.W_rel, qh)               # W_rel q
        sq = (Wq ** 2).sum(-1, keepdim=True)                              # [B,H,Tq,1]
        sk = (kh ** 2).sum(-1).unsqueeze(-2)                              # [B,H,1,Tk]
        sqdist = sq + sk - 2.0 * (Wq @ kh.transpose(-1, -2))              # ||W_rel q - k||^2
        E1 = (-(1.0 / Dh) * sqdist).clamp(max=0.0)                        # E1 <= 0, best 0
        return E0, E1

    def gate_logits(self, qh, kh, E0n, E1n) -> torch.Tensor:
        """W_gate [q; k; E~0; E~1] + c_gate, computed WITHOUT materialising the
        [B,H,Tq,Tk,2Dh+2] concat (the linear map splits exactly into 4 terms)."""
        Dh, H = self.d_head, self.n_heads
        wq, wk = self.W_gate[:, :Dh], self.W_gate[:, Dh:2 * Dh]
        we0, we1 = self.W_gate[:, 2 * Dh], self.W_gate[:, 2 * Dh + 1]
        lq = torch.einsum("bhtd,hd->bht", qh, wq).unsqueeze(-1)          # [B,H,Tq,1]
        lk = torch.einsum("bhtd,hd->bht", kh, wk).unsqueeze(-2)          # [B,H,1,Tk]
        return (lq + lk + we0.view(1, H, 1, 1) * E0n + we1.view(1, H, 1, 1) * E1n
                + self.c_gate.view(1, H, 1, 1))

    # -- L_v2.8 -----------------------------------------------------------------
    def _gate_loss(self, E0n, E1n, alpha_hat, a, has, km, target_idx) -> Dict:
        """L_gate = mean_{labelled (q,k)} (a_hat - a)^2            [w = 1 where beta=None]
                  + mean_{valid (t,p)} [ w (a_hat_t - a_t)^2 + lambda_d L_decision ]
        E0n / E1n arrive detached, so this loss only trains W_gate and c_gate."""
        B, H, Tq, Tk = E0n.shape
        zero = alpha_hat.sum() * 0.0
        reg_dense = ((alpha_hat - a) ** 2)[has].mean() if has.any() else zero

        if target_idx is None:                    # gold key = oracle winner
            S_star = ((1 - a) * E0n + a * E1n).masked_fill(~has, float("-inf"))
            q_has = has.any(-1)
            t = S_star.argmax(-1)
        else:
            ti = target_idx.to(device=E0n.device, dtype=torch.long)
            if ti.dim() == 2:
                ti = ti.unsqueeze(1).expand(B, H, Tq)
            q_has = ti >= 0
            t = ti.clamp_min(0)
            q_has = q_has & has.gather(-1, t.unsqueeze(-1)).squeeze(-1)

        beta, valid = decision_boundary(E0n, E1n, t)
        keyv = km.expand(B, H, Tq, Tk) if km is not None else torch.ones_like(valid)
        valid = valid & keyv & q_has.unsqueeze(-1)

        a_t = a.gather(-1, t.unsqueeze(-1))                    # [B,H,Tq,1]
        ah_t = alpha_hat.gather(-1, t.unsqueeze(-1))
        beta_safe = torch.where(valid, beta, torch.zeros_like(beta))   # no NaN in backward
        l_pair, d_b, w, s_b, l_dec = spotlight_decision_loss(
            ah_t, a_t, beta_safe, self.lambda_s, self.tau_s, self.tau_d, self.lambda_d)

        any_valid = bool(valid.any())
        spot = l_pair[valid].mean() if any_valid else zero
        nan = float("nan")
        side = ((ah_t.detach() - beta_safe) * (a_t - beta_safe) > 0)
        stats = {
            "loss_reg": float(reg_dense.detach()),
            "loss_spot": float(spot.detach()),
            "loss_decision": float(l_dec[valid].mean()) if any_valid else 0.0,
            "decision_side_acc": float(side[valid].float().mean()) if any_valid else nan,
            "n_decision_pairs": int(valid.sum()),
        }
        return {
            "loss_gate": reg_dense + spot,
            "stats": stats,
            "beta_list": beta.masked_fill(~valid, nan).detach(),
            "w_list": w.expand_as(beta).masked_fill(~valid, nan).detach(),
            "d_b_list": d_b.expand_as(beta).masked_fill(~valid, nan).detach(),
            "beta_valid": valid,
            "target_idx": t,
        }

    # -- forward ------------------------------------------------------------------
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                alpha_oracle: Optional[torch.Tensor] = None, return_details: bool = True, *,
                gamma_mix: float = 0.0, key_mask: Optional[torch.Tensor] = None,
                target_idx: Optional[torch.Tensor] = None) -> Dict:
        B, Tq, _ = q.shape
        Tk = k.size(1)
        H, Dh = self.n_heads, self.d_head
        qh, kh, vh = self._split(self.W_q(q)), self._split(self.W_k(k)), self._split(self.W_v(v))
        km = key_mask.to(torch.bool).view(B, 1, 1, Tk) if key_mask is not None else None

        # (1)(2) experts, (2b) independent per-query LayerNorm across keys
        E0, E1 = self.experts(qh, kh)
        E0n, E1n = per_query_layernorm(E0, km), per_query_layernorm(E1, km)

        # (6) gate on NORMALISED experts, detached inputs, logit clip BEFORE sigmoid
        logit = self.gate_logits(qh.detach(), kh.detach(), E0n.detach(), E1n.detach())
        logit = logit.clamp(-self.logit_clip, self.logit_clip)
        alpha_hat = torch.sigmoid(logit).clamp(0.0, self.alpha_max)
        ah_sg = alpha_hat.detach()                        # CE never reaches the gate

        # (7) inference scores
        S_tilde = (1.0 - ah_sg) * E0n + ah_sg * E1n

        S_mix, gate, a_ref = None, None, None
        if alpha_oracle is not None:
            ao = alpha_oracle.to(device=E0n.device, dtype=E0n.dtype)
            if ao.dim() == 3:
                ao = ao.unsqueeze(1)
            ao = ao.expand(B, H, Tq, Tk)
            has = ~torch.isnan(ao)
            if km is not None:
                has = has & km
            a = torch.nan_to_num(ao, nan=0.0).clamp(0.0, self.alpha_max)
            a_eff = torch.where(has, a, ah_sg)            # unlabelled pairs fall back to a_hat
            alpha_mix = gamma_mix * a_eff + (1.0 - gamma_mix) * ah_sg
            S_mix = (1.0 - alpha_mix) * E0n + alpha_mix * E1n
            scores = S_mix
            gate = self._gate_loss(E0n.detach(), E1n.detach(), alpha_hat, a, has, km, target_idx)
            a_ref = a.masked_fill(~has, float("nan"))
        else:
            scores = S_tilde

        # E~ are already unit-scale logits: softmax directly, no extra 1/sqrt(d)
        if km is not None:
            scores = scores.masked_fill(~km, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = (self.dropout(attn) @ vh).transpose(1, 2).reshape(B, Tq, H * Dh)
        output = self.W_o(ctx)

        loss_gate = gate["loss_gate"] if gate is not None else alpha_hat.new_zeros(())
        out = {"output": output, "alpha_hat": alpha_hat, "alpha_oracle": a_ref,
               "loss_gate": loss_gate, "stats": gate["stats"] if gate is not None else {}}
        if return_details:
            out.update({
                "E0_raw": E0, "E1_raw": E1, "E0_norm": E0n, "E1_norm": E1n,
                "S_tilde": S_tilde, "S_mix": S_mix, "attn": attn,
                "beta_list": gate["beta_list"] if gate else None,
                "w_list": gate["w_list"] if gate else None,
                "d_b_list": gate["d_b_list"] if gate else None,
                "beta_valid": gate["beta_valid"] if gate else None,
                "target_idx": gate["target_idx"] if gate else None,
            })
        return out


# ----------------------------------------------------------------------------
# Demo: "Paris is capital of France" -> high alpha ~0.9
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    tokens = ["[CLS]", "paris", "is", "capital", "of", "france", "[SEP]"]
    T, qi, ki = len(tokens), 1, 5
    x = torch.randn(1, T, 512)
    dea = DualExpertAttention(d_model=512, n_heads=8, dropout=0.0)

    # Oracle labels (illustrative values inside the paper's ranges; in real
    # training they come from DeBERTa-MNLI + MiniLM, see data.py)
    alpha = torch.full((1, T, T), float("nan"))
    alpha[0, qi, ki] = 0.90                      # paris -> france : strong relation
    alpha[0, qi, 2] = 0.02                       # paris -> is     : no relation
    alpha[0, qi, 4] = 0.02                       # paris -> of     : no relation
    target = torch.full((1, T), -1, dtype=torch.long)
    target[0, qi] = ki                           # gold key for "paris" is "france"

    with torch.no_grad():
        a0 = dea(x, x, x, alpha_oracle=alpha, target_idx=target)["alpha_hat"][0, :, qi, ki]
    print(f"alpha_hat(paris->france) at init  : {a0.mean():.3f}  (sigmoid(c_gate=-0.5)~0.38)")

    # Fit ONLY the gate on L_v2.8 (exactly what the gate's optimizer does in training)
    opt = torch.optim.Adam([dea.W_gate, dea.c_gate], lr=1e-2)
    for step in range(300):
        out = dea(x, x, x, alpha_oracle=alpha, gamma_mix=1.0, target_idx=target, return_details=False)
        opt.zero_grad()
        out["loss_gate"].backward()
        opt.step()

    with torch.no_grad():
        out = dea(x, x, x, alpha_oracle=alpha, gamma_mix=0.0, target_idx=target)
    ah = out["alpha_hat"][0, :, qi, ki]
    print(f"alpha_hat(paris->france) per head  : {[round(v, 3) for v in ah.tolist()]}")
    print(f"alpha_hat(paris->france) mean      : {ah.mean():.3f}   oracle 0.90  -> 10% looks / 90% works")
    print(f"alpha_hat(paris->is)     mean      : {out['alpha_hat'][0, :, qi, 2].mean():.3f}   oracle 0.02")
    b = out["beta_list"][0, 0, qi]
    bstr = ", ".join(f"{tokens[j]}={b[j]:.3f}" for j in range(T) if not torch.isnan(b[j]))
    print(f"head0 photo-finish beta vs 'france': {bstr or 'no crossing in [0,1]'}")
    print(f"gate stats: {out['stats']}")
    print(f"S~ row 'paris' (head0): {[round(v, 2) for v in out['S_tilde'][0, 0, qi].tolist()]}")
    print(f"attention row 'paris' (head0): {[round(v, 3) for v in out['attn'][0, 0, qi].tolist()]}")

    # Gradient isolation check
    dea.zero_grad(set_to_none=True)
    out = dea(x, x, x, alpha_oracle=alpha, gamma_mix=0.5, target_idx=target, return_details=False)
    out["output"].pow(2).mean().backward(retain_graph=True)        # stand-in for task CE
    assert dea.W_gate.grad is None or dea.W_gate.grad.abs().sum() == 0, "CE leaked into W_gate"
    assert dea.W_rel.grad is not None and dea.W_rel.grad.abs().sum() > 0, "W_rel got no CE grad"
    dea.zero_grad(set_to_none=True)
    out["loss_gate"].backward()
    assert dea.W_rel.grad is None or dea.W_rel.grad.abs().sum() == 0, "L_v2.8 leaked into W_rel"
    assert dea.W_gate.grad is not None and dea.W_gate.grad.abs().sum() > 0, "gate got no L_v2.8 grad"
    print("gradient isolation: OK  (W_rel <- CE only, W_gate/c_gate <- L_v2.8 only)")

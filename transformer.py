"""
transformer.py
==============
Full Transformer encoder built on DualExpertAttention (no nn.MultiheadAttention).

  * PositionalEncoding   sinusoidal
  * EncoderLayer         pre-LN, DualExpertAttention + FFN
  * TransformerModel     from-scratch encoder with CLS-classification or MLM head
  * DualExpertBert       HF bert-base-uncased (d_model=768, 12 heads, d_head=64)
                         with every self-attention swapped for DualExpertAttention
  * build_optimizer      two AdamW param groups: main 3e-4 (incl. W_rel), gate 1e-3
  * clip_grads_per_group grad clip 1.0 per group (a joint clip would couple the two)
  * gamma_mix_schedule   1 -> 0 linearly over 10k steps
  * verify_gradient_isolation
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_expert_attention import DualExpertAttention

CONFIG = dict(
    d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_len=512,
    lr_main=3e-4,        # W_rel (orthogonal init) + everything trained by task CE
    lr_gate=1e-3,        # W_gate (Xavier) + c_gate (init -0.5), trained by L_v2.8
    weight_decay=0.01, grad_clip=1.0, batch_size=128, seed=42, gamma_mix_steps=10_000,
)

GATE_PARAM_KEYS = ("W_gate", "c_gate")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = DualExpertAttention(d_model, n_heads, dropout=dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, key_mask=None, alpha_oracle=None, gamma_mix=0.0, target_idx=None,
                return_details=False):
        h = self.ln1(x)
        det = self.attn(h, h, h, alpha_oracle=alpha_oracle, return_details=return_details,
                        gamma_mix=gamma_mix, key_mask=key_mask, target_idx=target_idx)
        x = x + self.drop1(det["output"])
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x, det


def _finish(logits, labels, task, dets, n_sup):
    task_loss = None
    if labels is not None:
        if task == "cls":
            task_loss = F.cross_entropy(logits, labels)
        else:
            task_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                        ignore_index=-100)
    loss_gate = sum(d["loss_gate"] for d in dets) / max(1, n_sup)
    return {"logits": logits, "task_loss": task_loss, "loss_gate": loss_gate, "details": dets}


class TransformerModel(nn.Module):
    """Encoder-only Transformer whose every attention is Dual-Expert.

    task = "cls": logits from the [CLS] position (D_alpha relation classification)
    task = "mlm": tied-embedding LM head (Wikipedia pre-training from scratch)
    The same token-level oracle alpha [B,T,T] is fed to every supervised layer.
    """

    def __init__(self, vocab_size: int, d_model=512, n_heads=8, n_layers=6, d_ff=2048,
                 max_len=512, dropout=0.1, n_classes=2, pad_id=0, task="cls",
                 supervise_layers: Optional[List[int]] = None):
        super().__init__()
        self.task, self.d_model = task, d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.seg_emb = nn.Embedding(2, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.cls_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Dropout(dropout),
                                      nn.Linear(d_model, n_classes))
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.mlm_head.weight = self.tok_emb.weight                      # weight tying
        nn.init.normal_(self.tok_emb.weight, std=d_model ** -0.5)
        self.supervise = set(range(n_layers)) if supervise_layers is None else \
            {i % n_layers for i in supervise_layers}

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, alpha_oracle=None,
                gamma_mix=0.0, target_idx=None, labels=None, return_details=False) -> Dict:
        key_mask = attention_mask.bool() if attention_mask is not None else None
        x = self.tok_emb(input_ids) * math.sqrt(self.d_model)
        if token_type_ids is not None:
            x = x + self.seg_emb(token_type_ids)
        x = self.pos(x)
        dets = []
        for i, layer in enumerate(self.layers):
            ao = alpha_oracle if i in self.supervise else None
            x, det = layer(x, key_mask, ao, gamma_mix, target_idx, return_details)
            dets.append(det)
        x = self.ln_f(x)
        logits = self.cls_head(x[:, 0]) if self.task == "cls" else self.mlm_head(x)
        return _finish(logits, labels, self.task, dets, len(self.supervise) if alpha_oracle is not None else 1)


# ----------------------------------------------------------------------------
# HF BERT with Dual-Expert self-attention
# ----------------------------------------------------------------------------
def _hf_mask_to_key_mask(m: torch.Tensor) -> torch.Tensor:
    if m.dim() == 4:
        m = m[:, 0, 0, :]
    return m if m.dtype == torch.bool else (m > -1.0)


class HFDualExpertSelfAttention(nn.Module):
    """Replaces BertSelfAttention. BertSelfOutput (dense+dropout+LN) is kept, so
    DualExpertAttention runs with out_proj=False. Q/K/V weights are copied from
    the pretrained BERT; W_rel/W_gate are fresh (orthogonal / Xavier)."""

    def __init__(self, hf_self, hidden_size: int, n_heads: int, dropout: float):
        super().__init__()
        self.dea = DualExpertAttention(hidden_size, n_heads, dropout=dropout, out_proj=False)
        with torch.no_grad():
            for src, dst in ((hf_self.query, self.dea.W_q), (hf_self.key, self.dea.W_k),
                             (hf_self.value, self.dea.W_v)):
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
        self.ctx: Dict = {}
        self.last: Optional[Dict] = None

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        c = self.ctx
        km = c.get("key_mask")
        if km is None and attention_mask is not None:
            km = _hf_mask_to_key_mask(attention_mask)
        det = self.dea(hidden_states, hidden_states, hidden_states,
                       alpha_oracle=c.get("alpha_oracle"), return_details=c.get("return_details", False),
                       gamma_mix=c.get("gamma_mix", 0.0), key_mask=km, target_idx=c.get("target_idx"))
        self.last = det
        return (det["output"], None)       # (context, attn_weights) - works for old & new HF


class DualExpertBert(nn.Module):
    """bert-base-uncased (hidden 768) with Dual-Expert attention in all 12 layers.
    d_model is kept at 768 (12 heads x 64), so no projection adapter is needed."""

    def __init__(self, name: str = "bert-base-uncased", n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        try:
            self.bert = AutoModel.from_pretrained(name, attn_implementation="eager")
        except (TypeError, ValueError):
            self.bert = AutoModel.from_pretrained(name)
        cfg = self.bert.config
        self.task = "cls"
        for layer in self.bert.encoder.layer:
            layer.attention.self = HFDualExpertSelfAttention(
                layer.attention.self, cfg.hidden_size, cfg.num_attention_heads,
                cfg.attention_probs_dropout_prob)
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cfg.hidden_size, n_classes))

    def _adapters(self):
        return [m for m in self.modules() if isinstance(m, HFDualExpertSelfAttention)]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, alpha_oracle=None,
                gamma_mix=0.0, target_idx=None, labels=None, return_details=False) -> Dict:
        ctx = dict(key_mask=attention_mask.bool() if attention_mask is not None else None,
                   alpha_oracle=alpha_oracle, gamma_mix=gamma_mix, target_idx=target_idx,
                   return_details=return_details)
        ads = self._adapters()
        for ad in ads:
            ad.ctx = ctx
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        dets = [ad.last for ad in ads]
        for ad in ads:
            ad.ctx, ad.last = {}, None
        logits = self.cls_head(out.last_hidden_state[:, 0])
        return _finish(logits, labels, "cls", dets, len(ads) if alpha_oracle is not None else 1)


# ----------------------------------------------------------------------------
# Optimisation helpers
# ----------------------------------------------------------------------------
def is_gate_param(name: str) -> bool:
    return any(k in name for k in GATE_PARAM_KEYS)


def gate_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if is_gate_param(n)]


def build_optimizer(model: nn.Module, lr_main=3e-4, lr_gate=1e-3, weight_decay=0.01):
    main, gate = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (gate if is_gate_param(n) else main).append(p)
    return torch.optim.AdamW([
        {"params": main, "lr": lr_main, "weight_decay": weight_decay, "name": "main(W_rel+CE)"},
        {"params": gate, "lr": lr_gate, "weight_decay": 0.0, "name": "gate(L_v2.8)"},
    ])


def clip_grads_per_group(optimizer, max_norm=1.0):
    return [float(torch.nn.utils.clip_grad_norm_(g["params"], max_norm)) for g in optimizer.param_groups]


def gamma_mix_schedule(step: int, total: int = 10_000) -> float:
    """gamma_mix: 1 (stable oracle) -> 0 (matches inference a_hat) over `total` steps."""
    return max(0.0, 1.0 - step / float(total))


def build_model(cfg: Dict, tokenizer, task: str, n_classes: int = 2) -> nn.Module:
    if cfg.get("data_source") == "hf":
        return DualExpertBert(cfg.get("hf_model", "bert-base-uncased"), n_classes, cfg.get("dropout", 0.1))
    return TransformerModel(
        vocab_size=tokenizer.vocab_size, d_model=cfg.get("d_model", 512), n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 6), d_ff=cfg.get("d_ff", 2048), max_len=max(512, cfg.get("max_len", 128)),
        dropout=cfg.get("dropout", 0.1), n_classes=n_classes, pad_id=tokenizer.pad_token_id, task=task)


def _grad_sum(params) -> float:
    return float(sum(p.grad.abs().sum() for p in params if p.grad is not None))


def verify_gradient_isolation(model: nn.Module, batch: Dict, gamma_mix: float = 0.5) -> None:
    """Backprop CE and L_gate separately and assert the two param sets are disjoint."""
    gate = gate_parameters(model)
    rel = [p for n, p in model.named_parameters() if "W_rel" in n]
    model.zero_grad(set_to_none=True)
    out = model(**batch, gamma_mix=gamma_mix)
    out["task_loss"].backward(retain_graph=True)
    g_gate, g_rel = _grad_sum(gate), _grad_sum(rel)
    model.zero_grad(set_to_none=True)
    out["loss_gate"].backward()
    g_gate2, g_rel2 = _grad_sum(gate), _grad_sum(rel)
    model.zero_grad(set_to_none=True)
    assert g_gate == 0.0, f"task CE leaked into W_gate/c_gate ({g_gate})"
    assert g_rel2 == 0.0, f"L_v2.8 leaked into W_rel ({g_rel2})"
    print(f"[isolation OK] CE -> W_rel |g|={g_rel:.3e}, W_gate |g|=0 ; "
          f"L_v2.8 -> W_gate |g|={g_gate2:.3e}, W_rel |g|=0")


if __name__ == "__main__":
    set_seed(42)
    B, T, V = 4, 12, 1000
    model = TransformerModel(V, n_layers=2, task="cls")
    ids = torch.randint(5, V, (B, T)); am = torch.ones(B, T, dtype=torch.long); am[0, -3:] = 0
    from data import mock_alpha
    alpha = mock_alpha((B, T, T))
    alpha[~(am.bool()[:, :, None] & am.bool()[:, None, :])] = float("nan")
    batch = dict(input_ids=ids, attention_mask=am, token_type_ids=torch.zeros_like(ids),
                 alpha_oracle=alpha, labels=torch.randint(0, 2, (B,)))
    out = model(**batch, gamma_mix=1.0)
    print("logits", tuple(out["logits"].shape), "task_loss", float(out["task_loss"]),
          "loss_gate", float(out["loss_gate"]))
    verify_gradient_isolation(model, batch)
    opt = build_optimizer(model)
    print("param groups:", [(g["name"], len(g["params"]), g["lr"]) for g in opt.param_groups])

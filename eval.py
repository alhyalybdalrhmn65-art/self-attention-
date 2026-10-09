"""
eval.py
=======
Evaluate a Dual-Expert checkpoint on the held-out split.

Metrics
  task       : accuracy (cls) or masked-token accuracy (mlm)
  alpha      : MAE, Pearson, Spearman of a_hat vs oracle alpha on labelled pairs
  bands      : agreement of band(a_hat) vs band(alpha), bands none<0.10<=weak<0.50<=strong
  decision   : photo-finish side accuracy = fraction of valid (t,p) pairs where
               a_hat sits on the same side of beta as the oracle alpha
Inference path only: gamma_mix = 0, so S_mix == S~ (scores use a_hat).

python eval.py --ckpt runs/dea/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import build_dataloaders, tokenizer_from_state
from transformer import build_model

BANDS = [(0.0, 0.05, "#d0e6f7", "pure sim 45%"), (0.15, 0.30, "#fde2b8", "weak ~0.2 25%"),
         (0.75, 0.95, "#d4f0d0", "strong ~0.9 30%")]


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _band(x):
    return np.digitize(x, [0.10, 0.50])


def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def plot_alpha_histogram(alpha_hat, alpha_oracle, path, title="alpha_hat vs oracle alpha"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lo, hi, c, lab in BANDS:
        ax.axvspan(lo, hi, color=c, alpha=0.6, label=lab)
    bins = np.linspace(0, 1, 51)
    if alpha_oracle is not None and len(alpha_oracle):
        ax.hist(alpha_oracle, bins=bins, density=True, alpha=0.55, color="tab:green", label="oracle alpha")
    if alpha_hat is not None and len(alpha_hat):
        ax.hist(alpha_hat, bins=bins, density=True, histtype="step", lw=2, color="tab:orange", label="alpha_hat")
    ax.axvline(0.95, color="red", ls="--", lw=1)
    ax.text(0.955, ax.get_ylim()[1] * 0.9, "cap 0.95\n(never 1.0)", color="red", fontsize=8)
    ax.set_xlabel("alpha"); ax.set_ylabel("density"); ax.set_title(title); ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, task, device, max_batches: Optional[int] = None, layer: int = -1,
             max_points: int = 200_000) -> Dict:
    model.eval()
    n_ok = n_tot = 0
    losses, ah_all, ao_all, side, n_pairs = [], [], [], [], 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = to_device(batch, device)
        out = model(**batch, gamma_mix=0.0)
        if out["task_loss"] is not None:
            losses.append(float(out["task_loss"]))
        if task == "cls":
            n_ok += int((out["logits"].argmax(-1) == batch["labels"]).sum()); n_tot += len(batch["labels"])
        else:
            m = batch["labels"] != -100
            n_ok += int((out["logits"].argmax(-1)[m] == batch["labels"][m]).sum()); n_tot += int(m.sum())
        det = out["details"][layer]
        if det.get("alpha_oracle") is not None:
            ao = det["alpha_oracle"]; has = ~torch.isnan(ao)
            ah_all.append(det["alpha_hat"][has].float().cpu()); ao_all.append(ao[has].float().cpu())
            st = det["stats"]
            if st.get("n_decision_pairs", 0) > 0:
                side.append(st["decision_side_acc"] * st["n_decision_pairs"]); n_pairs += st["n_decision_pairs"]
    res = {"task_loss": float(np.mean(losses)) if losses else float("nan"),
           "accuracy": n_ok / max(1, n_tot)}
    ah = torch.cat(ah_all).numpy() if ah_all else np.array([])
    ao = torch.cat(ao_all).numpy() if ao_all else np.array([])
    if len(ah) > max_points:
        idx = np.random.default_rng(0).choice(len(ah), max_points, replace=False); ah, ao = ah[idx], ao[idx]
    if len(ah) > 1:
        res.update(alpha_mae=float(np.abs(ah - ao).mean()),
                   alpha_pearson=float(np.corrcoef(ah, ao)[0, 1]) if ah.std() > 0 and ao.std() > 0 else float("nan"),
                   alpha_spearman=float(np.corrcoef(_rank(ah), _rank(ao))[0, 1]) if ah.std() > 0 and ao.std() > 0 else float("nan"),
                   band_agreement=float((_band(ah) == _band(ao)).mean()),
                   alpha_hat_mean=float(ah.mean()), alpha_oracle_mean=float(ao.mean()))
    res["decision_side_acc"] = sum(side) / n_pairs if n_pairs else float("nan")
    res["n_decision_pairs"] = n_pairs
    model.train()
    return {"metrics": res, "alpha_hat": ah, "alpha_oracle": ao}


def save_checkpoint(path, model, cfg, tokenizer, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "tokenizer": tokenizer.state(), "step": step}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    tok = tokenizer_from_state(ck["tokenizer"])
    cfg = ck["cfg"]
    task = "mlm" if cfg["data_source"] == "wikipedia" else "cls"
    model = build_model(cfg, tok, task, 2).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def paris_demo(model, tok, device="cpu", text="Paris is capital of France", q="paris", k="france"):
    enc = tok.encode_pair(text, None, max_len=32)
    toks = enc["tokens"]
    qi = toks.index(tok.tokenize(q)[-1]); ki = toks.index(tok.tokenize(k)[0])
    ids = torch.tensor([enc["input_ids"]], device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), token_type_ids=torch.zeros_like(ids))
    per_layer = [float(d["alpha_hat"][0, :, qi, ki].mean()) for d in out["details"]]
    print(f"[{text}] alpha_hat({q}->{k}) per layer: {[round(a, 3) for a in per_layer]}  (paper target ~0.9)")
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_batches", type=int, default=None)
    a = ap.parse_args()
    model, tok, cfg = load_checkpoint(a.ckpt, a.device)
    data = build_dataloaders(cfg["data_source"], cfg["batch_size"], cfg["max_len"], cfg["oracle"], cfg["seed"],
                             tokenizer=tok, n_docs=cfg["n_docs"], dalpha_source=cfg["dalpha_source"],
                             n_per_source=cfg["n_per_source"], wiki_alpha=cfg["wiki_alpha"],
                             oracle_cache=cfg.get("oracle_cache"))
    r = evaluate(model, data[a.split], data["task"], a.device, a.max_batches)
    out_dir = a.out_dir or os.path.dirname(a.ckpt)
    print(json.dumps(r["metrics"], indent=2))
    with open(os.path.join(out_dir, f"eval_{a.split}.json"), "w") as f:
        json.dump(r["metrics"], f, indent=2)
    plot_alpha_histogram(r["alpha_hat"], r["alpha_oracle"], os.path.join(out_dir, f"alpha_hist_{a.split}.png"),
                         f"{a.split}: alpha_hat vs oracle")
    try:
        paris_demo(model, tok, a.device)
    except ValueError:
        print("[eval] paris demo skipped (tokens not in vocab)")


if __name__ == "__main__":
    main()

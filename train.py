"""
train.py
========
Train Dual-Expert Attention.

  # from scratch on D_alpha pairs (synthetic fallback when offline), paper config
  python train.py --data_source dalpha --oracle mock
  # real D_alpha + offline DeBERTa/MiniLM oracle (cached to jsonl)
  python train.py --data_source dalpha --dalpha_source real --oracle deberta --oracle_cache cache/alpha.jsonl
  # MLM from scratch on streamed Wikipedia (10k docs)
  python train.py --data_source wikipedia --max_len 128
  # fine-tune bert-base-uncased with Dual-Expert attention (d_model=768, 12 heads)
  python train.py --data_source hf --batch_size 32 --lr_main 2e-5
  # CPU smoke test
  python train.py --quick

loss = task_CE(S_mix) + L_v2.8(gate). Gradient isolation lives in the graph
(see dual_expert_attention.py), param groups get separate lr and separate clip.
gamma_mix: 1 -> 0 over --gamma_steps (10k). Logs alpha_hat vs oracle histograms.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from data import build_dataloaders
from eval import evaluate, plot_alpha_histogram, save_checkpoint, to_device, paris_demo
from transformer import (CONFIG, build_model, build_optimizer, clip_grads_per_group, gamma_mix_schedule,
                         set_seed, verify_gradient_isolation)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_source", default="dalpha", choices=["wikipedia", "dalpha", "hf"])
    ap.add_argument("--dalpha_source", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--oracle", default="mock", help="mock | deberta | path/to/alpha.jsonl")
    ap.add_argument("--oracle_cache", default=None)
    ap.add_argument("--wiki_alpha", default="mock", choices=["mock", "none"])
    ap.add_argument("--tokenizer", default="bert-base-uncased", help="HF name or 'simple'")
    ap.add_argument("--hf_model", default="bert-base-uncased")
    ap.add_argument("--n_docs", type=int, default=10_000)
    ap.add_argument("--n_per_source", type=int, default=2000)
    ap.add_argument("--d_model", type=int, default=CONFIG["d_model"])
    ap.add_argument("--n_heads", type=int, default=CONFIG["n_heads"])
    ap.add_argument("--n_layers", type=int, default=CONFIG["n_layers"])
    ap.add_argument("--d_ff", type=int, default=CONFIG["d_ff"])
    ap.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--lr_main", type=float, default=CONFIG["lr_main"])
    ap.add_argument("--lr_gate", type=float, default=CONFIG["lr_gate"])
    ap.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"])
    ap.add_argument("--grad_clip", type=float, default=CONFIG["grad_clip"])
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--gamma_steps", type=int, default=CONFIG["gamma_mix_steps"])
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--out_dir", default="runs/dea")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_isolation_check", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny CPU smoke config")
    a = ap.parse_args()
    if a.quick:
        a.d_model, a.n_heads, a.n_layers, a.d_ff = 128, 4, 2, 256
        a.batch_size, a.steps, a.gamma_steps, a.eval_every, a.log_every = 32, 300, 200, 150, 25
        a.n_per_source, a.n_docs = 500, 200
    return a


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    a = parse_args()
    set_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = vars(a)
    if a.steps < a.gamma_steps:
        print(f"[warn] steps={a.steps} < gamma_steps={a.gamma_steps}: gamma_mix never reaches 0")

    data = build_dataloaders(a.data_source, a.batch_size, a.max_len, a.oracle, a.seed,
                             tokenizer_name=a.hf_model if a.data_source == "hf" else a.tokenizer,
                             n_docs=a.n_docs, dalpha_source=a.dalpha_source, n_per_source=a.n_per_source,
                             wiki_alpha=a.wiki_alpha, oracle_cache=a.oracle_cache)
    tok, task = data["tokenizer"], data["task"]
    model = build_model(cfg, tok, task, data["n_classes"]).to(a.device)
    opt = build_optimizer(model, a.lr_main, a.lr_gate, a.weight_decay)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | groups: "
          + ", ".join(f"{g['name']} n={len(g['params'])} lr={g['lr']}" for g in opt.param_groups))

    it = cycle(data["train"])
    if not a.no_isolation_check and (a.data_source != "wikipedia" or a.wiki_alpha == "mock"):
        verify_gradient_isolation(model, to_device(next(it), a.device))

    log = open(os.path.join(a.out_dir, "train_log.jsonl"), "w")
    hist_hat, hist_or = [], []
    model.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        gamma = gamma_mix_schedule(step - 1, a.gamma_steps)
        batch = to_device(next(it), a.device)
        out = model(**batch, gamma_mix=gamma)
        loss = out["task_loss"] + out["loss_gate"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms = clip_grads_per_group(opt, a.grad_clip)
        opt.step()

        det = out["details"][-1]
        if det.get("alpha_oracle") is not None:
            ao = det["alpha_oracle"]; has = ~torch.isnan(ao)
            hh, oo = det["alpha_hat"].detach()[has], ao[has]
            if hh.numel() > 4000:
                idx = torch.randperm(hh.numel(), device=hh.device)[:4000]; hh, oo = hh[idx], oo[idx]
            hist_hat.append(hh.float().cpu()); hist_or.append(oo.float().cpu())

        if step % a.log_every == 0 or step == 1:
            st = det.get("stats", {})
            rec = {"step": step, "gamma_mix": round(gamma, 4), "task_loss": float(out["task_loss"]),
                   "loss_gate": float(out["loss_gate"]), "grad_norm_main": gnorms[0], "grad_norm_gate": gnorms[1],
                   **{k: v for k, v in st.items()}}
            if hist_hat:
                h, o = hist_hat[-1], hist_or[-1]
                rec.update(alpha_hat_mean=float(h.mean()), alpha_mae=float((h - o).abs().mean()))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step:6d} | g_mix {gamma:.3f} | CE {rec['task_loss']:.4f} | L_gate {rec['loss_gate']:.4f}"
                  f" | a_hat {rec.get('alpha_hat_mean', float('nan')):.3f} | MAE {rec.get('alpha_mae', float('nan')):.3f}"
                  f" | side {st.get('decision_side_acc', float('nan')):.3f} | {time.time() - t0:.0f}s")

        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(model, data["val"], task, a.device, max_batches=50)
            print(f"[eval step {step}] " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in r["metrics"].items()}))
            log.write(json.dumps({"step": step, "eval": r["metrics"]}) + "\n"); log.flush()
            if hist_hat:
                plot_alpha_histogram(torch.cat(hist_hat).numpy(), torch.cat(hist_or).numpy(),
                                     os.path.join(a.out_dir, f"alpha_hist_train_step{step}.png"),
                                     f"train step {step}: alpha_hat vs oracle (g_mix={gamma:.2f})")
                hist_hat, hist_or = [], []
            plot_alpha_histogram(r["alpha_hat"], r["alpha_oracle"],
                                 os.path.join(a.out_dir, f"alpha_hist_val_step{step}.png"), f"val step {step}")
            save_checkpoint(os.path.join(a.out_dir, "checkpoint.pt"), model, cfg, tok, step)

    log.close()
    print(f"done. checkpoint + logs in {a.out_dir}")
    try:
        paris_demo(model, tok, a.device)
    except ValueError:
        pass


if __name__ == "__main__":
    main()

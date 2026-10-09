"""
visualize.py  (for researchers)
===============================
Writes an attention_analysis/ folder:

  heatmap_E0tilde_<slug>.png    E~0 per head [H, Q, K]   (looks-like, normalised)
  heatmap_E1tilde_<slug>.png    E~1 per head [H, Q, K]   (works-like, normalised)
  heatmap_S_tilde_<slug>.png    S~ = (1-a_hat) E~0 + a_hat E~1 per head
  photo_finish_<slug>.png       S*_t(a') vs S*_p(a'), a' in [0,1]; beta crossing,
                                oracle alpha and a_hat marked
  alpha_histogram.png           a_hat over all token pairs vs the paper's graded density
  spotlight_w_vs_db.png         w(d_b) curve + real (d_b, w) pairs; signed L_decision
  alpha_hat_vs_oracle.png       per-sentence a_hat (per head) vs oracle alpha
  summary.json

Demo inputs: "Paris is capital of France", "car→wheel", "king→queen", "5+3=8", "dog→dogs".
Without --ckpt a fresh 2-layer model is built and ONLY its gate is warmed up on the
5 oracle labels (W_gate/c_gate, L_v2.8), so a_hat reflects the gate fit, not
task-trained W_rel. File names use ASCII (E0tilde) instead of 'Ẽ0' for portability.

python visualize.py                       # fresh model + gate warm-up
python visualize.py --ckpt runs/dea/checkpoint.pt --layer -1 --head 0
python visualize.py --oracle deberta      # real oracle alpha for the 5 pairs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import SimpleTokenizer, mock_alpha, mock_corpus, precompute_oracle_alpha
from dual_expert_attention import BETA_EPS, LAMBDA_D, LAMBDA_S, TAU_D, TAU_S, S_BETA_TEMP
from eval import BANDS, load_checkpoint
from transformer import TransformerModel, gate_parameters, set_seed

# (sentence, query word, key word, illustrative oracle alpha inside the paper's band)
DEMO = [("Paris is capital of France", "paris", "france", 0.90),
        ("car→wheel", "car", "wheel", 0.90),
        ("king→queen", "king", "queen", 0.90),
        ("5+3=8", "5+3", "8", 0.90),
        ("dog→dogs", "dog", "dogs", 0.02)]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def find_span(tokens, sub, start=0):
    for i in range(start, len(tokens) - len(sub) + 1):
        if tokens[i:i + len(sub)] == sub:
            return i, i + len(sub)
    raise ValueError(f"{sub} not found in {tokens}")


def build_demo_batch(tok, demo, device):
    encs, meta = [], []
    for text, qw, kw, a in demo:
        e = tok.encode_pair(text, None, max_len=32)
        qs = find_span(e["tokens"], tok.tokenize(qw), 1)
        ks = find_span(e["tokens"], tok.tokenize(kw), qs[1])
        encs.append(e)
        meta.append(dict(text=text, q=qw, k=kw, alpha=a, qi=qs[1] - 1, ki=ks[0], tokens=e["tokens"], L=len(e["tokens"])))
    N, T = len(encs), max(m["L"] for m in meta)
    ids = torch.full((N, T), tok.pad_token_id, dtype=torch.long)
    am = torch.zeros((N, T), dtype=torch.long)
    alpha = torch.full((N, T, T), float("nan"))
    tgt = torch.full((N, T), -1, dtype=torch.long)
    for i, (e, m) in enumerate(zip(encs, meta)):
        ids[i, :m["L"]] = torch.tensor(e["input_ids"]); am[i, :m["L"]] = 1
        alpha[i, m["qi"], m["ki"]] = m["alpha"]; tgt[i, m["qi"]] = m["ki"]
    batch = dict(input_ids=ids, attention_mask=am, token_type_ids=torch.zeros_like(ids),
                 alpha_oracle=alpha, target_idx=tgt)
    return {k: v.to(device) for k, v in batch.items()}, meta


def warmup_gate(model, batch, steps=300, lr=1e-2):
    gp = set(id(p) for p in gate_parameters(model))
    frozen = [p for p in model.parameters() if id(p) not in gp and p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    opt = torch.optim.Adam(gate_parameters(model), lr=lr)
    for s in range(steps):
        out = model(**batch, gamma_mix=0.0)
        opt.zero_grad(); out["loss_gate"].backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"  gate warm-up step {s:4d}  L_v2.8={float(out['loss_gate']):.4f}")
    for p in frozen:
        p.requires_grad_(True)


def heatmaps(det, i, m, out_dir, H):
    L, toks = m["L"], m["tokens"]
    for key, name, cmap in (("E0_norm", "E0tilde", "Blues"), ("E1_norm", "E1tilde", "Greens"),
                            ("S_tilde", "S_tilde", "magma")):
        X = det[key][i, :, :L, :L].cpu().numpy()
        nc = min(4, H); nr = math.ceil(H / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.0 * nr), squeeze=False)
        lab = {"E0tilde": r"$\tilde{E}_0$", "E1tilde": r"$\tilde{E}_1$", "S_tilde": r"$\tilde{S}$"}[name]
        for h in range(nr * nc):
            ax = axes[h // nc][h % nc]
            if h >= H:
                ax.axis("off"); continue
            im = ax.imshow(X[h], cmap=cmap, aspect="auto")
            ax.set_xticks(range(L)); ax.set_xticklabels(toks, rotation=90, fontsize=7)
            ax.set_yticks(range(L)); ax.set_yticklabels(toks, fontsize=7)
            ax.add_patch(plt.Rectangle((m["ki"] - .5, m["qi"] - .5), 1, 1, fill=False, ec="red", lw=1.5))
            ax.set_title(f"head {h}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{lab}  [H,Q,K]  '{m['text']}'  (red = {m['q']}→{m['k']})", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"heatmap_{name}_{slug(m['text'])}.png"), dpi=120)
        plt.close(fig)


def photo_finish(det, i, m, h, out_dir):
    L, t, qi = m["L"], m["ki"], m["qi"]
    e0 = det["E0_norm"][i, h, qi, :L].cpu().numpy(); e1 = det["E1_norm"][i, h, qi, :L].cpu().numpy()
    a, ah = m["alpha"], float(det["alpha_hat"][i, h, qi, t])
    best, best_beta = None, None
    for p in range(L):
        if p == t:
            continue
        den = (e1[t] - e0[t]) - (e1[p] - e0[p])
        if abs(den) < BETA_EPS:
            continue
        beta = (e0[p] - e0[t]) / (den + BETA_EPS * np.sign(den))
        if 0 <= beta <= 1 and (best_beta is None or abs(beta - a) < abs(best_beta - a)):
            best, best_beta = p, beta
    if best is None:                                  # no crossing: show strongest rival
        S = (1 - a) * e0 + a * e1; S[t] = -np.inf; best = int(np.argmax(S))
    x = np.linspace(0, 1, 201)
    St, Sp = (1 - x) * e0[t] + x * e1[t], (1 - x) * e0[best] + x * e1[best]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(0.95, 1.0, color="red", alpha=0.08, label="never (alpha>0.95)")
    ax.plot(x, St, lw=2.2, label=f"S*_t  target '{m['tokens'][t]}'")
    ax.plot(x, Sp, lw=2.2, ls="--", label=f"S*_p  distractor '{m['tokens'][best]}'")
    if best_beta is not None:
        yb = (1 - best_beta) * e0[t] + best_beta * e1[t]
        ax.plot([best_beta], [yb], "ko", ms=8); ax.axvline(best_beta, color="k", lw=1, ls=":")
        ax.annotate(f"β={best_beta:.3f}\nphoto-finish", (best_beta, yb), textcoords="offset points",
                    xytext=(8, 10), fontsize=9)
    else:
        ax.text(0.02, 0.95, "no crossing in [0,1]: one horse always wins", transform=ax.transAxes, fontsize=9, va="top")
    ax.axvline(a, color="tab:green", lw=2, label=f"oracle α={a:.2f}")
    ax.axvline(ah, color="tab:orange", lw=2, ls="-.", label=f"α̂={ah:.3f}")
    ax.set_xlabel("α' (DJ mixer knob)"); ax.set_ylabel("score S*(α')")
    ax.set_title(f"'{m['text']}'  query '{m['tokens'][qi]}', head {h}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"photo_finish_{slug(m['text'])}.png"), dpi=130); plt.close(fig)
    return {"distractor": m["tokens"][best], "beta": None if best_beta is None else float(best_beta)}


def alpha_histogram(det, batch, meta, out_dir):
    am = batch["attention_mask"].bool()
    valid = (am[:, :, None] & am[:, None, :]).unsqueeze(1).expand_as(det["alpha_hat"])
    ah = det["alpha_hat"][valid].cpu().numpy()
    ref = mock_alpha((200_000,)).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax in axes:
        for lo, hi, c, lab in BANDS:
            ax.axvspan(lo, hi, color=c, alpha=0.6, label=lab)
        ax.axvline(0.95, color="red", ls="--", lw=1)
    bins = np.linspace(0, 1, 51)
    axes[0].hist(ref, bins=bins, density=True, color="tab:green", alpha=0.7, label="graded oracle density")
    axes[0].set_title("oracle α density (paper: 45% / 25% / 30%)")
    axes[1].hist(ah, bins=bins, density=True, color="tab:orange", alpha=0.75, label="α̂, all token pairs, all heads")
    for m, i in zip(meta, range(len(meta))):
        v = float(det["alpha_hat"][i, :, m["qi"], m["ki"]].mean())
        axes[1].axvline(v, color="k", lw=1); axes[1].text(v, axes[1].get_ylim()[1] * 0.8, f" {m['q']}→{m['k']}",
                                                          rotation=90, fontsize=7, va="top")
    axes[1].set_title("α̂ on the demo sentences (labelled pairs marked)")
    for ax in axes:
        ax.set_xlabel("α"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_histogram.png"), dpi=130); plt.close(fig)


def spotlight(det, out_dir):
    d = np.linspace(0.01, 1.0, 300)
    w = 1 + LAMBDA_S * np.exp(-d / TAU_S)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(d, w, lw=2, label=r"$w = 1 + \lambda_s e^{-d_b/\tau_s}$  ($\lambda_s$=2, $\tau_s$=0.1)")
    if det.get("d_b_list") is not None:
        db, ww = det["d_b_list"].cpu().numpy().ravel(), det["w_list"].cpu().numpy().ravel()
        ok = ~np.isnan(db)
        axes[0].scatter(db[ok], ww[ok], s=10, c="tab:red", alpha=0.6, label=f"demo (t,p) pairs, n={int(ok.sum())}")
    axes[0].axhline(1, color="gray", lw=0.8, ls=":")
    axes[0].set_xlabel(r"$d_b = \sqrt{(\alpha-sg[\beta])^2 + 10^{-4}}$"); axes[0].set_ylabel("spotlight w")
    axes[0].set_title("spotlight near the cliff"); axes[0].legend(fontsize=8)
    ahat = np.linspace(0, 0.95, 300)
    for a, beta, c in ((0.9, 0.6, "tab:green"), (0.2, 0.5, "tab:blue"), (0.62, 0.6, "tab:purple")):
        s = np.tanh((a - beta) / S_BETA_TEMP)
        ld = np.log1p(np.exp(-s * (ahat - beta) / TAU_D))
        axes[1].plot(ahat, LAMBDA_D * ld, c=c, lw=2, label=f"α={a}, β={beta}  (s_β={s:+.2f})")
        axes[1].axvline(beta, c=c, ls=":", lw=1)
    axes[1].set_xlabel("α̂"); axes[1].set_ylabel(r"$\lambda_d\,L_{decision}$ (signed $s_\beta$, softplus)")
    axes[1].set_title("decision loss: α̂ must sit on α's side of β"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "spotlight_w_vs_db.png"), dpi=130); plt.close(fig)


def alpha_bars(det, meta, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(meta))
    per_head = [det["alpha_hat"][i, :, m["qi"], m["ki"]].cpu().numpy() for i, m in enumerate(meta)]
    ax.bar(x - 0.18, [m["alpha"] for m in meta], 0.36, label="oracle α", color="tab:green")
    ax.bar(x + 0.18, [ph.mean() for ph in per_head], 0.36, label="α̂ (mean over heads)", color="tab:orange")
    for i, ph in enumerate(per_head):
        ax.scatter(np.full(len(ph), i + 0.18), ph, s=8, c="k", zorder=3)
    ax.axhline(0.95, color="red", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{m['q']}→{m['k']}" for m in meta])
    ax.set_ylim(0, 1); ax.set_ylabel("α"); ax.legend(); ax.set_title("α̂ vs oracle α (dots = individual heads)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_hat_vs_oracle.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_dir", default="attention_analysis")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--oracle", default="demo", choices=["demo", "deberta"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    set_seed(42)
    os.makedirs(a.out_dir, exist_ok=True)

    demo = list(DEMO)
    if a.oracle == "deberta":
        recs = precompute_oracle_alpha([{"q_text": q, "k_text": k} for _, q, k, _ in demo], device=a.device)
        demo = [(t, q, k, r["alpha"]) for (t, q, k, _), r in zip(demo, recs)]

    if a.ckpt:
        model, tok, _ = load_checkpoint(a.ckpt, a.device)
        warm = a.warmup_steps or 0
    else:
        tok = SimpleTokenizer().build_vocab(mock_corpus())
        model = TransformerModel(tok.vocab_size, d_model=512, n_heads=8, n_layers=2, task="cls").to(a.device)
        warm = 300 if a.warmup_steps is None else a.warmup_steps
    model.eval()
    batch, meta = build_demo_batch(tok, demo, a.device)
    if warm:
        print(f"warming up the gate only ({warm} steps, W_gate/c_gate on L_v2.8)")
        warmup_gate(model, batch, warm)
    with torch.no_grad():
        out = model(**batch, gamma_mix=0.0, return_details=True)
    det = out["details"][a.layer]
    H = det["alpha_hat"].shape[1]

    summary = []
    for i, m in enumerate(meta):
        heatmaps(det, i, m, a.out_dir, H)
        pf = photo_finish(det, i, m, a.head % H, a.out_dir)
        ah = det["alpha_hat"][i, :, m["qi"], m["ki"]]
        summary.append({"text": m["text"], "pair": f"{m['q']}→{m['k']}", "oracle_alpha": m["alpha"],
                        "alpha_hat_mean": round(float(ah.mean()), 4),
                        "alpha_hat_per_head": [round(float(v), 4) for v in ah], **pf})
        print(f"{m['text']:<28s} {m['q']}→{m['k']:<7s} oracle α={m['alpha']:.2f}  α̂={float(ah.mean()):.3f}"
              f"  β(head {a.head % H}, vs '{pf['distractor']}')={pf['beta']}")
    alpha_histogram(det, batch, meta, a.out_dir)
    spotlight(det, a.out_dir)
    alpha_bars(det, meta, a.out_dir)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    n = len([f for f in os.listdir(a.out_dir) if f.endswith(".png")])
    print(f"wrote {n} PNGs to {a.out_dir}/")


if __name__ == "__main__":
    main()

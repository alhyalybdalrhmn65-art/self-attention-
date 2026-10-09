---
license: cc-by-4.0
language: [en, ar]
tags: [attention, dual-expert, relational-reasoning, graded-alpha, pytorch]
---
# Dual-Expert Attention v2.8.3: reference PyTorch implementation

**Learning When to Look Like vs. Work Like: A Dual-Expert Attention with a Graded Alpha Knob**
Abdulrahman Al-Hayali, Mosul, Iraq, v2.8.3 (May 2026). This repo implements Eq. (1)-(15) with the reviewer fixes, as a full Transformer with **no standard attention anywhere**.

## The idea in one screen

Standard attention asks one question: *do q and k look alike?* (`E0 = q·k/√d`). Great for `dog → dogs`, useless for `Paris → France` or `5+3 → 8`: they don't look alike, they **work together**. So every head has two experts:

| expert | formula | meaning |
|---|---|---|
| E0, looks-like | `q·k / √d_head` | unbounded similarity |
| E1, works-like | `-(1/d_head) ‖W_rel q − k‖²` | lock-and-key: does my key `W_rel` turn q into k? Best = 0 |

The scales are different (E0 unbounded, E1 ≤ 0), so each one is **normalised per query across keys with its own stats** (`Ẽ = (E − μ_k)/(σ_k + 1e-5)`). That is *not* `nn.LayerNorm(d_model)`, and all mixing uses Ẽ, never raw E.

### The DJ mixer α
`S*(q,k;α) = (1−α)·Ẽ0 + α·Ẽ1`. α is a **graded, continuous** knob, not a 0.2/0.9 label:

| α | mix | example | the feel |
|---|---|---|---|
| ≈0 `[0, 0.05]`, 45% of mass | 100% looks | `dog → dogs` | twins |
| ~0.2 `[0.15, 0.30]`, 25% | 80% looks / 20% works | `car → road` | **the cousin who looks like you but is actually your accountant** |
| ~0.9 `[0.75, 0.95]`, 30% | 10% looks / 90% works | `king → queen`, `car → wheel`, `Paris → France`, `5+3 → 8` | relation dominates |

**Never 1.0.** In March, α=1 made retrieval collapse (`queen` started retrieving `queen bee`), because pure relation forgets what the thing *is*. Both α and α̂ are clamped to `[0, 0.95]`, so ≥5% similarity anchor always survives.

The oracle α comes from a **frozen** library, never from the trainable model (otherwise W_rel could cheat):
`α = σ(Φ_rel − γ_nli·Sim)`, γ_nli = 0.3, Φ_rel = DeBERTa-v3-large-MNLI entailment logit (premise=text(q), hypothesis=text(k)), Sim = all-MiniLM cosine. It's pre-computed **offline** with `torch.no_grad()` (calling DeBERTa inside an O(N²) attention is infeasible), optionally isotonic-calibrated on human labels.

### Predicting α̂
`α̂ = σ(clip(W_gate[q; k; Ẽ0; Ẽ1] + c_gate, −10, 10))`, then clamp to 0.95. The **logit** is clipped (not the weights) so `α̂(1−α̂)` never vanishes. W_gate is per head, `1×(2·d_head+2)`, Xavier init; `c_gate = −0.5 ≈ σ⁻¹(mean α)`. The concat is never materialised: the linear map splits exactly into `q·w_q + k·w_k + w_E0·Ẽ0 + w_E1·Ẽ1`.

### β, the photo-finish
For a query, target t and distractor p are two straight lines as you turn the knob:
`S*_t(α') = (1−α')Ẽ0,t + α'Ẽ1,t`, `S*_p(α') = (1−α')Ẽ0,p + α'Ẽ1,p`.
They cross at
`β = (Ẽ0,p − Ẽ0,t) / (D + ε·sign(D))`, `D = (Ẽ1,t − Ẽ0,t) − (Ẽ1,p − Ẽ0,p)`, ε = 1e-6.
|D| < ε means parallel lines (β = None); β ∉ [0,1] means one horse always wins (skip). β depends only on Ẽ0, Ẽ1, never on the oracle. This was the v2.7 "3-day flip-flop bug": not a bug, just two horses crossing.

### Spotlight + decision loss (training only)
```
d_b        = sqrt((α − sg[β])² + 1e-4)
w          = sg[1 + 2·exp(−d_b / 0.1)]                 # shine light near the cliff
s_β        = tanh((α − sg[β]) / 0.05)                   # SIGNED (no abs)
L_decision = softplus(−s_β · (α̂ − sg[β]) / 0.1)        # softplus, hinge exploded in v2.7
L_v2.8     = w·(α̂ − α)² + 0.5·L_decision
```
If α is left of β, α̂ must be left too, otherwise the winner flips.

### Exposure-bias fix
`S_mix = (1−α_mix)Ẽ0 + α_mix Ẽ1`, `α_mix = γ_mix·α + (1−γ_mix)·α̂`, γ_mix annealed **1 → 0 over 10k steps**: early training uses the stable oracle, late training matches inference (fixes the 8% drop).

### Gradient isolation
* **W_rel** (and W_q/W_k/W_v, FFN, embeddings) learn **only from task CE** on S_mix.
* **W_gate, c_gate** learn **only from L_v2.8**.

This is enforced in the graph: the gate only sees *detached* q, k, Ẽ0, Ẽ1; the scores only use a *detached* α̂; β and w are detached. So one `backward()` of `CE + L_gate` equals two separate backward passes. On top of that the optimizer has two AdamW groups (3e-4 main, 1e-3 gate), clipped **separately** at 1.0 (a joint clip would couple them). `verify_gradient_isolation()` asserts it on the first batch of every run.

## Files
| file | what |
|---|---|
| `dual_expert_attention.py` | `DualExpertAttention`, Eq (1)-(15). `python dual_expert_attention.py` runs the Paris demo + isolation check |
| `transformer.py` | `PositionalEncoding`, `EncoderLayer`, `TransformerModel` (cls/mlm), `DualExpertBert` (bert-base-uncased, 768-d, 12 heads), optimizer, γ schedule |
| `data.py` | Wikipedia streaming / D_α (bAbI 16-18, CLUTRR, ConceptNet IsA/PartOf/Capitals, 1:1 negatives, 10/1/1 split), offline DeBERTa+MiniLM oracle, `mock_alpha()` |
| `train.py` | training loop, γ_mix annealing, α̂-vs-α histograms, checkpoints |
| `eval.py` | accuracy, α MAE/Pearson/Spearman, band agreement, photo-finish side accuracy, Paris demo |
| `visualize.py` | writes `attention_analysis/` (heatmaps of Ẽ0, Ẽ1, S̃ per head, photo-finish lines, α histogram, spotlight) |

## Quickstart
```bash
pip install -r requirements.txt
python dual_expert_attention.py            # Paris→France α̂ ≈ 0.9 after gate fit + isolation OK
python transformer.py                      # shapes + isolation + param groups
python train.py --quick                    # CPU smoke test
python train.py --data_source dalpha --oracle mock                          # paper config, synthetic D_α
python train.py --data_source dalpha --dalpha_source real --oracle deberta --oracle_cache cache/alpha.jsonl
python train.py --data_source wikipedia --max_len 128                       # MLM from scratch
python train.py --data_source hf --batch_size 32 --lr_main 2e-5             # fine-tune BERT
python eval.py --ckpt runs/dea/checkpoint.pt
python visualize.py [--ckpt runs/dea/checkpoint.pt] [--oracle deberta]
```
Defaults follow §8: d_model 512, 8 heads, AdamW 3e-4 (W_rel, orthogonal) / 1e-3 (W_gate, Xavier), c_gate −0.5, clip 1.0, batch 128, seed 42.

## Implementation decisions you should know about
1. **W_rel is per head**, `[n_heads, d_head, d_head]` (8×64×64 = 32k params/layer, not 262k), and both experts scale by `d_head` (= d_k).
2. **Softmax is applied directly to S** (Ẽ are already unit-scale), with no extra `1/√d`.
3. **Token-level oracle.** The pair α is written into the `[premise tokens × hypothesis tokens]` block; every other cell is NaN (no label). Unlabelled cells fall back to α̂ in `α_mix` and are excluded from L_v2.8.
4. **Gold key t.** Pass `target_idx`; if omitted, t = oracle winner `argmax_k S*(q,k;α)` among labelled keys.
5. **Gate loss** = dense `(α̂−α)²` on all labelled pairs (w = 1, where β = None) + mean over valid (t,p) of the spotlight + decision term.
6. **Wikipedia + mock α** only exercises the plumbing (random α carries no signal). Use D_α + DeBERTa for real results.
7. Real dataset loaders (bAbI/CLUTRR/ConceptNet on the HF hub) are best-effort; hub names move. On failure each source falls back to a synthetic generator, so the run never crashes. The synthetic ConceptNet bucket is a few hundred hand-written pairs, not 4k.
8. Heatmap file names use `E0tilde` instead of `Ẽ0` for filesystem portability.

## Discrepancies found in the source documents (resolved as listed)
| item | paper / model card | this code |
|---|---|---|
| α_mix order | model card: `(1−γ)α + γα̂` (would start on α̂) | `γ·α + (1−γ)·α̂`, γ 1→0 (paper §6) |
| s_β | model card: `tanh(|α−β|/0.05)` | signed, no abs (paper Eq 13) |
| ε·sign(·) in β | paper: "D = numerator diff" | sign(denominator) |
| E0/E1 scale | `d_model` | `d_head` |
| gate input / mixing | model card quickstart feeds raw E0,E1 to the gate and clamps after the linear layer | normalised Ẽ everywhere, clip logit before σ |
| strong band | [0.75,0.95] / [0.80,0.95] / [0.87,0.93] | mock uses [0.75,0.95] |
| weak band | [0.15,0.30] / [0.15,0.35] | mock uses [0.15,0.30] |
| licence | model card MIT, prompt CC-BY-4.0 | CC-BY-4.0 here, please pick one |

## Limitations
D_α is small (12k), ConceptNet covers 3 relations, and the oracle inherits DeBERTa/MiniLM biases (IsA may carry gender stereotypes). Random negatives are easy, not hard negatives near β. Memory is O(B·H·T²) for several tensors (E0, E1, Ẽ0, Ẽ1, α̂, β), so it's fine for T ≤ 256; longer contexts need chunking. No d=4096 scaling test yet.

## Citation
```bibtex
@misc{alhayali2026dualexpert,
  title  = {Learning When to Look Like vs. Work Like: A Dual-Expert Attention with a Graded Alpha Knob},
  author = {Al-Hayali, Abdulrahman}, year = {2026}, note = {v2.8.3, Mosul, Iraq. Independent researcher (azl\_y80)}
}
```

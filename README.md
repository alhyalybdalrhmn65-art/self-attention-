---
license: mit
language:
- en
- ar
tags:
- attention
- self-attention
- dual-expert
- relational-reasoning
- deberta
- babi
- clutrr
- conceptnet
- graded-alpha
library_name: pytorch
pipeline_tag: feature-extraction
---

# Learning When to Look Like vs. Work Like: Dual-Expert Attention with a Graded Alpha Knob
### v2.8.3 Human-Clear Final — Abdulrahman Al-Hayali — Mosul, Iraq — May 2026
**Contact:** `azl_y80` | Independent Researcher

> **⚠️ IMPORTANT: α is NOT fixed 0.2 or 0.9. This was the biggest misunderstanding in v2.7.**
> Those numbers were shorthand I wrote in my notebook for 80/20 vs 10/90.
> Real α is **graded, adaptive, continuous in [0,1]** evaluated by a frozen library L.
> `α(q,k) = σ( Φ_rel - γ_nli·Sim )` — peaks at `0` (45%), `~0.2=[0.15-0.30]` (25%), `~0.9=[0.80-0.95]` (30%) after isotonic regression on 2k human labels (κ=0.78).

## TL;DR — My Story

Standard attention asks one question: `E0 = q·k / √d` — do you look like me? Works for `dog → dogs`, fails for `Paris → France` or `5+3 → 8`.

I split it into two experts:
- **E0 = looks-like**
- **E1 = works-like = -(1/d)||W_rel q - k||²** — lock-and-key, does my key `W_rel` turn `q` into `k`? Best is 0 when `W_rel q = k`.

Then I needed a knob α. In March my retrieval collapsed — `queen` started retrieving `queen bee` — so I never use α=1.0. I keep 10% similarity as anchor.

This repo is the clean version after I wasted 3 days in v2.7 because target and distractor kept flip-flopping.

## How It Feels — The DJ Mixer

Think of α as a DJ mixer knob:

| α | What it means | Example | My shorthand note |
|---|---|---|---|
| **0** `[0, 0.05]` — 45% mass | Pure similarity | `dog → dogs` | Looks alike |
| **~0.2** `[0.15, 0.30]` — 25% mass | Weak relation — 80% looks, 20% works | `car → road` | Cousin who looks like you but is your accountant |
| **~0.9** `[0.87, 0.93]` — 30% mass | Strong relation — 10% looks, 90% works | `car → wheel`, `king → queen`, `Paris → France`, `5+3 → 8` |  |

**Never 1.0** — March collapse story above.

## Method

### 1. Two Experts — Why I needed them

```python
# v2.5 lesson: I removed sqrt(d_model) and gradients exploded
E0 = dot(q, k) / sqrt(d_model)  # (1)

# v2.4 lesson: q^T W k is symmetric, didn't work for directed Paris->France
E1 = -(1/d_model) * ||W_rel @ q - k||^2  # (2)
# W_rel orth init, d_model=512
```

Scales differ — E0 unbounded, E1 ≤0. I was confused for a week why E1 always dominated, until LayerNorm per query across keys:

```
Ẽ0 = (E0 - μ_q)/σ_q , Ẽ1 = (E1 - μ_q)/σ_q  # (2b)
```

### 2. Oracle α — Graded, Adaptive, Continuous

```
S*(q,k; α) = (1-α) Ẽ0 + α Ẽ1  # (3) DJ mixer

α(q,k) = L(q,k) = σ( Φ_rel(q,k) - γ_nli·Sim(q,k) ) ∈ [0,1]  # (4)
γ_nli = 0.3
Φ_rel = DeBERTa-v3-large MNLI logit, premise=text(q), hypothesis=text(k)
Sim = cosine(all-MiniLM) proxies E0 strength
```

L is frozen, not dependent on trainable W_rel — otherwise model cheats. [I used to call it γ and reused γ for annealing — my bad, now split to γ_nli and γ_mix]

Density peaks from isotonic regression, NOT hard bins.

### 3. Predicting α — My first attempt failed

```python
# α̂ = σ(W_θ q + b_θ)  # (5) collapsed to mean, useless
α̂(q,k) = σ( W_gate [q; k; E0; E1] + c_gate )  # (6)
# W_gate ∈ R^{1×(2d+2)}, clipped to [-10,10] else α̂(1-α̂)→0
S̃ = (1-α̂)E0 + α̂E1  # (7)
```

### 4. Decision Boundary β — Where Two Horses Cross

The 3-day bug in v2.7: S*_t and S*_p are just two lines crossing. Flip where they cross.

```
S*_t(α') = (1-α')Ẽ0,t + α'Ẽ1,t
S*_p(α') = (1-α')Ẽ0,p + α'Ẽ1,p  # (8)
Flip where S*_t = S*_p  # (9)
β_{t,p} = (Ẽ0,p - Ẽ0,t) / [ (Ẽ1,t-Ẽ0,t)-(Ẽ1,p-Ẽ0,p) + ε·sign(D) ]  # (10)
ε=1e-6 — early version I divided by zero and got NaNs for an epoch
If |D|<ε → parallel → β=None. If β∉[0,1] → target dominates always → skip
```

β is photo-finish where two horses cross. Independent of oracle α — depends only on Ẽ0,Ẽ1.

### 5. Spotlight Near The Cliff

If oracle α is near β, small error flips winner.

```
d_b = sqrt(|α - sg[β]|² + 1e-4)  # (11) sg=stop-gradient — I used to let W_rel move β away to cheat
w_{t,p} = sg[1 + λ_s exp(-d_b/τ_s)]  # (12) λ_s=2.0, τ_s=0.1 — spotlight near cliff
s_β = tanh(|α - sg[β]|/0.05)  # (13)
L_decision = softplus( -s_β (α̂ - sg[β])/τ_d )  # (14) τ_d=0.1 — hinge in v2.7 exploded
```

w = shine more light near cliff. L_decision = don't flip winner near photo-finish.

### 6. Final Loss & Annealing

```
L_{v2.8} = w_{t,p} (α̂ - α)² + λ_d L_decision  # (15) λ_d=0.5
S_mix = (1-α_mix)Ẽ0 + α_mix Ẽ1
α_mix = (1-γ_mix)α + γ_mix α̂, γ_mix annealed 1→0 over 10k steps
```

Fixes 8% drop from exposure bias. Early uses stable oracle, later matches inference. W_rel gets grads from task CE on S_mix, NOT from w or β thanks to sg[]. W_gate gets grads only from L_{v2.8}.

## Reproducibility

```
d_model=512, heads=8
AdamW lr 3e-4 for W_rel (orth init), 1e-3 for W_gate (Xavier)
c_gate init σ⁻¹(mean α)≈-0.5, grad clip 1.0, batch 128, seed 42
```

**D_α=12k pairs:** bAbI 16-18 4k, CLUTRR 4k, ConceptNet 4k (IsA, PartOf, Capitals). Negative 1:1 random keys. Split 10k/1k/1k. Human 2k subset for calibration κ=0.78.

**Cost:** Train = DeBERTa-v3-large MNLI + all-MiniLM once for oracle α. Inference = +262k params W_rel (512×512) + 1026 params W_gate, ~0.5µs/pair. β, w, spotlight are training-only.

## Quickstart

```python
import torch
import torch.nn as nn

class DualExpertAttention(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.W_rel = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.orthogonal_(self.W_rel)
        self.W_gate = nn.Linear(2*d_model+2, 1)

    def forward(self, q, k, E0, E1):
        E0_n = (E0 - E0.mean(-1, keepdim=True)) / (E0.std(-1, keepdim=True)+1e-6)
        E1_n = (E1 - E1.mean(-1, keepdim=True)) / (E1.std(-1, keepdim=True)+1e-6)
        feat = torch.cat([q, k, E0.unsqueeze(-1), E1.unsqueeze(-1)], dim=-1)
        alpha_hat = torch.sigmoid(self.W_gate(feat).squeeze(-1).clamp(-10,10))
        S = (1-alpha_hat)*E0_n + alpha_hat*E1_n
        return S, alpha_hat
```

## Dev Diary — Lessons I Learned The Hard Way

- v2.5 removed √d_model → gradients exploded
- v2.4 qᵀWk symmetric → failed directed Paris→France
- Divide by zero → NaNs for an epoch → fixed with ε·sign(D)
- Letting W_rel move β away to cheat → fixed with sg[β]
- Hinge loss exploded → softplus worked
- Training only on oracle α → 8% exposure bias drop → fixed with α_mix annealing 1→0 over 10k
- α=1.0 → queen→queen bee collapse in March → keep 10% similarity anchor

## Limitations & Ethics

- D_α small (12k), ConceptNet only 3 relations, inherits biases from DeBERTa-v3-large MNLI + all-MiniLM. IsA may carry gender stereotypes.
- Random negatives are trivial, not hard negatives near β.
- Human calibration only 2k, κ=0.78.
- No scaling test for d=4096 yet.

## Citation

```bibtex
@misc{alhayali2026dualexpert,
  title={Learning When to Look Like vs. Work Like: A Dual-Expert Attention with a Graded Alpha Knob},
  author={Al-Hayali, Abdulrahman},
  year={2026},
  version={v2.8.3 Human-Clear Final},
  address={Mosul, Iraq},
  howpublished={Independent Researcher — azl_y80}
}
```

---
*I wrote this in Mosul, v2.8.3 — trying to keep it clear for researchers but still my voice. α is graded, adaptive, evaluated from E0 strength via L, not fixed. 0.2 and 0.9 were just my shorthand for weak and strong ranges. — Abdulrahman*

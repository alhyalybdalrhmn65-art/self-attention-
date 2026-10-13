"""
data.py
=======
Data for Dual-Expert Attention.

--data_source wikipedia : streamed Wikipedia (first N docs) -> MLM from scratch.
                          Falls back to a built-in mock corpus when offline.
--data_source dalpha    : D_alpha relation pairs (bAbI 16-18, CLUTRR, ConceptNet
                          IsA/PartOf/Capitals), 1:1 random negatives, split 10k/1k/1k.
--data_source hf        : same D_alpha pairs, tokenised with bert-base-uncased, for
                          fine-tuning DualExpertBert (d_model=768).

ORACLE ALPHA IS NEVER COMPUTED INSIDE ATTENTION (O(N^2) DeBERTa calls is infeasible).
It is pre-computed offline, once per (q_text, k_text) pair, with frozen
DeBERTa-v3-large-MNLI + all-MiniLM under torch.no_grad():
    alpha = sigmoid(Phi_rel - gamma_nli * Sim),  gamma_nli = 0.3   (Eq 4)
optionally calibrated with isotonic regression on human labels, then clamped to
[0, 0.95]. The pair alpha is broadcast to the token block
[premise tokens (queries) x hypothesis tokens (keys)]; every other token pair is
NaN = "no oracle label" (gate gets no regression signal there).

mock_alpha() gives the paper's graded density for quick demos:
45% U[0,0.05], 25% U[0.15,0.30], 30% U[0.75,0.95].
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

GAMMA_NLI = 0.3
ALPHA_MAX = 0.95
DEFAULT_NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_SIM = "sentence-transformers/all-MiniLM-L6-v2"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


# =============================================================================
# Tokenizers (HF bert-base-uncased, or an offline word-level fallback)
# =============================================================================
class _PairEncoder:
    cls_token, sep_token = "[CLS]", "[SEP]"

    def encode_pair(self, text_a: str, text_b: Optional[str] = None, max_len: int = 64) -> Dict:
        ta = self.tokenize(text_a)
        tb = self.tokenize(text_b) if text_b is not None else []
        budget = max_len - (3 if text_b is not None else 2)
        while len(ta) + len(tb) > budget and (ta or tb):
            (ta if len(ta) >= len(tb) else tb).pop()
        tokens = [self.cls_token] + ta + [self.sep_token]
        span_a, span_b = (1, 1 + len(ta)), (0, 0)
        if text_b is not None:
            start = len(tokens)
            tokens += tb + [self.sep_token]
            span_b = (start, start + len(tb))
        ids = self.convert_tokens_to_ids(tokens)
        n_a = span_a[1] + 1
        return {"tokens": tokens, "input_ids": ids, "token_type_ids": [0] * n_a + [1] * (len(ids) - n_a),
                "attention_mask": [1] * len(ids), "span_a": span_a, "span_b": span_b}


class SimpleTokenizer(_PairEncoder):
    """Offline word-level tokenizer. Keeps arithmetic like '5+3' as one token."""
    kind = "simple"
    TOKEN_RE = re.compile(r"\d+(?:[+\-*/]\d+)+|\w+|[^\w\s]", re.UNICODE)

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = dict(vocab) if vocab else {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self._inv()

    def _inv(self):
        self.inv = {i: t for t, i in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        return self.TOKEN_RE.findall(text.lower())

    def build_vocab(self, texts: Sequence[str], max_size: int = 30000, min_freq: int = 1):
        c = Counter(tok for t in texts for tok in self.tokenize(t))
        for tok, f in c.most_common():
            if f < min_freq or len(self.vocab) >= max_size:
                break
            self.vocab.setdefault(tok, len(self.vocab))
        self._inv()
        return self

    def convert_tokens_to_ids(self, toks):
        unk = self.vocab["[UNK]"]
        return [self.vocab.get(t, unk) for t in toks]

    def convert_ids_to_tokens(self, ids):
        return [self.inv.get(int(i), "[UNK]") for i in ids]

    pad_token_id = property(lambda s: s.vocab["[PAD]"])
    unk_token_id = property(lambda s: s.vocab["[UNK]"])
    cls_token_id = property(lambda s: s.vocab["[CLS]"])
    sep_token_id = property(lambda s: s.vocab["[SEP]"])
    mask_token_id = property(lambda s: s.vocab["[MASK]"])
    vocab_size = property(lambda s: len(s.vocab))

    def state(self) -> Dict:
        return {"kind": "simple", "vocab": self.vocab}


class HFTokenizer(_PairEncoder):
    kind = "hf"

    def __init__(self, name: str = "bert-base-uncased"):
        from transformers import AutoTokenizer
        self.name, self.tok = name, AutoTokenizer.from_pretrained(name)
        self.cls_token, self.sep_token = self.tok.cls_token, self.tok.sep_token

    def tokenize(self, text):
        return self.tok.tokenize(text)

    def convert_tokens_to_ids(self, toks):
        return self.tok.convert_tokens_to_ids(list(toks))

    def convert_ids_to_tokens(self, ids):
        return self.tok.convert_ids_to_tokens([int(i) for i in ids])

    pad_token_id = property(lambda s: s.tok.pad_token_id)
    unk_token_id = property(lambda s: s.tok.unk_token_id)
    cls_token_id = property(lambda s: s.tok.cls_token_id)
    sep_token_id = property(lambda s: s.tok.sep_token_id)
    mask_token_id = property(lambda s: s.tok.mask_token_id)
    vocab_size = property(lambda s: len(s.tok))

    def state(self) -> Dict:
        return {"kind": "hf", "name": self.name}


def get_tokenizer(name: Optional[str] = "bert-base-uncased", texts: Optional[Sequence[str]] = None,
                  allow_fallback: bool = True):
    if name and name != "simple":
        try:
            return HFTokenizer(name)
        except Exception as e:                                   # offline / not installed
            if not allow_fallback:
                raise
            print(f"[data] HF tokenizer '{name}' unavailable ({type(e).__name__}); using SimpleTokenizer")
    return SimpleTokenizer().build_vocab(list(texts or []) + mock_corpus())


def tokenizer_from_state(state: Dict):
    return HFTokenizer(state["name"]) if state["kind"] == "hf" else SimpleTokenizer(state["vocab"])


# =============================================================================
# Graded oracle alpha: mock + real (DeBERTa-MNLI + MiniLM)
# =============================================================================
def mock_alpha(shape, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Paper density: 45% U[0,0.05] (pure similarity), 25% U[0.15,0.30] (weak,
    ~80/20), 30% U[0.75,0.95] (strong, ~10/90). Never above 0.95."""
    r = torch.rand(shape, generator=generator)
    x = torch.rand(shape, generator=generator)
    return torch.where(r < 0.45, 0.05 * x, torch.where(r < 0.70, 0.15 + 0.15 * x, 0.75 + 0.20 * x))


_BANDS = {"none": (0.0, 0.05), "weak": (0.15, 0.30), "strong": (0.75, 0.95)}


def mock_alpha_for_strength(strength: str, rng: random.Random) -> float:
    lo, hi = _BANDS.get(strength, _BANDS["none"])
    return rng.uniform(lo, hi)


@torch.no_grad()
def precompute_oracle_alpha(records: List[Dict], nli_model: str = DEFAULT_NLI, sim_model: str = DEFAULT_SIM,
                            gamma_nli: float = GAMMA_NLI, batch_size: int = 32, device: Optional[str] = None,
                            isotonic=None, cache_path: Optional[str] = None) -> List[Dict]:
    """Eq (4), offline and frozen: alpha = sigmoid(Phi_rel - gamma_nli * Sim).
    Phi_rel = entailment logit (premise=text(q), hypothesis=text(k)); Sim = cosine
    of mean-pooled all-MiniLM embeddings. Frozen models never see W_rel (no cheating)."""
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nli_tok = AutoTokenizer.from_pretrained(nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_model).to(device).eval()
    ent = next(int(i) for i, l in nli.config.id2label.items() if "entail" in str(l).lower())
    sim_tok = AutoTokenizer.from_pretrained(sim_model)
    sim = AutoModel.from_pretrained(sim_model).to(device).eval()
    for p in list(nli.parameters()) + list(sim.parameters()):
        p.requires_grad_(False)

    def embed(texts):
        e = sim_tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        h = sim(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp_min(1e-9), dim=-1)

    raw = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        prem, hyp = [r["q_text"] for r in chunk], [r["k_text"] for r in chunk]
        enc = nli_tok(prem, hyp, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        phi = nli(**enc).logits[:, ent].float()
        s = (embed(prem) * embed(hyp)).sum(-1)
        raw.extend(torch.sigmoid(phi - gamma_nli * s).cpu().tolist())
    cal = isotonic.predict(raw) if isotonic is not None else raw
    for r, a_raw, a in zip(records, raw, cal):
        r["alpha_raw"] = float(a_raw)
        r["alpha"] = float(min(max(a, 0.0), ALPHA_MAX))
    if cache_path:
        save_records(records, cache_path)
    return records


def fit_isotonic(raw_alpha: Sequence[float], human_alpha: Sequence[float]):
    """Calibrate raw oracle alpha against human labels (paper: 2k labels, kappa=0.78)."""
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0.0, y_max=ALPHA_MAX, out_of_bounds="clip").fit(raw_alpha, human_alpha)


def save_records(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_records(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# =============================================================================
# Built-in knowledge (mock corpus + synthetic D_alpha fallback)
# =============================================================================
CAPITALS = [("paris", "france"), ("berlin", "germany"), ("rome", "italy"), ("madrid", "spain"),
            ("baghdad", "iraq"), ("cairo", "egypt"), ("tokyo", "japan"), ("london", "england"),
            ("ankara", "turkey"), ("tehran", "iran"), ("damascus", "syria"), ("amman", "jordan"),
            ("riyadh", "saudi arabia"), ("beijing", "china"), ("moscow", "russia"), ("ottawa", "canada"),
            ("canberra", "australia"), ("lisbon", "portugal"), ("athens", "greece"), ("vienna", "austria"),
            ("oslo", "norway"), ("stockholm", "sweden"), ("helsinki", "finland"), ("dublin", "ireland"),
            ("warsaw", "poland"), ("prague", "czechia"), ("budapest", "hungary"), ("brussels", "belgium"),
            ("amsterdam", "netherlands"), ("bern", "switzerland"), ("seoul", "south korea"),
            ("hanoi", "vietnam"), ("bangkok", "thailand"), ("jakarta", "indonesia"), ("nairobi", "kenya"),
            ("lima", "peru"), ("santiago", "chile"), ("bogota", "colombia"), ("havana", "cuba"),
            ("beirut", "lebanon"), ("kabul", "afghanistan"), ("doha", "qatar"), ("muscat", "oman"),
            ("rabat", "morocco"), ("tunis", "tunisia"), ("algiers", "algeria"), ("khartoum", "sudan")]
PART_OF = [("car", "wheel"), ("car", "engine"), ("tree", "leaf"), ("hand", "finger"), ("house", "roof"),
           ("bicycle", "pedal"), ("book", "page"), ("face", "nose"), ("computer", "keyboard"), ("bird", "wing"),
           ("flower", "petal"), ("piano", "key"), ("shirt", "sleeve"), ("door", "handle"), ("city", "street"),
           ("guitar", "string"), ("plane", "wing"), ("knife", "blade"), ("fish", "fin"), ("year", "month"),
           ("week", "day"), ("tree", "branch"), ("foot", "toe"), ("ship", "deck"), ("chair", "leg")]
IS_A = [("dog", "animal"), ("rose", "flower"), ("oak", "tree"), ("apple", "fruit"), ("salmon", "fish"),
        ("eagle", "bird"), ("hammer", "tool"), ("violin", "instrument"), ("iron", "metal"),
        ("carrot", "vegetable"), ("cat", "mammal"), ("ant", "insect"), ("mars", "planet"), ("tennis", "sport"),
        ("chess", "game"), ("arabic", "language"), ("table", "furniture"), ("shirt", "clothing"),
        ("doctor", "profession"), ("tigris", "river")]
ANALOGY = [("king", "queen"), ("man", "woman"), ("prince", "princess"), ("father", "mother"),
           ("brother", "sister"), ("uncle", "aunt"), ("husband", "wife"), ("boy", "girl")]
WEAK = [("car", "road"), ("rain", "umbrella"), ("doctor", "hospital"), ("teacher", "school"),
        ("coffee", "morning"), ("book", "library"), ("ship", "sea"), ("pen", "paper"), ("snow", "winter"),
        ("train", "station"), ("bread", "bakery"), ("star", "night")]
SIMILAR = [("dog", "dogs"), ("car", "cars"), ("run", "running"), ("city", "cities"), ("walk", "walked"),
           ("big", "bigger"), ("happy", "happier"), ("tree", "trees"), ("play", "played"), ("fast", "faster")]
NAMES = ["ali", "sara", "omar", "lina", "yusuf", "maryam", "hassan", "noor", "karim", "huda", "adam",
         "layla", "zaid", "rana", "sami", "dana", "tariq", "mona", "faris", "reem"]
COLORS = ["white", "gray", "yellow", "green", "black", "red"]
ANIMALS = ["swan", "lion", "frog", "rhino", "mouse", "wolf"]
SIZED = ["box", "chest", "suitcase", "chocolate", "container", "jar", "bag", "crate"]
KIN = {("father", "father"): "grandfather", ("father", "mother"): "grandmother",
       ("mother", "father"): "grandfather", ("mother", "mother"): "grandmother",
       ("father", "brother"): "uncle", ("mother", "brother"): "uncle", ("father", "sister"): "aunt",
       ("mother", "sister"): "aunt", ("brother", "son"): "nephew", ("sister", "daughter"): "niece",
       ("son", "son"): "grandson", ("son", "daughter"): "granddaughter"}


def mock_corpus() -> List[str]:
    s = [f"{c} is the capital of {k}." for c, k in CAPITALS]
    s += [f"a {w} has a {p}." for w, p in PART_OF]
    s += [f"a {x} is a kind of {y}." for x, y in IS_A]
    s += [f"the {a} and the {b} are a pair." for a, b in ANALOGY]
    s += [f"you often find a {a} near a {b}." for a, b in WEAK]
    s += [f"the word {a} is like {b}." for a, b in SIMILAR]
    s += [f"{a}+{b}={a + b}" for a in range(10) for b in range(10)]
    s += ["paris is capital of france", "car→wheel", "king→queen", "5+3=8", "dog→dogs"]
    return s


# =============================================================================
# D_alpha: synthetic fallback generators
# =============================================================================
def _rec(q, k, label, source, relation, strength):
    return {"q_text": q, "k_text": k, "label": int(label), "source": source,
            "relation": relation, "strength": strength}


def _gen_babi_like(n: int, rng: random.Random) -> List[Dict]:
    out = []
    while len(out) < n:
        kind = rng.choice(["qa16", "qa18", "arith"])
        if kind == "qa16":                                   # basic induction
            a, b, c = rng.sample(NAMES, 3)
            an, other = rng.sample(ANIMALS, 2)
            col, ocol = rng.sample(COLORS, 2)
            story = (f"{a} is a {an}. {a} is {col}. {c} is a {other}. {c} is {ocol}. "
                     f"{b} is a {an}. what color is {b}?")
            out.append(_rec(story, col, 1, "babi", "qa16", "strong"))
        elif kind == "qa18":                                 # size reasoning
            x, y, z = rng.sample(SIZED, 3)
            if rng.random() < 0.5:
                story, ans = f"the {x} is bigger than the {y}. the {y} is bigger than the {z}. is the {x} bigger than the {z}?", "yes"
            else:
                story, ans = f"the {x} is bigger than the {y}. the {y} is bigger than the {z}. is the {z} bigger than the {x}?", "no"
            out.append(_rec(story, ans, 1, "babi", "qa18", "strong"))
        else:                                                # 5+3 -> 8
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            out.append(_rec(f"{a}+{b}", str(a + b), 1, "babi", "arith", "strong"))
    return out


def _gen_clutrr_like(n: int, rng: random.Random) -> List[Dict]:
    out, keys = [], list(KIN)
    while len(out) < n:
        r1, r2 = rng.choice(keys)
        a, b, c = rng.sample(NAMES, 3)
        story = f"{b} is the {r1} of {a}. {c} is the {r2} of {b}."
        out.append(_rec(story, f"{c} is the {KIN[(r1, r2)]} of {a}.", 1, "clutrr", f"{r1}-{r2}", "strong"))
    return out


def _gen_conceptnet_like(rng: random.Random) -> List[Dict]:
    out = [_rec(q, k, 1, "conceptnet", "capital", "strong") for q, k in CAPITALS]
    out += [_rec(q, k, 1, "conceptnet", "partof", "strong") for q, k in PART_OF]
    out += [_rec(q, k, 1, "conceptnet", "isa", "strong") for q, k in IS_A]
    out += [_rec(q, k, 1, "conceptnet", "analogy", "strong") for q, k in ANALOGY]
    out += [_rec(q, k, 1, "conceptnet", "weak", "weak") for q, k in WEAK]
    out += [_rec(q, k, 1, "conceptnet", "similar", "none") for q, k in SIMILAR]
    rng.shuffle(out)
    return out


# =============================================================================
# D_alpha: real datasets (best effort; dataset hub names change over time)
# =============================================================================
def _load_real_babi(n: int) -> List[Dict]:
    from datasets import load_dataset
    out = []
    for task in ("qa16", "qa17", "qa18"):
        ds = load_dataset("facebook/babi_qa", f"en-{task}", split="train", trust_remote_code=True)
        for ex in ds:
            st, ctx = ex["story"], []
            for typ, text, ans in zip(st["type"], st["text"], st["answer"]):
                if typ == 0:
                    ctx.append(text)
                else:
                    out.append(_rec(" ".join(ctx + [text]), ans, 1, "babi", task, "strong"))
            if len(out) >= n * (("qa16", "qa17", "qa18").index(task) + 1) // 3:
                break
    return out[:n]


def _load_real_clutrr(n: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("CLUTRR/v1", "gen_train234_test2to10", split="train", trust_remote_code=True)
    out = []
    for ex in ds:
        q = ex["query"].strip("()").replace("'", "").split(",")
        a, b = q[0].strip(), q[-1].strip()
        out.append(_rec(ex["clean_story"], f"{b} is the {ex['target_text']} of {a}.", 1, "clutrr",
                        ex["target_text"], "strong"))
        if len(out) >= n:
            break
    return out


def _load_real_conceptnet(n: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("conceptnet5", "conceptnet5", split="train", streaming=True, trust_remote_code=True)
    term = lambda c: c.split("/")[3].replace("_", " ")
    out = [_rec(q, k, 1, "conceptnet", "capital", "strong") for q, k in CAPITALS]
    for ex in ds:
        if ex.get("lang") != "en" or ex["rel"] not in ("/r/IsA", "/r/PartOf"):
            continue
        if not (ex["arg1"].startswith("/c/en/") and ex["arg2"].startswith("/c/en/")):
            continue
        x, y = term(ex["arg1"]), term(ex["arg2"])
        q, k = (y, x) if ex["rel"] == "/r/PartOf" else (x, y)     # whole -> part, like car -> wheel
        out.append(_rec(q, k, 1, "conceptnet", ex["rel"][3:].lower(), "strong"))
        if len(out) >= n:
            break
    return out


def _add_negatives(pos: List[Dict], rng: random.Random) -> List[Dict]:
    """1:1 random negatives: same query text, key text from another positive."""
    neg, ks = [], [r["k_text"] for r in pos]
    for r in pos:
        k = rng.choice(ks)
        while k == r["k_text"] and len(set(ks)) > 1:
            k = rng.choice(ks)
        neg.append(_rec(r["q_text"], k, 0, r["source"], "random_negative", "none"))
    return pos + neg


def build_dalpha(source: str = "synthetic", n_per_source: int = 2000, seed: int = 42) -> List[Dict]:
    """n_per_source POSITIVES per source (paper: 4k pairs per source incl. negatives -> 2k pos)."""
    rng = random.Random(seed)
    loaders = {"babi": (_load_real_babi, lambda: _gen_babi_like(n_per_source, rng)),
               "clutrr": (_load_real_clutrr, lambda: _gen_clutrr_like(n_per_source, rng)),
               "conceptnet": (_load_real_conceptnet, lambda: _gen_conceptnet_like(rng))}
    recs = []
    for name, (real, synth) in loaders.items():
        pos = None
        if source == "real":
            try:
                pos = real(n_per_source)
                print(f"[data] loaded real {name}: {len(pos)} positives")
            except Exception as e:
                print(f"[data] real {name} unavailable ({type(e).__name__}: {e}); using synthetic")
        if pos is None:
            pos = synth()
        recs += _add_negatives(pos, rng)
    rng.shuffle(recs)
    return recs


def split_records(recs: List[Dict], ratios=(10, 1, 1)) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    n, tot = len(recs), sum(ratios)
    a = n * ratios[0] // tot
    b = a + n * ratios[1] // tot
    return recs[:a], recs[a:b], recs[b:]


def attach_oracle(records: List[Dict], oracle: str = "mock", seed: int = 42, cache_path: Optional[str] = None):
    """oracle = 'mock' (graded band per relation strength) | 'deberta' | path/to/alpha.jsonl"""
    if oracle == "mock":
        rng = random.Random(seed + 1)
        for r in records:
            r["alpha"] = mock_alpha_for_strength(r["strength"], rng)
        return records
    if oracle == "deberta":
        if cache_path and os.path.exists(cache_path):
            return load_records(cache_path)
        return precompute_oracle_alpha(records, cache_path=cache_path)
    if os.path.exists(oracle):
        return load_records(oracle)
    raise ValueError(f"unknown oracle '{oracle}'")


# =============================================================================
# Wikipedia
# =============================================================================
def clean_wiki(text: str) -> List[str]:
    text = re.sub(r"\[\d+\]", " ", text)
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    sents = []
    for p in paras:
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if len(s.split()) >= 4]
    return sents


def load_wikipedia_texts(n_docs: int = 10_000) -> List[str]:
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        except Exception:
            ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= n_docs:
                break
            texts += clean_wiki(ex["text"])
        print(f"[data] Wikipedia: {n_docs} docs -> {len(texts)} sentences")
        return texts
    except Exception as e:
        print(f"[data] Wikipedia unavailable ({type(e).__name__}); using mock corpus")
        return mock_corpus() * 20


# =============================================================================
# Datasets + collate
# =============================================================================
class PairDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=64):
        self.records, self.tok, self.max_len = records, tokenizer, max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        enc = self.tok.encode_pair(r["q_text"], r["k_text"], self.max_len)
        enc["label"], enc["alpha"] = int(r["label"]), float(r["alpha"])
        return enc


def collate_pairs(batch, pad_id: int):
    B, T = len(batch), max(len(b["input_ids"]) for b in batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    tt = torch.zeros((B, T), dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    alpha = torch.full((B, T, T), float("nan"))
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        tt[i, :L] = torch.tensor(b["token_type_ids"])
        am[i, :L] = 1
        (a0, a1), (b0, b1) = b["span_a"], b["span_b"]
        if a1 > a0 and b1 > b0:
            alpha[i, a0:a1, b0:b1] = b["alpha"]       # premise tokens (q) -> hypothesis tokens (k)
    return {"input_ids": ids, "token_type_ids": tt, "attention_mask": am,
            "labels": torch.tensor([b["label"] for b in batch]), "alpha_oracle": alpha}


class WikiMLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.tok, self.chunks, buf, L = tokenizer, [], [], max_len - 2
        for t in texts:
            buf += tokenizer.tokenize(t)
            while len(buf) >= L:
                self.chunks.append(buf[:L])
                buf = buf[L:]
        if buf:
            self.chunks.append(buf)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        toks = [self.tok.cls_token] + self.chunks[i] + [self.tok.sep_token]
        return {"input_ids": self.tok.convert_tokens_to_ids(toks)}


def collate_mlm(batch, tok, mlm_prob=0.15, alpha_mode="mock"):
    B, T = len(batch), max(len(b["input_ids"]) for b in batch)
    ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        am[i, :len(b["input_ids"])] = 1
    special = (ids == tok.cls_token_id) | (ids == tok.sep_token_id) | (am == 0)
    pick = (torch.rand(B, T) < mlm_prob) & ~special
    for i in range(B):                                       # at least one target per row
        if not pick[i].any():
            cand = (~special[i]).nonzero().flatten()
            if len(cand):
                pick[i, cand[torch.randint(len(cand), (1,))]] = True
    labels = torch.where(pick, ids, torch.full_like(ids, -100))
    r = torch.rand(B, T)
    inp = ids.clone()
    inp[pick & (r < 0.8)] = tok.mask_token_id
    rnd = pick & (r >= 0.8) & (r < 0.9)
    inp[rnd] = torch.randint(5, tok.vocab_size, (int(rnd.sum()),))
    out = {"input_ids": inp, "attention_mask": am, "token_type_ids": torch.zeros_like(ids), "labels": labels}
    if alpha_mode == "mock":
        a = mock_alpha((B, T, T))
        valid = am.bool()[:, :, None] & am.bool()[:, None, :]
        out["alpha_oracle"] = a.masked_fill(~valid, float("nan"))
    return out


def build_dataloaders(data_source: str = "dalpha", batch_size: int = 128, max_len: int = 64,
                      oracle: str = "mock", seed: int = 42, tokenizer=None,
                      tokenizer_name: str = "bert-base-uncased", n_docs: int = 10_000,
                      dalpha_source: str = "synthetic", n_per_source: int = 2000,
                      wiki_alpha: str = "mock", oracle_cache: Optional[str] = None) -> Dict:
    g = torch.Generator().manual_seed(seed)
    if data_source == "wikipedia":
        texts = load_wikipedia_texts(n_docs)
        random.Random(seed).shuffle(texts)
        tok = tokenizer or get_tokenizer(tokenizer_name, texts)
        n = len(texts)
        parts = texts[: int(0.9 * n)], texts[int(0.9 * n): int(0.95 * n)], texts[int(0.95 * n):]
        dsets = [WikiMLMDataset(p or texts[:10], tok, max_len) for p in parts]
        col = partial(collate_mlm, tok=tok, alpha_mode=wiki_alpha)
        task = "mlm"
        if wiki_alpha == "mock":
            print("[data] wikipedia + mock alpha: graded RANDOM oracle, only exercises the plumbing")
    elif data_source in ("dalpha", "hf"):
        recs = attach_oracle(build_dalpha(dalpha_source, n_per_source, seed), oracle, seed, oracle_cache)
        tr, va, te = split_records(recs)
        if tokenizer is not None:
            tok = tokenizer
        elif data_source == "hf":
            tok = get_tokenizer(tokenizer_name, allow_fallback=False)
        else:
            tok = get_tokenizer(tokenizer_name, [r["q_text"] + " " + r["k_text"] for r in recs])
        dsets = [PairDataset(p, tok, max_len) for p in (tr, va, te)]
        col = partial(collate_pairs, pad_id=tok.pad_token_id)
        task = "cls"
    else:
        raise ValueError(data_source)
    mk = lambda d, sh: DataLoader(d, batch_size=batch_size, shuffle=sh, collate_fn=col, generator=g)
    return {"train": mk(dsets[0], True), "val": mk(dsets[1], False), "test": mk(dsets[2], False),
            "tokenizer": tok, "task": task, "n_classes": 2}


if __name__ == "__main__":
    a = mock_alpha((200_000,))
    print("mock alpha mass  [0,.05]=%.3f  [.15,.30]=%.3f  [.75,.95]=%.3f  max=%.3f" % (
        (a <= 0.05).float().mean(), ((a >= 0.15) & (a <= 0.30)).float().mean(),
        ((a >= 0.75) & (a <= 0.95)).float().mean(), a.max()))
    d = build_dataloaders("dalpha", batch_size=4, tokenizer_name="simple")
    b = next(iter(d["train"]))
    print({k: tuple(v.shape) for k, v in b.items()})
    print("labelled alpha cells in batch:", int((~torch.isnan(b["alpha_oracle"])).sum()))
    print("train/val/test sizes:", len(d["train"].dataset), len(d["val"].dataset), len(d["test"].dataset))

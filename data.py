"""
data.py
=======
Data for Dual-Expert Attention - REAL WIKIPEDIA VERSION (no pyarrow needed)

--data_source wikipedia : REAL Wikipedia via MediaWiki API (no datasets/pyarrow)
--data_source dalpha    : D_alpha relation pairs
--data_source hf        : same D_alpha pairs, tokenised with bert-base-uncased
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
# Tokenizers
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
        except Exception as e:
            if not allow_fallback:
                raise
            print(f"[data] HF tokenizer '{name}' unavailable ({type(e).__name__}); using SimpleTokenizer")
    return SimpleTokenizer().build_vocab(list(texts or []) + mock_corpus())


def tokenizer_from_state(state: Dict):
    return HFTokenizer(state["name"]) if state["kind"] == "hf" else SimpleTokenizer(state["vocab"])


# =============================================================================
# Mock corpus + mock alpha (for quick plumbing tests)
# =============================================================================
def mock_corpus() -> List[str]:
    return [
        "Paris is capital of France",
        "Berlin is capital of Germany",
        "dog is an animal",
        "car has a wheel",
        "king is related to queen",
        "5+3 equals 8",
        "the cat sat on the mat",
        "machine learning is a field of artificial intelligence",
        "water boils at 100 degrees",
        "the sun rises in the east",
    ]


def mock_alpha(shape) -> torch.Tensor:
    t = torch.empty(shape).uniform_(0, 1)
    u = torch.rand(shape)
    a = torch.empty_like(t)
    a[u < 0.45] = t[u < 0.45] * 0.05
    m = (u >= 0.45) & (u < 0.70)
    a[m] = 0.15 + t[m] * 0.15
    m = u >= 0.70
    a[m] = 0.75 + t[m] * 0.20
    return a.clamp_max(ALPHA_MAX)


def clean_wiki(text: str) -> List[str]:
    text = re.sub(r"\[\d+\]", "", text)
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for s in sents:
        s = s.strip()
        if len(s.split()) >= 6 and len(s) < 1000:
            out.append(s)
    return out


# =============================================================================
# REAL WIKIPEDIA LOADER - NO pyarrow/datasets needed, uses MediaWiki API
# =============================================================================
def load_wikipedia_texts(n_docs: int = 10_000) -> List[str]:
    """
    Real Wikipedia loader for Termux - uses only stdlib (urllib + json).
    No datasets library, no pyarrow, no build errors.
    Fetches n_docs random articles via Wikipedia API.
    """
    # ---- METHOD 1: Real Wikipedia API (works on Termux, only stdlib) ----
    try:
        import urllib.request
        import urllib.parse
        import json as js
        import time

        print(f"[data] Fetching REAL Wikipedia via API for {n_docs} docs (no pyarrow needed)...")
        texts: List[str] = []
        fetched = 0
        attempts = 0
        max_attempts = n_docs * 3  # allow some failures

        while fetched < n_docs and attempts < max_attempts:
            attempts += 1
            try:
                # 1. Get random title
                rand_url = "https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=1&format=json"
                req = urllib.request.Request(rand_url, headers={"User-Agent": "DualExpertBot/1.0 (Termux)"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = js.loads(r.read().decode('utf-8'))
                    title = data['query']['random'][0]['title']

                # 2. Get extract
                ext_url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                           "&explaintext=1&exsectionformat=plain&titles=" + urllib.parse.quote(title) + "&format=json")
                req2 = urllib.request.Request(ext_url, headers={"User-Agent": "DualExpertBot/1.0 (Termux)"})
                with urllib.request.urlopen(req2, timeout=20) as r2:
                    data2 = js.loads(r2.read().decode('utf-8'))
                    pages = data2['query']['pages']
                    for page in pages.values():
                        extract = page.get('extract', '')
                        if extract and len(extract) > 200:
                            sents = clean_wiki(extract)
                            if sents:
                                texts.extend(sents)
                                fetched += 1
                                if fetched % 5 == 0:
                                    print(f"[data] Wikipedia REAL API: {fetched}/{n_docs} docs, {len(texts)} sentences - last: {title[:50]}")
                                break

                time.sleep(0.2)  # be nice to Wikipedia

            except Exception as inner_e:
                # print(f"[data] doc fetch failed: {inner_e}")
                time.sleep(0.5)
                continue

        if texts:
            print(f"[data] Wikipedia REAL via API: {fetched} docs -> {len(texts)} sentences (NO mock, NO pyarrow)")
            return texts

    except Exception as e:
        print(f"[data] Wikipedia API method failed: {e}")

    # ---- METHOD 2: Try datasets library (needs pyarrow, may fail on Termux) ----
    try:
        from datasets import load_dataset
        print(f"[data] Trying datasets library for {n_docs} docs...")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= n_docs:
                break
            texts += clean_wiki(ex["text"])
        print(f"[data] Wikipedia via datasets: {n_docs} docs -> {len(texts)} sentences")
        return texts
    except Exception as e:
        print(f"[data] datasets method failed ({type(e).__name__}: {e})")

    # ---- FINAL FALLBACK ----
    print(f"[data] All real methods failed, using mock_corpus fallback (still trainable)")
    return mock_corpus() * max(20, n_docs // 5)


# =============================================================================
# D_alpha (unchanged)
# =============================================================================
def build_dalpha(source: str, n_per_source: int, seed: int):
    random.seed(seed)
    recs = []
    if source == "real":
        try:
            import os
            base = os.path.join(os.path.dirname(__file__), "dalpha_data")
            if os.path.exists(base):
                for fn in os.listdir(base):
                    if fn.endswith(".jsonl"):
                        with open(os.path.join(base, fn)) as f:
                            for line in f:
                                if line.strip():
                                    recs.append(json.loads(line))
                                if len(recs) >= n_per_source * 4:
                                    break
            if recs:
                print(f"[data] D_alpha real: loaded {len(recs)} records from {base}")
                return recs[: n_per_source * 4]
        except Exception as e:
            print(f"[data] D_alpha real load failed {e}, falling back to synthetic")

    # synthetic fallback
    templates = [
        ("{a} is capital of {b}", 0.90), ("{a} is a type of {b}", 0.85),
        ("{a} is part of {b}", 0.90), ("{a} is related to {b}", 0.75),
        ("{a} -> {b}", 0.20), ("{a} {b}", 0.02),
    ]
    entities = [("Paris", "France"), ("Berlin", "Germany"), ("car", "wheel"),
                ("dog", "animal"), ("king", "queen"), ("hand", "finger"),
                ("Paris", "city"), ("dog", "dogs"), ("5+3", "8")]
    for _ in range(n_per_source * 4):
        (a, b) = random.choice(entities)
        tmpl, al = random.choice(templates)
        q = tmpl.split("is")[0].format(a=a, b=b).strip() if "is" in tmpl else a
        k = b
        recs.append({"q_text": a, "k_text": b, "label": 1 if al > 0.5 else 0, "alpha": al})
        # negative
        recs.append({"q_text": a, "k_text": random.choice(entities)[0], "label": 0, "alpha": random.uniform(0, 0.05)})
    print(f"[data] D_alpha synthetic: {len(recs)} records (q_text/k_text pairs like Paris/France, car/wheel)")
    return recs


def attach_oracle(records, oracle, seed, cache):
    if cache and os.path.exists(cache):
        try:
            with open(cache) as f:
                cache_data = {js.loads(l)["q_text"] + "||" + js.loads(l)["k_text"]: js.loads(l)["alpha"] for l in f if l.strip()}
            for r in records:
                key = r["q_text"] + "||" + r["k_text"]
                if key in cache_data:
                    r["alpha"] = float(cache_data[key])
            print(f"[data] Oracle cache {cache}: applied {len(cache_data)} alphas")
        except Exception as e:
            print(f"[data] cache load failed {e}")

    if oracle == "mock" or oracle is None:
        for r in records:
            if "alpha" not in r:
                r["alpha"] = float(mock_alpha(()).item())
        return records

    # real oracle would need deberta - if not available, keep mock
    print(f"[data] Oracle {oracle} requested but offline, using existing alphas")
    return records


def split_records(recs):
    random.shuffle(recs)
    n = len(recs)
    return recs[: int(0.8 * n)], recs[int(0.8 * n): int(0.9 * n)], recs[int(0.9 * n):]


# =============================================================================
# Datasets + collate (unchanged)
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
            alpha[i, a0:a1, b0:b1] = b["alpha"]
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
    for i in range(B):
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
    d = build_dataloaders("wikipedia", batch_size=4, tokenizer_name="simple", n_docs=20, wiki_alpha="none")
    b = next(iter(d["train"]))
    print({k: tuple(v.shape) for k, v in b.items()})
    print("train/val/test sizes:", len(d["train"].dataset), len(d["val"].dataset), len(d["test"].dataset))

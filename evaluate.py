# -*- coding: utf-8 -*-
"""evaluate.py

A lightweight evaluation module for the CCD‑RAG system.
Call ``run_eval(embedder, test_file, k)`` from ``main.py`` after the index
has been built/loaded.

Metrics implemented
-------------------
1. **Recall@k**   – percentage of questions whose gold answer appears in the
   top‑k retrieved items (binary relevance).
2. **MRR**        – mean reciprocal rank of the first relevant document.
3. **nDCG**       – normalised discounted cumulative gain (binary rel.).
4. **Exact‑Match** – generator exact string match (SQuAD normalisation).
5. **F1**         – generator token‑level F1 (SQuAD style).

Assumptions
-----------
* ``test_file`` is an Excel/CSV with at least two columns named
  ``question`` and ``answer``.
* A *relevant retrieval hit* is counted when the gold answer text is a
  **substring** of the retrieved chunk's ``metadata['content']``
  (customise ``is_hit()`` if you use another criterion).
* ``QASystem`` lives in ``qa_system.py`` and wraps the Llama generator.

Edit the utility functions if your schema differs.
"""
from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from tqdm.auto import tqdm

# --- text normalisation helpers ------------------------------------------------

def _normalize(text: str) -> str:
    """Lower‑case, strip punctuation/articles/extra‑spaces (SQuAD style)."""
    text = str(text).lower() 
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(pred: str, gold) -> bool:
    """pred 與 gold 完全一致？ gold 可為 str 或 iterable[str]"""
    if isinstance(gold, (list, tuple, set)):
        return any(_normalize(pred) == _normalize(g) for g in gold)
    return _normalize(pred) == _normalize(gold)


def f1_score(pred: str, gold) -> float:
    """計算 token‑level F1；gold 可為多個答案，取最大值"""
    def _f1(a, b):
        a_tokens, b_tokens = _normalize(a).split(), _normalize(b).split()
        common = set(a_tokens) & set(b_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(a_tokens)
        rec  = len(common) / len(b_tokens)
        return 2 * prec * rec / (prec + rec)

    if isinstance(gold, (list, tuple, set)):
        return max(_f1(pred, g) for g in gold)
    return _f1(pred, gold)

# --- retrieval hit check -------------------------------------------------------

def is_hit(gold_answer: str, retrieved_md: Dict) -> bool:
    """Return *True* if this retrieved chunk counts as relevant.

    Current rule: gold answer substring appears in metadata["content"].
    Adjust to your own relevance signal if needed (e.g. compare IDs).
    """
    text = str(retrieved_md.get("content", "")).lower()
    return gold_answer.lower() in text

# --- main evaluation routine ---------------------------------------------------

def run_eval(embedder, test_file: str | Path, k: int = 10):
    """Run the full evaluation pipeline and print metrics.

    Parameters
    ----------
    embedder : ClipEmbeddingProcessor
        The already‑built embedding processor (has ``search`` method).
    test_file : str or Path
        Path to ``.xlsx`` or ``.csv`` containing *question* and *answer* columns.
    k : int
        Top‑k to retrieve.
    """
    from qa_system import QASystem  # local import to avoid circular deps.

    # --- load data ------------------------------------------------------------
    test_path = Path(test_file)
    if test_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(test_path)
    else:
        df = pd.read_csv(test_path)

    assert {"question", "answers"}.issubset(df.columns), (
        "test_file must contain 'question' and 'answers' columns")

    questions: Sequence[str] = df["question"].tolist()
    answers: Sequence[str] = df["answers"].tolist()

    # --- set up QA system (generator) ----------------------------------------
    qa = QASystem(embedder)  # uses default Llama in your repo

    # --- accumulators ---------------------------------------------------------
    hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    em_sum = 0
    f1_sum = 0.0

    for q, gold in tqdm(zip(questions, answers), total=len(questions), desc="eval"):
        # 1) retrieval ---------------------------------------------------------
        res = embedder.search(q, k=k)
        # metadatas: List[Dict] = res.get("metadatas", [])
        md_list: List[Dict] = (res.get("metadatas") or [[]])[0]

        hit_rank = None
        for rank, md in enumerate(md_list, start=1):
            if is_hit(gold, md):
                hit_rank = rank
                break

        if hit_rank is not None:
            hits += 1
            rr_sum += 1.0 / hit_rank
            ndcg_sum += 1.0 / math.log2(hit_rank + 1)

        # 2) generation --------------------------------------------------------
        pred = qa.display_response(q)
        em_sum += int(exact_match(pred, gold))
        f1_sum += f1_score(pred, gold)

    n = len(questions)
    metrics = {
        f"Recall@{k}": round(hits / n, 4),
        "MRR": round(rr_sum / n, 4),
        "nDCG": round(ndcg_sum / n, 4),
        "ExactMatch": round(em_sum / n, 4),
        "F1": round(f1_sum / n, 4),
    }

    print("\n===== Evaluation Result =====")
    print(json.dumps(metrics, indent=2))
    return metrics

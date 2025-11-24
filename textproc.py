from __future__ import annotations
import re
from collections import Counter
from typing import List, Tuple, Dict
from utils import PUNCTUATIONS

# Компилируем regex так, чтобы многосимвольные шли раньше (например, '...')
_PUNCT_TOKENS = sorted(PUNCTUATIONS, key=len, reverse=True)
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_TOKENS)))

def count_punctuations(text: str) -> dict[str, int]:
    return dict(Counter(_PUNCT_RE.findall(text)))

def extract_words(text: str) -> list[str]:
    cleaned = _PUNCT_RE.sub(" ", text)
    return [w.casefold() for w in cleaned.split()]

def list_accuracy(pred: list[str], target: list[str]) -> float:
    n_correct = sum(p == t for p, t in zip(pred, target))
    total = max(len(pred), len(target))
    return n_correct / total if total > 0 else 0.0

def _space_wrap_punct(text: str) -> str:
    tmp = text
    for p in _PUNCT_TOKENS:
        tmp = tmp.replace(p, f" {p} ")
    return tmp

def extract_punct_events(text: str) -> list[tuple[int, str]]:
    """
    Возвращает список (pos, punct), где pos — индекс границы после слова.
    Пример: слова = [w0 w1 w2]; запятая после w1 -> pos=2.
    """
    spaced = _space_wrap_punct(text)
    tokens = spaced.split()

    events: list[tuple[int, str]] = []
    word_count = 0
    for tok in tokens:
        if tok in PUNCTUATIONS:
            events.append((word_count, tok))
        else:
            word_count += 1
    return events

def punct_positions_by_type(text: str) -> dict[str, list[int]]:
    by_type: dict[str, list[int]] = {p: [] for p in PUNCTUATIONS}
    for pos, sym in extract_punct_events(text):
        by_type[sym].append(pos)
    for p in by_type:
        by_type[p].sort()
    return by_type

def punct_precision_metrics(ref_text: str, pred_text: str) -> tuple[float, dict[str, float]]:
    """
    Precision-метрика по позициям пунктуации (мультимножества).
    Возвращает: overall_precision, per_type_precision.
    """
    ref_pos = punct_positions_by_type(ref_text)
    pred_pos = punct_positions_by_type(pred_text)

    per_type: dict[str, float] = {}
    total_correct = 0
    total_predicted = 0
    total_ref = 0

    for p in PUNCTUATIONS:
        ref_list = ref_pos[p]
        pred_list = pred_pos[p]

        ref_counter = Counter(ref_list)
        pred_counter = Counter(pred_list)

        correct = sum(min(pred_counter[pos], ref_counter[pos]) for pos in pred_counter)
        predicted = sum(pred_counter.values())
        ref_count = sum(ref_counter.values())

        total_correct += correct
        total_predicted += predicted
        total_ref += ref_count

        if predicted > 0:
            per_type[p] = correct / predicted
        else:
            per_type[p] = 1.0 if ref_count == 0 else 0.0

    if total_predicted > 0:
        overall = total_correct / total_predicted
    else:
        overall = 1.0 if total_ref == 0 else 0.0

    return overall, per_type

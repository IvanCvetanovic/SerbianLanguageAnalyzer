from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

SentimentLabel = Literal["negative", "neutral", "positive"]
Mode = Literal["document", "sentences"]


@dataclass(frozen=True)
class SentimentResult:
    label: SentimentLabel
    confidence: float
    scores: Dict[SentimentLabel, float]


@dataclass(frozen=True)
class SentimentAnalysis:
    overall: SentimentResult
    sentences: List[SentimentResult]
    mode: Mode


class SerbianSentimentAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        use_fast_tokenizer: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=use_fast_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        id2label_raw = getattr(self.model.config, "id2label", None) or {}
        self.id2label: Dict[int, str] = {int(i): str(l).lower() for i, l in id2label_raw.items()}
        if not self.id2label:
            self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self._lock = threading.Lock()

    def analyze(
        self,
        text: str,
        mode: Mode = "sentences",
        aggregation: Literal["mean", "length_weighted_mean"] = "mean",
        max_length: int = 256,
    ) -> SentimentAnalysis:
        text = (text or "").strip()
        if not text:
            neutral = SentimentResult(
                label="neutral",
                confidence=1.0,
                scores={"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            )
            return SentimentAnalysis(overall=neutral, sentences=[], mode=mode)

        with self._lock:
            if mode == "document":
                overall = self._score_text(text, max_length=max_length)
                return SentimentAnalysis(overall=overall, sentences=[], mode="document")

            sents = self.split_sentences(text)
            if not sents:
                overall = self._score_text(text, max_length=max_length)
                return SentimentAnalysis(overall=overall, sentences=[], mode="document")

            sent_results = [self._score_text(s, max_length=max_length) for s in sents]
            overall = self._aggregate(sent_results, sents, method=aggregation)
            return SentimentAnalysis(overall=overall, sentences=sent_results, mode="sentences")

    def analyze_batch(
        self,
        texts: Sequence[str],
        mode: Mode = "document",
        aggregation: Literal["mean", "length_weighted_mean"] = "mean",
        max_length: int = 256,
    ) -> List[SentimentAnalysis]:
        return [self.analyze(t, mode=mode, aggregation=aggregation, max_length=max_length) for t in texts]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []
        t = re.sub(r"\s+", " ", t)
        parts = re.split(r"(?<=[.!?])\s+", t)
        return [p.strip() for p in parts if p and p.strip()]

    @torch.no_grad()
    def _score_text(self, text: str, max_length: int = 256) -> SentimentResult:
        text = (text or "").strip()
        if not text:
            return SentimentResult(
                label="neutral",
                confidence=1.0,
                scores={"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            )

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        scores: Dict[SentimentLabel, float] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for i, p in enumerate(probs):
            raw = self.id2label.get(i, "").lower()
            if "neg" in raw:
                scores["negative"] = float(p)
            elif "neu" in raw:
                scores["neutral"] = float(p)
            elif "pos" in raw:
                scores["positive"] = float(p)

        label: SentimentLabel = max(scores, key=scores.get)  # type: ignore[arg-type]
        return SentimentResult(label=label, confidence=float(scores[label]), scores=scores)

    @staticmethod
    def _aggregate(
        results: List[SentimentResult],
        sentences: List[str],
        method: Literal["mean", "length_weighted_mean"] = "mean",
    ) -> SentimentResult:
        if not results:
            return SentimentResult(
                label="neutral",
                confidence=1.0,
                scores={"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            )

        if method == "length_weighted_mean":
            weights = [max(1, len(s)) for s in sentences]
        else:
            weights = [1 for _ in results]

        wsum = float(sum(weights)) if sum(weights) > 0 else 1.0

        agg = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for r, w in zip(results, weights):
            agg["negative"] += r.scores["negative"] * w
            agg["neutral"] += r.scores["neutral"] * w
            agg["positive"] += r.scores["positive"] * w

        agg = {k: float(v / wsum) for k, v in agg.items()}
        label: SentimentLabel = max(agg, key=agg.get)  # type: ignore[arg-type]
        return SentimentResult(label=label, confidence=float(agg[label]), scores=agg)

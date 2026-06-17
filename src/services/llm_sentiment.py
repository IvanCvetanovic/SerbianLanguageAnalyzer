from __future__ import annotations

import re
import requests

from src.core.model_config import get_config, get_openai_client, OLLAMA_URL
from src.services.sentiment_analyzer import (
    SentimentAnalysis,
    SentimentLabel,
    SentimentResult,
    SerbianSentimentAnalyzer,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Exact prompt requested for the LLM sentiment engine.
_SYSTEM = "Ti si asistent za analizu sentimenta srpskog teksta. Odgovori SAMO jednom rečju: 'positive', 'negative' ili 'neutral'. Bez ikakvog objašnjenja."

_LABELS: tuple[SentimentLabel, ...] = ("positive", "negative", "neutral")


def _user_msg(text: str) -> str:
    return f"Analiziraj sentiment ovog teksta:\n\n{text}"


def _parse_label(raw: str) -> SentimentLabel:
    """Return the first of positive/negative/neutral that appears; default neutral."""
    cleaned = _THINK_RE.sub("", raw or "").lower()
    best: SentimentLabel = "neutral"
    best_idx = len(cleaned) + 1
    for label in _LABELS:
        idx = cleaned.find(label)
        if 0 <= idx < best_idx:
            best_idx = idx
            best = label
    return best


def _classify_remote(text: str, cfg: dict) -> SentimentLabel:
    r_cfg = cfg["remote"]
    client = get_openai_client(r_cfg["base_url"], r_cfg["api_key"])
    print(f"[VLLM-CALL] task=sentiment mode={cfg['mode']} url={r_cfg['base_url']}", flush=True)
    r = client.chat.completions.create(
        model=r_cfg["model"],
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_msg(text)},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return _parse_label(r.choices[0].message.content or "")


def _classify_local(text: str) -> SentimentLabel:
    payload = {
        "model": get_config()["local"]["model"],
        "prompt": f"{_SYSTEM}\n\n{_user_msg(text)}",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Ollama is not running or not reachable at {OLLAMA_URL}. "
            "Start Ollama before using the LLM sentiment engine."
        )
    r.raise_for_status()
    return _parse_label(r.json().get("response", ""))


def _classify(text: str) -> SentimentLabel:
    cfg = get_config()
    if cfg["mode"] == "remote":
        return _classify_remote(text, cfg)
    return _classify_local(text)


def _result_from_label(label: SentimentLabel) -> SentimentResult:
    scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    scores[label] = 1.0
    return SentimentResult(label=label, confidence=1.0, scores=scores)


def analyze_llm(text: str) -> SentimentAnalysis:
    """LLM sentiment mirroring the classifier's shape: a per-sentence result for each
    sentence (one LLM call each) plus an overall that is the mean aggregate of those
    per-sentence labels — the same aggregation the XLM classifier uses for its overall.
    """
    text = (text or "").strip()
    if not text:
        neutral = SentimentResult(
            label="neutral",
            confidence=1.0,
            scores={"negative": 0.0, "neutral": 1.0, "positive": 0.0},
        )
        return SentimentAnalysis(overall=neutral, sentences=[], mode="sentences")

    sents = SerbianSentimentAnalyzer.split_sentences(text)
    if not sents:
        overall = _result_from_label(_classify(text))
        return SentimentAnalysis(overall=overall, sentences=[], mode="document")

    sent_results = [_result_from_label(_classify(s)) for s in sents]
    overall = SerbianSentimentAnalyzer._aggregate(sent_results, sents, method="mean")
    return SentimentAnalysis(overall=overall, sentences=sent_results, mode="sentences")

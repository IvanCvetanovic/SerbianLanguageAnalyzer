import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "sadjava/multilingual-hate-speech-xlm-roberta"

LABELS: List[str] = [
    "race",
    "sexual_orientation",
    "gender",
    "physical_appearance",
    "religion",
    "class",
    "disability",
    "appropriate",
]

PROTECTED_LABELS = {
    "race",
    "sexual_orientation",
    "gender",
    "physical_appearance",
    "religion",
    "disability",
}

LABEL_THRESHOLDS: Dict[str, float] = {lbl: 0.35 for lbl in PROTECTED_LABELS}
A_APPROPRIATE: float = 0.70
MIN_PROTECTED_SIGNAL_FOR_LOW_APPROPRIATE: float = 0.20

VIOLENCE_PATTERNS = [
    r"\bpobiti\b",
    r"\bubiti\b",
    r"\bubij(te|mo|m|š|u)?\b",
    r"\bzaklati\b",
    r"\bistre(bi|biti|bljenje|bićemo)\b",
    r"\btreba\s+.*\b(pobiti|ubiti|zaklati)\b",
]

DISABILITY_SLUR_PATTERNS = [
    r"\bbogalj(e|i|a)?\b",
]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_overrides(text: str) -> List[str]:
    t = _normalize(text)
    reasons: List[str] = []

    for pat in VIOLENCE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            reasons.append("keyword: violence/threat detected")
            break

    for pat in DISABILITY_SLUR_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            reasons.append("keyword: disability slur detected")
            break

    return reasons


def _classify_policy(probs: List[float], labels: List[str]) -> Dict:
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    best_label = labels[best_idx]
    best_score = probs[best_idx]

    appropriate_idx = labels.index("appropriate")
    appropriate_score = probs[appropriate_idx]

    score_map = {lbl: score for lbl, score in zip(labels, probs)}
    protected_scores: List[Tuple[str, float]] = [(lbl, score_map[lbl]) for lbl in PROTECTED_LABELS]
    max_protected_label, max_protected_score = max(protected_scores, key=lambda x: x[1])

    crossed: List[Tuple[str, float, float]] = []
    for lbl, s in protected_scores:
        thr = LABEL_THRESHOLDS[lbl]
        if s >= thr:
            crossed.append((lbl, s, thr))

    reasons: List[str] = []
    if crossed:
        for lbl, s, thr in crossed:
            reasons.append(f"model: {lbl} >= {thr:.2f} ({s:.3f})")
        flagged_model = True
    else:
        flagged_model = (
            appropriate_score <= A_APPROPRIATE
            and max_protected_score >= MIN_PROTECTED_SIGNAL_FOR_LOW_APPROPRIATE
        )
        if flagged_model:
            reasons.append(
                f"model: appropriate <= {A_APPROPRIATE:.2f} AND "
                f"max(protected) >= {MIN_PROTECTED_SIGNAL_FOR_LOW_APPROPRIATE:.2f} "
                f"({max_protected_label}={max_protected_score:.3f})"
            )

    return {
        "best_label": best_label,
        "best_score": float(best_score),
        "appropriate_score": float(appropriate_score),
        "max_protected_label": max_protected_label,
        "max_protected_score": float(max_protected_score),
        "flagged_model": bool(flagged_model),
        "reasons_model": reasons,
    }


class HateSpeechDetector:
    _instance: Optional["HateSpeechDetector"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self._initialized = True

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        policy = _classify_policy(probs, LABELS)
        kw_reasons = _keyword_overrides(text)
        flagged_final = bool(policy["flagged_model"] or kw_reasons)

        scores = {lbl: float(p) for lbl, p in zip(LABELS, probs)}

        return {
            "text": text,
            "flagged": flagged_final,
            "reasons": policy["reasons_model"] + kw_reasons,
            "label": policy["best_label"],
            "score": policy["best_score"],
            "scores": scores,
            "appropriate_score": policy["appropriate_score"],
            "max_protected_label": policy["max_protected_label"],
            "max_protected_score": policy["max_protected_score"],
            "device": self.device,
            "model": self.model_name,
        }


_detector = HateSpeechDetector()


def analyze_hate_speech(text: str) -> Dict:
    return _detector.predict(text)


__all__ = [
    "HateSpeechDetector",
    "analyze_hate_speech",
    "MODEL_NAME",
    "LABELS",
]

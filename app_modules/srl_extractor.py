import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from app_modules.pipeline import get_nlp


ROLE_SET = [
    "Agent/Doer",
    "Patient/Theme",
    "Recipient/Beneficiary",
    "Time",
    "Location",
    "Source",
    "Destination",
    "Cause",
    "Purpose",
    "Manner",
    "Instrument",
    "Other",
]

TIME_WORDS = {"danas", "juče", "juce", "sutra", "sinoć", "sinoc", "uvek", "često", "cesto", "ponekad", "odmah"}
CONTROL_LEMMAS = {"želeti", "hteti", "moći", "morati", "trebati", "smeti", "pokušati", "planirati"}
CLITIC_RECIPIENTS = {"mi", "ti", "mu", "joj", "nam", "vam", "im", "se"}


@dataclass
class Token:
    i: int
    text: str
    lemma: str
    upos: str
    feats: str


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _dedup(vals: List[str]) -> List[str]:
    out = []
    for v in vals:
        v2 = _norm(v)
        if v2 and v2 not in out:
            out.append(v2)
    return out


def _starts_with_any(s: str, preps: List[str]) -> bool:
    s2 = _norm(s).lower()
    return any(s2.startswith(p + " ") or s2 == p for p in preps)


def _move(vals_from: List[str], vals_to: List[str], pred) -> None:
    for v in list(vals_from):
        if pred(v):
            vals_from.remove(v)
            vals_to.append(v)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def normalize_roles(obj: Dict[str, Any], sentence: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"sentence": sentence, "frames": []}

    frames = obj.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    out_frames = []
    for f in frames:
        if not isinstance(f, dict):
            continue

        pred = str(f.get("predicate", "")).strip()
        pred_lemma = str(f.get("predicate_lemma", "")).strip()
        pred_idx = f.get("predicate_index", None)

        # Coerce "3" -> 3
        if isinstance(pred_idx, str):
            s = pred_idx.strip()
            if s.isdigit():
                pred_idx = int(s)

        # If model accidentally returns 0-based index, make it 1-based minimum
        if isinstance(pred_idx, int) and pred_idx == 0:
            pred_idx = 1

        neg = bool(f.get("negated", False))

        roles = f.get("roles", {})
        if not isinstance(roles, dict):
            roles = {}

        clean_roles: Dict[str, List[str]] = {}
        for k, v in roles.items():
            k2 = str(k).strip()
            if k2 not in ROLE_SET:
                continue

            if isinstance(v, list):
                vals = [str(x).strip() for x in v if str(x).strip()]
            else:
                vals = [str(v).strip()] if str(v).strip() else []

            # dedup preserve order
            vals2 = []
            for x in vals:
                if x not in vals2:
                    vals2.append(x)

            if vals2:
                clean_roles[k2] = vals2

        if pred and isinstance(pred_idx, int) and pred_idx >= 1:
            out_frames.append(
                {
                    "predicate": pred,
                    "predicate_lemma": pred_lemma or pred,
                    "predicate_index": pred_idx,
                    "negated": neg,
                    "roles": clean_roles,
                }
            )

    return {"sentence": sentence, "frames": out_frames}


def postprocess_srl(sentence: str, frames: List[Dict[str, Any]], tokens: Optional[List[Token]]) -> List[Dict[str, Any]]:
    for f in frames:
        roles = f.get("roles", {}) or {}
        pred_text = _norm(f.get("predicate", ""))
        pred_lemma = _norm(f.get("predicate_lemma", "")).lower()

        for k in list(roles.keys()):
            if not isinstance(roles[k], list):
                roles[k] = [roles[k]]
            roles[k] = _dedup([str(x) for x in roles[k]])

        agent = roles.get("Agent/Doer", [])
        patient = roles.get("Patient/Theme", [])
        time = roles.get("Time", [])
        loc = roles.get("Location", [])
        src = roles.get("Source", [])
        dst = roles.get("Destination", [])
        cause = roles.get("Cause", [])

        _move(agent, time, lambda v: _norm(v).lower() in TIME_WORDS)

        def _drop_pred_text(v: str) -> bool:
            v2 = _norm(v)
            return (not v2) or (v2.lower() == pred_text.lower())

        roles["Agent/Doer"] = [v for v in agent if not _drop_pred_text(v)]
        roles["Patient/Theme"] = [v for v in patient if not _drop_pred_text(v)]

        _move(loc, cause, lambda v: _starts_with_any(v, ["zbog", "usled", "radi"]))
        _move(loc, src, lambda v: _starts_with_any(v, ["iz"]))
        _move(loc, dst, lambda v: _starts_with_any(v, ["u", "na"]))

        if pred_lemma in CONTROL_LEMMAS:
            roles.pop("Patient/Theme", None)

        if pred_lemma in {"otići", "doći"}:
            roles.pop("Patient/Theme", None)

        clitic_obj = {"ga", "je", "ju", "ih", "se", "me", "te", "nas", "vas"}
        pts = roles.get("Patient/Theme", [])
        if isinstance(pts, list) and len(pts) >= 2:
            has_np = any(len(_norm(x).split()) >= 2 for x in pts)
            if has_np:
                roles["Patient/Theme"] = [x for x in pts if _norm(x).lower() not in clitic_obj]
                if not roles["Patient/Theme"]:
                    roles.pop("Patient/Theme", None)

        if "Agent/Doer" in roles:
            roles["Agent/Doer"] = _dedup(roles["Agent/Doer"])
        if "Patient/Theme" in roles:
            roles["Patient/Theme"] = _dedup(roles["Patient/Theme"])
        if "Time" in roles:
            roles["Time"] = _dedup(time)
        if "Location" in roles:
            roles["Location"] = _dedup(loc)
        if "Source" in roles:
            roles["Source"] = _dedup(src)
        if "Destination" in roles:
            roles["Destination"] = _dedup(dst)
        if "Cause" in roles:
            roles["Cause"] = _dedup(cause)

        roles = {k: v for k, v in roles.items() if v}
        f["roles"] = roles

    return frames


def enrich_with_classla(sentence: str, frames: List[Dict[str, Any]], tokens: Optional[List[Token]]) -> List[Dict[str, Any]]:
    if not tokens:
        return frames

    by_i = {t.i: t for t in tokens}
    low = sentence.lower()

    def add(rolemap: Dict[str, List[str]], role: str, val: str):
        v = _norm(val)
        if not v:
            return
        rolemap.setdefault(role, [])
        if v not in rolemap[role]:
            rolemap[role].append(v)

    def find_token(text_lower: str) -> Optional[int]:
        for t in tokens:
            if t.text.lower() == text_lower:
                return t.i
        return None

    juce_i = find_token("juče") or find_token("juce")
    if juce_i and ("juče" in low or "juce" in low):
        for f in frames:
            roles = f.get("roles", {}) or {}
            if "Time" not in roles:
                add(roles, "Time", by_i[juce_i].text)
            f["roles"] = roles

    for f in frames:
        roles = f.get("roles", {}) or {}
        lemma = _norm(f.get("predicate_lemma", "")).lower()

        if lemma == "dati" and "Recipient/Beneficiary" not in roles:
            if "Agent/Doer" in roles and any(_norm(x).lower() in CLITIC_RECIPIENTS for x in roles["Agent/Doer"]):
                cl = [x for x in roles["Agent/Doer"] if _norm(x).lower() in CLITIC_RECIPIENTS]
                roles["Agent/Doer"] = [x for x in roles["Agent/Doer"] if _norm(x).lower() not in CLITIC_RECIPIENTS]
                for x in cl:
                    add(roles, "Recipient/Beneficiary", x)
                if not roles.get("Agent/Doer"):
                    roles.pop("Agent/Doer", None)

        loc = roles.get("Location", [])
        if loc:
            fixed = []
            for v in loc:
                v2 = _norm(v)
                if v2 and not _starts_with_any(v2, ["u", "na", "iz", "od", "kod"]):
                    if f"u {v2}" in sentence:
                        fixed.append(f"u {v2}")
                    elif f"na {v2}" in sentence:
                        fixed.append(f"na {v2}")
                    elif f"iz {v2}" in sentence:
                        fixed.append(f"iz {v2}")
                    else:
                        fixed.append(v2)
                else:
                    fixed.append(v2)
            roles["Location"] = _dedup(fixed)

        ag = roles.get("Agent/Doer", [])
        if ag:
            fixed = []
            for v in ag:
                v2 = _norm(v)
                if v2 and not _starts_with_any(v2, ["od"]):
                    if f"od {v2}" in sentence:
                        fixed.append(f"od {v2}")
                    else:
                        fixed.append(v2)
                else:
                    fixed.append(v2)
            roles["Agent/Doer"] = _dedup(fixed)

        if "zbog" in low and "Cause" not in roles:
            m = re.search(r"\bzbog\b\s+[^.,;!?]+", sentence, flags=re.IGNORECASE)
            if m:
                add(roles, "Cause", m.group(0))

        if lemma in {"otići", "doći"}:
            if "u " in low:
                m = re.search(r"\bu\b\s+[^.,;!?]+", sentence, flags=re.IGNORECASE)
                if m:
                    add(roles, "Destination", m.group(0))
            if "iz " in low:
                m = re.search(r"\biz\b\s+[^.,;!?]+", sentence, flags=re.IGNORECASE)
                if m:
                    add(roles, "Source", m.group(0))

        f["roles"] = {k: v for k, v in roles.items() if v}

    return frames


class SerbianSRLExtractor:
    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        use_classla: Optional[bool] = None,
        pipeline=None,
        timeout_s: int = 60,
    ):
        # Ollama config (YOU WERE MISSING THESE)
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

        self.use_classla = (os.getenv("USE_CLASSLA_SRL", "1").strip() == "1") if use_classla is None else bool(use_classla)
        self.timeout_s = int(timeout_s)

        # Reuse shared classla pipeline
        self._nlp = pipeline if pipeline is not None else (get_nlp() if self.use_classla else None)

        self._session = requests.Session()

    def _sent_to_tokens(self, sent) -> List[Token]:
        out: List[Token] = []
        for w in sent.words:
            out.append(
                Token(
                    i=int(w.id),
                    text=w.text,
                    lemma=w.lemma or w.text,
                    upos=w.upos or "",
                    feats=w.feats or "",
                )
            )
        return out

    def _guess_predicates(self, tokens: List[Token]) -> List[Dict[str, Any]]:
        preds = []
        for t in tokens:
            # IMPORTANT: include AUX (je, sam, nisam, etc.)
            if t.upos in {"VERB", "AUX"}:
                preds.append({"i": t.i, "text": t.text, "lemma": t.lemma})
            else:
                if "VerbForm=Part" in (t.feats or "") or "Voice=Pass" in (t.feats or ""):
                    if t.upos in {"ADJ", "VERB"}:
                        preds.append({"i": t.i, "text": t.text, "lemma": t.lemma})

        uniq = []
        seen = set()
        for p in preds:
            key = (p["i"], p["text"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        r = self._session.post(self.ollama_url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    def _build_prompt(self, sentence: str, tokens: Optional[List[Token]]) -> str:
        tokens_txt = ""
        preds_txt = ""

        if tokens:
            toks = [{"i": t.i, "text": t.text, "lemma": t.lemma, "upos": t.upos, "feats": t.feats} for t in tokens]
            preds = self._guess_predicates(tokens)
            tokens_txt = json.dumps(toks, ensure_ascii=False)
            preds_txt = json.dumps(preds, ensure_ascii=False)

        schema = {
            "sentence": sentence,
            "frames": [
                {
                    "predicate": "string",
                    "predicate_lemma": "string",
                    "predicate_index": 1,
                    "negated": False,
                    "roles": {r: ["span1", "span2"] for r in ROLE_SET},
                }
            ],
        }

        parts = [
            "You are an SRL (Semantic Role Labeling) extractor for Serbian.\n",
            "Return STRICT JSON ONLY. No markdown. No extra text.\n",
            "Output must follow this schema shape:\n",
            json.dumps(schema, ensure_ascii=False),
            "\n\nRules:\n",
            f"- Allowed role keys only: {json.dumps(ROLE_SET, ensure_ascii=False)}\n",
            "- roles values must be arrays of strings (text spans copied from the sentence).\n",
            "- Each frame must correspond to one predicate (verb, auxiliary, or verbal participle) in the sentence.\n",
            "- predicate_index is 1-based token index.\n",
            "- negated=true if predicate is negated in the sentence.\n",
            "- Prefer Source for 'iz + GEN', Destination for 'u/na + ACC', otherwise Location.\n",
            "- For coordinated verbs sharing an object, assign Patient/Theme to both.\n",
            "- For control/raising (e.g., 'želi da kupi'), propagate Agent/Doer to the embedded predicate.\n\n",
            "SENTENCE:\n",
            sentence,
            "\n\n",
        ]

        if tokens_txt:
            parts.extend(["TOKENS (1-based):\n", tokens_txt, "\n\n"])
        if preds_txt:
            parts.extend(["PREDICATES (suggested):\n", preds_txt, "\n\n"])

        parts.append("Now produce the JSON.")
        return "".join(parts)

    def split_sentences(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        if self.use_classla and self._nlp is not None:
            try:
                doc = self._nlp(text)
                sents = []
                for s in doc.sentences:
                    s_txt = " ".join([w.text for w in s.words]).strip()
                    if s_txt:
                        sents.append(s_txt)
                if sents:
                    return sents
            except Exception:
                pass

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        tokens: Optional[List[Token]] = None
        if self.use_classla and self._nlp is not None:
            doc = self._nlp(sentence)
            if doc.sentences:
                tokens = self._sent_to_tokens(doc.sentences[0])

        prompt = self._build_prompt(sentence, tokens)
        raw = self._call_ollama(prompt)
        obj = extract_json_object(raw)

        # If JSON extraction fails, return raw for debugging
        if obj is None:
            return {"sentence": sentence, "frames": [], "raw": raw}

        out = normalize_roles(obj or {}, sentence)
        out["frames"] = postprocess_srl(out["sentence"], out["frames"], tokens)
        out["frames"] = enrich_with_classla(out["sentence"], out["frames"], tokens)
        return out

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        sentences = self.split_sentences(text)
        out: List[Dict[str, Any]] = []

        for s in sentences:
            try:
                out.append(self.analyze_sentence(s))
            except Exception as e:
                # Don’t hide the reason (this helps you fix issues fast)
                out.append({"sentence": s, "frames": [], "error": str(e)})

        return out

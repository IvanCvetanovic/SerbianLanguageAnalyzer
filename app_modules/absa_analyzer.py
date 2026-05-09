from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from app_modules.model_config import get_config, get_openai_client, OLLAMA_URL

from app_modules.pipeline import get_nlp

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_ABSA_SYSTEM = (
    'Analiziraj aspekte i njihov sentiment u sledećem tekstu. Odgovor daj kao JSON niz u formatu:\n'
    '[{"sentence": "...", "aspects": [{"aspect": "...", "sentiment": "positive|negative|neutral", '
    '"confidence": 0.0-1.0, "evidence": "..."}]}]\n'
    'Odgovori SAMO JSON-om, bez ikakvog dodatnog teksta.'
)


def parse_absa_smart(text):
    if not text:
        return None
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    results = []
    i = 0
    while i < len(text):
        if text[i] == '[':
            depth = 0
            in_string = False
            escape = False
            start = i
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            chunk = json.loads(text[start:j + 1])
                            if isinstance(chunk, list):
                                results.extend(chunk)
                        except Exception:
                            pass
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return results if results else None


@dataclass
class ABSASpan:
    aspect: str
    sentiment: str
    confidence: float
    evidence: str

class SerbianABSA:
    def __init__(self, timeout_s: int = 180) -> None:
        self.timeout_s = int(timeout_s)

        # Keep lock: classla pipeline isn't guaranteed thread-safe
        self._lock = threading.Lock()

        # Reuse the single shared classla pipeline (tokenize,pos,lemma,ner,depparse)
        self._nlp = get_nlp()

        self.aspect_deprels = {"root", "nsubj", "obj", "iobj", "obl", "nmod", "conj"}

        self._adj_lemma_blacklist = {
            "spor", "brz", "tih", "lagan", "odličan", "dobar", "loš", "slab", "dosadan",
            "prelep", "ukusan", "prosečan", "visok", "skup", "razočaravajuć", "stabilan", "pregledan"
        }

        self._positive_cues = {
            "odličan", "odlična", "odlično", "sjajan", "sjajna", "sjajno", "super",
            "dobar", "dobra", "dobro", "brz", "brza", "brzo", "stabilan", "stabilna", "stabilno",
            "pregledan", "pregledna", "pregledno", "ukusan", "ukusna", "ukusno",
            "prelep", "prelepa", "prelepo", "lagan", "lagana", "lagano", "tih", "tiha", "tiho"
        }
        self._negative_cues = {
            "loš", "loša", "loše", "slab", "slaba", "slabo", "dosadan", "dosadna", "dosadno",
            "spor", "spora", "sporo", "razočaravajuć", "razočaravajuća", "razočaravajuće",
            "prosečan", "prosečna", "prosečno", "visok", "visoka", "visoko",
            "skup", "skupa", "skupo"
        }

    def _ollama_generate(self, prompt: str) -> str:
        payload = {
            "model": get_config()["local"]["model"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "top_p": 0.9, "stop": ["```"]},
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=(5, self.timeout_s))
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama is not running or not reachable at {OLLAMA_URL}. "
                "Start Ollama before using ABSA features."
            )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\{[\s\S]*\}", (text or "").strip())
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _cue_score(self, evidence: str) -> int:
        ev = self._norm(evidence)
        score = 0
        for w in self._positive_cues:
            if w in ev:
                score += 1
        for w in self._negative_cues:
            if w in ev:
                score -= 1
        if re.search(r"\b(ne|nije|nisu|nema)\b", ev):
            score *= -1
        return score

    def _postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict) or "aspects" not in data:
            return data
        fixed = []
        for item in data.get("aspects", []):
            if not isinstance(item, dict):
                continue
            aspect = item.get("aspect", "")
            sentiment = item.get("sentiment", "neutral")
            confidence = item.get("confidence", 0.5)
            evidence = item.get("evidence", "")
            s = self._cue_score(evidence)
            if s > 0 and sentiment == "negative":
                sentiment = "positive"
                confidence = max(float(confidence), 0.7)
            elif s < 0 and sentiment == "positive":
                sentiment = "negative"
                confidence = max(float(confidence), 0.7)
            fixed.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "confidence": float(confidence),
                    "evidence": evidence,
                }
            )
        data["aspects"] = fixed
        return data

    def _should_filter_aspect(self, w) -> bool:
        lemma = (w.lemma or w.text or "").strip().lower()
        text = (w.text or "").strip().lower()
        if w.upos == "NOUN" and w.deprel == "nmod" and (lemma in self._adj_lemma_blacklist or text in self._adj_lemma_blacklist):
            return True
        return False

    def _extract_aspects_from_sent(self, sent) -> List[str]:
        words = sent.words
        by_id = {w.id: w for w in words}

        amod_heads = set()
        for w in words:
            if w.deprel == "amod" and w.head in by_id:
                amod_heads.add(w.head)

        candidates: List[Tuple[str, str]] = []
        for w in words:
            if w.upos not in {"NOUN", "PROPN"}:
                continue
            if self._should_filter_aspect(w):
                continue
            if w.deprel in self.aspect_deprels or w.id in amod_heads:
                lemma = (w.lemma or w.text or "").strip()
                if lemma:
                    candidates.append((lemma.lower(), lemma))

        seen = set()
        aspects = []
        for key, lemma in candidates:
            if key in seen:
                continue
            seen.add(key)
            aspects.append(lemma)
        return aspects

    def _llama_absa(self, sentence_text: str, aspects: Sequence[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
        cfg = get_config()
        if cfg["mode"] == "remote":
            return self._llama_absa_remote(sentence_text, aspects, cfg)
        return self._llama_absa_local(sentence_text, aspects)

    def _llama_absa_local(self, sentence_text: str, aspects: Sequence[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
        prompt = f"""
Vrati ISKLJUČIVO validan JSON. Ne koristi markdown, ne koristi ``` i ne dodaj objašnjenja.

Radi ABSA samo za date aspekte (nemoj dodavati nove i nemoj menjati nazive).
Za svaki aspekt odredi sentiment: "positive" | "negative" | "neutral".
Dodaj "confidence" (0..1).
"evidence" mora biti TAČAN podstring iz originalne rečenice (na srpskom), max 10 reči; ako nema, stavi "".

JSON format:
{{
  "sentence": "...",
  "aspects": [
    {{"aspect": "...", "sentiment": "positive|negative|neutral", "confidence": 0.0, "evidence": "..."}}
  ]
}}

Rečenica:
{sentence_text}

Aspekti (tačno ovako):
{json.dumps(list(aspects), ensure_ascii=False)}
""".strip()
        raw = self._ollama_generate(prompt)
        data = self._extract_json(raw)
        return raw, data

    def _llama_absa_remote(self, sentence_text: str, aspects: Sequence[str], cfg: dict) -> Tuple[str, Optional[Dict[str, Any]]]:
        r_cfg = cfg["remote"]
        client = get_openai_client(r_cfg["base_url"], r_cfg["api_key"])
        print(f"[VLLM-CALL] task=absa mode={cfg['mode']} url={r_cfg['base_url']}", flush=True)
        user_msg = (
            f"Rečenica:\n{sentence_text}\n\n"
            f"Aspekti (tačno ovako):\n{json.dumps(list(aspects), ensure_ascii=False)}"
        )
        r = client.chat.completions.create(
            model=r_cfg["model"],
            messages=[{"role": "system", "content": _ABSA_SYSTEM},
                      {"role": "user",   "content": user_msg}],
            temperature=0.0,
            max_tokens=800,
        )
        raw = _THINK_RE.sub('', r.choices[0].message.content or "").strip()
        parsed = parse_absa_smart(raw)
        data = parsed[0] if parsed else None
        return raw, data

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return []

        # Safer with threads
        with self._lock:
            doc = self._nlp(text)

        out: List[Dict[str, Any]] = []
        for sent in doc.sentences:
            sent_text = sent.text
            aspects = self._extract_aspects_from_sent(sent)
            if not aspects:
                out.append({"sentence": sent_text, "aspects": []})
                continue

            raw, data = self._llama_absa(sent_text, aspects)
            if data is None:
                out.append({"sentence": sent_text, "aspects": [], "raw": raw})
                continue

            data = self._postprocess(data)
            out.append(data)

        return out


_WORD_RE = re.compile(r"[A-Za-zČĆŠĐŽčćšđžА-Яа-яЉЊЏЋЂљњџćђ]+", re.UNICODE)


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _safe_get(d: Dict[str, str], k: str, default: str = "/") -> str:
    v = d.get(k)
    return v if v else default


def enrich_absa_with_translations(
    absa: List[Dict[str, Any]],
    translator,
    translation_map: Dict[str, str],
    WordController,
    safe_default: str = "/",
) -> List[Dict[str, Any]]:
    if not absa:
        return absa

    cache: Dict[str, str] = {}

    def lemma_en(sr_word_or_lemma: str) -> str:
        key = (sr_word_or_lemma or "").strip()
        if not key:
            return safe_default
        if key in cache:
            return cache[key]

        latin = WordController.transliterate_cyrillic_to_latin([key])[0]
        lemma = WordController.lemmatize_words([latin])[0]
        en = _safe_get(translation_map, lemma, None)

        if not en or en == safe_default:
            try:
                en = getattr(translator.translate(lemma, src="sr", dest="en"), "text", safe_default) or safe_default
            except Exception:
                en = safe_default

        cache[key] = en
        return en

    def evidence_en(evidence: str) -> str:
        words = _tokenize_words(evidence)
        if not words:
            return ""
        latin_words = WordController.transliterate_cyrillic_to_latin(words)
        lemmas = WordController.lemmatize_words(latin_words)
        ens = [_safe_get(translation_map, lem, safe_default) for lem in lemmas]

        missing = [i for i, t in enumerate(ens) if not t or t == safe_default]
        if missing:
            try:
                to_translate = [lemmas[i] for i in missing]
                translated = translator.translate(to_translate, src="sr", dest="en")
                if not isinstance(translated, list):
                    translated = [translated]
                for idx, tr in zip(missing, translated):
                    ens[idx] = getattr(tr, "text", safe_default) or safe_default
            except Exception:
                pass

        return " ".join(ens)

    enriched = []
    for sent in absa:
        aspects = sent.get("aspects") or []
        new_aspects = []
        for a in aspects:
            aspect = (a.get("aspect") or "").strip()
            evidence = (a.get("evidence") or "").strip()
            a2 = dict(a)
            a2["aspect_en"] = lemma_en(aspect)
            a2["evidence_en"] = evidence_en(evidence)
            new_aspects.append(a2)
        s2 = dict(sent)
        s2["aspects"] = new_aspects
        enriched.append(s2)

    return enriched

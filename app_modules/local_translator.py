from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import List, Sequence, Union
import requests
from app_modules.model_config import get_config, get_openai_client, OLLAMA_URL

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

@dataclass
class _Translation:
    text: str

class LocalSrToEnTranslator:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _is_single_word(text: str) -> bool:
        return (" " not in text) and ("\n" not in text) and ("\t" not in text)

    @staticmethod
    def _looks_like_sr_infinitive(word: str) -> bool:
        w = word.lower()
        return w.endswith("ti") or w.endswith("ći")

    def _ollama_generate(self, prompt: str, temperature: float = 0.05) -> str:
        payload = {
            "model": get_config()["local"]["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            },
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=(5, 120))
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama is not running or not reachable at {OLLAMA_URL}. "
                "Start Ollama before using translation features."
            )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _translate_one_local(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self._is_single_word(text) and self._looks_like_sr_infinitive(text):
            prompt = f"""
Prevedi sledeći srpski glagol u infinitivu na engleski.
Vrati samo osnovni prevod glagola, bez objašnjenja.

Reč:
{text}

Prevod:
""".strip()
            return self._ollama_generate(prompt)

        prompt = f"""
Prevedi sledeći tekst sa srpskog na engleski.
Vrati samo prevod, bez objašnjenja.

Tekst:
{text}

Prevod:
""".strip()

        return self._ollama_generate(prompt)

    def _translate_one_remote(self, text: str, cfg: dict) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self._is_single_word(text) and self._looks_like_sr_infinitive(text):
            system = "Prevedi sledeću reč sa srpskog na engleski."
        else:
            system = "Prevedi sledeći tekst sa srpskog na engleski."

        r_cfg = cfg["remote"]
        client = get_openai_client(r_cfg["base_url"], r_cfg["api_key"])
        print(f"[VLLM-CALL] task=translation mode={cfg['mode']} url={r_cfg['base_url']}", flush=True)
        r = client.chat.completions.create(
            model=r_cfg["model"],
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": text}],
            temperature=0.3,
            max_tokens=400,
        )
        raw = r.choices[0].message.content or ""
        return _THINK_RE.sub('', raw).strip()

    def _translate_one(self, text: str) -> str:
        cfg = get_config()
        if cfg["mode"] == "remote":
            return self._translate_one_remote(text, cfg)
        return self._translate_one_local(text)

    def translate(
        self,
        text_or_texts: Union[str, Sequence[str]],
        src: str = "sr",
        dest: str = "en",
    ) -> Union[_Translation, List[_Translation]]:
        if (src or "").lower() != "sr" or (dest or "").lower() != "en":
            raise ValueError("LocalSrToEnTranslator supports only src='sr' and dest='en'.")

        with self._lock:
            if isinstance(text_or_texts, (list, tuple)):
                return [_Translation(self._translate_one(t)) for t in text_or_texts]
            return _Translation(self._translate_one(text_or_texts))

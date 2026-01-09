from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Sequence, Union
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

@dataclass
class _Translation:
    text: str

class LocalSrToEnTranslator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
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
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            },
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _translate_one(self, text: str) -> str:
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

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Sequence, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-sh-en"

_SINGLE_WORD_OVERRIDES = {
    "biti": "to be",
    "бити": "to be",
}

@dataclass
class _Translation:
    text: str

class LocalSrToEnTranslator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.device == "cuda":
            self.model = self.model.to(self.device).half()
        else:
            self.model = self.model.to(self.device)

        self.model.eval()
        self._lock = threading.Lock()

    @staticmethod
    def _is_single_word(text: str) -> bool:
        return (" " not in text) and ("\n" not in text) and ("\t" not in text)

    @staticmethod
    def _looks_like_sr_infinitive(word: str) -> bool:
        w = word.lower()
        return w.endswith("ti") or w.endswith("ći")

    def _mt_translate(self, text: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                early_stopping=True,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _translate_one(self, text: str, max_new_tokens: int = 128) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self._is_single_word(text):
            key = text.lower()
            if key in _SINGLE_WORD_OVERRIDES:
                return _SINGLE_WORD_OVERRIDES[key]

        if self._is_single_word(text) and self._looks_like_sr_infinitive(text):
            contextual = f"Glagol u infinitivu: {text}"
            out = self._mt_translate(contextual, max_new_tokens=max_new_tokens).strip()

            if ":" in out:
                candidate = out.split(":")[-1].strip()
                if candidate:
                    return candidate
            return out

        return self._mt_translate(text, max_new_tokens=max_new_tokens)

    def translate(
        self,
        text_or_texts: Union[str, Sequence[str]],
        src: str = "sr",
        dest: str = "en",
        max_new_tokens: int = 128,
    ) -> Union[_Translation, List[_Translation]]:
        if (src or "").lower() != "sr" or (dest or "").lower() != "en":
            raise ValueError("LocalSrToEnTranslator supports only src='sr' and dest='en'.")

        with self._lock:
            if isinstance(text_or_texts, (list, tuple)):
                return [_Translation(self._translate_one(t, max_new_tokens=max_new_tokens)) for t in text_or_texts]
            return _Translation(self._translate_one(text_or_texts, max_new_tokens=max_new_tokens))
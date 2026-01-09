import re
import numpy as np
import networkx as nx
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.1:8b"

_LAT_TO_CYR = {
    "dž": "џ", "Dž": "Џ", "DŽ": "Џ",
    "lj": "љ", "Lj": "Љ", "LJ": "Љ",
    "nj": "њ", "Nj": "Њ", "NJ": "Њ",
    "a": "а", "A": "А", "b": "б", "B": "Б", "c": "ц", "C": "Ц",
    "č": "ч", "Č": "Ч", "ć": "ћ", "Ć": "Ћ", "d": "д", "D": "Д",
    "đ": "ђ", "Đ": "Ђ", "e": "е", "E": "Е", "f": "ф", "F": "Ф",
    "g": "г", "G": "Г", "h": "х", "H": "Х", "i": "и", "I": "И",
    "j": "ј", "J": "Ј", "k": "к", "K": "К", "l": "л", "L": "Л",
    "m": "м", "M": "М", "n": "н", "N": "Н", "o": "о", "O": "О",
    "p": "п", "P": "П", "r": "р", "R": "Р", "s": "с", "S": "С",
    "š": "ш", "Š": "Ш", "t": "т", "T": "Т", "u": "у", "U": "У",
    "v": "в", "V": "В", "z": "з", "Z": "З", "ž": "ж", "Ž": "Ж",
}
_CYR_TO_LAT = {v: k for k, v in _LAT_TO_CYR.items()}
_lat_pattern = re.compile("|".join(sorted(_LAT_TO_CYR.keys(), key=len, reverse=True)))
_cyr_pattern = re.compile("|".join(sorted(_CYR_TO_LAT.keys(), key=len, reverse=True)))

def transliterate_to_cyrillic(text: str) -> str:
    return _lat_pattern.sub(lambda m: _LAT_TO_CYR[m.group(0)], text)

def transliterate_to_latin(text: str) -> str:
    return _cyr_pattern.sub(lambda m: _CYR_TO_LAT[m.group(0)], text)

def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    abbr = ["dr.", "mr.", "mrs.", "g.", "prof.", "itd.", "npr.", "str.", "br.", "ul."]
    PH = "§DOT§"
    tmp = text
    for a in abbr:
        tmp = tmp.replace(a, a.replace(".", PH))
    pattern = r'(?<=[\.!?])\s+(?=(?:[A-ZŠĐČĆŽ]|[А-ЯЉЊЏЋЂ]))'
    parts = re.split(pattern, tmp.strip())
    sents = [p.replace(PH, ".").strip() for p in parts if p.strip()]
    return sents

def build_similarity_matrix(sentences: list[str]) -> np.ndarray:
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(sentences)
    sim = (tfidf * tfidf.T).toarray()
    np.fill_diagonal(sim, 0)
    return sim

def extractive_summary(text: str, num_sentences: int = 2) -> list[str]:
    sents = split_sentences(text)
    if len(sents) <= num_sentences:
        return sents
    sim_mat = build_similarity_matrix(sents)
    graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
    selected = sorted(ranked[:num_sentences], key=lambda x: sents.index(x[1]))
    return [s for _, s in selected]

def _ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def abstractive_summary(text: str, translator, max_len: int = 60, min_len: int = 30, num_beams: int = 4) -> tuple[str, str]:
    is_latin = bool(re.search(r"[A-Za-z]", text))
    text_cyr = transliterate_to_cyrillic(text) if is_latin else text
    prompt = (
        "Sumiraj sledeći tekst na srpskom jeziku.\n"
        "Vrati samo sažetak, bez objašnjenja.\n"
        f"Sažetak neka bude između {min_len} i {max_len} reči.\n\n"
        f"Tekst:\n{text_cyr}\n\n"
        "Sažetak:"
    )
    summary = _ollama_generate(prompt, temperature=0.2)
    summary_latin = transliterate_to_latin(summary) if is_latin else summary
    translation = translator.translate(summary_latin, dest="en").text
    return summary_latin, translation

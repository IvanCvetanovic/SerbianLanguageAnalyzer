import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else -1
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum",
    device=DEVICE,
)

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
    pattern = r'(?<=[\.\?\!])\s+'
    sents = re.split(pattern, text.strip())
    return [s for s in sents if s]

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

def abstractive_summary(text: str, translator, max_len: int = 60, min_len: int = 30, num_beams: int = 4) -> tuple[str, str]:
    is_latin = bool(re.search(r"[A-Za-z]", text))
    text_cyr = transliterate_to_cyrillic(text) if is_latin else text
    prefixed = "Sumiraj ovaj tekst i samo to pošalji nazad: " + text_cyr
    out = summarizer(
        prefixed,
        max_length=max_len,
        min_length=min_len,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    summary = out[0]["summary_text"]
    summary_latin = transliterate_to_latin(summary) if is_latin else summary
    translation = translator.translate(summary_latin, dest='en').text
    return summary_latin, translation

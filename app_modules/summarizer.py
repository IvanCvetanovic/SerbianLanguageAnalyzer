import re
import numpy as np
import networkx as nx
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from app_modules.model_config import get_config, get_openai_client, OLLAMA_URL
from app_modules.transliteration import lat_to_cyr, cyr_to_lat

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


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
    return [p.replace(PH, ".").strip() for p in parts if p.strip()]


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
        "model": get_config()["local"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _vllm_generate(system: str, user: str, cfg: dict,
                   temperature: float = 0.3, max_tokens: int = 300) -> str:
    r_cfg = cfg["remote"]
    client = get_openai_client(r_cfg["base_url"], r_cfg["api_key"])
    print(f"[VLLM-CALL] task=summarization mode={cfg['mode']} url={r_cfg['base_url']}", flush=True)
    r = client.chat.completions.create(
        model=r_cfg["model"],
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _THINK_RE.sub('', r.choices[0].message.content or "").strip()


def abstractive_summary(text: str, translator, max_len: int = 60, min_len: int = 30) -> tuple[str, str]:
    is_latin = bool(re.search(r"[A-Za-z]", text))
    text_cyr = lat_to_cyr(text) if is_latin else text
    cfg = get_config()
    if cfg["mode"] == "remote":
        summary = _vllm_generate(
            system="Sažmi sledeći tekst u nekoliko rečenica.",
            user=text_cyr,
            cfg=cfg,
            temperature=0.3,
            max_tokens=300,
        )
    else:
        prompt = (
            "Sumiraj sledeći tekst na srpskom jeziku.\n"
            "Vrati samo sažetak, bez objašnjenja.\n"
            f"Sažetak neka bude između {min_len} i {max_len} reči.\n\n"
            f"Tekst:\n{text_cyr}\n\n"
            "Sažetak:"
        )
        summary = _ollama_generate(prompt, temperature=0.2)
    summary_latin = cyr_to_lat(summary) if is_latin else summary
    translation = translator.translate(summary_latin, dest="en").text
    return summary_latin, translation

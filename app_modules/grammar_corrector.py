import re
import requests
from app_modules.model_config import get_config, get_openai_client

OLLAMA_URL = "http://localhost:11434/api/generate"

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _correct_local(sentence: str) -> str | None:
    prompt = f"""
Ti si lektor za srpski jezik koji koristi ekavicu.
Ispravi sledeću rečenicu (pravopis, gramatika, slaganje).
Ne menjaj značenje.
Zadrži originalni redosled reči ako je gramatički ispravan.
Ako rečenica nema grešaka, vrati TAČNO ISTU rečenicu.
Vrati samo rečenicu, bez objašnjenja.

Rečenica: {sentence}
Ispravljeno:
""".strip()
    payload = {
        "model": get_config()["local"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05},
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    corrected = response.json()["response"].strip()
    if corrected == sentence:
        return None
    return corrected


def _correct_remote(sentence: str, cfg: dict) -> str | None:
    r_cfg = cfg["remote"]
    client = get_openai_client(r_cfg["base_url"], r_cfg["api_key"])
    print(f"[VLLM-CALL] task=grammar mode={cfg['mode']} url={r_cfg['base_url']}", flush=True)
    r = client.chat.completions.create(
        model=r_cfg["model"],
        messages=[
            {"role": "system", "content": "Ispravi gramatičke greške u sledećoj rečenici."},
            {"role": "user",   "content": sentence},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    corrected = _THINK_RE.sub('', r.choices[0].message.content or "").strip()
    if corrected == sentence:
        return None
    return corrected


def correct_sentence(sentence: str) -> str | None:
    cfg = get_config()
    if cfg["mode"] == "remote":
        return _correct_remote(sentence, cfg)
    return _correct_local(sentence)

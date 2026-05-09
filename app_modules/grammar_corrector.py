import json
import re
import requests
from app_modules.model_config import get_config, get_openai_client, OLLAMA_URL

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


def _parse_explain_json(raw: str) -> list[dict]:
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [
                {
                    "original":    str(c.get("original",    "")),
                    "corrected":   str(c.get("corrected",   "")),
                    "explanation": str(c.get("explanation", "")),
                }
                for c in result if isinstance(c, dict)
            ]
    except Exception:
        pass
    return []


def _explain_local(original: str, corrected: str) -> list[dict]:
    prompt = (
        "You are a Serbian grammar expert.\n"
        "For each word that changed between the original and corrected sentence, "
        "provide a brief English explanation of WHY it was corrected.\n"
        "Return ONLY a JSON array — no markdown, no extra text.\n"
        "Each element: {\"original\": \"...\", \"corrected\": \"...\", \"explanation\": \"...\"}\n"
        "If there are no changes, return [].\n\n"
        f"Original:  {original}\n"
        f"Corrected: {corrected}\n"
        "JSON:"
    )
    payload = {
        "model": get_config()["local"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return _parse_explain_json(response.json().get("response", ""))


def _explain_remote(original: str, corrected: str, cfg: dict) -> list[dict]:
    r_cfg  = cfg["remote"]
    client = get_openai_client(r_cfg["base_url"], r_cfg["api_key"])
    system = (
        "You are a Serbian grammar expert. "
        "For each word that changed between the original and corrected sentence, "
        "provide a brief English explanation. "
        'Return ONLY a JSON array: [{"original":"...","corrected":"...","explanation":"..."}]. '
        "If no changes, return []."
    )
    r = client.chat.completions.create(
        model=r_cfg["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Original: {original}\nCorrected: {corrected}"},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    raw = _THINK_RE.sub('', r.choices[0].message.content or "").strip()
    return _parse_explain_json(raw)


def explain_corrections(original: str, corrected: str) -> list[dict]:
    """One LLM call: explain every word-level change between original and corrected."""
    if not original or not corrected or original == corrected:
        return []
    cfg = get_config()
    if cfg["mode"] == "remote":
        return _explain_remote(original, corrected, cfg)
    return _explain_local(original, corrected)

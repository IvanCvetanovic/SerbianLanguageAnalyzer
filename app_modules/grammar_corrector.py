import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def correct_sentence(sentence: str) -> str | None:
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
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    corrected = response.json()["response"].strip()

    if corrected == sentence:
        return None

    return corrected

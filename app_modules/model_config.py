from __future__ import annotations

import json
from pathlib import Path
from openai import OpenAI

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_DEFAULTS: dict = {
    "mode": "local",
    "local": {
        "model": "llama3.1:8b",
    },
    "remote": {
        "base_url": "http://localhost:8001/v1",
        "model": "fulltune_14b_v3",
        "api_key": "not-needed",
    },
}


def get_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        cfg = {**_DEFAULTS, **data}
        cfg["local"]  = {**_DEFAULTS["local"],  **data.get("local",  {})}
        cfg["remote"] = {**_DEFAULTS["remote"], **data.get("remote", {})}
        return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        return {**_DEFAULTS, "local": dict(_DEFAULTS["local"]), "remote": dict(_DEFAULTS["remote"])}


def save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_backend_description() -> str:
    cfg = get_config()
    if cfg["mode"] == "remote":
        r = cfg["remote"]
        return f"Model backend: REMOTE ({r['base_url']}, model={r['model']})"
    return f"Model backend: LOCAL (Ollama, model={cfg['local']['model']})"


_client_cache: dict[tuple, OpenAI] = {}


def get_openai_client(base_url: str, api_key: str) -> OpenAI:
    key = (base_url, api_key)
    if key not in _client_cache:
        _client_cache[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _client_cache[key]

import os
import threading
from pathlib import Path
import whisper
from src.core.model_config import get_config

_whisper_lock  = threading.Lock()
_whisper_state = {"status": "idle", "model": None, "message": ""}


def _is_whisper_cached(model_size: str) -> bool:
    return (Path.home() / ".cache" / "whisper" / f"{model_size}.pt").exists()


def get_whisper_status():
    with _whisper_lock:
        return dict(_whisper_state)


class VoiceTranscriber:
    def __init__(self):
        self._model = None
        self._loaded_size = None
        self._lock = threading.Lock()

    def load(self, model_size: str) -> None:
        """Load (or swap to) a Whisper model. Blocks until complete."""
        with self._lock:
            if self._loaded_size == model_size and self._model is not None:
                return  # already loaded
        print(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        with self._lock:
            self._model = model
            self._loaded_size = model_size
        print(f"Whisper '{model_size}' ready.")

    def transcribe_audio_file(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found at '{audio_path}'."

        cfg_size = get_config().get("whisper_model", "medium")
        with self._lock:
            needs_load = self._model is None or self._loaded_size != cfg_size
            model = self._model if not needs_load else None

        if needs_load:
            self.load(cfg_size)
            with self._lock:
                model = self._model

        try:
            result = model.transcribe(audio_path, language="sr")
            return result["text"].strip()
        except Exception as e:
            return f"An error occurred during transcription: {e}"


def _load_whisper_background(transcriber: VoiceTranscriber, model_size: str):
    cached = _is_whisper_cached(model_size)
    msg = ("Loading model into memory…" if cached
           else f"Downloading {model_size} model — this may take a few minutes…")
    with _whisper_lock:
        _whisper_state.update({"status": "loading", "model": model_size, "message": msg})
    try:
        transcriber.load(model_size)
        with _whisper_lock:
            _whisper_state.update({"status": "ready", "model": model_size,
                                   "message": f"{model_size.capitalize()} model ready."})
    except Exception as e:
        with _whisper_lock:
            _whisper_state.update({"status": "error", "model": model_size,
                                   "message": f"Failed to load {model_size}: {e}"})


def start_whisper_background_loading(transcriber: VoiceTranscriber, model_size: str):
    threading.Thread(
        target=_load_whisper_background,
        args=(transcriber, model_size),
        daemon=True,
    ).start()

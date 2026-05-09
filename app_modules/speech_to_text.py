import os
import threading
import whisper
from app_modules.model_config import get_config


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
        print(f"Whisper '{model_size}' ready. ✅")

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

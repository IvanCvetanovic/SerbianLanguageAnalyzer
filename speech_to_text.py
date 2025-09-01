import whisper
import os

class VoiceTranscriber:
    def __init__(self, model_size: str = "small"):
        print(f"Loading Whisper model '{model_size}'...")
        try:
            self.model = whisper.load_model(model_size)
            print("Whisper model loaded successfully. ✅")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

    def transcribe_audio_file(self, audio_path: str) -> str:
        if not self.model:
            return "Error: Whisper model is not loaded."

        if not os.path.exists(audio_path):
            return f"Error: Audio file not found at '{audio_path}'."

        try:
            print(f"Transcribing audio file: {audio_path}")
            result = self.model.transcribe(audio_path, language="sr")
            transcribed_text = result["text"]
            print(f"Transcription successful.")
            return transcribed_text
        except Exception as e:
            error_message = f"An error occurred during transcription: {e}"
            print(error_message)
            return error_message
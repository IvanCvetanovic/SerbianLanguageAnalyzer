from pathlib import Path
from apiflask import APIFlask
from src.core.model_config import get_config, get_backend_description
from src.services.analysis_pipeline import transcriber
from src.services.speech_to_text import start_whisper_background_loading
from src.api.v1 import api_v1
from src.ui.routes import ui_bp

app = APIFlask(
    __name__,
    title='Serbian NLP Analysis API',
    version='1.0',
    template_folder='templates',
    static_folder='static',
    docs_path='/docs',
)
app.info = {
    'description': (
        'REST API for Serbian language analysis — sentiment, grammar correction, '
        'SRL, ABSA, NER, summarization, topic modelling, and voice transcription. '
        'Use POST /api/v1/analyze for the full pipeline (async), or individual '
        'module endpoints for targeted analysis.'
    ),
}
app.config['SPEC_FORMAT'] = 'json'
app.jinja_env.globals.update(zip=zip)

UPLOAD_FOLDER = Path(__file__).resolve().parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

app.register_blueprint(api_v1)
app.register_blueprint(ui_bp)

# Start Whisper model management in background
start_whisper_background_loading(
    transcriber,
    get_config().get("whisper_model", "medium")
)

print(get_backend_description())

if __name__ == "__main__":
    app.run(debug=False)

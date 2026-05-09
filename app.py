import sys
import uuid
import atexit
import signal
import random
import threading
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app_modules.sentence_generator import SentenceGenerator
from app_modules.fairy_tales import FairyTale
from app_modules.speech_to_text import VoiceTranscriber
from app_modules.model_config import get_config, save_config, get_backend_description
from app_modules.analysis_pipeline import run_analysis

import pyvis

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")
app.jinja_env.globals.update(zip=zip)

UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

pyvis_path = Path(pyvis.__file__).resolve().parent
@app.route('/lib/<path:filename>')
def pyvis_static(filename):
    return send_from_directory(pyvis_path / 'lib', filename)

transcriber = VoiceTranscriber()

# ── Whisper model management ──────────────────────────────────────────────────
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
_whisper_lock  = threading.Lock()
_whisper_state = {"status": "idle", "model": None, "message": ""}

def _is_whisper_cached(model_size: str) -> bool:
    return (Path.home() / ".cache" / "whisper" / f"{model_size}.pt").exists()

def _load_whisper_background(model_size: str):
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

# Start loading the configured Whisper model in the background at startup
threading.Thread(
    target=_load_whisper_background,
    args=(get_config().get("whisper_model", "medium"),),
    daemon=True,
).start()

print(get_backend_description())

executor = ThreadPoolExecutor(max_workers=2)
atexit.register(executor.shutdown, wait=False, cancel_futures=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

MAX_INPUT_CHARS    = 15_000
_JOB_TTL_SECONDS   = 3600

jobs     = {}
progress = {}


def _prune_old_jobs():
    cutoff = datetime.utcnow()
    stale = [
        jid for jid, job in list(jobs.items())
        if (cutoff - job["created_at"]).total_seconds() > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        jobs.pop(jid, None)
        progress.pop(jid, None)


@app.route("/", methods=["GET", "POST"])
def home():
    view_data = {
        "words": [],
        "lemmas": [],
        "ner_results": [],
        "original_input": "",
        "translated_sentence": "",
        "dependency_tree_img": None,
        "extractive_summary": [],
        "abstractive_summary": None,
        "topics": [],
        "zipped_data": [],
        "word_cloud_image": None,
        "ner_heatmap_image": None,
        "pos_sunburst_image": None,
        "error_message": None,
        "sentiment": None,
        "sentence_sentiments": [],
        "grammar_suggestion": None,
        "grammar_applied": False,
        "selected_features": ["translation", "summaries", "topic", "visuals", "graphs", "grammar"],
        "grammar_error": None
    }

    if request.method == "POST":
        selected_features = request.form.getlist("features")
        view_data["selected_features"] = selected_features

        input_string = ""
        if 'audio_file' in request.files and request.files['audio_file'].filename != '':
            file = request.files['audio_file']
            safe_name = secure_filename(file.filename) or "upload"
            filepath = Path(app.config['UPLOAD_FOLDER']) / safe_name
            file.save(filepath)
            try:
                input_string = transcriber.transcribe_audio_file(str(filepath))
            finally:
                try:
                    filepath.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            input_string = (request.form.get("input") or "").strip()

        if not input_string:
            view_data["error_message"] = "Input cannot be empty!"
            return render_template("index.html", **view_data)

        if len(input_string) > MAX_INPUT_CHARS:
            view_data["error_message"] = (
                f"Input is too long ({len(input_string):,} characters). "
                f"Please limit input to {MAX_INPUT_CHARS:,} characters."
            )
            return render_template("index.html", **view_data)

        _prune_old_jobs()
        job_id = str(uuid.uuid4())
        progress[job_id] = {"pct": 0, "stage": "Queued", "status": "running"}

        future = executor.submit(run_analysis, input_string, selected_features, job_id, progress)
        jobs[job_id] = {
            "future": future,
            "status": "running",
            "result": None,
            "created_at": datetime.utcnow(),
            "error": None
        }

        return render_template("index.html", job_id=job_id, **view_data)

    return render_template("index.html", **view_data)


@app.route("/progress/<job_id>")
def get_progress(job_id):
    info = progress.get(job_id)
    if not info:
        return jsonify({"error": "unknown job id"}), 404
    return jsonify(info)


@app.route("/results/<job_id>")
def show_results(job_id):
    if job_id not in progress:
        return "Job not found or expired.", 404
    job = jobs.get(job_id)
    if job:
        fut = job["future"]
        if fut.done() and job.get("result") is None:
            try:
                job["result"] = fut.result()
                job["status"] = "finished"
                if progress[job_id].get("status") != "failed":
                    progress[job_id].update({"status": "finished", "pct": 100, "stage": "Finished"})
            except Exception as e:
                job["status"] = "failed"
                progress[job_id].update({"status": "failed", "stage": "Error", "error_message": str(e)})
    return render_template("index.html", job_id=job_id,
                           original_input="", selected_features=[])


@app.route("/dependency_tree/<job_id>")
def get_dependency_tree(job_id):
    html = (progress.get(job_id, {})
                    .get("sections", {})
                    .get("dependency_tree", {})
                    .get("dependency_tree_img"))
    if not html:
        return "Dependency tree not available.", 404
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/get_random_sentence")
def get_random_sentence():
    fairy_tale = random.choice(list(FairyTale))
    sentence = SentenceGenerator.get_random_sentence(fairy_tale.get_url())
    return jsonify({"sentence": sentence})


@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request.'})
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})

    tmp_path = UPLOAD_FOLDER / 'temp_recording.webm'
    file.save(tmp_path)
    try:
        transcribed_text = transcriber.transcribe_audio_file(str(tmp_path))
        return jsonify({'text': transcribed_text})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/whisper-status")
def api_whisper_status():
    with _whisper_lock:
        return jsonify(dict(_whisper_state))


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(get_config())


@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body"}), 400
    if data.get("mode") not in ("local", "remote"):
        return jsonify({"error": "mode must be 'local' or 'remote'"}), 400
    cfg = get_config()
    old_whisper = cfg.get("whisper_model", "medium")

    cfg["mode"] = data["mode"]
    if isinstance(data.get("local"), dict):
        if "model" in data["local"]:
            cfg["local"]["model"] = str(data["local"]["model"]).strip()
    if isinstance(data.get("remote"), dict):
        for k in ("base_url", "model", "api_key"):
            if k in data["remote"]:
                cfg["remote"][k] = str(data["remote"][k]).strip()
    new_whisper = str(data.get("whisper_model", "")).strip()
    if new_whisper in WHISPER_MODELS:
        cfg["whisper_model"] = new_whisper

    save_config(cfg)

    if cfg.get("whisper_model", "medium") != old_whisper:
        threading.Thread(
            target=_load_whisper_background,
            args=(cfg["whisper_model"],),
            daemon=True,
        ).start()

    return jsonify({"ok": True, "config": cfg})


@app.route("/api/test-connection", methods=["POST"])
def api_test_connection():
    data = request.get_json(force=True) or {}
    mode = data.get("mode") or get_config()["mode"]
    if mode == "local":
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            return jsonify({"ok": True, "message": "Ollama is reachable."})
        except Exception as e:
            return jsonify({"ok": False, "message": f"Ollama not reachable: {e}"})
    else:
        base_url = str(data.get("base_url") or get_config()["remote"]["base_url"]).rstrip("/")
        api_key  = str(data.get("api_key")  or get_config()["remote"]["api_key"])
        try:
            headers = {} if api_key == "not-needed" else {"Authorization": f"Bearer {api_key}"}
            r = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            r.raise_for_status()
            return jsonify({"ok": True, "message": "Remote endpoint is reachable."})
        except Exception as e:
            return jsonify({"ok": False, "message": f"Remote not reachable: {e}"})


if __name__ == "__main__":
    app.run(debug=False)

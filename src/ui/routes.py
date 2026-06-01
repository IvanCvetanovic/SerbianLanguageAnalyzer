import uuid
import random
import requests
from pathlib import Path
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename

from src.services.sentence_generator import SentenceGenerator
from src.services.fairy_tales import FairyTale
from src.core.model_config import get_config, save_config
from src.infrastructure.task_service import task_service
from src.services.analysis_pipeline import run_analysis, transcriber
from src.services.speech_to_text import get_whisper_status, start_whisper_background_loading
from src.core.constants import WHISPER_MODELS, MAX_INPUT_CHARS

import pyvis

ui_bp = Blueprint('ui', __name__)

pyvis_path = Path(pyvis.__file__).resolve().parent


@ui_bp.route('/lib/<path:filename>')
def pyvis_static(filename):
    return send_from_directory(pyvis_path / 'lib', filename)


@ui_bp.route("/", methods=["GET", "POST"])
def home():
    view_data = {
        "words": [], "lemmas": [], "ner_results": [],
        "original_input": "", "translated_sentence": "",
        "dependency_tree_img": None, "extractive_summary": [],
        "abstractive_summary": None, "topics": [], "zipped_data": [],
        "word_cloud_image": None, "ner_heatmap_image": None,
        "pos_sunburst_image": None, "error_message": None,
        "sentiment": None, "sentence_sentiments": [],
        "grammar_suggestion": None, "grammar_applied": False,
        "selected_features": ["translation", "summaries", "topic", "visuals", "graphs", "grammar",
                              "sentiment", "hate", "absa", "srl", "table"],
        "grammar_error": None,
    }

    if request.method == "POST":
        selected_features = request.form.getlist("features")
        view_data["selected_features"] = selected_features

        input_string = ""
        if 'audio_file' in request.files and request.files['audio_file'].filename != '':
            file = request.files['audio_file']
            safe_name = secure_filename(file.filename) or "upload"
            filepath = Path(current_app.config['UPLOAD_FOLDER']) / safe_name
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

        if not selected_features:
            view_data["error_message"] = "Pick at least one feature to analyze."
            return render_template("index.html", **view_data)

        if len(input_string) > MAX_INPUT_CHARS:
            view_data["error_message"] = (
                f"Input is too long ({len(input_string):,} characters). "
                f"Please limit input to {MAX_INPUT_CHARS:,} characters."
            )
            return render_template("index.html", **view_data)

        job_id = task_service.submit_task(run_analysis, input_string, selected_features)
        return render_template("index.html", job_id=job_id, **view_data)

    return render_template("index.html", **view_data)


@ui_bp.route("/progress/<job_id>")
def get_progress(job_id):
    info = task_service.get_progress(job_id)
    if not info:
        return jsonify({"error": "unknown job id"}), 404
    return jsonify(info)


@ui_bp.route("/results/<job_id>")
def show_results(job_id):
    if not task_service.get_progress(job_id):
        return "Job not found or expired.", 404
    
    task_service.update_job_result(job_id)
    return render_template("index.html", job_id=job_id, original_input="", selected_features=[])


@ui_bp.route("/dependency_tree/<job_id>")
def get_dependency_tree(job_id):
    prog = task_service.get_progress(job_id)
    html = (prog.get("sections", {})
                .get("dependency_tree", {})
                .get("dependency_tree_img") if prog else None)
    if not html:
        return "Dependency tree not available.", 404
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@ui_bp.route("/get_random_sentence")
def get_random_sentence():
    fairy_tale = random.choice(list(FairyTale))
    sentence = SentenceGenerator.get_random_sentence(fairy_tale.get_url())
    return jsonify({"sentence": sentence})


@ui_bp.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request.'})
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    tmp_path = Path(current_app.config['UPLOAD_FOLDER']) / 'temp_recording.webm'
    file.save(tmp_path)
    try:
        return jsonify({'text': transcriber.transcribe_audio_file(str(tmp_path))})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@ui_bp.route("/api/whisper-status")
def api_whisper_status():
    return jsonify(get_whisper_status())


@ui_bp.route("/compare")
def compare_page():
    return render_template("compare.html", default_config=get_config())


@ui_bp.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify(get_config())


@ui_bp.route("/api/settings", methods=["POST"])
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
        start_whisper_background_loading(transcriber, cfg["whisper_model"])
    return jsonify({"ok": True, "config": cfg})


@ui_bp.route("/api/test-connection", methods=["POST"])
def api_test_connection():
    data     = request.get_json(force=True) or {}
    mode     = data.get("mode") or get_config()["mode"]
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

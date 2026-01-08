import os
import uuid
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, send_from_directory
from app_modules.local_translator import LocalSrToEnTranslator
from app_modules.sentiment_analyzer import SerbianSentimentAnalyzer
from app_modules.summarizer import extractive_summary, abstractive_summary
from app_modules.topic_modeller import get_topics
from app_modules.word_controller import WordController
from app_modules.sentence_generator import SentenceGenerator
from app_modules.fairy_tales import FairyTale
from app_modules.text_analyzer import TextAnalyzer
from app_modules.speech_to_text import VoiceTranscriber
from app_modules.graph_maker import Visualizer
from app_modules.hate_speech_detector import analyze_hate_speech

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

translator          = LocalSrToEnTranslator()
transcriber         = VoiceTranscriber(model_size="medium")
text_analyzer       = TextAnalyzer()
visualizer          = Visualizer()
sentiment_analyzer  = SerbianSentimentAnalyzer()

executor = ThreadPoolExecutor(max_workers=2)

jobs = {}
progress = {}

def _report(job_id: str, pct: int, stage: str):
    info = progress.get(job_id, {})
    info.update({"pct": int(pct), "stage": stage})
    progress[job_id] = info

def run_analysis(input_text, features, job_id: str):
    progress[job_id] = {"pct": 0, "stage": "Queued", "status": "running"}

    result = {
        "original_input": input_text,
        "translated_sentence": "",
        "words": [],
        "zipped_data": [],
        "extractive_summary": [],
        "abstractive_summary": None,
        "topics": [],
        "word_cloud_image": None,
        "ner_heatmap_image": None,
        "pos_sunburst_image": None,
        "dependency_tree_img": None,
        "ner_results": [],
        "error_message": None,
        "sentiment": None,
        "sentence_sentiments": [],
        "hate_speech": None
    }

    MAX_LINK_LOOKUPS = 40
    BATCH_TRANSLATE_SIZE = 50
    SAFE_DEFAULT = "/"

    try:
        _report(job_id, 5, "Tokenizing")
        words = WordController.split_into_words(input_text)
        result["words"] = words

        if "translation" in features:
            _report(job_id, 15, "Translating sentence")
            try:
                result["translated_sentence"] = WordController.translate_to_english(input_text, translator)
            except Exception:
                result["translated_sentence"] = SAFE_DEFAULT

        _report(job_id, 25, "Preparing words")
        latin_words = WordController.transliterate_cyrillic_to_latin(words)
        lemmas = WordController.lemmatize_words(latin_words)
        transliterated_lemmas = WordController.transliterate_latin_to_cyrillic(lemmas)
        unique_lemmas = list(dict.fromkeys(lemmas))

        _report(job_id, 35, "Reading local dictionaries")
        try:
            definitions_list = WordController.find_local_definitions(transliterated_lemmas)
        except Exception:
            definitions_list = [SAFE_DEFAULT] * len(lemmas)

        _report(job_id, 45, "Tagging morphology")
        try:
            word_types   = WordController.get_word_types(latin_words)
            word_numbers = WordController.get_word_numbers(latin_words)
            word_persons = WordController.get_word_persons(latin_words)
            word_cases   = WordController.get_word_cases(latin_words)
            word_genders = WordController.get_word_genders(latin_words)
        except Exception:
            n = len(latin_words)
            word_types   = [SAFE_DEFAULT] * n
            word_numbers = [SAFE_DEFAULT] * n
            word_persons = [SAFE_DEFAULT] * n
            word_cases   = [SAFE_DEFAULT] * n
            word_genders = [SAFE_DEFAULT] * n

        _report(job_id, 55, "NER & dependency parsing")
        try:
            result["ner_results"] = text_analyzer.analyze_named_entities(input_text)
            dp_results = text_analyzer.analyze_dependency_parsing(input_text)
            word_heads   = [dp.get("head", SAFE_DEFAULT)   for dp in dp_results]
            word_deprels = [dp.get("deprel", SAFE_DEFAULT) for dp in dp_results]
        except Exception:
            word_heads   = [SAFE_DEFAULT] * len(lemmas)
            word_deprels = [SAFE_DEFAULT] * len(lemmas)

        _report(job_id, 60, "Sentiment analysis")
        try:
            analysis = sentiment_analyzer.analyze(
                input_text,
                mode="sentences",
                aggregation="mean",
                max_length=256,
            )
            result["sentiment"] = {
                "label": analysis.overall.label,
                "confidence": analysis.overall.confidence,
                "scores": analysis.overall.scores,
            }

            sentences = sentiment_analyzer.split_sentences(input_text)
            result["sentence_sentiments"] = [
                {
                    "sentence": s,
                    "label": r.label,
                    "confidence": r.confidence,
                    "scores": r.scores,
                }
                for s, r in zip(sentences, analysis.sentences)
            ]
        except Exception:
            result["sentiment"] = None
            result["sentence_sentiments"] = []

        _report(job_id, 66, "Hate speech detection")
        try:
            raw_overall = analyze_hate_speech(input_text)

            overall_hate = {
                "flagged": raw_overall["flagged"],
                "label": raw_overall["label"],
                "score": raw_overall["score"],
                "scores": raw_overall["scores"],
                "reasons": raw_overall["reasons"],
            }

            sentences = sentiment_analyzer.split_sentences(input_text)
            sentence_hate = []
            for s in sentences:
                raw = analyze_hate_speech(s)
                sentence_hate.append({
                    "sentence": s,
                    "flagged": raw["flagged"],
                    "label": raw["label"],
                    "score": raw["score"],
                })

            result["hate_speech"] = {
                "overall": overall_hate,
                "sentences": sentence_hate
            }
        except Exception:
            result["hate_speech"] = None

        _report(job_id, 62, "Translating lemmas")
        translation_map = {lemma: SAFE_DEFAULT for lemma in unique_lemmas}
        try:
            for start in range(0, len(unique_lemmas), BATCH_TRANSLATE_SIZE):
                chunk = unique_lemmas[start:start + BATCH_TRANSLATE_SIZE]
                translated = translator.translate(chunk, src="sr", dest="en")
                if not isinstance(translated, list):
                    translated = [translated]
                for src_word, t in zip(chunk, translated):
                    translation_map[src_word] = getattr(t, "text", SAFE_DEFAULT) or SAFE_DEFAULT
        except Exception:
            pass

        _report(job_id, 68, "Finding online links")
        link_map = {lemma: SAFE_DEFAULT for lemma in unique_lemmas}
        try:
            to_check = unique_lemmas[:MAX_LINK_LOOKUPS]
            checked = WordController.process_links_for_lemmas(to_check)
            for lemma, url in zip(to_check, checked):
                link_map[lemma] = url or SAFE_DEFAULT
        except Exception:
            pass

        if "graphs" in features:
            _report(job_id, 75, "Rendering dependency tree")
            try:
                result["dependency_tree_img"] = text_analyzer.visualize_dependency_tree(input_text)
            except Exception:
                result["dependency_tree_img"] = None

        if "summaries" in features:
            _report(job_id, 82, "Summarizing")
            try:
                result["extractive_summary"] = extractive_summary(input_text, num_sentences=2)
            except Exception:
                result["extractive_summary"] = []
            try:
                result["abstractive_summary"] = abstractive_summary(input_text, translator)
            except Exception:
                result["abstractive_summary"] = None

        if "topic" in features:
            _report(job_id, 86, "Topic modelling")
            try:
                result["topics"] = get_topics([input_text], translator)
            except Exception:
                result["topics"] = []

        if "visuals" in features:
            _report(job_id, 92, "Building visualizations")
            try:
                doc_for_viz = text_analyzer.pipeline(input_text)
                if lemmas:
                    from collections import Counter
                    frequencies = Counter(lemmas)
                    result["word_cloud_image"] = visualizer.generate_word_cloud(frequencies)
                result["ner_heatmap_image"]  = visualizer.generate_ner_heatmap(doc_for_viz)
                result["pos_sunburst_image"] = visualizer.generate_pos_sunburst(doc_for_viz)
            except Exception:
                result["word_cloud_image"]   = None
                result["ner_heatmap_image"]  = None
                result["pos_sunburst_image"] = None

        _report(job_id, 96, "Assembling table")
        per_word_translations = [translation_map.get(lem, SAFE_DEFAULT) for lem in lemmas]
        per_word_links        = [link_map.get(lem, SAFE_DEFAULT)        for lem in lemmas]

        result["zipped_data"] = list(zip(
            per_word_translations,
            lemmas,
            definitions_list,
            per_word_links,
            word_types,
            word_numbers,
            word_persons,
            word_cases,
            word_genders,
            word_heads,
            word_deprels,
        ))

        progress[job_id].update({"pct": 100, "stage": "Finished", "status": "finished"})
        return result

    except Exception as e:
        result["error_message"] = str(e)
        progress[job_id].update({
            "status": "failed",
            "stage": "Error",
            "pct": progress[job_id].get("pct", 0)
        })
        return result

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
        "sentence_sentiments": []
    }

    if request.method == "POST":
        selected_features = request.form.getlist("features")

        input_string = ""
        if 'audio_file' in request.files and request.files['audio_file'].filename != '':
            file = request.files['audio_file']
            filepath = Path(app.config['UPLOAD_FOLDER']) / file.filename
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

        job_id = str(uuid.uuid4())
        progress[job_id] = {"pct": 0, "stage": "Queued", "status": "running"}

        future = executor.submit(run_analysis, input_string, selected_features, job_id)
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
    job = jobs.get(job_id)
    if not job:
        return "Job not found", 404

    fut = job["future"]
    if fut.done() and job["result"] is None:
        try:
            job["result"] = fut.result()
            job["status"] = "finished"
            if job_id in progress and progress[job_id].get("status") != "failed":
                progress[job_id].update({"status": "finished", "pct": 100, "stage": "Finished"})
        except Exception as e:
            job["status"] = "failed"
            job["result"] = {"error_message": str(e)}
            if job_id in progress:
                progress[job_id].update({"status": "failed", "stage": "Error"})

    if job["status"] != "finished":
        return render_template("index.html", error_message="Job not finished yet.")

    return render_template("index.html", **job["result"])


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

if __name__ == "__main__":
    app.run(debug=False)

import sys
import os
import random
import collections
from googletrans import Translator
from flask import Flask, render_template, request, jsonify, send_from_directory

from app_modules.summarizer import extractive_summary, abstractive_summary
from app_modules.topic_modeller import get_topics
from app_modules.word_controller import WordController
from app_modules.sentence_generator import SentenceGenerator
from app_modules.fairy_tales import FairyTale
from app_modules.text_analyzer import TextAnalyzer
from app_modules.speech_to_text import VoiceTranscriber
from app_modules.graph_maker import Visualizer


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

translator = Translator()
transcriber = VoiceTranscriber(model_size="medium")
text_analyzer = TextAnalyzer()
visualizer = Visualizer()

pyvis_path = os.path.join(os.path.dirname(__file__), ".venv/Lib/site-packages/pyvis")
app.add_url_rule(
    '/lib/path:filename>',
    'pyvis_static',
    lambda filename: send_from_directory(os.path.join(pyvis_path, 'lib'), filename)
)

@app.route("/", methods=["GET", "POST"])
def home():
    view_data = {
        "words": [], "lemmas": [], "ner_results": [], "original_input": "",
        "translated_sentence": "", "dependency_tree_img": None,
        "extractive_summary": [], "abstractive_summary": None, "topics": [],
        "zipped_data": [], "word_cloud_image": None, "ner_heatmap_image": None,
        "pos_sunburst_image": None, "error_message": None
    }

    if request.method == "POST":
        input_string = ""

        if 'audio_file' in request.files and request.files['audio_file'].filename != '':
            file = request.files['audio_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            input_string = transcriber.transcribe_audio_file(filepath)
            os.remove(filepath)
        else:
            input_string = request.form.get("input", "").strip()

        if not input_string:
            view_data["error_message"] = "Input cannot be empty!"
            return render_template("index.html", **view_data)

        view_data["original_input"] = input_string
        view_data["translated_sentence"] = WordController.translate_sentence(input_string, translator)
        view_data["words"] = WordController.split_into_words(input_string)
        
        latin_words = WordController.transliterate_cyrillic_to_latin(view_data["words"])
        lemmas = WordController.lemmatize_words(latin_words)
        transliterated_lemmas = WordController.transliterate_latin_to_cyrillic(lemmas)
        
        definitions = WordController.find_local_definitions(transliterated_lemmas)
        online_definitions = WordController.process_links_for_lemmas(transliterated_lemmas)
        translations = WordController.translate_words(lemmas, translator)
        word_types = WordController.get_word_types(latin_words)
        word_numbers = WordController.get_word_numbers(latin_words)
        word_persons = WordController.get_word_persons(latin_words)
        word_cases = WordController.get_word_cases(latin_words)
        word_genders = WordController.get_word_genders(latin_words)

        view_data["ner_results"] = text_analyzer.analyze_named_entities(input_string)
        dp_results = text_analyzer.analyze_dependency_parsing(input_string)
        word_heads = [dp["head"] for dp in dp_results]
        word_deprels = [dp["deprel"] for dp in dp_results]
        view_data["dependency_tree_img"] = text_analyzer.visualize_dependency_tree(input_string)

        view_data["extractive_summary"] = extractive_summary(input_string, num_sentences=2)
        view_data["abstractive_summary"] = abstractive_summary(input_string, translator)
        view_data["topics"] = get_topics([input_string], translator)

        doc_for_viz = text_analyzer.pipeline(input_string)
        
        if lemmas:
            frequencies = collections.Counter(lemmas)
            view_data["word_cloud_image"] = visualizer.generate_word_cloud(frequencies)
        
        view_data["ner_heatmap_image"] = visualizer.generate_ner_heatmap(doc_for_viz)
        view_data["pos_sunburst_image"] = visualizer.generate_pos_sunburst(doc_for_viz)

        view_data["zipped_data"] = zip(
            [translation[1] for translation in translations],
            lemmas,
            definitions,
            online_definitions,
            word_types,
            word_numbers,
            word_persons,
            word_cases,
            word_genders,
            word_heads,
            word_deprels,
        )

        return render_template("index.html", **view_data, zip=zip)

    return render_template("index.html", **view_data)

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

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_recording.webm')
        file.save(filepath)
        transcribed_text = transcriber.transcribe_audio_file(filepath)
        os.remove(filepath)
        return jsonify({'text': transcribed_text})
    
    return jsonify({'error': 'File processing failed.'})

if __name__ == "__main__":
    app.run(debug=False)
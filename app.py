from flask import Flask, render_template, request, jsonify, send_from_directory
from word_controller import WordController
from sentence_generator import SentenceGenerator
from fairy_tales import FairyTale
import os
import random
from text_analyzer import TextAnalyzer  

app = Flask(__name__)

text_analyzer = TextAnalyzer()

# Serve PyVis static files (utils.js) from the PyVis package
pyvis_path = os.path.join(os.path.dirname(__file__), ".venv/Lib/site-packages/pyvis")
app.add_url_rule(
    '/lib/<path:filename>',
    'pyvis_static',
    lambda filename: send_from_directory(os.path.join(pyvis_path, 'lib'), filename)
)

@app.route("/", methods=["GET", "POST"])
def home():
    words = []
    lemmas = []
    word_types = []
    word_numbers = []
    word_persons = []
    word_cases = []
    word_genders = []
    word_heads = []
    word_deprels = []
    ner_results = []
    original_input = ""
    translated_sentence = ""
    dependency_tree_img = None
    transliterated_lemmas = []
    definitions = []

    if request.method == "POST":
        input_string = request.form["input"]
        original_input = input_string

        translated_sentence = WordController.translate_sentence(input_string)
        words = WordController.split_into_words(input_string)

        lemmas = WordController.lemmatize_words(words)
        
        # Call transliteration function
        transliterated_lemmas = WordController.transliterate_lemmas(lemmas)
        
        # Find definitions locally
        definitions = WordController.find_local_definitions(transliterated_lemmas)

        online_definitions = WordController.process_links_for_lemmas(transliterated_lemmas)
        print("Online Definitions:", online_definitions)
        
        translations = WordController.translate_words(lemmas)
        word_types = WordController.get_word_types(words)
        word_numbers = WordController.get_word_numbers(words)
        word_persons = WordController.get_word_persons(words)
        word_cases = WordController.get_word_cases(words)
        word_genders = WordController.get_word_genders(words)

        ner_results = text_analyzer.analyze_named_entities(input_string)
        dp_results = text_analyzer.analyze_dependency_parsing(input_string)

        word_heads = [dp["head"] for dp in dp_results]
        word_deprels = [dp["deprel"] for dp in dp_results]

        # Generate the new dependency tree image using PyVis
        dependency_tree_img = text_analyzer.visualize_dependency_tree(input_string)

        # Prepare zipped data with "Local Definition" column
        zipped_data = zip(
            [translation[1] for translation in translations],  # Translation
            lemmas,  # Lemma (Base Form)
            definitions,  # Local Definition
            online_definitions,  # Online Definition
            word_types,  # Word Type
            word_numbers,
            word_persons,
            word_cases,
            word_genders,
            word_heads,
            word_deprels,
        )

        return render_template(
            "index.html",
            words=words,
            lemmas=zipped_data,
            original_input=original_input,
            translated_sentence=translated_sentence,
            ner_results=ner_results,
            dependency_tree_img=dependency_tree_img,  # Pass PyVis graph to the template
            zip=zip,
            online_definitions=online_definitions
        )

    return render_template("index.html", words=words, lemmas=lemmas)

@app.route("/get_random_sentence")
def get_random_sentence():
    fairy_tale = random.choice(list(FairyTale))
    url = fairy_tale.get_url()

    print(f"Selected fairy tale URL: {url}")
    sentence = SentenceGenerator.get_random_sentence(url)
    print(f"Generated sentence: {sentence}")

    return jsonify({"sentence": sentence})

if __name__ == "__main__":
    app.run(debug=False)

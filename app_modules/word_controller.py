from cyrtranslit import to_latin, to_cyrillic
import csv
import requests

from app_modules.pipeline import get_nlp


class WordController:

    @staticmethod
    def split_into_words(input_string):
        doc = get_nlp()(input_string)

        words = []
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append(word.text)

        print(words)
        return words

    @staticmethod
    def translate_to_english(text_or_texts, translator):
        res = translator.translate(text_or_texts, src="sr", dest="en")
        if isinstance(res, list):
            return [r.text for r in res]
        return res.text

    @staticmethod
    def lemmatize_words(words):
        lemmatized_words = []
        nlp = get_nlp()  # reuse pipeline reference

        for word in words:
            doc = nlp(word)
            lemma = doc.sentences[0].words[0].lemma
            lemmatized_words.append(lemma)

        return lemmatized_words

    @staticmethod
    def transliterate_latin_to_cyrillic(words):
        return [to_cyrillic(word, "sr") for word in words]

    @staticmethod
    def transliterate_cyrillic_to_latin(words):
        return [to_latin(word, "sr") for word in words]

    @staticmethod
    def process_links_for_lemmas(transliterated_lemmas):
        base_url = "http://serbiandictionary.com/translate/"
        results = []

        for lemma in transliterated_lemmas:
            url = base_url + lemma
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    results.append(url)
                else:
                    results.append("/")
            except requests.RequestException:
                results.append("/")

        return results

    @staticmethod
    def find_local_definitions(
        transliterated_lemmas,
        csv_file_path=r"C:\Users\PrOfSeS\Desktop\Master Thesis Project\data\Serbian-Wordnet.csv",
    ):
        definitions = []

        with open(csv_file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            wordnet_dict = {}

            for row in reader:
                word = row[0].strip()
                definition = row[2].strip()

                wordnet_dict.setdefault(word, []).append(definition)

            for lemma in transliterated_lemmas:
                definitions.append(wordnet_dict.get(lemma, "/"))

        return definitions

    @staticmethod
    def get_word_types(words):
        word_types = []
        nlp = get_nlp()

        for word in words:
            doc = nlp(word)
            pos_tag = doc.sentences[0].words[0].upos
            word_types.append(pos_tag)

        return word_types

    @staticmethod
    def get_word_numbers(words):
        word_numbers = []
        nlp = get_nlp()

        for word in words:
            doc = nlp(word)
            feats = doc.sentences[0].words[0].feats

            if feats:
                feats_dict = dict(feat.split("=") for feat in feats.split("|"))
                number = feats_dict.get("Number", "/")
            else:
                number = "/"

            word_numbers.append(number)

        return word_numbers

    @staticmethod
    def get_word_persons(words):
        word_persons = []
        nlp = get_nlp()

        for word in words:
            doc = nlp(word)
            pos_tag = doc.sentences[0].words[0].upos

            if pos_tag in {"VERB", "AUX"}:
                feats = doc.sentences[0].words[0].feats
                if feats:
                    feats_dict = dict(feat.split("=") for feat in feats.split("|"))
                    person = feats_dict.get("Person", "/")
                else:
                    person = "/"
            else:
                person = "/"

            word_persons.append(person)

        return word_persons

    @staticmethod
    def get_word_cases(words):
        word_cases = []
        nlp = get_nlp()

        for word in words:
            doc = nlp(word)
            feats = doc.sentences[0].words[0].feats

            if feats:
                feats_dict = dict(feat.split("=") for feat in feats.split("|"))
                case = feats_dict.get("Case", "/")
            else:
                case = "/"

            word_cases.append(case)

        return word_cases

    @staticmethod
    def get_word_genders(words):
        word_genders = []
        nlp = get_nlp()

        for word in words:
            doc = nlp(word)
            feats = doc.sentences[0].words[0].feats

            if feats:
                feats_dict = dict(feat.split("=") for feat in feats.split("|"))
                gender = feats_dict.get("Gender", "/")
            else:
                gender = "/"

            word_genders.append(gender)

        return word_genders

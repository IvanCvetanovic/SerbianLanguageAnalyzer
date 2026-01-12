from cyrtranslit import to_latin, to_cyrillic
import csv
import urllib.parse
import re

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

    _WORDNET_LATIN_CACHE = {}

    @staticmethod
    def _load_wordnet_latin(csv_file_path):
        if csv_file_path in WordController._WORDNET_LATIN_CACHE:
            return WordController._WORDNET_LATIN_CACHE[csv_file_path]

        known = set()
        with open(csv_file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                word_cyr = (row[0] or "").strip()
                if not word_cyr:
                    continue

                word_lat = to_latin(word_cyr, "sr").lower()
                known.add(word_lat)

        WordController._WORDNET_LATIN_CACHE[csv_file_path] = known
        return known

    @staticmethod
    def process_links_for_lemmas(
        lemmas,
        csv_file_path=r"C:\Users\PrOfSeS\Desktop\Master Thesis Project\data\Serbian-Wordnet.csv",
        safe_default="/",
    ):
        base_url = "https://en.pons.com/translate/serbian-english/"
        allowed = re.compile(r"[^0-9A-Za-zŠĐČĆŽšđčćž]+")

        known_words_latin = WordController._load_wordnet_latin(csv_file_path)

        results = []
        for lemma in (lemmas or []):
            if lemma is None:
                results.append(safe_default)
                continue

            w = str(lemma).strip()
            if not w:
                results.append(safe_default)
                continue

            w_latin = to_latin(w, "sr").lower()
            w_clean = allowed.sub(" ", w_latin).strip()
            if not w_clean:
                results.append(safe_default)
                continue

            parts = w_clean.split()
            if len(parts) != 1:
                results.append(safe_default)
                continue

            token = parts[0]

            if token not in known_words_latin:
                results.append(safe_default)
                continue

            token_enc = urllib.parse.quote(token, safe="")
            results.append(base_url + token_enc)

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

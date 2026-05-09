from pathlib import Path
import csv
import urllib.parse
import re

from app_modules.pipeline import get_nlp
from app_modules.transliteration import cyr_to_lat, words_to_latin, words_to_cyrillic

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_WORDNET_CSV = str(_DATA_DIR / "Serbian-Wordnet.csv")


class WordController:

    @staticmethod
    def split_into_words(input_string):
        doc = get_nlp()(input_string)

        words = []
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append(word.text)

        return words

    @staticmethod
    def translate_to_english(text_or_texts, translator):
        res = translator.translate(text_or_texts, src="sr", dest="en")
        if isinstance(res, list):
            return [r.text for r in res]
        return res.text

    @staticmethod
    def _batch_analyze(words):
        """Run the NLP pipeline once on all words instead of once per word.

        Each word is placed on its own paragraph so classla treats it as a
        separate sentence, giving the same result as individual calls but
        with a single pipeline invocation.
        """
        if not words:
            return []
        nlp = get_nlp()
        combined = "\n\n".join(str(w) for w in words)
        doc = nlp(combined)
        analyzed = [sent.words[0] for sent in doc.sentences if sent.words]
        # Pad so the returned list always aligns with the input list
        while len(analyzed) < len(words):
            analyzed.append(None)
        return analyzed

    @staticmethod
    def lemmatize_words(words):
        analyzed = WordController._batch_analyze(words)
        return [
            (w.lemma if w is not None else word)
            for w, word in zip(analyzed, words)
        ]

    @staticmethod
    def transliterate_latin_to_cyrillic(words):
        return words_to_cyrillic(words)

    @staticmethod
    def transliterate_cyrillic_to_latin(words):
        return words_to_latin(words)

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

                word_lat = cyr_to_lat(word_cyr).lower()
                known.add(word_lat)

        WordController._WORDNET_LATIN_CACHE[csv_file_path] = known
        return known

    _WORDNET_DICT_CACHE = {}

    @staticmethod
    def _load_wordnet_dict(csv_file_path):
        if csv_file_path in WordController._WORDNET_DICT_CACHE:
            return WordController._WORDNET_DICT_CACHE[csv_file_path]

        wordnet_dict = {}
        with open(csv_file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 3:
                    continue
                word = row[0].strip()
                definition = row[2].strip()
                wordnet_dict.setdefault(word, []).append(definition)

        WordController._WORDNET_DICT_CACHE[csv_file_path] = wordnet_dict
        return wordnet_dict

    @staticmethod
    def process_links_for_lemmas(
        lemmas,
        csv_file_path=_DEFAULT_WORDNET_CSV,
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

            w_latin = cyr_to_lat(w).lower()
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
        csv_file_path=_DEFAULT_WORDNET_CSV,
    ):
        wordnet_dict = WordController._load_wordnet_dict(csv_file_path)
        return [wordnet_dict.get(lemma, "/") for lemma in transliterated_lemmas]

    @staticmethod
    def get_word_types(words):
        analyzed = WordController._batch_analyze(words)
        return [
            (w.upos if w is not None else "/")
            for w in analyzed
        ]

    @staticmethod
    def get_word_numbers(words):
        analyzed = WordController._batch_analyze(words)
        result = []
        for w in analyzed:
            if w is not None and w.feats:
                feats_dict = dict(feat.split("=") for feat in w.feats.split("|") if "=" in feat)
                result.append(feats_dict.get("Number", "/"))
            else:
                result.append("/")
        return result

    @staticmethod
    def get_word_persons(words):
        analyzed = WordController._batch_analyze(words)
        result = []
        for w in analyzed:
            if w is not None and w.upos in {"VERB", "AUX"} and w.feats:
                feats_dict = dict(feat.split("=") for feat in w.feats.split("|") if "=" in feat)
                result.append(feats_dict.get("Person", "/"))
            else:
                result.append("/")
        return result

    @staticmethod
    def get_word_cases(words):
        analyzed = WordController._batch_analyze(words)
        result = []
        for w in analyzed:
            if w is not None and w.feats:
                feats_dict = dict(feat.split("=") for feat in w.feats.split("|") if "=" in feat)
                result.append(feats_dict.get("Case", "/"))
            else:
                result.append("/")
        return result

    @staticmethod
    def get_word_genders(words):
        analyzed = WordController._batch_analyze(words)
        result = []
        for w in analyzed:
            if w is not None and w.feats:
                feats_dict = dict(feat.split("=") for feat in w.feats.split("|") if "=" in feat)
                result.append(feats_dict.get("Gender", "/"))
            else:
                result.append("/")
        return result

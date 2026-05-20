import csv
import re
from pathlib import Path
from typing import Dict, List, Set

from src.core.transliteration import cyr_to_lat, lat_to_cyr

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_WORDNET_CSV = str(_DATA_DIR / "Serbian-Wordnet.csv")
_DEFAULT_STOPWORDS_CSV = str(_DATA_DIR / "SSWdictionary.csv")


class DictionaryRepository:
    def __init__(self, 
                 wordnet_csv_path: str = _DEFAULT_WORDNET_CSV,
                 stopwords_csv_path: str = _DEFAULT_STOPWORDS_CSV):
        self.wordnet_csv_path = wordnet_csv_path
        self.stopwords_csv_path = stopwords_csv_path
        self._latin_cache: Set[str] | None = None
        self._dict_cache: Dict[str, List[str]] | None = None
        self._stopwords_cache: List[str] | None = None

    def _load_latin_cache(self) -> Set[str]:
        if self._latin_cache is not None:
            return self._latin_cache

        known = set()
        try:
            with open(self.wordnet_csv_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    if not row:
                        continue
                    word_cyr = (row[0] or "").strip()
                    if not word_cyr:
                        continue
                    word_lat = cyr_to_lat(word_cyr).lower()
                    known.add(word_lat)
        except (FileNotFoundError, IOError):
            pass

        self._latin_cache = known
        return known

    def _load_dict_cache(self) -> Dict[str, List[str]]:
        if self._dict_cache is not None:
            return self._dict_cache

        wordnet_dict = {}
        try:
            with open(self.wordnet_csv_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) < 3:
                        continue
                    word = row[0].strip()
                    definition = row[2].strip()
                    wordnet_dict.setdefault(word, []).append(definition)
        except (FileNotFoundError, IOError):
            pass

        self._dict_cache = wordnet_dict
        return wordnet_dict

    def is_word_known(self, word_latin: str) -> bool:
        """Check if a Latin-script Serbian word exists in the dictionary."""
        return word_latin.lower() in self._load_latin_cache()

    def get_definitions(self, lemma: str) -> List[str]:
        """Get local definitions for a Cyrillic lemma."""
        return self._load_dict_cache().get(lemma, [])

    def get_serbian_stopwords(self) -> List[str]:
        """Load and cache Serbian stopwords from SSWdictionary.csv."""
        if self._stopwords_cache is not None:
            return self._stopwords_cache

        stopwords = []
        try:
            with open(self.stopwords_csv_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    w = row[0].strip().lower()
                    w = re.sub(r'^[^\w]+|[^\w]+$', '', w)
                    if len(w) >= 2:
                        stopwords.append(w)
                        # Add Cyrillic transliteration to support both scripts
                        stopwords.append(lat_to_cyr(w))
        except (FileNotFoundError, IOError):
            pass

        self._stopwords_cache = list(set(stopwords))
        return self._stopwords_cache


# Singleton instance for default usage
default_repo = DictionaryRepository()

import requests
from bs4 import BeautifulSoup
import random
import re

_sentence_cache: dict[str, list[str]] = {}

class SentenceGenerator:
    EXCLUDED_SENTENCES = [
        "Priča može da sadržava nasilne, agresivne ili neprikladne elemente u radnji.",
        "Roditeljima se savetuje da prvo pročitaju bajku da bi utvrdili da li je primerena uzrastu deteta, jer postoji mogućnost da je dete osetljivo na takve teme.",
        "Srpske narodne pripovijetke, 1853.",
        "Srpske narodne pripovijetke, 1870.",
        "Objavljena je 1853.",
        "Pročitajte originalnu celu priču online!",
        "Sadrži klasične motive hrabrosti i upornosti kao i druge narodne bajke objavljene u zbirci."
    ]
    MAX_WORDS = 15

    @staticmethod
    def get_random_sentence(url):
        if url not in _sentence_cache:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                story_div = soup.find('div', class_='entry-content')

                if not story_div:
                    _sentence_cache[url] = []
                else:
                    text = story_div.get_text(separator=' ', strip=True)
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    sentences = [s for s in sentences if s and s not in SentenceGenerator.EXCLUDED_SENTENCES]
                    _sentence_cache[url] = [
                        s for s in sentences if len(re.findall(r'\b\w+\b', s)) <= SentenceGenerator.MAX_WORDS
                    ]
            except Exception as e:
                return f"An error occurred: {e}"

        sentences = _sentence_cache.get(url, [])
        if sentences:
            encoded = random.choice(sentences).encode('utf-8')
            return encoded.decode('utf-8', 'ignore')
        return "No suitable sentence found."

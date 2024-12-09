import requests
from bs4 import BeautifulSoup
import random
import re

class SentenceGenerator:
    EXCLUDED_SENTENCES = [
        "Priča može da sadržava nasilne, agresivne ili neprikladne elemente u radnji.",
        "Roditeljima se savetuje da prvo pročitaju bajku da bi utvrdili da li je primerena uzrastu deteta, jer postoji mogućnost da je dete osetljivo na takve teme.",
        "Srpske narodne pripovijetke, 1853.",
        "Srpske narodne pripovijetke, 1870.",
        "Objavljena je 1853.",
        "Pročitajte originalnu celu priču online!",
        "Sadrži klasične motive hrabrosti i upornosti kao i druge narodne bajke objavljene u zbirci."
    ]
    MAX_WORDS = 25  # Max words per sentence allowed for random generation.

    @staticmethod
    def get_random_sentence(url):
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            story_div = soup.find('div', class_='entry-content')

            if not story_div:
                return "Content not found on the page."

            text = story_div.get_text(separator=' ', strip=True)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s for s in sentences if s and s not in SentenceGenerator.EXCLUDED_SENTENCES]

            short_sentences = [
                s for s in sentences if len(re.findall(r'\b\w+\b', s)) <= SentenceGenerator.MAX_WORDS
            ]

            if short_sentences:
                encoded_sentence = random.choice(short_sentences).encode('utf-8')
                return encoded_sentence.decode('utf-8', 'ignore')  # Ignore annoying decoding errors
            else:
                return "No suitable sentence found."
        
        except Exception as e:
            return f"An error occurred: {e}"

from cyrtranslit import to_latin, to_cyrillic
import csv
import requests
from app_modules.pipeline import nlp

class WordController:
    nlp = nlp

    @staticmethod
    def split_into_words(input_string):
        doc = WordController.nlp(input_string)
        
        words = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                words.append(word.text)
        
        print(words)
        return words


    @staticmethod
    def translate_words(words, translator):
        translations = []
        for word in words:
            translated_word = translator.translate(word, src="sr", dest="en").text
            translations.append((word, translated_word))

        return translations
    
    @staticmethod
    def translate_sentence(sentence, translator):
        translation = translator.translate(sentence, src="sr", dest="en")
        return translation.text

    @staticmethod
    def lemmatize_words(words):
        lemmatized_words = []
        for word in words:
            doc = WordController.nlp(word)
            lemma = doc.sentences[0].words[0].lemma
            lemmatized_words.append(lemma)

        return lemmatized_words
    

    def transliterate_latin_to_cyrillic(words):
        cyrillic_words = []
        for word in words:
            cyrillic_words.append(to_cyrillic(word, 'sr'))
        return cyrillic_words

    def transliterate_cyrillic_to_latin(words):
        latin_words = []
        for word in words:
            latin_words.append(to_latin(word, 'sr'))
        return latin_words
    
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
    def find_local_definitions(transliterated_lemmas, csv_file_path="C:\\Users\\PrOfSeS\\Desktop\\Master Thesis Project\\data\\Serbian-Wordnet.csv"):
        definitions = []

        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            wordnet_dict = {}

            for row in reader:
                word = row[0].strip()
                definition = row[2].strip()

                if word not in wordnet_dict:
                    wordnet_dict[word] = []

                wordnet_dict[word].append(definition)

            for lemma in transliterated_lemmas:
                if lemma in wordnet_dict:
                    definitions.append(wordnet_dict[lemma])
                else:
                    definitions.append("/")

        return definitions

    @staticmethod
    def get_word_types(words):
        word_types = []
        for word in words:
            doc = WordController.nlp(word)
            pos_tag = doc.sentences[0].words[0].upos
            word_types.append(pos_tag)
        return word_types
    
    @staticmethod
    def get_word_numbers(words):
        word_numbers = []
        for word in words:
            doc = WordController.nlp(word)
            feats = doc.sentences[0].words[0].feats 
            
            if feats:
                feats_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in feats.split('|')}
                number = feats_dict.get("Number", "/")
            else:
                number = "/"
            
            word_numbers.append(number)
        return word_numbers
    
    @staticmethod
    def get_word_persons(words):
        word_persons = []
        for word in words:
            doc = WordController.nlp(word)
            pos_tag = doc.sentences[0].words[0].upos

            if pos_tag in ["VERB", "AUX"]:
                feats = doc.sentences[0].words[0].feats 
                if feats:
                    feats_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in feats.split('|')}
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
        for word in words:
            doc = WordController.nlp(word)
            feats = doc.sentences[0].words[0].feats
            
            if feats:
                feats_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in feats.split('|')}
                case = feats_dict.get("Case", "/")
            else:
                case = "/" 
            
            word_cases.append(case)
        return word_cases
    
    @staticmethod
    def get_word_genders(words):
        word_genders = []
        for word in words:
            doc = WordController.nlp(word)
            feats = doc.sentences[0].words[0].feats
            
            if feats:
                feats_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in feats.split('|')}
                gender = feats_dict.get("Gender", "/") 
            else:
                gender = "/" 
            
            word_genders.append(gender)
        return word_genders
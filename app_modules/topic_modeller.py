import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def load_serbian_stopwords(csv_path: str) -> list[str]:
    stopwords = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            w = row[0].strip().lower()
            w = re.sub(r'^[^\w]+|[^\w]+$', '', w)
            if len(w) >= 2:
                stopwords.append(w)
    return stopwords

@staticmethod
def translate_to_english(text_or_texts, translator):
    res = translator.translate(text_or_texts, src="sr", dest="en")
    if isinstance(res, list):
        return [r.text for r in res]
    return res.text


def get_topics(
    docs: list[str],
    translator,
    n_topics: int = 1,
    n_top_words: int = 8,
    stopwords_csv: str = "C:\\Users\\PrOfSeS\\Desktop\\Master Thesis Project\\data\\SSWdictionary.csv",
    translate: bool = True
) -> list[list[str]]:
    sr_stop = load_serbian_stopwords(stopwords_csv)
    vectorizer = CountVectorizer(
        stop_words=sr_stop,
        token_pattern=r"(?u)\b\w\w+\b",
        max_df=1.0,
        min_df=1
    )
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        random_state=42
    )
    lda.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = []

    for comp in lda.components_:
        top_idxs = comp.argsort()[-n_top_words:][::-1]
        sr_words = [terms[i] for i in top_idxs]
        
        if translate:
            en_words = translate_to_english(sr_words, translator)
            formatted_topic = [f"{sr} - ({en})" for sr, en in zip(sr_words, en_words)]
            topics.append(formatted_topic)
        else:
            topics.append(sr_words)

    return topics
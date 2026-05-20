from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.infrastructure.repositories import default_repo


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
    repository=default_repo,
    translate: bool = True
) -> list[list[str]]:
    sr_stop = repository.get_serbian_stopwords()
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

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[: -n_top_words - 1 : -1]
        top_words = [terms[i] for i in top_indices]

        if translate and translator:
            try:
                translated_words = translate_to_english(top_words, translator)
                # Pair them up for the UI to display both
                combined = [f"{sr} ({en})" for sr, en in zip(top_words, translated_words)]
                topics.append(combined)
            except Exception:
                topics.append(top_words)
        else:
            topics.append(top_words)

    return topics

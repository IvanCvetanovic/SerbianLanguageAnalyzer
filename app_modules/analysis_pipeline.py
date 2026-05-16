import logging
from collections import Counter

from app_modules.local_translator import LocalSrToEnTranslator
from app_modules.sentiment_analyzer import SerbianSentimentAnalyzer
from app_modules.summarizer import extractive_summary, abstractive_summary
from app_modules.topic_modeller import get_topics
from app_modules.word_controller import WordController
from app_modules.text_analyzer import TextAnalyzer
from app_modules.graph_maker import Visualizer
from app_modules.hate_speech_detector import analyze_hate_speech
from app_modules.grammar_corrector import correct_sentence as grammar_correct_sentence
from app_modules.absa_analyzer import SerbianABSA, enrich_absa_with_translations
from app_modules.srl_extractor import SerbianSRLExtractor
from app_modules.speech_to_text import VoiceTranscriber
from app_modules.pipeline import get_nlp
from app_modules.model_config import set_config_override

_log = logging.getLogger(__name__)

SAFE_DEFAULT = "/"
MAX_LINK_LOOKUPS = 40
BATCH_TRANSLATE_SIZE = 50

translator         = LocalSrToEnTranslator()
text_analyzer      = TextAnalyzer()
visualizer         = Visualizer()
sentiment_analyzer = SerbianSentimentAnalyzer()
absa_analyzer      = SerbianABSA()
srl_extractor      = SerbianSRLExtractor(pipeline=get_nlp())
transcriber        = VoiceTranscriber()


def _report(progress: dict, job_id: str, pct: int, stage: str):
    info = progress.get(job_id, {})
    info.update({"pct": int(pct), "stage": stage})
    progress[job_id] = info


def _set_section(progress: dict, job_id: str, key: str, data):
    if job_id in progress:
        progress[job_id].setdefault("sections", {})[key] = data


def run_analysis(input_text, features, job_id: str, progress: dict, config_override: dict | None = None):
    if config_override is not None:
        set_config_override(config_override)
    try:
        return _run_analysis_inner(input_text, features, job_id, progress)
    finally:
        set_config_override(None)


def _run_analysis_inner(input_text, features, job_id: str, progress: dict):
    progress[job_id] = {"pct": 0, "stage": "Queued", "status": "running", "sections": {}}

    result = {
        "original_input": input_text,
        "translated_sentence": "",
        "words": [],
        "zipped_data": [],
        "extractive_summary": [],
        "abstractive_summary": None,
        "topics": [],
        "word_cloud_image": None,
        "ner_heatmap_image": None,
        "pos_sunburst_image": None,
        "dependency_tree_img": None,
        "ner_results": [],
        "error_message": None,
        "sentiment": None,
        "sentence_sentiments": [],
        "hate_speech": None,
        "grammar_suggestion": None,
        "grammar_applied": False,
        "selected_features": features,
        "grammar_error": None,
        "absa": None,
        "srl": None,
    }

    try:
        # ── Phase 1: top-of-page sections (all standalone, appear in visual order) ──

        if "grammar" in features:
            _report(progress, job_id, 5, "Grammar correction")
            try:
                result["grammar_suggestion"] = grammar_correct_sentence(input_text)
            except Exception as e:
                _log.exception("grammar correction failed")
                result["grammar_suggestion"] = None
                result["grammar_error"] = str(e)
            _set_section(progress, job_id, "grammar", {
                "grammar_suggestion": result["grammar_suggestion"],
                "grammar_error":      result["grammar_error"],
                "original_input":     input_text,
            })

        _report(progress, job_id, 10, "Tokenizing")
        words = WordController.split_into_words(input_text)
        result["words"] = words
        _set_section(progress, job_id, "input", {"original_input": input_text, "translated_sentence": ""})

        if "translation" in features:
            _report(progress, job_id, 15, "Translating sentence")
            try:
                result["translated_sentence"] = WordController.translate_to_english(input_text, translator)
            except Exception as e:
                _log.exception("translation failed")
                result["translated_sentence"] = SAFE_DEFAULT
                result["translation_error"] = str(e)
            _set_section(progress, job_id, "input", {
                "original_input": input_text,
                "translated_sentence": result["translated_sentence"],
                "translation_error": result.get("translation_error"),
            })

        _report(progress, job_id, 20, "Sentiment analysis")
        try:
            analysis = sentiment_analyzer.analyze(
                input_text,
                mode="sentences",
                aggregation="mean",
                max_length=256,
            )
            result["sentiment"] = {
                "label": analysis.overall.label,
                "confidence": analysis.overall.confidence,
                "scores": analysis.overall.scores,
            }
            sentences = sentiment_analyzer.split_sentences(input_text)
            result["sentence_sentiments"] = [
                {
                    "sentence": s,
                    "label": r.label,
                    "confidence": r.confidence,
                    "scores": r.scores,
                }
                for s, r in zip(sentences, analysis.sentences)
            ]
        except Exception as e:
            _log.exception("sentiment analysis failed")
            result["sentiment"] = None
            result["sentence_sentiments"] = []
            result["sentiment_error"] = str(e)
        _set_section(progress, job_id, "sentiment", {
            "sentiment": result["sentiment"],
            "sentence_sentiments": result["sentence_sentiments"],
            "sentiment_error": result.get("sentiment_error"),
        })

        _report(progress, job_id, 25, "Hate speech detection")
        try:
            raw_overall = analyze_hate_speech(input_text)
            overall_hate = {
                "flagged": raw_overall["flagged"],
                "label": raw_overall["label"],
                "score": raw_overall["score"],
                "scores": raw_overall["scores"],
                "reasons": raw_overall["reasons"],
            }
            sentences = sentiment_analyzer.split_sentences(input_text)
            sentence_hate = []
            for s in sentences:
                raw = analyze_hate_speech(s)
                sentence_hate.append({
                    "sentence": s,
                    "flagged": raw["flagged"],
                    "label": raw["label"],
                    "score": raw["score"],
                })
            result["hate_speech"] = {"overall": overall_hate, "sentences": sentence_hate}
        except Exception as e:
            _log.exception("hate speech detection failed")
            result["hate_speech"] = None
            result["hate_speech_error"] = str(e)
        _set_section(progress, job_id, "hate_speech", {
            "hate_speech": result["hate_speech"],
            "hate_speech_error": result.get("hate_speech_error"),
        })

        _report(progress, job_id, 30, "Aspect-based sentiment analysis")
        try:
            result["absa"] = absa_analyzer.analyze(input_text)
        except Exception as e:
            _log.exception("ABSA failed")
            result["absa"] = None
            result["absa_error"] = str(e)
        _set_section(progress, job_id, "absa", {
            "absa": result["absa"],
            "absa_error": result.get("absa_error"),
        })

        if "summaries" in features:
            _report(progress, job_id, 35, "Summarizing")
            try:
                result["extractive_summary"] = extractive_summary(input_text, num_sentences=2)
            except Exception as e:
                _log.exception("extractive summary failed")
                result["extractive_summary"] = []
                result["extractive_summary_error"] = str(e)
            try:
                result["abstractive_summary"] = abstractive_summary(input_text, translator)
            except Exception as e:
                _log.exception("abstractive summary failed")
                result["abstractive_summary"] = None
                result["abstractive_summary_error"] = str(e)
            _set_section(progress, job_id, "summaries", {
                "extractive_summary": result["extractive_summary"],
                "abstractive_summary": result["abstractive_summary"],
                "extractive_summary_error": result.get("extractive_summary_error"),
                "abstractive_summary_error": result.get("abstractive_summary_error"),
            })

        if "topic" in features:
            _report(progress, job_id, 40, "Topic modelling")
            try:
                result["topics"] = get_topics([input_text], translator)
            except Exception:
                result["topics"] = []
            _set_section(progress, job_id, "topics", {"topics": result["topics"]})

        _report(progress, job_id, 45, "Semantic role labeling")
        try:
            result["srl"] = srl_extractor.analyze(input_text)
        except Exception:
            result["srl"] = None
        _set_section(progress, job_id, "srl", {"srl": result["srl"]})

        # ── Phase 2: per-word pipeline (needed for word table + visuals) ────────────

        _report(progress, job_id, 50, "Preparing words")
        latin_words = WordController.transliterate_cyrillic_to_latin(words)
        lemmas = WordController.lemmatize_words(latin_words)
        transliterated_lemmas = WordController.transliterate_latin_to_cyrillic(lemmas)
        unique_lemmas = list(dict.fromkeys(lemmas))

        _report(progress, job_id, 55, "Reading local dictionaries")
        try:
            definitions_list = WordController.find_local_definitions(transliterated_lemmas)
        except Exception:
            definitions_list = [SAFE_DEFAULT] * len(lemmas)

        _report(progress, job_id, 60, "Tagging morphology")
        try:
            word_types   = WordController.get_word_types(latin_words)
            word_numbers = WordController.get_word_numbers(latin_words)
            word_persons = WordController.get_word_persons(latin_words)
            word_cases   = WordController.get_word_cases(latin_words)
            word_genders = WordController.get_word_genders(latin_words)
        except Exception:
            n = len(latin_words)
            word_types   = [SAFE_DEFAULT] * n
            word_numbers = [SAFE_DEFAULT] * n
            word_persons = [SAFE_DEFAULT] * n
            word_cases   = [SAFE_DEFAULT] * n
            word_genders = [SAFE_DEFAULT] * n

        _report(progress, job_id, 65, "NER & dependency parsing")
        try:
            result["ner_results"] = text_analyzer.analyze_named_entities(input_text)
            dp_results = text_analyzer.analyze_dependency_parsing(input_text)
            word_heads   = [dp.get("head", SAFE_DEFAULT)   for dp in dp_results]
            word_deprels = [dp.get("deprel", SAFE_DEFAULT) for dp in dp_results]
        except Exception:
            word_heads   = [SAFE_DEFAULT] * len(lemmas)
            word_deprels = [SAFE_DEFAULT] * len(lemmas)

        _report(progress, job_id, 70, "Translating lemmas")
        translation_map = {lemma: SAFE_DEFAULT for lemma in unique_lemmas}
        try:
            for start in range(0, len(unique_lemmas), BATCH_TRANSLATE_SIZE):
                chunk = unique_lemmas[start:start + BATCH_TRANSLATE_SIZE]
                translated = translator.translate(chunk, src="sr", dest="en")
                if not isinstance(translated, list):
                    translated = [translated]
                for src_word, t in zip(chunk, translated):
                    translation_map[src_word] = getattr(t, "text", SAFE_DEFAULT) or SAFE_DEFAULT
        except Exception:
            pass

        try:
            if result.get("absa"):
                result["absa"] = enrich_absa_with_translations(
                    result["absa"],
                    translator=translator,
                    translation_map=translation_map,
                    WordController=WordController,
                    safe_default=SAFE_DEFAULT,
                )
                _set_section(progress, job_id, "absa", {"absa": result["absa"]})
        except Exception:
            pass

        _report(progress, job_id, 75, "Finding online links")
        link_map = {lemma: SAFE_DEFAULT for lemma in unique_lemmas}
        try:
            to_check = unique_lemmas[:MAX_LINK_LOOKUPS]
            checked = WordController.process_links_for_lemmas(to_check)
            for lemma, url in zip(to_check, checked):
                link_map[lemma] = url or SAFE_DEFAULT
        except Exception:
            pass

        _report(progress, job_id, 80, "Assembling table")
        per_word_translations = [translation_map.get(lem, SAFE_DEFAULT) for lem in lemmas]
        per_word_links        = [link_map.get(lem, SAFE_DEFAULT)        for lem in lemmas]

        result["zipped_data"] = [
            list(row) for row in zip(
                per_word_translations,
                lemmas,
                definitions_list,
                per_word_links,
                word_types,
                word_numbers,
                word_persons,
                word_cases,
                word_genders,
                word_heads,
                word_deprels,
            )
        ]

        _set_section(progress, job_id, "words_table", {
            "words":       result["words"],
            "zipped_data": result["zipped_data"],
            "ner_results": result["ner_results"],
        })

        # ── Phase 3: bottom-of-page visuals ─────────────────────────────────────────

        if "graphs" in features:
            _report(progress, job_id, 85, "Rendering dependency tree")
            try:
                result["dependency_tree_img"] = text_analyzer.visualize_dependency_tree(input_text)
            except Exception:
                result["dependency_tree_img"] = None
            _set_section(progress, job_id, "dependency_tree", {"dependency_tree_img": result["dependency_tree_img"]})

        if "visuals" in features:
            _report(progress, job_id, 90, "Building visualizations")
            try:
                doc_for_viz = text_analyzer.pipeline(input_text)
                if lemmas:
                    frequencies = Counter(lemmas)
                    result["word_cloud_image"] = visualizer.generate_word_cloud(frequencies)
                result["ner_heatmap_image"]  = visualizer.generate_ner_heatmap(doc_for_viz)
                result["pos_sunburst_image"] = visualizer.generate_pos_sunburst(doc_for_viz)
            except Exception:
                result["word_cloud_image"]   = None
                result["ner_heatmap_image"]  = None
                result["pos_sunburst_image"] = None
            _set_section(progress, job_id, "visuals", {
                "word_cloud_image":   result["word_cloud_image"],
                "ner_heatmap_image":  result["ner_heatmap_image"],
                "pos_sunburst_image": result["pos_sunburst_image"],
            })

        progress[job_id].update({"pct": 100, "stage": "Finished", "status": "finished"})
        return result

    except Exception as e:
        result["error_message"] = str(e)
        progress[job_id].update({
            "status": "failed",
            "stage": "Error",
            "pct": progress[job_id].get("pct", 0),
            "error_message": str(e),
        })
        return result

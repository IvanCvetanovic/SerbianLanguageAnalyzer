from apiflask import Schema
from apiflask.fields import String, List, Boolean, Float, Integer, Nested, Dict

VALID_FEATURES = ['grammar', 'translation', 'summaries', 'topic', 'visuals', 'graphs']


# ── Input schemas ─────────────────────────────────────────────────────────────

class TextIn(Schema):
    text = String(
        required=True,
        metadata={'description': 'Serbian text to analyze (Latin or Cyrillic script).'},
    )


class ModelConfigIn(Schema):
    mode            = String(load_default=None, allow_none=True,
                             metadata={'description': 'local or remote. Defaults to the global setting.'})
    local_model     = String(load_default=None, allow_none=True,
                             metadata={'description': 'Ollama model name, e.g. llama3.1:8b'})
    remote_base_url = String(load_default=None, allow_none=True,
                             metadata={'description': 'vLLM base URL, e.g. http://host:8001/v1'})
    remote_model    = String(load_default=None, allow_none=True,
                             metadata={'description': 'Model name on the remote endpoint.'})
    remote_api_key  = String(load_default=None, allow_none=True,
                             metadata={'description': 'API key (use "not-needed" for open endpoints).'})


class AnalyzeIn(Schema):
    text = String(required=True, metadata={'description': 'Serbian text to analyze.'})
    features = List(
        String(),
        load_default=list(VALID_FEATURES),
        metadata={
            'description': (
                'Analysis modules to enable. '
                f'Any subset of: {VALID_FEATURES}. Defaults to all.'
            ),
        },
    )
    model_config = Nested(
        ModelConfigIn,
        load_default=None,
        allow_none=True,
        metadata={'description': 'Optional per-request model override. Omit to use the global config.'},
    )


class ExplainIn(Schema):
    original  = String(required=True, metadata={'description': 'Original uncorrected sentence.'})
    corrected = String(required=True, metadata={'description': 'Grammar-corrected sentence.'})


# ── Output schemas ────────────────────────────────────────────────────────────

class JobCreatedOut(Schema):
    job_id   = String(metadata={'description': 'Use to poll /api/v1/jobs/{job_id}.'})
    poll_url = String()


class JobStatusOut(Schema):
    status        = String(metadata={'description': 'running | finished | failed'})
    pct           = Integer(metadata={'description': 'Completion 0–100.'})
    stage         = String()
    result        = Dict(allow_none=True)
    error_message = String(allow_none=True)


class SentimentScoresOut(Schema):
    negative = Float()
    neutral  = Float()
    positive = Float()


class SentimentOverallOut(Schema):
    label      = String()
    confidence = Float()
    scores     = Nested(SentimentScoresOut)


class SentenceSentimentOut(Schema):
    sentence   = String()
    label      = String()
    confidence = Float()
    scores     = Nested(SentimentScoresOut)


class SentimentOut(Schema):
    overall             = Nested(SentimentOverallOut)
    sentence_sentiments = List(Nested(SentenceSentimentOut))


class GrammarOut(Schema):
    suggestion     = String(allow_none=True)
    has_suggestion = Boolean()


class ExplainChangeOut(Schema):
    original    = String()
    corrected   = String()
    explanation = String()


class ExplainOut(Schema):
    changes = List(Nested(ExplainChangeOut))


# ── Per-module output schemas ─────────────────────────────────────────────────

class NEREntityOut(Schema):
    text   = String()
    entity = String()

class NEROut(Schema):
    ner = List(Nested(NEREntityOut))


class SRLFrameOut(Schema):
    predicate       = String()
    predicate_lemma = String()
    predicate_index = Integer()
    negated         = Boolean()
    roles           = Dict(keys=String(), values=List(String()))

class SRLSentenceOut(Schema):
    sentence = String()
    frames   = List(Nested(SRLFrameOut))

class SRLOut(Schema):
    srl = List(Nested(SRLSentenceOut))


class ABSAAspectOut(Schema):
    aspect      = String()
    sentiment   = String()
    confidence  = Float()
    evidence    = String()
    aspect_en   = String(allow_none=True)
    evidence_en = String(allow_none=True)

class ABSASentenceOut(Schema):
    sentence = String()
    aspects  = List(Nested(ABSAAspectOut))

class ABSAOut(Schema):
    absa = List(Nested(ABSASentenceOut))


class AbstractiveSummaryOut(Schema):
    text_sr = String(allow_none=True)
    text_en = String(allow_none=True)

class SummarizeOut(Schema):
    extractive  = List(String())
    abstractive = Nested(AbstractiveSummaryOut, allow_none=True)


class TopicsOut(Schema):
    topics = List(String())


class TranscribeQueryIn(Schema):
    analyze  = Boolean(load_default=False,
                       metadata={'description': 'Also run the full analysis pipeline on the transcript.'})
    features = List(String(), load_default=None,
                    metadata={'description': 'Comma-separated feature list for the analysis pipeline.'})

class TranscribeOut(Schema):
    text     = String()
    job_id   = String(allow_none=True)
    poll_url = String(allow_none=True)


class TranslateOut(Schema):
    translation = String()


class HateSpeechScoresOut(Schema):
    race               = Float()
    sexual_orientation = Float()
    gender             = Float()
    physical_appearance = Float()
    religion           = Float()
    disability         = Float()
    appropriate        = Float()

class HateSpeechResultOut(Schema):
    flagged  = Boolean()
    label    = String()
    score    = Float()
    scores   = Nested(HateSpeechScoresOut)
    reasons  = List(String())

class HateSpeechSentenceOut(Schema):
    sentence = String()
    flagged  = Boolean()
    label    = String()
    score    = Float()

class HateSpeechOut(Schema):
    overall   = Nested(HateSpeechResultOut)
    sentences = List(Nested(HateSpeechSentenceOut))


# ── Result formatter (raw pipeline dict → clean API dict) ─────────────────────

def _val(v):
    return v if (v and v != '/') else None


def _fmt_words(words: list, zipped: list) -> list:
    out = []
    for surface, row in zip(words, zipped):
        out.append({
            'surface':        surface,
            'translation_en': _val(row[0]),
            'lemma':          row[1],
            'definition':     _val(str(row[2])),
            'link':           _val(row[3]),
            'pos':            _val(row[4]),
            'number':         _val(row[5]),
            'person':         _val(row[6]),
            'case':           _val(row[7]),
            'gender':         _val(row[8]),
            'head':           _val(row[9]),
            'deprel':         _val(row[10]),
        })
    return out


def format_result(raw: dict) -> dict:
    grammar = None
    if raw.get('grammar_suggestion'):
        grammar = {
            'suggestion': raw['grammar_suggestion'],
            'error':      raw.get('grammar_error'),
        }

    ab = raw.get('abstractive_summary')
    abstractive = None
    if ab:
        if isinstance(ab, (list, tuple)) and len(ab) >= 2:
            abstractive = {'text_sr': ab[0], 'text_en': ab[1]}
        else:
            abstractive = {'text_sr': str(ab), 'text_en': None}

    return {
        'schema_version':      '1.0',
        'language':            'sr',
        'input':               raw.get('original_input', ''),
        'input_translation':   raw.get('translated_sentence') or None,
        'features_run':        raw.get('selected_features', []),
        'words':               _fmt_words(raw.get('words', []), raw.get('zipped_data', [])),
        'sentiment':           raw.get('sentiment'),
        'sentence_sentiments': raw.get('sentence_sentiments', []),
        'hate_speech':         raw.get('hate_speech'),
        'absa':                raw.get('absa'),
        'srl':                 raw.get('srl'),
        'grammar':             grammar,
        'extractive_summary':  raw.get('extractive_summary', []),
        'abstractive_summary': abstractive,
        'topics':              raw.get('topics', []),
        'ner':                 raw.get('ner_results', []),
        'error':               raw.get('error_message'),
    }

from apiflask import Schema
from apiflask.fields import String, List, Boolean, Float, Integer, Nested, Dict

VALID_FEATURES = ['grammar', 'translation', 'summaries', 'topic', 'visuals', 'graphs']


# ── Input schemas ─────────────────────────────────────────────────────────────

class TextIn(Schema):
    text = String(
        required=True,
        metadata={'description': 'Serbian text to analyze (Latin or Cyrillic script).'},
    )


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

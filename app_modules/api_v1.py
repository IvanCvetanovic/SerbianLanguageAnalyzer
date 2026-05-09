import os
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

from apiflask import APIBlueprint, abort
from flask import request, jsonify

from app_modules.job_store import jobs, progress, executor, prune_old_jobs
from app_modules.analysis_pipeline import (
    run_analysis,
    sentiment_analyzer,
    absa_analyzer,
    srl_extractor,
    text_analyzer,
    translator,
    transcriber,
)
from app_modules.summarizer import extractive_summary, abstractive_summary
from app_modules.topic_modeller import get_topics
from app_modules.grammar_corrector import correct_sentence, explain_corrections
from app_modules.api_schema import (
    AnalyzeIn, TextIn, ExplainIn,
    JobCreatedOut, JobStatusOut,
    SentimentOut, GrammarOut, ExplainOut,
    VALID_FEATURES,
    format_result,
)

api_v1 = APIBlueprint('api_v1', __name__, url_prefix='/api/v1')


# ── Full async pipeline ───────────────────────────────────────────────────────

@api_v1.post('/analyze')
@api_v1.input(AnalyzeIn, arg_name='body')
@api_v1.output(JobCreatedOut, status_code=202)
@api_v1.doc(
    summary='Submit text for full analysis',
    description=(
        'Runs the complete NLP pipeline asynchronously. '
        'Returns a job_id immediately — poll `/api/v1/jobs/{job_id}` for progress and results.'
    ),
)
def analyze(body):
    prune_old_jobs()
    job_id = str(uuid.uuid4())
    progress[job_id] = {'pct': 0, 'stage': 'Queued', 'status': 'running'}
    future = executor.submit(run_analysis, body['text'], body['features'], job_id, progress)
    jobs[job_id] = {
        'future': future, 'status': 'running', 'result': None,
        'created_at': datetime.utcnow(), 'error': None,
    }
    return {'job_id': job_id, 'poll_url': f'/api/v1/jobs/{job_id}'}


@api_v1.get('/jobs/<job_id>')
@api_v1.output(JobStatusOut)
@api_v1.doc(
    summary='Poll analysis job',
    description='Returns current progress. When `status=finished`, the `result` field contains the full analysis.',
)
def get_job(job_id):
    if job_id not in progress:
        abort(404, message='Job not found or expired.')
    info = dict(progress[job_id])
    job  = jobs.get(job_id)
    if job and job['future'].done() and job.get('result') is None:
        try:
            job['result'] = job['future'].result()
            job['status'] = 'finished'
            if info.get('status') != 'failed':
                info.update({'status': 'finished', 'pct': 100, 'stage': 'Finished'})
                progress[job_id].update(info)
        except Exception as e:
            info.update({'status': 'failed', 'error_message': str(e)})
            progress[job_id].update(info)
    if info.get('status') == 'finished' and job and job.get('result'):
        info['result'] = format_result(job['result'])
    return info


# ── Per-module endpoints ──────────────────────────────────────────────────────

@api_v1.post('/sentiment')
@api_v1.input(TextIn, arg_name='body')
@api_v1.output(SentimentOut)
@api_v1.doc(
    summary='Sentiment analysis',
    description='Returns overall and per-sentence sentiment with confidence scores.',
)
def api_sentiment(body):
    text     = body['text']
    analysis = sentiment_analyzer.analyze(text, mode='sentences', aggregation='mean', max_length=256)
    sents    = sentiment_analyzer.split_sentences(text)
    return {
        'overall': {
            'label': analysis.overall.label,
            'confidence': analysis.overall.confidence,
            'scores': analysis.overall.scores,
        },
        'sentence_sentiments': [
            {'sentence': s, 'label': r.label, 'confidence': r.confidence, 'scores': r.scores}
            for s, r in zip(sents, analysis.sentences)
        ],
    }


@api_v1.post('/grammar')
@api_v1.input(TextIn, arg_name='body')
@api_v1.output(GrammarOut)
@api_v1.doc(
    summary='Grammar correction',
    description='Returns a corrected sentence. `suggestion` is null when no corrections are needed.',
)
def api_grammar(body):
    suggestion = correct_sentence(body['text'])
    return {'suggestion': suggestion, 'has_suggestion': suggestion is not None}


@api_v1.post('/grammar/explain')
@api_v1.input(ExplainIn, arg_name='body')
@api_v1.output(ExplainOut)
@api_v1.doc(
    summary='Explain grammar corrections',
    description=(
        'Given the original and corrected sentence, returns a brief English explanation '
        'for each changed word. Fires one LLM call.'
    ),
)
def api_grammar_explain(body):
    changes = explain_corrections(body['original'], body['corrected'])
    return {'changes': changes}


@api_v1.post('/srl')
@api_v1.input(TextIn, arg_name='body')
@api_v1.doc(
    summary='Semantic role labeling',
    description='Returns predicate–argument frames for each sentence in the input.',
)
def api_srl(body):
    return {'srl': srl_extractor.analyze(body['text'])}


@api_v1.post('/absa')
@api_v1.input(TextIn, arg_name='body')
@api_v1.doc(
    summary='Aspect-based sentiment analysis',
    description='Returns aspect/sentiment/confidence triples for each sentence.',
)
def api_absa(body):
    return {'absa': absa_analyzer.analyze(body['text'])}


@api_v1.post('/ner')
@api_v1.input(TextIn, arg_name='body')
@api_v1.doc(
    summary='Named entity recognition',
    description='Returns a list of named entity spans with their labels (PERSON, ORG, LOC, …).',
)
def api_ner(body):
    return {'ner': text_analyzer.analyze_named_entities(body['text'])}


@api_v1.post('/summarize')
@api_v1.input(TextIn, arg_name='body')
@api_v1.doc(
    summary='Text summarization',
    description=(
        'Returns an extractive summary (selected sentences) and an abstractive summary '
        '(LLM-generated) with an English translation.'
    ),
)
def api_summarize(body):
    text    = body['text']
    ex      = extractive_summary(text, num_sentences=2)
    ab      = abstractive_summary(text, translator)
    return {
        'extractive': ex,
        'abstractive': {
            'text_sr': ab[0] if ab else None,
            'text_en': ab[1] if ab else None,
        },
    }


@api_v1.post('/topics')
@api_v1.input(TextIn, arg_name='body')
@api_v1.doc(
    summary='Topic modelling',
    description='Returns the top keyword topics extracted from the text.',
)
def api_topics(body):
    topics = get_topics([body['text']], translator)
    return {'topics': topics[0] if topics else []}


@api_v1.post('/transcribe')
@api_v1.doc(
    summary='Transcribe audio',
    description=(
        'Accepts a WAV/WebM/MP3 audio file via `multipart/form-data` (field name: `audio`). '
        'Returns Whisper transcription. '
        'Add `?analyze=true` to also submit the transcript to the full analysis pipeline '
        '(returns a `job_id` in addition to `text`).'
    ),
)
def api_transcribe():
    if 'audio' not in request.files:
        abort(400, message='Missing audio. Send as multipart/form-data with field name "audio".')
    f = request.files['audio']
    if not f.filename:
        abort(400, message='Empty filename.')

    suffix = Path(f.filename).suffix or '.webm'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.save(tmp.name)
        tmp.close()
        text = transcriber.transcribe_audio_file(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if request.args.get('analyze', '').lower() in ('1', 'true', 'yes'):
        features_param = request.args.get('features', '')
        features = [x.strip() for x in features_param.split(',') if x.strip()] or list(VALID_FEATURES)
        prune_old_jobs()
        job_id = str(uuid.uuid4())
        progress[job_id] = {'pct': 0, 'stage': 'Queued', 'status': 'running'}
        future = executor.submit(run_analysis, text, features, job_id, progress)
        jobs[job_id] = {
            'future': future, 'status': 'running', 'result': None,
            'created_at': datetime.utcnow(), 'error': None,
        }
        return jsonify({'text': text, 'job_id': job_id, 'poll_url': f'/api/v1/jobs/{job_id}'})

    return jsonify({'text': text})

// ─── DOM refs ───────────────────────────────────────────────────────────────
const textInput      = document.querySelector('input[name="input"]');
const mainForm       = document.querySelector('form[action="/"][method="post"]:not([enctype])');
const loaderOverlay  = document.getElementById('loader-overlay');
const stageEl        = document.getElementById('loader-stage');
const barEl          = document.getElementById('progress-bar');
const finishedNote   = document.getElementById('finished-note');
const failedNote     = document.getElementById('failed-note');
const resultsContainer = document.getElementById('results-container');
const slimProgress   = document.getElementById('slim-progress');
const slimBar        = document.getElementById('slim-bar');
const slimStage      = document.getElementById('slim-stage');
const jobErrorBanner = document.getElementById('job-error-banner');

// ─── Utilities ──────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function sectionSpinner(label) {
  return `<div class="section-loading"><div class="section-spinner"></div><span>${escHtml(label)}</span></div>`;
}

function showEl(el) { if (el) el.style.display = ''; }
function hideEl(el) { if (el) el.style.display = 'none'; }

/** Populate a section div: set its innerHTML, make it visible, show results container */
function showSection(id, html) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = html;
  el.style.display = '';
  resultsContainer.style.display = 'flex';  // reveal results area on first section
}

// ─── Random sentence & mic ───────────────────────────────────────────────────
function fetchRandomSentence() {
  fetch('/get_random_sentence')
    .then(r => r.json())
    .then(data => { textInput.value = data.sentence; mainForm.submit(); })
    .catch(err => console.error('Error fetching random sentence:', err));
}

const micBtn = document.getElementById('mic-btn');
let mediaRecorder, audioChunks = [], isRecording = false;
micBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => sendAudioToServer(new Blob(audioChunks, { type: 'audio/webm' }));
      mediaRecorder.start();
      isRecording = true;
      micBtn.textContent = '🛑';
      micBtn.style.backgroundColor = '#dc3545';
    } catch (err) {
      console.error('Mic error:', err);
      alert('Could not access the microphone. Please check browser permissions.');
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    micBtn.textContent = '🎤';
    micBtn.style.backgroundColor = '#58b0fd';
  }
}

function sendAudioToServer(audioBlob) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  textInput.placeholder = 'Transcribing, please wait...';
  micBtn.disabled = true;
  fetch('/analyze_voice', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.text) { textInput.value = data.text; mainForm.submit(); }
      else { alert('Failed to transcribe audio: ' + data.error); }
    })
    .catch(err => console.error('Send audio error:', err))
    .finally(() => { micBtn.disabled = false; textInput.placeholder = 'Enter text...'; audioChunks = []; });
}

// ─── Overlay ─────────────────────────────────────────────────────────────────
// Overlay intentionally not shown on submit — results appear progressively via section spinners.

function showOverlay() {
  hideEl(finishedNote);
  hideEl(failedNote);
  stageEl.textContent = 'Queued…';
  barEl.style.width = '0%';
  loaderOverlay.style.display = 'flex';
}

// ─── Section renderers ────────────────────────────────────────────────────────

function renderInput(data) {
  let html = `<div class="summary card-blue" style="margin:0; height:100%; box-sizing:border-box;"><h2>Original Input:</h2><p>${escHtml(data.original_input)}</p>`;
  if (data.translated_sentence && data.translated_sentence !== '/') {
    html += `<h2 style="margin-top:15px;">Translated Input:</h2><p>${escHtml(data.translated_sentence)}</p>`;
  }
  html += '</div>';
  showSection('section-input', html);
}

function renderGrammar(data) {
  if (!data.grammar_suggestion) {
    hideEl(document.getElementById('section-grammar'));
    return;
  }
  const html = `<div class="summary card-teal" style="margin:0; height:100%; box-sizing:border-box;">
    <h2>Grammar suggestion</h2>
    <p><strong>Suggested:</strong></p>
    <p id="grammar-suggested">${escHtml(data.grammar_suggestion)}</p>
    <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;">
      <button type="button" class="text-submit" id="btn-use-corrected">Use suggested version</button>
    </div>
  </div>`;
  showSection('section-grammar', html);
  document.getElementById('btn-use-corrected').addEventListener('click', () => {
    const inp = document.querySelector('input[name="input"]');
    if (inp) { inp.value = data.grammar_suggestion.trim(); inp.focus(); }
  });
}

function renderSentiment(data) {
  if (!data.sentiment && !data.sentence_sentiments?.length) return;
  let html = '<div class="summary card-green" style="margin:0; height:100%; box-sizing:border-box;"><h2>Sentiment</h2>';
  if (data.sentiment) {
    html += `<p><strong>Overall:</strong> ${escHtml(data.sentiment.label)}
             (confidence: ${data.sentiment.confidence.toFixed(3)})</p>
             <ul>
               <li>Negative: ${data.sentiment.scores.negative.toFixed(3)}</li>
               <li>Neutral:  ${data.sentiment.scores.neutral.toFixed(3)}</li>
               <li>Positive: ${data.sentiment.scores.positive.toFixed(3)}</li>
             </ul>`;
  }
  if (data.sentence_sentiments?.length) {
    html += '<h3 style="margin-top:15px;">Sentence-level</h3><ul>';
    for (const item of data.sentence_sentiments) {
      html += `<li style="margin-bottom:12px;">
        <div><strong>Sentence:</strong> ${escHtml(item.sentence)}</div>
        <div><strong>${escHtml(item.label)}</strong> (${item.confidence.toFixed(3)})
        — neg ${item.scores.negative.toFixed(3)},
          neu ${item.scores.neutral.toFixed(3)},
          pos ${item.scores.positive.toFixed(3)}</div>
      </li>`;
    }
    html += '</ul>';
  }
  html += '</div>';
  showSection('section-sentiment', html);
}

function renderHateSpeech(data) {
  if (!data.hate_speech) return;
  const hs = data.hate_speech;
  let html = '<div class="summary card-red" style="margin:0; height:100%; box-sizing:border-box;"><h2>Hate / Abuse Detection</h2>';
  if (hs.overall) {
    html += `<p><strong>Status:</strong> ${hs.overall.flagged
      ? '<span style="color:#b00020;">🚩 Abusive / Problematic</span>'
      : '<span style="color:#2e7d32;">✅ Not Abusive</span>'}</p>
      <p><strong>Top label:</strong> ${escHtml(hs.overall.label)} (${hs.overall.score.toFixed(3)})</p><ul>`;
    for (const [lbl, score] of Object.entries(hs.overall.scores)) {
      html += `<li>${escHtml(lbl)}: ${score.toFixed(3)}</li>`;
    }
    html += '</ul>';
  }
  if (hs.sentences?.length) {
    html += '<h3 style="margin-top:15px;">Sentence-level</h3><ul>';
    for (const item of hs.sentences) {
      html += `<li style="margin-bottom:12px;">
        <div><strong>Sentence:</strong> ${escHtml(item.sentence)}</div>
        <div><strong>${escHtml(item.label)}</strong> (${item.score.toFixed(3)})
        ${item.flagged
          ? '<span style="color:#b00020;">🚩</span>'
          : '<span style="color:#2e7d32;">✅</span>'}</div>
      </li>`;
    }
    html += '</ul>';
  }
  html += '</div>';
  showSection('section-hate-speech', html);
}

function renderAbsa(data) {
  if (!data.absa?.length) return;
  let html = '<div class="summary card-purple" style="width:100%; margin:0;"><h2>Aspect-Based Sentiment Analysis</h2>';
  for (const sent of data.absa) {
    html += `<div style="margin-bottom:20px;"><p><strong>Sentence:</strong> ${escHtml(sent.sentence)}</p>`;
    if (sent.aspects?.length) {
      html += `<table style="margin-top:10px;"><thead><tr>
        <th>Aspect</th><th>Sentiment</th><th>Confidence</th><th>Evidence</th>
      </tr></thead><tbody>`;
      for (const a of sent.aspects) {
        const c = a.sentiment === 'positive' ? '#2e7d32' : a.sentiment === 'negative' ? '#b00020' : '#616161';
        html += `<tr>
          <td>${escHtml(a.aspect)}${a.aspect_en ? `<div style="color:#666;font-size:.9em;">${escHtml(a.aspect_en)}</div>` : ''}</td>
          <td><span style="color:${c}">${escHtml(a.sentiment)}</span></td>
          <td>${a.confidence.toFixed(2)}</td>
          <td>${escHtml(a.evidence)}${a.evidence_en ? `<div style="color:#666;font-size:.9em;">${escHtml(a.evidence_en)}</div>` : ''}</td>
        </tr>`;
      }
      html += '</tbody></table>';
    } else {
      html += '<p style="color:#777;">No aspects detected.</p>';
    }
    html += '</div>';
  }
  html += '</div>';
  showSection('section-absa', html);
}

function renderSrl(data) {
  if (!data.srl?.length) return;
  let html = '<div class="summary card-blue" style="width:100%; margin:0;"><h2>Semantic Role Labeling (SRL)</h2>';
  for (const item of data.srl) {
    html += `<div style="margin-bottom:18px;padding-top:10px;border-top:1px solid #e0e0e0;">
      <p style="margin:0 0 8px 0;"><strong>Sentence:</strong> ${escHtml(item.sentence)}</p>`;
    if (item.frames?.length) {
      for (const fr of item.frames) {
        html += `<div style="margin:10px 0 14px 0;padding:10px;background:#fafafa;border:1px solid #eee;border-radius:8px;">
          <div style="margin-bottom:8px;"><strong>Predicate:</strong> ${escHtml(fr.predicate)}
          <span style="color:#666;">(lemma: ${escHtml(fr.predicate_lemma)}, idx: ${fr.predicate_index},
          negated: ${fr.negated ? 'true' : 'false'})</span></div>`;
        if (fr.roles && Object.keys(fr.roles).length) {
          html += '<table style="margin:0;"><thead><tr><th style="width:220px;">Role</th><th>Spans</th></tr></thead><tbody>';
          for (const [role, spans] of Object.entries(fr.roles)) {
            html += `<tr><td><strong>${escHtml(role)}</strong></td>
              <td>${spans?.length ? spans.map(escHtml).join(', ') : '/'}</td></tr>`;
          }
          html += '</tbody></table>';
        } else {
          html += '<p style="margin:0;color:#777;">No roles found for this predicate.</p>';
        }
        html += '</div>';
      }
    } else {
      html += '<p style="color:#777;margin:0;">No SRL frames found.</p>';
    }
    html += '</div>';
  }
  html += '</div>';
  showSection('section-srl', html);
}

function renderSummaries(data) {
  if (!data.extractive_summary?.length && !data.abstractive_summary) return;
  let html = '<div style="display:flex;gap:20px;flex-wrap:wrap;width:100%;">';
  if (data.extractive_summary?.length) {
    html += '<div style="flex:1;min-width:320px;"><div class="summary card-orange" style="margin:0; height:100%; box-sizing:border-box;"><h2>Extractive Summary</h2><ul>';
    for (const s of data.extractive_summary) html += `<li>${escHtml(s)}</li>`;
    html += '</ul></div></div>';
  }
  if (data.abstractive_summary) {
    html += `<div style="flex:1;min-width:320px;"><div class="summary card-orange" style="margin:0; height:100%; box-sizing:border-box;">
      <h2>Abstractive Summary</h2>
      <p><strong>Original:</strong> ${escHtml(data.abstractive_summary[0])}</p>
      <p><strong>Translated:</strong> ${escHtml(data.abstractive_summary[1])}</p>
    </div></div>`;
  }
  html += '</div>';
  showSection('section-summaries', html);
}

function renderTopics(data) {
  if (!data.topics?.length) return;
  let html = '<div class="summary card-teal" style="margin:0; width:100%; box-sizing:border-box;"><h2>Topic Modelling</h2><ul>';
  for (const term of data.topics[0]) html += `<li>${escHtml(term)}</li>`;
  html += '</ul></div>';
  showSection('section-topics', html);
}

function renderWordsTable(data) {
  if (!data.words?.length) return;
  const nerMap = {};
  for (const n of (data.ner_results || [])) nerMap[n.text] = n.entity;
  let html = `<table style="font-size:.88em;"><thead><tr>
    <th>Original Word</th><th>Translation</th><th>Lemma (Base Form)</th>
    <th>Local Definition</th><th>Online Definition</th><th>Word Type</th>
    <th>Number</th><th>Person</th><th>Case</th><th>Gender</th>
    <th>Head</th><th>Dependency Relation</th><th>Named Entity</th>
  </tr></thead><tbody>`;
  for (let i = 0; i < data.words.length; i++) {
    const w = data.words[i];
    const z = data.zipped_data[i];
    const link = z[3] && z[3] !== '/'
      ? `<a href="${escHtml(z[3])}" target="_blank">${escHtml(z[3])}</a>`
      : '/';
    html += `<tr>
      <td>${escHtml(w)}</td>
      <td>${escHtml(z[0])}</td>
      <td>${escHtml(z[1])}</td>
      <td>${escHtml(String(z[2]))}</td>
      <td>${link}</td>
      <td>${escHtml(z[4])}</td>
      <td>${escHtml(z[5])}</td>
      <td>${escHtml(z[6])}</td>
      <td>${escHtml(z[7])}</td>
      <td>${escHtml(z[8])}</td>
      <td>${escHtml(z[9])}</td>
      <td>${escHtml(z[10])}</td>
      <td>${escHtml(nerMap[w] || 'O')}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  showSection('section-words-table', html);
}

function renderVisuals(data) {
  const items = [];
  if (data.word_cloud_image)
    items.push(`<div class="chart-container" style="flex:1; min-width:300px; width:auto; max-width:none; margin:0;"><h2>Word Cloud</h2><img src="data:image/png;base64, ${data.word_cloud_image}" alt="Word Cloud"></div>`);
  if (data.ner_heatmap_image)
    items.push(`<div class="chart-container" style="flex:1; min-width:300px; width:auto; max-width:none; margin:0;"><h2>NER Heatmap</h2><img src="data:image/png;base64, ${data.ner_heatmap_image}" alt="NER Heatmap"></div>`);
  if (data.pos_sunburst_image)
    items.push(`<div class="chart-container" style="flex:1; min-width:300px; width:auto; max-width:none; margin:0;"><h2>POS Sunburst Chart</h2><img src="data:image/png;base64, ${data.pos_sunburst_image}" alt="POS Sunburst Chart"></div>`);
  if (items.length) {
    const html = `<div style="display:flex; flex-wrap:wrap; gap:20px; width:100%;">${items.join('')}</div>`;
    showSection('section-visuals', html);
  }
}

function renderDependencyTree(data) {
  if (!data.dependency_tree_img) return;
  const el = document.getElementById('section-dependency-tree');
  if (!el) return;
  // Serve the pyvis HTML from a real Flask endpoint so the browser fully lays out
  // the iframe before vis.js runs — this ensures container.offsetWidth is correct.
  const iframe = document.createElement('iframe');
  iframe.src = `/dependency_tree/${window.currentJobId}`;
  iframe.style.cssText = 'width:100%; height:540px; border:none; display:block;';
  const container = document.createElement('div');
  container.className = 'chart-container';
  container.style.cssText = 'width:100%; max-width:none; box-sizing:border-box;';
  container.innerHTML = '<h2>Dependency Tree</h2>';
  container.appendChild(iframe);
  el.innerHTML = '';
  el.appendChild(container);
  el.style.display = '';
  resultsContainer.style.display = 'flex';
}

// ─── Section dispatch ─────────────────────────────────────────────────────────
const RENDERERS = {
  input:           renderInput,
  grammar:         renderGrammar,
  sentiment:       renderSentiment,
  hate_speech:     renderHateSpeech,
  absa:            renderAbsa,
  srl:             renderSrl,
  summaries:       renderSummaries,
  topics:          renderTopics,
  words_table:     renderWordsTable,
  visuals:         renderVisuals,
  dependency_tree: renderDependencyTree,
};

function applySection(key, data) {
  const fn = RENDERERS[key];
  if (fn) fn(data);
}

// ─── Spinner initialisation ───────────────────────────────────────────────────
const SECTION_LABELS = {
  input:           'Loading input…',
  grammar:         'Checking grammar…',
  sentiment:       'Analyzing sentiment…',
  hate_speech:     'Detecting hate speech…',
  absa:            'Analyzing aspects…',
  srl:             'Extracting semantic roles…',
  summaries:       'Summarizing…',
  topics:          'Modelling topics…',
  words_table:     'Building word table…',
  visuals:         'Generating visualizations…',
  dependency_tree: 'Rendering dependency tree…',
};

// Features that always produce a section vs. those gated on a checkbox
const ALWAYS_SECTIONS  = ['input', 'sentiment', 'hate_speech', 'absa', 'srl', 'words_table'];
const FEATURE_SECTIONS = { grammar: 'grammar', summaries: 'summaries', topic: 'topics', visuals: 'visuals', graphs: 'dependency_tree' };

function sectionElId(key) {
  // section keys use underscores; HTML IDs use 'section-' + dashes
  return 'section-' + key.replace(/_/g, '-');
}

function initSectionSpinners() {
  const features = window.selectedFeatures || [];
  const pending = new Set(ALWAYS_SECTIONS);
  for (const [feat, sec] of Object.entries(FEATURE_SECTIONS)) {
    if (features.includes(feat)) pending.add(sec);
  }
  for (const key of pending) {
    const el = document.getElementById(sectionElId(key));
    if (!el) continue;
    el.innerHTML = sectionSpinner(SECTION_LABELS[key] || 'Loading…');
    el.style.display = '';
  }
  // Show results container so spinners are visible while overlay is up
  resultsContainer.style.display = 'flex';
}

// ─── Poll loop ────────────────────────────────────────────────────────────────
if (window.currentJobId) {
  initSectionSpinners();
  // Show the slim progress bar immediately — no full-screen overlay during analysis
  if (slimProgress) slimProgress.style.display = 'flex';
  resultsContainer.style.display = 'flex';

  let done = false;
  const lastSections = {};

  const poll = setInterval(() => {
    fetch(`/progress/${window.currentJobId}`)
      .then(r => r.json())
      .then(info => {
        if (info.error) throw new Error(info.error);

        const pct = Math.max(0, Math.min(100, info.pct || 0));

        // Keep slim bar in sync
        if (slimBar)   slimBar.style.width   = pct + '%';
        if (slimStage) slimStage.textContent = info.stage || 'Working…';

        // Apply any new or updated sections
        if (info.sections) {
          for (const [key, data] of Object.entries(info.sections)) {
            const sig = JSON.stringify(data);
            if (sig !== JSON.stringify(lastSections[key])) {
              lastSections[key] = data;
              applySection(key, data);
            }
          }
        }

        if ((info.status === 'finished' || info.status === 'failed') && !done) {
          done = true;
          clearInterval(poll);
          hideEl(slimProgress);

          if (info.status === 'failed') {
            const msg = info.error_message || 'Analysis failed. Please try again.';
            if (jobErrorBanner) {
              jobErrorBanner.textContent = msg;
              jobErrorBanner.style.display = '';
            }
          }
        }
      })
      .catch(err => {
        console.error('Progress poll error:', err);
        clearInterval(poll);
        hideEl(slimProgress);
        if (jobErrorBanner) {
          jobErrorBanner.textContent = 'Lost connection to server. Please try again.';
          jobErrorBanner.style.display = '';
        }
      });
  }, 800);
}

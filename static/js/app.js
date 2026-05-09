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
      ? '<span class="text-danger">🚩 Abusive / Problematic</span>'
      : '<span class="text-success">✅ Not Abusive</span>'}</p>
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
          ? '<span class="text-danger">🚩</span>'
          : '<span class="text-success">✅</span>'}</div>
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
        html += `<tr>
          <td>${escHtml(a.aspect)}${a.aspect_en ? `<div class="secondary-label">${escHtml(a.aspect_en)}</div>` : ''}</td>
          <td><span class="sentiment-${a.sentiment}">${escHtml(a.sentiment)}</span></td>
          <td>${a.confidence.toFixed(2)}</td>
          <td>${escHtml(a.evidence)}${a.evidence_en ? `<div class="secondary-label">${escHtml(a.evidence_en)}</div>` : ''}</td>
        </tr>`;
      }
      html += '</tbody></table>';
    } else {
      html += '<p class="text-muted">No aspects detected.</p>';
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
    html += `<div class="srl-sentence">
      <p style="margin:0 0 8px 0;"><strong>Sentence:</strong> ${escHtml(item.sentence)}</p>`;
    if (item.frames?.length) {
      for (const fr of item.frames) {
        html += `<div class="srl-frame">
          <div style="margin-bottom:8px;"><strong>Predicate:</strong> ${escHtml(fr.predicate)}
          <span class="text-muted">(lemma: ${escHtml(fr.predicate_lemma)}, idx: ${fr.predicate_index},
          negated: ${fr.negated ? 'true' : 'false'})</span></div>`;
        if (fr.roles && Object.keys(fr.roles).length) {
          html += '<table style="margin:0;"><thead><tr><th style="width:220px;">Role</th><th>Spans</th></tr></thead><tbody>';
          for (const [role, spans] of Object.entries(fr.roles)) {
            html += `<tr><td><strong>${escHtml(role)}</strong></td>
              <td>${spans?.length ? spans.map(escHtml).join(', ') : '/'}</td></tr>`;
          }
          html += '</tbody></table>';
        } else {
          html += '<p class="text-muted" style="margin:0;">No roles found for this predicate.</p>';
        }
        html += '</div>';
      }
    } else {
      html += '<p class="text-muted" style="margin:0;">No SRL frames found.</p>';
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

// ─── Settings modal ───────────────────────────────────────────────────────────
const settingsBtn     = document.getElementById('settings-btn');
const settingsOverlay = document.getElementById('settings-overlay');
const settingsClose   = document.getElementById('settings-close');
const settingsSave    = document.getElementById('settings-save');
const settingsTest    = document.getElementById('settings-test');
const settingsStatus  = document.getElementById('settings-status');
const localFields         = document.getElementById('local-fields');
const localModelSelect    = document.getElementById('local-model');
const localModelRefresh   = document.getElementById('local-model-refresh');
const remoteFields        = document.getElementById('remote-fields');
const modeLocal           = document.getElementById('mode-local');
const modeRemote          = document.getElementById('mode-remote');
const remoteUrl           = document.getElementById('remote-url');
const remoteModel         = document.getElementById('remote-model');
const remoteKey           = document.getElementById('remote-key');

function showSettingsStatus(msg, ok) {
  settingsStatus.textContent = msg;
  settingsStatus.style.background = ok ? '#e8f5e9' : '#ffebee';
  settingsStatus.style.color      = ok ? '#2e7d32' : '#b00020';
  settingsStatus.style.border     = ok ? '1px solid #a5d6a7' : '1px solid #ef9a9a';
  settingsStatus.style.display    = 'block';
}

function syncModeFields() {
  const isRemote = modeRemote.checked;
  remoteFields.style.display = isRemote ? '' : 'none';
  localFields.style.display  = isRemote ? 'none' : '';
}

modeLocal.addEventListener('change', syncModeFields);
modeRemote.addEventListener('change', syncModeFields);

function fetchOllamaModels(selectValue) {
  fetch('http://localhost:11434/api/tags')
    .then(r => r.json())
    .then(data => {
      const models = (data.models || []).map(m => m.name).filter(Boolean);
      if (!models.length) return;
      localModelSelect.innerHTML = '';
      models.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === selectValue) opt.selected = true;
        localModelSelect.appendChild(opt);
      });
      if (selectValue && !models.includes(selectValue)) {
        const opt = document.createElement('option');
        opt.value = selectValue;
        opt.textContent = selectValue;
        opt.selected = true;
        localModelSelect.insertBefore(opt, localModelSelect.firstChild);
      }
    })
    .catch(() => {});
}

localModelRefresh.addEventListener('click', () => fetchOllamaModels(localModelSelect.value));

function openSettings() {
  settingsStatus.style.display = 'none';
  fetch('/api/settings')
    .then(r => r.json())
    .then(cfg => {
      (cfg.mode === 'remote' ? modeRemote : modeLocal).checked = true;
      syncModeFields();
      const savedLocalModel = cfg.local?.model || 'llama3.1:8b';
      localModelSelect.value = savedLocalModel;
      fetchOllamaModels(savedLocalModel);
      if (cfg.remote) {
        remoteUrl.value   = cfg.remote.base_url || '';
        remoteModel.value = cfg.remote.model    || '';
        remoteKey.value   = cfg.remote.api_key  || '';
      }
    })
    .catch(() => { modeLocal.checked = true; syncModeFields(); })
    .finally(() => { settingsOverlay.style.display = 'flex'; });
}

function closeSettings() {
  settingsOverlay.style.display = 'none';
}

settingsBtn.addEventListener('click', openSettings);
settingsClose.addEventListener('click', closeSettings);
settingsOverlay.addEventListener('click', e => { if (e.target === settingsOverlay) closeSettings(); });

settingsSave.addEventListener('click', () => {
  const payload = {
    mode: modeRemote.checked ? 'remote' : 'local',
    local: {
      model: localModelSelect.value.trim(),
    },
    remote: {
      base_url: remoteUrl.value.trim(),
      model:    remoteModel.value.trim(),
      api_key:  remoteKey.value.trim(),
    },
  };
  settingsSave.disabled = true;
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
    .then(r => r.json())
    .then(d => showSettingsStatus(d.ok ? 'Settings saved.' : 'Error: ' + (d.error || '?'), d.ok))
    .catch(() => showSettingsStatus('Failed to save settings.', false))
    .finally(() => { settingsSave.disabled = false; });
});

// ─── Theme toggle ─────────────────────────────────────────────────────────────
const themeToggle = document.getElementById('theme-toggle');

function applyTheme(dark) {
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  themeToggle.textContent = dark ? '☀️ Light' : '🌙 Dark';
}

applyTheme(document.documentElement.getAttribute('data-theme') === 'dark');

themeToggle.addEventListener('click', () => {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
  applyTheme(!isDark);
});

settingsTest.addEventListener('click', () => {
  const payload = {
    mode:     modeRemote.checked ? 'remote' : 'local',
    base_url: remoteUrl.value.trim(),
    api_key:  remoteKey.value.trim(),
  };
  settingsTest.disabled = true;
  settingsTest.textContent = 'Testing…';
  fetch('/api/test-connection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
    .then(r => r.json())
    .then(d => showSettingsStatus(d.message, d.ok))
    .catch(() => showSettingsStatus('Connection test failed.', false))
    .finally(() => { settingsTest.disabled = false; settingsTest.textContent = 'Test connection'; });
});

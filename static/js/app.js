const textInput     = document.querySelector('input[name="input"]');
const mainForm      = document.querySelector('form[action="/"][method="post"]:not([enctype])');
const loaderOverlay = document.getElementById('loader-overlay');
const stageEl       = document.getElementById('loader-stage');
const barEl         = document.getElementById('progress-bar');
const finishedNote  = document.getElementById('finished-note');
const failedNote    = document.getElementById('failed-note');

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
      else { console.error('Server error:', data.error); alert('Failed to transcribe audio: ' + data.error); }
    })
    .catch(err => console.error('Send audio error:', err))
    .finally(() => {
      micBtn.disabled = false;
      textInput.placeholder = 'Enter text...';
      audioChunks = [];
    });
}

const allForms = document.querySelectorAll('form');
allForms.forEach(form => form.addEventListener('submit', () => { showOverlay(); }));

function showOverlay() {
  finishedNote.style.display = 'none';
  failedNote.style.display = 'none';
  stageEl.textContent = 'Queued…';
  barEl.style.width = '0%';
  loaderOverlay.style.display = 'flex';
}

if (window.currentJobId) {
  showOverlay();
  let done = false;
  const poll = setInterval(() => {
    fetch(`/progress/${window.currentJobId}`)
      .then(r => r.json())
      .then(info => {
        if (info.error) throw new Error(info.error);
        const pct = Math.max(0, Math.min(100, info.pct || 0));
        barEl.style.width = pct + '%';
        stageEl.textContent = info.stage ? `${info.stage}…` : 'Working…';

        if (info.status === 'finished' && !done) {
          done = true;
          finishedNote.style.display = 'block';
          stageEl.textContent = 'Finished';
          barEl.style.width = '100%';
          clearInterval(poll);
          setTimeout(() => {
            window.location.href = `/results/${window.currentJobId}`;
          }, 1000);
        } else if (info.status === 'failed' && !done) {
          done = true;
          failedNote.style.display = 'block';
          stageEl.textContent = 'Error';
          clearInterval(poll);
        }
      })
      .catch(err => {
        console.error('Progress error:', err);
        clearInterval(poll);
        failedNote.style.display = 'block';
        stageEl.textContent = 'Error';
      });
  }, 800);
}

document.addEventListener("DOMContentLoaded", () => {
  const useBtn = document.getElementById("btn-use-corrected");
  const keepBtn = document.getElementById("btn-keep-original");
  const suggestedEl = document.getElementById("grammar-suggested");
  const inputEl = document.querySelector('input[name="input"]');

  if (useBtn && suggestedEl && inputEl) {
    useBtn.addEventListener("click", () => {
      inputEl.value = suggestedEl.textContent.trim();
      inputEl.focus();
    });
  }

  if (keepBtn && inputEl) {
    keepBtn.addEventListener("click", () => {
      inputEl.focus();
    });
  }
});

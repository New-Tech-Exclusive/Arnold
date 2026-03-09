/* ── State ──────────────────────────────────────────────────────────────── */
let abortController = null;
let isGenerating = false;

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('userInput');
const sendBtn     = document.getElementById('sendBtn');
const stopBtn     = document.getElementById('stopBtn');
const newChatBtn  = document.getElementById('newChatBtn');
const modelInfoEl = document.getElementById('modelInfo');

// Sidebar status
const stageVal   = document.getElementById('stageVal');
const ageVal     = document.getElementById('ageVal');
const paramsVal  = document.getElementById('paramsVal');
const sessionVal = document.getElementById('sessionVal');
const statusLink = document.getElementById('statusLink');

/* ── Slider wiring ──────────────────────────────────────────────────────── */
['temperature', 'top_k', 'top_p', 'max_tokens', 'rep_penalty'].forEach(id => {
  const slider = document.getElementById(id);
  const output = document.getElementById(`${id}-val`);
  slider.addEventListener('input', () => { output.value = slider.value; });
});

function getSamplingParams() {
  return {
    temperature:       parseFloat(document.getElementById('temperature').value),
    top_k:             parseInt(document.getElementById('top_k').value),
    top_p:             parseFloat(document.getElementById('top_p').value),
    max_tokens:        parseInt(document.getElementById('max_tokens').value),
    repetition_penalty:parseFloat(document.getElementById('rep_penalty').value),
  };
}

/* ── Model info ─────────────────────────────────────────────────────────── */
async function loadModelInfo() {
  try {
    const res = await fetch('/api/info');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const info = await res.json();

    const params = info.parameters >= 1e6
      ? `${(info.parameters / 1e6).toFixed(1)}M`
      : `${(info.parameters / 1e3).toFixed(0)}K`;

    modelInfoEl.textContent =
      `Arnold · ${info.stage} · age=${info.age} · ${params} params · ${info.device}`;

    stageVal.textContent   = info.stage;
    ageVal.textContent     = info.age;
    paramsVal.textContent  = params;
    sessionVal.textContent = info.session_active ? 'active' : 'idle';

    statusLink.href = '/api/status';
  } catch (e) {
    modelInfoEl.textContent = 'Arnold (failed to load info)';
  }
}

/* ── Message rendering ──────────────────────────────────────────────────── */
function removeWelcome() {
  const w = messagesEl.querySelector('.welcome');
  if (w) w.remove();
}

function appendUserBubble(text) {
  removeWelcome();
  const turn = document.createElement('div');
  turn.className = 'turn turn-user';
  turn.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(turn);
  scrollToBottom();
  return turn;
}

function appendAssistantBubble() {
  const turn = document.createElement('div');
  turn.className = 'turn turn-assistant';
  turn.innerHTML = `
    <div class="avatar">A</div>
    <div>
      <div class="bubble cursor" id="streamBubble"></div>
      <div class="meta" id="streamMeta"></div>
    </div>`;
  messagesEl.appendChild(turn);
  scrollToBottom();
  return {
    bubble: document.getElementById('streamBubble'),
    meta:   document.getElementById('streamMeta'),
  };
}

function appendErrorBubble(msg) {
  const div = document.createElement('div');
  div.className = 'turn';
  div.innerHTML = `<div class="error-msg">⚠ ${escapeHtml(msg)}</div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function escapeHtml(s) {
  return s
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ── Send ───────────────────────────────────────────────────────────────── */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isGenerating) return;

  inputEl.value = '';
  autoResize();

  appendUserBubble(text);
  const { bubble, meta } = appendAssistantBubble();

  setGenerating(true);

  abortController = new AbortController();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, ...getSamplingParams() }),
      signal: abortController.signal,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Server error ${res.status}: ${errText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (!raw) continue;

        let event;
        try { event = JSON.parse(raw); }
        catch { continue; }

        if (event.error) {
          bubble.classList.remove('cursor');
          appendErrorBubble(event.error);
          break;
        }

        if (event.token !== undefined) {
          fullText += event.token;
          bubble.textContent = fullText;
          bubble.classList.add('cursor');
          scrollToBottom();
        }

        if (event.done) {
          bubble.classList.remove('cursor');
          meta.textContent =
            `${event.tokens} tok · ${event.tok_per_s} tok/s · ${event.time_s}s`
            + ` · novelty=${event.novelty} · reinforcement=${event.reinforcement}`
            + ` · stage=${event.stage} · age=${event.age}`;

          // Keep the status panel up to date
          stageVal.textContent = event.stage;
          ageVal.textContent   = event.age;
        }
      }
    }

  } catch (e) {
    if (e.name !== 'AbortError') {
      bubble.classList.remove('cursor');
      appendErrorBubble(e.message || String(e));
    } else {
      bubble.classList.remove('cursor');
      if (!bubble.textContent) bubble.textContent = '[stopped]';
    }
  } finally {
    setGenerating(false);
    abortController = null;
  }
}

/* ── Stop ───────────────────────────────────────────────────────────────── */
function stopGeneration() {
  if (abortController) {
    abortController.abort();
  }
}

/* ── New chat ───────────────────────────────────────────────────────────── */
async function newChat() {
  if (isGenerating) stopGeneration();

  try {
    const res = await fetch('/api/new_session', { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    console.info('New session started:', data);
  } catch (e) {
    console.warn('new_session error:', e);
  }

  // Clear the UI
  messagesEl.innerHTML = `
    <div class="welcome">
      <h2>Arnold</h2>
      <p>A biologically-inspired adaptive model.<br>Everything you say shapes how it thinks.</p>
    </div>`;

  inputEl.value = '';
  autoResize();
  loadModelInfo();
  inputEl.focus();
}

/* ── UI state helpers ───────────────────────────────────────────────────── */
function setGenerating(v) {
  isGenerating = v;
  sendBtn.disabled = v;
  sendBtn.style.display = v ? 'none' : '';
  stopBtn.style.display  = v ? '' : 'none';
}

/* ── Textarea auto-resize ───────────────────────────────────────────────── */
function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}

inputEl.addEventListener('input', autoResize);

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

/* ── Bindings ───────────────────────────────────────────────────────────── */
sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', stopGeneration);
newChatBtn.addEventListener('click', newChat);

/* ── Init ───────────────────────────────────────────────────────────────── */
loadModelInfo();
setInterval(loadModelInfo, 30_000);   // refresh status every 30 s
inputEl.focus();

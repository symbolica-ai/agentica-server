async function fetchPending() {
  const res = await fetch('/mock/pending');
  return await res.json();
}

async function fetchSent() {
  try {
    const res = await fetch('/mock/sent');
    if (!res.ok) return [];
    return await res.json();
  } catch (e) {
    return [];
  }
}

function formatDate(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

let selected = null;
let selectedData = null;
const urlParams = new URLSearchParams(window.location.search);
let selectedSentId = null;
let autoSubmitting = false;

function getPrefillParam() {
  const v = urlParams.get('prefill');
  return v ? decodeURIComponent(v) : '';
}

function renderPending(list) {
  const ul = document.getElementById('pending-list');
  ul.innerHTML = '';
  list.forEach((item) => {
    const li = document.createElement('li');
    const created = new Date(item.created_at * 1000).toLocaleTimeString();
    const top = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = item.model || '(model)';
    const close = document.createElement('button');
    close.textContent = '×';
    close.title = 'Clear (send 300)';
    close.style.float = 'right';
    close.style.background = 'transparent';
    close.style.color = '#e6edf3';
    close.style.border = 'none';
    close.style.cursor = 'pointer';
    close.onclick = (e) => {
      e.stopPropagation();
      clearPending(item.request_id);
    };
    top.appendChild(strong);
    top.appendChild(close);
    const time = document.createElement('div');
    time.style.color = '#99a3b2';
    time.style.fontSize = '12px';
    time.textContent = created;
    const count = document.createElement('div');
    count.style.color = '#99a3b2';
    count.style.fontSize = '12px';
    count.textContent = `messages: ${item.messages.length}`;
    li.appendChild(top);
    li.appendChild(time);
    li.appendChild(count);
    li.onclick = () => select(item.request_id, item);
    ul.appendChild(li);
  });
}

function select(id, data) {
  selected = id;
  selectedData = data;
  document.getElementById('empty').style.display = 'none';
  const details = document.getElementById('details');
  details.classList.remove('hidden');
  document.getElementById('meta-model').textContent = data.model || '';
  document.getElementById('meta-stop').textContent = Array.isArray(data.stop) ? data.stop.join(', ') : (data.stop || '');
  document.getElementById('meta-max').textContent = data.max_completion_tokens || '';
  renderTranscript(data.messages);
  renderRequestJSON(id);
  initToggle();
}

function renderTranscript(messages) {
  const t = document.getElementById('transcript');
  t.innerHTML = '';
  const prefill = getPrefillParam();

  let tailAssistantSpan = null; // assistant span only if the last message is assistant

  messages.forEach((m, idx) => {
    const line = document.createElement('div');
    line.className = 'line';
    const pill = document.createElement('span');
    pill.className = `pill ${m.role}`;
    pill.textContent = m.role;
    const span = document.createElement('span');
    const content = (m.reasoning_content ? `<think>${m.reasoning_content}</think>\n` : '') + (m.content || '');
    span.textContent = content;
    line.appendChild(pill);
    line.appendChild(span);
    t.appendChild(line);

    // Only keep a reference if this is the last message and it's assistant
    if (idx === messages.length - 1 && m.role === 'assistant') {
      tailAssistantSpan = span;
    }
  });

  // Inline assistant editor appended directly after the last assistant content if available
  const editor = document.createElement('div');
  editor.id = 'assistant-input';
  editor.className = 'inline-input';
  editor.contentEditable = 'true';
  editor.setAttribute('data-placeholder', 'Continue here...');
  editor.textContent = prefill || '';

  if (tailAssistantSpan) {
    // Append editor directly without adding extra text nodes/spaces
    tailAssistantSpan.parentElement.appendChild(editor);
  } else {
    // No prior assistant line; create one inline
    const container = document.createElement('div');
    container.className = 'line';
    const pill = document.createElement('span');
    pill.className = 'pill assistant';
    pill.textContent = 'assistant';
    container.appendChild(pill);
    container.appendChild(editor);
    t.appendChild(container);
  }

  // Focus and ensure caret is visible; autoscroll to bottom
  editor.focus();
  const transcriptEl = document.getElementById('transcript');
  if (transcriptEl) {
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
  editor.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'enter') {
      e.preventDefault();
      sendResponse();
    }
  });
  editor.addEventListener('input', () => maybeAutoSubmitOnStop(editor));
  const actions = document.createElement('div');
  actions.className = 'actions';
  const btn = document.createElement('button');
  btn.id = 'send';
  btn.textContent = 'Generate';
  btn.onclick = sendResponse;
  actions.appendChild(btn);
  t.appendChild(actions);
}

async function sendResponse() {
  if (!selected) return;
  if (autoSubmitting) return;
  const text = document.getElementById('assistant-input').value;
  // Support contentEditable editor fallback
  const editable = document.getElementById('assistant-input');
  const textCE = editable && editable.textContent !== undefined ? editable.textContent : null;
  let finalText = (text !== undefined ? text : '') || (textCE || '');
  // Stop token handling: if stop is present and found in finalText, trim at first occurrence
  const stop = selectedData && selectedData.stop ? selectedData.stop : null;
  if (stop) {
    const stops = Array.isArray(stop) ? stop : [stop];
    let earliest = null;
    let which = null;
    for (const s of stops) {
      const idx = finalText.indexOf(s);
      if (idx !== -1 && (earliest === null || idx < earliest)) {
        earliest = idx;
        which = s;
      }
    }
    if (earliest !== null) {
      finalText = finalText.slice(0, earliest);
      if (editable && editable.textContent !== undefined) {
        editable.textContent = finalText;
      }
    }
  }
  if (!finalText) return;
  const status = document.getElementById('status-select') ? document.getElementById('status-select').value : '200';
  autoSubmitting = true;
  const res = await fetch('/mock/respond', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: selected, content: finalText, status })
  });
  const j = await res.json();
  if (j.ok) {
    // Immediately refresh lists and clear selection
    selected = null;
    selectedData = null;
    await tick();
    setTimeout(() => { if (!selected) { tick(); } }, 300);
    document.getElementById('details').classList.add('hidden');
    document.getElementById('empty').style.display = 'flex';
  }
  autoSubmitting = false;
}

async function tick() {
  const [list, sent] = await Promise.all([fetchPending(), fetchSent()]);
  renderPending(list);
  if (Array.isArray(list)) {
    const selectedEntry = selected ? list.find((x) => x.request_id === selected) : null;
    const newestActive = list.find((x) => !x.completed);
    if (!selectedEntry || (selectedEntry && selectedEntry.completed)) {
      if (newestActive) {
        select(newestActive.request_id, newestActive);
      }
    }
  }
  renderSent(sent);
}

function renderSent(sent) {
  if (!Array.isArray(sent)) sent = [];
  const ul = document.getElementById('sent-list');
  const pre = document.getElementById('sent-viewer');
  ul.innerHTML = '';
  // keep pre content if selection persists
  sent
    .slice()
    .reverse()
    .forEach((entry, idx) => {
      const li = document.createElement('li');
      const created = new Date(entry.created_at * 1000).toLocaleTimeString();
      const resp = entry.response || {};
      const model = entry.model || resp.model || '(model)';
      const id = entry.id || resp.id || `status-${entry.status != null ? entry.status : ''}`;
      const status = entry.status != null ? String(entry.status) : '';
      li.textContent = `${created} — ${model} — ${status}`;
      li.onclick = () => {
        selectedSentId = id;
        pre.textContent = JSON.stringify(resp, null, 2);
      };
      if (selectedSentId === id) {
        li.style.borderColor = '#7aa2f7';
      }
      ul.appendChild(li);
    });

  // Restore viewer for current selection if present
  if (selectedSentId) {
    const current = sent.find((e) => (e.id === selectedSentId) || (e.response && e.response.id === selectedSentId));
    if (current) {
      pre.textContent = JSON.stringify((current.response || {}), null, 2);
    }
  }
}

function maybeAutoSubmitOnStop(editor) {
  if (!selectedData || !selectedData.stop) return;
  if (autoSubmitting) return;
  const stop = selectedData.stop;
  const stops = Array.isArray(stop) ? stop : [stop];
  const text = editor.textContent || '';
  let earliest = null;
  for (const s of stops) {
    const idx = text.indexOf(s);
    if (idx !== -1 && (earliest === null || idx < earliest)) earliest = idx;
  }
  if (earliest !== null) {
    editor.textContent = text.slice(0, earliest);
    sendResponse();
  }
}

async function renderRequestJSON(requestId) {
  try {
    const res = await fetch(`/mock/pending/${encodeURIComponent(requestId)}`);
    const pre = document.getElementById('request-json');
    if (!res.ok) {
      if (pre) pre.textContent = '(no request payload found)';
      return;
    }
    const body = await res.json();
    if (pre) pre.textContent = JSON.stringify(body, null, 2);
  } catch (e) {
    const pre = document.getElementById('request-json');
    if (pre) pre.textContent = '(error loading request JSON)';
  }
}

function initTabs() {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach((tab) => {
    tab.onclick = () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const which = tab.dataset.tab;
      document.querySelectorAll('.tabpane').forEach((p) => p.classList.remove('active'));
      const pane = document.getElementById(`tab-${which}`);
      if (pane) pane.classList.add('active');
    };
  });
}

function initToggle() {
  const btn = document.getElementById('toggle-view');
  const transcript = document.getElementById('tab-transcript');
  const request = document.getElementById('tab-request');
  let showing = 'transcript';
  btn.textContent = 'JSON view';
  btn.onclick = () => {
    if (showing === 'transcript') {
      showing = 'request';
      transcript.classList.remove('active');
      request.classList.add('active');
      btn.textContent = 'Transcript view';
      if (selected) renderRequestJSON(selected);
    } else {
      showing = 'transcript';
      request.classList.remove('active');
      transcript.classList.add('active');
      btn.textContent = 'JSON view';
      const editor = document.getElementById('assistant-input');
      if (editor) {
        editor.focus();
        const transcriptEl2 = document.getElementById('transcript');
        if (transcriptEl2) transcriptEl2.scrollTop = transcriptEl2.scrollHeight;
      }
    }
  };
}

async function clearPending(requestId) {
  await fetch('/mock/respond', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, content: 'cleared', status: 300 })
  });
  // Force refresh list and clear selection if we cleared the selected item
  if (selected === requestId) {
    selected = null;
    selectedData = null;
    document.getElementById('details').classList.add('hidden');
    document.getElementById('empty').style.display = 'flex';
  }
  await tick();
  setTimeout(() => { if (!selected) { tick(); } }, 200);
}

tick();
setInterval(tick, 1500);



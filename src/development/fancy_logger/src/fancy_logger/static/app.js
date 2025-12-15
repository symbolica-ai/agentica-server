const escapeHtml = unsafe => {
  return unsafe
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
};

(() => {
  const typePanel = document.getElementById('typePanel');
  const typeToggle = document.getElementById('typeToggle');
  const typePanelContainer = document.getElementById('typePanelContainer');
  const clearBtn = document.getElementById('clearBtn');
  const refreshBtn = document.getElementById('refreshBtn');
  const uidList = document.getElementById('uidList');
  const iidList = document.getElementById('iidList');
  const messagesEl = document.getElementById('messages');
  const selectionInfo = document.getElementById('selectionInfo');

  let selectedUid = '';
  let selectedIid = '';
  const typeStates = new Map(); // type -> 'off' | 'include' | 'exclude'
  let lastVersion = 0;
  let savedWidth = 0;
  const TYPE_STATE_KEY = 'fancy_logger_type_states_v1';

  function saveTypeStates() {
    const obj = {};
    for (const [t, s] of typeStates) {
      if (s && s !== 'off') obj[t] = s;
    }
    try { localStorage.setItem(TYPE_STATE_KEY, JSON.stringify(obj)); } catch {}
  }

  function loadTypeStates(types) {
    let obj = {};
    try {
      const raw = localStorage.getItem(TYPE_STATE_KEY);
      if (raw) obj = JSON.parse(raw);
    } catch {}
    typeStates.clear();
    for (const t of types) {
      const s = obj[t];
      if (s === 'include' || s === 'exclude') typeStates.set(t, s);
      else typeStates.set(t, 'off');
    }
  }

  // Deterministic pastel color from string
  function hashString(str) {
    let h = 2166136261 >>> 0; // FNV-1a basis
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  }
  function typeColor(type) {
    const h32 = hashString(type) >>> 0;
    const norm = h32 / 4294967296; // [0,1)
    const phi = 0.61803398875; // golden ratio conjugate
    const buckets = 28; // evenly spaced pastel hues
    const bucket = Math.floor(((norm + phi) % 1) * buckets);
    const hue = (bucket * (360 / buckets)) % 360;
    const s = 68;
    const l = 88;
    const borderL = 62;
    return {
      bg: `hsl(${hue} ${s}% ${l}%)`,
      border: `hsl(${hue} ${s}% ${borderL}%)`,
    };
  }

  const setSelectionInfo = () => {
    const includes = Array.from(typeStates).filter(([, s]) => s === 'include').map(([t]) => t);
    const excludes = Array.from(typeStates).filter(([, s]) => s === 'exclude').map(([t]) => t);
    const parts = [];
    parts.push(`Include: ${includes.length ? includes.join(', ') : '—'}`);
    parts.push(`Exclude: ${excludes.length ? excludes.join(', ') : '—'}`);
    parts.push(`UID: ${selectedUid || 'All'}`);
    parts.push(`IID: ${selectedIid || 'All'}`);
    selectionInfo.textContent = parts.join(' · ');
  };

  function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  function renderUidTabs(uids) {
    uidList.innerHTML = '';
    const allBtn = el('button', `tab ${selectedUid ? '' : 'active'}`.trim(), 'All');
    allBtn.addEventListener('click', () => {
      selectedUid = '';
      selectedIid = '';
      renderIidTabs([]);
      renderMessages();
      renderUidTabs(uids);
    });
    uidList.appendChild(allBtn);

    for (const uid of uids) {
      const btn = el('button', `tab ${selectedUid === uid ? 'active' : ''}`.trim(), uid);
      btn.title = uid;
      btn.addEventListener('click', () => {
        selectedUid = uid;
        selectedIid = '';
        renderMessages();
        loadIids(uid);
        renderUidTabs(uids);
      });
      uidList.appendChild(btn);
    }
  }

  function renderIidTabs(iids) {
    iidList.innerHTML = '';
    const allBtn = el('button', `subtab ${selectedIid ? '' : 'active'}`.trim(), 'All');
    allBtn.addEventListener('click', () => {
      selectedIid = '';
      renderMessages();
      renderIidTabs(iids);
    });
    iidList.appendChild(allBtn);

    for (const iid of iids) {
      const btn = el('button', `subtab ${selectedIid === iid ? 'active' : ''}`.trim(), iid);
      btn.title = iid;
      btn.addEventListener('click', () => {
        selectedIid = iid;
        renderMessages();
        renderIidTabs(iids);
      });
      iidList.appendChild(btn);
    }
  }

  function renderTypePanel(types) {
    typePanel.innerHTML = '';
    for (const t of types) {
      if (!typeStates.has(t)) typeStates.set(t, 'off');
      const state = () => typeStates.get(t);
      const colors = typeColor(t);
      const btn = el('button', `type-tile ${state()}`.trim(), t);
      btn.style.setProperty('--type-bg', colors.bg);
      btn.style.setProperty('--type-border', colors.border);
      btn.title = `Click to cycle: off → include → exclude → off`;
      btn.addEventListener('click', () => {
        const curr = state();
        const next = curr === 'off' ? 'include' : curr === 'include' ? 'exclude' : 'off';
        typeStates.set(t, next);
        btn.className = `type-tile ${next}`.trim();
        saveTypeStates();
        setSelectionInfo();
        renderMessages();
      });
      typePanel.appendChild(btn);
    }
    // add bottom-left resizer handle
    const resizer = document.createElement('div');
    resizer.className = 'resizer bl';
    typePanel.appendChild(resizer);

    let dragging = false;
    let startX = 0;
    let startLeft = 0, startWidth = 0, startRight = 0;
    const margin = 12;

    function onMouseMove(e) {
      if (!dragging) return;
      const dx = e.clientX - startX;

      // Horizontal: bottom-left handle moves left edge
      let newLeft = startLeft + dx;
      const minWidth = 240;
      const maxWidth = Math.min(window.innerWidth - margin, 1200);
      // Clamp left so width stays within [minWidth, maxWidth]
      newLeft = Math.min(newLeft, startRight - minWidth);
      newLeft = Math.max(newLeft, startRight - maxWidth);
      newLeft = Math.max(newLeft, margin);
      const newWidth = startRight - newLeft;

      typePanel.style.left = `${newLeft}px`;
      typePanel.style.width = `${newWidth}px`;
      savedWidth = newWidth;
    }

    function onMouseUp() {
      dragging = false;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp, true);
    }

    resizer.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragging = true;
      startX = e.clientX;
      const rect = typePanel.getBoundingClientRect();
      startLeft = rect.left;
      startWidth = rect.width;
      startRight = rect.right;
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp, true);
    });
  }

  function positionTypePanel() {
    // Ensure panel is open/displayed for measurement
    const wasOpen = typePanelContainer.classList.contains('open');
    if (!wasOpen) typePanelContainer.classList.add('open');
    typePanel.style.visibility = 'hidden';
    typePanel.style.display = 'grid';
    if (savedWidth > 0) typePanel.style.width = `${savedWidth}px`;
    // Use fixed positioning anchored under the toggle button, clamped to viewport
    const r = typeToggle.getBoundingClientRect();
    const panelWidth = typePanel.offsetWidth;
    const margin = 12;
    const left = Math.min(Math.max(margin, r.left), Math.max(margin, window.innerWidth - panelWidth - margin));
    const top = r.bottom + 8;
    typePanel.style.position = 'fixed';
    typePanel.style.left = `${left}px`;
    typePanel.style.top = `${top}px`;
    typePanel.style.visibility = '';
    if (!wasOpen) typePanelContainer.classList.remove('open');
  }

  async function loadTypes() {
    const types = await fetchJSON('/types');
    loadTypeStates(types);
    renderTypePanel(types);
  }

  async function loadUids() {
    const uids = await fetchJSON('/uids');
    renderUidTabs(uids);
  }

  async function loadIids(uid) {
    if (!uid) {
      renderIidTabs([]);
      return;
    }
    const iids = await fetchJSON(`/iids?uid=${encodeURIComponent(uid)}`);
    renderIidTabs(iids);
  }

  function extractInnerType(m) {
    const body = m && m.body;
    if (!body) return undefined;
    if (typeof body === 'object' && body.type) return body.type;
    if (typeof body === 'string') {
      try {
        const parsed = JSON.parse(body);
        if (parsed && typeof parsed === 'object' && parsed.type) return parsed.type;
      } catch {}
    }
    return undefined;
  }

  function renderMessagesList(messages) {
    messagesEl.innerHTML = '';
    if (!messages.length) {
      messagesEl.appendChild(el('div', 'empty', 'No messages'));
      return;
    }
    for (const m of messages) {
      const outerType = m.type || '';
      const innerType = extractInnerType(m) || '';
      const colorKey = outerType || innerType || 'unknown';
      const colors = typeColor(String(colorKey));

      const item = el('div', 'message');
      item.style.background = colors.bg;
      item.style.borderLeft = `8px solid ${colors.border}`;

      const header = el('div', 'message-header');
      const title = el('div', 'message-title');
      title.textContent = `${m.timestamp || ''} ${outerType || innerType}`.trim();
      const meta = el('div', 'message-meta');
      meta.textContent = `uid=${m.uid || ''} iid=${m.iid || ''}`.trim();
      header.appendChild(title);
      header.appendChild(meta);

      let body = el('pre', 'message-body');
      // Pretty-print inference request/response when available
      const prettyMaybeJson = (v) => {
        if (typeof v === 'string') {
          try {
            return JSON.stringify(JSON.parse(v), null, 2);
          } catch {
            return v;
          }
        }
        if (v && typeof v === 'object') return JSON.stringify(v, null, 2);
        return String(v ?? '');
      };
      const maybeJson = (v) => {
        if (typeof v === 'string') {
          try {
            return JSON.parse(v);
          } catch {
            return v;
          }
        }
        return v;
      };

      if (m.type === 'sm_inference_request' && m.request !== undefined) {
        body.textContent = prettyMaybeJson(m.request);
      } else if (m.type === 'sm_inference_response' && m.response !== undefined) {
        body.textContent = prettyMaybeJson(m.response);
      } else if (m.type === 'sm_create_agentic_function') {
        const name = escapeHtml(String(m.name ?? ''));
        const model = escapeHtml(String(m.model ?? ''));
        const jsonMode = String(m.json ?? '');
        const streaming = String(m.streaming ?? '');
        const persist = String(m.persist ?? '');
        const prompt = m.prompt != null ? escapeHtml(String(m.prompt)) : '';
        body = el('div', 'message-body');
        body.innerHTML = `
          <div><strong>Name:</strong> ${name}</div>
          <div><strong>Model:</strong> ${model}</div>
          <div><strong>JSON:</strong> ${jsonMode}</div>
          <div><strong>Streaming:</strong> ${streaming}</div>
          <div><strong>Persist:</strong> ${persist}</div>
          ${prompt ? `<details><summary><strong>Prompt</strong></summary><pre>${prompt}</pre></details>` : ''}
        `;
      } else if (m.type === 'sm_create_agent') {
        const doc = m.doc != null ? escapeHtml(String(m.doc)) : '';
        const model = escapeHtml(String(m.model ?? ''));
        const jsonMode = String(m.json ?? '');
        const streaming = String(m.streaming ?? '');
        body = el('div', 'message-body');
        body.innerHTML = `
          <div><strong>Model:</strong> ${model}</div>
          <div><strong>JSON:</strong> ${jsonMode}</div>
          <div><strong>Streaming:</strong> ${streaming}</div>
          ${doc ? `<details><summary><strong>Doc</strong></summary><pre>${doc}</pre></details>` : ''}
        `;
      }  else if (m.type === 'sm_monad' && m.body !== undefined) {
        const b = maybeJson(m.body)
        if (b.type === 'delta') {
          const delta = b.args[0];
          body = el('div', 'message-body');
          let html = `
            <span class="delta-type">delta</span>
            Role: <span class="role">${delta.role}</span>${delta.username ? ` (${delta.username})` : ''}
            <pre class="content">${escapeHtml(delta.content)}</pre>
          `;
          if (delta.tool_calls) {
            for (const tool_call of delta.tool_calls) {
              if (typeof tool_call.content === 'string') {
                tool_call.content = JSON.parse(tool_call.content);
              }
              html += `
                <strong>Tool Calls</strong>
                <pre>${JSON.stringify(tool_call, null, 2)}</pre>
              `;
            }
          }
          body.innerHTML = html;
        } else {
          body.textContent = prettyMaybeJson(m.body);
        }
        console.log("body", b);
      } else if (m.body !== undefined && m.body !== null) {
        body.textContent = String(m.body);
      } else {
        body.textContent = JSON.stringify(m, null, 2);
      }

      item.appendChild(header);
      item.appendChild(body);
      messagesEl.appendChild(item);
    }
  }

  async function renderMessages() {
    setSelectionInfo();
    const params = new URLSearchParams();
    for (const [t, s] of typeStates) {
      if (s === 'include') params.append('include', t);
      if (s === 'exclude') params.append('exclude', t);
    }
    if (selectedUid) params.set('uid', selectedUid);
    if (selectedIid) params.set('iid', selectedIid);
    params.set('limit', '500');
    const url = `/messages?${params.toString()}`;
    const data = await fetchJSON(url);
    renderMessagesList(data);
  }

  async function maybeRefresh() {
    try {
      const { version } = await fetchJSON('/version');
      if (typeof version !== 'number') return;
      if (version !== lastVersion) {
        lastVersion = version;
        // Rehydrate
        await loadTypes();
        await loadUids();
        await loadIids(selectedUid);
        await renderMessages();
      }
    } catch (e) {
      // ignore transient errors
    }
  }

  // Events
  typeToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const nowOpen = !typePanelContainer.classList.contains('open');
    typePanelContainer.classList.toggle('open');
    if (nowOpen) {
      positionTypePanel();
    } else {
      // Clear inline styles so CSS can fully hide the panel
      typePanel.style.display = '';
      typePanel.style.visibility = '';
      typePanel.style.left = '';
      typePanel.style.top = '';
      typePanel.style.position = '';
    }
  });
  document.addEventListener('click', (e) => {
    if (!typePanelContainer.contains(e.target) && e.target !== typeToggle) {
      typePanelContainer.classList.remove('open');
      typePanel.style.display = '';
      typePanel.style.visibility = '';
      typePanel.style.left = '';
      typePanel.style.top = '';
      typePanel.style.position = '';
    }
  });
  window.addEventListener('resize', () => {
    if (typePanelContainer.classList.contains('open')) positionTypePanel();
  });
  window.addEventListener('scroll', () => {
    if (typePanelContainer.classList.contains('open')) positionTypePanel();
  }, true);
  refreshBtn.addEventListener('click', () => {
    renderMessages();
  });

  clearBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/logs', { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (e) {
      // ignore
    }
    // Reset selections and reload everything
    selectedUid = '';
    selectedIid = '';
    await loadTypes();
    await loadUids();
    await loadIids(selectedUid);
    await renderMessages();
  });

  // Initial load
  (async () => {
    await loadTypes();
    await loadUids();
    await loadIids(selectedUid);
    await renderMessages();
    try {
      const v = await fetchJSON('/version');
      if (typeof v.version === 'number') lastVersion = v.version;
    } catch {}
    setInterval(maybeRefresh, 1000);
  })();
})();

const $ = (id) => document.getElementById(id);
const log = $('log'), q = $('q');
let modelName = '', sid = null, session = null, projects = [], ctxSelected = new Set(), idxPid = null, busy = false;

/* ---------- helpers ---------- */
// Build DOM without innerHTML so names/paths/model output can never inject markup.
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  n.append(...kids);
  return n;
}

function toast(msg, kind = 'error') {
  const t = el('div', {class: 'toast ' + kind, role: kind === 'error' ? 'alert' : 'status'}, msg);
  $('toasts').appendChild(t);
  setTimeout(() => t.remove(), kind === 'error' ? 8000 : 4000);
}

// Turn a failed response into a readable message (FastAPI sends {"detail": ...}).
async function errorText(r) {
  let detail = '';
  try {
    const j = await r.json();
    detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail || j);
  } catch (e) { /* not JSON */ }
  const hint = {
    413: 'File too large for its type.',
    409: 'The index was built with another embedding model: run "Index project" again.',
    502: 'Ollama did not answer properly. Is it running, and are the models pulled?',
    400: 'Invalid request.',
  }[r.status];
  return [hint, detail].filter(Boolean).join(' ') || `HTTP ${r.status}`;
}

async function request(url, opts) {
  let r;
  try { r = await fetch(url, opts); }
  catch (e) { throw new Error('Cannot reach the server. Check the network connection.'); }
  if (!r.ok) throw new Error(await errorText(r));
  return r;
}
const api = async (url, opts) => (await request(url, opts)).json();
const json = (method, body) => ({method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message); } };

/* ---------- theme ---------- */
const media = window.matchMedia('(prefers-color-scheme: dark)');
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === 'dark' : media.matches;
};
function paintThemeBtn() { $('themeBtn').textContent = isDark() ? '☀' : '☾'; }
$('themeBtn').onclick = () => {
  const next = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch (e) { /* private mode */ }
  paintThemeBtn();
};
media.addEventListener('change', paintThemeBtn);
paintThemeBtn();

/* ---------- safe markdown ---------- */
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

// Input is escaped first, so only the tags produced here can reach innerHTML.
function inline(s) {
  const codes = [];
  s = esc(s).replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + codes[i] + '</code>');
}

function markdown(src) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const isBlockStart = (l) => /^\s*(```|#{1,4}\s|[-*+]\s|\d+[.)]\s|>)/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = line.match(/^\s*```\s*([\w+#.-]*)/))) {
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i++]);
      i++; // closing fence (an unclosed fence, e.g. mid-stream, runs to the end)
      out.push(`<div class="code"><div class="code-h"><span>${esc(m[1] || 'code')}</span>` +
        `<button type="button" class="ghost copy">Copy</button></div><pre><code>${esc(code.join('\n'))}</code></pre></div>`);
    } else if ((m = line.match(/^(#{1,4})\s+(.*)/))) {
      out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`);
      i++;
    } else if (/^\s*([-*+]|\d+[.)])\s/.test(line)) {
      const ordered = /^\s*\d/.test(line), items = [];
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s/.test(lines[i]))
        items.push(`<li>${inline(lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, ''))}</li>`);
      out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`);
    } else if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${inline(quote.join(' '))}</blockquote>`);
    } else if (!line.trim()) {
      i++;
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && (!para.length || !isBlockStart(lines[i]))) para.push(lines[i++]);
      out.push(`<p>${para.map(inline).join('<br>')}</p>`);
    }
  }
  return out.join('');
}

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  // plain-http LAN pages have no Clipboard API
  const ta = el('textarea', {'aria-hidden': 'true', style: 'position:fixed;opacity:0'});
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand('copy');
  ta.remove();
  return ok ? Promise.resolve() : Promise.reject(new Error('copy failed'));
}

log.addEventListener('click', async (e) => {
  const btn = e.target.closest('button.copy');
  if (!btn) return;
  try {
    await copyText(btn.closest('.code').querySelector('code').textContent);
    btn.textContent = 'Copied';
  } catch (err) { btn.textContent = 'Failed'; }
  setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
});

/* ---------- sessions ---------- */
function toggleNav(open) {
  document.body.classList.toggle('nav-open', open);
  if (open) $('side').querySelector('button, a[href], input, select, textarea')?.focus();
  else if ($('side').contains(document.activeElement)) $('menuBtn').focus();
}
$('menuBtn').onclick = () => toggleNav(true);
$('scrim').onclick = () => toggleNav(false);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') toggleNav(false); });

const loadSessions = guard(async () => {
  const rows = await api('/api/sessions');
  const box = $('sessions');
  box.replaceChildren();
  if (!rows.length) box.append(el('div', {class: 'muted', style: 'padding:8px 10px'}, 'No chats yet.'));
  rows.forEach((s) => {
    const del = async () => {
      if (!confirm('Delete this session?')) return;
      try { await request('/api/sessions/' + s.id, {method: 'DELETE'}); }
      catch (e) { toast(e.message); return; }
      if (sid === s.id) { sid = null; session = null; render(); }
      loadSessions();
    };
    const open = el('button', {class: 'open', type: 'button', onclick: () => openSession(s.id)}, s.title);
    if (s.id === sid) open.setAttribute('aria-current', 'true');
    const item = el('div', {class: 'item' + (s.id === sid ? ' active' : ''), oncontextmenu: (e) => { e.preventDefault(); del(); }},
      open, el('button', {class: 'icon del danger', type: 'button', 'aria-label': 'Delete ' + s.title, title: 'Delete', onclick: del}, '✕'));
    box.appendChild(item);
  });
});

const newSession = guard(async () => {
  const r = await api('/api/sessions', json('POST', {title: 'New chat'}));
  await openSession(r.id);
});

const openSession = guard(async (id) => {
  sid = id;
  session = await api('/api/sessions/' + id);
  render();
  loadSessions();
  toggleNav(false);
});

function render() {
  log.replaceChildren($('empty'));
  $('title').textContent = session ? session.title : 'No session';
  const proj = session && session.project_id
    ? 'Project: ' + ((projects.find((p) => p.id === session.project_id) || {}).name || '#' + session.project_id) : '';
  $('sub').textContent = [modelName, proj].filter(Boolean).join(' \u00b7 ');
  $('chips').replaceChildren(...((session && session.context) || []).map((f) => el('span', {class: 'chip'}, f)));
  if (session) session.messages.forEach((m) => bubble(m.role === 'user' ? 'user' : 'bot', m.content));
  $('empty').hidden = !!(session && session.messages.length);
}

function setBody(node, cls, text) {
  if (cls === 'user') node.textContent = text;
  else node.innerHTML = markdown(text);  // escaped by markdown() first
}

function bubble(cls, text) {
  $('empty').hidden = true;
  const d = el('div', {class: 'msg ' + cls});
  setBody(d, cls, text);
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

/* ---------- chat ---------- */
q.addEventListener('input', () => {
  q.style.height = 'auto';
  q.style.height = Math.min(q.scrollHeight, 200) + 'px';
});
q.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('ask').requestSubmit(); }
});

$('ask').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true;
  $('send').disabled = true;
  try {
    if (!sid) await newSession();
    if (!sid) return;
    q.value = '';
    q.style.height = 'auto';
    bubble('user', text);
    await stream(text);
  } finally {
    busy = false;
    $('send').disabled = false;
    q.focus();
    loadSessions();
  }
};

async function stream(text) {
  const out = bubble('bot', '');
  out.innerHTML = '<span class="typing" role="status" aria-label="Assistant is typing"><i></i><i></i><i></i></span>';
  let full = '', queued = false;
  const paint = () => {
    queued = false;
    setBody(out, 'bot', full);
    log.scrollTop = log.scrollHeight;
  };
  try {
    const res = await fetch('/api/chat', json('POST', {session_id: sid, content: text}));
    if (!res.ok) throw new Error(await errorText(res));
    const reader = res.body.getReader(), dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      full += dec.decode(value, {stream: true});
      if (!queued) { queued = true; requestAnimationFrame(paint); }
    }
    paint();
    if (!full.trim()) throw new Error('Empty reply.');
  } catch (err) {
    const down = !full && (err instanceof TypeError || err.message === 'Empty reply.');
    const msg = down ? 'No reply. Ollama may not be running (or the model is not pulled).' : err.message;
    if (full) paint(); else { out.classList.add('err'); out.textContent = msg; }
    toast(msg);
  }
}

/* ---------- dialogs ---------- */
document.querySelectorAll('dialog').forEach((d) => {
  d.querySelectorAll('[data-close]').forEach((b) => b.addEventListener('click', () => d.close()));
  d.addEventListener('click', (e) => { if (e.target === d) d.close(); });  // backdrop click
});
$('newBtn').onclick = newSession;
$('filesBtn').onclick = () => { $('filesDlg').showModal(); loadFiles(); };
// arrows: these handlers are declared further down (const, so not yet initialised here)
$('projBtn').onclick = () => openProjects();
$('ctxBtn').onclick = () => openContext();
$('addProj').onclick = () => addProject();
$('saveCtx').onclick = () => saveContext();

/* ---------- file transfer ---------- */
function fmtSize(n) {
  return n >= 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB';
}

const loadFiles = guard(async () => {
  const rows = await api('/api/files');
  const box = $('fileList');
  box.replaceChildren();
  if (!rows.length) box.append(el('div', {class: 'muted'}, 'No files yet.'));
  rows.forEach((f) => {
    box.append(el('div', {class: 'line'},
      el('span', {class: 'grow', title: f.name}, f.name),
      el('span', {class: 'muted'}, fmtSize(f.size)),
      el('button', {class: 'ghost', 'aria-label': 'Download ' + f.name, onclick: () => { location.href = '/api/files/' + encodeURIComponent(f.name); }}, '↓'),
      el('button', {class: 'ghost danger', 'aria-label': 'Delete ' + f.name, onclick: guard(async () => {
        await request('/api/files/' + encodeURIComponent(f.name), {method: 'DELETE'});
        loadFiles();
      })}, '✕')));
  });
});

$('upInput').onchange = async (e) => {
  for (const f of e.target.files) {
    const fd = new FormData();
    fd.append('file', f);
    await guard(() => request('/api/upload', {method: 'POST', body: fd}))();
  }
  e.target.value = '';
  loadFiles();
};

/* ---------- projects ---------- */
const openProjects = guard(async () => {
  projects = await api('/api/projects');
  const box = $('projList');
  box.replaceChildren();
  if (!projects.length) box.append(el('div', {class: 'muted'}, 'No projects yet.'));
  projects.forEach((p) => {
    box.append(el('div', {class: 'line'},
      el('span', {class: 'grow', title: p.path}, p.name, el('br'), el('span', {class: 'muted'}, p.path)),
      el('button', {class: 'ghost', onclick: () => openIndex(p)}, 'Index'),
      el('button', {class: 'ghost danger', 'aria-label': 'Delete project ' + p.name, onclick: guard(async () => {
        if (!confirm(`Remove project "${p.name}" and its index?`)) return;
        await request('/api/projects/' + p.id, {method: 'DELETE'});
        openProjects();
      })}, '✕')));
  });
  if (!$('projDlg').open) $('projDlg').showModal();
});

const addProject = guard(async () => {
  await api('/api/projects', json('POST', {name: $('projName').value, path: $('projPath').value}));
  $('projName').value = ''; $('projPath').value = '';
  openProjects();
});

/* ---------- project index (RAG) ---------- */
const OUTCOMES = {
  indexed: 'indexed', captioned: 'images described', unchanged: 'unchanged', skipped: 'skipped',
  skipped_pdf_no_text: 'PDFs without text layer', skipped_pdf_unreadable: 'PDFs encrypted/corrupt',
  skipped_image_failed: 'images the vision model failed on', skipped_image_disabled: 'images (vision off)',
  shadowed: 'kept uploaded version', deleted: 'removed (gone from disk)', missing: 'missing', failed: 'failed',
};
const RESULT_TEXT = {
  indexed: 'Indexed.', captioned: 'Image described and indexed.', unchanged: 'Already up to date.',
  skipped: 'Skipped: empty, binary or too large.',
  skipped_pdf_no_text: 'Skipped: this PDF has no text layer (scanned?).',
  skipped_pdf_unreadable: 'Skipped: this PDF is encrypted or corrupt.',
  skipped_image_failed: 'Skipped: the vision model could not describe the image (is it pulled?).',
  skipped_image_disabled: 'Skipped: image support is off (VISION_MODEL is empty).',
  shadowed: 'Not replaced: an uploaded file uses this path.', deleted: 'Removed from the index.', missing: 'Not found.',
};

function showResult(title, lines, bad = false) {
  const box = $('idxResult');
  box.hidden = false;
  box.className = 'result' + (bad ? ' bad' : '');
  box.replaceChildren(el('b', {}, title), el('ul', {}, ...lines.map((l) => el('li', {}, l))));
}

function openIndex(p) {
  idxPid = p.id;
  $('idxH').textContent = 'Index: ' + p.name;
  $('idxResult').hidden = true;
  loadIndex();
  $('idxDlg').showModal();
}

const loadIndex = guard(async () => {
  const rows = await api(`/api/projects/${idxPid}/index`);
  const box = $('idxFiles');
  box.replaceChildren();
  if (!rows.length) { box.append(el('div', {class: 'muted'}, 'Nothing indexed yet. Click "Index project" or add files.')); return; }
  rows.forEach((f) => {
    box.append(el('div', {class: 'line'},
      el('span', {class: 'grow', title: f.path}, f.path),
      el('span', {class: 'badge' + (f.chunks ? ' ok' : ' warn')}, f.chunks ? `${f.chunks} chunk${f.chunks > 1 ? 's' : ''}` : 'no text'),
      f.source === 'upload' ? el('span', {class: 'badge'}, 'upload') : '',
      f.source === 'scan' ? el('button', {class: 'ghost', 'aria-label': 'Re-sync ' + f.path, title: 'Re-sync from disk', onclick: () => resync(f.path)}, '↻') : '',
      el('button', {class: 'ghost danger', 'aria-label': 'Remove ' + f.path + ' from index', title: 'Remove from index', onclick: () => unindex(f.path)}, '✕')));
  });
});

const resync = guard(async (path) => {
  const r = await api(`/api/projects/${idxPid}/index/file?path=${encodeURIComponent(path)}`, {method: 'POST'});
  toast(`${path}: ${RESULT_TEXT[r.result] || r.result}`, r.result.startsWith('skipped') ? 'error' : 'ok');
  loadIndex();
});

const unindex = guard(async (path) => {
  await request(`/api/projects/${idxPid}/index/file?path=${encodeURIComponent(path)}`, {method: 'DELETE'});
  loadIndex();
});

$('indexBtn').onclick = async () => {
  const btn = $('indexBtn');
  btn.disabled = true;
  btn.textContent = 'Indexing…';
  showResult('Indexing…', ['Large PDFs and images (described by a vision model) can take a while.']);
  try {
    const s = await api(`/api/projects/${idxPid}/index`, {method: 'POST'});
    const lines = Object.entries(s).filter(([k, v]) => v && OUTCOMES[k]).map(([k, v]) => `${v} ${OUTCOMES[k]}`);
    showResult(s.failed ? 'Finished with errors' : 'Done', lines.length ? lines : ['Nothing to index.'], !!s.failed);
  } catch (e) {
    showResult('Indexing stopped', [e.message], true);
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Index project';
    loadIndex();
  }
};

$('projUpload').onchange = async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  const lines = [];
  let bad = false;
  showResult('Uploading…', [`${files.length} file(s)`]);
  for (const f of files) {
    const fd = new FormData();
    fd.append('file', f);
    try {
      const r = await api(`/api/projects/${idxPid}/upload`, {method: 'POST', body: fd});
      lines.push(`${f.name}: ${RESULT_TEXT[r.result] || r.result}`);
      bad = bad || r.result.startsWith('skipped');
    } catch (err) {
      lines.push(`${f.name}: ${err.message}`);
      bad = true;
    }
  }
  showResult(bad ? 'Uploaded, with problems' : 'Uploaded', lines, bad);
  loadIndex();
};

/* ---------- context ---------- */
const openContext = guard(async () => {
  if (!sid) { toast('Open or start a chat first.', 'ok'); return; }
  projects = await api('/api/projects');
  const sel = $('ctxProject');
  sel.replaceChildren(el('option', {value: ''}, 'No project'));
  projects.forEach((p) => {
    const o = el('option', {value: p.id}, p.name);
    if (p.id === session.project_id) o.selected = true;
    sel.appendChild(o);
  });
  ctxSelected = new Set(session.context || []);
  await loadProjectFiles();
  $('ctxDlg').showModal();
});

$('ctxProject').onchange = () => {
  ctxSelected = new Set();   // paths belong to the old project, drop them
  loadProjectFiles();
};
$('ctxFilter').oninput = () => loadProjectFiles();

const loadProjectFiles = guard(async () => {
  const pid = $('ctxProject').value;
  if (!pid) { $('ctxFiles').replaceChildren(el('div', {class: 'muted'}, 'Pick a project.')); updateCtxCount(); return; }
  paintFiles(await api(`/api/projects/${pid}/files?q=${encodeURIComponent($('ctxFilter').value)}`));
});

$('ctxGrep').onkeydown = guard(async (e) => {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  const pid = $('ctxProject').value;
  if (pid) paintFiles(await api(`/api/projects/${pid}/grep?q=${encodeURIComponent(e.target.value)}`));
});

function paintFiles(files) {
  const box = $('ctxFiles');
  box.replaceChildren();
  updateCtxCount();
  if (!files.length) { box.append(el('div', {class: 'muted'}, 'Nothing matched.')); return; }
  files.slice(0, 400).forEach((f) => {
    const cb = el('input', {type: 'checkbox', value: f});
    cb.checked = ctxSelected.has(f);
    // Track in a Set, not the DOM: filtering re-renders the list, and a file
    // selected under an earlier filter must survive that.
    cb.onchange = () => { cb.checked ? ctxSelected.add(f) : ctxSelected.delete(f); updateCtxCount(); };
    box.append(el('label', {class: 'pick'}, cb, f));
  });
}

function updateCtxCount() {
  const n = ctxSelected.size;
  $('ctxCount').textContent = n ? `${n} file${n > 1 ? 's' : ''} selected` : 'No files selected';
}

const saveContext = guard(async () => {
  const pid = $('ctxProject').value;
  await api(`/api/sessions/${sid}/context`, json('PUT', {files: [...ctxSelected]}));
  // persist project choice on the session
  await request(`/api/sessions/${sid}/project?pid=${pid || ''}`, {method: 'PUT'});
  session = await api('/api/sessions/' + sid);
  render();
  $('ctxDlg').close();
});

/* ---------- start ---------- */
(async () => {
  try {
    const h = await api('/api/health');
    modelName = h.model;
  } catch (e) { toast(e.message); }
  projects = await api('/api/projects').catch(() => []);
  render();
  loadSessions();
})();

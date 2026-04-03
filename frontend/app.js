'use strict';

// ─────────────────────────── STATE ───────────────────────────
const state = {
  sampleTickets: [],
  currentResult: null,
  queue: [],
  stats: { total: 0, auto: 0, review: 0, escalated: 0 },
  stepData: {},   // agent key → result data
};

const AGENT_META = {
  novelty_detector:  { num: 1, label: 'Novelty Detector',      icon: '◈', verifier: false },
  classifier:        { num: 2, label: 'Classifier',             icon: '⊞', verifier: false },
  researcher:        { num: 3, label: 'Researcher (RAG)',       icon: '⊛', verifier: false },
  responder:         { num: 4, label: 'Responder',              icon: '⊡', verifier: false },
  grounding_checker: { num: 5, label: 'Grounding Checker',      icon: '✓', verifier: true  },
  confidence_scorer: { num: 6, label: 'Confidence Scorer',      icon: '⊕', verifier: true  },
};

const WS_URL = `ws://${location.host}/ws/process`;

// ─────────────────────────── INIT ───────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await fetchStatus();
  await fetchSampleTickets();
});

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const dot  = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');

    if (data.llm === 'ollama') {
      dot.className  = 'status-dot online';
      text.textContent = `Ollama · ${data.models[0] ?? 'unknown'}`;
    } else {
      dot.className  = 'status-dot demo';
      text.textContent = 'Demo mode (Ollama not running)';
    }
  } catch {
    document.querySelector('.status-dot').className = 'status-dot offline';
    document.querySelector('.status-text').textContent = 'Connection error';
  }
}

async function fetchSampleTickets() {
  try {
    const res = await fetch('/api/sample-tickets');
    state.sampleTickets = await res.json();
    const sel = document.getElementById('sampleSelect');
    state.sampleTickets.forEach((t, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = t.subject;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to load sample tickets', e);
  }
}

// ─────────────────────────── SAMPLE LOADER ───────────────────────────
function loadSample() {
  const sel = document.getElementById('sampleSelect');
  const idx = parseInt(sel.value);
  if (isNaN(idx)) return;
  const t = state.sampleTickets[idx];
  document.getElementById('ticketSubject').value   = t.subject ?? '';
  document.getElementById('ticketBody').value      = t.body ?? '';
  document.getElementById('ticketCustomer').value  = `${t.customer ?? ''} · ${t.company ?? ''}`.replace(/^ · | · $/g, '');
  if (t.plan) {
    const planSel = document.getElementById('ticketPlan');
    [...planSel.options].forEach(o => { o.selected = o.value === t.plan; });
  }
}

// ─────────────────────────── PIPELINE RUNNER ───────────────────────────
function processTicket() {
  const subject  = document.getElementById('ticketSubject').value.trim();
  const body     = document.getElementById('ticketBody').value.trim();
  const plan     = document.getElementById('ticketPlan').value;

  if (!body) {
    shakeElement(document.getElementById('ticketBody'));
    return;
  }

  const ticket = subject ? `${subject}\n\n${body}` : body;

  // Reset UI
  document.getElementById('emptyState').style.display      = 'none';
  document.getElementById('pipelineContainer').style.display = 'block';
  document.getElementById('resultCard').style.display       = 'none';
  state.stepData = {};

  initPipelineSteps();
  disableButton(true);

  const ws = new WebSocket(WS_URL);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handlePipelineMessage(msg);
  };

  ws.onopen = () => {
    ws.send(JSON.stringify({ ticket, plan }));
  };

  ws.onerror = () => {
    disableButton(false);
    showError('WebSocket connection failed. Is the server running?');
  };

  ws.onclose = () => {
    disableButton(false);
  };
}

function initPipelineSteps() {
  const container = document.getElementById('pipelineSteps');
  container.innerHTML = '';

  Object.entries(AGENT_META).forEach(([key, meta]) => {
    const card = document.createElement('div');
    card.className   = 'step-card state-pending';
    card.id          = `step-${key}`;
    card.dataset.key = key;

    card.innerHTML = `
      <div class="step-header" onclick="toggleStep('${key}')">
        <div class="step-num">${meta.num}</div>
        <div class="step-label">
          ${meta.verifier ? '<span style="font-size:0.65rem;background:#fef9c3;color:#a16207;border-radius:10px;padding:1px 6px;margin-right:4px;font-weight:600;">VERIFY</span>' : ''}
          ${meta.label}
        </div>
        <div class="step-status">
          <span class="step-status-text">Waiting</span>
        </div>
        <svg class="step-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </div>
      <div class="step-body" id="step-body-${key}"></div>
    `;

    container.appendChild(card);
  });
}

function handlePipelineMessage(msg) {
  if (msg.type === 'start') return;

  if (msg.type === 'agent_start') {
    setStepState(msg.agent, 'running');
    return;
  }

  if (msg.type === 'agent_done') {
    state.stepData[msg.agent] = msg;
    setStepState(msg.agent, msg.flag ? 'flagged' : 'done');
    renderStepBody(msg.agent, msg);
    return;
  }

  if (msg.type === 'complete') {
    state.currentResult = msg;
    renderResult(msg);
    updateStats(msg);
    return;
  }

  if (msg.type === 'error') {
    showError(msg.message);
  }
}

// ─────────────────────────── STEP STATE MANAGEMENT ───────────────────────────
function setStepState(key, state_) {
  const card = document.getElementById(`step-${key}`);
  if (!card) return;
  card.className = `step-card state-${state_}`;

  const statusEl = card.querySelector('.step-status-text');
  const statusWrap = card.querySelector('.step-status');

  // Remove existing spinner
  const existingSpinner = statusWrap.querySelector('.spinner');
  if (existingSpinner) existingSpinner.remove();

  if (state_ === 'running') {
    statusEl.textContent = 'Processing';
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    statusWrap.prepend(spinner);
  } else if (state_ === 'done') {
    statusEl.textContent = 'Done';
  } else if (state_ === 'flagged') {
    statusEl.textContent = 'Flagged';
  } else if (state_ === 'error') {
    statusEl.textContent = 'Error';
  }
}

function toggleStep(key) {
  const card = document.getElementById(`step-${key}`);
  if (!card || card.classList.contains('state-pending') || card.classList.contains('state-running')) return;
  card.classList.toggle('expanded');
}

// ─────────────────────────── STEP BODY RENDERERS ───────────────────────────
function renderStepBody(key, data) {
  const body = document.getElementById(`step-body-${key}`);
  if (!body) return;

  switch (key) {
    case 'novelty_detector': body.innerHTML = renderNoveltyBody(data); break;
    case 'classifier':       body.innerHTML = renderClassifierBody(data); break;
    case 'researcher':       body.innerHTML = renderResearcherBody(data); break;
    case 'responder':        body.innerHTML = renderResponderBody(data); break;
    case 'grounding_checker': body.innerHTML = renderGroundingBody(data); break;
    case 'confidence_scorer': body.innerHTML = renderScorerBody(data); break;
  }

  // Auto-expand on done
  const card = document.getElementById(`step-${key}`);
  if (card) card.classList.add('expanded');

  // Update status badge in header
  updateStepBadge(key, data);
}

function updateStepBadge(key, data) {
  const card = document.getElementById(`step-${key}`);
  if (!card) return;
  const statusWrap = card.querySelector('.step-status');

  let badge = '';
  if (key === 'novelty_detector') {
    badge = data.is_novel
      ? `<span class="step-badge badge-novel">Novel</span>`
      : `<span class="step-badge badge-known">Known</span>`;
  } else if (key === 'classifier') {
    const p = data.priority ?? '';
    badge = `<span class="step-badge badge-${p.toLowerCase()}">${capitalize(p)}</span>`;
  } else if (key === 'grounding_checker') {
    const pct = Math.round((data.grounding_score ?? 0) * 100);
    const cls = pct >= 80 ? 'badge-known' : pct >= 50 ? 'badge-medium' : 'badge-high';
    badge = `<span class="step-badge ${cls}">${pct}% grounded</span>`;
  } else if (key === 'confidence_scorer') {
    const r = data.routing ?? '';
    const cls = r === 'AUTO_APPROVE' ? 'badge-auto' : r === 'LIGHT_REVIEW' ? 'badge-light' : 'badge-full';
    badge = `<span class="step-badge ${cls}">${data.routing_label ?? r}</span>`;
  } else if (data.elapsed_ms != null) {
    badge = `<span style="font-size:0.72rem;color:var(--slate-400)">${data.elapsed_ms}ms</span>`;
  }

  const existing = statusWrap.querySelector('.step-badge');
  if (existing) existing.remove();

  const statusText = statusWrap.querySelector('.step-status-text');
  if (badge) {
    const tmp = document.createElement('div');
    tmp.innerHTML = badge;
    statusWrap.insertBefore(tmp.firstChild, statusText);
    statusText.textContent = '';
  }
}

function renderNoveltyBody(d) {
  const simPct = Math.round((d.similarity_to_known ?? 0) * 100);
  const fillClass = d.is_novel ? 'fill-red' : 'fill-green';
  return `
    <div class="step-detail-grid">
      ${detailItem('Result',     d.is_novel ? '⚠ Novel ticket' : '✓ Known pattern')}
      ${detailItem('Similarity', `${simPct}% to nearest known`)}
      ${detailItem('Threshold',  `${Math.round((d.threshold ?? 0.45) * 100)}%`)}
      ${detailItem('Method',     d.method ?? '—')}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Similarity to known distribution</span><span>${simPct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${simPct}%"></div></div>
    </div>
    ${d.nearest_ticket ? `<div style="font-size:0.78rem;color:var(--slate-500);margin-top:0.4rem">Nearest known: <em>${escHtml(d.nearest_ticket)}</em></div>` : ''}
    ${d.note ? `<div style="font-size:0.78rem;color:var(--slate-500);margin-top:0.5rem;padding:0.5rem;background:var(--slate-50);border-radius:6px;border:1px solid var(--slate-200)">${escHtml(d.note)}</div>` : ''}
  `;
}

function renderClassifierBody(d) {
  const confPct = Math.round((d.confidence ?? 0) * 100);
  const fillClass = confPct >= 75 ? 'fill-green' : confPct >= 55 ? 'fill-amber' : 'fill-red';
  return `
    <div class="step-detail-grid">
      ${detailItem('Category', capitalize(d.category ?? '—'))}
      ${detailItem('Priority', capitalize(d.priority ?? '—'))}
      ${detailItem('Confidence', `${confPct}%`)}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Classifier confidence</span><span>${confPct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${confPct}%"></div></div>
    </div>
    ${d.reasoning ? `<div style="font-size:0.78rem;color:var(--slate-500);margin-top:0.5rem">${escHtml(d.reasoning)}</div>` : ''}
  `;
}

function renderResearcherBody(d) {
  const articles = d.articles ?? [];
  return `
    <div style="font-size:0.78rem;color:var(--slate-500);margin-bottom:0.6rem">
      Retrieved ${articles.length} KB article(s) using ${d.method ?? 'semantic'} search.
    </div>
    <div class="step-articles">
      ${articles.map(a => `
        <span class="article-chip">
          ${escHtml(a.id)} — ${escHtml(a.title)}
          <span class="article-relevance">${Math.round((a.relevance ?? 0) * 100)}%</span>
        </span>
      `).join('')}
    </div>
  `;
}

function renderResponderBody(d) {
  return `
    <div style="font-size:0.78rem;color:var(--slate-500);margin-bottom:0.4rem">
      Draft response (pre-verification):
    </div>
    <div class="step-draft">${escHtml(d.draft ?? '')}</div>
  `;
}

function renderGroundingBody(d) {
  const pct = Math.round((d.grounding_score ?? 0) * 100);
  const fillClass = pct >= 80 ? 'fill-green' : pct >= 50 ? 'fill-amber' : 'fill-red';
  const sentences = d.sentences ?? [];
  return `
    <div class="step-detail-grid">
      ${detailItem('Claims checked',  d.claims_checked ?? 0)}
      ${detailItem('Claims grounded', d.claims_grounded ?? 0)}
      ${detailItem('Removed',         (d.ungrounded ?? []).length)}
      ${detailItem('Grounding score', `${pct}%`)}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Response grounding</span><span>${pct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${pct}%"></div></div>
    </div>
    ${sentences.length > 0 ? `
      <div style="margin-top:0.75rem;font-size:0.73rem;color:var(--slate-500);margin-bottom:0.35rem;font-weight:500">SENTENCE-LEVEL GROUNDING</div>
      ${sentences.slice(0, 8).map(s => `
        <div class="grounding-row">
          <div class="grounding-dot ${s.grounded ? 'ok' : 'bad'}"></div>
          <div style="flex:1">
            <span style="color:var(--slate-600)">${escHtml(s.sentence.slice(0, 120))}${s.sentence.length > 120 ? '…' : ''}</span>
            <span style="font-size:0.68rem;color:var(--slate-400);margin-left:0.4rem">(${Math.round((s.score ?? 0) * 100)}% · ${s.source ?? ''})</span>
          </div>
        </div>
      `).join('')}
    ` : ''}
  `;
}

function renderScorerBody(d) {
  const sc = d.scores ?? {};
  const finalPct = Math.round((d.final_confidence ?? 0) * 100);
  const fillClass = finalPct >= 70 ? 'fill-green' : finalPct >= 50 ? 'fill-amber' : 'fill-red';
  return `
    <div class="step-detail-grid">
      ${detailItem('Classifier score', `${Math.round((sc.classifier ?? 0) * 100)}%`)}
      ${detailItem('Grounding score',  `${Math.round((sc.grounding ?? 0) * 100)}%`)}
      ${detailItem('Novelty score',    `${Math.round((sc.novelty ?? 0) * 100)}%`)}
      ${detailItem('Final confidence', `${finalPct}%`)}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Final pipeline confidence</span><span>${finalPct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${finalPct}%"></div></div>
    </div>
    <div style="font-size:0.78rem;color:var(--slate-500);margin-top:0.5rem;padding:0.5rem;background:var(--slate-50);border-radius:6px;border:1px solid var(--slate-200)">
      <strong>Routing: ${escHtml(d.routing_label ?? d.routing ?? '—')}</strong><br/>
      ${escHtml(d.routing_reason ?? '')}
    </div>
  `;
}

// ─────────────────────────── RESULT CARD ───────────────────────────
function renderResult(data) {
  document.getElementById('resultCard').style.display = 'block';

  const cat  = capitalize(data.category  ?? '—');
  const pri  = capitalize(data.priority  ?? '—');
  document.getElementById('resultMeta').textContent = `Category: ${cat}  ·  Priority: ${pri}`;

  // Routing badge
  const rb = document.getElementById('routingBadge');
  const r  = data.routing ?? '';
  rb.textContent = data.routing_label ?? r;
  rb.className   = `routing-badge ${r === 'AUTO_APPROVE' ? 'auto' : r === 'LIGHT_REVIEW' ? 'light' : 'full'}`;

  // Confidence bar
  const pct = Math.round((data.final_confidence ?? 0) * 100);
  document.getElementById('confidenceValue').textContent = `${pct}%`;
  const fill = document.getElementById('confidenceFill');
  fill.style.width = `${pct}%`;
  fill.className = `confidence-bar-fill ${pct >= 70 ? 'fill-green' : pct >= 50 ? 'fill-amber' : 'fill-red'}`;

  // Routing reason
  document.getElementById('routingReason').textContent = data.routing_reason ?? '';

  // Grounding pills
  const pillsEl = document.getElementById('groundingPills');
  const gPct    = Math.round((data.grounding_score ?? 0) * 100);
  const gClass  = gPct >= 80 ? 'pill-green' : gPct >= 60 ? 'pill-amber' : 'pill-slate';
  pillsEl.innerHTML = `
    <span class="pill pill-blue">📋 ${data.claims_checked ?? 0} claims checked</span>
    <span class="pill ${gClass}">✓ ${gPct}% grounded</span>
    ${data.is_novel ? '<span class="pill pill-amber">⚠ Novel ticket</span>' : ''}
  `;

  // Response
  document.getElementById('responseBody').textContent = data.verified_response ?? '';

  // Scroll to result
  setTimeout(() => {
    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

// ─────────────────────────── STATS ───────────────────────────
function updateStats(data) {
  state.stats.total++;
  if (data.routing === 'AUTO_APPROVE') state.stats.auto++;
  else if (data.routing === 'LIGHT_REVIEW') state.stats.review++;
  else state.stats.escalated++;

  document.getElementById('statTotal').textContent     = state.stats.total;
  document.getElementById('statAuto').textContent      = state.stats.auto;
  document.getElementById('statReview').textContent    = state.stats.review;
  document.getElementById('statEscalated').textContent = state.stats.escalated;
  document.getElementById('sessionStats').style.display = 'flex';
}

// ─────────────────────────── QUEUE ───────────────────────────
function toggleQueue() {
  const drawer  = document.getElementById('queueDrawer');
  const overlay = document.getElementById('queueOverlay');
  const isOpen  = drawer.classList.contains('open');
  drawer.classList.toggle('open', !isOpen);
  overlay.classList.toggle('visible', !isOpen);
}

function sendToQueue() {
  if (!state.currentResult) return;

  const subject  = document.getElementById('ticketSubject').value.trim() || 'Untitled ticket';
  const body     = document.getElementById('ticketBody').value.trim();
  const customer = document.getElementById('ticketCustomer').value.trim();

  const item = {
    id:       `Q${String(state.queue.length + 1).padStart(3, '0')}`,
    subject,
    body,
    customer,
    routing:  state.currentResult.routing,
    category: state.currentResult.category,
    priority: state.currentResult.priority,
    confidence: state.currentResult.final_confidence,
    response: state.currentResult.verified_response,
    timestamp: new Date().toLocaleTimeString(),
  };

  state.queue.push(item);
  renderQueue();

  const badge = document.getElementById('queueCount');
  badge.textContent = state.queue.length;

  toggleQueue();
}

function renderQueue() {
  const list = document.getElementById('queueList');

  if (state.queue.length === 0) {
    list.innerHTML = '<div class="queue-empty">No tickets in queue</div>';
    return;
  }

  list.innerHTML = state.queue.slice().reverse().map(item => {
    const routingClass = item.routing === 'AUTO_APPROVE' ? 'badge-auto' : item.routing === 'LIGHT_REVIEW' ? 'badge-light' : 'badge-full';
    return `
      <div class="queue-ticket">
        <div class="queue-ticket-header">
          <div class="queue-ticket-subject">${escHtml(item.subject)}</div>
          <span class="step-badge ${routingClass}" style="font-size:0.65rem;white-space:nowrap">${escHtml(capitalize(item.routing?.replace('_', ' ') ?? ''))}</span>
        </div>
        <div class="queue-ticket-meta">${escHtml(item.id)} · ${escHtml(item.customer || 'Unknown')} · ${item.timestamp}</div>
        <div class="queue-ticket-preview">${escHtml(item.body)}</div>
        <div class="queue-ticket-footer">
          <span class="pill pill-blue" style="font-size:0.68rem">${capitalize(item.category ?? '—')}</span>
          <span class="pill pill-slate" style="font-size:0.68rem">${Math.round((item.confidence ?? 0) * 100)}% confidence</span>
        </div>
      </div>
    `;
  }).join('');
}

// ─────────────────────────── MODAL ───────────────────────────
function openModal(title, content) {
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalBody').innerHTML    = content;
  document.getElementById('modalOverlay').classList.add('visible');
  document.getElementById('agentModal').classList.add('visible');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('visible');
  document.getElementById('agentModal').classList.remove('visible');
}

// ─────────────────────────── UTILITIES ───────────────────────────
function resetPipeline() {
  document.getElementById('pipelineContainer').style.display = 'none';
  document.getElementById('resultCard').style.display        = 'none';
  document.getElementById('emptyState').style.display        = 'block';
  document.getElementById('sampleSelect').value              = '';
  document.getElementById('ticketSubject').value             = '';
  document.getElementById('ticketBody').value                = '';
  document.getElementById('ticketCustomer').value            = '';
  state.currentResult = null;
  state.stepData      = {};
}

function disableButton(disabled) {
  const btn = document.getElementById('btnProcess');
  btn.disabled = disabled;
  btn.innerHTML = disabled
    ? `<div class="spinner" style="width:16px;height:16px;border-color:rgba(255,255,255,0.3);border-top-color:#fff"></div> Processing…`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Pipeline`;
}

function copyResponse() {
  const text = document.getElementById('responseBody').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target.closest('button');
    const orig = btn.innerHTML;
    btn.innerHTML = '✓ Copied';
    setTimeout(() => { btn.innerHTML = orig; }, 1500);
  });
}

function showError(msg) {
  const container = document.getElementById('pipelineSteps');
  const errEl     = document.createElement('div');
  errEl.style.cssText = 'padding:1rem;background:#fee2e2;border:1px solid #fecaca;border-radius:8px;font-size:0.83rem;color:#b91c1c;margin-top:0.5rem';
  errEl.textContent   = `Error: ${msg}`;
  container.appendChild(errEl);
}

function shakeElement(el) {
  el.style.animation = 'none';
  el.offsetHeight;
  el.style.animation = 'shake 0.3s ease';
  el.style.borderColor = 'var(--red-600)';
  setTimeout(() => {
    el.style.borderColor = '';
  }, 1200);
}

function detailItem(key, val) {
  return `
    <div class="detail-item">
      <div class="detail-key">${escHtml(key)}</div>
      <div class="detail-val">${escHtml(String(val))}</div>
    </div>
  `;
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Inject shake keyframes
const style = document.createElement('style');
style.textContent = `
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    20%       { transform: translateX(-6px); }
    40%       { transform: translateX(6px); }
    60%       { transform: translateX(-4px); }
    80%       { transform: translateX(4px); }
  }
`;
document.head.appendChild(style);

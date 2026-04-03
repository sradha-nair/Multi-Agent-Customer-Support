'use strict';

// ─────────────────────────── STATE ───────────────────────────
const state = {
  sampleTickets: [],
  currentResult: null,
  queue: [],
  stats: { total: 0, auto: 0, review: 0, escalated: 0 },
  stepData: {},
  abortController: null,   // lets us cancel an in-flight request
};

const AGENT_META = {
  novelty_detector:  { num: 1, label: 'Novelty Detector',   verifier: false },
  classifier:        { num: 2, label: 'Classifier',          verifier: false },
  researcher:        { num: 3, label: 'Researcher (RAG)',    verifier: false },
  responder:         { num: 4, label: 'Responder',           verifier: false },
  grounding_checker: { num: 5, label: 'Grounding Checker',   verifier: true  },
  confidence_scorer: { num: 6, label: 'Confidence Scorer',   verifier: true  },
};

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
    } else if (data.llm === 'groq') {
      dot.className  = 'status-dot online';
      text.textContent = `Groq · ${data.models[0] ?? 'llama-3.1-8b'}`;
    } else {
      dot.className  = 'status-dot demo';
      text.textContent = 'Demo mode (no LLM configured)';
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
async function processTicket() {
  const subject  = document.getElementById('ticketSubject').value.trim();
  const body     = document.getElementById('ticketBody').value.trim();
  const plan     = document.getElementById('ticketPlan').value;

  if (!body) {
    shakeElement(document.getElementById('ticketBody'));
    return;
  }

  const ticket = subject ? `${subject}\n\n${body}` : body;

  // Reset UI
  document.getElementById('emptyState').style.display       = 'none';
  document.getElementById('pipelineContainer').style.display = 'block';
  document.getElementById('resultCard').style.display        = 'none';
  state.stepData = {};

  initPipelineSteps();
  disableButton(true);

  // Cancel any previous in-flight request
  if (state.abortController) state.abortController.abort();
  state.abortController = new AbortController();

  try {
    const response = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticket, plan }),
      signal: state.abortController.signal,
    });

    if (!response.ok) {
      showError(`Server error: ${response.status} ${response.statusText}`);
      disableButton(false);
      return;
    }

    // Parse SSE stream via ReadableStream
    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on newlines; keep any incomplete line in buffer
      const lines = buffer.split('\n');
      buffer = lines.pop();   // last element may be incomplete

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') { disableButton(false); return; }
        try {
          handlePipelineMessage(JSON.parse(raw));
        } catch {
          // Malformed JSON line — skip silently
        }
      }
    }

  } catch (err) {
    if (err.name !== 'AbortError') {
      showError(`Connection failed: ${err.message}. Is the server running?`);
    }
  } finally {
    disableButton(false);
  }
}

// ─────────────────────────── STEP STATE MANAGEMENT ───────────────────────────
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
          ${meta.verifier ? '<span class="verify-tag">VERIFY</span>' : ''}
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

function setStepState(key, state_) {
  const card = document.getElementById(`step-${key}`);
  if (!card) return;
  card.className = `step-card state-${state_}`;

  const statusWrap = card.querySelector('.step-status');
  const statusEl   = card.querySelector('.step-status-text');

  const existingSpinner = statusWrap.querySelector('.spinner');
  if (existingSpinner) existingSpinner.remove();

  if (state_ === 'running') {
    statusEl.textContent = 'Processing';
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    statusWrap.prepend(spinner);
  } else if (state_ === 'done')    { statusEl.textContent = 'Done'; }
  else if (state_ === 'flagged')   { statusEl.textContent = 'Flagged'; }
  else if (state_ === 'error')     { statusEl.textContent = 'Error'; }
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
    case 'novelty_detector':  body.innerHTML = renderNoveltyBody(data);   break;
    case 'classifier':        body.innerHTML = renderClassifierBody(data); break;
    case 'researcher':        body.innerHTML = renderResearcherBody(data); break;
    case 'responder':         body.innerHTML = renderResponderBody(data);  break;
    case 'grounding_checker': body.innerHTML = renderGroundingBody(data);  break;
    case 'confidence_scorer': body.innerHTML = renderScorerBody(data);     break;
  }

  // Auto-expand on completion
  const card = document.getElementById(`step-${key}`);
  if (card) card.classList.add('expanded');

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
    badge = `<span style="font-size:.72rem;color:var(--slate-400)">${data.elapsed_ms}ms</span>`;
  }

  const existing = statusWrap.querySelector('.step-badge, span[style]');
  if (existing) existing.remove();

  const statusText = statusWrap.querySelector('.step-status-text');
  if (badge) {
    const tmp = document.createElement('div');
    tmp.innerHTML = badge;
    statusWrap.insertBefore(tmp.firstChild, statusText);
    statusText.textContent = '';
  }
}

// ─────────────────────────── AGENT RENDERERS ───────────────────────────
function renderNoveltyBody(d) {
  const simPct = Math.round((d.similarity_to_known ?? 0) * 100);
  const fillClass = d.is_novel ? 'fill-red' : 'fill-green';
  return `
    <div class="step-detail-grid">
      ${detailItem('Result',     d.is_novel ? '⚠ Novel ticket' : '✓ Known pattern')}
      ${detailItem('Similarity', `${simPct}% to nearest`)}
      ${detailItem('Threshold',  `${Math.round((d.threshold ?? 0.45) * 100)}%`)}
      ${detailItem('Method',     d.method ?? '—')}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Similarity to known distribution</span><span>${simPct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${simPct}%"></div></div>
    </div>
    ${d.nearest_ticket ? `<p class="step-note">Nearest known: <em>${escHtml(d.nearest_ticket)}</em></p>` : ''}
    ${d.note ? `<p class="step-note">${escHtml(d.note)}</p>` : ''}
  `;
}

function renderClassifierBody(d) {
  const confPct = Math.round((d.confidence ?? 0) * 100);
  const fillClass = confPct >= 75 ? 'fill-green' : confPct >= 55 ? 'fill-amber' : 'fill-red';
  return `
    <div class="step-detail-grid">
      ${detailItem('Category',   capitalize(d.category ?? '—'))}
      ${detailItem('Priority',   capitalize(d.priority ?? '—'))}
      ${detailItem('Confidence', `${confPct}%`)}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Classifier confidence</span><span>${confPct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${confPct}%"></div></div>
    </div>
    ${d.reasoning ? `<p class="step-note">${escHtml(d.reasoning)}</p>` : ''}
  `;
}

function renderResearcherBody(d) {
  const articles = d.articles ?? [];
  return `
    <p class="step-note">Retrieved ${articles.length} KB article(s) using ${d.method ?? 'semantic'} search.</p>
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
    <p class="step-note">Draft response (pre-verification):</p>
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
      <p class="step-note" style="font-weight:600;margin-top:.6rem">Sentence-level grounding:</p>
      ${sentences.slice(0, 8).map(s => `
        <div class="grounding-row">
          <div class="grounding-dot ${s.grounded ? 'ok' : 'bad'}"></div>
          <div>
            <span style="color:var(--slate-600)">${escHtml(s.sentence.slice(0, 120))}${s.sentence.length > 120 ? '…' : ''}</span>
            <span style="font-size:.68rem;color:var(--slate-400);margin-left:.4rem">(${Math.round((s.score ?? 0) * 100)}% · ${s.source ?? ''})</span>
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
      ${detailItem('Classifier',   `${Math.round((sc.classifier ?? 0) * 100)}%`)}
      ${detailItem('Grounding',    `${Math.round((sc.grounding ?? 0) * 100)}%`)}
      ${detailItem('Novelty',      `${Math.round((sc.novelty ?? 0) * 100)}%`)}
      ${detailItem('Final conf.',  `${finalPct}%`)}
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Final pipeline confidence</span><span>${finalPct}%</span></div>
      <div class="score-bar-track"><div class="score-bar-fill ${fillClass}" style="width:${finalPct}%"></div></div>
    </div>
    <div class="step-note" style="padding:.55rem .7rem;background:var(--beige-50);border:1px solid var(--beige-200);border-radius:6px;margin-top:.5rem">
      <strong>Routing: ${escHtml(d.routing_label ?? d.routing ?? '—')}</strong><br/>
      ${escHtml(d.routing_reason ?? '')}
    </div>
  `;
}

// ─────────────────────────── RESULT CARD ───────────────────────────
function renderResult(data) {
  document.getElementById('resultCard').style.display = 'block';

  const cat = capitalize(data.category ?? '—');
  const pri = capitalize(data.priority  ?? '—');
  document.getElementById('resultMeta').textContent = `Category: ${cat}  ·  Priority: ${pri}`;

  const rb = document.getElementById('routingBadge');
  const r  = data.routing ?? '';
  rb.textContent = data.routing_label ?? r;
  rb.className   = `routing-badge ${r === 'AUTO_APPROVE' ? 'auto' : r === 'LIGHT_REVIEW' ? 'light' : 'full'}`;

  const pct = Math.round((data.final_confidence ?? 0) * 100);
  document.getElementById('confidenceValue').textContent = `${pct}%`;
  const fill = document.getElementById('confidenceFill');
  fill.style.width = `${pct}%`;
  fill.className   = `confidence-bar-fill ${pct >= 70 ? 'fill-green' : pct >= 50 ? 'fill-amber' : 'fill-red'}`;

  document.getElementById('routingReason').textContent = data.routing_reason ?? '';

  const gPct   = Math.round((data.grounding_score ?? 0) * 100);
  const gClass = gPct >= 80 ? 'pill-green' : gPct >= 60 ? 'pill-amber' : 'pill-slate';
  document.getElementById('groundingPills').innerHTML = `
    <span class="pill pill-blue">📋 ${data.claims_checked ?? 0} claims checked</span>
    <span class="pill ${gClass}">✓ ${gPct}% grounded</span>
    ${data.is_novel ? '<span class="pill pill-amber">⚠ Novel ticket</span>' : ''}
  `;

  document.getElementById('responseBody').textContent = data.verified_response ?? '';

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

  state.queue.push({
    id:         `Q${String(state.queue.length + 1).padStart(3, '0')}`,
    subject, body, customer,
    routing:    state.currentResult.routing,
    category:   state.currentResult.category,
    priority:   state.currentResult.priority,
    confidence: state.currentResult.final_confidence,
    response:   state.currentResult.verified_response,
    timestamp:  new Date().toLocaleTimeString(),
  });

  document.getElementById('queueCount').textContent = state.queue.length;
  renderQueue();
  toggleQueue();
}

function renderQueue() {
  const list = document.getElementById('queueList');
  if (state.queue.length === 0) {
    list.innerHTML = '<div class="queue-empty">No tickets in queue</div>';
    return;
  }
  list.innerHTML = state.queue.slice().reverse().map(item => {
    const cls = item.routing === 'AUTO_APPROVE' ? 'badge-auto' : item.routing === 'LIGHT_REVIEW' ? 'badge-light' : 'badge-full';
    return `
      <div class="queue-ticket">
        <div class="queue-ticket-header">
          <div class="queue-ticket-subject">${escHtml(item.subject)}</div>
          <span class="step-badge ${cls}" style="font-size:.65rem;white-space:nowrap">${escHtml(capitalize(item.routing?.replace('_', ' ') ?? ''))}</span>
        </div>
        <div class="queue-ticket-meta">${escHtml(item.id)} · ${escHtml(item.customer || 'Unknown')} · ${item.timestamp}</div>
        <div class="queue-ticket-preview">${escHtml(item.body)}</div>
        <div class="queue-ticket-footer">
          <span class="pill pill-blue" style="font-size:.68rem">${capitalize(item.category ?? '—')}</span>
          <span class="pill pill-slate" style="font-size:.68rem">${Math.round((item.confidence ?? 0) * 100)}% confidence</span>
        </div>
      </div>`;
  }).join('');
}

// ─────────────────────────── MODAL ───────────────────────────
function closeModal() {
  document.getElementById('modalOverlay').classList.remove('visible');
  document.getElementById('agentModal').classList.remove('visible');
}

// ─────────────────────────── UTILITIES ───────────────────────────
function resetPipeline() {
  if (state.abortController) state.abortController.abort();
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
    ? `<div class="spinner" style="width:16px;height:16px;border-color:rgba(255,255,255,.3);border-top-color:#fff"></div> Processing…`
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
  const errEl = document.createElement('div');
  errEl.style.cssText = 'padding:1rem;background:#fee2e2;border:1px solid #fecaca;border-radius:8px;font-size:.83rem;color:#b91c1c;margin-top:.5rem';
  errEl.textContent   = `Error: ${msg}`;
  container.appendChild(errEl);
}

function shakeElement(el) {
  el.style.animation  = 'none';
  el.offsetHeight;
  el.style.animation  = 'shake .3s ease';
  el.style.borderColor = 'var(--red-600)';
  setTimeout(() => { el.style.borderColor = ''; }, 1200);
}

function detailItem(key, val) {
  return `
    <div class="detail-item">
      <div class="detail-key">${escHtml(key)}</div>
      <div class="detail-val">${escHtml(String(val))}</div>
    </div>`;
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Inject keyframe animations
const style = document.createElement('style');
style.textContent = `
  @keyframes shake {
    0%,100% { transform:translateX(0); }
    20%      { transform:translateX(-6px); }
    40%      { transform:translateX(6px); }
    60%      { transform:translateX(-4px); }
    80%      { transform:translateX(4px); }
  }
  .verify-tag {
    font-size:.63rem; background:var(--amber-100); color:var(--amber-700);
    border-radius:10px; padding:1px 6px; margin-right:4px; font-weight:600;
  }
  .step-note {
    font-size:.78rem; color:var(--slate-500); margin-top:.4rem; line-height:1.5;
  }
`;
document.head.appendChild(style);

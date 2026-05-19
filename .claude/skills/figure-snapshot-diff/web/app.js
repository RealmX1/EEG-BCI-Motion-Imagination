// Paper Figure History — vanilla JS frontend.
// Polls /api/index every 2s for staging changes; loads /api/figures/<fig_id>
// on click; renders a slider where the user can pick any two versions to
// compare; provides Accept/Reject on each staging entry.

const POLL_MS = 2000;

const state = {
  index: { figures: {}, last_updated: null },
  selectedFigId: null,
  currentManifest: null,
  currentComments: [],   // full comments list for selected fig
  beforeChoice: null,
  afterChoice: null,
  // Set by pickDefaultChoices() when it had to skip byte-identical predecessors
  // to find a meaningful Before for the auto-selected pair. Cleared on any
  // user-driven choice change (dropdown / history-modal jump). Shape:
  //   null  — natural default (Before is the immediate predecessor of After)
  //   { mode: 'skipped', skippedCount, naturalChoice, actualChoice }
  //   { mode: 'no_different', skippedCount }
  autoSkipInfo: null,
  pollTimer: null,
  historyModalOpen: false,
  historyFilter: "all",  // "all" | "on_pair" | "other_pairs" | "history"
};

// -----------------------------------------------------------------------
// API helpers
// -----------------------------------------------------------------------

async function apiGet(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error(`GET ${path} → ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
    cache: "no-store",
  });
  if (!resp.ok) throw new Error(`POST ${path} → ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

function imageUrl(figId, filename) {
  // server enforces path safety; trim leading "_staging/" already part of filename
  return `/api/images/${encodeURIComponent(figId)}/${filename
    .split("/").map(encodeURIComponent).join("/")}`;
}

// -----------------------------------------------------------------------
// Comment draft persistence (localStorage, keyed by (fig_id, before_sha,
// after_sha)). Survives pair switch, fig switch, browser reload, etc.
// Cleared only on successful submit of that specific pair.
// -----------------------------------------------------------------------
function draftKey(figId, beforeSha, afterSha) {
  return `figdiff:comment_draft:${figId}:${beforeSha || "_"}:${afterSha || "_"}`;
}
function loadDraft(figId, beforeSha, afterSha) {
  if (!figId) return "";
  try { return localStorage.getItem(draftKey(figId, beforeSha, afterSha)) || ""; }
  catch { return ""; }
}
function saveDraft(figId, beforeSha, afterSha, text) {
  if (!figId) return;
  try {
    const key = draftKey(figId, beforeSha, afterSha);
    if (text && text.length > 0) localStorage.setItem(key, text);
    else localStorage.removeItem(key);
  } catch {}
}
function clearDraft(figId, beforeSha, afterSha) {
  if (!figId) return;
  try { localStorage.removeItem(draftKey(figId, beforeSha, afterSha)); } catch {}
}

// -----------------------------------------------------------------------
// UI rendering
// -----------------------------------------------------------------------

function setStatus(text, kind = "") {
  const el = document.getElementById("status-text");
  el.textContent = text;
  el.className = kind;
}

function showToast(msg, isError = false) {
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.className = `toast show${isError ? " error" : ""}`;
  setTimeout(() => { toast.className = "toast"; }, 3500);
}

function renderSidebar() {
  const ul = document.getElementById("fig-list");
  ul.innerHTML = "";
  const figs = state.index.figures || {};
  // Maintain order from /api/index
  for (const figId of Object.keys(figs)) {
    const f = figs[figId];
    const li = document.createElement("li");
    if (figId === state.selectedFigId) li.classList.add("active");
    li.dataset.figId = figId;
    const commentN = f.comment_open_n || 0;
    li.innerHTML = `
      <div class="li-text">
        <div class="label">${escapeHtml(f.paper_label || figId)}</div>
        <div class="caption">${escapeHtml((f.caption || "").slice(0, 44))}${(f.caption || "").length > 44 ? "…" : ""}</div>
      </div>
      <div class="li-badges">
        <span class="badge stage-badge ${f.staging_n > 0 ? "" : "zero"}" title="${f.staging_n} pending staging">${f.staging_n}</span>
        <span class="badge comment-badge ${commentN > 0 ? "" : "zero"}" title="${commentN} open comments">${commentN === 0 ? "·" : "💬" + commentN}</span>
      </div>
    `;
    li.addEventListener("click", () => selectFigure(figId));
    ul.appendChild(li);
  }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtBytes(n) {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function renderMain() {
  const main = document.getElementById("main-area");
  const m = state.currentManifest;
  if (!m) {
    main.innerHTML = `<div class="empty-state">Select a figure on the left to view its version chain.</div>`;
    return;
  }
  const trunk = m.trunk || [];
  const staging = m.staging || [];
  const rejected = m.rejected || [];

  const beforeOptions = buildOptions(trunk, staging, rejected, state.beforeChoice);
  const afterOptions  = buildOptions(trunk, staging, rejected, state.afterChoice);

  const secondaryHtml = (m.secondary_generators || []).map(s =>
    `<li><strong>${escapeHtml(s.label)}</strong>: <code>${escapeHtml(s.command)}</code></li>`
  ).join("");

  // Build the unified Version History table: trunk (accepted, normal style) +
  // rejected (greyed, REJECTED tag). Sort by created_at descending so newest
  // entries are at top — tip is always first among accepted.
  // Rejected entries keep their rejected_id (r1, r2, …) instead of a v-prefixed
  // version label since they were never accepted.
  const versionRows = [];
  trunk.slice().reverse().forEach((t, ridx) => {
    const isTrunkTip = ridx === 0;
    versionRows.push(`
      <tr class="${isTrunkTip ? "tip" : ""}">
        <td><strong>${escapeHtml(t.version)}</strong>${isTrunkTip ? " ← tip" : ""}</td>
        <td><span class="version-tag tag-accepted" title="In trunk — paper draft cites this">accepted</span></td>
        <td>${escapeHtml(t.tag || "")}</td>
        <td>${escapeHtml((t.created_at || "").slice(0, 16).replace("T", " "))}</td>
        <td><code title="${escapeHtml(t.sha256 || "")}">${escapeHtml((t.sha256 || "").slice(0, 8))}…</code></td>
      </tr>`);
  });
  rejected.slice().reverse().forEach((r) => {
    versionRows.push(`
      <tr class="rejected-row" title="Rejected — still available for compare/comment.${r.rejected_reason ? "\nReason: " + r.rejected_reason.replace(/"/g, "&quot;") : ""}">
        <td><strong>${escapeHtml(r.rejected_id)}</strong></td>
        <td><span class="version-tag tag-rejected">rejected</span></td>
        <td>${escapeHtml(r.tag || "")}</td>
        <td>${escapeHtml((r.rejected_at || r.created_at || "").slice(0, 16).replace("T", " "))}</td>
        <td><code title="${escapeHtml(r.sha256 || "")}">${escapeHtml((r.sha256 || "").slice(0, 8))}…</code></td>
      </tr>`);
  });

  main.innerHTML = `
    <div class="col-info">
      <div class="fig-header">
        <h2>${escapeHtml(m.paper_label || m.fig_id)}</h2>
        <div class="caption">${escapeHtml(m.caption || "")}</div>
        <div class="meta">canonical: <code>${escapeHtml(m.canonical_output_path || "")}</code></div>
        <details>
          <summary style="cursor:pointer">Generator command${(m.secondary_generators || []).length ? " (and secondaries)" : ""}</summary>
          <div style="padding: 0.3rem 0">
            <strong>Primary:</strong> <code>${escapeHtml((m.generator || {}).command || "")}</code>
            ${secondaryHtml ? `<ul style="margin-top:0.3rem;padding-left:1.1rem">${secondaryHtml}</ul>` : ""}
          </div>
        </details>
      </div>

      <div class="staging-section">
        <h3>Staging — pending decision (${staging.length})</h3>
        ${staging.length === 0 ? `<div class="staging-empty">No staged candidates. Use ← / → or click another figure to continue.</div>` :
          staging.map(e => renderStagingEntry(m.fig_id, e)).join("")}
      </div>

      <div id="comments-area" class="comments-area"></div>

      <div class="trunk-section">
        <h3>Version history (${trunk.length} accepted${rejected.length ? ", " + rejected.length + " rejected" : ""})</h3>
        <div class="trunk-scroll">
          <table class="trunk-table">
            <thead><tr>
              <th>ID</th><th>Status</th><th>Tag</th><th>When</th><th>SHA</th>
            </tr></thead>
            <tbody>
              ${versionRows.join("")}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="col-compare">
      <div class="compare-controls">
        <div>
          <label>Before</label>
          <select class="before-select" id="before-select">${beforeOptions}</select>
        </div>
        <div>
          <label>After</label>
          <select class="after-select" id="after-select">${afterOptions}</select>
        </div>
      </div>
      <div id="slider-host"></div>
    </div>
  `;

  // Wire up dropdowns
  const bSel = document.getElementById("before-select");
  const aSel = document.getElementById("after-select");
  bSel.value = state.beforeChoice || bSel.value;
  aSel.value = state.afterChoice || aSel.value;
  state.beforeChoice = bSel.value;
  state.afterChoice = aSel.value;
  bSel.addEventListener("change", () => {
    state.beforeChoice = bSel.value;
    // User overrode the default selection — clear the skip banner.
    state.autoSkipInfo = null;
    renderSlider();
    renderCommentsArea();
  });
  aSel.addEventListener("change", () => {
    state.afterChoice = aSel.value;
    state.autoSkipInfo = null;
    renderSlider();
    renderCommentsArea();
  });

  // Wire up accept/reject
  for (const btn of document.querySelectorAll("[data-action]")) {
    btn.addEventListener("click", () => handleAction(btn.dataset.figId, btn.dataset.stagedId, btn.dataset.action));
  }

  renderSlider();
  renderCommentsArea();
}

function labelForChoice(choice) {
  const m = state.currentManifest;
  if (!m || !choice) return "";
  if (choice.startsWith("trunk:")) {
    const t = (m.trunk || []).find(x => x.version === choice.slice(6));
    if (!t) return choice;
    return t.tag ? `${t.version} (${t.tag})` : t.version;
  }
  if (choice.startsWith("staging:")) {
    const e = (m.staging || []).find(x => x.staged_id === choice.slice(8));
    if (!e) return choice;
    return e.tag ? `${e.staged_id} (${e.tag})` : e.staged_id;
  }
  if (choice.startsWith("rejected:")) {
    const e = (m.rejected || []).find(x => x.rejected_id === choice.slice(9));
    if (!e) return choice;
    return e.tag ? `${e.rejected_id} (${e.tag}, REJECTED)` : `${e.rejected_id} (REJECTED)`;
  }
  return choice;
}

function renderCommentEntry(c, fromOtherPair = false) {
  const statusBadge = `<span class="status-badge status-${c.status}">${escapeHtml(c.status)}</span>`;
  const supersedes = c.supersedes_comment_ids || [];
  const supersedesLine = supersedes.length
    ? `<div class="comment-supersedes">↑ Supersedes ${supersedes.map(s => `<code>${escapeHtml(s)}</code>`).join(", ")}</div>`
    : "";
  let resolutionLine = "";
  if (c.status === "claimed") {
    resolutionLine = `<div class="resolution">Claimed by ${escapeHtml(c.claimed_by || "")} at ${escapeHtml((c.claimed_at || "").slice(0, 16).replace("T", " "))}</div>`;
  } else if (c.status === "resolved") {
    if (c.resolved_by_comment_id) {
      resolutionLine = `<div class="resolution superseded">⤳ Superseded by <code>${escapeHtml(c.resolved_by_comment_id)}</code>${c.resolution_notes ? " · " + escapeHtml(c.resolution_notes) : ""}</div>`;
    } else {
      resolutionLine = `<div class="resolution">Resolved by ${escapeHtml(c.resolved_by_staged_id || "(manual)")}${c.resolution_notes ? " · " + escapeHtml(c.resolution_notes) : ""}</div>`;
    }
  }
  return `
    <div class="comment-entry status-${c.status}">
      <div class="comment-header">
        <strong>${escapeHtml(c.id)}</strong>
        ${statusBadge}
        <span class="author">${escapeHtml(c.author || "")}</span>
        <span class="when">${escapeHtml((c.created_at || "").slice(0, 16).replace("T", " "))}</span>
      </div>
      ${fromOtherPair ? `<div class="pair-ref-inline">on: <code>${escapeHtml(c.before_label || c.before_sha.slice(0, 8))} → ${escapeHtml(c.after_label || c.after_sha.slice(0, 8))}</code></div>` : ""}
      <div class="comment-text">${escapeHtml(c.text)}</div>
      ${supersedesLine}
      ${resolutionLine}
    </div>
  `;
}

function renderCommentsArea() {
  const el = document.getElementById("comments-area");
  if (!el || !state.currentManifest) return;
  const m = state.currentManifest;
  const beforeSha = shaForChoice(m, state.beforeChoice);
  const afterSha = shaForChoice(m, state.afterChoice);
  const beforeLabel = labelForChoice(state.beforeChoice);
  const afterLabel = labelForChoice(state.afterChoice);

  // Bucket comments by relevance:
  // - matching:    on the currently-selected pair (all statuses, all kinds — there
  //                are no kinds anymore; this is the primary thing user sees)
  // - otherActive: open|claimed comments on OTHER pairs (collapsible)
  // - history:     resolved / superseded comments (collapsible audit trail)
  const matching = state.currentComments.filter(
    c => c.before_sha === beforeSha && c.after_sha === afterSha
  );
  const otherActive = state.currentComments.filter(
    c => !(c.before_sha === beforeSha && c.after_sha === afterSha)
      && (c.status === "open" || c.status === "claimed")
  );
  const history = state.currentComments.filter(c => c.status === "resolved");

  // Supersedes picker: any active (open|claimed) comment is a candidate. The
  // submitter explicitly marks "this new comment retracts c1, c2" when intent
  // genuinely conflicts. Default behavior is ADDITIVE — new comments don't
  // implicitly override anything.
  const supersedesCandidates = state.currentComments.filter(
    c => c.status === "open" || c.status === "claimed"
  );

  const supersedesPicker = supersedesCandidates.length ? `
    <details class="supersedes-picker">
      <summary>Supersede earlier (${supersedesCandidates.length} active) — optional</summary>
      <div class="supersedes-list">
        ${supersedesCandidates.map(c => `
          <label class="supersedes-option">
            <input type="checkbox" name="supersedes" value="${escapeHtml(c.id)}" />
            <code>${escapeHtml(c.id)}</code>
            <span class="supersedes-text">${escapeHtml(c.text.slice(0, 100))}${c.text.length > 100 ? "…" : ""}</span>
          </label>
        `).join("")}
      </div>
    </details>` : "";

  el.innerHTML = `
    <div class="comments-header">
      <h3>Comments</h3>
      <span class="pair-ref" title="Currently-selected pair (this comment will be keyed to its SHA-256)">
        <code>${escapeHtml(beforeLabel)}</code> → <code>${escapeHtml(afterLabel)}</code>
      </span>
    </div>

    <div class="comments-subsection">
      <div class="subsection-label">On this pair (${matching.length})</div>
      <div class="comments-list">
        ${matching.length
          ? matching.map(c => renderCommentEntry(c, false)).join("")
          : `<div class="empty-state-mini">No comments on this pair yet.</div>`}
      </div>
    </div>

    <form id="comment-form" class="comment-form">
      <textarea id="comment-text" placeholder="Free-form. Both critique of this pair AND requests for future changes welcome in the same comment — the body is yours. Comment is keyed by SHA of the current pair; it survives accept/reject and version renames. Agents reading this figure's context-bundle will see it as 'must address'."></textarea>
      ${supersedesPicker}
      <div class="comment-form-actions">
        <div class="left-actions">
          <button type="button" id="copy-ref-btn" class="btn btn-secondary" title="Copy a markdown reference of the currently-compared pair (fig_id, version IDs, SHA, file paths, exploration commands) to clipboard — paste into an agent chat for deeper discussion.">📎 Copy ref</button>
          <button type="button" id="history-btn" class="btn btn-secondary" title="Open all comments on this figure, grouped by pair. Click a pair to jump to it in the slider.">📜 History (${state.currentComments.length})</button>
          <span class="form-hint">Ctrl/⌘+Enter to submit</span>
        </div>
        <button type="submit" class="btn">Add comment</button>
      </div>
    </form>

    ${otherActive.length ? `
      <details class="other-comments">
        <summary>${otherActive.length} other active comment${otherActive.length === 1 ? "" : "s"} on other pairs</summary>
        <div class="other-list">${otherActive.map(c => renderCommentEntry(c, true)).join("")}</div>
      </details>` : ""}

    ${history.length ? `
      <details class="other-comments history">
        <summary>${history.length} resolved / superseded (history)</summary>
        <div class="other-list">${history.map(c => renderCommentEntry(c, true)).join("")}</div>
      </details>` : ""}
  `;

  const form = document.getElementById("comment-form");
  const ta = document.getElementById("comment-text");
  if (ta) {
    // Restore draft for this specific (fig, before, after) triplet, if any.
    // Survives pair-switch, fig-switch, and browser reload.
    const restored = loadDraft(m.fig_id, beforeSha, afterSha);
    if (restored) ta.value = restored;
    ta.addEventListener("input", () => {
      // Sync save on each keystroke — localStorage writes are < 1ms for text of
      // this size, and sync ensures we never lose data when the user switches
      // dropdowns (which immediately destroys this textarea via re-render).
      saveDraft(m.fig_id, beforeSha, afterSha, ta.value);
    });
  }
  if (form) {
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const supersedes = Array.from(
        form.querySelectorAll("input[name='supersedes']:checked")
      ).map(cb => cb.value);
      submitComment(ta.value, supersedes);
    });
    ta.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        const supersedes = Array.from(
          form.querySelectorAll("input[name='supersedes']:checked")
        ).map(cb => cb.value);
        submitComment(ta.value, supersedes);
      }
    });
  }
  const copyRefBtn = document.getElementById("copy-ref-btn");
  if (copyRefBtn) copyRefBtn.addEventListener("click", () => copyPairReference());
  const historyBtn = document.getElementById("history-btn");
  if (historyBtn) historyBtn.addEventListener("click", () => openHistoryModal());
}

async function submitComment(text, supersedes = []) {
  text = (text || "").trim();
  if (!text) {
    showToast("Comment text required", true);
    return;
  }
  const m = state.currentManifest;
  const beforeSha = shaForChoice(m, state.beforeChoice);
  const afterSha = shaForChoice(m, state.afterChoice);
  if (!beforeSha || !afterSha) {
    showToast("Pick Before and After versions first", true);
    return;
  }
  try {
    const resp = await apiPost(`/api/figures/${encodeURIComponent(m.fig_id)}/comments`, {
      before_sha: beforeSha,
      after_sha: afterSha,
      before_label: labelForChoice(state.beforeChoice),
      after_label: labelForChoice(state.afterChoice),
      text,
      supersedes: supersedes.length ? supersedes : undefined,
    });
    // Successfully submitted — drop the draft for this pair so future visits
    // don't restore a stale copy of what was just sent.
    clearDraft(m.fig_id, beforeSha, afterSha);
    state.currentComments = resp.comments || [];
    renderCommentsArea();
    const supersedesMsg = supersedes.length ? ` (superseded ${supersedes.join(", ")})` : "";
    showToast(`Comment ${resp.entry.id} saved${supersedesMsg}`);
    await refreshIndex();
  } catch (e) {
    showToast(`Comment add failed: ${e.message}`, true);
  }
}

function buildOptions(trunk, staging, rejected, selected) {
  const parts = [];
  parts.push("<optgroup label='Trunk (accepted)'>");
  for (const t of trunk) {
    parts.push(`<option value="trunk:${t.version}"${selected === `trunk:${t.version}` ? " selected" : ""}>${escapeHtml(t.version)} — ${escapeHtml(t.tag || "")} (${escapeHtml((t.created_at || "").slice(0, 10))})</option>`);
  }
  parts.push("</optgroup>");
  if (staging.length) {
    parts.push("<optgroup label='Staging (pending decision)'>");
    for (const e of staging) {
      parts.push(`<option value="staging:${e.staged_id}"${selected === `staging:${e.staged_id}` ? " selected" : ""}>${escapeHtml(e.staged_id)} — ${escapeHtml(e.tag || "")} (${escapeHtml((e.created_at || "").slice(0, 10))})</option>`);
    }
    parts.push("</optgroup>");
  }
  if (rejected.length) {
    // Greyed via CSS option:disabled-look (but options stay selectable).
    parts.push("<optgroup class='rejected-optgroup' label='Rejected (kept for compare/learning)'>");
    for (const e of rejected) {
      parts.push(`<option class="rejected-option" value="rejected:${e.rejected_id}"${selected === `rejected:${e.rejected_id}` ? " selected" : ""}>${escapeHtml(e.rejected_id)} — ${escapeHtml(e.tag || "")} (${escapeHtml((e.rejected_at || e.created_at || "").slice(0, 10))})</option>`);
    }
    parts.push("</optgroup>");
  }
  return parts.join("");
}

function resolveChoiceToFilename(choice) {
  if (!choice) return null;
  const m = state.currentManifest;
  if (choice.startsWith("trunk:")) {
    const ver = choice.slice(6);
    const t = (m.trunk || []).find(x => x.version === ver);
    return t ? t.filename : null;
  }
  if (choice.startsWith("staging:")) {
    const sid = choice.slice(8);
    const e = (m.staging || []).find(x => x.staged_id === sid);
    return e ? e.filename : null;
  }
  if (choice.startsWith("rejected:")) {
    const rid = choice.slice(9);
    const e = (m.rejected || []).find(x => x.rejected_id === rid);
    return e ? e.filename : null;
  }
  return null;
}

function renderAutoSkipBanner(info) {
  // Banner shown at the top of the comparison area when pickDefaultChoices()
  // had to skip byte-identical predecessors (or found none) to pick a default
  // Before. Purely informational — never blocks rendering.
  if (!info) return "";
  if (info.mode === "skipped") {
    return `
      <div class="auto-skip-warning">
        ⓘ 默认 Before 为 <code>${escapeHtml(labelForChoice(info.actualChoice) || info.actualChoice)}</code>，
        跳过了 ${info.skippedCount} 个与 After 字节相同的中间版本
        （相邻前一版 <code>${escapeHtml(labelForChoice(info.naturalChoice) || info.naturalChoice)}</code> 与 After 内容相同）。
      </div>
    `;
  }
  if (info.mode === "no_different") {
    return `
      <div class="auto-skip-warning">
        ⓘ 该图共有 ${info.skippedCount + 1} 个版本，但全部字节相同。
        无可对比的差异版本，仅显示单图。
      </div>
    `;
  }
  return "";
}

function renderSlider() {
  const host = document.getElementById("slider-host");
  if (!host) return;
  const m = state.currentManifest;
  const beforeFn = resolveChoiceToFilename(state.beforeChoice);
  const afterFn  = resolveChoiceToFilename(state.afterChoice);
  if (!beforeFn || !afterFn) {
    host.innerHTML = `<div class="identical-warning">Pick Before and After versions above.</div>`;
    return;
  }
  const topBanner = renderAutoSkipBanner(state.autoSkipInfo);
  // Identical-image detection: degrade to single-image render instead of
  // blocking. Slider has nothing to wipe between when both halves are the
  // same bytes; users still want to see what the image looks like.
  const beforeSha = shaForChoice(m, state.beforeChoice);
  const afterSha  = shaForChoice(m, state.afterChoice);
  if (beforeSha && afterSha && beforeSha === afterSha) {
    const url = imageUrl(m.fig_id, afterFn);
    // When the top banner already explains "all versions are byte-identical",
    // suppress the inline warning to avoid showing two messages saying the
    // same thing. The inline warning is still useful when a manual selection
    // (or skipped-default with multiple distinct versions) lands on identical
    // bytes, because the top banner won't cover that case.
    const suppressInline = state.autoSkipInfo && state.autoSkipInfo.mode === "no_different";
    const inlineWarning = suppressInline ? "" : `
      <div class="identical-warning">
        <strong>⚠ Byte-identical</strong> — Before 与 After SHA-256 相同，
        仅显示单张图像（无对比滑块）。要看对比，请在上方下拉框选择不同版本。
      </div>
    `;
    host.innerHTML = `
      ${topBanner}
      ${inlineWarning}
      <div class="single-image-wrap">
        <img src="${url}" alt="version preview" />
      </div>
    `;
    // Mirror slider-wrap's image-load aspect-ratio handling so the box doesn't
    // collapse to the 4/3 fallback for tall/wide figures.
    const wrap = host.querySelector(".single-image-wrap");
    const img = wrap ? wrap.querySelector("img") : null;
    if (wrap && img) {
      const apply = () => {
        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
          wrap.style.aspectRatio = String(img.naturalWidth / img.naturalHeight);
        }
      };
      if (img.complete) apply();
      else {
        img.addEventListener("load", apply, { once: true });
        img.addEventListener("error", apply, { once: true });
      }
    }
    return;
  }

  const beforeUrl = imageUrl(m.fig_id, beforeFn);
  const afterUrl  = imageUrl(m.fig_id, afterFn);
  host.innerHTML = `
    ${topBanner}
    <div class="slider-wrap">
      <span class="badge-overlay before">BEFORE</span>
      <span class="badge-overlay after">AFTER</span>
      <img-comparison-slider>
        <img slot="first"  src="${beforeUrl}" alt="before" />
        <img slot="second" src="${afterUrl}"  alt="after" />
        <svg slot="handle" xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="-15 -15 30 30">
          <circle r="13" fill="#ff6b35" stroke="white" stroke-width="2"/>
          <path d="M-7 -4 L-2 0 L-7 4 Z M7 -4 L2 0 L7 4 Z" fill="white"/>
        </svg>
      </img-comparison-slider>
    </div>
  `;

  // Once both images load, lock slider-wrap aspect-ratio to the *taller*
  // image's natural aspect (i.e. smaller w/h). With both images set to
  // 100%×100% + object-fit:contain inside, this guarantees the taller image
  // fills the wrap (no bottom truncation) and the shorter image letterboxes
  // horizontally — matching what build_compare_page.py does server-side via
  // PIL. Without this, the library's shadow-DOM defaults pin the second image
  // to width:100% height:auto top:0, causing top-align + bottom overflow when
  // the two image aspect ratios differ.
  const wrap = host.querySelector(".slider-wrap");
  if (wrap) {
    const imgs = wrap.querySelectorAll("img-comparison-slider img");
    const aspects = [];
    let pending = imgs.length;
    const finalize = () => {
      if (aspects.length === 0) return;
      // Smaller w/h = taller image; that aspect ensures both fit.
      wrap.style.aspectRatio = String(Math.min(...aspects));
    };
    imgs.forEach((img) => {
      const onResolved = () => {
        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
          aspects.push(img.naturalWidth / img.naturalHeight);
        }
        if (--pending === 0) finalize();
      };
      if (img.complete) {
        onResolved();
      } else {
        img.addEventListener("load", onResolved, { once: true });
        img.addEventListener("error", onResolved, { once: true });
      }
    });
  }
}

function shaForChoice(m, choice) {
  if (!choice) return null;
  if (choice.startsWith("trunk:")) {
    const t = (m.trunk || []).find(x => x.version === choice.slice(6));
    return t ? t.sha256 : null;
  }
  if (choice.startsWith("staging:")) {
    const e = (m.staging || []).find(x => x.staged_id === choice.slice(8));
    return e ? e.sha256 : null;
  }
  if (choice.startsWith("rejected:")) {
    const e = (m.rejected || []).find(x => x.rejected_id === choice.slice(9));
    return e ? e.sha256 : null;
  }
  return null;
}

function renderStagingEntry(figId, e) {
  return `
    <div class="staging-entry">
      <div class="info">
        <div><strong>${escapeHtml(e.staged_id)}</strong> — tag: ${escapeHtml(e.tag || "")}</div>
        <div>created: ${escapeHtml((e.created_at || "").slice(0, 19).replace("T", " "))} · proposed by: ${escapeHtml(e.proposed_by || "")}</div>
        ${e.source_cmd ? `<div>cmd: <code>${escapeHtml(e.source_cmd)}</code></div>` : ""}
        ${e.notes ? `<div style="color:#666; font-style:italic; margin-top:0.2rem">${escapeHtml(e.notes)}</div>` : ""}
        <div style="color:#888; font-size:0.8rem; margin-top:0.2rem">sha-256: <code>${escapeHtml((e.sha256 || "").slice(0, 16))}…</code></div>
      </div>
      <div class="actions">
        <button class="btn" data-fig-id="${escapeHtml(figId)}" data-staged-id="${escapeHtml(e.staged_id)}" data-action="accept">Accept</button>
        <button class="btn btn-danger" data-fig-id="${escapeHtml(figId)}" data-staged-id="${escapeHtml(e.staged_id)}" data-action="reject">Reject</button>
      </div>
    </div>
  `;
}

async function handleAction(figId, stagedId, action) {
  // Disable all buttons during request
  for (const b of document.querySelectorAll("[data-action]")) b.disabled = true;
  try {
    const resp = await apiPost(`/api/figures/${figId}/${action}`, { staged_id: stagedId });
    state.currentManifest = resp.manifest;
    const stagingLeft = (resp.manifest.staging || []).length;
    const verb = action === "accept" ? "Accepted" : "Rejected";
    // Reset choices (the staging entry is no longer in `staging[]`; for reject
    // it has moved to `rejected[]` and is still selectable via that optgroup,
    // for accept it has moved to `trunk[]` as the new tip).
    state.beforeChoice = null;
    state.afterChoice = null;
    pickDefaultChoices();
    renderMain();
    await refreshIndex();
    const tailMsg = action === "reject"
      ? ` (kept as ${resp.entry.rejected_id} for compare/comment)`
      : "";
    // Auto-advance when current figure has no more pending staging
    if (stagingLeft === 0) {
      const next = findNextPendingFigId(figId);
      if (next) {
        showToast(`${verb} ${figId}/${stagedId}${tailMsg} · auto-advanced to ${next}`);
        await selectFigure(next);
      } else {
        showToast(`${verb} ${figId}/${stagedId}${tailMsg} · all figures reviewed! 🎉`);
      }
    } else {
      showToast(`${verb} ${figId}/${stagedId}${tailMsg} · ${stagingLeft} staging entry remaining on this figure`);
    }
  } catch (e) {
    showToast(`${action} failed: ${e.message}`, true);
  } finally {
    for (const b of document.querySelectorAll("[data-action]")) b.disabled = false;
  }
}

// Find next figure with staging_n > 0 after `fromFigId`, wrapping around the list.
// Excludes `fromFigId` itself (we just emptied it).
function findNextPendingFigId(fromFigId) {
  const order = Object.keys(state.index.figures || {});
  if (order.length === 0) return null;
  const startIdx = order.indexOf(fromFigId);
  for (let i = 1; i <= order.length; i++) {
    const idx = ((startIdx === -1 ? -1 : startIdx) + i + order.length) % order.length;
    const fid = order[idx];
    if (fid === fromFigId) continue;
    if ((state.index.figures[fid].staging_n || 0) > 0) {
      return fid;
    }
  }
  return null;
}

// -----------------------------------------------------------------------
// Copy-reference: build markdown blob for the currently-compared pair and
// copy to clipboard. Pasting this into an agent chat is sufficient context for
// the agent to run `history_cli.py context-bundle <fig_id>` and reach full
// state (comments, supersedes, rejected attempts, etc.) on its own.
// -----------------------------------------------------------------------
function buildPairReferenceMarkdown() {
  const m = state.currentManifest;
  if (!m) return null;
  const beforeChoice = state.beforeChoice;
  const afterChoice = state.afterChoice;
  if (!beforeChoice || !afterChoice) return null;

  function describe(choice) {
    if (!choice) return null;
    const trunk = m.trunk || [];
    const staging = m.staging || [];
    const rejected = m.rejected || [];
    if (choice.startsWith("trunk:")) {
      const t = trunk.find(x => x.version === choice.slice(6));
      if (!t) return null;
      return {
        id: t.version,
        bucket: "trunk",
        tag: t.tag || "",
        sha: t.sha256 || "",
        filename: t.filename || "",
        relpath: `paper/figures/_history/${m.fig_id}/${t.filename || ""}`,
      };
    }
    if (choice.startsWith("staging:")) {
      const e = staging.find(x => x.staged_id === choice.slice(8));
      if (!e) return null;
      return {
        id: e.staged_id,
        bucket: "staging",
        tag: e.tag || "",
        sha: e.sha256 || "",
        filename: e.filename || "",
        relpath: `paper/figures/_history/${m.fig_id}/${e.filename || ""}`,
      };
    }
    if (choice.startsWith("rejected:")) {
      const e = rejected.find(x => x.rejected_id === choice.slice(9));
      if (!e) return null;
      return {
        id: e.rejected_id,
        bucket: "rejected",
        tag: e.tag || "",
        sha: e.sha256 || "",
        filename: e.filename || "",
        relpath: e.filename ? `paper/figures/_history/${m.fig_id}/${e.filename}` : "(file missing)",
      };
    }
    return null;
  }

  const b = describe(beforeChoice);
  const a = describe(afterChoice);
  if (!b || !a) return null;

  const shortSha = (s) => (s ? s.slice(0, 8) + "…" : "(none)");
  const canonical = m.canonical_output_path || "";
  const generatorCmd = (m.generator || {}).command || "";
  const label = m.paper_label || m.fig_id;

  const lines = [];
  lines.push(`### Figure ${label} (${m.fig_id}) — currently compared pair`);
  lines.push("");
  lines.push(`**Before**: ${b.id} — ${b.tag || "(no tag)"}${b.bucket === "rejected" ? " [REJECTED]" : (b.bucket === "staging" ? " [STAGING]" : "")} (sha \`${shortSha(b.sha)}\`)`);
  lines.push(`- file: \`${b.relpath}\``);
  lines.push("");
  lines.push(`**After**: ${a.id} — ${a.tag || "(no tag)"}${a.bucket === "rejected" ? " [REJECTED]" : (a.bucket === "staging" ? " [STAGING]" : "")} (sha \`${shortSha(a.sha)}\`)`);
  lines.push(`- file: \`${a.relpath}\``);
  lines.push("");
  if (canonical) lines.push(`**Canonical (paper draft 引用路径)**: \`${canonical}\``);
  if (generatorCmd) lines.push(`**Generator**: \`${generatorCmd}\``);
  lines.push("");
  lines.push(`**Agent 探索命令**:`);
  lines.push(`- 完整上下文 (manifest + active/settled/superseded/rejected_attempts + 所有评论):`);
  lines.push(`  \`uv run python .claude/skills/figure-snapshot-diff/scripts/history_cli.py context-bundle ${m.fig_id}\``);
  lines.push(`- 仅版本列表:`);
  lines.push(`  \`uv run python .claude/skills/figure-snapshot-diff/scripts/history_cli.py list ${m.fig_id}\``);
  lines.push(`- UI 浏览: <http://127.0.0.1:8765/> (启动: \`uv run python .claude/skills/figure-snapshot-diff/scripts/history_server.py --port 8765\`)`);
  return lines.join("\n");
}

async function copyPairReference() {
  const md = buildPairReferenceMarkdown();
  if (!md) {
    showToast("Pick Before and After versions first", true);
    return;
  }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(md);
    } else {
      // Fallback for non-https / older browsers
      const ta = document.createElement("textarea");
      ta.value = md;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    showToast(`Reference copied (${md.length} chars) — paste into an agent chat`);
  } catch (e) {
    showToast(`Copy failed: ${e.message}`, true);
  }
}

// -----------------------------------------------------------------------
// History modal: shows all comments on this figure, grouped by pair, with a
// "jump to this pair" button on each group. Closes on Esc / backdrop / X.
// Renders into #modal-host so the underlying comment-form is untouched
// (textarea draft survives open→close).
// -----------------------------------------------------------------------
function openHistoryModal() {
  if (!state.currentManifest) {
    showToast("Select a figure first", true);
    return;
  }
  state.historyModalOpen = true;
  renderHistoryModal();
}

function closeHistoryModal() {
  state.historyModalOpen = false;
  const host = document.getElementById("modal-host");
  if (host) host.innerHTML = "";
}

function renderHistoryModal() {
  const host = document.getElementById("modal-host");
  if (!host) return;
  const m = state.currentManifest;
  if (!m) return;

  const currentBeforeSha = shaForChoice(m, state.beforeChoice);
  const currentAfterSha = shaForChoice(m, state.afterChoice);

  const all = state.currentComments || [];
  const counts = {
    all: all.length,
    on_pair: all.filter(c => c.before_sha === currentBeforeSha && c.after_sha === currentAfterSha).length,
    other_pairs: all.filter(c => !(c.before_sha === currentBeforeSha && c.after_sha === currentAfterSha) && (c.status === "open" || c.status === "claimed")).length,
    history: all.filter(c => c.status === "resolved").length,
  };

  // Pick filtered set
  let filtered = all;
  if (state.historyFilter === "on_pair") {
    filtered = all.filter(c => c.before_sha === currentBeforeSha && c.after_sha === currentAfterSha);
  } else if (state.historyFilter === "other_pairs") {
    filtered = all.filter(c => !(c.before_sha === currentBeforeSha && c.after_sha === currentAfterSha) && (c.status === "open" || c.status === "claimed"));
  } else if (state.historyFilter === "history") {
    filtered = all.filter(c => c.status === "resolved");
  }

  // Group by (before_sha, after_sha) pair, preserve order by latest comment created_at desc
  const groups = new Map();
  for (const c of filtered) {
    const key = `${c.before_sha}__${c.after_sha}`;
    if (!groups.has(key)) {
      groups.set(key, {
        before_sha: c.before_sha,
        after_sha: c.after_sha,
        before_label: c.before_label || (c.before_sha || "").slice(0, 8),
        after_label: c.after_label || (c.after_sha || "").slice(0, 8),
        latest_ts: c.created_at || "",
        comments: [],
      });
    }
    const g = groups.get(key);
    g.comments.push(c);
    if ((c.created_at || "") > g.latest_ts) g.latest_ts = c.created_at;
  }
  const groupList = Array.from(groups.values()).sort((x, y) => (y.latest_ts || "").localeCompare(x.latest_ts || ""));

  function filterBtn(key, label, n) {
    const active = state.historyFilter === key ? "active" : "";
    return `<button type="button" class="filter-btn ${active}" data-filter="${key}">${escapeHtml(label)} (${n})</button>`;
  }

  function pairResolves(beforeSha, afterSha) {
    // Returns true if both SHAs can be resolved to a version in the current chain.
    const find = (sha) => {
      if (!sha) return null;
      return (m.trunk || []).find(t => t.sha256 === sha)
          || (m.staging || []).find(s => s.sha256 === sha)
          || (m.rejected || []).find(r => r.sha256 === sha);
    };
    return !!(find(beforeSha) && find(afterSha));
  }

  const groupsHtml = groupList.length
    ? groupList.map(g => {
        const isCurrent = (g.before_sha === currentBeforeSha && g.after_sha === currentAfterSha);
        const canJump = pairResolves(g.before_sha, g.after_sha) && !isCurrent;
        const jumpBtn = isCurrent
          ? `<button type="button" class="jump-btn" disabled title="This is the current pair">⇄ (current)</button>`
          : `<button type="button" class="jump-btn" data-before-sha="${escapeHtml(g.before_sha || "")}" data-after-sha="${escapeHtml(g.after_sha || "")}" ${canJump ? "" : "disabled title='At least one SHA is no longer in this figure\\'s version chain'"}>⇄ Jump to this pair</button>`;
        const commentsHtml = g.comments
          .slice()
          .sort((x, y) => (y.created_at || "").localeCompare(x.created_at || ""))
          .map(c => renderCommentEntry(c, false))
          .join("");
        return `
          <div class="history-pair-group">
            <div class="history-pair-header">
              <div class="pair-label ${isCurrent ? "current-pair" : ""}" title="before sha: ${escapeHtml(g.before_sha || "")}\nafter sha: ${escapeHtml(g.after_sha || "")}">
                <code>${escapeHtml(g.before_label)}</code> → <code>${escapeHtml(g.after_label)}</code>${isCurrent ? " · current" : ""}
              </div>
              ${jumpBtn}
            </div>
            <div class="pair-comments">${commentsHtml}</div>
          </div>`;
      }).join("")
    : `<div class="modal-empty">No comments in this filter.</div>`;

  host.innerHTML = `
    <div class="modal-backdrop" id="history-modal-backdrop">
      <div class="modal-dialog" role="dialog" aria-modal="true" aria-label="Comment history">
        <div class="modal-header">
          <h2>History — ${escapeHtml(m.paper_label || m.fig_id)} (${counts.all} comment${counts.all === 1 ? "" : "s"})</h2>
          <button type="button" class="modal-close" id="history-modal-close" title="Close (Esc)">×</button>
        </div>
        <div class="modal-body">
          <div class="history-filter-bar">
            ${filterBtn("all", "All", counts.all)}
            ${filterBtn("on_pair", "On current pair", counts.on_pair)}
            ${filterBtn("other_pairs", "Other pairs (active)", counts.other_pairs)}
            ${filterBtn("history", "Resolved / Superseded", counts.history)}
          </div>
          ${groupsHtml}
        </div>
      </div>
    </div>
  `;

  // Wire close handlers
  const backdrop = document.getElementById("history-modal-backdrop");
  const dialog = backdrop ? backdrop.querySelector(".modal-dialog") : null;
  backdrop.addEventListener("click", (ev) => {
    // Only close when click target IS the backdrop itself, not inside dialog
    if (ev.target === backdrop) closeHistoryModal();
  });
  const closeBtn = document.getElementById("history-modal-close");
  if (closeBtn) closeBtn.addEventListener("click", closeHistoryModal);

  // Filter buttons
  for (const btn of host.querySelectorAll(".filter-btn")) {
    btn.addEventListener("click", () => {
      state.historyFilter = btn.dataset.filter;
      renderHistoryModal();
    });
  }

  // Jump buttons
  for (const btn of host.querySelectorAll(".jump-btn[data-before-sha]")) {
    btn.addEventListener("click", () => {
      jumpToPair(btn.dataset.beforeSha, btn.dataset.afterSha);
    });
  }
}

function jumpToPair(beforeSha, afterSha) {
  const m = state.currentManifest;
  if (!m) return;
  const findChoice = (sha) => {
    if (!sha) return null;
    const t = (m.trunk || []).find(x => x.sha256 === sha);
    if (t) return `trunk:${t.version}`;
    const s = (m.staging || []).find(x => x.sha256 === sha);
    if (s) return `staging:${s.staged_id}`;
    const r = (m.rejected || []).find(x => x.sha256 === sha);
    if (r) return `rejected:${r.rejected_id}`;
    return null;
  };
  const bChoice = findChoice(beforeSha);
  const aChoice = findChoice(afterSha);
  if (!bChoice || !aChoice) {
    showToast("Version no longer in chain — cannot jump", true);
    return;
  }
  state.beforeChoice = bChoice;
  state.afterChoice = aChoice;
  // User explicitly jumped to a historical pair — banner no longer applies.
  state.autoSkipInfo = null;
  closeHistoryModal();
  renderMain();
  showToast(`Jumped to ${labelForChoice(bChoice)} → ${labelForChoice(aChoice)}`);
}

// -----------------------------------------------------------------------
// State / lifecycle
// -----------------------------------------------------------------------

async function selectFigure(figId) {
  state.selectedFigId = figId;
  if (state.historyModalOpen) closeHistoryModal();  // stale modal would show prior fig's comments
  try {
    const [manifest, comments] = await Promise.all([
      apiGet(`/api/figures/${encodeURIComponent(figId)}`),
      apiGet(`/api/figures/${encodeURIComponent(figId)}/comments`),
    ]);
    state.currentManifest = manifest;
    state.currentComments = comments.comments || [];
    pickDefaultChoices();
    renderSidebar();
    renderMain();
  } catch (e) {
    showToast(`Failed to load ${figId}: ${e.message}`, true);
  }
}

async function refreshComments() {
  if (!state.selectedFigId) return;
  try {
    const data = await apiGet(`/api/figures/${encodeURIComponent(state.selectedFigId)}/comments`);
    state.currentComments = data.comments || [];
    renderCommentsArea();
  } catch (e) {
    // silent — sidebar will surface connection errors
  }
}

function pickDefaultChoices() {
  // SHA-aware default selection: After = the "latest/pending" (staging[0] if any,
  // else trunk tip), Before = the closest predecessor with a *different* SHA.
  // If every candidate has the same SHA as After, fall back to Before == After
  // so renderSlider() degrades to single-image render. When the chosen Before
  // is NOT the immediate predecessor (i.e. some identical versions were
  // skipped), record the skip into state.autoSkipInfo so the UI can warn at
  // the top of the comparison area.
  const m = state.currentManifest;
  state.autoSkipInfo = null;
  if (!m) {
    state.beforeChoice = null;
    state.afterChoice = null;
    return;
  }
  const trunk = m.trunk || [];
  const staging = m.staging || [];

  // 1. Determine latestPending and the candidate ancestor list (closest first).
  let latestPendingChoice, latestPendingSha, candidates;
  if (staging.length) {
    const s = staging[0];
    latestPendingChoice = `staging:${s.staged_id}`;
    latestPendingSha = s.sha256;
    // Trunk from tip backwards. (Other staging entries are alternative proposals
    // against the same trunk tip; not useful as the "natural predecessor".)
    candidates = trunk.slice().reverse().map(t => ({
      choice: `trunk:${t.version}`, sha: t.sha256,
    }));
  } else if (trunk.length) {
    const tip = trunk[trunk.length - 1];
    latestPendingChoice = `trunk:${tip.version}`;
    latestPendingSha = tip.sha256;
    candidates = trunk.slice(0, -1).reverse().map(t => ({
      choice: `trunk:${t.version}`, sha: t.sha256,
    }));
  } else {
    state.beforeChoice = null;
    state.afterChoice = null;
    return;
  }

  state.afterChoice = latestPendingChoice;

  // 2. Find first candidate with a different SHA.
  const firstDifferentIdx = candidates.findIndex(c => c.sha !== latestPendingSha);

  if (firstDifferentIdx === -1) {
    // No meaningful comparison possible; let renderSlider degrade to single-image.
    state.beforeChoice = latestPendingChoice;
    if (candidates.length > 0) {
      state.autoSkipInfo = {
        mode: "no_different",
        skippedCount: candidates.length,
      };
    }
    return;
  }

  state.beforeChoice = candidates[firstDifferentIdx].choice;
  if (firstDifferentIdx > 0) {
    // Skipped some byte-identical predecessors.
    state.autoSkipInfo = {
      mode: "skipped",
      skippedCount: firstDifferentIdx,
      naturalChoice: candidates[0].choice,
      actualChoice: candidates[firstDifferentIdx].choice,
    };
  }
}

async function refreshIndex() {
  try {
    const before = JSON.stringify(state.index.figures || {});
    state.index = await apiGet("/api/index");
    setStatus(`connected · last update ${(state.index.last_updated || "").slice(11, 19)}`, "ok");
    const after = JSON.stringify(state.index.figures || {});
    if (before !== after) {
      renderSidebar();
      // If selected fig's staging changed, also refresh manifest
      if (state.selectedFigId) {
        const f = state.index.figures[state.selectedFigId];
        if (f && state.currentManifest && f.staging_n !== (state.currentManifest.staging || []).length) {
          await selectFigure(state.selectedFigId);
        }
      }
    }
  } catch (e) {
    setStatus(`connection error: ${e.message}`, "err");
  }
}

async function init() {
  document.getElementById("reload-btn").addEventListener("click", () => refreshIndex());
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.historyModalOpen) {
      ev.preventDefault();
      closeHistoryModal();
    }
  });
  await refreshIndex();
  renderSidebar();
  state.pollTimer = setInterval(refreshIndex, POLL_MS);
}

init();

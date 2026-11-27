// Kala Studio seller page: caption approval on top of /api/seller and the approval workflow routes.
// Plain JS, no build step. Every write sends X-Seller-User (no real login yet: demo only).
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const FIELDS = ["name", "description", "tags"];
const FIELD_LABEL = { name: "Nama", description: "Deskripsi", tags: "Tag" };
const STATUS_LABEL = { pending: "menunggu", applied: "diterapkan", rejected: "ditolak", cancelled: "dibatalkan", stale: "basi", reverted: "dikembalikan",
  open: "diproses agent", proposed: "sudah diusulkan" };
const STOP_REASON = {
  applied: ["ok", "Disetujui. Caption baru sudah tampil di toko."],
  stale: ["bad", "Tidak diterapkan: caption produk sudah berubah lebih dulu, jadi usulan ini basi."],
  cancelled: ["", "Dibatalkan."],
  max_rejections: ["bad", "Sudah ditolak 5 kali, alur berhenti. Buat usulan baru bila masih perlu."],
  revision_failed: ["bad", "Agent dua kali gagal menulis revisi yang valid, alur berhenti."],
};
const USER_PATTERN = /^[\w.@-]{1,64}$/;

const state = { status: "pending", products: {}, busy: false };

// ---------- API ----------
class ApiError extends Error {
  constructor(status, body) {
    super(body?.detail ? (typeof body.detail === "string" ? body.detail : "permintaan tidak valid") : `HTTP ${status}`);
    this.status = status;
    this.code = body?.code;
  }
}

async function api(path, { method = "GET", body } = {}) {
  const headers = {};
  if (method !== "GET") {
    const user = currentUser();
    if (!user) throw new ApiError(0, { detail: "isi nama seller dulu (kanan atas)" });
    headers["X-Seller-User"] = user;
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}
const proposalUrl = (pid, rest = "") => `/api/seller/caption-proposals/${encodeURIComponent(pid)}${rest}`;
const requestUrl = (rid, rest = "") => `/api/seller/caption-requests/${encodeURIComponent(rid)}${rest}`;
const threadUrl = (id, rest = "") => id.startsWith("CR-") ? requestUrl(id, rest) : proposalUrl(id, rest);

// ---------- seller name ----------
function currentUser() {
  const value = $("#user").value.trim();
  return USER_PATTERN.test(value) ? value : "";
}
function initUser() {
  let saved = "";
  try { saved = localStorage.getItem("kala-seller-user") || ""; } catch { /* storage unavailable */ }
  $("#user").value = saved || "seller-demo";
  $("#user").addEventListener("change", () => {
    try { localStorage.setItem("kala-seller-user", $("#user").value.trim()); } catch { /* ignore */ }
  });
}
const who = (actor) => actor === "agent:caption-approval" ? "agent (revisi)" : String(actor || "").replace(/^seller:/, "");
const when = (iso) => iso ? new Date(iso).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" }) : "";

// ---------- diff ----------
function wordDiff(oldText, newText) {
  const a = String(oldText ?? "").split(/(\s+)/), b = String(newText ?? "").split(/(\s+)/);
  const n = a.length, m = b.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  let i = 0, j = 0, left = "", right = "";
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { left += esc(a[i]); right += esc(b[j]); i++; j++; }
    else if (j < m && (i >= n || lcs[i][j + 1] >= lcs[i + 1][j])) { right += /^\s+$/.test(b[j]) ? esc(b[j]) : `<ins>${esc(b[j])}</ins>`; j++; }
    else { left += /^\s+$/.test(a[i]) ? esc(a[i]) : `<del>${esc(a[i])}</del>`; i++; }
  }
  return [left, right];
}

function tagDiff(oldTags = [], newTags = []) {
  const before = new Set(oldTags), after = new Set(newTags);
  const left = oldTags.map((t) => after.has(t) ? `<span>${esc(t)}</span>` : `<del>${esc(t)}</del>`).join("");
  const right = newTags.map((t) => before.has(t) ? `<span>${esc(t)}</span>` : `<ins>${esc(t)}</ins>`).join("");
  return [`<div class="tags">${left}</div>`, `<div class="tags">${right}</div>`];
}

function changeBlock(field, change, allowed) {
  const [left, right] = field === "tags" ? tagDiff(change.old, change.new) : wordDiff(change.old, change.new);
  const len = (v) => field === "tags" ? `${(v || []).length} tag` : `${String(v ?? "").length} karakter`;
  return `
    <div class="change">
      <h3>${FIELD_LABEL[field] || esc(field)} ${allowed ? "" : "<small>(di luar cakupan revisi)</small>"}</h3>
      <div class="sides">
        <div class="side"><span class="label">Sekarang</span>${left}<div class="count">${len(change.old)}</div></div>
        <div class="side new"><span class="label">Usulan</span>${right}<div class="count">${len(change.new)}</div></div>
      </div>
    </div>`;
}

// ---------- queue ----------
async function renderQueue() {
  const params = new URLSearchParams({ limit: "100" });
  if (state.status) params.set("status", state.status);
  const requestParams = new URLSearchParams({ limit: "50" });
  if (state.status) requestParams.set("status", "open");   // "Menunggu": requests the agent is still on
  let proposals, requests;
  try {
    [{ proposals }, { requests }] = await Promise.all([
      api(`/api/seller/caption-proposals?${params}`), api(`/api/seller/caption-requests?${requestParams}`)]);
  } catch (error) { $("#queue").innerHTML = `<p class="empty">Gagal memuat: ${esc(error.message)}</p>`; return; }
  const selected = selectedId();
  const requestItems = requests.filter((r) => r.status !== "proposed").map((r) => {
    const product = state.products[r.product_id] || {};
    return `
      <a class="q-item request" href="#/${esc(r.request_id)}" ${r.request_id === selected ? 'aria-current="true"' : ""}>
        <img src="${esc(product.thumbnail || "")}" alt="">
        <div>
          <div class="title">${esc(product.name || r.product_id)}</div>
          <div class="sub"><span class="status ${esc(r.status)}">${STATUS_LABEL[r.status] || esc(r.status)}</span>
            <span class="mono">${esc(r.request_id)}</span>
            <span>· caption baru: ${r.fields.map((f) => FIELD_LABEL[f] || f).join(", ")}</span></div>
        </div>
      </a>`;
  });
  $("#queue").innerHTML = proposals.length || requestItems.length ? requestItems.join("") + proposals.map((p) => {
    const product = state.products[p.product_id] || {};
    return `
      <a class="q-item" href="#/${esc(p.proposal_id)}" ${p.proposal_id === selected ? 'aria-current="true"' : ""}>
        <img src="${esc(product.thumbnail || "")}" alt="">
        <div>
          <div class="title">${esc(product.name || p.product_id)}</div>
          <div class="sub"><span class="status ${esc(p.status)}">${STATUS_LABEL[p.status] || esc(p.status)}</span>
            <span class="mono">${esc(p.proposal_id)}</span>
            <span>· ${Object.keys(p.changes).map((f) => FIELD_LABEL[f] || f).join(", ")}</span>
            ${p.rejections_in_chain ? `<span>· ${p.rejections_in_chain}/5 ditolak</span>` : ""}</div>
        </div>
      </a>`;
  }).join("") : `<p class="empty">${state.status === "pending" ? "Tidak ada usulan yang menunggu." : "Belum ada usulan."}</p>`;
}

// ---------- detail ----------
const selectedId = () => decodeURIComponent((location.hash.match(/^#\/(C[PR]-\d+)$/) || [])[1] || "");

async function renderDetail() {
  const pid = selectedId();
  const root = $("#detail");
  if (!pid) { root.innerHTML = `<p class="empty">Pilih usulan di kiri.</p>`; return; }
  root.innerHTML = `<p class="working"><span class="spinner"></span> Memuat ${esc(pid)}…</p>`;
  if (pid.startsWith("CR-")) return renderRequest(pid);
  let proposal, approval = null;
  try {
    proposal = await api(proposalUrl(pid));
    try { approval = await api(proposalUrl(pid, "/approval")); }
    catch (error) { if (error.code !== "approval_not_found") throw error; }
  } catch (error) { root.innerHTML = `<p class="notice bad">Gagal memuat ${esc(pid)}: ${esc(error.message)}</p>`; return; }
  const [log, facts] = await Promise.all([auditLog(proposal.product_id), productFacts(proposal.product_id)]);
  root.innerHTML = detailHtml(proposal, approval, log, facts);
  bindDetail(proposal, approval);
}

const auditLog = (productId) => api(`/api/seller/audit-log?product_id=${encodeURIComponent(productId)}&limit=30`)
  .then((r) => r.entries).catch(() => []);
const productFacts = (productId) => api(`/api/seller/products/${encodeURIComponent(productId)}/facts`)
  .then((r) => r.facts).catch(() => ({}));

function detailHtml(p, approval, log, facts) {
  const product = state.products[p.product_id] || {};
  const waitingHere = approval?.waiting_for === "decision" && approval.request.proposal.proposal_id === p.proposal_id;
  const askingHere = approval?.waiting_for === "answer" && approval.proposal_id === p.proposal_id;
  const fields = waitingHere ? approval.request.fields : Object.keys(p.changes);
  const notices = [];
  if (approval?.waiting_for === "decision" && !waitingHere) {
    const next = approval.request.proposal.proposal_id;
    notices.push(`<p class="notice">Versi ini sudah diputuskan. Yang menunggu sekarang: <a href="#/${esc(next)}">${esc(next)} →</a></p>`);
  }
  if (waitingHere && approval.request.error) notices.push(`<p class="notice bad">Jawaban terakhir ditolak: ${esc(approval.request.error)}</p>`);
  if (approval?.state === "error") notices.push(`<p class="notice bad">Langkah agent gagal (mis. LLM tidak tersedia). Penolakanmu sudah tercatat. <button class="link-btn" id="retry">Coba lagi</button></p>`);
  if (approval?.state === "done" && approval.proposal_id === p.proposal_id && STOP_REASON[approval.stop_reason]) {
    const [kind, text] = STOP_REASON[approval.stop_reason];
    notices.push(`<p class="notice ${kind}">${text}${approval.draft_error ? ` <span class="mono">${esc(approval.draft_error)}</span>` : ""}</p>`);
  }
  if (p.status === "rejected") notices.push(`<p class="notice">Ditolak oleh ${esc(who(p.decided_by))}: ${p.feedback ? `“${esc(p.feedback)}”` : "tanpa alasan"}</p>`);
  if (p.status === "reverted") notices.push(`<p class="notice">Dikembalikan oleh ${esc(who(p.reverted_by))} · ${esc(when(p.reverted_at))}</p>`);

  const canRevert = p.status === "applied";
  const others = FIELDS.filter((f) => !fields.includes(f));
  return `
    <div class="detail-head">
      <img src="${esc(product.thumbnail || "")}" alt="">
      <div>
        <h2>${esc(product.name || p.product_id)}</h2>
        <div class="sub">
          <span class="status ${esc(p.status)}">${STATUS_LABEL[p.status] || esc(p.status)}</span>
          <span class="mono">${esc(p.proposal_id)}</span>
          ${p.revision_of ? `<span>revisi dari <a href="#/${esc(p.revision_of)}" class="mono">${esc(p.revision_of)}</a></span>` : ""}
          <span>· oleh ${esc(who(p.created_by))} · ${esc(when(p.created_at))}</span>
          <span>· ${p.rejections_in_chain}/5 ditolak</span>
          <a class="link-btn" href="/#/p/${esc(p.product_id)}" target="_blank" rel="noopener">lihat di toko ↗</a>
        </div>
      </div>
    </div>
    ${notices.join("")}
    <div class="reason"><b>Alasan ${p.created_by?.startsWith("agent") ? "agent" : "pengusul"}</b>${esc(p.reason)}</div>
    ${seenImages(p.inputs?.images)}
    ${usedFacts(p.inputs?.seller_facts)}
    ${askingHere ? questionForm(approval.request) : ""}
    ${Object.entries(p.changes).map(([field, change]) => changeBlock(field, change, fields.includes(field))).join("")}
    ${waitingHere ? `
      <div class="actions" id="actions">
        <button class="primary" data-decision="approve">Setujui & terapkan</button>
        <button class="ghost" id="reject-open">Tolak…</button>
        <button class="ghost danger" data-decision="cancel">Batalkan</button>
      </div>
      <form class="reject-form" id="reject-form" hidden>
        <label for="feedback">Apa yang perlu diubah? Agent akan menulis revisi.</label>
        <textarea id="feedback" rows="3" maxlength="1000" placeholder="mis. terlalu panjang, sebut bahannya"></textarea>
        ${others.length ? `<div class="fields"><span class="hint">Boleh juga ubah:</span>${others.map((f) =>
          `<label><input type="checkbox" name="add" value="${f}"> ${FIELD_LABEL[f]}</label>`).join("")}</div>` : ""}
        <p class="hint">Revisi hanya boleh mengubah: ${fields.map((f) => FIELD_LABEL[f] || f).join(", ")}${others.length ? " (+ yang dicentang)" : ""}.</p>
        <div><button class="primary" type="submit">Tolak & minta revisi</button></div>
      </form>` : ""}
    ${canRevert ? `<div class="actions"><button class="ghost danger" id="revert">Kembalikan caption sebelumnya</button>
      <span class="hint">Hanya bisa selama toko masih menampilkan caption dari usulan ini.</span></div>` : ""}
    ${factsPanel(facts)}
    <details class="history">
      <summary>Riwayat produk (audit log)</summary>
      <table class="log"><thead><tr><th>Waktu</th><th>Oleh</th><th>Aksi</th><th>Usulan</th></tr></thead><tbody>
        ${log.map((e) => `<tr><td>${esc(when(e.at))}</td><td>${esc(who(e.actor))}</td><td>${esc(e.action)}</td>
          <td class="mono">${e.proposal_id ? `<a href="#/${esc(e.proposal_id)}">${esc(e.proposal_id)}</a>` : ""}</td></tr>`).join("")}
      </tbody></table>
    </details>`;
}

function seenImages(images) {   // C3b: the exact images the model saw, served by hash
  if (!images?.length) return "";
  return `<div class="seen"><b>Gambar yang dilihat model</b><div class="seen-list">${images.map((img) => `
    <a href="/api/seller/images/${esc(img.sha256)}" target="_blank" rel="noopener" title="${esc(img.source_url || "")}">
      <img src="/api/seller/images/${esc(img.sha256)}" alt="" loading="lazy">
      <span class="mono">${esc(img.sha256.slice(0, 8))} · ${img.width}×${img.height}</span></a>`).join("")}</div></div>`;
}

function usedFacts(facts) {   // C3b: what the seller told the agent, as the draft used it
  const entries = Object.entries(facts || {});
  if (!entries.length) return "";
  return `<div class="seen"><b>Fakta dari seller yang dipakai</b><dl class="facts">${entries.map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${v === null ? "<em>seller tidak tahu</em>" : esc(v)}</dd>`).join("")}</dl></div>`;
}

function factsPanel(facts) {   // memory: the latest answer per key
  const entries = Object.entries(facts || {});
  if (!entries.length) return "";
  return `<details class="history"><summary>Yang sudah kamu ceritakan ke agent (${entries.length})</summary>
    <table class="log"><thead><tr><th>Fakta</th><th>Jawaban</th><th>Oleh</th><th>Dari</th></tr></thead><tbody>
    ${entries.map(([k, f]) => `<tr><td class="mono">${esc(k)}</td><td>${f.value === null ? "<em>tidak tahu</em>" : esc(f.value)}</td>
      <td>${esc(who(f.by))} · ${esc(when(f.at))}</td><td class="mono">${f.source ? `<a href="#/${esc(f.source)}">${esc(f.source)}</a>` : "langsung"}</td></tr>`).join("")}
    </tbody></table></details>`;
}

function questionForm(request) {   // the graph asks before writing (interrupt kind "question")
  return `
    <form class="question-form" id="question-form">
      <p class="q-title">Agent butuh info dulu sebelum menulis caption</p>
      ${request.error ? `<p class="notice bad">Jawaban ditolak: ${esc(typeof request.error === "string" ? request.error : "tidak valid")}</p>` : ""}
      ${request.questions.map((q) => `
        <label class="question">${esc(q.question)}
          <textarea name="${esc(q.key)}" rows="2" maxlength="300" placeholder="kosongkan kalau tidak tahu"></textarea>
          <span class="hint mono">${esc(q.key)}</span>
        </label>`).join("")}
      <p class="hint">Jawabanmu disimpan sebagai fakta produk, jadi agent tidak perlu bertanya lagi lain kali.
        Yang dikosongkan dicatat sebagai "tidak tahu": caption akan menyebut bahwa data tambahan dibutuhkan.</p>
      <div class="actions"><button class="primary" type="submit">Kirim jawaban</button></div>
    </form>`;
}

function bindQuestions(id) {
  const form = $("#question-form");
  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const answers = Object.fromEntries([...form.querySelectorAll("textarea")].map((t) => [t.name, t.value.trim() || null]));
    run(id, () => api(threadUrl(id, "/approval/answer"), { method: "POST", body: { answers } }), "Agent sedang menulis…");
  });
}

// ---------- caption request (C3b: the agent writes from scratch) ----------
async function renderRequest(rid) {
  const root = $("#detail");
  let request, approval = null;
  try {
    request = await api(requestUrl(rid));
    try { approval = await api(requestUrl(rid, "/approval")); }
    catch (error) { if (error.code !== "approval_not_found") throw error; }
  } catch (error) { root.innerHTML = `<p class="notice bad">Gagal memuat ${esc(rid)}: ${esc(error.message)}</p>`; return; }
  const [log, facts] = await Promise.all([auditLog(request.product_id), productFacts(request.product_id)]);
  const product = state.products[request.product_id] || {};
  const notices = [];
  if (approval?.state === "error") notices.push(`<p class="notice bad">Langkah agent gagal (mis. LLM tidak tersedia). <button class="link-btn" id="retry">Coba lagi</button></p>`);
  if (approval?.stop_reason === "cancelled" || request.status === "cancelled") notices.push(`<p class="notice">Permintaan dibatalkan.</p>`);
  const next = approval?.waiting_for === "decision" ? approval.request.proposal.proposal_id : request.proposal_id;
  if (next) notices.push(`<p class="notice ok">Agent sudah menulis usulan: <a href="#/${esc(next)}">${esc(next)} →</a></p>`);
  root.innerHTML = `
    <div class="detail-head">
      <img src="${esc(product.thumbnail || "")}" alt="">
      <div>
        <h2>${esc(product.name || request.product_id)}</h2>
        <div class="sub">
          <span class="status ${esc(request.status)}">${STATUS_LABEL[request.status] || esc(request.status)}</span>
          <span class="mono">${esc(request.request_id)}</span>
          <span>· caption baru: ${request.fields.map((f) => FIELD_LABEL[f] || f).join(", ")}</span>
          <span>· oleh ${esc(who(request.created_by))} · ${esc(when(request.created_at))}</span>
        </div>
      </div>
    </div>
    ${notices.join("")}
    <div class="reason"><b>Permintaan</b>${esc(request.instruction || "(tanpa instruksi khusus)")}</div>
    ${approval?.waiting_for === "answer" ? questionForm(approval.request) : ""}
    ${request.status === "open" ? `<div class="actions"><button class="ghost danger" id="cancel-request">Batalkan permintaan</button></div>` : ""}
    ${factsPanel(facts)}
    <details class="history"><summary>Riwayat produk (audit log)</summary>
      <table class="log"><thead><tr><th>Waktu</th><th>Oleh</th><th>Aksi</th><th>Usulan</th></tr></thead><tbody>
        ${log.map((e) => `<tr><td>${esc(when(e.at))}</td><td>${esc(who(e.actor))}</td><td>${esc(e.action)}</td>
          <td class="mono">${e.proposal_id ? `<a href="#/${esc(e.proposal_id)}">${esc(e.proposal_id)}</a>` : esc(e.detail?.request_id || "")}</td></tr>`).join("")}
      </tbody></table>
    </details>`;
  bindQuestions(rid);
  $("#retry", root)?.addEventListener("click", () => run(rid, () => api(requestUrl(rid, "/approval/retry"), { method: "POST" }), "Agent sedang mencoba lagi…"));
  $("#cancel-request", root)?.addEventListener("click", () => run(rid, () => api(requestUrl(rid, "/cancel"), { method: "POST" }), "Membatalkan…", "Permintaan dibatalkan."));
}

function bindDetail(p, approval) {
  const root = $("#detail");
  bindQuestions(p.proposal_id);
  root.querySelectorAll("[data-decision]").forEach((button) =>
    button.addEventListener("click", () => decide(p.proposal_id, { decision: button.dataset.decision })));
  $("#reject-open", root)?.addEventListener("click", () => {
    const form = $("#reject-form", root);
    form.hidden = !form.hidden;
    if (!form.hidden) $("#feedback", root).focus();
  });
  $("#reject-form", root)?.addEventListener("submit", (event) => {
    event.preventDefault();
    const add_fields = [...root.querySelectorAll('input[name="add"]:checked')].map((box) => box.value);
    decide(p.proposal_id, { decision: "reject", feedback: $("#feedback", root).value.trim(), add_fields });
  });
  $("#retry", root)?.addEventListener("click", () => run(p.proposal_id, () => api(proposalUrl(p.proposal_id, "/approval/retry"), { method: "POST" }),
    "Agent sedang mencoba lagi…"));
  $("#revert", root)?.addEventListener("click", () => run(p.proposal_id, () => api(proposalUrl(p.proposal_id, "/revert"), { method: "POST", body: {} }),
    "Mengembalikan caption…", "Caption sebelumnya tampil lagi di toko."));
}

function decide(pid, body) {
  const label = body.decision === "reject" ? "Agent sedang menulis revisi…" : body.decision === "approve" ? "Menerapkan caption…" : "Membatalkan…";
  run(pid, () => api(proposalUrl(pid, "/approval/decision"), { method: "POST", body }), label);
}

async function run(pid, call, label, doneText) {
  if (state.busy) return;
  state.busy = true;
  const root = $("#detail");
  root.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  const actions = $("#actions", root) || root.querySelector(".actions");
  actions?.insertAdjacentHTML("beforeend", `<span class="working"><span class="spinner"></span>${esc(label)}</span>`);
  try {
    const result = await call();
    await loadProducts();
    const next = result?.waiting_for === "decision" ? result.request.proposal.proposal_id : null;
    if (next && next !== pid) {
      toast(`${pid.startsWith("CR-") ? "Usulan" : "Revisi"} ${next} siap diputuskan`);
      location.hash = `#/${next}`;   // hashchange renders it
    } else if (result?.waiting_for === "answer" && result.request_id && pid !== result.request_id && !result.proposal_id) {
      location.hash = `#/${result.request_id}`;
    } else {
      if (doneText) toast(doneText);
      await renderDetail();
    }
  } catch (error) {
    toast(error.message);
    await renderDetail();
  } finally {
    state.busy = false;
    renderQueue();
  }
}

// ---------- new proposal (demo) ----------
function openNew(open) {
  $("#new-drawer").hidden = !open;
  $("#new-backdrop").hidden = !open;
}
async function submitRequest(event) {
  event.preventDefault();
  const fields = [...event.target.querySelectorAll('input[name="req-field"]:checked')].map((box) => box.value);
  if (!fields.length) { toast("Pilih minimal satu bagian caption"); return; }
  const body = { product_id: $("#req-product").value, fields, instruction: $("#req-instruction").value.trim() };
  const button = event.target.querySelector("button[type=submit]");
  button.disabled = true;
  button.insertAdjacentHTML("afterend", `<span class="working"><span class="spinner"></span>Agent sedang membaca produk…</span>`);
  try {
    const view = await api("/api/seller/caption-requests", { method: "POST", body });
    openNew(false);
    event.target.reset();
    const next = view.waiting_for === "decision" ? view.request.proposal.proposal_id : view.request_id;
    toast(view.waiting_for === "answer" ? "Agent punya pertanyaan untukmu" : view.state === "error" ? "Langkah agent gagal, coba lagi" : `Usulan ${next} siap diputuskan`);
    location.hash = `#/${next}`;
    renderQueue();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; event.target.querySelector(".working")?.remove(); }
}

async function submitNew(event) {
  event.preventDefault();
  const value = (id) => $(id).value.trim();
  const body = { product_id: value("#new-product"), reason: value("#new-reason") };
  if (value("#new-name")) body.name = value("#new-name");
  if (value("#new-description")) body.description = value("#new-description");
  if (value("#new-tags")) body.tags = value("#new-tags").split(",").map((t) => t.trim()).filter(Boolean);
  if (!body.name && !body.description && !body.tags) { toast("Isi minimal satu: nama, deskripsi, atau tag"); return; }
  try {
    const created = await api("/api/seller/caption-proposals", { method: "POST", body });
    openNew(false);
    event.target.reset();
    toast(`Usulan ${created.proposal_id} dibuat`);
    location.hash = `#/${created.proposal_id}`;
    renderQueue();
  } catch (error) { toast(error.message); }
}

let toastTimer;
function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

// ---------- init ----------
async function loadProducts() {   // names as the shop shows them now (approved captions included)
  try {
    const { items } = await api("/api/products?page_size=60");
    for (const item of items) state.products[item.product_id] = item;
    for (const select of [$("#new-product"), $("#req-product")]) {
      const chosen = select.value;
      select.innerHTML = items.map((p) => `<option value="${esc(p.product_id)}">${esc(p.product_id)} · ${esc(p.name)}</option>`).join("");
      if (chosen) select.value = chosen;
    }
  } catch { /* the page still works without thumbnails */ }
}

async function init() {
  initUser();
  await loadProducts();
  $("#status-chips").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-status]");
    if (!chip) return;
    state.status = chip.dataset.status;
    $("#status-chips").querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
    renderQueue();
  });
  $("#new-open").addEventListener("click", () => openNew(true));
  $("#new-close").addEventListener("click", () => openNew(false));
  $("#new-backdrop").addEventListener("click", () => openNew(false));
  $("#new-form").addEventListener("submit", submitNew);
  $("#request-form").addEventListener("submit", submitRequest);
  window.addEventListener("hashchange", () => { renderDetail(); renderQueue(); });
  await renderQueue();
  if (!selectedId()) {
    const first = $("#queue .q-item");
    if (first) location.hash = first.getAttribute("href"); else renderDetail();
  } else renderDetail();
}

init();

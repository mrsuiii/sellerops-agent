// Kala Studio demo store. Plain JS, no build step; talks to the FastAPI backend.
"use strict";

const $ = (sel) => document.querySelector(sel);
const rupiah = (n) => "Rp" + n.toLocaleString("id-ID");
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const CATEGORY_LABEL = {
  bags: "Tas", dresses: "Dress", fragrances: "Parfum", jewellery: "Perhiasan", shirts: "Kemeja",
  shoes: "Sepatu", sunglasses: "Kacamata", tops: "Atasan", watches: "Jam tangan",
};
const PAGE_SIZE = 12;

const state = { q: "", category: "", page: 1 };

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

// ---------- Cart (kept in the browser; checkout is a demo) ----------
const cart = {
  load() {
    try { return JSON.parse(localStorage.getItem("kala-cart") || "{}"); } catch { return {}; }
  },
  save(items) {
    try { localStorage.setItem("kala-cart", JSON.stringify(items)); } catch { /* storage unavailable */ }
    renderCart();
  },
  add(product) {
    const items = cart.load();
    const line = items[product.product_id] || { ...pick(product), qty: 0 };
    line.qty = Math.min(line.qty + 1, product.stock);
    items[product.product_id] = line;
    cart.save(items);
    toast(`${product.name} masuk keranjang`);
  },
  setQty(id, qty) {
    const items = cart.load();
    if (!items[id]) return;
    if (qty <= 0) delete items[id]; else items[id].qty = Math.min(qty, items[id].stock);
    cart.save(items);
  },
};
const pick = (p) => ({ product_id: p.product_id, name: p.name, thumbnail: p.thumbnail, final_price_idr: p.final_price_idr, stock: p.stock });

function renderCart() {
  const lines = Object.values(cart.load());
  $("#cart-count").textContent = lines.reduce((n, l) => n + l.qty, 0);
  $("#cart-total").textContent = rupiah(lines.reduce((n, l) => n + l.qty * l.final_price_idr, 0));
  $("#checkout").disabled = lines.length === 0;
  $("#cart-items").innerHTML = lines.length ? lines.map((l) => `
    <div class="line">
      <img src="${esc(l.thumbnail)}" alt="">
      <div>
        <a href="#/p/${esc(l.product_id)}">${esc(l.name)}</a>
        <div class="qty">
          <button data-qty="${esc(l.product_id)}" data-delta="-1" aria-label="Kurangi">−</button>
          <span>${l.qty}</span>
          <button data-qty="${esc(l.product_id)}" data-delta="1" aria-label="Tambah">+</button>
        </div>
      </div>
      <strong>${rupiah(l.qty * l.final_price_idr)}</strong>
    </div>`).join("") : `<p class="empty">Keranjang masih kosong.</p>`;
}

function openCart(open) {
  $("#cart").hidden = !open;
  $("#cart-backdrop").hidden = !open;
}

// ---------- Shared bits ----------
function card(p, extra = "") {
  const badge = p.stock <= 0 ? `<span class="badge out">Habis</span>`
    : p.discount_percent >= 1 ? `<span class="badge">-${p.discount_percent}%</span>` : "";
  return `
    <a class="card" href="#/p/${esc(p.product_id)}">
      <div class="img"><img src="${esc(p.thumbnail)}" alt="${esc(p.name)}" loading="lazy">${badge}</div>
      <span class="brand">${esc(p.brand)}</span>
      <span class="name">${esc(p.name)}</span>
      ${priceHtml(p)}
      <span class="rating">★ ${p.rating.toFixed(1)}</span>
      ${extra}
    </a>`;
}

function priceHtml(p) {
  return `<span class="price"><strong>${rupiah(p.final_price_idr)}</strong>${
    p.discount_percent >= 1 ? `<s>${rupiah(p.price_idr)}</s>` : ""}</span>`;
}

let toastTimer;
function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2200);
}

// ---------- Shop view ----------
async function loadFacets() {
  const data = await api("/api/categories");
  const total = data.categories.reduce((n, c) => n + c.count, 0);
  const chips = [{ category: "", count: total }, ...data.categories];
  $("#category-chips").innerHTML = chips.map((c) => `
    <button class="chip" data-category="${esc(c.category)}" aria-pressed="${c.category === state.category}">
      ${esc(c.category ? CATEGORY_LABEL[c.category] || c.category : "Semua")}<small>${c.count}</small>
    </button>`).join("");
  $("#f-brand").insertAdjacentHTML("beforeend", data.brands.map((b) => `<option>${esc(b)}</option>`).join(""));
}

function currentParams() {
  const params = new URLSearchParams({ page: state.page, page_size: PAGE_SIZE, sort: $("#f-sort").value });
  const add = (key, value) => { if (value) params.set(key, value); };
  add("q", state.q);
  add("category", state.category);
  add("audience", $("#f-audience").value);
  add("brand", $("#f-brand").value);
  add("max_price", $("#f-max-price").value);
  add("min_rating", $("#f-min-rating").value);
  if ($("#f-in-stock").checked) params.set("in_stock", "true");
  return params;
}

async function loadProducts() {
  const data = await api(`/api/products?${currentParams()}`);
  $("#result-meta").textContent = `${data.total} produk${state.q ? ` untuk “${state.q}”` : ""}`;
  $("#grid").innerHTML = data.items.length ? data.items.map((p) => card(p)).join("")
    : `<p class="empty">Tidak ada produk yang cocok. Coba longgarkan filternya.</p>`;
  const pages = Math.ceil(data.total / PAGE_SIZE);
  $("#pager").innerHTML = pages > 1 ? Array.from({ length: pages }, (_, i) =>
    `<button data-page="${i + 1}" ${i + 1 === state.page ? 'aria-current="page"' : ""}>${i + 1}</button>`).join("") : "";
  document.querySelectorAll(".chip").forEach((el) => el.setAttribute("aria-pressed", el.dataset.category === state.category));
}

function refresh() { state.page = 1; loadProducts(); }

// ---------- Product view ----------
async function showProduct(id) {
  const view = $("#view-product");
  let p;
  try { p = await api(`/api/products/${encodeURIComponent(id)}`); } catch {
    view.innerHTML = `<a class="back" href="#/">← Kembali</a><p class="empty">Produk tidak ditemukan.</p>`;
    return;
  }
  const images = p.images.length ? p.images : [p.thumbnail];
  view.innerHTML = `
    <a class="back" href="#/">← Kembali ke katalog</a>
    <div class="product">
      <div class="gallery">
        <div class="main"><img id="main-img" src="${esc(images[0])}" alt="${esc(p.name)}"></div>
        ${images.length > 1 ? `<div class="thumbs">${images.map((src, i) =>
          `<button data-src="${esc(src)}" aria-current="${i === 0}"><img src="${esc(src)}" alt=""></button>`).join("")}</div>` : ""}
      </div>
      <div class="info">
        <span class="brand">${esc(p.brand)} · ${esc(CATEGORY_LABEL[p.category] || p.category)}</span>
        <h1>${esc(p.name)}</h1>
        ${priceHtml(p)}
        <p class="rating">★ ${p.rating.toFixed(1)} · ${p.reviews.length} ulasan</p>
        <p>${esc(p.description)}</p>
        <dl class="meta">
          <dt>Stok</dt><dd>${p.stock > 0 ? `${p.stock} tersedia` : "Habis"}</dd>
          <dt>SKU</dt><dd>${esc(p.sku || p.product_id)}</dd>
          <dt>Garansi</dt><dd>${esc(p.warranty || "-")}</dd>
          <dt>Pengiriman</dt><dd>${esc(p.shipping || "-")}</dd>
          <dt>Retur</dt><dd>${esc(p.return_policy || "-")}</dd>
        </dl>
        <button class="primary" id="add-to-cart" ${p.stock > 0 ? "" : "disabled"}>${p.stock > 0 ? "Tambah ke keranjang" : "Stok habis"}</button>
        <div class="reviews">
          <h2 class="section-title" style="margin-top:0">Ulasan</h2>
          ${p.reviews.map((r) => `
            <div class="review">★ ${r.rating} — ${esc(r.comment)}<br><span class="who">${esc(r.reviewer)} · ${esc(r.date)}</span></div>`).join("")}
        </div>
      </div>
    </div>
    <h2 class="section-title">Mungkin kamu suka</h2>
    <div class="grid" id="similar"><p class="empty">Memuat rekomendasi…</p></div>`;

  $("#add-to-cart").addEventListener("click", () => cart.add(p));
  view.querySelectorAll(".thumbs button").forEach((btn) => btn.addEventListener("click", () => {
    $("#main-img").src = btn.dataset.src;
    view.querySelectorAll(".thumbs button").forEach((b) => b.setAttribute("aria-current", b === btn));
  }));

  const rec = await api("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ similar_to: p.product_id, limit: 4 }),
  });
  $("#similar").innerHTML = rec.items.length ? rec.items.map((r) =>
    card(r, r.reasons.length ? `<span class="reasons">${r.reasons.map((x) => `<span>${esc(x)}</span>`).join("")}</span>` : "")).join("")
    : `<p class="empty">Belum ada rekomendasi.</p>`;
}

// ---------- Routing ----------
function route() {
  const match = location.hash.match(/^#\/p\/(.+)$/);
  $("#view-shop").hidden = Boolean(match);
  $("#view-product").hidden = !match;
  if (match) { showProduct(decodeURIComponent(match[1])); window.scrollTo(0, 0); }
}

// ---------- Wiring ----------
$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  state.q = $("#search-input").value.trim();
  if (location.hash.startsWith("#/p/")) location.hash = "#/";
  refresh();
});
$("#category-chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  state.category = chip.dataset.category;
  refresh();
});
["#f-audience", "#f-brand", "#f-max-price", "#f-min-rating", "#f-in-stock", "#f-sort"].forEach((sel) =>
  $(sel).addEventListener("change", refresh));
$("#f-reset").addEventListener("click", () => {
  ["#f-audience", "#f-brand", "#f-max-price", "#f-min-rating"].forEach((sel) => { $(sel).value = ""; });
  $("#f-sort").value = "relevance";
  $("#f-in-stock").checked = false;
  state.category = "";
  refresh();
});
$("#pager").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-page]");
  if (!btn) return;
  state.page = Number(btn.dataset.page);
  loadProducts();
  $(".results").scrollIntoView({ behavior: "smooth" });
});
$("#cart-open").addEventListener("click", () => openCart(true));
$("#cart-close").addEventListener("click", () => openCart(false));
$("#cart-backdrop").addEventListener("click", () => openCart(false));
$("#cart-items").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-qty]");
  if (!btn) return;
  const line = cart.load()[btn.dataset.qty];
  if (line) cart.setQty(btn.dataset.qty, line.qty + Number(btn.dataset.delta));
});
$("#checkout").addEventListener("click", () => {
  cart.save({});
  openCart(false);
  toast("Pesanan demo dibuat. Tidak ada pembayaran sungguhan.");
});
window.addEventListener("hashchange", route);

renderCart();
loadFacets().then(loadProducts);
route();

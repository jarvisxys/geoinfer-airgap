/* GeoInfer web UI: offline, no external libraries. */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const TILE_STRIDE = 448; // tile 512 - overlap 64: used for the tile estimate

// ---------------------------------------------------------------- API helpers
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  }
  const r = await fetch(path, init);
  const isJson = (r.headers.get("content-type") || "").includes("json");
  const body = isJson ? await r.json() : null;
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    if (body && body.detail) {
      msg = typeof body.detail === "string" ? body.detail
        : body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    }
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return body;
}

let toastTimer = null;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === "bad" ? 6000 : 3500);
}

const fmt = {
  km2(m2) { return m2 >= 1e5 ? `${(m2 / 1e6).toFixed(2)} km²` : `${Math.round(m2).toLocaleString()} m²`; },
  px(n) { return n >= 1e6 ? `${(n / 1e6).toFixed(1)} Mpx` : `${Math.round(n).toLocaleString()} px`; },
  s(x) { return x == null ? "–" : x < 60 ? `${x.toFixed(1)} s` : `${Math.floor(x / 60)} min ${Math.round(x % 60)} s`; },
  time(iso) { return iso ? new Date(iso).toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : ""; },
};

// ---------------------------------------------------------------- state
const S = {
  status: null, scenes: [], scene: null, meta: null, layer: "optical",
  tool: "rect", aoi: null, draft: null, drawing: false,
  mode: null, prio: 0, job: null, overlay: null, showOverlay: true, fill: true,
  view: { cx: 0, cy: 0, z: 0 },
};

// ---------------------------------------------------------------- UTM -> lon/lat (cursor readout)
function utmToLonLat(E, N, zone, south) {
  const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996;
  const e2 = f * (2 - f), ep2 = e2 / (1 - e2);
  const x = E - 500000, y = south ? N - 10000000 : N;
  const mu = y / k0 / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 ** 3 / 256));
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const p1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu)
    + (21 * e1 * e1 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu)
    + (151 * e1 ** 3 / 96) * Math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * Math.sin(8 * mu);
  const sin = Math.sin(p1), cos = Math.cos(p1), tan = Math.tan(p1);
  const C1 = ep2 * cos * cos, T1 = tan * tan;
  const N1 = a / Math.sqrt(1 - e2 * sin * sin);
  const R1 = a * (1 - e2) / Math.pow(1 - e2 * sin * sin, 1.5);
  const D = x / (N1 * k0);
  const lat = p1 - (N1 * tan / R1) * (D * D / 2
    - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24
    + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720);
  const lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
    + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) / cos;
  return [(zone - 1) * 6 - 177 + lon * 180 / Math.PI, lat * 180 / Math.PI];
}
function pxToLonLat(c, r) {
  const m = S.meta;
  if (!m || !m.epsg || !((m.epsg > 32600 && m.epsg <= 32660) || (m.epsg > 32700 && m.epsg <= 32760))) return null;
  const [a, b, cc, d, e, f] = m.transform;
  const E = a * c + b * r + cc, N = d * c + e * r + f;
  return utmToLonLat(E, N, m.epsg % 100, m.epsg > 32700);
}

// ---------------------------------------------------------------- map viewer
const canvas = $("#map");
const ctx = canvas.getContext("2d");
const tiles = new Map();
let W = 0, H = 0, dpr = 1, drawQueued = false;

function resize() {
  const r = canvas.getBoundingClientRect();
  dpr = window.devicePixelRatio || 1;
  W = r.width; H = r.height;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  requestDraw();
}
window.addEventListener("resize", resize);

const scale = () => Math.pow(2, S.view.z - S.meta.max_zoom); // screen px per scene px
const toScene = (sx, sy) => [S.view.cx + (sx - W / 2) / scale(), S.view.cy + (sy - H / 2) / scale()];
const toScreen = (x, y) => [(x - S.view.cx) * scale() + W / 2, (y - S.view.cy) * scale() + H / 2];

function zoomLimits() {
  const fit = S.meta.max_zoom + Math.log2(Math.min(W / S.meta.width, H / S.meta.height));
  return [Math.min(fit, S.meta.max_zoom) - 1, S.meta.max_zoom + 2];
}
function fitScene() {
  if (!S.meta) return;
  S.view.cx = S.meta.width / 2; S.view.cy = S.meta.height / 2;
  S.view.z = S.meta.max_zoom + Math.log2(0.94 * Math.min(W / S.meta.width, H / S.meta.height));
  requestDraw();
}
function fitBox(x0, y0, x1, y1) {
  S.view.cx = (x0 + x1) / 2; S.view.cy = (y0 + y1) / 2;
  const s = 0.8 * Math.min(W / Math.max(1, x1 - x0), H / Math.max(1, y1 - y0));
  const [lo, hi] = zoomLimits();
  S.view.z = Math.max(lo, Math.min(hi, S.meta.max_zoom + Math.log2(s)));
  requestDraw();
}
function zoomAt(sx, sy, dz) {
  const [lo, hi] = zoomLimits();
  const nz = Math.max(lo, Math.min(hi, S.view.z + dz));
  const [x, y] = toScene(sx, sy);
  S.view.z = nz;
  S.view.cx = x - (sx - W / 2) / scale(); S.view.cy = y - (sy - H / 2) / scale();
  requestDraw();
}

function tileImg(level, tx, ty) {
  const key = `${S.scene.id}/${S.layer}/${level}/${tx}/${ty}`;
  let t = tiles.get(key);
  if (!t) {
    t = new Image();
    t.onload = requestDraw;
    t.src = `/v1/scenes/${encodeURIComponent(S.scene.id)}/tiles/${S.layer}/${level}/${tx}/${ty}.png`;
    tiles.set(key, t);
    if (tiles.size > 1500) tiles.delete(tiles.keys().next().value);
  }
  return t;
}
function drawLevel(level, load) {
  const m = S.meta, span = m.tile * Math.pow(2, m.max_zoom - level);
  const [x0, y0] = toScene(0, 0), [x1, y1] = toScene(W, H);
  const n = Math.ceil(m.width / span), k = Math.ceil(m.height / span);
  for (let ty = Math.max(0, Math.floor(y0 / span)); ty <= Math.min(k - 1, Math.floor(y1 / span)); ty++) {
    for (let tx = Math.max(0, Math.floor(x0 / span)); tx <= Math.min(n - 1, Math.floor(x1 / span)); tx++) {
      const key = `${S.scene.id}/${S.layer}/${level}/${tx}/${ty}`;
      const img = load ? tileImg(level, tx, ty) : tiles.get(key);
      if (!img || !img.complete || !img.naturalWidth) continue;
      const [sx, sy] = toScreen(tx * span, ty * span);
      const sz = span * scale();
      ctx.drawImage(img, sx, sy, sz + 0.5, sz + 0.5);
    }
  }
}
function pathRing(ring, close = true) {
  ring.forEach(([x, y], i) => {
    const [sx, sy] = toScreen(x, y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  });
  if (close) ctx.closePath();
}
function draw() {
  drawQueued = false;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!S.meta) return;
  const m = S.meta;
  const level = Math.max(0, Math.min(m.max_zoom, Math.ceil(S.view.z - 0.25)));
  ctx.imageSmoothingEnabled = S.view.z < m.max_zoom;
  for (let l = 0; l < level; l++) drawLevel(l, l === 0);
  drawLevel(level, true);
  // scene outline
  const [ax, ay] = toScreen(0, 0), [bx, by] = toScreen(m.width, m.height);
  ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1;
  ctx.strokeRect(ax, ay, bx - ax, by - ay);
  // buildings
  if (S.overlay && S.showOverlay) {
    ctx.lineWidth = Math.max(1, Math.min(2, scale() * 2));
    ctx.strokeStyle = "#eb6834"; ctx.fillStyle = "rgba(235,104,52,.35)";
    for (const b of S.overlay.buildings) {
      ctx.beginPath();
      b.rings.forEach((r) => pathRing(r));
      if (S.fill) ctx.fill("evenodd");
      ctx.stroke();
    }
  }
  // area of interest
  const ring = S.draft || S.aoi;
  if (ring && ring.length) {
    ctx.beginPath(); pathRing(ring, !S.drawing || S.tool !== "poly");
    ctx.fillStyle = "rgba(242,165,65,.14)"; ctx.fill();
    ctx.strokeStyle = "#f2a541"; ctx.lineWidth = 2.5; ctx.setLineDash(S.draft ? [6, 4] : []);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#f2a541";
    for (const [x, y] of ring) { const [sx, sy] = toScreen(x, y); ctx.fillRect(sx - 3, sy - 3, 6, 6); }
  }
  updateScale();
}
function requestDraw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(draw); } }

function updateScale() {
  const mPerPx = S.meta.gsd_m / scale();
  const target = 110 * mPerPx;
  const p = Math.pow(10, Math.floor(Math.log10(target)));
  const nice = [1, 2, 5, 10].map((k) => k * p).filter((v) => v <= target).pop() || p;
  $("#scale-bar").style.width = `${nice / mPerPx}px`;
  $("#scale-label").textContent = nice >= 1000 ? `${nice / 1000} km` : `${nice} m`;
}

// ---------------------------------------------------------------- map interaction
let drag = null, spaceDown = false;
const evPos = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
const clampPt = ([x, y]) => [Math.max(0, Math.min(S.meta.width, x)), Math.max(0, Math.min(S.meta.height, y))];
const activeTool = () => (spaceDown ? "pan" : S.tool);

canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  if (!S.meta) return;
  canvas.setPointerCapture(e.pointerId);
  const [sx, sy] = evPos(e);
  const tool = e.button !== 0 ? "pan" : activeTool();
  if (tool === "rect" && drawingAllowed()) {
    const p = clampPt(toScene(sx, sy));
    drag = { kind: "rect", p0: p }; S.drawing = true; S.draft = [p, p, p, p];
  } else if (tool === "poly" && drawingAllowed()) {
    const p = clampPt(toScene(sx, sy));
    if (!S.drawing) { S.drawing = true; S.draft = [p, p]; } else { S.draft.splice(S.draft.length - 1, 0, p); }
  } else {
    drag = { kind: "pan", sx, sy, cx: S.view.cx, cy: S.view.cy, moved: false };
  }
  requestDraw();
});
canvas.addEventListener("pointermove", (e) => {
  if (!S.meta) return;
  const [sx, sy] = evPos(e);
  const [x, y] = toScene(sx, sy);
  const ll = pxToLonLat(x, y);
  $("#cursor-ll").textContent = ll ? `${ll[1].toFixed(5)}°N  ${ll[0].toFixed(5)}°E`
    : `col ${Math.round(x)}, row ${Math.round(y)}`;
  if (drag && drag.kind === "pan") {
    if (Math.abs(sx - drag.sx) + Math.abs(sy - drag.sy) > 3) drag.moved = true;
    S.view.cx = drag.cx - (sx - drag.sx) / scale(); S.view.cy = drag.cy - (sy - drag.sy) / scale();
    requestDraw();
  } else if (drag && drag.kind === "rect") {
    const [x0, y0] = drag.p0, [x1, y1] = clampPt([x, y]);
    S.draft = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    requestDraw();
  } else if (S.drawing && S.tool === "poly") {
    S.draft[S.draft.length - 1] = clampPt([x, y]);
    requestDraw();
  }
});
canvas.addEventListener("pointerup", (e) => {
  if (!S.meta) return;
  if (drag && drag.kind === "rect") {
    const [x0, y0] = S.draft[0], [x1, y1] = S.draft[2];
    S.drawing = false; drag = null;
    if (Math.abs(x1 - x0) * scale() < 6 || Math.abs(y1 - y0) * scale() < 6) { S.draft = null; requestDraw(); return; }
    const [a, b] = [Math.min(x0, x1), Math.min(y0, y1)], [c, d] = [Math.max(x0, x1), Math.max(y0, y1)];
    setAoi([[a, b], [c, b], [c, d], [a, d]]);
    S.draft = null;
  } else if (drag && drag.kind === "pan") {
    const click = !drag.moved && e.button === 0;
    drag = null;
    if (click) clickAt(...evPos(e));
  }
});
canvas.addEventListener("dblclick", (e) => { e.preventDefault(); finishPolygon(); });
canvas.addEventListener("wheel", (e) => {
  if (!S.meta) return;
  e.preventDefault();
  zoomAt(...evPos(e), -Math.sign(e.deltaY) * (e.ctrlKey ? 0.15 : 0.5));
}, { passive: false });

function finishPolygon() {
  if (!S.drawing || S.tool !== "poly") return;
  const pts = S.draft.slice(0, -1).filter((p, i, a) => i === 0 || Math.hypot(p[0] - a[i - 1][0], p[1] - a[i - 1][1]) > 0.5);
  S.drawing = false; S.draft = null;
  if (pts.length >= 3) setAoi(pts); else { toast("A polygon needs at least 3 points.", "bad"); requestDraw(); }
}
function cancelDrawing() { S.drawing = false; S.draft = null; drag = null; requestDraw(); }

function clickAt(sx, sy) {
  const pop = $("#popup");
  pop.hidden = true;
  if (!S.overlay || !S.showOverlay) return;
  const [x, y] = toScene(sx, sy);
  const hit = S.overlay.buildings.find((b) => pointInRing(x, y, b.rings[0]));
  if (!hit) return;
  pop.innerHTML = `<b>Building ${hit.id == null ? "" : hit.id}</b>${fmt.km2(hit.area_m2 || 0)} · confidence ${
    hit.score == null ? "–" : `${Math.round(hit.score * 100)} %`}`;
  pop.style.left = `${sx}px`; pop.style.top = `${sy}px`; pop.hidden = false;
}
function pointInRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  if (e.key === "Escape") { if (!$("#help").hidden) closeHelp(); else cancelDrawing(); }
  else if (e.key === "Enter") finishPolygon();
  else if (e.key === " ") { spaceDown = true; canvas.style.cursor = "grab"; e.preventDefault(); }
  else if (!S.meta) return;
  else if (e.key === "+" || e.key === "=") zoomAt(W / 2, H / 2, 0.5);
  else if (e.key === "-") zoomAt(W / 2, H / 2, -0.5);
  else if (e.key === "f" || e.key === "F") fitScene();
  else if ("hHrRpP".includes(e.key) && drawingAllowed()) setTool({ h: "pan", r: "rect", p: "poly" }[e.key.toLowerCase()]);
});
document.addEventListener("keyup", (e) => { if (e.key === " ") { spaceDown = false; updateCursor(); } });

$("#zoom-in").onclick = () => zoomAt(W / 2, H / 2, 0.5);
$("#zoom-out").onclick = () => zoomAt(W / 2, H / 2, -0.5);
$("#zoom-fit").onclick = fitScene;

// ---------------------------------------------------------------- steps
function setStep(n, state) { // state: locked | open | done
  const el = $(`#step-${n}`);
  el.classList.toggle("locked", state === "locked");
  el.classList.toggle("done", state === "done");
}
function drawingAllowed() { return !!S.meta && !S.job; }
function updateCursor() {
  canvas.style.cursor = !S.meta ? "default" : activeTool() === "pan" || !drawingAllowed() ? "grab" : "crosshair";
}

// ---------------------------------------------------------------- step 1: scenes
async function loadScenes(selectId) {
  try {
    S.scenes = await api("/v1/scenes");
  } catch (e) { $("#scene-list").innerHTML = `<p class="msg bad">${e.message}</p>`; return; }
  const box = $("#scene-list");
  if (!S.scenes.length) {
    box.innerHTML = `<p class="muted">No scenes yet. Add an image under <em>Add images</em>.</p>`;
    return;
  }
  box.innerHTML = "";
  for (const sc of S.scenes) {
    const b = document.createElement("button");
    b.className = "scene-card" + (S.scene && S.scene.id === sc.id ? " active" : "");
    const w = (sc.width * sc.gsd_m / 1000).toFixed(1), h = (sc.height * sc.gsd_m / 1000).toFixed(1);
    b.innerHTML = `<span class="t">${sc.id}<span class="badge ${sc.sensor}">${sc.sensor}</span></span>
      <span class="d">${w} × ${h} km · ${sc.gsd_m} m pixels</span>
      <span class="d">${sc.bounds[1].toFixed(3)}°N, ${sc.bounds[0].toFixed(3)}°E</span>`;
    b.onclick = () => selectScene(sc.id);
    box.appendChild(b);
  }
  if (selectId) selectScene(selectId);
}

async function selectScene(id, keepJob = false) {
  try {
    S.meta = await api(`/v1/scenes/${encodeURIComponent(id)}/map`);
  } catch (e) { toast(`Could not open ${id}: ${e.message}`, "bad"); return; }
  S.scene = S.scenes.find((s) => s.id === id) || { id };
  S.layer = S.meta.layers[0];
  if (!keepJob) { S.aoi = null; S.overlay = null; S.job = null; }
  $$(".scene-card").forEach((c) => c.classList.toggle("active", c.querySelector(".t").firstChild.textContent === id));
  $("#map-empty").hidden = true; $("#map-tools").hidden = false; $("#map-info").hidden = false; $("#legend").hidden = false;
  renderLayers(); renderModes(); fitScene(); updateAoi();
  setStep(1, "done");
  setStep(2, keepJob ? "done" : "open");
  if (!keepJob) { setStep(3, "locked"); setStep(4, "locked"); resetResults(); setTool("rect"); }
  updateCursor();
}
function renderLayers() {
  const box = $("#layer-switch");
  box.innerHTML = "";
  if (S.meta.layers.length < 2) return;
  for (const l of S.meta.layers) {
    const b = document.createElement("button");
    b.textContent = l; b.className = l === S.layer ? "active" : ""; b.title = `Show the ${l} image`;
    b.onclick = () => { S.layer = l; renderLayers(); requestDraw(); };
    box.appendChild(b);
  }
}

// ---------------------------------------------------------------- step 2: area
function setTool(t) {
  S.tool = t; cancelDrawing();
  $$("[data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
  $("#tool-hint").textContent = {
    pan: "Drag to move the map, scroll to zoom. Choose a drawing tool to select an area.",
    rect: "Drag on the map to draw a rectangle. Right-drag or hold Space to move the map.",
    poly: "Click to add corners; double-click or press Enter to finish, Esc to cancel.",
  }[t] || "";
  updateCursor();
}
$$("[data-tool]").forEach((b) => b.addEventListener("click", () => {
  if (!S.meta) return;
  if (b.dataset.tool === "view") {
    const [x0, y0] = clampPt(toScene(0, 0)), [x1, y1] = clampPt(toScene(W, H));
    setAoi([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    return;
  }
  setTool(b.dataset.tool);
}));

function polyArea(r) {
  let a = 0;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) a += (r[j][0] + r[i][0]) * (r[j][1] - r[i][1]);
  return Math.abs(a / 2);
}
function setAoi(ring) {
  S.aoi = ring.map(clampPt);
  setTool("pan");
  updateAoi();
  requestDraw();
}
function updateAoi() {
  const box = $("#aoi-box");
  if (!S.aoi) { box.hidden = true; setStep(2, S.meta ? "open" : "locked"); setStep(3, "locked"); $("#run-btn").disabled = true; return; }
  box.hidden = false;
  const xs = S.aoi.map((p) => p[0]), ys = S.aoi.map((p) => p[1]);
  const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys);
  const px = w * h, area = polyArea(S.aoi) * S.meta.gsd_m ** 2;
  const nTiles = Math.ceil(w / TILE_STRIDE) * Math.ceil(h / TILE_STRIDE);
  $("#aoi-area").textContent = fmt.km2(area);
  $("#aoi-size").textContent = fmt.px(px);
  $("#aoi-tiles").textContent = nTiles.toLocaleString();
  const max = S.status ? S.status.max_aoi_px : 150e6;
  const msg = $("#aoi-msg");
  let ok = true;
  if (area < 100) { msg.textContent = "The area is very small. Draw a larger one."; msg.className = "msg bad"; ok = false; }
  else if (px > max) { msg.textContent = `Too large: at most ${fmt.px(max)} (about ${Math.sqrt(max * S.meta.gsd_m ** 2 / 1e6).toFixed(1)} km square).`; msg.className = "msg bad"; ok = false; }
  else if (px > 16e6) { msg.textContent = `Large area: expect a few minutes (${nTiles} tiles).`; msg.className = "msg warn"; }
  else { msg.textContent = "Area ready."; msg.className = "msg ok"; }
  setStep(2, ok ? "done" : "open");
  setStep(3, ok ? "open" : "locked");
  $("#run-btn").disabled = !ok || !!S.job;
  if (ok) reveal("#step-3");
}
function reveal(sel) { setTimeout(() => $(sel).scrollIntoView({ behavior: "smooth", block: "nearest" }), 60); }
$("#aoi-clear").onclick = () => { S.aoi = null; setTool("rect"); updateAoi(); requestDraw(); };

$("#c-apply").onclick = async () => {
  if (!S.meta) return;
  const v = ["#c-w", "#c-s", "#c-e", "#c-n"].map((id) => parseFloat($(id).value));
  if (v.some(Number.isNaN)) { toast("Enter all four coordinates in decimal degrees.", "bad"); return; }
  const [w, s, e, n] = v;
  if (w >= e || s >= n) { toast("West must be less than East, South less than North.", "bad"); return; }
  try {
    const r = await api(`/v1/scenes/${encodeURIComponent(S.scene.id)}/convert`, { json: { to: "px", points: [[w, n], [e, n], [e, s], [w, s]] } });
    const xs = r.points.map((p) => p[0]), ys = r.points.map((p) => p[1]);
    if (Math.max(...xs) < 0 || Math.min(...xs) > S.meta.width || Math.max(...ys) < 0 || Math.min(...ys) > S.meta.height) {
      toast("Those coordinates are outside this scene.", "bad"); return;
    }
    setAoi(r.points); fitBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys));
  } catch (err) { toast(err.message, "bad"); }
};

// ---------------------------------------------------------------- step 3: run
const MODE_HINT = {
  optical: "Most accurate on clear imagery.",
  sar: "Radar: works through cloud and at night, less precise outlines.",
  fused: "Radar + optical: the most robust when there are clouds.",
};
function renderModes() {
  const box = $("#mode-seg");
  box.innerHTML = "";
  const models = S.status ? S.status.models : [];
  S.mode = S.meta.modes.includes("fused") ? "fused" : S.meta.modes[0];
  for (const m of S.meta.modes) {
    const b = document.createElement("button");
    b.className = "seg-btn" + (m === S.mode ? " active" : "");
    b.textContent = m === "sar" ? "SAR" : m[0].toUpperCase() + m.slice(1);
    b.dataset.mode = m;
    b.onclick = () => { S.mode = m; $$("#mode-seg .seg-btn").forEach((x) => x.classList.toggle("active", x === b)); $("#mode-hint").textContent = MODE_HINT[m]; };
    if (models.length && !models.some((id) => id.includes(`_${m}_`))) { b.disabled = true; b.title = "no model installed"; }
    box.appendChild(b);
  }
  $("#mode-hint").textContent = MODE_HINT[S.mode];
}
$$("#prio-seg .seg-btn").forEach((b) => b.addEventListener("click", () => {
  S.prio = +b.dataset.prio;
  $$("#prio-seg .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
}));

$("#run-btn").onclick = async () => {
  const btn = $("#run-btn"), msg = $("#run-msg");
  btn.disabled = true; btn.textContent = "Running…"; msg.textContent = "Submitting…"; msg.className = "msg";
  try {
    const ll = await api(`/v1/scenes/${encodeURIComponent(S.scene.id)}/convert`, { json: { to: "lonlat", points: S.aoi } });
    const ring = [...ll.points, ll.points[0]];
    const r = await api("/v1/jobs", { json: { scene_id: S.scene.id, task: "segment_buildings", sensor_mode: S.mode, priority: S.prio, aoi: { type: "Polygon", coordinates: [ring] } } });
    msg.textContent = ""; setStep(3, "done"); startJob(r.id);
  } catch (e) { msg.textContent = e.message; msg.className = "msg bad"; btn.disabled = false; btn.textContent = "Find buildings"; }
};
function runReady(again) {
  const btn = $("#run-btn");
  btn.textContent = again ? "Run again (compare another model)" : "Find buildings";
  btn.disabled = !S.aoi;
}

// ---------------------------------------------------------------- step 4: results
const STAGES = [["prepare_s", "Read + normalise", "#2a78d6"], ["infer_s", "Model (GPU)", "#13828b"],
  ["stitch_s", "Stitch", "#1baf7a"], ["postprocess_s", "Clean + polygons", "#eda100"], ["write_s", "Write", "#eb6834"]];
let pollTimer = null;
function resetResults() {
  clearTimeout(pollTimer);
  $("#job-progress").hidden = true; $("#result-box").hidden = true; $("#result-empty").hidden = false;
  $("#popup").hidden = true;
}
function startJob(id) {
  S.job = { id, t0: Date.now() };
  setStep(4, "open"); resetResults();
  $("#job-progress").hidden = false; $("#result-empty").hidden = true;
  $("#run-btn").disabled = true; $("#run-btn").textContent = "Running…";
  updateCursor(); reveal("#step-4"); pollJob();
}
async function pollJob() {
  if (!S.job) return;
  let j;
  try { j = await api(`/v1/jobs/${S.job.id}`); } catch (e) { pollTimer = setTimeout(pollJob, 2000); return; }
  const order = ["queued", "running", "ready"];
  $$(".pipeline span").forEach((s) => {
    s.classList.toggle("on", order.indexOf(s.dataset.st) <= order.indexOf(j.status));
    s.classList.toggle("fail", j.status === "failed" && s.dataset.st === "ready");
  });
  const bar = $("#job-bar").parentElement;
  bar.classList.toggle("indeterminate", j.status === "running");
  $("#job-bar").style.width = j.status === "queued" ? "8%" : j.status === "ready" ? "100%" : "";
  const el = Math.round((Date.now() - S.job.t0) / 1000);
  if (j.status === "failed") {
    $("#job-line").innerHTML = `<span class="msg bad">Failed: ${j.error || "unknown error"}</span>`;
    S.job = null; updateCursor(); runReady(false); setStep(3, "open");
    return;
  }
  if (j.status !== "ready") {
    $("#job-line").textContent = j.status === "queued" ? `Waiting for the worker… ${el} s` : `Processing on the GPU… ${el} s`;
    pollTimer = setTimeout(pollJob, 1500);
    return;
  }
  $("#job-line").textContent = `Done in ${fmt.s(j.timings ? j.timings.total_s : el)} · ${j.n_tiles} tiles`;
  await showResult(j);
}
async function showResult(j) {
  try { S.overlay = await api(`/v1/jobs/${j.id}/overlay`); } catch (e) { toast(e.message, "bad"); return; }
  setStep(4, "done");
  $("#result-box").hidden = false; $("#result-empty").hidden = true;
  $("#r-count").textContent = S.overlay.count.toLocaleString();
  $("#r-area").textContent = fmt.km2(S.overlay.total_area_m2);
  const aoiArea = S.aoi ? polyArea(S.aoi) * S.meta.gsd_m ** 2 : null;
  $("#r-share").textContent = aoiArea ? `${(100 * S.overlay.total_area_m2 / aoiArea).toFixed(1)} %` : "–";
  const t = j.timings || {};
  const total = STAGES.reduce((a, [k]) => a + (t[k] || 0), 0) || 1;
  $("#r-timings").innerHTML = `<div class="timings">${STAGES.map(([k, , c]) => `<span style="width:${100 * (t[k] || 0) / total}%;background:${c}" title="${k}"></span>`).join("")}</div>
    <div class="timing-legend">${STAGES.map(([k, n, c]) => `<span><i style="background:${c}"></i>${n} ${fmt.s(t[k] || 0)}</span>`).join("")}</div>`;
  $("#dl-geojson").href = `/v1/jobs/${j.id}/result?format=geojson`;
  $("#dl-geojson").setAttribute("download", `${j.id}_buildings.geojson`);
  $("#dl-mask").href = `/v1/jobs/${j.id}/result?format=cog`;
  S.job = null; updateCursor(); runReady(true); reveal("#step-4");
  const all = S.overlay.buildings.flatMap((b) => b.rings[0]);
  if (all.length && !S.aoi) fitBox(Math.min(...all.map((p) => p[0])), Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[0])), Math.max(...all.map((p) => p[1])));
  requestDraw();
  toast(`${S.overlay.count} buildings found.`, "ok");
}
$("#r-show").onchange = (e) => { S.showOverlay = e.target.checked; requestDraw(); };
$("#r-fill").onchange = (e) => { S.fill = e.target.checked; requestDraw(); };
$("#new-area").onclick = () => {
  S.overlay = null; S.aoi = null; S.job = null; resetResults();
  setStep(4, "locked"); setTool("rect"); updateAoi(); runReady(false); requestDraw();
};

// ---------------------------------------------------------------- tabs, help
function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "upload") loadInbox();
  if (name === "history") loadHistory();
}
$$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));
function closeHelp() { $("#help").hidden = true; }
$("#help-btn").onclick = () => { $("#help").hidden = false; };
$("#help-close").onclick = closeHelp;
$("#help").addEventListener("click", (e) => { if (e.target.id === "help") closeHelp(); });

// ---------------------------------------------------------------- upload / inbox
const dz = $("#dropzone"), fileInput = $("#file-input");
dz.onclick = () => fileInput.click();
dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } };
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });
fileInput.onchange = () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); fileInput.value = ""; };

function putFile(file, overwrite) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const t0 = Date.now();
    xhr.open("PUT", `/v1/inbox/${encodeURIComponent(file.name)}${overwrite ? "?overwrite=true" : ""}`);
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round(100 * e.loaded / e.total);
      $("#up-bar").style.width = `${pct}%`; $("#up-pct").textContent = `${pct} %`;
      const mbps = e.loaded / 1048576 / Math.max(0.1, (Date.now() - t0) / 1000);
      $("#up-speed").textContent = `${(e.loaded / 1048576).toFixed(0)} of ${(e.total / 1048576).toFixed(0)} MB · ${mbps.toFixed(0)} MB/s`;
    };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(Object.assign(new Error(body && body.detail ? body.detail : `upload failed (${xhr.status})`), { status: xhr.status }));
    };
    xhr.onerror = () => reject(new Error("upload failed: the service did not answer"));
    xhr.send(file);
  });
}
async function uploadFile(file) {
  if (!/\.tiff?$/i.test(file.name)) { toast("Only GeoTIFF files (.tif, .tiff) can be added.", "bad"); return; }
  const maxGb = S.status ? S.status.max_upload_gb : 20;
  if (file.size > maxGb * 2 ** 30) { toast(`The file is larger than ${maxGb} GB.`, "bad"); return; }
  $("#up-progress").hidden = false; $("#up-name").textContent = file.name;
  $("#up-bar").style.width = "0"; $("#up-pct").textContent = ""; $("#up-speed").textContent = "";
  try {
    let r;
    try { r = await putFile(file, false); } catch (e) {
      if (e.status !== 409 || !confirm(`${file.name} is already in the inbox. Replace it?`)) throw e;
      r = await putFile(file, true);
    }
    $("#up-speed").textContent = `Added as ${r.name} (${r.size_mb} MB). Checking the image…`;
    await inspectFile(r.name);
    loadInbox();
  } catch (e) { $("#up-speed").innerHTML = `<span class="msg bad">${e.message}</span>`; }
}

const ICON = { pass: "✓", info: "i", warn: "!", fail: "✕" };
let prepTimer = null;
async function inspectFile(name) {
  const card = $("#inspect-card");
  card.hidden = false; $("#ins-title").textContent = name; $("#ins-sub").textContent = "Checking…";
  $("#ins-checks").innerHTML = ""; $("#prep-row").hidden = true; $("#reg-form").hidden = true;
  let r;
  try { r = await api(`/v1/inbox/${encodeURIComponent(name)}/inspect`); } catch (e) { $("#ins-sub").textContent = e.message; return; }
  $("#ins-sub").textContent = r.width ? `${r.width.toLocaleString()} × ${r.height.toLocaleString()} px · ${r.bands} bands · ${r.dtype}${r.gsd_m ? ` · ${r.gsd_m} m pixels` : ""}` : "";
  $("#ins-checks").innerHTML = r.checks.map((c) => `<li class="${c.level}"><span class="ic">${ICON[c.level]}</span><span>${c.message}</span></li>`).join("");
  const needsPrep = r.checks.some((c) => c.fixable && (c.level === "fail" || c.level === "warn"));
  const canPrep = r.checks.some((c) => c.fixable) && !r.checks.some((c) => c.level === "fail" && !c.fixable);
  $("#prep-row").hidden = !canPrep || name.endsWith("_prepared.tif");
  $("#prep-btn").textContent = needsPrep ? "Prepare image (needed)" : "Prepare image (optional, faster)";
  $("#prep-btn").onclick = () => prepareFile(name);
  if (r.ok) showRegister(name, r); else $("#reg-msg").textContent = "";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
async function prepareFile(name) {
  $("#prep-btn").disabled = true; $("#prep-msg").textContent = "Preparing… this can take a few minutes for large images.";
  try { await api(`/v1/inbox/${encodeURIComponent(name)}/prepare`, { method: "POST" }); } catch (e) { $("#prep-msg").textContent = e.message; $("#prep-btn").disabled = false; return; }
  const poll = async () => {
    const st = await api(`/v1/inbox/${encodeURIComponent(name)}/prepare`);
    if (st.state === "running") { prepTimer = setTimeout(poll, 1000); return; }
    $("#prep-btn").disabled = false;
    if (st.state === "done") { toast(`Prepared: ${st.output}`, "ok"); loadInbox(); inspectFile(st.output); }
    else $("#prep-msg").innerHTML = `<span class="msg bad">${st.error || "failed"}</span>`;
  };
  poll();
}
async function showRegister(name, r) {
  const f = $("#reg-form");
  f.hidden = false; $("#reg-msg").textContent = "";
  $("#reg-id").value = name.replace(/\.tiff?$/i, "").replace(/[^A-Za-z0-9_.-]/g, "_").slice(0, 60);
  $("#reg-sensor").value = r.suggested_sensor || "optical";
  const inbox = await api("/v1/inbox").catch(() => []);
  $("#reg-sar").innerHTML = inbox.filter((x) => x.name !== name).map((x) => `<option>${x.name}</option>`).join("");
  const sync = () => {
    const s = $("#reg-sensor").value;
    $("#reg-sar-row").hidden = s !== "fused"; $("#reg-orient-row").hidden = s === "optical";
  };
  $("#reg-sensor").onchange = sync; sync();
  f.onsubmit = async (e) => {
    e.preventDefault();
    const sensor = $("#reg-sensor").value;
    const body = { id: $("#reg-id").value.trim(), path: name, sensor };
    if (sensor === "fused") body.sar_path = $("#reg-sar").value;
    if (sensor !== "optical") body.orientation = +$("#reg-orient").value;
    $("#reg-msg").textContent = "Adding… (computing a checksum of the file)"; $("#reg-msg").className = "msg";
    try {
      await api("/v1/scenes", { json: body });
      toast(`Scene ${body.id} added.`, "ok");
      showTab("workflow"); await loadScenes(body.id);
    } catch (err) { $("#reg-msg").textContent = err.message; $("#reg-msg").className = "msg bad"; }
  };
}
async function loadInbox() {
  const box = $("#inbox-list");
  let files;
  try { files = await api("/v1/inbox"); } catch (e) { box.innerHTML = `<p class="msg bad">${e.message}</p>`; return; }
  if (!files.length) { box.innerHTML = `<p class="muted">The inbox is empty.</p>`; return; }
  box.innerHTML = "";
  for (const f of files) {
    const d = document.createElement("div");
    d.className = "item";
    d.innerHTML = `<span class="n">${f.name}</span><span class="acts"></span>
      <span class="s">${f.size_mb.toLocaleString()} MB · ${f.modified}${f.scenes.length ? ` · scene: ${f.scenes.join(", ")}` : " · not added yet"}</span>`;
    const acts = d.querySelector(".acts");
    const chk = document.createElement("button"); chk.className = "link"; chk.textContent = "Check";
    chk.onclick = () => inspectFile(f.name); acts.appendChild(chk);
    if (!f.scenes.length) {
      const del = document.createElement("button"); del.className = "link danger"; del.textContent = "Delete";
      del.onclick = async () => {
        if (!confirm(`Delete ${f.name} from the inbox?`)) return;
        try { await fetch(`/v1/inbox/${encodeURIComponent(f.name)}`, { method: "DELETE" }); loadInbox(); } catch (e) { toast(e.message, "bad"); }
      };
      acts.appendChild(del);
    }
    box.appendChild(d);
  }
}

// ---------------------------------------------------------------- history
async function loadHistory() {
  const box = $("#job-list");
  let jobs;
  try { jobs = await api("/v1/jobs?limit=50"); } catch (e) { box.innerHTML = `<p class="msg bad">${e.message}</p>`; return; }
  if (!jobs.length) { box.innerHTML = `<p class="muted">No jobs yet.</p>`; return; }
  box.innerHTML = "";
  const prio = (p) => (p >= 10 ? "urgent" : p >= 5 ? "high" : "normal");
  for (const j of jobs) {
    const d = document.createElement("div");
    d.className = "item clickable";
    d.innerHTML = `<span class="n">${j.scene_id} · ${j.sensor_mode}</span><span class="badge ${j.status}">${j.status}</span>
      <span class="s">${fmt.time(j.created_at)} · ${prio(j.priority)} priority${j.total_s ? ` · ${fmt.s(j.total_s)}` : ""}${j.n_tiles ? ` · ${j.n_tiles} tiles` : ""}</span>`;
    d.title = j.error || "Open on the map";
    d.onclick = async () => {
      showTab("workflow");
      if (!S.scene || S.scene.id !== j.scene_id) await selectScene(j.scene_id, true);
      S.aoi = null; updateAoi(); setStep(2, "done"); setStep(3, "done");
      if (j.status === "ready") {
        setStep(4, "open"); resetResults(); $("#job-progress").hidden = true;
        await showResult(await api(`/v1/jobs/${j.id}`));
      } else startJob(j.id);
    };
    box.appendChild(d);
  }
}
$("#hist-refresh").onclick = loadHistory;

// ---------------------------------------------------------------- status chips
async function pollStatus() {
  try {
    const s = await api("/v1/status");
    const first = !S.status;
    S.status = s;
    const chip = (id, cls, text) => { const c = $(id); c.className = `chip ${cls}`; c.lastChild.textContent = text; };
    chip("#chip-service", "ok", "Service ready");
    const w = s.workers[0];
    chip("#chip-worker", s.worker ? "ok" : "warn",
      s.worker ? `Worker · ${(w && w.gpu ? w.gpu.replace(/NVIDIA |GeForce |with Max-Q Design/g, "") : "CPU").trim()}` : "Worker starting…");
    const q = s.queue;
    chip("#chip-queue", q.running || q.queued ? "warn" : "ok", q.running || q.queued ? `${q.running} running · ${q.queued} waiting` : "Queue empty");
    if (first && S.meta) { renderModes(); updateAoi(); }
  } catch (e) {
    const c = $("#chip-service"); c.className = "chip bad"; c.lastChild.textContent = "Service offline";
  }
  setTimeout(pollStatus, 4000);
}

// ---------------------------------------------------------------- start
resize();
pollStatus();
loadScenes();
updateCursor();

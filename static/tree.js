// @ts-nocheck
"use strict";

/* =========================
 *  Global State
 * ========================= */
let treeData = null;           // { name, parents[], children[], _collapsedParents, _collapsedChildren }
let focusNode = null;          // 現在のフォーカスノード（クリックで移動）
let expanded = new Set();      // 展開済みノード名
let selected = [];             // Scholar 検索用チップ
let centerStack = [];          // 中心履歴（戻る用）

let svg, rootG, contentG;      // SVGとその中身
let zoom;
const MIN_SCALE = 0.3, MAX_SCALE = 3.0;

let KNOWN_META = {};           // meta から取得する単一ソース
const metaCache = {};          // canon(term) -> {short, full, desc}
let tooltipEl = null;

// ラベル（角丸矩形）用
const LABEL_FONT_SIZE = 13;   // px
const LABEL_PAD_X = 12;       // 左右パディング
const LABEL_PAD_Y = 6;        // 上下パディング
const LABEL_MIN_W = 44;       // 最小幅
const LABEL_MIN_H = 28;       // 最小高さ
const LABEL_CORNER_R = 12;    // 角丸半径

/* =========================
 *  Utilities
 * ========================= */
function $(sel){ return document.querySelector(sel); }

function showToast(msg, ms=2200){
  const t = $("#toast");
  if(!t) return;
  t.textContent = msg;
  t.style.display = "block";
  setTimeout(()=> t.style.display = "none", ms);
}

function canon(s){
  return (s || "")
    .normalize("NFKC")
    .toLowerCase()
    .trim()
    .replace(/・/g, " ")
    .replace(/[\-‐-‒–—―]/g, "-")
    .replace(/[\s_/]+/g, " ")
    .replace(/[^\p{L}\p{N}\s-]/gu, ""); // ← 日本語を残す
}

function escapeHtml(s){
  return (s || "").replace(/[&<>"']/g, (m) => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;"
  }[m]));
}

function isAsciiLike(s){
  return /^[\x00-\x7F]+$/.test(s || "");
}

// Scholar 用クリーンナップ（※完全一致の引用は使わない）
function cleanTerm(s){
  return String(s || "")
    .replace(/["“”]+/g, "")   // 引用符を除去
    .replace(/\s+/g, " ")     // 連続空白を1つへ
    .trim();
}

/* =========================
 *  Chips helper (with logging)
 * ========================= */
function drawChips(){
  const box = $("#chips");
  if (!box) return;
  box.innerHTML = "";
  selected.forEach((term, idx) => {
    const span = document.createElement("span");
    span.className = "chip";
    span.title = "クリックで削除";
    span.textContent = term;
    span.onclick = ()=> {
      const removed = selected[idx];
      selected.splice(idx,1);
      drawChips();
      logUI("chip_remove", {
        term: removed,
        selected: [...selected]
      });
    };
    box.appendChild(span);
  });

  const enBtn = document.getElementById("searchEN");
  const jaBtn = document.getElementById("searchJA");
  const disabled = selected.length === 0;
  if (enBtn) enBtn.disabled = disabled;
  if (jaBtn) jaBtn.disabled = disabled;
}

// チップ追加をまとめて処理（ログ付き）
function addSelectedTerm(term, source){
  const t = (term || "").trim();
  if (!t) return;
  if (selected.includes(t)) return;
  selected.push(t);
  drawChips();
  logUI("chip_add", {
    term: t,
    selected: [...selected],
    source: source || null
  });
}

/* ラベル略称（略語優先） */
function computeShortLabel(name){
  const key = canon(name);
  if (key && metaCache[key]?.short) return metaCache[key].short;
  if (key && KNOWN_META[key]?.short) return KNOWN_META[key].short;

  const m1 = name.match(/^([A-Za-z][A-Za-z0-9\-]{2,10})[（(]/);
  if (m1) return m1[1];
  const m2 = name.match(/[（(]([A-Za-z][A-Za-z0-9\-]{2,10})[)）]$/);
  if (m2) return m2[1];
  const m3 = name.match(/[A-Z]{2,6}/);
  if (m3) return m3[0];
  return name.length > 12 ? name.slice(0,10) + "…" : name;
}

/* 用語メタの取得（初回のみ describe、既知メタ優先） */
async function describeTerm(term){
  const key = canon(term);
  if (!key) return { short: computeShortLabel(term), full: term, desc: "" };
  if (metaCache[key]) return metaCache[key];
  if (KNOWN_META[key]) { metaCache[key] = KNOWN_META[key]; return KNOWN_META[key]; }

  try{
    const res = await fetch("describe",{
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ term })
    });
    const meta = await res.json();
    const safe = {
      short: (meta.short || computeShortLabel(term)).slice(0, 20),
      full : meta.full || term,
      desc : (meta.desc || "").slice(0, 200)
    };
    metaCache[key] = safe;
    return safe;
  }catch(_){
    const fb = { short: computeShortLabel(term), full: term, desc: "" };
    metaCache[key] = fb;
    return fb;
  }
}

/* ★説明の先読み：ノード展開直後にバックグラウンドで describe を叩く */
function warmDescribe(terms){
  (terms || []).forEach(t => {
    const key = canon(t);
    if (!key) return;
    if (metaCache[key]) return;
    if (KNOWN_META[key]) return;
    // 結果は metaCache に入るので await しない
    describeTerm(t).catch(()=>{});
  });
}

/* =========================
 *  Translation (batch) via translate
 * ========================= */
async function translateTerms(terms){
  const uniq = Array.from(new Set((terms || []).map(t => (t||"").trim()).filter(Boolean)));
  if (!uniq.length) return {};
  try{
    const res = await fetch("translate", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ terms: uniq })
    });
    const obj = await res.json(); // { map: { "元語": ["en1","en2"], ... } }
    return obj?.map || {};
  }catch(e){
    console.warn("translateTerms failed", e);
    return {};
  }
}

/* =========================
 *  Backend API wrappers
 * ========================= */
async function apiReset(root){
  await fetch("expand", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ node: root, mode: "reset" })
  });
}
async function apiRecenter(root){
  await fetch("expand", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ node: root, mode: "recenter" })
  });
}
async function apiExpand(nodeName, mode){
  const res = await fetch("expand", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ node: nodeName, mode })
  });
  return await res.json(); // { parents:[], children:[] }
}

/* =========================
 *  UI: chips & scholar
 * ========================= */
function scholarSearchJA(){
  if (!selected.length) { showToast("検索要素が空です"); return; }
  const q = selected.map(cleanTerm).join(" ");
  logUI("search_ja", {
    selected: [...selected],
    query: q
  });
  const url = "https://scholar.google.com/scholar?hl=ja&q=" + encodeURIComponent(q);
  window.open(url, "_blank", "noopener");
}
window.scholarSearchJA = scholarSearchJA;

async function scholarSearchEN(){
  if (!selected.length) { showToast("検索要素が空です"); return; }
  const tmap = await translateTerms(selected);
  const groups = [];
  for (const src of selected){
    let terms = [];
    const srcClean = cleanTerm(src);
    if (isAsciiLike(srcClean)) {
      terms.push(srcClean);           // 英語っぽければ元語も含める
    }
    const ens = (tmap[src] || []).slice(0,2).map(cleanTerm);
    for (const e of ens){
      if (e && !terms.includes(e)) terms.push(e);
    }
    if (terms.length === 0) terms = [srcClean];     // 非ASCIIなら元語そのまま
    groups.push(terms.length > 1 ? "(" + terms.join(" OR ") + ")" : terms[0]);
  }
  const q = groups.join(" ");
  logUI("search_en", {
    selected: [...selected],
    query: q
  });
  const url = "https://scholar.google.com/scholar?hl=en&q=" + encodeURIComponent(q);
  window.open(url, "_blank", "noopener");
}
window.scholarSearchEN = scholarSearchEN;

/* =========================
 *  SVG / Layout
 * ========================= */
function ensureSvg(){
  svg = d3.select("#canvas");
  if (!svg.node()) return;

  svg.selectAll("*").remove();
  rootG = svg.append("g");
  contentG = rootG.append("g");

  zoom = d3.zoom().scaleExtent([MIN_SCALE, MAX_SCALE]).on("zoom", (ev)=> {
    rootG.attr("transform", ev.transform);
  });
  svg.call(zoom).on("dblclick.zoom", null);
}

function fitToContents(instant=false){
  if (!svg || !contentG) return;
  let bbox; try{ bbox = contentG.node().getBBox(); }catch(_){ return; }
  const pad = 80;
  const vw = svg.node().clientWidth || window.innerWidth;
  const vh = svg.node().clientHeight || window.innerHeight;
  const bw = Math.max(1, bbox.width + pad*2);
  const bh = Math.max(1, bbox.height + pad*2);
  let scale = Math.min(vw/bw, vh/bh);
  scale = Math.max(MIN_SCALE, Math.min(scale, 1.0));
  const cx = bbox.x + bbox.width/2;
  const cy = bbox.y + bbox.height/2;
  const tx = (vw/2) - (cx * scale);
  const ty = (vh/2) - (cy * scale);
  const t = d3.zoomIdentity.translate(tx,ty).scale(scale);
  svg.transition().duration(instant ? 0 : 300).call(zoom.transform, t);
}
window.fitToContents = fitToContents;

function resetZoom(){
  if (!svg) return;
  const t = d3.zoomIdentity;
  svg.transition().duration(200).call(zoom.transform, t);
}
window.resetZoom = resetZoom;

/* =========================
 *  Tooltip
 * ========================= */
function ensureTooltip(){
  if (tooltipEl) return;
  tooltipEl = document.createElement("div");
  Object.assign(tooltipEl.style, {
    position:"fixed", pointerEvents:"auto",
    background:"#fff", border:"1px solid #ddd", borderRadius:"8px",
    padding:"8px 10px", boxShadow:"0 4px 12px rgba(0,0,0,.12)",
    fontSize:"12px", maxWidth:"360px", zIndex:"9999", display:"none"
  });
  document.body.appendChild(tooltipEl);
}

function showTooltip(ev, meta, term){
  if(!tooltipEl) return;
  const full = meta.full || "";
  const desc = meta.desc || "";
  tooltipEl.innerHTML =
    `<div style="font-weight:600;margin-bottom:4px;">${escapeHtml(full)}</div>`+
    `<div style="color:#444;line-height:1.4;">${escapeHtml(desc || "説明なし")}</div>`;

  const m = 14;
  let x = ev.clientX + m, y = ev.clientY + m;
  const vw = window.innerWidth, vh = window.innerHeight;
  const r = tooltipEl.getBoundingClientRect();
  if (x + r.width + 10 > vw) x = ev.clientX - r.width - m;
  if (y + r.height + 10 > vh) y = ev.clientY - r.height - m;
  tooltipEl.style.left = x + "px";
  tooltipEl.style.top  = y + "px";
  tooltipEl.style.display = "block";
}

function hideTooltip(){ if(tooltipEl) tooltipEl.style.display = "none"; }

/* =========================
 *  Render (左右ツリー：左=親 / 右=子)
 * ========================= */
function render(){
  ensureSvg();
  ensureTooltip();
  if (!treeData) return;

  // アクセサ：折りたたみ状態を反映
  const parentAcc = (d)=> d._collapsedParents ? [] : (d.parents || []);
  const childAcc  = (d)=> d._collapsedChildren ? [] : (d.children || []);

  // 親側（左）レイアウト
  const parentRoot = d3.hierarchy(treeData, parentAcc);
  d3.tree().nodeSize([90, 180])(parentRoot);
  parentRoot.descendants().forEach((d)=> { d.y = -Math.abs(d.depth * 250); });

  // 子側（右）レイアウト
  const childRoot  = d3.hierarchy(treeData, childAcc);
  d3.tree().nodeSize([90, 180])(childRoot);
  childRoot.descendants().forEach((d)=> { d.y = Math.abs(d.depth * 250); });

  // マージ（root は重複するので childRoot は先頭を除外）
  let nodes = parentRoot.descendants().concat(childRoot.descendants().slice(1));
  let links = parentRoot.links().concat(childRoot.links());

  // リンク
  contentG.selectAll(".link")
    .data(links, d => d.source.data.name + "→" + d.target.data.name)
    .join("path")
    .attr("class","link")
    .attr("fill","none")
    .attr("stroke","#bbb")
    .attr("stroke-width",2)
    .attr("d", d3.linkHorizontal().x(d=>d.y).y(d=>d.x))
    .attr("stroke-opacity",0.9);

  // ノード（角丸矩形）
  contentG.selectAll(".node")
    .data(nodes, d => d.data.name + "@" + d.depth + (d.y<0?"L":"R"))
    .join(enter=>{
      const g = enter.append("g").attr("class","node")
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .style("cursor","pointer");

      g.each(function(d){
        const gsel = d3.select(this);
        const label = computeShortLabel(d.data.name);

        // 1) テキストを先置きして実測
        const textEl = gsel.append("text")
          .attr("text-anchor","middle")
          .attr("dy", 4)
          .style("font-size", `${LABEL_FONT_SIZE}px`)
          .style("user-select","none")
          .style("pointer-events","none")
          .text(label);

        let tw = label.length * 7; // フォールバック
        try { tw = textEl.node().getComputedTextLength(); } catch(_){}

        const w = Math.max(LABEL_MIN_W, tw + LABEL_PAD_X * 2);
        const h = Math.max(LABEL_MIN_H, LABEL_FONT_SIZE + LABEL_PAD_Y * 2);

        // 2) 角丸矩形（pill）
        const fill = (focusNode && d.data.name === focusNode.name) ? "#ffd54f"
                   : (selected.includes(d.data.name) ? "#a5d6a7" : "#90caf9");

        gsel.insert("rect","text")
          .attr("x", -w / 2)
          .attr("y", -h / 2)
          .attr("width",  w)
          .attr("height", h)
          .attr("rx", LABEL_CORNER_R).attr("ry", LABEL_CORNER_R)
          .attr("fill", fill)
          .attr("stroke", "#333").attr("stroke-width", 1.2)
          .on("click", async (ev)=>{
            // Shift+クリックで中心据え（履歴は維持）
            if (ev.shiftKey){
              recenterOn(d.data.name);
              return;
            }
            // Alt+クリックで折りたたみ（上級者向けショートカットとして残す）
            if (ev.altKey){
              const sideKey = (d.y < 0) ? "_collapsedParents" : "_collapsedChildren";
              d.data[sideKey] = !d.data[sideKey];
              render(); return;
            }
            // 通常クリック：展開 or フォーカス移動
            const dir = (d.y < 0) ? "parent" : "child";
            focusNode = d.data;
            if (expanded.has(d.data.name)) { render(); }
            else { await expandNode(d.data, dir); }
          })
          .on("dblclick", (ev)=>{
            ev.stopPropagation();
            const term = d.data.name;
            if (!selected.includes(term)) {
              addSelectedTerm(term, "node_dblclick");
              render();
            }
          })
          .on("mouseover", async (ev)=>{
            // まずは即座にプレースホルダ表示
            showTooltip(ev, { full: d.data.name, desc:"読み込み中..." }, d.data.name);
            const meta = await describeTerm(d.data.name);
            showTooltip(ev, meta, d.data.name);
          })
          .on("mousemove", (ev)=>{
            const meta = metaCache[canon(d.data.name)] || { full: d.data.name, desc:"" };
            showTooltip(ev, meta, d.data.name);
          })
          .on("mouseout", hideTooltip);

        // 3) 折りたたみインジケータ（矩形サイズに追従 & クリックで開閉）
        const hasKids = (d.y < 0)
          ? (d.data.parents?.length > 0)
          : (d.data.children?.length > 0);
        const collapsed = (d.y < 0)
          ? d.data._collapsedParents
          : d.data._collapsedChildren;

        const indicator = gsel.append("text")
          .attr("text-anchor","middle")
          .attr("dy", (h / 2) + 14)
          .style("font-size", "14px")
          .style("fill", hasKids ? "#333" : "#aaa")
          .style("cursor", hasKids ? "pointer" : "default")
          .text(() => {
            return hasKids ? (collapsed ? "＋" : "−") : "";
          });

        if (hasKids) {
          indicator.on("click", (ev) => {
            ev.stopPropagation();  // 親のクリックイベントに伝播させない
            const sideKey = (d.y < 0) ? "_collapsedParents" : "_collapsedChildren";
            d.data[sideKey] = !d.data[sideKey];
            render();
          });
        }
      });
      return g;
    })
    .attr("transform", d => `translate(${d.y},${d.x})`);

  fitToContents(true);
}

/* =========================
 *  Expand & Recenter
 * ========================= */
async function expandNode(nodeData, mode){
  const out = await apiExpand(nodeData.name, mode);

  let newNames = [];
  if (mode === "parent"){
    const parents = (out.parents || []);
    nodeData.parents = parents.map(n => ({
      name:n, parents:[], children:[], _collapsedParents:false, _collapsedChildren:false
    }));
    newNames = parents;
  } else {
    const children = (out.children || []);
    nodeData.children = children.map(n => ({
      name:n, parents:[], children:[], _collapsedParents:false, _collapsedChildren:false
    }));
    newNames = children;
  }

  expanded.add(nodeData.name);

  // ★ 展開されたノードの説明を先読み
  warmDescribe(newNames);

  render();
}

async function recenterOn(term, opts={push:true}){
  const t = (term || "").trim();
  if (!t) return;
  if (opts.push && treeData?.name) centerStack.push(treeData.name);

  // クライアント側状態だけ初期化（履歴はサーバに残す）
  treeData = { name: t, parents: [], children: [], _collapsedParents: false, _collapsedChildren: false };
  focusNode = treeData;
  expanded.clear();
  render();

  // サーバへ recenter（history は維持される）
  await apiRecenter(t);
  await expandNode(treeData, "parent");
  await expandNode(treeData, "child");
}

/* =========================
 *  Init
 * ========================= */
async function loadKnownMeta(){
  try{
    const res = await fetch("meta");
    KNOWN_META = await res.json();
  }catch(e){
    console.warn("meta load failed", e);
    KNOWN_META = {};
  }
}

async function init(){
  await loadKnownMeta();

  $("#run")?.addEventListener("click", async ()=>{
    const q = ($("#q")?.value || "").trim();
    if (!q) { showToast("キーワードを入力してください"); return; }

    // 完全新規検索：サーバ履歴もリセット
    treeData = { name:q, parents:[], children:[], _collapsedParents:false, _collapsedChildren:false };
    focusNode = treeData;
    expanded.clear();
    ensureSvg();

    await apiReset(q);
    await expandNode(treeData, "parent");
    await expandNode(treeData, "child");
  });

  $("#fit")?.addEventListener("click", ()=> fitToContents(false));
  $("#reset")?.addEventListener("click", ()=>{
    resetZoom();
    showToast("ズームをリセットしました");
  });

  $("#back")?.addEventListener("click", ()=>{
    const prev = centerStack.pop();
    if (!prev) { showToast("戻る履歴がありません"); return; }
    recenterOn(prev, {push:false});
  });

  // 日本語/英語 検索ボタン
  $("#searchJA")?.addEventListener("click", ()=> scholarSearchJA());
  $("#searchEN")?.addEventListener("click", ()=> scholarSearchEN());

  // Enter で実行
  $("#q")?.addEventListener("keydown", (e)=>{
    if (e.key === "Enter") $("#run").click();
  });
  document.addEventListener("keydown", (e)=>{
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      if (e.shiftKey) scholarSearchEN();
      else scholarSearchJA();
    }
  });
}

/* =========================
 *  UI logging -> log_ui
 * ========================= */
async function logUI(eventType, payload) {
  try {
    await fetch("log_ui", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_type: eventType,
        payload: payload || {}
      })
    });
  } catch (e) {
    console.warn("log_ui failed", e);
  }
}

document.addEventListener("DOMContentLoaded", init);

// Entry point: load the graph, lay it out, wire the query panel and node interactions.

import { $, h, fetchJSON, fmtInt, lineRange } from "./util.js";
import { readTheme } from "./colors.js";
import { buildGraph, applyTheme, createView } from "./graph.js";
import { cacheKey, seedPositions, loadCachedPositions, savePositions, runLayout } from "./layout.js";
import { renderResults, renderChunk } from "./render.js";
import { readRequest, postQuery, highlightFromResults, wireContextShortcut } from "./query.js";

const panel = $("#panel");
const status = $("#status");
let lastResults = null; // last query doc, re-rendered when leaving the chunk view
let chunkSeq = 0; // drops stale /api/chunk responses

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function showPanel(nodes) {
  panel.replaceChildren(...nodes.filter(Boolean));
  panel.scrollTop = 0;
}

function renderMeta(meta) {
  const edgeSource = meta.edge_source === "subchunk"
    ? `subchunk (${meta.subchunk_size_tokens} tok)`
    : "chunk (whole-chunk cosine)";
  const items = [
    ["corpus", meta.root],
    ["chunks", fmtInt(meta.nodes)],
    ["edges", fmtInt(meta.edges)],
    ["communities", fmtInt(meta.communities)],
    ["edge source", edgeSource],
    ["embedding", meta.embedding_model],
  ];
  $("#meta").replaceChildren(...items.map(([k, v]) =>
    h("div", {}, h("dt", { text: k }), h("dd", { text: String(v) }))));
}

async function main() {
  const overlay = $("#overlay");
  let payload;
  try {
    payload = await fetchJSON("/api/graph");
  } catch (e) {
    overlay.textContent = `Could not load the graph: ${e.message}`;
    overlay.classList.add("error");
    return;
  }
  renderMeta(payload.meta);
  document.title = `ragx · ${payload.meta.root}`;

  const { graph, palette } = buildGraph(payload);
  let theme = readTheme();
  applyTheme(graph, palette, theme);
  const key = cacheKey(payload.meta);
  const cached = loadCachedPositions(graph, key);
  if (!cached) seedPositions(graph);
  overlay.hidden = true;
  if (graph.order === 0) {
    overlay.hidden = false;
    overlay.textContent = "The index has no chunks.";
  }

  const view = createView($("#graph"), graph, theme);
  window.ragxView = view; // handle for debugging from the console

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    theme = readTheme();
    applyTheme(graph, palette, theme);
    view.setTheme(theme);
  });

  // --- layout -----------------------------------------------------------------
  const layoutBtn = $("#layout-btn");
  const layoutStatus = $("#layout-status");
  let layout = null;
  const startLayout = () => {
    layoutBtn.textContent = "Stop layout";
    layout = runLayout(graph, {
      budgetMs: Math.min(10000, 3000 + graph.order * 0.35),
      onTick: ({ elapsed, done, reason }) => {
        const secs = (elapsed / 1000).toFixed(1);
        if (!done) {
          layoutStatus.textContent = `layout running ${secs}s`;
          return;
        }
        layout = null;
        layoutStatus.textContent = `layout ${reason} · ${secs}s`;
        layoutBtn.textContent = "Re-run layout";
        savePositions(graph, key);
      },
    });
  };
  layoutBtn.disabled = graph.order === 0;
  layoutBtn.addEventListener("click", () => (layout ? layout.stop() : startLayout()));
  $("#fit-btn").addEventListener("click", () => view.fit());
  if (cached) layoutStatus.textContent = "layout from cache";
  else if (graph.order) startLayout();

  // --- hover tooltip ------------------------------------------------------------
  const tooltip = $("#tooltip");
  const stage = $("#graph");
  let mouse = { x: 0, y: 0 };
  const placeTooltip = () => {
    const r = stage.getBoundingClientRect();
    const x = mouse.x - r.left + 14;
    const y = mouse.y - r.top + 14;
    tooltip.style.left = `${Math.max(0, Math.min(x, r.width - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(0, Math.min(y, r.height - tooltip.offsetHeight - 8))}px`;
  };
  stage.addEventListener("pointermove", (e) => {
    mouse = { x: e.clientX, y: e.clientY };
    if (!tooltip.hidden) placeTooltip();
  });
  view.sigma.on("enterNode", ({ node }) => {
    const a = graph.getNodeAttributes(node);
    tooltip.replaceChildren(
      h("div", { class: "loc", text: `${a.file}:${lineRange(a.lineStart, a.lineEnd)}` }),
      h("div", { text: a.title || "(empty)" }));
    tooltip.hidden = false;
    placeTooltip();
  });
  view.sigma.on("leaveNode", () => { tooltip.hidden = true; });

  // --- node selection -------------------------------------------------------------
  const backToResults = () => {
    view.deselect();
    showPanel(lastResults
      ? renderResults(lastResults, { onFocus: (k) => view.focus(k) })
      : [h("p", { class: "hint", text: "Run a query or click a node." })]);
  };
  async function selectNode(nodeKey, focus) {
    view.select(nodeKey);
    if (focus) view.focus(nodeKey, false);
    const seq = ++chunkSeq;
    showPanel([h("p", { class: "hint", text: "Loading chunk…" })]);
    try {
      const chunk = await fetchJSON(`/api/chunk/${encodeURIComponent(nodeKey)}`);
      if (seq !== chunkSeq) return;
      const a = graph.getNodeAttributes(nodeKey);
      showPanel(renderChunk(chunk, {
        color: a.color,
        nodeInfo: (k) => (graph.hasNode(k) ? graph.getNodeAttributes(k) : null),
        onNeighbor: (k) => graph.hasNode(k) && selectNode(k, true),
        onBack: lastResults ? backToResults : null,
      }));
    } catch (e) {
      if (seq === chunkSeq) showPanel([h("p", { class: "status error", text: e.message })]);
    }
  }
  view.sigma.on("clickNode", ({ node }) => selectNode(node, false));
  view.sigma.on("clickStage", () => {
    if (view.state.selected) backToResults();
  });

  // --- query ----------------------------------------------------------------------------
  wireContextShortcut();
  const submit = $("#q-submit");
  $("#query-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    let req;
    try {
      req = readRequest();
    } catch (err) {
      setStatus(err.message, true);
      return;
    }
    submit.disabled = true;
    setStatus("Searching…");
    const t0 = performance.now();
    try {
      const out = await postQuery(req);
      lastResults = out;
      const { rank, pathNodes, pathEdges } = highlightFromResults(graph, out);
      view.setResults(rank, pathNodes, pathEdges);
      const chunks = out.results.reduce((n, f) => n + f.chunks.length, 0);
      const ms = out.elapsed_ms ?? Math.round(performance.now() - t0);
      const paths = req.explain ? ` · ${pathEdges.size} path edges` : "";
      setStatus(`${out.results.length} files · ${chunks} chunks · ${fmtInt(ms)} ms${paths}`);
      showPanel(renderResults(out, { onFocus: (k) => view.focus(k) }));
    } catch (err) {
      setStatus(err.message, true);
    } finally {
      submit.disabled = false;
    }
  });
  $("#q-clear").addEventListener("click", () => {
    lastResults = null;
    view.clear();
    setStatus("");
    showPanel([h("p", { class: "hint", text: "Cleared. Run a query or click a node." })]);
  });
}

main();

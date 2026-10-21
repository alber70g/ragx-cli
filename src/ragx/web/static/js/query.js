// Query form -> POST /api/query request, and result -> graph highlight mapping.

import { $, fetchJSON } from "./util.js";
import { chunkLabel } from "./render.js";

function intOrNull(el) {
  const v = el.value.trim();
  if (v === "") return null;
  const n = Number(v);
  if (!Number.isInteger(n) || n < 0) throw new Error(`${el.id.replace("q-", "")} must be a whole number ≥ 0`);
  return n;
}

export function wireContextShortcut() {
  $("#q-context").addEventListener("input", (e) => {
    $("#q-before").value = e.target.value;
    $("#q-after").value = e.target.value;
  });
}

export function readRequest() {
  const req = {
    text: $("#q-text").value.trim(),
    graph: $("#q-graph").checked,
    expand: $("#q-expand").checked,
    rerank: $("#q-rerank").checked,
    explain: $("#q-explain").checked,
  };
  for (const [key, id] of [["top", "#q-top"], ["hops", "#q-hops"], ["before", "#q-before"], ["after", "#q-after"]]) {
    const v = intOrNull($(id));
    if (v !== null) req[key] = v; // omitted -> server uses the config default
  }
  if (!req.text) throw new Error("Enter a query.");
  if (req.top === 0) throw new Error("top must be at least 1");
  return req;
}

export function postQuery(req) {
  return fetchJSON("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

// Maps a ragx.query.v3 doc onto the graph: rank labels per result chunk, plus (with
// explain) the seed -> ... -> chunk traversal path, highlighted as nodes/edges.
// `explain.path` is the authoritative seed-first chunk-id chain from the traversal's
// own parent links; when it's missing, fall back to just the [parent, chunk] edge.
export function highlightFromResults(graph, out) {
  const rank = new Map();
  const pathNodes = new Set();
  const pathEdges = new Set();
  const addPath = (nodes) => {
    for (let i = 0; i < nodes.length; i++) {
      pathNodes.add(nodes[i]);
      if (i && graph.hasEdge(nodes[i - 1], nodes[i])) pathEdges.add(graph.edge(nodes[i - 1], nodes[i]));
    }
  };
  out.results.forEach((f, fi) => f.chunks.forEach((c, ci) => {
    const key = String(c.chunk_id);
    if (!graph.hasNode(key)) return;
    rank.set(key, chunkLabel(fi + 1, ci));
    const ex = c.explain;
    if (!ex) return;
    const chain = ex.path ?? (ex.parent != null ? [ex.parent, c.chunk_id] : []);
    addPath(chain.map(String).filter((n) => graph.hasNode(n)));
  }));
  return { rank, pathNodes, pathEdges };
}

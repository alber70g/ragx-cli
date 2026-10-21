// Side-panel renderers: query results (grep-style context) and the chunk detail view.
// Pure DOM builders; callbacks are passed in, nothing here touches the graph.

import { h, fmtScore, lineRange } from "./util.js";

function row(cls, no, sep, text) {
  return h("div", { class: `row ${cls}` },
    h("span", { class: "no", text: String(no) }),
    h("span", { class: "sep", text: sep }),
    h("span", { class: "tx", text: text === "" ? " " : text }));
}

function spanRows(span) {
  if (!span) return [];
  return span.text.split("\n").map((t, i) => row("ctx", span.line_start + i, "-", t));
}

// Chunk text as numbered lines. A chunk slice that ends in "\n" yields one extra empty
// element; drop it when it falls past line_end.
function chunkRows(text, lineStart, lineEnd, cls = "hit") {
  const lines = text.split("\n");
  if (lines.length > lineEnd - lineStart + 1 && lines[lines.length - 1] === "") lines.pop();
  return lines.map((t, i) => row(cls, lineStart + i, ":", t));
}

// grep -B/-A layout: muted context before, emphasized chunk lines, muted context after.
export function grepBlock(c) {
  const ctx = c.context || {};
  const rows = [...spanRows(ctx.before), ...chunkRows(c.text, c.line_start, c.line_end)];
  if (c.truncated) {
    rows.push(h("div", { class: "gap", text: `… chunk text truncated; chunk ends at line ${c.line_end}` }));
  }
  rows.push(...spanRows(ctx.after));
  return h("div", { class: "grep" }, rows);
}

function explainText(e) {
  if (!e) return null;
  if (e.parent == null) return "seed (vector hit)";
  const w = typeof e.edge_weight === "number" ? ` w ${e.edge_weight.toFixed(3)}` : "";
  const via = e.parent === e.seed ? "" : ` via #${e.parent}`;
  const chain = Array.isArray(e.path) && e.path.length > 2 ? ` (${e.path.length - 1} hops)` : "";
  return `hop ${e.hop} from seed #${e.seed}${via}${chain}${w}`;
}

export function chunkLabel(fileRank, idx) {
  return idx === 0 ? String(fileRank) : `${fileRank}.${idx + 1}`;
}

export function renderResults(out, { onFocus }) {
  const frag = [];
  if (out.variants && out.variants.length > 1) {
    frag.push(h("div", { class: "variants" }, `variants: ${out.variants.slice(1).join(" · ")}`));
  }
  if (!out.results.length) frag.push(h("p", { class: "hint", text: "No results." }));
  out.results.forEach((f, fi) => {
    const chunks = f.chunks.map((c, ci) => {
      const b = c.breakdown || {};
      const head = h("div", {
        class: "chunk-head", role: "button", tabindex: "0",
        title: "Show this chunk in the graph",
        onclick: () => onFocus(String(c.chunk_id)),
        onkeydown: (e) => { if (e.key === "Enter") onFocus(String(c.chunk_id)); },
      },
        h("span", { class: "rank", text: chunkLabel(fi + 1, ci) }),
        h("span", { class: "loc", text: `lines ${lineRange(c.line_start, c.line_end)}` }),
        h("span", { text: `score ${fmtScore(c.score)}` }),
        h("span", { text: `vec ${fmtScore(b.vector)} · heat ${fmtScore(b.heat)} · rerank ${fmtScore(b.rerank)}` }),
        c.explain ? h("span", { class: "explain", text: explainText(c.explain) }) : null);
      return h("div", { class: "chunk" }, head, grepBlock(c));
    });
    frag.push(h("section", { class: "file" },
      h("div", { class: "file-head" },
        h("span", { class: "rank", text: String(fi + 1) }),
        h("span", { class: "path", text: f.file }),
        h("span", { class: "score", text: fmtScore(f.score) })),
      chunks));
  });
  return frag;
}

// chunk: GET /api/chunk payload. nodeInfo(key) -> graph attributes for neighbor labels.
export function renderChunk(chunk, { color, nodeInfo, onNeighbor, onBack }) {
  const comm = chunk.community == null ? "none (isolate)" : String(chunk.community);
  const neighbors = chunk.neighbors.map((n) => {
    const a = nodeInfo(String(n.chunk_id));
    const loc = a ? `${a.file}:${lineRange(a.lineStart, a.lineEnd)}` : `chunk #${n.chunk_id}`;
    return h("li", {},
      h("button", { type: "button", class: "link", onclick: () => onNeighbor(String(n.chunk_id)) },
        h("span", { class: "loc", text: `${n.weight.toFixed(3)}  ${loc}` }),
        a && a.title ? h("span", { class: "label", text: a.title }) : null));
  });
  return [
    onBack ? h("button", { type: "button", class: "link", onclick: onBack, text: "← Back to results" }) : null,
    h("div", { class: "detail-head" },
      h("span", { class: "path", text: chunk.file }),
      h("div", { class: "detail-meta" },
        h("span", { text: `lines ${lineRange(chunk.line_start, chunk.line_end)}` }),
        h("span", {}, h("span", { class: "swatch", style: `background:${color}` }), `community ${comm}`),
        h("span", { text: `chunk #${chunk.chunk_id}` }))),
    h("div", { class: "grep" }, chunkRows(chunk.text, chunk.line_start, chunk.line_end, "plain")),
    h("h2", { text: `Neighbors (${chunk.neighbors.length})` }),
    neighbors.length ? h("ul", { class: "neighbors" }, neighbors) : h("p", { class: "hint", text: "No edges." }),
  ];
}

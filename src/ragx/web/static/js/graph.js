// Graph model + sigma renderer. Nodes = chunks (colored by community, sized by degree),
// edges = sub-chunk similarity links. Dimming/highlighting is done in sigma reducers driven
// by `view.state`, so a query or selection never mutates the base node/edge attributes.

import { communityColors, mix, rgba } from "./colors.js";

const { Graph } = window.graphology;
const Sigma = window.Sigma;

export function buildGraph(payload) {
  const graph = new Graph({ type: "undirected", multi: false, allowSelfLoops: false });
  const sizes = new Map();
  let maxDeg = 1;
  for (const n of payload.nodes) {
    if (n.community != null) sizes.set(n.community, (sizes.get(n.community) || 0) + 1);
    maxDeg = Math.max(maxDeg, n.degree);
  }
  const big = payload.nodes.length > 5000;
  const [minSize, span] = big ? [1.5, 3.5] : [2.5, 5];
  for (const n of payload.nodes) {
    graph.addNode(String(n.id), {
      x: 0, y: 0,
      size: minSize + span * Math.sqrt(n.degree / maxDeg),
      community: n.community, file: n.file, lineStart: n.line_start, lineEnd: n.line_end,
      title: n.label, degree: n.degree,
    });
  }
  let minW = Infinity;
  let maxW = -Infinity;
  for (const [, , w] of payload.edges) {
    minW = Math.min(minW, w);
    maxW = Math.max(maxW, w);
  }
  const wSpan = maxW > minW ? maxW - minW : 1;
  for (const [s, t, w] of payload.edges) {
    const a = String(s);
    const b = String(t);
    if (a === b || !graph.hasNode(a) || !graph.hasNode(b) || graph.hasEdge(a, b)) continue;
    graph.addEdgeWithKey(`${a}-${b}`, a, b, { weight: w, t: (w - minW) / wSpan });
  }
  return { graph, palette: communityColors(sizes) };
}

// Base colors live on the attributes so the reducers stay cheap; recomputed on theme change.
export function applyTheme(graph, palette, theme) {
  // Many overlapping edges add up to a grey wash: lower per-edge alpha as the edge count grows.
  const dense = Math.min(1, Math.max(0.25, Math.sqrt(5000 / Math.max(graph.size, 1))));
  graph.updateEachNodeAttributes((_, a) => {
    const c = a.community == null ? theme.neutral : palette.get(a.community);
    a.color = c;
    a.dimColor = mix(c, theme.bg, theme.dark ? 0.74 : 0.78);
    return a;
  });
  graph.updateEachEdgeAttributes((_, a) => {
    a.color = rgba(theme.edge, (0.05 + 0.3 * a.t) * dense);
    a.size = 0.4 + 0.9 * a.t;
    return a;
  });
}

export function createView(container, graph, theme) {
  const state = {
    mode: null, // null | "results" | "select"
    rank: new Map(), // node -> result rank (1-based)
    pathNodes: new Set(),
    pathEdges: new Set(),
    selected: null,
    focusSet: new Set(), // selected node + its neighbors
    selectEdges: new Set(),
    focused: null, // node flagged by clicking a result row
    theme,
    dimEdge: rgba(theme.edge, 0.05),
  };

  const nodeReducer = (key, data, attrs) => {
    const out = { labelColor: state.theme.label, backdropColor: state.theme.bg };
    const r = state.rank.get(key);
    if (state.mode) {
      const lit = state.mode === "select"
        ? state.focusSet.has(key)
        : r !== undefined || state.pathNodes.has(key);
      if (!lit) return { color: attrs.dimColor, size: data.size * 0.75, zIndex: 0, label: null };
      out.zIndex = 1;
      out.depth = "topNodes";
    }
    if (r !== undefined) {
      out.size = data.size * 1.5 + 4;
      out.label = String(r);
      out.labelVisibility = "visible";
      out.zIndex = 2;
    }
    if (key === state.selected || key === state.focused) {
      out.size = (out.size ?? data.size) + 5;
      out.highlighted = true;
      out.zIndex = 3;
    }
    return out;
  };

  const edgeReducer = (key) => {
    if (!state.mode) return {};
    if (state.mode === "select") {
      if (state.selectEdges.has(key)) {
        return { color: rgba(state.theme.accent, 0.75), size: 1.3, zIndex: 1, depth: "topEdges" };
      }
    } else if (state.pathEdges.has(key)) {
      return { color: state.theme.path, size: 2.6, zIndex: 2, depth: "topEdges" };
    }
    return { color: state.dimEdge, size: 0.35, zIndex: 0 };
  };

  const sigma = new Sigma(graph, container, {
    // Highlighted edges ("topEdges") draw above dimmed nodes but below lit ones; with the
    // default order (all edges under all nodes) dense dimmed nodes chop paths into dashes.
    primitives: { depthLayers: ["edges", "nodes", "topEdges", "topNodes"] },
    nodeReducer,
    edgeReducer,
    settings: {
      itemSizesReference: "screen", // node/edge sizes in pixels, not graph units
      hideEdgesOnMove: graph.size > 20000,
      renderEdgeLabels: false,
      enableEdgeEvents: false,
      labelDensity: 1,
      labelRenderedSizeThreshold: 0,
      minCameraRatio: 0.01,
      maxCameraRatio: 4,
    },
  });

  const refresh = () => sigma.refresh({ skipIndexation: true });

  return {
    sigma,
    state,
    setTheme(t) {
      state.theme = t;
      state.dimEdge = rgba(t.edge, 0.05);
      sigma.refresh();
    },
    setResults(rank, pathNodes, pathEdges) {
      Object.assign(state, { mode: "results", rank, pathNodes, pathEdges, focused: null });
      state.selected = null;
      refresh();
    },
    select(key) {
      state.selected = key;
      state.focusSet = new Set([key, ...graph.neighbors(key)]);
      state.selectEdges = new Set(graph.edges(key));
      state.mode = "select";
      refresh();
    },
    deselect() {
      state.selected = null;
      state.mode = state.rank.size ? "results" : null;
      refresh();
    },
    clear() {
      Object.assign(state, {
        mode: null, rank: new Map(), pathNodes: new Set(), pathEdges: new Set(),
        selected: null, focused: null,
      });
      refresh();
    },
    focus(key, flag = true) {
      if (flag) {
        state.focused = key;
        refresh();
      }
      const d = sigma.getNodeDisplayData(key);
      if (!d) return;
      const cam = sigma.getCamera();
      cam.animate({ x: d.x, y: d.y, ratio: Math.min(cam.ratio, 0.25) }, { duration: 450 });
    },
    fit() {
      sigma.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1, angle: 0 }, { duration: 400 });
    },
  };
}

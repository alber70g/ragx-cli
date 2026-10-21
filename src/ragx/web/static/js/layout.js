// Initial placement, incremental ForceAtlas2 and the localStorage position cache.

import { storageGet, storageSet } from "./util.js";

const FA2 = window.graphologyLibrary.layoutForceAtlas2;
const CACHE_PREFIX = "ragx.layout.v1:";

export function cacheKey(meta) {
  return `${CACHE_PREFIX}${meta.root}:${meta.nodes}:${meta.edges}:${meta.embedding_model}`;
}

// Seed: community centers on a circle, members scattered around their center. FA2 then
// starts close to the clustered end state instead of untangling a random cloud.
export function seedPositions(graph) {
  const members = new Map();
  graph.forEachNode((key, a) => {
    const c = a.community ?? "none";
    if (!members.has(c)) members.set(c, []);
    members.get(c).push(key);
  });
  const groups = [...members.entries()].sort((a, b) => b[1].length - a[1].length);
  const R = 12 * Math.sqrt(graph.order);
  groups.forEach(([, keys], i) => {
    const angle = (2 * Math.PI * i) / groups.length;
    const cx = R * Math.cos(angle);
    const cy = R * Math.sin(angle);
    const r = 4 * Math.sqrt(keys.length);
    for (const key of keys) {
      const t = Math.random() * 2 * Math.PI;
      const d = r * Math.sqrt(Math.random());
      graph.mergeNodeAttributes(key, { x: cx + d * Math.cos(t), y: cy + d * Math.sin(t) });
    }
  });
}

// Returns true when every node got a cached position.
export function loadCachedPositions(graph, key) {
  const raw = storageGet(key);
  if (!raw) return false;
  let pos;
  try {
    pos = JSON.parse(raw).pos;
  } catch {
    return false;
  }
  if (!pos || graph.someNode((k) => !Array.isArray(pos[k]))) return false;
  graph.updateEachNodeAttributes((k, a) => {
    a.x = pos[k][0];
    a.y = pos[k][1];
    return a;
  });
  return true;
}

export function savePositions(graph, key) {
  const pos = {};
  graph.forEachNode((k, a) => {
    pos[k] = [Math.round(a.x * 10) / 10, Math.round(a.y * 10) / 10];
  });
  return storageSet(key, JSON.stringify({ pos }), CACHE_PREFIX);
}

function settingsFor(graph) {
  const settings = FA2.inferSettings(graph);
  settings.barnesHutOptimize = graph.order > 2000;
  settings.edgeWeightInfluence = 1; // edge `weight` (similarity) pulls stronger links closer
  return settings;
}

// Mean node displacement since `xs/ys` relative to the layout extent (0 = no movement).
function makeMovementProbe(graph) {
  const xs = new Float64Array(graph.order);
  const ys = new Float64Array(graph.order);
  return {
    snapshot() {
      let i = 0;
      graph.forEachNode((_, a) => {
        xs[i] = a.x;
        ys[i++] = a.y;
      });
    },
    movement() {
      let i = 0;
      let moved = 0;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      graph.forEachNode((_, a) => {
        moved += Math.hypot(a.x - xs[i], a.y - ys[i]);
        i++;
        if (a.x < minX) minX = a.x;
        if (a.x > maxX) maxX = a.x;
        if (a.y < minY) minY = a.y;
        if (a.y > maxY) maxY = a.y;
      });
      return moved / graph.order / Math.max(maxX - minX, maxY - minY, 1);
    },
  };
}

// Runs ForceAtlas2 until node movement settles or `budgetMs` runs out, reporting through
// onTick({elapsed, iterations, done, reason}). Prefers the library's web-worker supervisor
// (FA2Layout) so the page stays interactive; falls back to slices of iterations per
// animation frame when a worker can't be created (e.g. a CSP that blocks blob: workers).
export function runLayout(graph, opts) {
  try {
    return runInWorker(graph, opts);
  } catch {
    return runInFrames(graph, opts);
  }
}

function runInWorker(graph, { budgetMs, onTick }) {
  const sup = new window.graphologyLibrary.FA2Layout(graph, { settings: settingsFor(graph) });
  const probe = makeMovementProbe(graph);
  const t0 = performance.now();
  let calm = 0;
  let timer = null;
  const finish = (reason) => {
    clearInterval(timer);
    sup.kill();
    onTick({ elapsed: performance.now() - t0, done: true, reason });
  };
  probe.snapshot();
  sup.start();
  timer = setInterval(() => {
    const elapsed = performance.now() - t0;
    // Probed every 0.5 s: mean movement under 0.2 % of the extent on 3 probes in a row = settled.
    calm = probe.movement() < 0.002 ? calm + 1 : 0;
    probe.snapshot();
    if (calm >= 3 && elapsed > 1500) return finish("converged");
    if (elapsed > budgetMs) return finish("time budget");
    onTick({ elapsed, done: false });
  }, 500);
  return { stop: () => sup.isRunning() && finish("stopped") };
}

function runInFrames(graph, { budgetMs, onTick }) {
  const settings = settingsFor(graph);
  const probe = makeMovementProbe(graph);
  let stopped = false;
  let perFrame = 1;
  let calm = 0;
  const t0 = performance.now();
  const finish = (reason) => {
    stopped = true;
    onTick({ elapsed: performance.now() - t0, done: true, reason });
  };
  const frame = () => {
    if (stopped) return;
    probe.snapshot();
    const s = performance.now();
    FA2.assign(graph, { iterations: perFrame, settings });
    const took = performance.now() - s;
    const m = probe.movement() / perFrame;
    // Adapt the slice so one frame costs ~16 ms of layout work.
    perFrame = Math.max(1, Math.min(50, Math.round(16 / Math.max(took / perFrame, 0.05))));
    calm = m < 0.0004 ? calm + 1 : 0;
    const elapsed = performance.now() - t0;
    if (calm >= 5) return finish("converged");
    if (elapsed > budgetMs) return finish("time budget");
    onTick({ elapsed, done: false });
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return { stop: () => !stopped && finish("stopped") };
}

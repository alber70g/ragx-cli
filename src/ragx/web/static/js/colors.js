// Community palette and theme-dependent canvas colors.

// Hand-picked categorical colors for the largest communities; mid lightness so they read
// on both the light and the dark background. Smaller communities get golden-angle hues.
const BASE = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
  "#9c755f", "#3c8dbc", "#d4a017", "#8cd17d", "#d37295", "#499894", "#e6739f", "#86bcb6",
  "#c4651e", "#7a6fca", "#5fa04e", "#c75d8c",
];

function hslToHex(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h / 30) % 12;
    const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(c * 255).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

// communities: Map<communityId, size>. Largest community gets the first palette color.
export function communityColors(communities) {
  const ordered = [...communities.entries()].sort((a, b) => b[1] - a[1]).map(([c]) => c);
  const out = new Map();
  ordered.forEach((c, i) => {
    if (i < BASE.length) out.set(c, BASE[i]);
    else out.set(c, hslToHex((i * 137.508) % 360, 0.55, i % 2 ? 0.5 : 0.62));
  });
  return out;
}

function parseHex(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Blend `hex` toward `bg` by t (0 = hex, 1 = bg). Used for dimmed nodes: an opaque blend
// keeps dimmed nodes from stacking into bright blobs where many overlap.
export function mix(hex, bg, t) {
  const a = parseHex(hex);
  const b = parseHex(bg);
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

export function rgba(hex, alpha) {
  const [r, g, b] = parseHex(hex);
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

// Canvas colors come from the CSS custom properties, so styles.css is the single source.
export function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    dark: window.matchMedia("(prefers-color-scheme: dark)").matches,
    bg: v("--graph-bg"),
    edge: v("--graph-edge"),
    accent: v("--accent"),
    path: v("--path"),
    neutral: v("--neutral-node"),
    label: v("--fg"),
  };
}

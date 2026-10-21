// DOM + fetch helpers. All text goes in via textContent, never innerHTML: chunk text is
// user content, so nothing here parses HTML.

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export async function fetchJSON(url, init) {
  let res;
  try {
    res = await fetch(url, init);
  } catch (e) {
    throw new Error(`Network error: ${e.message}`);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON body: fall through to status-based message
  }
  if (!res.ok) {
    const msg = body && typeof body.error === "string" ? body.error : res.statusText;
    throw new Error(`${res.status}: ${msg}`);
  }
  return body;
}

export const fmtInt = (n) => Number(n).toLocaleString("en-US");
export const fmtScore = (x) => (typeof x === "number" ? x.toFixed(3) : "–");

export function lineRange(a, b) {
  return a === b ? `${a}` : `${a}–${b}`;
}

// localStorage can throw (private mode, blocked storage, quota): every access is guarded.
export function storageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function storageSet(key, value, dropPrefix) {
  try {
    const ls = window.localStorage;
    if (dropPrefix) {
      for (let i = ls.length - 1; i >= 0; i--) {
        const k = ls.key(i);
        if (k && k.startsWith(dropPrefix) && k !== key) ls.removeItem(k);
      }
    }
    ls.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

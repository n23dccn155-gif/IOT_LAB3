// Run: node LAB3/frontend/tests/overlay.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const window = { devicePixelRatio: 2 };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../ui.js'), 'utf8'), { window });

function fixture(width, height) {
  const calls = [];
  const ctx = new Proxy({ measureText: () => ({ width: 90 }) }, {
    get(target, key) {
      return key in target ? target[key] : (...args) => calls.push([key, ...args]);
    }
  });
  const canvas = { clientWidth: width, clientHeight: height, getContext: () => ctx,
    getBoundingClientRect: () => ({ left: 10, top: 20, width, height }) };
  return { overlay: new window.LAB3Ui.Overlay(canvas), calls, canvas };
}

const face = { x: 100, y: 80, width: 200, height: 240,
  label: 'without_mask', confidence: 0.847 };
const f = fixture(640, 360);
f.overlay.syncSize(1280, 720);
assert.equal(f.canvas.width, 1280); // DPR affects resolution, not CSS coordinates.
f.overlay.draw([face], { mirrored: false });
assert.deepEqual(f.calls.find(c => c[0] === 'strokeRect'), ['strokeRect', 50, 40, 100, 120]);
f.calls.length = 0;
f.overlay.draw([face], { mirrored: true });
assert.deepEqual(f.calls.find(c => c[0] === 'strokeRect'), ['strokeRect', 490, 40, 100, 120]);
assert.deepEqual(f.calls.find(c => c[0] === 'setTransform'), ['setTransform', 2, 0, 0, 2, 0, 0]);
assert.ok(!f.calls.some(c => c[0] === 'scale'), 'Text must not be reflected');
assert.ok(f.calls.some(c => c[0] === 'fillText' && c[1].includes('84.7%')));
let point = f.overlay.pointToImage(550, 120, 1280, 720, true);
assert.equal(point.x, 200);
assert.equal(point.y, 200);

const letterbox = fixture(640, 480);
letterbox.overlay.syncSize(1280, 720);
letterbox.overlay.draw([face], {});
assert.deepEqual(letterbox.calls.find(c => c[0] === 'strokeRect'), ['strokeRect', 50, 100, 100, 120]);
assert.equal(letterbox.overlay.pointToImage(110, 40, 1280, 720, false), null);
point = letterbox.overlay.pointToImage(110, 180, 1280, 720, false);
assert.equal(point.x, 200);
assert.equal(point.y, 200);

const portrait = fixture(360, 640);
portrait.overlay.syncSize(1280, 720, 1);
portrait.overlay.draw([face], { mirrored: true });
assert.deepEqual(portrait.calls.find(c => c[0] === 'strokeRect'),
  ['strokeRect', 275.625, 241.25, 56.25, 67.5]);
console.log('PASS: face boxes, readable mirrored labels, DPR, letterboxing, manual coordinates, portrait resize');

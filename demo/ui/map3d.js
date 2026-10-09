// 3-D knowledge map: the questions as a Three.js point cloud, part labels as DOM tags above each island.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Restrained categorical palette: 16 OKLCH hues, alternating two lightness levels (0.55 / 0.69) so neighbours separate.
const PALETTE = ['#B9454C', '#E6784A', '#A0600B', '#C09506', '#767700', '#76AE44', '#04884D', '#12B49B',
  '#118186', '#14ACD3', '#0B77BA', '#7296F8', '#725EC2', '#BB7DDD', '#A44A94', '#E3709C'];
const GREY = [0.765, 0.788, 0.824]; // dimmed points
const BG = [1, 1, 1];
const MAX_PARTS = 32;
const SPAN = 10;      // world size of the [0,1] cube
const FOV = 30;
const POINT_PX = 9;   // point diameter (stage px) at the orbit target's depth
const LABEL_TOP = 66; // keep labels clear of the map header
const MODES = {
  idle: { a: 0.92, s: 1, c: 1, g: 0 },
  dim: { a: 0.5, s: 0.7, c: 0, g: 0 },
  lit: { a: 1, s: 1.3, c: 1, g: 1 },
};
const ORDER = { dim: 0, idle: 1, lit: 2 };
// Candidate label positions around the island-top anchor, in (half label width, label height) steps.
const SLOTS = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [0, -2], [0, 2], [-1, -1], [1, -1], [-1, 1], [1, 1],
  [-2, 0], [2, 0], [0, -3], [0, 3], [-2, -1], [2, -1], [-2, 1], [2, 1], [-1, -2], [1, -2], [-1, 2], [1, 2]];
const SVG_NS = 'http://www.w3.org/2000/svg';

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const rgb01 = hex => { const n = parseInt(hex.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; };
const luminance = rgb => {
  const [r, g, b] = rgb.map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const INK = [17 / 255, 20 / 255, 24 / 255];
const oklab = hex => {
  const [r, g, b] = rgb01(hex).map(c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
};
const LAB = PALETTE.map(oklab);
const colorGap = (a, b) => Math.hypot(LAB[a][0] - LAB[b][0], LAB[a][1] - LAB[b][1], LAB[a][2] - LAB[b][2]);
const textOn = rgb => { const L = luminance(rgb); return 1.05 / (L + 0.05) >= (L + 0.05) / (luminance(INK) + 0.05) ? '#FFFFFF' : '#111418'; };

const VERT = /* glsl */ `
  attribute float aPart;
  attribute vec3 aColor;
  uniform vec4 uParts[${MAX_PARTS}]; // per part: alpha, size, colour amount, glow
  uniform float uSize, uScale, uRef, uFogNear, uFogFar;
  uniform vec3 uGrey, uBg;
  varying vec3 vColor;
  varying float vAlpha, vGlow, vPx;
  void main() {
    vec4 st = uParts[int(aPart + 0.5)];
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = -mv.z;
    vPx = uSize * st.y * uScale * (uRef / depth) * (1.0 + st.w * 1.6);
    gl_PointSize = vPx;
    vGlow = st.w;
    float fog = smoothstep(uFogNear, uFogFar, depth) * 0.6;
    vColor = mix(mix(uGrey, aColor, st.z), uBg, fog);
    vAlpha = st.x;
  }`;
const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha, vGlow, vPx;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float rc = 1.0 / (1.0 + vGlow * 1.6);
    float aa = 2.0 / max(vPx, 1.0);
    float core = 1.0 - smoothstep(rc - aa, rc + aa, d);
    float halo = vGlow * 0.2 * (1.0 - smoothstep(rc, 1.0, d));
    float a = max(core * vAlpha, halo);
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor, a);
  }`;

/** Gives spatially close parts clearly different colours (greedy max-min of OKLab distance to the nearest islands). */
function assignColors(parts) {
  const n = parts.length, P = PALETTE.length;
  const dist = (i, j) => parts[i].center.distanceTo(parts[j].center);
  const crowd = i => parts.map((_, j) => (j === i ? Infinity : dist(i, j))).sort((a, b) => a - b).slice(0, 3).reduce((a, b) => a + b, 0);
  const order = [...parts.keys()].sort((a, b) => crowd(a) - crowd(b));
  const pick = new Array(n).fill(-1);
  const used = new Set();
  for (const i of order) {
    const near = order.filter(j => pick[j] >= 0).sort((a, b) => dist(i, a) - dist(i, b)).slice(0, 5);
    let best = 0, bestScore = -Infinity;
    for (let c = 0; c < P; c++) {
      if (used.has(c) && used.size < P) continue;
      const gaps = near.map(j => colorGap(c, pick[j]));
      const score = (gaps.length ? Math.min(...gaps.slice(0, 3)) * 100 : 0) + gaps.reduce((a, b, r) => a + b / (r + 1), 0);
      if (score > bestScore) { bestScore = score; best = c; }
    }
    pick[i] = best;
    used.add(best);
  }
  parts.forEach((p, i) => { p.color = PALETTE[pick[i]]; p.rgb = rgb01(p.color); p.on = textOn(p.rgb); });
}

const _v = new THREE.Vector3();
const _w = new THREE.Vector3();

export class Map3D {
  constructor({ host, canvas, labels, leaders }) {
    this.host = host;
    this.canvas = canvas;
    this.labelsEl = labels;
    this.leadersEl = leaders;
    this.reserved = []; // map-local boxes labels must avoid (landed markers), set by the flights each frame
    this.parts = [];
    this.byId = new Map();
    this.frameFns = new Set();
    this.reduced = false;
    this.rotateSpeed = 0.6;
    this.ox = 0; this.oy = 0;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setClearColor(new THREE.Color().setRGB(...BG, THREE.SRGBColorSpace), 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(FOV, this.w / this.h, 0.1, 500);
    this.camera.position.set(0, 0, 30);
    this.controls = new OrbitControls(this.camera, canvas);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, enablePan: false, rotateSpeed: 0.5, zoomSpeed: 0.6 });

    this.uniforms = {
      uParts: { value: Array.from({ length: MAX_PARTS }, () => new THREE.Vector4(...Object.values(MODES.idle))) },
      uSize: { value: POINT_PX }, uScale: { value: 1 }, uRef: { value: 30 },
      uFogNear: { value: 20 }, uFogFar: { value: 40 },
      uGrey: { value: new THREE.Vector3(...GREY) }, uBg: { value: new THREE.Vector3(...BG) },
    };
    this.material = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false });
    this.last = performance.now();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  get w() { return this.host.clientWidth; }
  get h() { return this.host.clientHeight; }

  toWorld(xyz) {
    const [x = 0.5, y = 0.5, z = 0.5] = xyz ?? [];
    return new THREE.Vector3((x - 0.5) * SPAN, (0.5 - y) * SPAN, (z - 0.5) * SPAN);
  }

  part(id) { return this.byId.get(id); }
  camDist() { return this.camera.position.distanceTo(this.controls.target); }
  camUp() { return new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion); }
  onFrame(fn) { this.frameFns.add(fn); return () => this.frameFns.delete(fn); }
  setRotation(speed) { this.rotateSpeed = speed; }

  /** World point -> stage coordinates (px on the 1920x1080 stage). */
  project(v) {
    _v.copy(v).project(this.camera);
    return { x: this.ox + ((_v.x + 1) / 2) * this.w, y: this.oy + ((1 - _v.y) / 2) * this.h, depth: this.camera.position.distanceTo(v), behind: _v.z > 1 };
  }

  /** The world point at `dist` from the camera along the ray through a stage point (which may lie outside the canvas). */
  rayPoint(pt, dist) {
    _w.set(((pt.x - this.ox) / this.w) * 2 - 1, -(((pt.y - this.oy) / this.h) * 2 - 1), 0.5).unproject(this.camera).sub(this.camera.position).normalize();
    return this.camera.position.clone().addScaledVector(_w, dist);
  }

  setMap(map) {
    const src = (map?.parts ?? []).slice(0, MAX_PARTS);
    this.parts = src.map((p, i) => ({ id: p.id, label: p.label || p.id, i, src: p, pts: [], mode: 'idle', vis: { ...MODES.idle } }));
    this.byId = new Map(this.parts.map(p => [p.id, p]));
    const pts = (map?.points ?? []).filter(pt => this.byId.has(pt.part));
    const pos = new Float32Array(pts.length * 3), idx = new Float32Array(pts.length), col = new Float32Array(pts.length * 3);
    this.pointPart = new Uint8Array(pts.length);
    pts.forEach((pt, i) => {
      const v = this.toWorld([pt.x, pt.y, pt.z]);
      const p = this.byId.get(pt.part);
      pos.set([v.x, v.y, v.z], i * 3);
      idx[i] = p.i;
      this.pointPart[i] = p.i;
      p.pts.push(v);
    });
    for (const p of this.parts) {
      if (p.pts.length) {
        p.center = p.pts.reduce((a, v) => a.add(v), new THREE.Vector3()).divideScalar(p.pts.length);
        const ds = p.pts.map(v => v.distanceTo(p.center)).sort((a, b) => a - b);
        p.radius = ds[Math.floor(ds.length * 0.7)];
      } else {
        p.center = this.toWorld([p.src.x, p.src.y, p.src.z]);
        p.radius = 0.5;
      }
    }
    assignColors(this.parts);
    pts.forEach((pt, i) => col.set(this.byId.get(pt.part).rgb, i * 3));

    if (this.points) { this.scene.remove(this.points); this.points.geometry.dispose(); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aPart', new THREE.BufferAttribute(idx, 1));
    geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint16Array(pts.length), 1));
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.reorder();

    // Frame the cloud so it stays in view while it spins about the vertical axis: centre on a robust bounding box,
    // then fit the (rotation-invariant) height and the horizontal radius.
    const all = this.parts.flatMap(p => p.pts);
    const pct = (arr, q) => arr[Math.min(arr.length - 1, Math.floor(arr.length * q))];
    const axis = k => all.map(v => v[k]).sort((a, b) => a - b);
    const [xs, ys, zs] = ['x', 'y', 'z'].map(axis);
    const center = all.length
      ? new THREE.Vector3((pct(xs, 0.01) + pct(xs, 0.99)) / 2, (pct(ys, 0.01) + pct(ys, 0.99)) / 2, (pct(zs, 0.01) + pct(zs, 0.99)) / 2)
      : new THREE.Vector3();
    const halfY = all.length ? Math.max(center.y - pct(ys, 0.01), pct(ys, 0.99) - center.y) : SPAN / 2;
    const rXZ = all.length ? pct(all.map(v => Math.hypot(v.x - center.x, v.z - center.z)).sort((a, b) => a - b), 0.985) : SPAN / 2;
    const R = Math.hypot(halfY, rXZ);
    const tanV = Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * 0.86; // leave room for the map header
    const tanH = Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * (this.w / this.h) * 0.95;
    const dist = Math.max(halfY / tanV + rXZ * 0.55, rXZ / tanH + rXZ * 0.35);
    this.controls.target.copy(center);
    this.home = new THREE.Vector3(Math.sin(0.42) * Math.cos(0.2), Math.sin(0.2), Math.cos(0.42) * Math.cos(0.2)).multiplyScalar(dist).add(center);
    this.camera.position.copy(this.home);
    this.controls.minDistance = dist * 0.45;
    this.controls.maxDistance = dist * 1.8;
    this.controls.update();
    this.uniforms.uRef.value = dist;
    this.uniforms.uFogNear.value = dist - R * 0.3;
    this.uniforms.uFogFar.value = dist + R * 1.1;

    this.labelsEl.textContent = '';
    if (this.leadersEl) this.leadersEl.textContent = '';
    for (const p of this.parts) {
      if (this.leadersEl) {
        p.leader = document.createElementNS(SVG_NS, 'line');
        p.leader.setAttribute('opacity', '0');
        this.leadersEl.append(p.leader);
      }
      const el = document.createElement('div');
      el.className = 'plabel';
      el.textContent = p.label;
      el.style.setProperty('--c', p.color);
      el.style.setProperty('--on', p.on);
      this.labelsEl.append(el);
      p.el = el;
      p.lw = el.offsetWidth;
      p.lh = el.offsetHeight;
      p.slot = 0;
      p.shown = true;
      p.leaderOn = false;
    }
  }

  resize(k) {
    const ratio = clamp(k * (window.devicePixelRatio || 1), 0.5, 3);
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(this.w, this.h, false);
    this.camera.aspect = this.w / this.h;
    this.camera.setViewOffset(this.w, this.h, 0, -LABEL_TOP / 2, this.w, this.h);
    this.camera.updateProjectionMatrix();
    this.uniforms.uScale.value = ratio;
    this.ox = this.host.offsetLeft + this.host.clientLeft;
    this.oy = this.host.offsetTop + this.host.clientTop;
    this.leadersEl?.setAttribute('viewBox', `0 0 ${this.w} ${this.h}`);
  }

  setAll(mode) { for (const p of this.parts) this.setMode(p, mode); }
  light(id) { const p = this.byId.get(id); if (p) this.setMode(p, 'lit'); }
  setMode(p, mode) {
    if (p.mode === mode) return;
    p.mode = mode;
    p.el?.classList.toggle('lit', mode === 'lit');
    p.el?.classList.toggle('dim', mode === 'dim');
    this.orderDirty = true;
  }

  /** Draw order: dimmed parts first, lit parts last, so lit islands sit on top. */
  reorder() {
    const index = this.points?.geometry.index;
    if (!index) return;
    const rank = this.parts.map(p => ORDER[p.mode]);
    const ids = [...this.pointPart.keys()].sort((a, b) => rank[this.pointPart[a]] - rank[this.pointPart[b]] || a - b);
    index.array.set(ids);
    index.needsUpdate = true;
    this.orderDirty = false;
  }

  loop(now) {
    requestAnimationFrame(this.loop);
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.controls.autoRotate = !this.reduced && this.rotateSpeed > 0;
    this.controls.autoRotateSpeed = this.rotateSpeed;
    this.controls.update(dt);
    if (this.orderDirty) this.reorder();
    const f = this.reduced ? 1 : 1 - Math.exp(-dt / 0.16);
    for (const p of this.parts) {
      const t = MODES[p.mode], v = p.vis;
      for (const key in t) v[key] += (t[key] - v[key]) * f;
      this.uniforms.uParts.value[p.i].set(v.a, v.s, v.c, v.g);
    }
    this.renderer.render(this.scene, this.camera);
    this.placeLabels();
    for (const fn of this.frameFns) fn(now);
  }

  /**
   * Labels sit just above each island. On overlap a label tries nearby slots (with a thin leader line back to its
   * island); lit parts are placed first, then nearer parts. A label fades out only when no slot is free.
   */
  placeLabels() {
    if (!this.parts.length) return;
    const W = this.w, H = this.h, up = this.camUp();
    const items = this.parts.map(p => {
      _v.copy(p.center).addScaledVector(up, p.radius).project(this.camera);
      return { p, x: ((_v.x + 1) / 2) * W, y: ((1 - _v.y) / 2) * H, behind: _v.z > 1, depth: this.camera.position.distanceTo(p.center) };
    });
    items.sort((a, b) => (b.p.mode === 'lit') - (a.p.mode === 'lit') || a.depth - b.depth);
    const placed = [...this.reserved];
    const fits = (box, pad) => placed.every(q => box[2] + pad <= q[0] || q[2] + pad <= box[0] || box[3] + pad <= q[1] || q[3] + pad <= box[1]);
    const boxAt = (p, x, y, slot) => {
      const [sx, sy] = SLOTS[slot];
      const bx = clamp(x - p.lw / 2 + sx * (p.lw / 2 + 14), 8, W - p.lw - 8);
      const by = clamp(y - p.lh - 6 + sy * (p.lh + 6), LABEL_TOP, H - p.lh - 8);
      return [bx, by, bx + p.lw, by + p.lh];
    };
    for (const { p, x, y, behind } of items) {
      let chosen = null;
      for (const slot of [p.slot, ...SLOTS.keys()]) {
        const box = boxAt(p, x, y, slot);
        if (fits(box, p.shown && slot === p.slot ? -2 : 6)) { chosen = { slot, box }; break; }
      }
      if (!chosen && p.mode === 'lit') chosen = { slot: 0, box: boxAt(p, x, y, 0) };
      const show = !!chosen && !behind;
      if (show) {
        const [x0, y0, x1, y1] = chosen.box;
        placed.push(chosen.box);
        p.slot = chosen.slot;
        p.el.style.transform = `translate3d(${x0.toFixed(1)}px,${y0.toFixed(1)}px,0)`;
        // Leader from the island top to the nearest point of a displaced label.
        const lx = clamp(x, x0 + 6, x1 - 6), ly = clamp(y, y0, y1);
        const far = Math.hypot(lx - x, ly - y) > 16;
        if (p.leader) {
          if (far) {
            p.leader.setAttribute('x1', x.toFixed(1)); p.leader.setAttribute('y1', y.toFixed(1));
            p.leader.setAttribute('x2', lx.toFixed(1)); p.leader.setAttribute('y2', ly.toFixed(1));
            p.leader.style.stroke = p.mode === 'lit' ? p.color : '';
          }
          if (far !== p.leaderOn) { p.leaderOn = far; p.leader.setAttribute('opacity', far ? '1' : '0'); }
        }
      } else if (p.leaderOn) {
        p.leaderOn = false;
        p.leader?.setAttribute('opacity', '0');
      }
      if (show !== p.shown) { p.shown = show; p.el.classList.toggle('off', !show); }
    }
  }
}

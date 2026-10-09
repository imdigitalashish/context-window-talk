// The one authored animation: each clause flies in 3-D from its chip in the prompt panel to where it lands,
// leaving a trail; the whole prompt flies last. Recomputed every frame, so trails stay pinned to their chips
// and to their landing points while the cloud rotates.
import { Vector3 } from 'three';

const NS = 'http://www.w3.org/2000/svg';
const SAMPLES = 32;
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const bezier = (a, c, b, s, out) => out.copy(a).multiplyScalar((1 - s) * (1 - s)).addScaledVector(c, 2 * (1 - s) * s).addScaledVector(b, s * s);
const tmp = new Vector3();

export class Flights {
  constructor({ map, svg, layer }) {
    this.map = map;
    this.svg = svg;
    this.layer = layer;
    this.legs = [];
    this.waiters = [];
    map?.onFrame(now => this.frame(now));
  }

  clear() {
    this.legs = [];
    if (this.map) this.map.reserved = [];
    this.svg.textContent = '';
    this.layer.textContent = '';
    this.waiters.forEach(w => w.resolve(false));
    this.waiters = [];
  }

  /**
   * legs: [{ from: {x,y} stage px, xyz, color, on, touched, n, kind: 'clause'|'whole', label }]
   * Resolves true once every leg has landed, false if cleared first.
   */
  play(legs, { reduced = false, stagger = 450, duration = 1200, onLand = () => {} } = {}) {
    const t0 = performance.now() + 20;
    const added = legs.map((leg, i) => {
      const P = this.map ? this.spread(this.map.toWorld(leg.xyz)) : new Vector3();
      const path = document.createElementNS(NS, 'path');
      path.setAttribute('class', `trail${leg.kind === 'whole' ? ' whole' : leg.touched ? '' : ' miss'}`);
      if (leg.kind !== 'whole' && leg.touched) path.setAttribute('stroke', leg.color);
      this.svg.append(path);
      const mark = document.createElement('div');
      mark.className = `mark${leg.kind === 'whole' ? ' whole' : leg.touched ? '' : ' miss'}`;
      if (leg.kind === 'whole') {
        const tag = document.createElement('b');
        tag.textContent = leg.label ?? 'whole prompt';
        mark.append(tag);
      } else {
        mark.textContent = String(leg.n);
      }
      mark.style.setProperty('--c', leg.color);
      mark.style.setProperty('--on', leg.on ?? '#fff');
      return { ...leg, i, P, path, mark, start: reduced ? 0 : t0 + i * stagger, dur: reduced ? 0 : duration, state: 'wait', onLand };
    });
    this.legs.push(...added);
    return new Promise(resolve => {
      const w = { left: added.length, resolve };
      added.forEach(l => { l.waiter = w; });
      if (!added.length || !this.map) {
        added.forEach(l => l.onLand(l.i));
        resolve(true);
        return;
      }
      this.waiters.push(w);
    });
  }

  /** Nudges a landing point that would sit on top of an earlier one, so both markers stay readable. */
  spread(P) {
    const taken = this.legs.map(l => l.P);
    const right = new Vector3(1, 0, 0).applyQuaternion(this.map.camera.quaternion);
    const up = this.map.camUp();
    const min = 0.75; // world units; about one marker diameter at the default zoom
    for (let ring = 1; ring <= 3; ring++) {
      if (taken.every(Q => Q.distanceTo(P) >= min)) return P;
      for (let a = 0; a < 8; a++) {
        const ang = (a * Math.PI) / 4;
        const c = P.clone().addScaledVector(right, Math.cos(ang) * min * ring).addScaledVector(up, Math.sin(ang) * min * ring);
        if (taken.every(Q => Q.distanceTo(c) >= min)) return c;
      }
    }
    return P;
  }

  frame(now) {
    const reserved = [];
    for (const L of this.legs) {
      if (now < L.start) continue;
      const t = L.dur ? Math.min(1, (now - L.start) / L.dur) : 1;
      if (L.state === 'wait') { L.state = 'fly'; this.layer.append(L.mark); }
      this.draw(L, ease(t));
      if (t >= 1 && L.state === 'fly') {
        L.state = 'landed';
        L.mark.classList.add('landed');
        L.path.classList.add('done');
        L.onLand(L.i);
        if (--L.waiter.left === 0) { L.waiter.resolve(true); this.waiters = this.waiters.filter(w => w !== L.waiter); }
      }
      if (L.state === 'landed' && L.head) {
        const x = L.head.x - this.map.ox, y = L.head.y - this.map.oy, r = L.kind === 'whole' ? 19 : 22;
        reserved.push([x - r, y - r, x + r, y + r]);
        if (L.kind === 'whole') {
          L.tagW ??= L.mark.firstChild?.offsetWidth ?? 150;
          reserved.push([x - L.tagW / 2 - 4, y - r - 48, x + L.tagW / 2 + 4, y - r + 2]);
        }
      }
    }
    if (this.map) this.map.reserved = reserved;
  }

  draw(L, e) {
    const m = this.map;
    const start = m.rayPoint(L.from, m.camDist() * 0.4);
    const ctrl = start.clone().lerp(L.P, 0.5).addScaledVector(m.camUp(), start.distanceTo(L.P) * 0.14);
    let d = '';
    for (let i = 0; i <= SAMPLES; i++) {
      const p = m.project(bezier(start, ctrl, L.P, (e * i) / SAMPLES, tmp));
      d += `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
    }
    L.path.setAttribute('d', d);
    const h = (L.head = m.project(bezier(start, ctrl, L.P, e, tmp)));
    const s = Math.min(1.7, Math.max(0.85, m.camDist() / h.depth));
    L.mark.style.transform = `translate(${h.x.toFixed(1)}px,${h.y.toFixed(1)}px) scale(${s.toFixed(3)})`;
  }
}

/**
 * Occupancy NPZ → inside/outside point layers (file labels).
 * Does not import occupancy Python.
 */

import * as THREE from "../vendor/three.module.js";
import { parseOccupancyNpz } from "./npy_npz.js";

export const DRAW_CAP = 200000;

export const COLOR_INSIDE = new THREE.Color(0xffaa00);
export const COLOR_OUTSIDE = new THREE.Color(0x4a6d8c);
export const COLOR_FN = new THREE.Color(0xe85d5d);
export const COLOR_FP = new THREE.Color(0x7ec8e3);

/**
 * @param {File|null|undefined} file
 * @returns {boolean}
 */
export function isNpzFile(file) {
  const name = String(file && file.name ? file.name : "").toLowerCase();
  return name.endsWith(".npz");
}

/**
 * Even stride so a huge file still paints, with a stable subset.
 * @param {number} n
 * @param {number} cap
 * @returns {number[]|null}
 */
export function subsampleIndices(n, cap) {
  if (n <= cap) {
    return null;
  }
  const indices = new Array(cap);
  const step = n / cap;
  for (let i = 0; i < cap; i += 1) {
    indices[i] = Math.min(n - 1, Math.floor(i * step));
  }
  return indices;
}

/**
 * @param {Float32Array} points
 * @param {Float32Array} labels
 * @param {boolean} wantInside
 * @param {number[]|null} pick
 * @returns {Float32Array}
 */
function collectPositions(points, labels, wantInside, pick) {
  const n = labels.length;
  const chosen = [];
  const visit = pick || null;
  const count = visit ? visit.length : n;
  for (let d = 0; d < count; d += 1) {
    const i = visit ? visit[d] : d;
    if ((labels[i] > 0.5) !== wantInside) {
      continue;
    }
    chosen.push(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
  }
  return new Float32Array(chosen);
}

/**
 * @param {Float32Array} xyz
 * @param {THREE.Color} color
 * @param {string} name
 * @param {number} size
 * @returns {THREE.Points|null}
 */
function makeLayer(xyz, color, name, size) {
  if (xyz.length < 3) {
    return null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
  const material = new THREE.PointsMaterial({
    size,
    color,
    sizeAttenuation: true,
  });
  const pts = new THREE.Points(geometry, material);
  pts.name = name;
  return pts;
}

/**
 * Build (or rebuild) inside/outside layers from full NPZ arrays.
 * @param {Float32Array} points
 * @param {Float32Array|Uint8Array} labels
 * @param {number} cap
 * @param {number} pointSize
 * @param {{inside?: THREE.Color, outside?: THREE.Color}} [colors]
 * @returns {{inside: THREE.Points|null, outside: THREE.Points|null, nDraw: number, nDrawInside: number, nDrawOutside: number, radius: number}}
 */
export function buildNpzLayers(points, labels, cap, pointSize, colors) {
  const n = labels.length;
  const pick = subsampleIndices(n, cap);
  const cin = (colors && colors.inside) || COLOR_INSIDE;
  const cout = (colors && colors.outside) || COLOR_OUTSIDE;
  const insideXyz = collectPositions(points, labels, true, pick);
  const outsideXyz = collectPositions(points, labels, false, pick);
  const nDrawInside = insideXyz.length / 3;
  const nDrawOutside = outsideXyz.length / 3;
  const inside = makeLayer(insideXyz, cin, "npz-inside", pointSize);
  const outside = makeLayer(outsideXyz, cout, "npz-outside", pointSize);
  let radius = 1;
  const probe = inside || outside;
  if (probe) {
    probe.geometry.computeBoundingSphere();
    if (probe.geometry.boundingSphere) {
      radius = Math.max(probe.geometry.boundingSphere.radius, 1e-3);
    }
  }
  return {
    inside,
    outside,
    nDraw: nDrawInside + nDrawOutside,
    nDrawInside,
    nDrawOutside,
    radius,
  };
}

/**
 * Wrong points only: FN (gt inside, pred outside) and FP (gt outside, pred inside).
 * @param {Float32Array} points
 * @param {Float32Array} gt
 * @param {Uint8Array|Float32Array} pred
 * @param {number} cap
 * @param {number} pointSize
 */
export function buildErrorLayers(points, gt, pred, cap, pointSize) {
  const n = gt.length;
  const pick = subsampleIndices(n, cap);
  const fn = [];
  const fp = [];
  const visit = pick;
  const count = visit ? visit.length : n;
  for (let d = 0; d < count; d += 1) {
    const i = visit ? visit[d] : d;
    const g = gt[i] > 0.5;
    const p = pred[i] > 0.5;
    if (g === p) {
      continue;
    }
    if (g) {
      fn.push(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    } else {
      fp.push(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    }
  }
  const inside = makeLayer(new Float32Array(fn), COLOR_FN, "npz-inside", pointSize);
  const outside = makeLayer(new Float32Array(fp), COLOR_FP, "npz-outside", pointSize);
  return {
    inside,
    outside,
    nDraw: fn.length / 3 + fp.length / 3,
    nDrawInside: fn.length / 3,
    nDrawOutside: fp.length / 3,
    radius: 1,
  };
}

/**
 * @param {File} file
 * @returns {Promise<{
 *   points: Float32Array,
 *   labels: Float32Array,
 *   n: number,
 *   nInside: number,
 *   nOutside: number,
 *   meshPath: string,
 *   layers: ReturnType<typeof buildNpzLayers>
 * }>}
 */
export async function parseNpzFile(file) {
  const buffer = await file.arrayBuffer();
  const parsed = await parseOccupancyNpz(buffer);
  const n = parsed.n;
  if (n === 0) {
    throw new Error("NPZ has no points");
  }
  let nInside = 0;
  for (let i = 0; i < n; i += 1) {
    if (parsed.labels[i] > 0.5) {
      nInside += 1;
    }
  }
  const layers = buildNpzLayers(parsed.points, parsed.labels, DRAW_CAP, 0.05);
  return {
    points: parsed.points,
    labels: parsed.labels,
    n,
    nInside,
    nOutside: n - nInside,
    nDraw: layers.nDraw,
    meshPath: parsed.meshPath,
    fileName: file && file.name ? file.name : "",
    layers,
  };
}

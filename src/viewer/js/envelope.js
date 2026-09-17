/**
 * Envelope overlay: purple dots plus a short tick along each face normal.
 * Same samples as occupancy (XYZ + unit normal). Does not classify.
 */

import * as THREE from "../vendor/three.module.js";
import { base64ToFloat32, readHelperJson } from "./model_panel.js";

export const COLOR_ENVELOPE = new THREE.Color(0xb56bff);
export const N_SURFACE_MIN = 256;
export const N_SURFACE_MAX = 4096;
export const N_SURFACE_DEFAULT = 1024;

/**
 * Clamp the envelope count slider to the helper range.
 * @param {string|number} raw
 * @returns {number}
 */
export function clampNSurface(raw) {
  const n = Math.round(Number(raw));
  if (!Number.isFinite(n)) {
    return N_SURFACE_DEFAULT;
  }
  return Math.min(N_SURFACE_MAX, Math.max(N_SURFACE_MIN, n));
}

/**
 * Tick length from the cloud's AABB diagonal (visible, not mesh-scale locked).
 * @param {Float32Array} xyz
 * @returns {number}
 */
export function envelopeTickLength(xyz) {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < xyz.length; i += 3) {
    const x = xyz[i];
    const y = xyz[i + 1];
    const z = xyz[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
  return Math.max(diag * 0.04, 1e-4);
}

/**
 * Short segment from each sample along its stored face normal.
 * @param {Float32Array} xyz
 * @param {Float32Array} nrm
 * @param {number} length
 * @returns {THREE.LineSegments}
 */
export function makeEnvelopeNormalTicks(xyz, nrm, length) {
  const n = xyz.length / 3;
  const segs = new Float32Array(n * 6);
  const len = Number(length);
  for (let i = 0; i < n; i += 1) {
    const i3 = i * 3;
    const o = i * 6;
    segs[o] = xyz[i3];
    segs[o + 1] = xyz[i3 + 1];
    segs[o + 2] = xyz[i3 + 2];
    segs[o + 3] = xyz[i3] + nrm[i3] * len;
    segs[o + 4] = xyz[i3 + 1] + nrm[i3 + 1] * len;
    segs[o + 5] = xyz[i3 + 2] + nrm[i3 + 2] * len;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(segs, 3));
  const material = new THREE.LineBasicMaterial({
    color: COLOR_ENVELOPE,
    transparent: true,
    opacity: 0.75,
  });
  const lines = new THREE.LineSegments(geometry, material);
  lines.name = "envelope-normals";
  return lines;
}

/**
 * Purple dots plus normal ticks in one group (``envelope-overlay``).
 * @param {Float32Array} xyz
 * @param {Float32Array|null} nrm
 * @param {number} pointSize
 * @returns {THREE.Group}
 */
export function makeEnvelopeLayer(xyz, pointSize, nrm) {
  const group = new THREE.Group();
  group.name = "envelope-overlay";
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
  const material = new THREE.PointsMaterial({
    size: pointSize,
    color: COLOR_ENVELOPE,
    sizeAttenuation: true,
  });
  const pts = new THREE.Points(geometry, material);
  pts.name = "envelope-points";
  group.add(pts);
  if (nrm && nrm.length === xyz.length) {
    group.add(makeEnvelopeNormalTicks(xyz, nrm, envelopeTickLength(xyz)));
  }
  return group;
}

/**
 * @param {{objText: string, nSurface: number}} payload
 * @returns {Promise<{points: Float32Array, normals: Float32Array|null, n: number}>}
 */
export async function envelopeObjOnHelper(payload) {
  const res = await fetch("/api/envelope-obj", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      obj_text: payload.objText,
      n_surface: payload.nSurface,
    }),
  });
  const body = await readHelperJson(res);
  if (!body || !body.points_b64) {
    throw new Error("helper returned no envelope points");
  }
  const xyz = base64ToFloat32(body.points_b64);
  const n = Number(body.n) || xyz.length / 3;
  if (xyz.length !== n * 3) {
    throw new Error("envelope buffer length does not match n");
  }
  let normals = null;
  if (body.normals_b64) {
    normals = base64ToFloat32(body.normals_b64);
    if (normals.length !== n * 3) {
      throw new Error("envelope normal buffer length does not match n");
    }
  }
  return {
    points: xyz,
    normals: normals,
    n: n,
  };
}

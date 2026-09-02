/**
 * Envelope overlay: purple dots hug sharp edges (helper).
 * Does not classify occupancy and does not import occupancy Python.
 */

import * as THREE from "../vendor/three.module.js";
import { base64ToFloat32, readHelperJson } from "./model_panel.js";

export const COLOR_ENVELOPE = new THREE.Color(0xb56bff);
export const N_SURFACE_MIN = 256;
export const N_SURFACE_MAX = 4096;
export const N_SURFACE_DEFAULT = 1024;
export const MIX_MIN = 0;
export const MIX_MAX = 100;
export const MIX_DEFAULT = 100;

/**
 * Clamp the Faces↔Edges mix slider (0 = faces, 100 = edges).
 * @param {string|number} raw
 * @returns {number}
 */
export function clampEnvelopeMix(raw) {
  const n = Math.round(Number(raw));
  if (!Number.isFinite(n)) {
    return MIX_DEFAULT;
  }
  return Math.min(MIX_MAX, Math.max(MIX_MIN, n));
}

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
 * @param {Float32Array} xyz
 * @param {number} pointSize
 * @returns {THREE.Points}
 */
export function makeEnvelopeLayer(xyz, pointSize) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
  const material = new THREE.PointsMaterial({
    size: pointSize,
    color: COLOR_ENVELOPE,
    sizeAttenuation: true,
  });
  const pts = new THREE.Points(geometry, material);
  pts.name = "envelope-points";
  return pts;
}

/**
 * @param {{objText: string, nSurface: number, mix: number}} payload
 * @returns {Promise<{points: Float32Array, n: number, nCreases: number, nArea: number, nEdge: number}>}
 */
export async function envelopeObjOnHelper(payload) {
  const res = await fetch("/api/envelope-obj", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      obj_text: payload.objText,
      n_surface: payload.nSurface,
      mix: payload.mix,
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
  return {
    points: xyz,
    n: n,
    nCreases: Number(body.n_creases) || 0,
    nArea: Number(body.n_area) || 0,
    nEdge: Number(body.n_edge) || 0,
  };
}

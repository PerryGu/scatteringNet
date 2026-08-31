/**
 * Job B: fill an OBJ AABB lattice, then infer.
 * Slider 0 = spacing 0.40 (coarse), 100 = 0.05 (dense).
 */

import {
  float32ToBase64,
  base64ToFloat32,
  base64ToUint8,
  readHelperJson,
} from "./model_panel.js";

export const SPACING_COARSE = 0.4;
export const SPACING_FINE = 0.05;

/**
 * @param {number} value 0-100
 * @returns {number}
 */
export function spacingFromSlider(value) {
  const t = Math.min(1, Math.max(0, Number(value) / 100));
  return SPACING_COARSE + (SPACING_FINE - SPACING_COARSE) * t;
}

/**
 * @param {{objText: string, spacing: number}} payload
 */
export async function fillObjOnHelper(payload) {
  const res = await fetch("/api/fill-obj", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      obj_text: payload.objText,
      spacing: payload.spacing,
    }),
  });
  const body = await readHelperJson(res);
  if (!body || !body.points_b64) {
    throw new Error("helper returned no fill points");
  }
  const points = base64ToFloat32(body.points_b64);
  if (points.length < 3) {
    throw new Error("fill produced no points");
  }
  return {
    points,
    n: Number(body.n) || points.length / 3,
    usedSpacing: Number(body.used_spacing) || payload.spacing,
    grid: body.grid || null,
  };
}

/**
 * @param {{
 *   checkpoint: string,
 *   name: string,
 *   objText: string,
 *   points: Float32Array
 * }} payload
 */
export async function inferObjOnHelper(payload) {
  const res = await fetch("/api/infer-obj", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      checkpoint: payload.checkpoint,
      name: payload.name,
      obj_text: payload.objText,
      points_b64: float32ToBase64(payload.points),
    }),
  });
  const body = await readHelperJson(res);
  if (!body || !body.pred_b64) {
    throw new Error("helper returned no predictions");
  }
  const pred = base64ToUint8(body.pred_b64);
  const n = payload.points.length / 3;
  if (pred.length !== n) {
    throw new Error("prediction length " + pred.length + " does not match fill " + n);
  }
  return { pred, metrics: body };
}

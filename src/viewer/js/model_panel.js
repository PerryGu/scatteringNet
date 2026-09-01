/**
 * List models/<run_id>/best.pt and POST NPZ points to the helper.
 * Also holds wire codecs shared with Job B.
 */

/**
 * @param {Uint8Array} bytes
 * @returns {string}
 */
function bytesToBase64(bytes) {
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/**
 * @param {Float32Array} data
 * @returns {string}
 */
export function float32ToBase64(data) {
  return bytesToBase64(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

/**
 * @param {Uint8Array} data
 * @returns {string}
 */
export function uint8ToBase64(data) {
  return bytesToBase64(data);
}

/**
 * @param {string} b64
 * @returns {Uint8Array}
 */
export function base64ToUint8(b64) {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

/**
 * @param {string} b64
 * @returns {Float32Array}
 */
export function base64ToFloat32(b64) {
  const bytes = base64ToUint8(b64);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error("points buffer is not float32");
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/**
 * File labels as uint8 {0,1} for the helper.
 * @param {Float32Array|Uint8Array} labels
 * @returns {Uint8Array}
 */
export function labelsToUint8(labels) {
  const n = labels.length;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = labels[i] > 0.5 ? 1 : 0;
  }
  return out;
}

/**
 * Parse a helper JSON body. Throws using the helper error field when status is not OK.
 * @param {Response} res
 */
export async function readHelperJson(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const detail =
      body && body.error ? String(body.error) : res.status + " " + res.statusText;
    throw new Error(detail);
  }
  return body;
}

/**
 * Slider 0-100 maps to a sigmoid cut in [0, 1]. Default 50 = 0.50.
 * @param {string|number} raw
 * @returns {number}
 */
export function thresholdFromCutSlider(raw) {
  const v = Number(raw);
  if (!Number.isFinite(v)) {
    return 0.5;
  }
  return Math.min(1, Math.max(0, v / 100));
}

/**
 * Hard inside labels from stored sigmoid probabilities.
 * @param {Float32Array} probs
 * @param {number} threshold
 * @returns {Uint8Array}
 */
export function predFromProbs(probs, threshold) {
  const t = Math.min(1, Math.max(0, Number(threshold)));
  const pred = new Uint8Array(probs.length);
  for (let i = 0; i < probs.length; i += 1) {
    pred[i] = probs[i] >= t ? 1 : 0;
  }
  return pred;
}

/**
 * Optional float32 sigmoid vector from the helper (one value per query).
 * @param {{prob_b64?: string}} body
 * @param {number} n
 * @returns {Float32Array|null}
 */
export function decodeProbB64(body, n) {
  if (!body || typeof body.prob_b64 !== "string" || !body.prob_b64) {
    return null;
  }
  const probs = base64ToFloat32(body.prob_b64);
  if (probs.length !== n) {
    throw new Error(
      "probability length " + probs.length + " does not match points " + n
    );
  }
  return probs;
}

/**
 * @returns {Promise<{id: string, path: string, shape_encoder?: string}[]>}
 */
export async function fetchModelList() {
  const res = await fetch("/api/models");
  const body = await readHelperJson(res);
  return Array.isArray(body.models) ? body.models : [];
}

/**
 * @param {{
 *   checkpoint: string,
 *   name: string,
 *   meshPath: string,
 *   points: Float32Array,
 *   labels: Float32Array
 * }} payload
 */
export async function inferNpzOnHelper(payload) {
  const res = await fetch("/api/infer-npz", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      checkpoint: payload.checkpoint,
      name: payload.name,
      mesh_path: payload.meshPath,
      points_b64: float32ToBase64(payload.points),
      labels_b64: uint8ToBase64(labelsToUint8(payload.labels)),
    }),
  });
  const body = await readHelperJson(res);
  if (!body || !body.pred_b64) {
    throw new Error("helper returned no predictions");
  }
  const pred = base64ToUint8(body.pred_b64);
  if (pred.length !== payload.labels.length) {
    throw new Error(
      "prediction length " + pred.length + " does not match points " + payload.labels.length
    );
  }
  const probs = decodeProbB64(body, payload.labels.length);
  return { pred, probs, metrics: body };
}

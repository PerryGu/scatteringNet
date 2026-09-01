/**
 * Face-token overlay: teal triangles + normal ticks from the occupancy mesh head.
 * Does not classify occupancy and does not import occupancy Python.
 */

import * as THREE from "../vendor/three.module.js";
import { base64ToFloat32, readHelperJson } from "./model_panel.js";

export const COLOR_FACES = new THREE.Color(0x2ec4b6);
export const COLOR_NORMALS = new THREE.Color(0x7ef0e4);
export const N_FACES_MIN = 64;
export const N_FACES_MAX = 1024;
export const N_FACES_DEFAULT = 256;

/**
 * @param {string|number} raw
 * @returns {number}
 */
export function clampNFaces(raw) {
  const n = Math.round(Number(raw));
  if (!Number.isFinite(n)) {
    return N_FACES_DEFAULT;
  }
  return Math.min(N_FACES_MAX, Math.max(N_FACES_MIN, n));
}

/**
 * Unpack ``(n, 12)`` tokens into a group: translucent faces + normal ticks.
 * @param {Float32Array} tokens
 * @param {number} n
 * @param {number} tick
 * @returns {THREE.Group}
 */
export function makeFaceTokenGroup(tokens, n, tick) {
  const group = new THREE.Group();
  group.name = "face-tokens";
  const triPos = new Float32Array(n * 9);
  const linePos = new Float32Array(n * 6);
  const t = Number(tick) || 0.05;
  for (let i = 0; i < n; i += 1) {
    const o = i * 12;
    const v0x = tokens[o];
    const v0y = tokens[o + 1];
    const v0z = tokens[o + 2];
    const v1x = tokens[o + 3];
    const v1y = tokens[o + 4];
    const v1z = tokens[o + 5];
    const v2x = tokens[o + 6];
    const v2y = tokens[o + 7];
    const v2z = tokens[o + 8];
    const nx = tokens[o + 9];
    const ny = tokens[o + 10];
    const nz = tokens[o + 11];
    const tp = i * 9;
    triPos[tp] = v0x;
    triPos[tp + 1] = v0y;
    triPos[tp + 2] = v0z;
    triPos[tp + 3] = v1x;
    triPos[tp + 4] = v1y;
    triPos[tp + 5] = v1z;
    triPos[tp + 6] = v2x;
    triPos[tp + 7] = v2y;
    triPos[tp + 8] = v2z;
    const cx = (v0x + v1x + v2x) / 3;
    const cy = (v0y + v1y + v2y) / 3;
    const cz = (v0z + v1z + v2z) / 3;
    const lp = i * 6;
    linePos[lp] = cx;
    linePos[lp + 1] = cy;
    linePos[lp + 2] = cz;
    linePos[lp + 3] = cx + nx * t;
    linePos[lp + 4] = cy + ny * t;
    linePos[lp + 5] = cz + nz * t;
  }
  const triGeom = new THREE.BufferGeometry();
  triGeom.setAttribute("position", new THREE.BufferAttribute(triPos, 3));
  triGeom.computeVertexNormals();
  const mesh = new THREE.Mesh(
    triGeom,
    new THREE.MeshBasicMaterial({
      color: COLOR_FACES,
      transparent: true,
      opacity: 0.42,
      side: THREE.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    })
  );
  mesh.name = "face-token-tris";
  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
  const lines = new THREE.LineSegments(
    lineGeom,
    new THREE.LineBasicMaterial({ color: COLOR_NORMALS })
  );
  lines.name = "face-token-normals";
  group.add(mesh);
  group.add(lines);
  return group;
}

/**
 * @param {{objText: string, nFaces: number}} payload
 * @returns {Promise<{tokens: Float32Array, n: number, nMesh: number, nUnique: number, tick: number}>}
 */
export async function facesObjOnHelper(payload) {
  const res = await fetch("/api/faces-obj", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      obj_text: payload.objText,
      n_faces: payload.nFaces,
    }),
  });
  const body = await readHelperJson(res);
  if (!body || !body.tokens_b64) {
    throw new Error("helper returned no face tokens");
  }
  const tokens = base64ToFloat32(body.tokens_b64);
  const n = Number(body.n) || 0;
  if (tokens.length !== n * 12) {
    throw new Error("face-token buffer length does not match n");
  }
  return {
    tokens: tokens,
    n: n,
    nMesh: Number(body.n_mesh) || 0,
    nUnique: Number(body.n_unique) || n,
    tick: Number(body.tick) || 0.05,
  };
}

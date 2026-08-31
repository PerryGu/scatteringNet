/**
 * Viewer Step 2: parse a user-picked OBJ into a Three.js group.
 * Does not import occupancy Python. NPZ handling is in load_npz.js.
 */

import * as THREE from "../vendor/three.module.js";
import { OBJLoader } from "../vendor/loaders/OBJLoader.js";

/** Quiet solid, double-sided so thin occupancy shells read from both sides. */
const MESH_COLOR = 0xb8c0cc;

/**
 * @param {File|null|undefined} file
 * @returns {boolean}
 */
export function isObjFile(file) {
  const name = String(file && file.name ? file.name : "").toLowerCase();
  return name.endsWith(".obj");
}

/**
 * Read Wavefront OBJ text and return a styled group.
 * @param {string} text
 * @param {{opacity?: number, name?: string}} [opts]
 * @returns {THREE.Object3D}
 */
export function parseObjText(text, opts) {
  if (!String(text).trim()) {
    throw new Error("OBJ file is empty");
  }
  const group = new OBJLoader().parse(text);
  return styleObjGroup(group, opts);
}

/**
 * Read the file as text and parse Wavefront OBJ.
 * @param {File} file
 * @returns {Promise<THREE.Group>}
 */
export async function parseObjFile(file) {
  return parseObjText(await file.text());
}

/**
 * One MeshStandardMaterial on every mesh (ignore missing MTL).
 * Overlay meshes (Show object) use opacity < 1 so NPZ points stay visible.
 * @param {THREE.Object3D} group
 * @param {{opacity?: number, name?: string}} [opts]
 * @returns {THREE.Object3D}
 */
export function styleObjGroup(group, opts) {
  const opacity = opts && opts.opacity != null ? opts.opacity : 1;
  const name = opts && opts.name ? opts.name : "loaded-obj";
  const material = new THREE.MeshStandardMaterial({
    color: MESH_COLOR,
    metalness: 0.08,
    roughness: 0.48,
    side: THREE.DoubleSide,
    transparent: opacity < 0.999,
    opacity,
    depthWrite: opacity >= 0.999,
  });
  let meshCount = 0;
  group.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    child.material = material;
    child.castShadow = false;
    child.receiveShadow = false;
    meshCount += 1;
  });
  if (meshCount === 0) {
    throw new Error("OBJ has no faces to draw");
  }
  group.name = name;
  return group;
}

/**
 * Point the orbit camera at the mesh AABB (dataset coordinates, no occupancy normalize).
 * @param {THREE.PerspectiveCamera} camera
 * @param {import("../vendor/controls/OrbitControls.js").OrbitControls} controls
 * @param {THREE.Object3D} object
 * @returns {{center: THREE.Vector3, radius: number, yMin: number}}
 */
export function fitCameraToObject(camera, controls, object) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) {
    throw new Error("OBJ bounding box is empty");
  }
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 1e-6);

  camera.near = Math.max(radius / 200, 1e-4);
  camera.far = Math.max(radius * 80, 50);
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  camera.position.set(
    center.x + radius * 1.45,
    center.y + radius * 0.95,
    center.z + radius * 1.45
  );
  controls.minDistance = Math.max(radius * 0.05, 1e-3);
  controls.maxDistance = radius * 40;
  controls.update();

  return { center, radius, yMin: box.min.y };
}

/**
 * Drop GPU buffers from a previous OBJ so a second load does not leak.
 * @param {THREE.Object3D|null} root
 */
export function disposeObject3d(root) {
  if (!root) {
    return;
  }
  const materials = new Set();
  root.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    const mat = child.material;
    if (!mat) {
      return;
    }
    if (Array.isArray(mat)) {
      mat.forEach((m) => materials.add(m));
    } else {
      materials.add(mat);
    }
  });
  materials.forEach((m) => m.dispose());
}

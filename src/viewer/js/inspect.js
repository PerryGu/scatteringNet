/**
 * Step 5 inspect helpers: mesh opacity, wire overlay, point size, layer vis.
 */

import * as THREE from "../vendor/three.module.js";

/**
 * @param {THREE.Object3D|null} root
 * @param {(mesh: THREE.Mesh) => void} fn
 */
export function forEachMesh(root, fn) {
  if (!root) {
    return;
  }
  root.traverse((child) => {
    if (child.isMesh) {
      fn(child);
    }
  });
}

/**
 * Edges overlay parented to each mesh so it follows the solid.
 * @param {THREE.Object3D} root
 * @param {boolean} visible
 */
export function ensureWireOverlays(root, visible) {
  forEachMesh(root, (mesh) => {
    if (mesh.userData.wire) {
      mesh.userData.wire.visible = visible;
      return;
    }
    const edges = new THREE.EdgesGeometry(mesh.geometry, 25);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({
        color: 0xe4e8ee,
        transparent: true,
        opacity: 0.9,
      })
    );
    line.name = "wire-overlay";
    line.visible = visible;
    mesh.add(line);
    mesh.userData.wire = line;
  });
}

/**
 * @param {THREE.Object3D|null} root
 * @param {number} opacity 0–1
 */
export function applyMeshOpacity(root, opacity) {
  const clamped = Math.min(1, Math.max(0, opacity));
  forEachMesh(root, (mesh) => {
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    mats.forEach((mat) => {
      if (!mat || mat.isLineBasicMaterial) {
        return;
      }
      mat.opacity = clamped;
      mat.transparent = clamped < 0.999;
      mat.depthWrite = clamped >= 0.999;
      mat.needsUpdate = true;
    });
  });
}

/**
 * @param {THREE.Object3D|null} root
 * @param {number} size
 */
export function applyPointSize(root, size) {
  if (!root) {
    return;
  }
  root.traverse((child) => {
    if (child.isPoints && child.material) {
      child.material.size = size;
      child.material.needsUpdate = true;
    }
  });
}

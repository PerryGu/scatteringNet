/**
 * Occupancy viewer: orbit, OBJ/NPZ inspect, Job A (NPZ infer), Job B (OBJ fill + infer).
 * Does not import occupancy Python in the browser.
 */

import * as THREE from "../vendor/three.module.js";
import { OrbitControls } from "../vendor/controls/OrbitControls.js";
import { bindRecentsHover, persistRecentFile, resolveRecentFile } from "./recents.js";
import {
  disposeObject3d,
  fitCameraToObject,
  isObjFile,
  parseObjText,
} from "./load_obj.js";
import {
  isNpzFile,
  parseNpzFile,
  buildNpzLayers,
  buildErrorLayers,
} from "./load_npz.js";
import { fetchMeshObjText, meshBasename } from "./show_object.js";
import { applyMeshOpacity, applyPointSize, ensureWireOverlays, forEachMesh } from "./inspect.js";
import {
  fetchModelList,
  inferNpzOnHelper,
  predFromProbs,
  thresholdFromCutSlider,
} from "./model_panel.js";
import { fillObjOnHelper, inferObjOnHelper, spacingFromSlider } from "./obj_infer.js";
import { clampNSurface, envelopeObjOnHelper, makeEnvelopeLayer } from "./envelope.js";
import { clampNFaces, facesObjOnHelper, makeFaceTokenGroup } from "./faces.js";
import { fetchUiPrefs, postUiPrefs } from "./ui_prefs.js";

const container = document.getElementById("canvas-container");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const dropHint = document.getElementById("drop-hint");
const openWrap = document.getElementById("open-wrap");
const recentsList = document.getElementById("recents-list");
const btnOpen = document.getElementById("btn-open");
const fileInput = document.getElementById("file-input");
const objFallbackInput = document.getElementById("obj-fallback-input");
const npzPanel = document.getElementById("npz-panel");
const npzCounts = document.getElementById("npz-counts");
const npzNInside = document.getElementById("npz-n-inside");
const npzNOutside = document.getElementById("npz-n-outside");
const npzPath = document.getElementById("npz-path");
const inspectPanel = document.getElementById("inspect-panel");
const togMesh = document.getElementById("tog-mesh");
const togInside = document.getElementById("tog-inside");
const togOutside = document.getElementById("tog-outside");
const togWire = document.getElementById("tog-wire");
const sldOpacity = document.getElementById("sld-opacity");
const valOpacity = document.getElementById("val-opacity");
const sldPsize = document.getElementById("sld-psize");
const valPsize = document.getElementById("val-psize");
const sldCap = document.getElementById("sld-cap");
const valCap = document.getElementById("val-cap");
const selModel = document.getElementById("sel-model");
const btnRun = document.getElementById("btn-run");
const btnViewTruth = document.getElementById("btn-view-truth");
const btnViewPred = document.getElementById("btn-view-pred");
const btnViewErrors = document.getElementById("btn-view-errors");
const labInside = document.getElementById("lab-inside");
const labOutside = document.getElementById("lab-outside");
const legWordIn = document.getElementById("leg-word-in");
const legWordOut = document.getElementById("leg-word-out");
const legSwatchIn = document.getElementById("leg-swatch-in");
const legSwatchOut = document.getElementById("leg-swatch-out");
const fillPanel = document.getElementById("fill-panel");
const btnFill = document.getElementById("btn-fill");
const sldDensity = document.getElementById("sld-density");
const valDensity = document.getElementById("val-density");
const btnEnvelope = document.getElementById("btn-envelope");
const sldEnvelope = document.getElementById("sld-envelope");
const valEnvelope = document.getElementById("val-envelope");
const btnFaces = document.getElementById("btn-faces");
const sldFaces = document.getElementById("sld-faces");
const valFaces = document.getElementById("val-faces");
const sldCut = document.getElementById("sld-cut");
const valCut = document.getElementById("val-cut");

let scene = null;
let camera = null;
let controls = null;
let keyLight = null;
let loadedRoot = null;
let shownMesh = null;
let currentMeshPath = "";
let npzState = null;
let viewMode = "truth";
let inferBusy = false;
let fillBusy = false;
let fillTimer = 0;
let fillPending = false;
let envelopeBusy = false;
let envelopeTimer = 0;
let envelopeVisible = false;
let facesBusy = false;
let facesTimer = 0;
let facesVisible = false;
let objText = "";
let objFileName = "";
let basePointSize = 0.05;
let wireOn = false;
let gridHelper = null;
let axesHelper = null;
let prefsReady = false;
let prefsTimer = 0;

function cancelPendingFill() {
  window.clearTimeout(fillTimer);
  fillTimer = 0;
  fillPending = false;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function hideNpzPanel() {
  npzPanel.hidden = true;
  npzCounts.textContent = "";
  npzNInside.textContent = "";
  npzNOutside.textContent = "";
  npzPath.textContent = "";
  npzPath.removeAttribute("title");
  currentMeshPath = "";
}

function hideInspectPanel() {
  inspectPanel.hidden = true;
}

function isFillJob() {
  return !!(npzState && npzState.source === "fill");
}

function isNpzJob() {
  return !!(npzState && npzState.source !== "fill");
}

function syncFillPanel() {
  const hasMesh = !!objText;
  const jobB = hasMesh && !isNpzJob();
  if (fillPanel) {
    fillPanel.hidden = !hasMesh;
  }
  if (btnFill) {
    btnFill.disabled = !jobB || fillBusy || inferBusy;
  }
  if (btnEnvelope) {
    btnEnvelope.disabled = !hasMesh || envelopeBusy || inferBusy;
    btnEnvelope.classList.toggle("on", envelopeVisible);
  }
  if (btnFaces) {
    btnFaces.disabled = !hasMesh || facesBusy || inferBusy;
    btnFaces.classList.toggle("on", facesVisible);
  }
}

function syncModelPanel() {
  const hasModel = !!(selModel && selModel.value);
  const hasPred = !!(npzState && npzState.pred);
  const canRun = isNpzJob() || isFillJob();
  btnRun.disabled = !canRun || !hasModel || inferBusy || fillBusy;
  btnViewTruth.disabled = !isNpzJob();
  btnViewPred.disabled = !hasPred;
  btnViewErrors.disabled = !isNpzJob() || !hasPred;
  [btnViewTruth, btnViewPred, btnViewErrors].forEach((btn) => {
    if (!btn) {
      return;
    }
    btn.classList.toggle("on", btn.dataset.view === viewMode);
  });
  syncFillPanel();
}

function collectUiPrefs() {
  return {
    mesh: !!(togMesh && togMesh.checked),
    inside: !!(togInside && togInside.checked),
    outside: !!(togOutside && togOutside.checked),
    wireframe: !!(togWire && togWire.checked),
    opacity: Number(sldOpacity && sldOpacity.value),
    point_size: Number(sldPsize && sldPsize.value),
    draw_cap: Number(sldCap && sldCap.value),
    density: Number(sldDensity && sldDensity.value),
    envelope_n: clampNSurface(sldEnvelope && sldEnvelope.value),
    faces_n: clampNFaces(sldFaces && sldFaces.value),
    inside_cut: Number(sldCut && sldCut.value),
    model_id: selModel && selModel.value ? String(selModel.value) : "",
  };
}

function applyUiPrefs(prefs) {
  if (!prefs) {
    return;
  }
  if (togMesh) {
    togMesh.checked = !!prefs.mesh;
  }
  if (togInside) {
    togInside.checked = !!prefs.inside;
  }
  if (togOutside) {
    togOutside.checked = !!prefs.outside;
  }
  if (togWire) {
    togWire.checked = !!prefs.wireframe;
    wireOn = togWire.checked;
  }
  if (sldOpacity && prefs.opacity != null) {
    sldOpacity.value = String(prefs.opacity);
  }
  if (sldPsize && prefs.point_size != null) {
    sldPsize.value = String(prefs.point_size);
  }
  if (sldCap && prefs.draw_cap != null) {
    sldCap.value = String(prefs.draw_cap);
    if (valCap) {
      valCap.textContent = Number(sldCap.value).toLocaleString();
    }
  }
  if (sldDensity && prefs.density != null) {
    sldDensity.value = String(prefs.density);
  }
  if (sldEnvelope && prefs.envelope_n != null) {
    sldEnvelope.value = String(clampNSurface(prefs.envelope_n));
  }
  if (sldFaces && prefs.faces_n != null) {
    sldFaces.value = String(clampNFaces(prefs.faces_n));
  }
  if (sldCut && prefs.inside_cut != null) {
    sldCut.value = String(prefs.inside_cut);
  }
  updateDensityLabel();
  updateEnvelopeLabel();
  updateFacesLabel();
  updateCutLabel();
  applyInspectToMesh();
  applyInspectToPoints();
}

function scheduleSaveUiPrefs() {
  if (!prefsReady) {
    return;
  }
  window.clearTimeout(prefsTimer);
  prefsTimer = window.setTimeout(() => {
    prefsTimer = 0;
    postUiPrefs(collectUiPrefs()).catch(() => {});
  }, 200);
}

function currentInsideCut() {
  return thresholdFromCutSlider(sldCut && sldCut.value);
}

function updateCutLabel() {
  if (valCut) {
    valCut.textContent = currentInsideCut().toFixed(2);
  }
}

/**
 * Acc / inside-IoU / FN / FP at the current hard cut (viewer inspect only).
 * @param {Float32Array|Uint8Array} labels
 * @param {Uint8Array} pred
 */
function scoreAtPred(labels, pred) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  const n = labels.length;
  for (let i = 0; i < n; i += 1) {
    const gt = labels[i] > 0.5;
    const p = pred[i] > 0;
    if (gt && p) {
      tp += 1;
    } else if (!gt && p) {
      fp += 1;
    } else if (gt && !p) {
      fn += 1;
    } else {
      tn += 1;
    }
  }
  const den = tp + fp + fn;
  return {
    nFn: fn,
    nFp: fp,
    acc: n ? (tp + tn) / n : 0,
    iou: den > 0 ? tp / den : 0,
  };
}

/**
 * Re-threshold stored sigmoid probs with the Inside-cut slider (no GPU).
 * @param {{rebuild?: boolean, status?: boolean}} [opts]
 * @returns {boolean}
 */
function applyStoredProbs(opts) {
  if (!npzState || !npzState.probs) {
    return false;
  }
  const t = currentInsideCut();
  npzState.pred = predFromProbs(npzState.probs, t);
  if (isFillJob()) {
    npzState.nInside = countPredClass(npzState.pred, true);
    npzState.nOutside = countPredClass(npzState.pred, false);
  } else {
    const scored = scoreAtPred(npzState.labels, npzState.pred);
    npzState.nFn = scored.nFn;
    npzState.nFp = scored.nFp;
    npzState.cutAcc = scored.acc;
    npzState.cutIou = scored.iou;
  }
  const rebuild = !opts || opts.rebuild !== false;
  if (rebuild && (viewMode === "pred" || viewMode === "errors" || isFillJob())) {
    rebuildNpzLayers();
  }
  if (opts && opts.status) {
    setStatus(statusForCurrentCut());
  }
  return true;
}

function statusForCurrentCut() {
  const t = currentInsideCut().toFixed(2);
  if (!npzState || !npzState.pred) {
    return "Inside cut " + t + ".";
  }
  if (isFillJob()) {
    return (
      "Inside cut " +
      t +
      ": inside=" +
      Number(npzState.nInside || 0).toLocaleString() +
      " outside=" +
      Number(npzState.nOutside || 0).toLocaleString() +
      "."
    );
  }
  const acc = Number(npzState.cutAcc);
  const iou = Number(npzState.cutIou);
  return (
    "Inside cut " +
    t +
    ": acc=" +
    (Number.isFinite(acc) ? acc.toFixed(4) : "—") +
    " iou=" +
    (Number.isFinite(iou) ? iou.toFixed(4) : "—") +
    ". Flip Truth / Errors."
  );
}

/**
 * Keep helper 0.5 pred if probs are missing (old helper).
 * @param {{pred: Uint8Array, probs: Float32Array|null, metrics: object}} result
 */
function attachInferResult(result) {
  const n = npzState.n;
  if (result.probs && result.probs.length === n) {
    npzState.probs = result.probs;
    applyStoredProbs({ rebuild: false });
    return;
  }
  npzState.probs = null;
  npzState.pred = result.pred;
  if (isFillJob()) {
    npzState.nInside = Number(result.metrics.n_inside) || countPredClass(result.pred, true);
    npzState.nOutside = Number(result.metrics.n_outside) || countPredClass(result.pred, false);
  } else {
    npzState.nFn = result.metrics.n_fn;
    npzState.nFp = result.metrics.n_fp;
    npzState.cutAcc = Number(result.metrics.accuracy);
    npzState.cutIou = Number(result.metrics.inside_iou);
  }
}

function countPredClass(pred, wantInside) {
  let n = 0;
  for (let i = 0; i < pred.length; i += 1) {
    if ((pred[i] > 0.5) === wantInside) {
      n += 1;
    }
  }
  return n;
}

function applyLegendForView() {
  const errorMode = viewMode === "errors";
  if (labInside) {
    labInside.textContent = errorMode ? "Missed" : "Inside";
  }
  if (labOutside) {
    labOutside.textContent = errorMode ? "False+" : "Outside";
  }
  if (legWordIn) {
    legWordIn.textContent = errorMode ? "missed" : "inside";
  }
  if (legWordOut) {
    legWordOut.textContent = errorMode ? "false+" : "outside";
  }
  if (legSwatchIn) {
    legSwatchIn.classList.toggle("err-fn", errorMode);
  }
  if (legSwatchOut) {
    legSwatchOut.classList.toggle("err-fp", errorMode);
  }
}

function resetViewMode() {
  viewMode = "truth";
  applyLegendForView();
  syncModelPanel();
}

function setLayerVisible(name, on) {
  if (!loadedRoot) {
    return;
  }
  const obj = loadedRoot.getObjectByName(name);
  if (obj) {
    obj.visible = on;
  }
}

function meshTarget() {
  return shownMesh || (loadedRoot && loadedRoot.name === "loaded-obj" ? loadedRoot : null);
}

function syncInspectEnabled() {
  const hasNpz = !!npzState;
  const hasMesh = !!meshTarget();
  togMesh.disabled = !hasMesh;
  togInside.disabled = !hasNpz;
  togOutside.disabled = !hasNpz;
  togWire.disabled = !hasMesh;
  sldOpacity.disabled = !hasMesh;
  sldPsize.disabled = !hasNpz && !envelopeVisible;
  sldCap.disabled = !hasNpz;
  inspectPanel.hidden = !loadedRoot;
  syncModelPanel();
}

function applyInspectToMesh() {
  const mesh = meshTarget();
  if (!mesh) {
    return;
  }
  forEachMesh(mesh, (m) => {
    m.visible = togMesh.checked;
  });
  const opacity = Number(sldOpacity.value) / 100;
  applyMeshOpacity(mesh, opacity);
  wireOn = !!(togWire && togWire.checked);
  ensureWireOverlays(mesh, wireOn);
  valOpacity.textContent = Math.round(opacity * 100) + "%";
}

function currentPointSize() {
  return basePointSize * (Number(sldPsize.value) / 100);
}

function applyInspectToPoints() {
  if (!loadedRoot) {
    return;
  }
  applyPointSize(loadedRoot, currentPointSize());
  setLayerVisible("npz-inside", togInside.checked);
  setLayerVisible("npz-outside", togOutside.checked);
  valPsize.textContent = (Number(sldPsize.value) / 100).toFixed(1) + "×";
}

function removeEnvelopeLayer() {
  if (!loadedRoot) {
    envelopeVisible = false;
    return;
  }
  const old = loadedRoot.getObjectByName("envelope-points");
  if (old) {
    loadedRoot.remove(old);
    disposeObject3d(old);
  }
}

function attachEnvelopeLayer(xyz) {
  if (!loadedRoot) {
    return;
  }
  removeEnvelopeLayer();
  loadedRoot.add(makeEnvelopeLayer(xyz, currentPointSize()));
  envelopeVisible = true;
  applyInspectToPoints();
}

function removeFaceTokenLayer() {
  if (!loadedRoot) {
    facesVisible = false;
    return;
  }
  const old = loadedRoot.getObjectByName("face-tokens");
  if (old) {
    loadedRoot.remove(old);
    disposeObject3d(old);
  }
}

function attachFaceTokenLayer(tokens, n, tick) {
  if (!loadedRoot) {
    return;
  }
  removeFaceTokenLayer();
  loadedRoot.add(makeFaceTokenGroup(tokens, n, tick));
  facesVisible = true;
}

function rebuildNpzLayers() {
  if (!npzState || !loadedRoot) {
    return;
  }
  const cap = Number(sldCap.value);
  valCap.textContent = cap.toLocaleString();
  ["npz-inside", "npz-outside"].forEach((name) => {
    const old = loadedRoot.getObjectByName(name);
    if (old) {
      loadedRoot.remove(old);
      disposeObject3d(old);
    }
  });
  let layers;
  if (viewMode === "errors" && npzState.pred) {
    layers = buildErrorLayers(
      npzState.points,
      npzState.labels,
      npzState.pred,
      cap,
      currentPointSize()
    );
  } else {
    const labels =
      viewMode === "pred" && npzState.pred ? npzState.pred : npzState.labels;
    layers = buildNpzLayers(npzState.points, labels, cap, currentPointSize());
  }
  if (layers.inside) {
    loadedRoot.add(layers.inside);
  }
  if (layers.outside) {
    loadedRoot.add(layers.outside);
  }
  npzState.nDraw = layers.nDraw;
  showNpzPanel(npzState);
  applyInspectToPoints();
  syncInspectEnabled();
}

/**
 * @param {{n: number, nDraw: number, nInside: number, nOutside: number, meshPath: string}} info
 */
function showNpzPanel(info) {
  npzPanel.hidden = false;
  const drawn =
    info.nDraw < info.n
      ? " · drawing " + info.nDraw.toLocaleString()
      : "";
  // Total sits above the color legend; class counts sit next to each swatch.
  npzCounts.textContent = "Total points: " + info.n.toLocaleString() + drawn;
  if (viewMode === "pred" && info.pred) {
    npzNInside.textContent = countPredClass(info.pred, true).toLocaleString();
    npzNOutside.textContent = countPredClass(info.pred, false).toLocaleString();
  } else if (viewMode === "errors" && info.pred) {
    npzNInside.textContent = Number(info.nFn || 0).toLocaleString();
    npzNOutside.textContent = Number(info.nFp || 0).toLocaleString();
  } else {
    npzNInside.textContent = info.nInside.toLocaleString();
    npzNOutside.textContent = info.nOutside.toLocaleString();
  }
  if (info.source === "fill") {
    npzPath.textContent =
      "fill · spacing " +
      (info.usedSpacing != null ? Number(info.usedSpacing).toFixed(2) : "—");
    npzPath.title = objFileName || "";
  } else if (info.meshPath) {
    npzPath.textContent = info.meshPath;
    npzPath.title = info.meshPath;
    currentMeshPath = info.meshPath;
  } else {
    npzPath.textContent = "(no mesh_path in this NPZ)";
    npzPath.title = "";
    currentMeshPath = "";
  }
}

function disposeHelper(helper) {
  if (!helper) {
    return;
  }
  scene.remove(helper);
  if (helper.geometry) {
    helper.geometry.dispose();
  }
  const mat = helper.material;
  if (Array.isArray(mat)) {
    mat.forEach((m) => m.dispose());
  } else if (mat) {
    mat.dispose();
  }
}

/** Floor grid and axes scaled to the current content. */
function setHelpers(radius, yMin, center) {
  disposeHelper(gridHelper);
  disposeHelper(axesHelper);
  const span = Math.max(radius * 4, 4);
  gridHelper = new THREE.GridHelper(span, 20, 0xaabbcc, 0x556677);
  gridHelper.position.y = yMin;
  if (center) {
    gridHelper.position.x = center.x;
    gridHelper.position.z = center.z;
  }
  axesHelper = new THREE.AxesHelper(Math.max(radius * 0.45, 1));
  if (center) {
    axesHelper.position.copy(center);
    axesHelper.position.y = yMin;
  }
  scene.add(gridHelper);
  scene.add(axesHelper);
}

function replaceContent(group) {
  if (loadedRoot) {
    scene.remove(loadedRoot);
    disposeObject3d(loadedRoot);
  }
  loadedRoot = group;
  shownMesh = null;
  envelopeVisible = false;
  facesVisible = false;
  scene.add(loadedRoot);
  const framed = fitCameraToObject(camera, controls, loadedRoot);
  setHelpers(framed.radius, framed.yMin, framed.center);
  keyLight.position.copy(camera.position);
}

function attachShownMesh(meshGroup) {
  if (!loadedRoot) {
    throw new Error("Load an NPZ before attaching a mesh");
  }
  if (shownMesh) {
    loadedRoot.remove(shownMesh);
    disposeObject3d(shownMesh);
  }
  shownMesh = meshGroup;
  loadedRoot.add(shownMesh);
  const framed = fitCameraToObject(camera, controls, loadedRoot);
  setHelpers(framed.radius, framed.yMin, framed.center);
  keyLight.position.copy(camera.position);
  applyInspectToMesh();
  syncInspectEnabled();
}

function resetEmptyView() {
  camera.near = 0.1;
  camera.far = 1000;
  camera.position.set(10, 8, 10);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.minDistance = 2;
  controls.maxDistance = 50;
  controls.update();
  keyLight.position.set(5, 8, 5);
  setHelpers(5, 0, null);
}

function clearView() {
  if (!scene) {
    return;
  }
  if (loadedRoot) {
    scene.remove(loadedRoot);
    disposeObject3d(loadedRoot);
    loadedRoot = null;
  }
  shownMesh = null;
  npzState = null;
  objText = "";
  objFileName = "";
  inferBusy = false;
  fillBusy = false;
  envelopeBusy = false;
  envelopeVisible = false;
  facesBusy = false;
  facesVisible = false;
  cancelPendingFill();
  resetViewMode();
  hideNpzPanel();
  hideInspectPanel();
  resetEmptyView();
  dropHint.textContent = "Drop an OBJ or NPZ onto the view";
  setStatus("View cleared. Open or drop an OBJ or NPZ.");
}

function rememberFile(file, handle) {
  return persistRecentFile(file, handle || null);
}

/**
 * @param {File} file
 * @param {FileSystemFileHandle|null} [handle]
 */
async function openUserFile(file, handle) {
  if (!file) {
    setStatus("No file selected.");
    return;
  }
  cancelPendingFill();
  if (isObjFile(file)) {
    setStatus("Loading “" + file.name + "”…");
    try {
      const text = await file.text();
      const group = parseObjText(text);
      npzState = null;
      objText = text;
      objFileName = file.name;
      hideNpzPanel();
      replaceContent(group);
      shownMesh = group;
      applyInspectToMesh();
      syncInspectEnabled();
      resetViewMode();
      updateDensityLabel();
      await rememberFile(file, handle);
      dropHint.textContent = "Drop an OBJ or NPZ to replace";
      setStatus("Loaded “" + file.name + "”. Fill points, then Run model.");
    } catch (err) {
      setStatus("Could not load “" + file.name + "”: " + err);
    }
    return;
  }
  if (isNpzFile(file)) {
    setStatus("Loading “" + file.name + "”…");
    try {
      const loaded = await parseNpzFile(file);
      npzState = loaded;
      npzState.fileName = file.name;
      npzState.source = "npz";
      npzState.pred = null;
      npzState.probs = null;
      objText = "";
      objFileName = "";
      viewMode = "truth";
      applyLegendForView();
      const group = new THREE.Group();
      group.name = "loaded-npz";
      if (loaded.layers.inside) {
        group.add(loaded.layers.inside);
      }
      if (loaded.layers.outside) {
        group.add(loaded.layers.outside);
      }
      replaceContent(group);
      basePointSize = Math.max(loaded.layers.radius / 90, 0.02);
      showNpzPanel(loaded);
      applyInspectToPoints();
      syncInspectEnabled();
      await rememberFile(file, handle);
      dropHint.textContent = "Drop an OBJ or NPZ to replace";
      let meshOk = true;
      if (loaded.meshPath) {
        meshOk = await showLinkedObject(true);
      }
      if (!meshOk) {
        return;
      }
      const sampleNote =
        loaded.nDraw < loaded.n ? " Sampled for display." : "";
      const meshNote =
        loaded.meshPath && shownMesh ? " Mesh from NPZ path." : "";
      setStatus(
        "Loaded “" + file.name + "”." + sampleNote + meshNote + " Left-drag to orbit."
      );
    } catch (err) {
      setStatus("Could not load “" + file.name + "”: " + err);
    }
    return;
  }
  setStatus("Not an OBJ or NPZ: “" + file.name + "”.");
}

async function attachObjText(text) {
  objText = String(text || "");
  const meshGroup = parseObjText(objText, { opacity: 0.42, name: "shown-obj" });
  attachShownMesh(meshGroup);
  syncFillPanel();
}

function promptObjFallback(reason) {
  const want = meshBasename(currentMeshPath);
  setStatus(reason + (want ? " Pick “" + want + "”." : " Pick the matching OBJ."));
  if (objFallbackInput) {
    objFallbackInput.click();
  }
}

/**
 * Fetch the OBJ stored in the NPZ mesh_path. Returns false if the helper
 * failed (fallback picker is already prompted).
 * @param {boolean} [quiet] skip the success status (caller sets its own)
 */
async function showLinkedObject(quiet) {
  if (!currentMeshPath) {
    if (!quiet) {
      setStatus("This NPZ has no mesh_path.");
    }
    return false;
  }
  setStatus("Loading mesh from “" + currentMeshPath + "”…");
  try {
    const text = await fetchMeshObjText(currentMeshPath);
    await attachObjText(text);
    if (!quiet) {
      setStatus("Mesh: “" + meshBasename(currentMeshPath) + "”. Left-drag to orbit.");
    }
    return true;
  } catch (err) {
    promptObjFallback("Helper could not load the mesh (" + err + ").");
    return false;
  }
}

function updateDensityLabel() {
  if (!valDensity || !sldDensity) {
    return;
  }
  valDensity.textContent = spacingFromSlider(sldDensity.value).toFixed(2);
}

function updateFacesLabel() {
  if (!valFaces || !sldFaces) {
    return;
  }
  valFaces.textContent = String(clampNFaces(sldFaces.value));
}

function scheduleFacesFromSlider() {
  updateFacesLabel();
  if (!facesVisible || !objText) {
    return;
  }
  window.clearTimeout(facesTimer);
  facesTimer = window.setTimeout(() => {
    facesTimer = 0;
    runFaces({ refresh: true });
  }, 180);
}

async function runFaces(opts) {
  const refresh = !!(opts && opts.refresh);
  if (!objText) {
    return;
  }
  if (!refresh && facesVisible) {
    removeFaceTokenLayer();
    facesVisible = false;
    syncFillPanel();
    setStatus("Faces hidden.");
    return;
  }
  if (facesBusy || inferBusy) {
    return;
  }
  facesBusy = true;
  syncFillPanel();
  const nFaces = clampNFaces(sldFaces && sldFaces.value);
  setStatus("Building " + nFaces.toLocaleString() + " face tokens…");
  try {
    const overlay = await facesObjOnHelper({
      objText: objText,
      nFaces: nFaces,
    });
    attachFaceTokenLayer(overlay.tokens, overlay.n, overlay.tick);
    const tiled =
      overlay.nUnique < overlay.n
        ? " (" + overlay.nUnique.toLocaleString() + " unique, tiled)"
        : "";
    setStatus(
      "Faces “" +
        (objFileName || "mesh") +
        "”: " +
        overlay.n.toLocaleString() +
        " tokens" +
        tiled +
        " of " +
        overlay.nMesh.toLocaleString() +
        " mesh triangles."
    );
    scheduleSaveUiPrefs();
  } catch (err) {
    setStatus("Faces failed: " + err);
  }
  facesBusy = false;
  syncFillPanel();
}

function updateEnvelopeLabel() {
  if (!valEnvelope || !sldEnvelope) {
    return;
  }
  valEnvelope.textContent = String(clampNSurface(sldEnvelope.value));
}

function scheduleEnvelopeFromSlider() {
  updateEnvelopeLabel();
  if (!envelopeVisible || !objText) {
    return;
  }
  window.clearTimeout(envelopeTimer);
  envelopeTimer = window.setTimeout(() => {
    envelopeTimer = 0;
    runEnvelope({ refresh: true });
  }, 180);
}

async function runEnvelope(opts) {
  const refresh = !!(opts && opts.refresh);
  if (!objText) {
    return;
  }
  if (!refresh && envelopeVisible) {
    removeEnvelopeLayer();
    envelopeVisible = false;
    syncFillPanel();
    setStatus("Envelope hidden.");
    syncInspectEnabled();
    return;
  }
  if (envelopeBusy || inferBusy) {
    return;
  }
  envelopeBusy = true;
  syncFillPanel();
  const nSurface = clampNSurface(sldEnvelope && sldEnvelope.value);
  setStatus("Sampling " + nSurface.toLocaleString() + " envelope points…");
  try {
    const sampled = await envelopeObjOnHelper({
      objText: objText,
      nSurface: nSurface,
    });
    attachEnvelopeLayer(sampled.points);
    syncInspectEnabled();
    setStatus(
      "Envelope “" +
        (objFileName || "mesh") +
        "”: " +
        sampled.n.toLocaleString() +
        " purple surface points."
    );
    scheduleSaveUiPrefs();
  } catch (err) {
    setStatus("Envelope failed: " + err);
  }
  envelopeBusy = false;
  syncFillPanel();
}

/** Debounce live density so drag does not POST on every tick. */
function scheduleFillFromDensity() {
  updateDensityLabel();
  if (!objText || isNpzJob()) {
    return;
  }
  fillPending = true;
  window.clearTimeout(fillTimer);
  fillTimer = window.setTimeout(() => {
    fillTimer = 0;
    runFill();
  }, 180);
}

async function runFill() {
  if (!objText || isNpzJob()) {
    return;
  }
  if (fillBusy || inferBusy) {
    fillPending = true;
    return;
  }
  fillPending = false;
  fillBusy = true;
  syncFillPanel();
  const spacing = spacingFromSlider(sldDensity.value);
  setStatus("Filling box at spacing " + spacing.toFixed(2) + "…");
  try {
    const filled = await fillObjOnHelper({ objText: objText, spacing: spacing });
    const n = filled.n;
    const labels = new Float32Array(n);
    npzState = {
      source: "fill",
      points: filled.points,
      labels: labels,
      pred: null,
      probs: null,
      n: n,
      nInside: 0,
      nOutside: n,
      nDraw: n,
      meshPath: "",
      fileName: objFileName,
      usedSpacing: filled.usedSpacing,
    };
    viewMode = "truth";
    applyLegendForView();
    if (togInside) {
      togInside.checked = true;
    }
    if (togOutside) {
      togOutside.checked = true;
    }
    rebuildNpzLayers();
    let maxAbs = 1;
    for (let i = 0; i < filled.points.length; i += 1) {
      const a = Math.abs(filled.points[i]);
      if (a > maxAbs) {
        maxAbs = a;
      }
    }
    basePointSize = Math.max(maxAbs / 90, 0.02);
    applyInspectToPoints();
    setStatus(
      "Filled " +
        n.toLocaleString() +
        " points (spacing " +
        Number(filled.usedSpacing).toFixed(2) +
        "). Run model to classify."
    );
    scheduleSaveUiPrefs();
  } catch (err) {
    setStatus("Fill failed: " + err);
  }
  fillBusy = false;
  if (fillPending) {
    runFill();
    return;
  }
  syncInspectEnabled();
}

async function fillModelSelect(preferredId) {
  if (!selModel) {
    return;
  }
  const previous = (preferredId && String(preferredId)) || selModel.value;
  try {
    const models = await fetchModelList();
    selModel.innerHTML = "";
    if (!models.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No models/<run>/best.pt";
      selModel.appendChild(opt);
      syncModelPanel();
      return;
    }
    models.forEach((row) => {
      const opt = document.createElement("option");
      opt.value = row.id;
      const enc = row.shape_encoder ? String(row.shape_encoder) : "";
      let label = row.id;
      if (enc === "mesh") {
        label += " (faces)";
      } else if (enc === "surface") {
        label += " (envelope)";
      }
      opt.textContent = label;
      selModel.appendChild(opt);
    });
    if (previous && models.some((row) => row.id === previous)) {
      selModel.value = previous;
    }
  } catch (err) {
    selModel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Helper has no model list";
    selModel.appendChild(opt);
  }
  syncModelPanel();
}

function setViewMode(mode) {
  if (!npzState) {
    return;
  }
  if (isFillJob() && mode !== "pred") {
    return;
  }
  if (mode !== "truth" && !npzState.pred) {
    return;
  }
  viewMode = mode;
  applyLegendForView();
  rebuildNpzLayers();
  syncModelPanel();
}

async function runInfer() {
  if (!npzState || inferBusy) {
    return;
  }
  const checkpoint = selModel && selModel.value;
  if (!checkpoint) {
    setStatus("Select a model under models/<run>/best.pt.");
    return;
  }
  inferBusy = true;
  syncModelPanel();
  setStatus("Running “" + checkpoint + "” on " + npzState.n.toLocaleString() + " points…");
  try {
    if (isFillJob()) {
      if (!objText) {
        throw new Error("OBJ text is missing; open the object again.");
      }
      const result = await inferObjOnHelper({
        checkpoint,
        name: objFileName || npzState.fileName || "",
        objText: objText,
        points: npzState.points,
      });
      attachInferResult(result);
      viewMode = "pred";
      applyLegendForView();
      rebuildNpzLayers();
      setStatus("Prediction from “" + checkpoint + "”. " + statusForCurrentCut());
    } else {
      const result = await inferNpzOnHelper({
        checkpoint,
        name: npzState.fileName || "",
        meshPath: npzState.meshPath || currentMeshPath || "",
        points: npzState.points,
        labels: npzState.labels,
      });
      attachInferResult(result);
      viewMode = "pred";
      applyLegendForView();
      rebuildNpzLayers();
      setStatus("Prediction from “" + checkpoint + "”. " + statusForCurrentCut());
    }
  } catch (err) {
    setStatus("Run failed: " + err);
  }
  inferBusy = false;
  syncModelPanel();
  if (fillPending) {
    runFill();
  }
}

async function pickWithOpenButton() {
  if (typeof window.showOpenFilePicker === "function") {
    try {
      const handles = await window.showOpenFilePicker({
        multiple: false,
        types: [
          {
            description: "OBJ or NPZ",
            accept: { "application/octet-stream": [".obj", ".npz"] },
          },
        ],
      });
      const handle = handles[0];
      const file = await handle.getFile();
      await openUserFile(file, handle);
      return;
    } catch (err) {
      if (err && err.name === "AbortError") {
        return;
      }
    }
  }
  fileInput.click();
}

try {
  bindRecentsHover(openWrap, recentsList, {
    onSelect: async (name) => {
      const cached = await resolveRecentFile(name);
      if (cached) {
        await openUserFile(cached);
        return;
      }
      setStatus("“" + name + "” is not in the cache yet. Use Open once; after that the menu opens it directly.");
    },
    onClear: () => {
      clearView();
    },
  });
} catch (err) {
  setStatus("UI init failed: " + err);
}

async function restoreUiPrefs() {
  try {
    const prefs = await fetchUiPrefs();
    applyUiPrefs(prefs);
    await fillModelSelect(prefs.model_id);
  } catch (err) {
    await fillModelSelect();
  }
  prefsReady = true;
}

restoreUiPrefs();

btnOpen.addEventListener("click", () => {
  pickWithOpenButton();
});

if (selModel) {
  selModel.addEventListener("change", () => {
    syncModelPanel();
    scheduleSaveUiPrefs();
  });
}
if (btnRun) {
  btnRun.addEventListener("click", () => {
    runInfer();
  });
}
if (btnViewTruth) {
  btnViewTruth.addEventListener("click", () => {
    setViewMode("truth");
  });
}
if (btnViewPred) {
  btnViewPred.addEventListener("click", () => {
    setViewMode("pred");
  });
}
if (btnViewErrors) {
  btnViewErrors.addEventListener("click", () => {
    setViewMode("errors");
  });
}

if (btnFill) {
  btnFill.addEventListener("click", () => {
    window.clearTimeout(fillTimer);
    fillTimer = 0;
    runFill();
  });
}
if (btnEnvelope) {
  btnEnvelope.addEventListener("click", () => {
    window.clearTimeout(envelopeTimer);
    envelopeTimer = 0;
    runEnvelope();
  });
}
if (sldEnvelope) {
  sldEnvelope.addEventListener("input", () => {
    scheduleEnvelopeFromSlider();
    scheduleSaveUiPrefs();
  });
  updateEnvelopeLabel();
}
if (btnFaces) {
  btnFaces.addEventListener("click", () => {
    window.clearTimeout(facesTimer);
    facesTimer = 0;
    runFaces();
  });
}
if (sldFaces) {
  sldFaces.addEventListener("input", () => {
    scheduleFacesFromSlider();
    scheduleSaveUiPrefs();
  });
  updateFacesLabel();
}
if (sldDensity) {
  sldDensity.addEventListener("input", () => {
    scheduleFillFromDensity();
    scheduleSaveUiPrefs();
  });
  updateDensityLabel();
}
if (sldCut) {
  sldCut.addEventListener("input", () => {
    updateCutLabel();
    applyStoredProbs({ status: true });
    scheduleSaveUiPrefs();
  });
  updateCutLabel();
}

togMesh.addEventListener("change", () => {
  applyInspectToMesh();
  scheduleSaveUiPrefs();
});
togInside.addEventListener("change", () => {
  applyInspectToPoints();
  scheduleSaveUiPrefs();
});
togOutside.addEventListener("change", () => {
  applyInspectToPoints();
  scheduleSaveUiPrefs();
});
togWire.addEventListener("change", () => {
  applyInspectToMesh();
  scheduleSaveUiPrefs();
});
sldOpacity.addEventListener("input", () => {
  applyInspectToMesh();
  scheduleSaveUiPrefs();
});
sldPsize.addEventListener("input", () => {
  applyInspectToPoints();
  scheduleSaveUiPrefs();
});
sldCap.addEventListener("change", () => {
  rebuildNpzLayers();
  scheduleSaveUiPrefs();
});

if (objFallbackInput) {
  objFallbackInput.addEventListener("change", async () => {
    const file = objFallbackInput.files && objFallbackInput.files[0];
    objFallbackInput.value = "";
    if (!file) {
      return;
    }
    const want = meshBasename(currentMeshPath);
    if (want && file.name !== want) {
      setStatus("Expected “" + want + "”, got “" + file.name + "”. Loading it anyway.");
    }
    try {
      await attachObjText(await file.text());
      setStatus("Mesh (picked file): “" + file.name + "”.");
    } catch (err) {
      setStatus("Could not load picked OBJ: " + err);
    }
  });
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!file) {
    return;
  }
  openUserFile(file);
});

container.addEventListener("dragover", (event) => {
  event.preventDefault();
  container.classList.add("drag-over");
});

container.addEventListener("dragleave", (event) => {
  if (event.target === container) {
    container.classList.remove("drag-over");
  }
});

container.addEventListener("drop", (event) => {
  event.preventDefault();
  container.classList.remove("drag-over");
  const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
  if (!file) {
    setStatus("Drop an .obj or .npz file onto the view.");
    return;
  }
  openUserFile(file);
});

try {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2a32);

  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(10, 8, 10);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;
  controls.minDistance = 2;
  controls.maxDistance = 50;
  controls.target.set(0, 0, 0);

  setHelpers(5, 0, null);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
  keyLight.position.set(5, 8, 5);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x8899aa, 0.35);
  fillLight.position.set(-6, 3, -4);
  scene.add(fillLight);

  function onResize() {
    const w = Math.max(container.clientWidth, 16);
    const h = Math.max(container.clientHeight, 16);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  window.addEventListener("resize", onResize);
  onResize();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
  setStatus("Open or drop an OBJ or NPZ. Left-drag to orbit.");
} catch (err) {
  setStatus("3D init failed: " + err);
}

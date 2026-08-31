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
import { fetchModelList, inferNpzOnHelper } from "./model_panel.js";
import { fillObjOnHelper, inferObjOnHelper, spacingFromSlider } from "./obj_infer.js";
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
  const jobB = !!objText && !isNpzJob();
  if (fillPanel) {
    fillPanel.hidden = !jobB;
  }
  if (btnFill) {
    btnFill.disabled = !jobB || fillBusy || inferBusy;
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
  updateDensityLabel();
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
  sldPsize.disabled = !hasNpz;
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
  const meshGroup = parseObjText(text, { opacity: 0.42, name: "shown-obj" });
  attachShownMesh(meshGroup);
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
      opt.textContent = row.id;
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
      npzState.pred = result.pred;
      npzState.nInside = Number(result.metrics.n_inside) || countPredClass(result.pred, true);
      npzState.nOutside = Number(result.metrics.n_outside) || countPredClass(result.pred, false);
      viewMode = "pred";
      applyLegendForView();
      rebuildNpzLayers();
      setStatus(
        "Prediction from “" +
          checkpoint +
          "”. inside=" +
          npzState.nInside.toLocaleString() +
          " outside=" +
          npzState.nOutside.toLocaleString() +
          "."
      );
    } else {
      const result = await inferNpzOnHelper({
        checkpoint,
        name: npzState.fileName || "",
        meshPath: npzState.meshPath || currentMeshPath || "",
        points: npzState.points,
        labels: npzState.labels,
      });
      npzState.pred = result.pred;
      npzState.nFn = result.metrics.n_fn;
      npzState.nFp = result.metrics.n_fp;
      viewMode = "pred";
      applyLegendForView();
      rebuildNpzLayers();
      const acc = Number(result.metrics.accuracy);
      const iou = Number(result.metrics.inside_iou);
      setStatus(
        "Prediction from “" +
          checkpoint +
          "”. acc=" +
          acc.toFixed(4) +
          " iou=" +
          iou.toFixed(4) +
          ". Flip Truth / Errors."
      );
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
if (sldDensity) {
  sldDensity.addEventListener("input", () => {
    scheduleFillFromDensity();
    scheduleSaveUiPrefs();
  });
  updateDensityLabel();
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

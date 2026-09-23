/**
 * Persistent occupancy orbit (Y-up). Gradio Model3D remounts the whole
 * WebGL context on every new GLB — that is the gray flash + camera snap.
 * This script owns one Engine for the page lifetime and only replaces meshes.
 *
 * Gradio 6 strips <script> inside gr.HTML. This file is passed to launch(js=)
 * so it actually runs. Python writes a tiny #sn-cmd span (path + reset).
 * The canvas host HTML is never updated, so the Engine is not remounted.
 */
(async function () {
  /* Gradio can re-run launch(js=) when the cmd span updates. A second
     Engine would start at 45°/70° and snapshotCam() would wipe the orbit. */
  if (window.__snOrbit && window.__snOrbit.alive && window.__snOrbit.alive()) {
    return;
  }
  const BG = [42 / 255, 42 / 255, 50 / 255, 1];
  const START_ALPHA = 45;
  const START_BETA = 70;
  /* Pull back so samples (dog / horse / torus) fit in frame. Load still
     does not move the camera; Reset view returns here. */
  const START_RADIUS = 60;
  const GRID_HALF = 16;
  const GRID_DIVS = 20;
  const AXIS_LEN = 4;
  const BABYLON_SRCS = [
    [
      "https://cdn.babylonjs.com/babylon.js",
      "https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js",
    ],
    [
      "https://cdn.jsdelivr.net/npm/babylonjs@7/babylon.js",
      "https://cdn.jsdelivr.net/npm/babylonjs@7/babylonjs.loaders.min.js",
    ],
  ];

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      const el = document.createElement("script");
      el.src = src;
      el.async = true;
      el.onload = function () {
        resolve();
      };
      el.onerror = function () {
        reject(new Error(src));
      };
      document.head.appendChild(el);
    });
  }

  async function loadBabylon() {
    if (window.BABYLON && window.BABYLON.SceneLoader) {
      return;
    }
    let last = null;
    for (let i = 0; i < BABYLON_SRCS.length; i += 1) {
      try {
        await loadScript(BABYLON_SRCS[i][0]);
        await loadScript(BABYLON_SRCS[i][1]);
        if (window.BABYLON && window.BABYLON.SceneLoader) {
          return;
        }
      } catch (err) {
        last = err;
      }
    }
    throw last || new Error("Babylon.js failed to load");
  }

  function waitFor(id) {
    return new Promise(function (resolve) {
      function tick() {
        const el = document.getElementById(id);
        if (el) {
          resolve(el);
          return;
        }
        window.setTimeout(tick, 50);
      }
      tick();
    });
  }

  function cmdSpan() {
    const direct = document.getElementById("sn-cmd");
    if (direct) {
      return direct;
    }
    const wrap = document.getElementById("sn-cmd-wrap");
    return wrap ? wrap.querySelector("span") : null;
  }

  const DEFAULT_PSIZE = 8;

  function cmdParts() {
    const el = cmdSpan();
    if (!el) {
      return ["", 0, DEFAULT_PSIZE, 0, 0];
    }
    const bits = String(el.textContent || "").trim().split("|");
    const path = bits[0] || "";
    const reset = bits.length > 1 ? Number(bits[1]) : 0;
    const psize = bits.length > 2 ? Number(bits[2]) : DEFAULT_PSIZE;
    const wire = bits.length > 3 ? Number(bits[3]) : 0;
    const forceDefault = bits.length > 4 ? Number(bits[4]) : 0;
    return [path, reset, psize, wire, forceDefault];
  }

  function fieldPath() {
    return cmdParts()[0];
  }

  function fieldReset() {
    return cmdParts()[1] || 0;
  }

  function fieldPsize() {
    const n = cmdParts()[2];
    if (!Number.isFinite(n) || n < 1) {
      return DEFAULT_PSIZE;
    }
    return Math.max(1, Math.min(24, n));
  }

  function fieldWire() {
    return cmdParts()[3] ? 1 : 0;
  }

  function fieldForceDefault() {
    return cmdParts()[4] ? 1 : 0;
  }

  function fileUrl(absPath) {
    const posix = String(absPath).replace(/\\/g, "/");
    return "/gradio_api/file=" + posix;
  }

  function pickObj(fileList) {
    if (!fileList) {
      return null;
    }
    for (let i = 0; i < fileList.length; i += 1) {
      if (fileList[i] && /\.obj$/i.test(fileList[i].name || "")) {
        return fileList[i];
      }
    }
    return null;
  }

  function objFileInput() {
    const root = document.getElementById("sn-obj-file");
    return root ? root.querySelector("input[type=file]") : null;
  }

  function sendObjToGradio(file) {
    const input = objFileInput();
    if (!input || !file) {
      return;
    }
    const dt = new DataTransfer();
    dt.items.add(file);
    input.value = "";
    input.files = dt.files;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function bindLoadBtn(wrap) {
    if (!wrap || wrap.dataset.snLoadBound === "1") {
      return;
    }
    wrap.dataset.snLoadBound = "1";
    wrap.addEventListener(
      "click",
      function () {
        const input = objFileInput();
        if (input) {
          input.click();
        }
      },
      true
    );
  }

  function bindViewDrop(host) {
    if (!host || host.dataset.snDropBound === "1") {
      return;
    }
    host.dataset.snDropBound = "1";
    host.addEventListener(
      "dragover",
      function (e) {
        if (!e.dataTransfer) {
          return;
        }
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        host.classList.add("sn-drop-over");
      },
      true
    );
    host.addEventListener(
      "dragleave",
      function (e) {
        if (!host.contains(e.relatedTarget)) {
          host.classList.remove("sn-drop-over");
        }
      },
      true
    );
    host.addEventListener(
      "drop",
      function (e) {
        e.preventDefault();
        host.classList.remove("sn-drop-over");
        sendObjToGradio(pickObj(e.dataTransfer && e.dataTransfer.files));
      },
      true
    );
  }

  waitFor("sn-orbit-host").then(bindViewDrop);
  waitFor("sn-load-obj").then(bindLoadBtn);
  await loadBabylon();
  if (window.BABYLON && window.BABYLON.SceneLoader) {
    window.BABYLON.SceneLoader.OnPluginActivatedObservable.add(function (plugin) {
      if (plugin && String(plugin.name || "").toLowerCase() === "gltf") {
        plugin.loadCameras = false;
      }
    });
  }
  let canvas = document.getElementById("sn-orbit");
  if (!canvas) {
    /* Gradio may strip <canvas> from HTML; the host div is enough. */
    let host = document.getElementById("sn-orbit-host");
    if (!host) {
      const wrap = await waitFor("sn-orbit-wrap");
      host = document.createElement("div");
      host.id = "sn-orbit-host";
      host.style.cssText =
        "width:100%;height:640px;background:#2a2a32;border-radius:8px;overflow:hidden;";
      wrap.appendChild(host);
    }
    bindViewDrop(host);
    canvas = document.createElement("canvas");
    canvas.id = "sn-orbit";
    canvas.style.cssText = "width:100%;height:100%;display:block;";
    host.appendChild(canvas);
  }
  if (canvas && canvas.parentElement && !document.getElementById("sn-drop-hint")) {
    const hint = document.createElement("div");
    hint.id = "sn-drop-hint";
    hint.textContent = "Drop an OBJ file here";
    canvas.parentElement.appendChild(hint);
  }
  const BABYLON = window.BABYLON;

  const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true }, true);
  const scene = new BABYLON.Scene(engine);
  scene.useRightHandedSystem = true;
  scene.clearColor = new BABYLON.Color4(BG[0], BG[1], BG[2], BG[3]);
  scene.ambientColor = new BABYLON.Color3(0.35, 0.35, 0.38);

  const camera = new BABYLON.ArcRotateCamera(
    "cam",
    BABYLON.Tools.ToRadians(START_ALPHA),
    BABYLON.Tools.ToRadians(START_BETA),
    START_RADIUS,
    BABYLON.Vector3.Zero(),
    scene
  );
  camera.lowerRadiusLimit = 0.05;
  camera.minZ = 0.01;
  camera.wheelPrecision = 40;
  camera.attachControl(canvas, true);

  const light = new BABYLON.HemisphericLight("hemi", new BABYLON.Vector3(0.2, 1, 0.15), scene);
  light.intensity = 0.95;

  /* Fixed world floor. Not part of the GLB — a per-mesh grid is what
     made the camera look like it jumped when a new OBJ loaded. */
  (function addWorldHelpers() {
    const half = GRID_HALF;
    const n = GRID_DIVS;
    const mid = n / 2;
    const cell = new BABYLON.Color3(85 / 255, 102 / 255, 119 / 255);
    const center = new BABYLON.Color3(170 / 255, 187 / 255, 204 / 255);
    for (let i = 0; i <= n; i += 1) {
      const t = -half + (2 * half * i) / n;
      const col = i === mid ? center : cell;
      const gx = BABYLON.MeshBuilder.CreateLines(
        "sn_gx_" + i,
        { points: [new BABYLON.Vector3(t, 0, -half), new BABYLON.Vector3(t, 0, half)] },
        scene
      );
      gx.color = col;
      const gz = BABYLON.MeshBuilder.CreateLines(
        "sn_gz_" + i,
        { points: [new BABYLON.Vector3(-half, 0, t), new BABYLON.Vector3(half, 0, t)] },
        scene
      );
      gz.color = col;
    }
    const axis = [
      [new BABYLON.Vector3(AXIS_LEN, 0, 0), new BABYLON.Color3(1, 0, 0)],
      [new BABYLON.Vector3(0, AXIS_LEN, 0), new BABYLON.Color3(0, 1, 0)],
      [new BABYLON.Vector3(0, 0, AXIS_LEN), new BABYLON.Color3(0, 0, 1)],
    ];
    for (let i = 0; i < axis.length; i += 1) {
      const line = BABYLON.MeshBuilder.CreateLines(
        "sn_ax_" + i,
        { points: [BABYLON.Vector3.Zero(), axis[i][0]] },
        scene
      );
      line.color = axis[i][1];
    }
  })();

  let imported = [];
  let lastPath = "";
  let lastReset = null;
  let lastPsize = null;
  let lastWire = null;
  let loadGen = 0;
  let loadBusy = false;

  function applyCam(shot) {
    if (!shot) {
      return;
    }
    if (camera.inertialAlphaOffset !== undefined) {
      camera.inertialAlphaOffset = 0;
      camera.inertialBetaOffset = 0;
      camera.inertialRadiusOffset = 0;
    }
    if (camera.inertialPanningX !== undefined) {
      camera.inertialPanningX = 0;
      camera.inertialPanningY = 0;
    }
    camera.alpha = shot.a;
    camera.beta = shot.b;
    camera.radius = shot.r;
    camera.setTarget(new BABYLON.Vector3(shot.tx, shot.ty, shot.tz));
    scene.activeCamera = camera;
  }

  function applyDefaultCam() {
    applyCam({
      a: BABYLON.Tools.ToRadians(START_ALPHA),
      b: BABYLON.Tools.ToRadians(START_BETA),
      r: START_RADIUS,
      tx: 0,
      ty: 0,
      tz: 0,
    });
  }

  function frameDefault() {
    applyDefaultCam();
  }

  function isOccPoints(mesh) {
    /* Grid/axes are GL_LINES with idx.length === vert count. Treating those
       as a point cloud turns the floor into dots. Only occupancy points. */
    if (!mesh) {
      return false;
    }
    return String(mesh.name || "").indexOf("occ_points") >= 0;
  }

  function isOccShell(mesh) {
    if (!mesh) {
      return false;
    }
    return String(mesh.name || "").indexOf("occ_shell") >= 0;
  }

  function applyWireframe(on) {
    /* Inspect viewer: THREE.EdgesGeometry(geom, 25) on the solid, not
       material.wireframe (that draws every triangle). Babylon's epsilon is
       cos(threshold): draw an edge when adjacent face normals diverge. */
    const want = !!on;
    const epsilon = Math.cos((25 * Math.PI) / 180);
    const edgeColor = new BABYLON.Color4(228 / 255, 232 / 255, 238 / 255, 0.9);
    for (let i = 0; i < imported.length; i += 1) {
      const mesh = imported[i];
      if (!isOccShell(mesh)) {
        continue;
      }
      const mats = mesh.material
        ? Array.isArray(mesh.material)
          ? mesh.material
          : [mesh.material]
        : [];
      for (let m = 0; m < mats.length; m += 1) {
        if (mats[m]) {
          mats[m].wireframe = false;
        }
      }
      if (want) {
        mesh.enableEdgesRendering(epsilon);
        mesh.edgesWidth = 1.5;
        mesh.edgesColor = edgeColor;
      } else if (mesh.disableEdgesRendering) {
        mesh.disableEdgesRendering();
      }
    }
  }
  function applyPointSize(px) {
    const size = Math.max(1, Math.min(24, Number(px) || DEFAULT_PSIZE));
    for (let i = 0; i < imported.length; i += 1) {
      const mesh = imported[i];
      if (!isOccPoints(mesh)) {
        continue;
      }
      const mats = mesh.material
        ? Array.isArray(mesh.material)
          ? mesh.material
          : [mesh.material]
        : [];
      for (let m = 0; m < mats.length; m += 1) {
        const mat = mats[m];
        if (!mat) {
          continue;
        }
        mat.pointsCloud = true;
        mat.pointSize = size;
        mat.disableLighting = true;
        if (mat.unlit !== undefined) {
          mat.unlit = true;
        }
      }
    }
  }

  function clearImported() {
    for (let i = 0; i < imported.length; i += 1) {
      if (imported[i] && imported[i].dispose) {
        imported[i].dispose(false, true);
      }
    }
    imported = [];
  }

  async function loadGlb(url, px, wire) {
    /* Freeze the orbit. Import must not frame, target, or replace the camera. */
    const pose = {
      a: camera.alpha,
      b: camera.beta,
      r: camera.radius,
      tx: camera.target.x,
      ty: camera.target.y,
      tz: camera.target.z,
    };
    const gen = (loadGen += 1);
    let result;
    try {
      result = await BABYLON.SceneLoader.ImportMeshAsync("", url, "", scene);
    } catch (err) {
      applyCam(pose);
      throw err;
    }
    if (gen !== loadGen) {
      (result.meshes || []).forEach(function (m) {
        m.dispose(false, true);
      });
      applyCam(pose);
      return;
    }
    clearImported();
    imported = [];
    (result.meshes || []).forEach(function (m) {
      const n = String(m.name || "");
      if (n.indexOf("occ_empty") >= 0) {
        m.dispose(false, true);
        return;
      }
      imported.push(m);
    });
    (result.cameras || []).forEach(function (cam) {
      if (cam && cam.dispose) {
        cam.dispose();
      }
    });
    scene.activeCamera = camera;
    applyPointSize(px);
    applyWireframe(wire);
    applyCam(pose);
  }

  async function tick() {
    if (loadBusy) {
      return;
    }
    const path = fieldPath();
    const resetN = fieldReset();
    const px = fieldPsize();
    const wire = fieldWire();
    if (!path) {
      return;
    }
    const pathChanged = path !== lastPath;
    const resetChanged = lastReset !== null && resetN !== lastReset;
    const sizeChanged = lastPsize !== null && px !== lastPsize;
    const wireChanged = lastWire !== null && wire !== lastWire;
    if (!pathChanged && !resetChanged && !sizeChanged && !wireChanged) {
      lastReset = resetN;
      lastPsize = px;
      lastWire = wire;
      return;
    }
    if (pathChanged) {
      lastPath = path;
      lastReset = resetN;
      lastPsize = px;
      lastWire = wire;
      loadBusy = true;
      try {
        await loadGlb(fileUrl(path), px, wire);
      } catch (err) {
        console.error("sn-orbit load", err);
      } finally {
        loadBusy = false;
      }
      return;
    }
    lastReset = resetN;
    lastPsize = px;
    lastWire = wire;
    if (resetChanged) {
      applyDefaultCam();
    }
    if (sizeChanged || resetChanged) {
      applyPointSize(px);
    }
    if (wireChanged || resetChanged) {
      applyWireframe(wire);
    }
  }

  window.addEventListener("resize", function () {
    engine.resize();
  });
  engine.runRenderLoop(function () {
    scene.render();
  });

  applyDefaultCam();

  window.__snOrbit = {
    alive: function () {
      return !!(
        engine &&
        !engine.isDisposed &&
        canvas &&
        canvas.isConnected &&
        engine.getRenderingCanvas() === canvas
      );
    },
  };

  await tick();
  window.setInterval(tick, 200);
})();

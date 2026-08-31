/**
 * Last five successful loads: names in localStorage, file bytes in IndexedDB
 * so a recent click can replace the current OBJ/NPZ after a refresh.
 */

const STORAGE_KEY = "scatteringNet.viewer.recents";
const MAX_RECENTS = 5;
const IDB_NAME = "scatteringNet.viewer.files";
const IDB_STORE = "files";
/** Skip IndexedDB for huge catalogs so the UI stays responsive. */
const MAX_IDB_BYTES = 80 * 1024 * 1024;

/** Same-tab File objects and optional FileSystemFileHandle. */
const memoryFiles = new Map();
const memoryHandles = new Map();

/**
 * @returns {string[]}
 */
export function loadRecents() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.map((x) => String(x)).filter((s) => s.length > 0).slice(0, MAX_RECENTS);
  } catch {
    return [];
  }
}

/**
 * @param {string[]} names
 */
export function saveRecents(names) {
  const trimmed = names.map((x) => String(x)).filter((s) => s.length > 0).slice(0, MAX_RECENTS);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
}

/**
 * Prepend a display name after a successful load.
 * @param {string} name
 */
export function pushRecent(name) {
  const label = String(name || "").trim();
  if (!label) {
    return loadRecents();
  }
  const prev = loadRecents().filter((n) => n !== label);
  const next = [label, ...prev].slice(0, MAX_RECENTS);
  saveRecents(next);
  return next;
}

/**
 * @returns {Promise<IDBDatabase>}
 */
function openRecentsDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: "name" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Keep the File (and Chrome file handle when we have one) so recents reopen
 * without a second Explorer dialog.
 * @param {File} file
 * @param {FileSystemFileHandle|null} [handle]
 */
export async function persistRecentFile(file, handle) {
  const label = String(file && file.name ? file.name : "").trim();
  if (!label) {
    return;
  }
  memoryFiles.set(label, file);
  if (handle) {
    memoryHandles.set(label, handle);
  }
  pushRecent(label);
  try {
    const db = await openRecentsDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      const store = tx.objectStore(IDB_STORE);
      const record = {
        name: label,
        mime: file.type || "",
        savedAt: Date.now(),
        handle: handle || null,
        blob: file.size <= MAX_IDB_BYTES ? file : null,
      };
      store.put(record);
      const keep = new Set(loadRecents());
      const keysReq = store.getAllKeys();
      keysReq.onsuccess = () => {
        const keys = keysReq.result || [];
        for (let i = 0; i < keys.length; i += 1) {
          if (!keep.has(keys[i])) {
            store.delete(keys[i]);
          }
        }
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // Private mode, quota, or handle not cloneable: same-tab Map still works.
  }
}

/**
 * @param {FileSystemFileHandle} handle
 * @returns {Promise<File|null>}
 */
async function fileFromHandle(handle) {
  if (!handle || typeof handle.getFile !== "function") {
    return null;
  }
  try {
    let perm = "granted";
    if (typeof handle.queryPermission === "function") {
      perm = await handle.queryPermission({ mode: "read" });
    }
    if (perm !== "granted" && typeof handle.requestPermission === "function") {
      perm = await handle.requestPermission({ mode: "read" });
    }
    if (perm !== "granted") {
      return null;
    }
    return await handle.getFile();
  } catch {
    return null;
  }
}

/**
 * @param {string} name
 * @returns {Promise<File|null>}
 */
export async function resolveRecentFile(name) {
  const label = String(name || "").trim();
  if (!label) {
    return null;
  }
  if (memoryFiles.has(label)) {
    return memoryFiles.get(label) || null;
  }
  if (memoryHandles.has(label)) {
    const fromHandle = await fileFromHandle(memoryHandles.get(label));
    if (fromHandle) {
      memoryFiles.set(label, fromHandle);
      return fromHandle;
    }
  }
  try {
    const db = await openRecentsDb();
    const rec = await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readonly");
      const req = tx.objectStore(IDB_STORE).get(label);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
    if (!rec) {
      return null;
    }
    if (rec.handle) {
      const fromHandle = await fileFromHandle(rec.handle);
      if (fromHandle) {
        memoryFiles.set(label, fromHandle);
        memoryHandles.set(label, rec.handle);
        return fromHandle;
      }
    }
    if (!rec.blob) {
      return null;
    }
    const file = new File([rec.blob], label, { type: rec.mime || rec.blob.type || "" });
    memoryFiles.set(label, file);
    return file;
  } catch {
    return null;
  }
}

/**
 * Names that can be reopened without a new file picker.
 * @returns {Promise<Set<string>>}
 */
export async function listCachedRecentNames() {
  const cached = new Set(memoryFiles.keys());
  try {
    const db = await openRecentsDb();
    const keys = await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readonly");
      const req = tx.objectStore(IDB_STORE).getAllKeys();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
    for (let i = 0; i < keys.length; i += 1) {
      cached.add(String(keys[i]));
    }
  } catch {
    // IndexedDB unavailable.
  }
  return cached;
}

/**
 * @param {HTMLElement} listEl
 * @param {Set<string>} [cachedNames]
 */
export function renderRecents(listEl, cachedNames) {
  const names = loadRecents();
  const cached = cachedNames || new Set();
  listEl.replaceChildren();
  if (names.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No recent files yet";
    listEl.appendChild(li);
  } else {
    for (const name of names) {
      const li = document.createElement("li");
      const ready = cached.has(name);
      li.className = ready ? "recent-item" : "recent-item stale";
      li.dataset.name = name;
      li.textContent = name;
      li.title = ready
        ? name
        : name + " — not cached; click Open and pick this file";
      listEl.appendChild(li);
    }
  }
  const clear = document.createElement("li");
  clear.className = "recent-clear";
  clear.textContent = "Clear view";
  clear.title = "Remove the current mesh or points";
  listEl.appendChild(clear);
}

/**
 * Show the popover while the pointer is on the Open control or the panel.
 * @param {HTMLElement} wrap
 * @param {HTMLElement} listEl
 * @param {{onSelect?: (name: string) => void, onClear?: () => void}} [handlers]
 */
export function bindRecentsHover(wrap, listEl, handlers) {
  let hideTimer = 0;
  const onSelect = handlers && handlers.onSelect;
  const onClear = handlers && handlers.onClear;

  const hide = () => {
    wrap.classList.remove("recents-open");
  };

  const show = () => {
    window.clearTimeout(hideTimer);
    wrap.classList.add("recents-open");
    listCachedRecentNames().then((cached) => {
      renderRecents(listEl, cached);
    });
  };

  const scheduleHide = () => {
    hideTimer = window.setTimeout(hide, 180);
  };

  wrap.addEventListener("mouseenter", show);
  wrap.addEventListener("mouseleave", scheduleHide);
  wrap.addEventListener("focusin", show);
  wrap.addEventListener("focusout", scheduleHide);

  listEl.addEventListener("click", (event) => {
    const clearItem =
      event.target && event.target.closest && event.target.closest("li.recent-clear");
    if (clearItem) {
      hide();
      if (onClear) {
        onClear();
      }
      return;
    }
    const item =
      event.target && event.target.closest && event.target.closest("li.recent-item");
    if (!item || !onSelect || item.classList.contains("stale")) {
      return;
    }
    hide();
    onSelect(item.dataset.name || item.textContent || "");
  });
}

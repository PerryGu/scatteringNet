/**
 * Step 4: fetch OBJ text from the local helper using NPZ mesh_path.
 */

/**
 * @param {string} stored
 * @returns {string}
 */
export function meshBasename(stored) {
  const parts = String(stored || "").replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || "";
}

/**
 * @param {string} storedPath
 * @returns {Promise<string>}
 */
export async function fetchMeshObjText(storedPath) {
  const url = "/api/mesh?path=" + encodeURIComponent(storedPath);
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.status + " " + res.statusText;
    try {
      const body = await res.json();
      if (body && body.error) {
        detail = String(body.error);
      }
    } catch {
      // keep status text
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  const text = await res.text();
  if (!String(text).trim()) {
    throw new Error("Helper returned an empty OBJ");
  }
  return text;
}

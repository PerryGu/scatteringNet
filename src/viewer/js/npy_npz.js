/**
 * Minimal NPZ/NPY reader for occupancy files (points, labels, mesh_path).
 * Uses the browser ZIP layout numpy writes; no occupancy Python.
 */

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

/**
 * @param {Uint8Array} bytes
 * @returns {Promise<Record<string, Uint8Array>>}
 */
export async function unzipToFiles(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  /** @type {Record<string, Uint8Array>} */
  const files = {};
  let offset = 0;
  while (offset + 30 <= bytes.byteLength) {
    const sig = view.getUint32(offset, true);
    if (sig === 0x02014b50 || sig === 0x06054b50) {
      break;
    }
    if (sig !== 0x04034b50) {
      throw new Error("File is not NPZ (ZIP)");
    }
    const flags = view.getUint16(offset + 6, true);
    const method = view.getUint16(offset + 8, true);
    const compSize = view.getUint32(offset + 18, true);
    const nameLen = view.getUint16(offset + 26, true);
    const extraLen = view.getUint16(offset + 28, true);
    const name = new TextDecoder("utf-8").decode(
      bytes.subarray(offset + 30, offset + 30 + nameLen)
    );
    const dataStart = offset + 30 + nameLen + extraLen;
    if (flags & 0x08) {
      throw new Error("NPZ uses ZIP data descriptors; this reader does not");
    }
    const compressed = bytes.subarray(dataStart, dataStart + compSize);
    let raw;
    if (method === 0) {
      raw = compressed;
    } else if (method === 8) {
      raw = await inflateRaw(compressed);
    } else {
      throw new Error("Unsupported ZIP method " + method + " for " + name);
    }
    files[name.replace(/\\/g, "/")] = raw;
    offset = dataStart + compSize;
  }
  return files;
}

/**
 * @param {Uint8Array} compressed
 * @returns {Promise<Uint8Array>}
 */
async function inflateRaw(compressed) {
  if (typeof DecompressionStream !== "function") {
    throw new Error("This browser cannot inflate compressed NPZ");
  }
  const stream = new Blob([compressed]).stream().pipeThrough(
    new DecompressionStream("deflate-raw")
  );
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * @param {Uint8Array} bytes
 * @returns {{descr: string, fortran: boolean, shape: number[], data: Uint8Array}}
 */
export function parseNpy(bytes) {
  for (let i = 0; i < NPY_MAGIC.length; i += 1) {
    if (bytes[i] !== NPY_MAGIC[i]) {
      throw new Error("Not an NPY array");
    }
  }
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let headerLen;
  let headerStart;
  if (major <= 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  }
  const header = new TextDecoder("ascii").decode(
    bytes.subarray(headerStart, headerStart + headerLen)
  );
  const descrMatch = /'descr':\s*'([^']+)'/.exec(header);
  const orderMatch = /'fortran_order':\s*(True|False)/.exec(header);
  const shapeMatch = /'shape':\s*\(([^)]*)\)/.exec(header);
  if (!descrMatch || !orderMatch || !shapeMatch) {
    throw new Error("NPY header missing descr/shape");
  }
  const shapeParts = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const shape = shapeParts.map((s) => Number(s));
  return {
    descr: descrMatch[1],
    fortran: orderMatch[1] === "True",
    shape,
    data: bytes.subarray(headerStart + headerLen),
  };
}

/**
 * @param {Record<string, Uint8Array>} files
 * @param {string} key
 * @returns {Uint8Array}
 */
function npyBytes(files, key) {
  const exact = files[key] || files[key + ".npy"];
  if (exact) {
    return exact;
  }
  const found = Object.keys(files).find((name) => {
    const base = name.split("/").pop() || name;
    return base === key || base === key + ".npy";
  });
  if (!found) {
    throw new Error("NPZ has no '" + key + "' array");
  }
  return files[found];
}

/**
 * @param {Uint8Array} data
 * @param {boolean} little
 * @param {number} count
 * @returns {Float32Array}
 */
function readFloat32(data, little, count) {
  const out = new Float32Array(count);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let i = 0; i < count; i += 1) {
    out[i] = view.getFloat32(i * 4, little);
  }
  return out;
}

/**
 * @param {Uint8Array} data
 * @param {boolean} little
 * @param {number} count
 * @returns {Float32Array}
 */
function readFloat64As32(data, little, count) {
  const out = new Float32Array(count);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let i = 0; i < count; i += 1) {
    out[i] = view.getFloat64(i * 8, little);
  }
  return out;
}

/**
 * @param {ReturnType<typeof parseNpy>} arr
 * @returns {Float32Array}
 */
export function npyToFloat32(arr) {
  if (arr.fortran) {
    throw new Error("Fortran-order arrays are not supported");
  }
  const n = arr.shape.reduce((a, b) => a * b, 1);
  const little = !arr.descr.startsWith(">");
  if (arr.descr.includes("f4")) {
    return readFloat32(arr.data, little, n);
  }
  if (arr.descr.includes("f8")) {
    return readFloat64As32(arr.data, little, n);
  }
  if (arr.descr.includes("u1") || arr.descr === "|b1" || arr.descr.endsWith("i1")) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      out[i] = arr.data[i];
    }
    return out;
  }
  if (arr.descr.includes("i4") || arr.descr.includes("u4")) {
    const out = new Float32Array(n);
    const view = new DataView(arr.data.buffer, arr.data.byteOffset, arr.data.byteLength);
    const signed = arr.descr.includes("i4");
    for (let i = 0; i < n; i += 1) {
      out[i] = signed ? view.getInt32(i * 4, little) : view.getUint32(i * 4, little);
    }
    return out;
  }
  throw new Error("Unsupported NPY dtype " + arr.descr);
}

/**
 * 0-d unicode / bytestring used for mesh_path. Pickled object arrays are skipped.
 * @param {ReturnType<typeof parseNpy>} arr
 * @returns {string}
 */
export function npyToString(arr) {
  const descr = arr.descr;
  if (descr.includes("O")) {
    return "";
  }
  const little = !descr.startsWith(">");
  if (descr.includes("U")) {
    const um = /U(\d+)/.exec(descr);
    const maxChars = um ? Number(um[1]) : Math.floor(arr.data.byteLength / 4);
    const nChars = Math.min(maxChars, Math.floor(arr.data.byteLength / 4));
    const view = new DataView(arr.data.buffer, arr.data.byteOffset, arr.data.byteLength);
    let text = "";
    for (let i = 0; i < nChars; i += 1) {
      const cp = view.getUint32(i * 4, little);
      if (cp === 0) {
        break;
      }
      text += String.fromCodePoint(cp);
    }
    return text.trim();
  }
  if (descr.includes("S") || descr.includes("a")) {
    let end = arr.data.indexOf(0);
    if (end < 0) {
      end = arr.data.length;
    }
    return new TextDecoder("utf-8").decode(arr.data.subarray(0, end)).trim();
  }
  throw new Error("mesh_path dtype is not a string (" + descr + ")");
}

/**
 * Occupancy arrays from a user-picked NPZ (same keys as data_npz.py).
 * @param {ArrayBuffer} buffer
 * @returns {Promise<{points: Float32Array, labels: Float32Array, meshPath: string, n: number}>}
 */
export async function parseOccupancyNpz(buffer) {
  const files = await unzipToFiles(new Uint8Array(buffer));
  const pointsArr = parseNpy(npyBytes(files, "points"));
  const labelsArr = parseNpy(npyBytes(files, "labels"));
  if (pointsArr.shape.length !== 2 || pointsArr.shape[1] !== 3) {
    throw new Error("points must have shape (N, 3), got (" + pointsArr.shape.join(", ") + ")");
  }
  const n = pointsArr.shape[0];
  const labelCount = labelsArr.shape.reduce((a, b) => a * b, 1);
  if (labelCount !== n) {
    throw new Error("labels length " + labelCount + " does not match N=" + n);
  }
  let meshPath = "";
  try {
    meshPath = npyToString(parseNpy(npyBytes(files, "mesh_path")));
  } catch {
    meshPath = "";
  }
  return {
    points: npyToFloat32(pointsArr),
    labels: npyToFloat32(labelsArr),
    meshPath,
    n,
  };
}

/**
 * Load/save viewer UI prefs (checkboxes, sliders, selected model).
 * File lives next to the page as ui_prefs.json via the helper.
 */

export const UI_PREF_DEFAULTS = {
  mesh: true,
  inside: true,
  outside: true,
  wireframe: false,
  opacity: 100,
  point_size: 100,
  draw_cap: 200000,
  density: 71,
  model_id: "",
};

/**
 * @returns {Promise<typeof UI_PREF_DEFAULTS>}
 */
export async function fetchUiPrefs() {
  try {
    const res = await fetch("/api/ui-prefs");
    if (!res.ok) {
      return { ...UI_PREF_DEFAULTS };
    }
    const body = await res.json();
    const prefs = body && body.prefs && typeof body.prefs === "object" ? body.prefs : {};
    return { ...UI_PREF_DEFAULTS, ...prefs };
  } catch {
    return { ...UI_PREF_DEFAULTS };
  }
}

/**
 * @param {typeof UI_PREF_DEFAULTS} prefs
 */
export async function postUiPrefs(prefs) {
  const res = await fetch("/api/ui-prefs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(prefs),
  });
  if (!res.ok) {
    throw new Error("could not save ui prefs");
  }
}

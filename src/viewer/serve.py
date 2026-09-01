"""Serve the occupancy viewer, optional mesh helper, and open the browser."""

from __future__ import annotations

import json
import mimetypes
import os
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

PORT_START = 8765
PORT_END = 8775
ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
MAX_OBJ_BYTES = 32 * 1024 * 1024
MAX_INFER_JSON_BYTES = 80 * 1024 * 1024
MAX_UI_PREFS_JSON_BYTES = 16_384
JS_MIME = "text/javascript"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pick_port() -> int:
    for port in range(PORT_START, PORT_END + 1):
        if not _port_in_use(port):
            return port
    raise OSError(
        f"no free port in {PORT_START}-{PORT_END}; close old python http.server windows"
    )


def _yaml_quoted_or_plain(raw: str) -> str:
    """Take a YAML scalar; ignore an inline ``#`` comment after a quoted string."""
    text = str(raw).strip()
    if not text:
        return ""
    quote = text[0]
    if quote in "\"'":
        end = 1
        while end < len(text):
            if text[end] == quote:
                return text[1:end]
            end += 1
        return text[1:].strip()
    return text.split("#", 1)[0].strip().strip("\"'")


def _read_data_dir() -> Path | None:
    """data_dir from repo config.yaml without importing occupancy config/torch."""
    cfg_path = REPO / "config.yaml"
    if not cfg_path.is_file():
        return None
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith("data_dir:"):
            continue
        raw = _yaml_quoted_or_plain(stripped.split(":", 1)[1])
        if not raw:
            return None
        path = Path(raw)
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            return None
        return None
    return None


class Handler(SimpleHTTPRequestHandler):
    """Static viewer files, mesh fetch, NPZ infer (Job A), OBJ fill/envelope/infer (Job B)."""

    data_dir: Path | None = None
    models_dir: Path | None = None

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html",
        ".css": "text/css",
        ".js": JS_MIME,
        ".mjs": JS_MIME,
        ".json": "application/json",
        ".wasm": "application/wasm",
    }

    def guess_type(self, path: str) -> str:
        clean = path.split("?", 1)[0].lower()
        if clean.endswith(".js") or clean.endswith(".mjs"):
            return JS_MIME
        return super().guess_type(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_request(self, code: object = "-", size: object = "-") -> None:
        path = getattr(self, "path", "")
        if path.startswith("/api/"):
            self.log_message('"%s" %s %s', self.requestline, str(code), str(size))
            return
        ctype = self.guess_type(path)
        self.log_message('"%s" %s %s  Content-Type=%s', self.requestline, str(code), str(size), ctype)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            models = []
            if self.models_dir is not None:
                try:
                    from model_access import list_viewer_models

                    models = list_viewer_models(
                        self.models_dir, runs_root=REPO / "runs"
                    )
                except Exception:
                    models = []
            self._send_json(
                200,
                {
                    "ok": True,
                    "helper": self.data_dir is not None,
                    "data_dir": str(self.data_dir) if self.data_dir else "",
                    "models_dir": str(self.models_dir) if self.models_dir else "",
                    "n_models": len(models),
                },
            )
            return
        if parsed.path == "/api/models":
            self._serve_models()
            return
        if parsed.path == "/api/mesh":
            self._serve_mesh(urllib.parse.parse_qs(parsed.query))
            return
        if parsed.path == "/api/ui-prefs":
            self._ui_prefs_get()
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/infer-npz":
            self._infer_npz()
            return
        if parsed.path == "/api/fill-obj":
            self._fill_obj()
            return
        if parsed.path == "/api/envelope-obj":
            self._envelope_obj()
            return
        if parsed.path == "/api/faces-obj":
            self._faces_obj()
            return
        if parsed.path == "/api/infer-obj":
            self._infer_obj()
            return
        if parsed.path == "/api/ui-prefs":
            self._ui_prefs_post()
            return
        self.send_error(404, "unknown POST")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_models(self) -> None:
        try:
            from model_access import list_viewer_models

            root = self.models_dir
            models = (
                list_viewer_models(root, runs_root=REPO / "runs")
                if root is not None
                else []
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, "models": models})

    def _read_json_body(self, max_bytes: int | None = None) -> dict:
        length_raw = self.headers.get("Content-Length")
        if not length_raw:
            raise ValueError("missing Content-Length")
        length = int(length_raw)
        limit = MAX_INFER_JSON_BYTES if max_bytes is None else int(max_bytes)
        if length < 0 or length > limit:
            raise ValueError("request too large")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def _ui_prefs_get(self) -> None:
        try:
            from ui_prefs import load_ui_prefs

            prefs = load_ui_prefs(ROOT)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, "prefs": prefs})

    def _ui_prefs_post(self) -> None:
        try:
            payload = self._read_json_body(max_bytes=MAX_UI_PREFS_JSON_BYTES)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON: " + str(exc)})
            return
        try:
            from ui_prefs import save_ui_prefs

            prefs = save_ui_prefs(ROOT, payload)
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except OSError as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, "prefs": prefs})

    def _infer_npz(self) -> None:
        if self.models_dir is None:
            self._send_json(503, {"error": "helper has no models/ folder"})
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON: " + str(exc)})
            return
        run_id = str(payload.get("checkpoint") or payload.get("run_id") or "").strip()
        if not run_id:
            self._send_json(400, {"error": "missing checkpoint"})
            return
        try:
            from infer_job import (
                decode_labels_b64,
                decode_points_b64,
                infer_uploaded_npz,
            )

            points = decode_points_b64(str(payload.get("points_b64") or ""))
            labels = decode_labels_b64(str(payload.get("labels_b64") or ""), int(points.shape[0]))
            result = infer_uploaded_npz(
                run_id=run_id,
                models_root=self.models_dir,
                data_dir=self.data_dir,
                npz_name=str(payload.get("name") or ""),
                mesh_path=str(payload.get("mesh_path") or ""),
                points=points,
                labels=labels,
            )
        except ImportError as exc:
            self._send_json(
                503,
                {
                    "error": "PyTorch infer needs the conda env (scatteringNet). "
                    + str(exc),
                },
            )
            return
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except KeyError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _fill_obj(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON: " + str(exc)})
            return
        try:
            from obj_fill import fill_from_obj_text

            spacing = float(payload.get("spacing") or 0)
            result = fill_from_obj_text(str(payload.get("obj_text") or ""), spacing)
        except ImportError as exc:
            self._send_json(503, {"error": "fill helper import failed: " + str(exc)})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _envelope_obj(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON: " + str(exc)})
            return
        try:
            from envelope_job import envelope_from_obj_text

            n_surface = int(payload.get("n_surface") or 0)
            result = envelope_from_obj_text(
                str(payload.get("obj_text") or ""), n_surface
            )
        except ImportError as exc:
            self._send_json(
                503, {"error": "envelope helper import failed: " + str(exc)}
            )
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _faces_obj(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON: " + str(exc)})
            return
        try:
            from faces_job import faces_overlay_from_obj_text

            n_faces = int(payload.get("n_faces") or 0)
            result = faces_overlay_from_obj_text(
                str(payload.get("obj_text") or ""), n_faces
            )
        except ImportError as exc:
            self._send_json(
                503, {"error": "faces helper import failed: " + str(exc)}
            )
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _infer_obj(self) -> None:
        if self.models_dir is None:
            self._send_json(503, {"error": "helper has no models/ folder"})
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON: " + str(exc)})
            return
        run_id = str(payload.get("checkpoint") or payload.get("run_id") or "").strip()
        if not run_id:
            self._send_json(400, {"error": "missing checkpoint"})
            return
        try:
            from infer_job import decode_points_b64, infer_uploaded_obj

            points = decode_points_b64(str(payload.get("points_b64") or ""))
            result = infer_uploaded_obj(
                run_id=run_id,
                models_root=self.models_dir,
                data_dir=self.data_dir,
                obj_name=str(payload.get("name") or ""),
                obj_text=str(payload.get("obj_text") or ""),
                points=points,
            )
        except ImportError as exc:
            self._send_json(
                503,
                {
                    "error": "PyTorch infer needs the conda env (scatteringNet). "
                    + str(exc),
                },
            )
            return
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _serve_mesh(self, query: dict[str, list[str]]) -> None:
        if self.data_dir is None:
            self._send_json(503, {"error": "helper has no data_dir (check config.yaml)"})
            return
        stored = ""
        if "path" in query and query["path"]:
            stored = urllib.parse.unquote(query["path"][0]).strip()
        if not stored:
            self._send_json(400, {"error": "missing path"})
            return
        try:
            from mesh_access import resolve_viewer_mesh

            obj_path = resolve_viewer_mesh(stored, self.data_dir)
        except ImportError as exc:
            self._send_json(503, {"error": "helper import failed: " + str(exc)})
            return
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        size = obj_path.stat().st_size
        if size > MAX_OBJ_BYTES:
            self._send_json(413, {"error": f"OBJ too large ({size} bytes)"})
            return
        data = obj_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    os.chdir(ROOT)
    port = _pick_port()
    url = f"http://127.0.0.1:{port}/"
    guessed_js, _ = mimetypes.guess_type("dummy.js")
    data_dir = _read_data_dir()
    models_dir = REPO / "models"
    Handler.data_dir = data_dir
    Handler.models_dir = models_dir

    def open_browser() -> None:
        time.sleep(0.5)
        webbrowser.open(url)

    print(f"viewer root: {ROOT}")
    print(f"open: {url}")
    print(f"Python default MIME for .js: {guessed_js!r}")
    print(f"This server will send .js as: {JS_MIME!r}")
    if data_dir is not None:
        print(f"helper data_dir: {data_dir}")
    else:
        print("helper data_dir: (missing) Show object cannot fetch meshes")
    print(f"helper models/: {models_dir}")
    if not models_dir.is_dir():
        print("NOTE: no models/ folder yet; Run model will have an empty list.")
    print(f"ui prefs: {ROOT / 'ui_prefs.json'}")
    if guessed_js not in ("text/javascript", "application/javascript", "application/ecmascript"):
        print("NOTE: Windows often serves .js as text/plain; Chrome then never runs the module.")
    print("Leave this window open. Close it (and any old server window) to stop.")
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        threading.Thread(target=open_browser, daemon=True).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()

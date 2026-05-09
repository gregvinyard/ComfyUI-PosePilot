"""
ComfyUI-PosePilot — Pose library integration for ComfyUI.

Provides two nodes:
  PosePilotSelector  — browse a PosePilot library folder and select a pose
  PosePilot3D        — Load3D-style viewport + server-side OpenPose rendering

Custom routes (registered via aiohttp):
  GET  /posepilot/browse              — open a tkinter folder-picker dialog
  GET  /posepilot/list_libraries        — list library names in models/pose-pilot/
  GET  /posepilot/list_glbs?library=... — list GLB filenames for a named library
  GET  /posepilot/library?path=...    — return filtered manifest entries as JSON
  GET  /posepilot/thumbnail?path=...  — serve a pose thumbnail PNG
  GET  /posepilot/get_views?glb_path= — return saved favorite views for a GLB
  POST /posepilot/save_view           — append a view to a GLB's favorites
  POST /posepilot/delete_view         — remove a view from a GLB's favorites
"""
from __future__ import annotations

from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO

# Serve the web/ directory so posepilot.js is loaded as a frontend extension
WEB_DIRECTORY = "./web"

from .nodes_selector import PosePilotSelector
from .nodes_3d import PosePilot3D


# ---------------------------------------------------------------------------
# Custom HTTP routes
# ---------------------------------------------------------------------------

def _build_routes():
    from aiohttp import web

    routes = web.RouteTableDef()

    @routes.get("/posepilot/browse")
    async def browse_folder(request: web.Request) -> web.Response:
        """Open a system folder-picker and return the chosen path."""
        import asyncio
        import tkinter as tk
        from tkinter import filedialog

        loop = asyncio.get_event_loop()

        def _pick():
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(title="Select PosePilot library folder")
            root.destroy()
            return path or ""

        path = await loop.run_in_executor(None, _pick)
        return web.json_response({"path": path})

    @routes.get("/posepilot/list_libraries")
    async def list_libraries(request: web.Request) -> web.Response:
        """Return sorted library names from models/pose-pilot/ (subdirs with manifest.json)."""
        import folder_paths as _fp
        from pathlib import Path
        pose_pilot_dir = Path(_fp.models_dir) / "pose-pilot"
        if not pose_pilot_dir.is_dir():
            return web.json_response({"libraries": []})
        libs = sorted(
            d.name for d in pose_pilot_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        )
        return web.json_response({"libraries": libs})

    @routes.get("/posepilot/list_glbs")
    async def list_glbs(request: web.Request) -> web.Response:
        """Return sorted GLB filenames for a library.

        Accepts either:
          ?library=<name>   — looks in models/pose-pilot/<name>/poses/  (new)
          ?path=<full_path> — looks in <path>/poses/                    (legacy)
        """
        import folder_paths as _fp
        from pathlib import Path
        library = request.rel_url.query.get("library", "")
        path    = request.rel_url.query.get("path", "")
        if library:
            poses_dir = Path(_fp.models_dir) / "pose-pilot" / library / "poses"
        elif path:
            poses_dir = Path(path) / "poses"
        else:
            return web.json_response({"files": []})
        if not poses_dir.is_dir():
            return web.json_response({"files": []})
        files = sorted(f.name for f in poses_dir.glob("*.glb"))
        return web.json_response({"files": files})

    @routes.get("/posepilot/library")
    async def library_info(request: web.Request) -> web.Response:
        """Return manifest entries (with optional filters) as JSON."""
        import json
        from pathlib import Path

        library_path = request.rel_url.query.get("path", "")
        manifest_file = Path(library_path) / "manifest.json"
        if not manifest_file.exists():
            return web.json_response({"error": "manifest not found"}, status=404)

        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)

        entries = list(manifest.get("entries", {}).values())

        # Optional filters
        q = request.rel_url.query
        ps  = q.get("pose_set", "")
        pos = q.get("posture", "")
        gen = q.get("gender", "")
        eng = q.get("energy", "")

        def _keep(e):
            t = e.get("tags", {})
            if ps  and t.get("pose_set") != ps:  return False
            if pos and t.get("posture")  != pos: return False
            if gen and t.get("gender")   != gen: return False
            if eng and t.get("energy")   != eng: return False
            return True

        visible = [{"pose_id": e["pose_id"], "tags": e["tags"],
                    "source": e["source"], "confidence": e["extraction"]["confidence"]}
                   for e in entries if _keep(e)]
        return web.json_response({"count": len(visible), "entries": visible})

    @routes.get("/posepilot/thumbnail")
    async def serve_thumbnail(request: web.Request) -> web.Response:
        """Serve a pose thumbnail PNG by absolute path."""
        from pathlib import Path
        path = request.rel_url.query.get("path", "")
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".png":
            return web.Response(status=404)
        data = p.read_bytes()
        return web.Response(body=data, content_type="image/png")

    @routes.get("/posepilot/get_views")
    async def get_views(request: web.Request) -> web.Response:
        """Return saved favorite views for a GLB (reads <stem>.views.json sidecar)."""
        import json
        from pathlib import Path
        glb_path = request.rel_url.query.get("glb_path", "")
        if not glb_path:
            return web.json_response({"views": []})
        views_file = Path(glb_path).with_suffix(".views.json")
        if not views_file.exists():
            return web.json_response({"views": []})
        try:
            data = json.loads(views_file.read_text(encoding="utf-8"))
            return web.json_response({"views": data.get("views", [])})
        except Exception:
            return web.json_response({"views": []})

    @routes.post("/posepilot/save_view")
    async def save_view(request: web.Request) -> web.Response:
        """Append a camera view to a GLB's .views.json sidecar."""
        import json
        from pathlib import Path
        body = await request.json()
        glb_path = body.get("glb_path", "")
        if not glb_path:
            return web.json_response({"ok": False, "error": "glb_path required"}, status=400)
        label = (body.get("label") or "").strip()
        camera = body.get("camera")
        if not camera:
            return web.json_response({"ok": False, "error": "camera required"}, status=400)
        views_file = Path(glb_path).with_suffix(".views.json")
        if views_file.exists():
            data = json.loads(views_file.read_text(encoding="utf-8"))
        else:
            data = {"views": []}
        data["views"].append({"label": label, "camera": camera})
        views_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return web.json_response({"ok": True, "count": len(data["views"])})

    @routes.post("/posepilot/delete_view")
    async def delete_view(request: web.Request) -> web.Response:
        """Remove a view by 0-based index from a GLB's .views.json sidecar."""
        import json
        from pathlib import Path
        body = await request.json()
        glb_path = body.get("glb_path", "")
        index = body.get("index", -1)
        if not glb_path:
            return web.json_response({"ok": False, "error": "glb_path required"}, status=400)
        views_file = Path(glb_path).with_suffix(".views.json")
        if not views_file.exists():
            return web.json_response({"ok": False, "error": "no views file"}, status=404)
        data = json.loads(views_file.read_text(encoding="utf-8"))
        views = data.get("views", [])
        if not (0 <= index < len(views)):
            return web.json_response({"ok": False, "error": "index out of range"}, status=400)
        views.pop(index)
        data["views"] = views
        views_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return web.json_response({"ok": True, "count": len(views)})

    return routes


# ---------------------------------------------------------------------------
# Extension entry point
# ---------------------------------------------------------------------------

class PosePilotExtension(ComfyExtension):

    @override
    async def on_load(self) -> None:
        try:
            import folder_paths as _fp
            from pathlib import Path as _Path
            pose_pilot_dir = _Path(_fp.models_dir) / "pose-pilot"
            pose_pilot_dir.mkdir(parents=True, exist_ok=True)

            from server import PromptServer
            routes = _build_routes()
            PromptServer.instance.app.add_routes(routes)
            print("[PosePilot] Canonical library folder:", pose_pilot_dir)
            print("[PosePilot] Custom routes registered: browse, list_libraries, list_glbs, library, thumbnail, get_views, save_view, delete_view")
        except Exception as e:
            print(f"[PosePilot] Warning: could not register custom routes: {e}")

    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [PosePilotSelector, PosePilot3D]


async def comfy_entrypoint() -> PosePilotExtension:
    return PosePilotExtension()

"""
PosePilot3D node — Load3D viewer + server-side OpenPose rendering.

Usage in ComfyUI:
  1. Connect PosePilotSelector's glb_path → this node's glb_path input.
  2. Set model_file to "3d/posepilot_current.glb" once; the Selector will
     swap the underlying file each time you cycle poses — no combo change needed.
  3. Adjust the 3-D camera in the viewer, then execute to get the OpenPose
     skeleton image, normal map, shaded render, and camera_info.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import nodes as comfy_nodes
import folder_paths
from comfy_api.latest import IO, Types


def _load_image_as_tensor(annotated_path: str):
    """Load a ComfyUI annotated image path and return (tensor, mask) via LoadImage."""
    node = comfy_nodes.LoadImage()
    img, mask = node.load_image(image=annotated_path)
    return img, mask


def _blank_tensors(width: int, height: int):
    """Return (image_tensor, mask_tensor) filled with zeros."""
    import torch
    img  = torch.zeros(1, height, width, 3, dtype=torch.float32)
    mask = torch.zeros(1, height, width,    dtype=torch.float32)
    return img, mask


def _default_camera(joint_positions: np.ndarray) -> dict:
    """Frontal perspective camera centred on the model's bounding box."""
    ys = joint_positions[:, 1]
    cy = float((ys.min() + ys.max()) / 2)
    return {
        "position": {"x": 0.0, "y": cy, "z": 2.5},
        "target":   {"x": 0.0, "y": cy, "z": 0.0},
        "zoom": 1.0,
        "cameraType": "perspective",
    }


def _load_favorite_camera(glb_path: str, view_preset: str) -> dict | None:
    """
    Return the saved camera dict for view_preset (format "N: label"), or None.
    Reads the .views.json sidecar file next to the GLB.
    """
    import json
    import re

    match = re.match(r"^(\d+):", view_preset)
    if not match:
        return None
    idx = int(match.group(1)) - 1  # 1-based label → 0-based index

    views_file = Path(glb_path).with_suffix(".views.json")
    if not views_file.exists():
        return None
    try:
        data = json.loads(views_file.read_text(encoding="utf-8"))
        views = data.get("views", [])
        if 0 <= idx < len(views):
            return views[idx].get("camera")
    except Exception:
        pass
    return None


class PosePilot3D(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")
        os.makedirs(input_dir, exist_ok=True)
        base_path = Path(folder_paths.get_input_directory())
        input_path = Path(input_dir)

        def _norm(p: Path) -> str:
            return str(p.relative_to(base_path)).replace("\\", "/")

        files = sorted(
            _norm(fp)
            for fp in input_path.rglob("*")
            if fp.suffix.lower() in {".gltf", ".glb", ".obj", ".fbx", ".stl"}
        )

        # Ensure the canonical PosePilot file is always the default/first option
        # (Selector copies the active pose here; it may not exist at startup time)
        canonical = "3d/posepilot_current.glb"
        if canonical not in files:
            files.insert(0, canonical)

        return IO.Schema(
            node_id="PosePilot3D",
            display_name="Pose Pilot: Viewfinder",
            category="PosePilot",
            is_experimental=True,
            description=(
                "Integrated 3-D viewer (like Load3D) that also renders an OpenPose "
                "skeleton image from the pose parameters. Connect glb_path from "
                "PosePilotSelector to enable server-side FK and projection."
            ),
            inputs=[
                IO.Combo.Input(
                    "model_file",
                    options=files,
                    default=canonical,
                    upload=IO.UploadType.model,
                    tooltip="3-D model displayed in the viewport. Auto-managed when glb_path is connected.",
                ),
                IO.Load3D.Input("image", optional=True),
                IO.String.Input(
                    "glb_path",
                    default="",
                    multiline=False,
                    tooltip="Absolute path to the .glb from PosePilotSelector (used for FK)",
                ),
                IO.Int.Input("width",  default=1024, min=64, max=4096, step=64),
                IO.Int.Input("height", default=1024, min=64, max=4096, step=64),
                IO.Boolean.Input("include_face",  default=False,
                                 tooltip="Overlay jaw and eye keypoints"),
                IO.Boolean.Input("include_hands", default=False,
                                 tooltip="Overlay finger-tip keypoints"),
                IO.Combo.Input(
                    "view_preset",
                    options=["(live camera)"],
                    tooltip=(
                        "Saved favorite view to use for OpenPose rendering. "
                        "'(live camera)' uses the viewport camera as usual. "
                        "Populated automatically when glb_path is connected. "
                        "Use the Save/Delete buttons below to manage favorites."
                    ),
                ),
                IO.String.Input(
                    "view_label",
                    default="",
                    multiline=False,
                    tooltip="Optional label for the view you are about to save (e.g. 'front', 'side').",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="openpose",
                                tooltip="OpenPose COCO-18 skeleton on black background"),
                IO.Image.Output(display_name="normal_map"),
                IO.Image.Output(display_name="shaded_render"),
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.Mask.Output(display_name="mask"),
                IO.String.Output(display_name="mesh_path"),
                IO.File3DAny.Output(display_name="model_3d"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_file: str,
        glb_path: str,
        width: int,
        height: int,
        include_face: bool,
        include_hands: bool,
        image=None,
        view_preset: str = "(live camera)",
        view_label: str = "",
        **kwargs,
    ) -> IO.NodeOutput:
        import torch
        from .glb_utils import load_glb_joints
        from .openpose_render import render_openpose, bgr_to_tensor

        # ── Load3D widget output (may be absent or a string before first render) ─
        scene = image if isinstance(image, dict) and "image" in image else None

        if scene is not None:
            img_tensor, _ = _load_image_as_tensor(
                folder_paths.get_annotated_filepath(scene["image"])
            )
            _, mask_tensor = _load_image_as_tensor(
                folder_paths.get_annotated_filepath(scene["mask"])
            )
            normal_tensor, _ = _load_image_as_tensor(
                folder_paths.get_annotated_filepath(scene["normal"])
            )
            camera_info = scene["camera_info"]
        else:
            img_tensor = mask_tensor = normal_tensor = None
            camera_info = None

        # ── FK from GLB ───────────────────────────────────────────────────
        effective_glb = glb_path.strip()
        if not effective_glb:
            fallback = folder_paths.get_annotated_filepath(model_file)
            if not Path(fallback).exists():
                raise FileNotFoundError(
                    "No GLB found. Connect PosePilotSelector → glb_path, or execute the "
                    "Selector node first so it creates posepilot_current.glb."
                )
            effective_glb = fallback
        joint_positions, _, _ = load_glb_joints(effective_glb)

        # Fall back to a default frontal camera if the widget hasn't rendered yet
        if camera_info is None:
            camera_info = _default_camera(joint_positions)
            print("[PosePilot3D] Load3D widget not yet rendered — using default frontal camera.")

        # ── Override camera from saved favorite view ───────────────────────
        if view_preset != "(live camera)" and glb_path.strip():
            camera_info = _load_favorite_camera(glb_path.strip(), view_preset) or camera_info

        # ── OpenPose rendering ────────────────────────────────────────────
        openpose_bgr = render_openpose(
            joint_positions,
            camera_info,
            width,
            height,
            include_face=include_face,
            include_hands=include_hands,
        )
        openpose_tensor = bgr_to_tensor(openpose_bgr)

        # Blank placeholders when the 3D widget hasn't rendered
        if img_tensor is None:
            img_tensor, mask_tensor = _blank_tensors(width, height)
            normal_tensor, _ = _blank_tensors(width, height)

        # ── File3D output ─────────────────────────────────────────────────
        file_3d = Types.File3D(folder_paths.get_annotated_filepath(model_file))

        return IO.NodeOutput(
            openpose_tensor,
            normal_tensor,
            img_tensor,
            camera_info,
            mask_tensor,
            model_file,
            file_3d,
        )

    process = execute  # legacy compatibility

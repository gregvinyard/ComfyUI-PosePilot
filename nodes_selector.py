"""
PosePilotSelector — pick a pose from the canonical models/pose-pilot/ location.

Usage:
  1. Add pose libraries to <ComfyUI>/models/pose-pilot/ using the pose-extractor CLI:
       uv run pose-extractor ingest <images> --library-root <comfyui>/models/pose-pilot --library-name my_lib
  2. Select the library from the first dropdown.
  3. Select a pose GLB from the second dropdown.
  4. Use the ⚙ cycle mode (fixed / next / random) to browse poses like a seed.
  5. Execute: outputs the canonical GLB path and shows a thumbnail preview.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from comfy_api.latest import IO, UI


def _list_libraries(pose_pilot_dir: Path) -> list[str]:
    """Return sorted library names (subdirs of pose-pilot/ that contain manifest.json)."""
    if not pose_pilot_dir.is_dir():
        return [""]
    libs = sorted(
        d.name for d in pose_pilot_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )
    return libs if libs else [""]


class PosePilotSelector(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        import folder_paths as _fp
        pose_pilot_dir = Path(_fp.models_dir) / "pose-pilot"
        libraries = _list_libraries(pose_pilot_dir)

        return IO.Schema(
            node_id="PosePilotSelector",
            display_name="Pose Pilot: Select Pose",
            category="PosePilot",
            description=(
                "Select a pose from a library in models/pose-pilot/. "
                "Add libraries there with the pose-extractor CLI, then pick a library "
                "and pose from the dropdowns. Use the ⚙ cycle control to step through "
                "poses (fixed / next / random)."
            ),
            inputs=[
                IO.Combo.Input(
                    "library",
                    options=libraries,
                    tooltip=(
                        "Pose library folder inside models/pose-pilot/. "
                        "Populated at startup; add new libraries with the pose-extractor CLI."
                    ),
                ),
                IO.Combo.Input(
                    "pose_file",
                    options=[""],
                    control_after_generate=True,
                    tooltip=(
                        "GLB filename to load. Auto-populated when a library is selected. "
                        "Use the ⚙ cycle mode to step through poses."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="glb_path",
                    tooltip="Absolute path to the source GLB in models/pose-pilot/.",
                ),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        import folder_paths as _fp
        import torch

        library  = kwargs.get("library",  "").strip()
        pose_file = kwargs.get("pose_file", "").strip()

        pose_pilot_dir = Path(_fp.models_dir) / "pose-pilot"

        if not library or library == "":
            raise ValueError(
                "No library selected. "
                "Add a library to models/pose-pilot/ using the pose-extractor CLI, "
                "then reload ComfyUI."
            )

        lib_dir = pose_pilot_dir / library
        if not lib_dir.is_dir():
            raise FileNotFoundError(f"Library folder not found: {lib_dir}")

        if not pose_file or pose_file == "":
            raise ValueError(
                "No pose selected. "
                "Set the library first so the pose dropdown populates."
            )

        glb_src = lib_dir / "poses" / pose_file
        if not glb_src.exists():
            raise FileNotFoundError(f"GLB not found: {glb_src}")

        # Copy to a fixed name in input/3d/ for the Load3D browser widget.
        # ComfyUI's /view endpoint only serves input/, output/, and temp/ — files in
        # models/ cannot be served to the browser directly, so this copy is required.
        # It is always one file (overwritten), never accumulates.
        input_3d = Path(_fp.get_input_directory()) / "3d"
        input_3d.mkdir(exist_ok=True)
        shutil.copy2(str(glb_src), str(input_3d / "posepilot_current.glb"))

        # Thumbnail preview
        pose_id = glb_src.stem
        png_path = lib_dir / "poses" / f"{pose_id}.png"
        if png_path.exists():
            from PIL import Image
            img = Image.open(str(png_path)).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            thumb_tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            thumb_tensor = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        # Output the SOURCE path so per-pose .views.json sidecars work correctly.
        return IO.NodeOutput(
            str(glb_src),
            ui=UI.PreviewImage(thumb_tensor, cls=cls),
        )

    process = execute  # legacy ComfyUI compatibility

# ComfyUI-PosePilot

A ComfyUI custom node pack for loading and positioning 3D SMPL-X pose models in a
viewport and exporting OpenPose skeleton images for use with ControlNet.

Designed to work alongside the
[pose-extractor](https://github.com/gregvinyard/PosePilot) CLI tool, which builds
pose libraries from source images.

---

## Nodes

### Pose Pilot: Select Pose

Browses a pose library and selects a GLB for the Viewfinder.

- **library** — dropdown of libraries found in `<ComfyUI>/models/pose-pilot/`
- **pose_file** — dropdown of poses within the selected library; supports the ⚙ cycle
  control (fixed / next / random) for stepping through poses like a seed

Outputs an absolute `glb_path` string wired into the Viewfinder node.

### Pose Pilot: Viewfinder

Load3D-style 3D viewport with server-side OpenPose rendering.

- Position the model in the 3D viewer, then execute to get the OpenPose skeleton image
- OpenPose rendering is done server-side from the actual joint hierarchy (forward
  kinematics on the SMPL-X mesh), not from a 2D projection estimate
- Outputs: **openpose** image, **normal map**, **shaded render**, **camera info**,
  **mask**, and the **model path**

#### Favorite Views

Save and restore named camera positions on a per-pose basis:

1. Position the camera in the viewport and execute the node once to capture it
2. Type an optional label in **view_label** (e.g. `front`, `side`)
3. Click **Save View** — the view appears in the **view_preset** dropdown
4. Select a saved preset → the OpenPose output uses that camera on next execute
5. Click **Update View (execute to apply)** to push the saved camera back into the
   viewport widget
6. Click **Delete View** to remove the currently selected preset

Favorite views are stored as `<pose_id>.views.json` sidecar files next to each GLB in
the library, so they travel with the library and are not lost if you move it.

---

## Installation

### Via ComfyUI Manager

Search for **ComfyUI-PosePilot** in the ComfyUI Manager and install.

### Manual

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/gregvinyard/ComfyUI-PosePilot.git
pip install -r ComfyUI-PosePilot/requirements.txt
```

Restart ComfyUI.

---

## Setup

### 1. Build a pose library

Use the [pose-extractor](https://github.com/gregvinyard/PosePilot) CLI to extract
SMPL-X poses from images and write them directly into the canonical library location:

```powershell
uv run pose-extractor ingest <image_folder> `
    --library-root "C:\path\to\ComfyUI\models\pose-pilot" `
    --library-name my_poses `
    --backend hybrid
```

ComfyUI-PosePilot automatically scans `<ComfyUI>/models/pose-pilot/` for libraries on
startup. Any subdirectory containing a `manifest.json` appears in the **library**
dropdown after reloading ComfyUI.

### 2. Wire up the workflow

```
[Pose Pilot: Select Pose]
        glb_path  ──────────────────────────────────────────┐
                                                             ▼
                                             [Pose Pilot: Viewfinder]
                                                     openpose ──► [ControlNet]
```

Set `model_file` in the Viewfinder to `3d/posepilot_current.glb` once — the Selector
updates the underlying file automatically when you change poses, so the Viewfinder
combo never needs to change.

---

## Requirements

- ComfyUI (recent build with `comfy_api.latest` support)
- `pygltflib >= 1.16.0` (installed via `requirements.txt`)
- A pose library built with the pose-extractor CLI (SMPL-X skinned GLB format)

---

## Compatibility

Tested on Windows 11 with ComfyUI running on an NVIDIA RTX 5080. The nodes are
CPU-only at runtime (all heavy lifting is done by the pose-extractor offline); no GPU
is required inside ComfyUI.

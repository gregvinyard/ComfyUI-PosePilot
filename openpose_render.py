"""
Project SMPL-X FK joints to screen space and render an OpenPose skeleton image.

Coordinate conventions:
- GLB joints are in the Rx(180°)-flipped frame: head has lower Y than hips,
  but the flip is self-consistent so the camera_info from the Load3D viewer
  (which sees the same GLB) will project them correctly.
- camera_info format: {position:{x,y,z}, target:{x,y,z}, zoom:float, cameraType:str}
- three.js PerspectiveCamera: FOV=35°, zoom scales the frustum (NOT the FOV angle).
  top = near * tan(FOV/2) / zoom  →  f_y = zoom / tan(FOV/2)
"""
from __future__ import annotations

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# SMPL-X (55-joint) → OpenPose COCO-18 mapping
#
# SMPL-X joint indices used here (Mixamo names from glb_export.py):
#   0  Hips            12 Neck         16 LeftArm (shoulder ball)
#   1  LeftUpLeg       13 LeftShoulder 17 RightArm (shoulder ball)
#   2  RightUpLeg      14 RightShoulder 18 LeftForeArm (elbow)
#   4  LeftLeg (knee)  15 Head         19 RightForeArm (elbow)
#   5  RightLeg (knee) 20 LeftHand (wrist)
#   7  LeftFoot        21 RightHand (wrist)
#   8  RightFoot       23 LeftEye
#  10  LeftToeBase     24 RightEye
#  11  RightToeBase    25-39 left hand fingers
#                      40-54 right hand fingers
#
# OpenPose COCO-18:
#   0 Nose  1 Neck  2 RShoulder  3 RElbow  4 RWrist
#   5 LShoulder  6 LElbow  7 LWrist
#   8 RHip  9 RKnee  10 RAnkle
#  11 LHip  12 LKnee  13 LAnkle
#  14 REye  15 LEye  16 REar  17 LEar
# ---------------------------------------------------------------------------

# Joint indices from SMPL-X for the 16 directly-mapped keypoints (skip 0,16,17)
_SMPLX_FOR_OP = {
    1:  12,   # Neck
    2:  17,   # RShoulder → RightArm ball
    3:  19,   # RElbow
    4:  21,   # RWrist
    5:  16,   # LShoulder → LeftArm ball
    6:  18,   # LElbow
    7:  20,   # LWrist
    8:  2,    # RHip
    9:  5,    # RKnee
    10: 8,    # RAnkle
    11: 1,    # LHip
    12: 4,    # LKnee
    13: 7,    # LAnkle
    14: 24,   # REye
    15: 23,   # LEye
}

# Face joint indices (into the 55-joint array)
#   22: jaw  23: LeftEye  24: RightEye
_FACE_JOINTS = [22, 23, 24]
_FACE_EYE_L, _FACE_EYE_R, _FACE_JAW = 23, 24, 22

# Hand finger chains: (wrist_joint_idx, [proximal, mid, distal])
# Left hand  — wrist = joint 20 (LeftHand)
_LEFT_HAND_CHAINS = [
    (20, [25, 26, 27]),   # Index
    (20, [28, 29, 30]),   # Middle
    (20, [31, 32, 33]),   # Pinky
    (20, [34, 35, 36]),   # Ring
    (20, [37, 38, 39]),   # Thumb
]
# Right hand — wrist = joint 21 (RightHand)
_RIGHT_HAND_CHAINS = [
    (21, [40, 41, 42]),   # Index
    (21, [43, 44, 45]),   # Middle
    (21, [46, 47, 48]),   # Pinky
    (21, [49, 50, 51]),   # Ring
    (21, [52, 53, 54]),   # Thumb
]
# Per-finger BGR colors (Index, Middle, Pinky, Ring, Thumb)
_FINGER_COLORS_BGR = [
    (0,  165, 255),   # orange
    (0,  255,  0),    # green
    (255,  0,  0),    # blue
    (0,  255, 255),   # yellow
    (255,  0, 255),   # magenta
]

# Limb pairs (a, b) in OpenPose-18 index space
_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]

# Per-limb BGR colors (matches standard OpenPose palette)
_LIMB_COLORS_BGR = [
    (85, 0, 255), (85, 0, 255), (255, 85, 0), (255, 170, 0),
    (85, 0, 255), (85, 255, 0), (0, 255, 0),
    (170, 0, 255), (255, 0, 255), (255, 0, 170),
    (0, 0, 255), (0, 85, 255), (0, 170, 255),
    (0, 255, 255), (0, 255, 170),
    (0, 255, 255), (0, 255, 85),
]

# Per-keypoint BGR colors
_KP_COLORS_BGR = [
    (85, 0, 255), (0, 0, 255), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (0, 0, 255), (0, 85, 255), (0, 170, 255),
    (0, 255, 255), (0, 255, 255), (0, 255, 170), (0, 255, 85),
]


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build a 4×4 view matrix from eye position and look-at target."""
    up = np.array([0.0, 1.0, 0.0])
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    norm_r = np.linalg.norm(r)
    if norm_r < 1e-8:
        # Camera looking straight up/down — use Z as up
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        norm_r = np.linalg.norm(r)
    r /= norm_r
    u = np.cross(r, f)
    return np.array([
        [ r[0],  r[1],  r[2], -np.dot(r, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [0,      0,     0,     1              ],
    ], dtype=np.float64)


def project_points(
    pts3d: np.ndarray,
    camera_info: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Project N 3-D world points to (N, 2) pixel coordinates.

    Uses the same projection as three.js Load3D viewer:
      - Perspective: FOV=35°, zoom scales frustum (f_y = zoom/tan(FOV/2))
      - Orthographic: frustum height = 10/zoom units

    Returns float32 (N, 2) pixel coords; points behind camera have large z after NDC.
    """
    pos = camera_info["position"]
    tgt = camera_info["target"]
    zoom = float(camera_info.get("zoom", 1.0))
    cam_type = camera_info.get("cameraType", "perspective")

    # The GLB was exported with an Rx(180°) flip that negates Y (and Z) relative to
    # the original SMPL-X Y-up convention.  Negate Y in both the world points and the
    # camera to restore standard Y-up before projecting, so head appears at the top.
    pts3d = pts3d.copy()
    pts3d[:, 1] = -pts3d[:, 1]

    eye    = np.array([pos["x"], -pos["y"], pos["z"]], dtype=np.float64)
    target = np.array([tgt["x"], -tgt["y"], tgt["z"]], dtype=np.float64)

    V = _look_at(eye, target)

    aspect = width / height
    near, far = 0.01, 10000.0

    if cam_type == "perspective":
        fov_rad = np.radians(35.0)
        f_y = zoom / np.tan(fov_rad / 2.0)
        f_x = f_y / aspect
        P = np.array([
            [f_x, 0,   0,                        0                      ],
            [0,   f_y, 0,                        0                      ],
            [0,   0,  -(far + near) / (far - near), -2*far*near / (far - near)],
            [0,   0,  -1,                        0                      ],
        ], dtype=np.float64)
    else:  # orthographic
        fh = 10.0 / zoom
        fw = fh * aspect
        P = np.array([
            [2/fw, 0,    0,                        0                    ],
            [0,    2/fh, 0,                        0                    ],
            [0,    0,   -2 / (far - near),         -(far + near) / (far - near)],
            [0,    0,    0,                         1                   ],
        ], dtype=np.float64)

    N = len(pts3d)
    p4 = np.ones((N, 4), dtype=np.float64)
    p4[:, :3] = pts3d
    clip = (P @ V @ p4.T).T  # (N, 4)

    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, :3] / w

    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return np.stack([px, py], axis=1).astype(np.float32)


def build_openpose_18(
    joint_positions: np.ndarray,
    camera_info: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Build 18 OpenPose keypoints (pixel coords) from 55 SMPL-X world positions.

    Nose  (index 0): midpoint of eyes
    REar  (index 16): offset outward from RightEye along eye-to-eye axis × 0.9
    LEar  (index 17): offset outward from LeftEye  along eye-to-eye axis × 0.9

    Returns float32 (18, 2).
    """
    px = project_points(joint_positions, camera_info, width, height)  # (55, 2)

    # Synthesise nose / ears in world space, then project
    leye_w = joint_positions[23]
    reye_w = joint_positions[24]
    eye_diff = reye_w - leye_w                        # points right (subject POV)
    nose_w  = (leye_w + reye_w) * 0.5
    rear_w  = reye_w + eye_diff * 0.9
    lear_w  = leye_w - eye_diff * 0.9

    synth = project_points(np.stack([nose_w, rear_w, lear_w]), camera_info, width, height)
    nose_px, rear_px, lear_px = synth[0], synth[1], synth[2]

    kp = np.zeros((18, 2), dtype=np.float32)
    kp[0]  = nose_px
    kp[16] = rear_px
    kp[17] = lear_px
    for op_i, smplx_i in _SMPLX_FOR_OP.items():
        kp[op_i] = px[smplx_i]

    return kp


def render_openpose(
    joint_positions: np.ndarray,
    camera_info: dict,
    width: int,
    height: int,
    include_face: bool = False,
    include_hands: bool = False,
) -> np.ndarray:
    """
    Render a strict OpenPose-18 skeleton image (black background, BGR uint8).

    Optionally overlays additional face and hand keypoints.

    Returns: (height, width, 3) uint8 BGR numpy array.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    kp = build_openpose_18(joint_positions, camera_info, width, height)

    radius    = max(2, int(min(width, height) * 0.012))
    thickness = max(2, int(min(width, height) * 0.006))

    # Draw limbs
    for i, (a, b) in enumerate(_LIMBS):
        ax, ay = int(kp[a][0]), int(kp[a][1])
        bx, by = int(kp[b][0]), int(kp[b][1])
        if (0 <= ax < width and 0 <= ay < height and
                0 <= bx < width and 0 <= by < height):
            color = _LIMB_COLORS_BGR[i]
            cv2.line(canvas, (ax, ay), (bx, by), color, thickness)

    # Draw keypoints
    for i, (x, y) in enumerate(kp):
        xi, yi = int(x), int(y)
        if 0 <= xi < width and 0 <= yi < height:
            cv2.circle(canvas, (xi, yi), radius, _KP_COLORS_BGR[i], -1)

    # --- optional face keypoints ---
    if include_face:
        # Project jaw + both eyes
        fi = project_points(joint_positions[[_FACE_EYE_L, _FACE_EYE_R, _FACE_JAW]],
                            camera_info, width, height)
        leye_px = (int(fi[0][0]), int(fi[0][1]))
        reye_px = (int(fi[1][0]), int(fi[1][1]))
        jaw_px  = (int(fi[2][0]), int(fi[2][1]))
        fr = max(2, radius // 2)

        def _in(pt): return 0 <= pt[0] < width and 0 <= pt[1] < height

        # Draw connecting lines to suggest face shape (eye–eye + each eye to jaw)
        if _in(leye_px) and _in(reye_px):
            cv2.line(canvas, leye_px, reye_px, (200, 200, 200), max(1, thickness // 2))
        if _in(leye_px) and _in(jaw_px):
            cv2.line(canvas, leye_px, jaw_px,  (200, 200, 200), max(1, thickness // 2))
        if _in(reye_px) and _in(jaw_px):
            cv2.line(canvas, reye_px, jaw_px,  (200, 200, 200), max(1, thickness // 2))

        # Draw eye and jaw dots
        for pt, color in [(leye_px, (255, 255, 255)),   # white: left eye
                          (reye_px, (255, 255, 255)),   # white: right eye
                          (jaw_px,  (180, 180, 180))]:  # gray: jaw
            if _in(pt):
                cv2.circle(canvas, pt, fr, color, -1)

    # --- optional hand keypoints: full finger-chain skeletons ---
    if include_hands:
        # Pre-project all 55 joints (we need wrists 20/21 too)
        all_px = project_points(joint_positions, camera_info, width, height)
        hr = max(2, radius // 2)
        ht = max(1, thickness // 2)

        def _draw_chain(wrist_idx, chain, color):
            pts = [wrist_idx] + chain
            for seg_a, seg_b in zip(pts, pts[1:]):
                ax, ay = int(all_px[seg_a][0]), int(all_px[seg_a][1])
                bx, by = int(all_px[seg_b][0]), int(all_px[seg_b][1])
                if (0 <= ax < width and 0 <= ay < height and
                        0 <= bx < width and 0 <= by < height):
                    cv2.line(canvas, (ax, ay), (bx, by), color, ht)
            for j in chain:
                xi, yi = int(all_px[j][0]), int(all_px[j][1])
                if 0 <= xi < width and 0 <= yi < height:
                    cv2.circle(canvas, (xi, yi), hr, color, -1)

        for i, (wrist, chain) in enumerate(_LEFT_HAND_CHAINS):
            _draw_chain(wrist, chain, _FINGER_COLORS_BGR[i])
        for i, (wrist, chain) in enumerate(_RIGHT_HAND_CHAINS):
            _draw_chain(wrist, chain, _FINGER_COLORS_BGR[i])

    return canvas  # BGR uint8


def bgr_to_tensor(img: np.ndarray) -> "torch.Tensor":
    """(H, W, 3) BGR uint8 → (1, H, W, 3) float32 RGB [0, 1] tensor."""
    import torch
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
    return t

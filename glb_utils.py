"""
Load a PosePilot skinned GLB and compute FK joint world positions.
"""
from __future__ import annotations

import numpy as np
from pygltflib import GLTF2

NUM_JOINTS = 55


def _quat_to_rot4(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] → 4×4 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),     0],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),     0],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y), 0],
        [0,                 0,                  0,                  1],
    ], dtype=np.float64)


def load_glb_joints(glb_path: str) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Walk the 55-joint hierarchy in a PosePilot GLB and return world positions.

    Returns:
        joint_positions: (55, 3) float32 world-space XYZ
        parents:         list[int], length 55, -1 for root
        names:           list[str], length 55, joint names
    """
    gltf = GLTF2().load(glb_path)

    local_t = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    local_q = np.zeros((NUM_JOINTS, 4), dtype=np.float64)  # [x,y,z,w]
    names: list[str] = []

    for j in range(NUM_JOINTS):
        node = gltf.nodes[j]
        local_t[j] = node.translation if node.translation else [0.0, 0.0, 0.0]
        local_q[j] = node.rotation    if node.rotation    else [0.0, 0.0, 0.0, 1.0]
        names.append(node.name or f"joint_{j}")

    # Recover parent array from children lists
    parents = [-1] * NUM_JOINTS
    for j in range(NUM_JOINTS):
        children = gltf.nodes[j].children or []
        for c in children:
            if c < NUM_JOINTS:
                parents[c] = j

    # Forward kinematics: world_T[j] = world_T[parent] @ T_local @ R_local
    world_T = np.tile(np.eye(4, dtype=np.float64), (NUM_JOINTS, 1, 1))
    for j in range(NUM_JOINTS):
        p = parents[j]
        parent_world = world_T[p] if p >= 0 else np.eye(4, dtype=np.float64)
        T_local = np.eye(4, dtype=np.float64)
        T_local[:3, 3] = local_t[j]
        R_local = _quat_to_rot4(local_q[j])
        world_T[j] = parent_world @ T_local @ R_local

    joint_positions = world_T[:, :3, 3].astype(np.float32)
    return joint_positions, parents, names

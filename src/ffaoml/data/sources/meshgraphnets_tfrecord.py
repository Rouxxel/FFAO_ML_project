"""
#############################################################################
### MeshGraphNets TFRecord parsing
###
### @file meshgraphnets_tfrecord.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Decode one trajectory from DeepMind ``cylinder_flow`` TFRecords using ``meta.json``.
TensorFlow is required at runtime (optional ``mesh`` extra / local install).
"""

# Native imports
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np

"""META-----------------------------------------------------------"""

# Built-in cylinder_flow ``meta.json`` field layout (trajectory_length may differ).
CYLINDER_FLOW_META: dict[str, Any] = {
    "trajectory_length": 600,
    "field_names": ["mesh_pos", "node_type", "cells", "velocity", "pressure"],
    "features": {
        "mesh_pos": {"type": "static", "shape": [1, None, 2], "dtype": "float32"},
        "node_type": {"type": "static", "shape": [1, None, 1], "dtype": "int32"},
        "cells": {"type": "static", "shape": [1, None, 3], "dtype": "int32"},
        "velocity": {"type": "dynamic", "shape": [None, None, 2], "dtype": "float32"},
        "pressure": {"type": "dynamic", "shape": [None, None, 1], "dtype": "float32"},
    },
}


def load_meta(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``meta.json`` from disk or return the built-in cylinder_flow template."""
    if path is None:
        return dict(CYLINDER_FLOW_META)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to read MeshGraphNets TFRecords. "
            "Install tensorflow or use the numpy fixture under tests/fixtures/."
        ) from exc
    return tf


def parse_tfrecord_example(
    serialized: bytes, meta: dict[str, Any]
) -> dict[str, np.ndarray]:
    """
    Parse a single TFRecord example into numpy arrays.

    Static fields are squeezed to (N, …); dynamic fields are (T, N, C).
    """
    tf = _require_tensorflow()
    field_names = list(meta["field_names"])
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in field_names}
    parsed = tf.io.parse_single_example(serialized, feature_lists)
    return _parse_cylinder_flow_heuristic(parsed, meta, tf)


def _parse_cylinder_flow_heuristic(
    parsed: Any,
    meta: dict[str, Any],
    tf: Any,
) -> dict[str, np.ndarray]:
    """Decode fields using sizes inferred from velocity (T, N, 2)."""
    traj_len = int(meta.get("trajectory_length", 600))

    def _decode(key: str, dtype: str) -> np.ndarray:
        raw = parsed[key].values
        return tf.io.decode_raw(raw, getattr(tf, dtype)).numpy()

    vel_flat = _decode("velocity", "float32")
    n_nodes = vel_flat.size // (traj_len * 2)
    velocity = vel_flat.reshape(traj_len, n_nodes, 2)

    pres_flat = _decode("pressure", "float32")
    pressure = pres_flat.reshape(traj_len, n_nodes, 1)

    pos_flat = _decode("mesh_pos", "float32")
    mesh_pos = pos_flat.reshape(n_nodes, 2)

    nt_flat = _decode("node_type", "int32")
    node_type = nt_flat.reshape(n_nodes)

    cells_flat = _decode("cells", "int32")
    n_cells = cells_flat.size // 3
    cells = cells_flat.reshape(n_cells, 3)

    return {
        "mesh_pos": mesh_pos.astype(np.float32),
        "node_type": node_type.astype(np.int32),
        "cells": cells.astype(np.int32),
        "velocity": velocity.astype(np.float32),
        "pressure": pressure.astype(np.float32),
    }


def read_first_tfrecord_example(
    tfrecord_path: str | Path,
    *,
    meta_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Read and parse the first serialized example from a ``.tfrecord`` file."""
    tf = _require_tensorflow()
    meta = load_meta(meta_path)
    path = str(tfrecord_path)
    dataset = tf.data.TFRecordDataset([path])
    for raw in dataset.take(1):
        return parse_tfrecord_example(bytes(raw.numpy()), meta)
    raise ValueError(f"No records in {tfrecord_path}")


def describe_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Human-readable summary of parsed field shapes and dtypes."""
    lines = []
    for key in sorted(arrays):
        arr = arrays[key]
        lines.append(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
    return "\n".join(lines)

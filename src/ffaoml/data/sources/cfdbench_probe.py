"""
#############################################################################
### CFDBench upstream layout probe (Stage 3)
###
### @file cfdbench_probe.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Read one interpolated CFDBench ``case*`` directory (``u.npy``, ``v.npy``, ``case.json``)
without importing into project Zarr. Used by ``scripts/inspect_cfdbench_sample.py``.
"""

# Native imports
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np

# Project imports
from ffaoml.contracts import (
    CFDBENCH_CASE_PARAMS_FILE,
    CFDBENCH_UPSTREAM_VELOCITY_X,
    CFDBENCH_UPSTREAM_VELOCITY_Y,
)

"""DISCOVERY-----------------------------------------------------------"""

CYLINDER_SUBSETS: tuple[str, ...] = ("prop", "bc", "geo")


def is_cfdbench_case_dir(path: Path) -> bool:
    """Return True when ``path`` looks like a CFDBench ``case*`` folder."""
    return (
        path.is_dir()
        and (path / CFDBENCH_UPSTREAM_VELOCITY_X).is_file()
        and (path / CFDBENCH_UPSTREAM_VELOCITY_Y).is_file()
    )


def find_first_cylinder_case(data_root: Path) -> Path:
    """
    Locate the first ``cylinder/<subset>/case*`` directory under ``data_root``.

    Parameters:
        data_root (Path): CFDBench ``data/`` root (contains ``cylinder/``).

    Returns:
        Path: Case directory with ``u.npy`` and ``v.npy``.

    Raises:
        FileNotFoundError: When no case directory exists.
    """
    root = Path(data_root)
    if is_cfdbench_case_dir(root):
        return root
    if root.name == "cylinder":
        cylinder = root
    elif (root / "cylinder").is_dir():
        cylinder = root / "cylinder"
    else:
        cylinder = None
    if cylinder is None:
        raise FileNotFoundError(f"No cylinder/ tree under {data_root}")
    for subset in CYLINDER_SUBSETS:
        subset_dir = cylinder / subset
        if not subset_dir.is_dir():
            continue
        for case_dir in sorted(subset_dir.glob("case*")):
            if is_cfdbench_case_dir(case_dir):
                return case_dir
    raise FileNotFoundError(
        f"No CFDBench cylinder case under {data_root} "
        f"(expected cylinder/{{prop,bc,geo}}/case*/u.npy)"
    )


"""LOAD-----------------------------------------------------------"""


def load_cfdbench_case(case_dir: Path) -> dict[str, Any]:
    """
    Load velocity arrays and ``case.json`` from one CFDBench case folder.

    Parameters:
        case_dir (Path): Directory containing ``u.npy``, ``v.npy``, and ``case.json``.

    Returns:
        dict: Keys ``u``, ``v``, ``params``, ``case_dir``, ``re_estimate``.
    """
    case_path = Path(case_dir)
    if not is_cfdbench_case_dir(case_path):
        raise FileNotFoundError(
            f"Missing {CFDBENCH_UPSTREAM_VELOCITY_X} or "
            f"{CFDBENCH_UPSTREAM_VELOCITY_Y} in {case_path}"
        )
    u = np.load(case_path / CFDBENCH_UPSTREAM_VELOCITY_X)
    v = np.load(case_path / CFDBENCH_UPSTREAM_VELOCITY_Y)
    params_path = case_path / CFDBENCH_CASE_PARAMS_FILE
    params: dict[str, Any] = {}
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))
    return {
        "u": u,
        "v": v,
        "params": params,
        "case_dir": case_path,
        "re_estimate": estimate_reynolds_from_params(params),
    }


def estimate_reynolds_from_params(params: dict[str, Any]) -> float | None:
    """
    Compute Re = ρ U D / μ when CFDBench prop fields are present.

    Uses ``vel_in`` (or ``u_in``), ``density``, ``viscosity``, and ``radius``
    (diameter = 2 × radius) when available.
    """
    if not params:
        return None
    vel = params.get("vel_in", params.get("u_in"))
    rho = params.get("density")
    mu = params.get("viscosity")
    radius = params.get("radius")
    if vel is None or rho is None or mu is None or radius is None:
        re = params.get("re")
        return float(re) if re is not None else None
    diameter = 2.0 * float(radius)
    mu_val = float(mu)
    if mu_val == 0.0:
        return None
    return float(rho) * float(vel) * diameter / mu_val


def describe_cfdbench_case(case_dir: Path) -> str:
    """Human-readable summary of shapes, grid size, and parameters."""
    loaded = load_cfdbench_case(case_dir)
    u = loaded["u"]
    v = loaded["v"]
    params = loaded["params"]
    re_est = loaded["re_estimate"]
    lines = [
        f"case_dir: {case_dir}",
        f"u ({CFDBENCH_UPSTREAM_VELOCITY_X}): shape={u.shape} dtype={u.dtype}",
        f"v ({CFDBENCH_UPSTREAM_VELOCITY_Y}): shape={v.shape} dtype={v.dtype}",
        f"case.json keys: {sorted(params.keys()) if params else '(missing)'}",
    ]
    if u.ndim == 3 and v.ndim == 3:
        lines.append(f"grid: T={u.shape[0]}, ny={u.shape[1]}, nx={u.shape[2]}")
    if re_est is not None:
        lines.append(f"Re (estimate from case.json): {re_est:.4g}")
    lines.append(
        "project map: u -> velocity_x, v -> velocity_y; "
        "pressure/vorticity filled at import (see CONTRACTS.md)"
    )
    return "\n".join(lines)

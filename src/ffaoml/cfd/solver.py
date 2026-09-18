"""CFD solver adapter interface and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf

from ffaoml.cfd.types import SimulationCase, SimulationResult
from ffaoml.physics.reynolds import reynolds_number


class Solver(Protocol):
    """Export time-resolved 2D fields in the shared Zarr layout."""

    name: str

    def run(self, case: SimulationCase) -> SimulationResult: ...


def case_from_config(
    cfg: DictConfig,
    *,
    sim_id: str,
    output_root: str | Path,
    split: str = "train",
) -> SimulationCase:
    """Build a ``SimulationCase`` from Hydra ``simulation`` (and paths) groups."""
    sim = cfg.simulation
    n_steps = int(round(float(sim.simulation_time) / float(sim.time_step)))
    n_steps = max(n_steps, 1)
    return SimulationCase(
        sim_id=sim_id,
        output_root=Path(output_root),
        re=float(
            cfg.dataset.re
            if OmegaConf.select(cfg, "dataset.re") is not None
            else reynolds_number(
                float(sim.inlet_velocity),
                float(sim.cylinder_diameter),
                float(sim.viscosity),
            )
        ),
        u_inlet=float(sim.inlet_velocity),
        nu=float(sim.viscosity),
        diameter=float(sim.cylinder_diameter),
        nx=int(sim.mesh_resolution_x),
        ny=int(sim.mesh_resolution_y),
        dt=float(sim.time_step),
        n_steps=n_steps,
        seed=int(cfg.get("seed", 0)),
        domain_width=float(sim.domain_width),
        domain_height=float(sim.domain_height),
        split=split,
    )


def get_solver(backend: str) -> Solver:
    """Return a solver implementation by ``simulation.backend`` name."""
    from ffaoml.cfd.adapters.openfoam import OpenFoamSolver
    from ffaoml.cfd.dedalus_backend import DedalusSolver
    from ffaoml.cfd.fd_backend import FiniteDifferenceSolver
    from ffaoml.cfd.stub_backend import StubSolver

    key = backend.lower().strip()
    registry: dict[str, Solver] = {
        "stub": StubSolver(),
        "fd": FiniteDifferenceSolver(),
        "dedalus": DedalusSolver(),
        "openfoam": OpenFoamSolver(),
    }
    if key not in registry:
        options = ", ".join(sorted(registry))
        raise ValueError(f"unknown CFD backend '{backend}'; options: {options}")
    return registry[key]


def run_from_config(
    cfg: DictConfig,
    *,
    sim_id: str,
    output_root: str | Path,
) -> SimulationResult:
    """Convenience: compose case + factory + run."""
    backend = str(cfg.simulation.backend)
    case = case_from_config(cfg, sim_id=sim_id, output_root=output_root)
    return get_solver(backend).run(case)


def config_to_plain(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

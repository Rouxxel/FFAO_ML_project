"""Synthetic fields for schema / pipeline tests (not physical CFD)."""

from __future__ import annotations

import numpy as np

from ffaoml.cfd.types import SimulationCase, SimulationResult
from ffaoml.data.io import write_field_store


class StubSolver:
    name = "stub"

    def run(self, case: SimulationCase) -> SimulationResult:
        rng = np.random.default_rng(case.seed)
        ny, nx = case.ny, case.nx
        time = np.arange(case.n_steps, dtype=float) * case.dt
        y = np.linspace(0.0, case.domain_height, ny, dtype=float)
        x = np.linspace(0.0, case.domain_width, nx, dtype=float)
        dx = case.domain_width / max(nx - 1, 1)
        dy = case.domain_height / max(ny - 1, 1)

        yy, xx = np.meshgrid(y, x, indexing="ij")
        cylinder = (xx - case.domain_width * 0.2) ** 2 + (
            yy - case.domain_height * 0.5
        ) ** 2
        solid_mask = cylinder < (case.diameter * 0.5) ** 2

        base_omega = np.sin(2.0 * np.pi * xx / case.domain_width) * np.cos(
            2.0 * np.pi * yy / case.domain_height
        )
        omega = np.stack(
            [
                base_omega * np.cos(0.4 * t) + 0.05 * rng.standard_normal((ny, nx))
                for t in time
            ],
            axis=0,
        )
        omega = np.where(solid_mask, 0.0, omega)

        velocity_x = np.stack(
            [1.0 - 0.1 * np.sin(0.3 * t) * base_omega for t in time],
            axis=0,
        )
        velocity_y = np.stack(
            [0.05 * np.cos(0.3 * t) * base_omega for t in time],
            axis=0,
        )
        velocity_x = np.where(solid_mask, 0.0, velocity_x)
        velocity_y = np.where(solid_mask, 0.0, velocity_y)
        pressure = np.stack(
            [0.1 * np.sin(0.2 * t) * base_omega for t in time],
            axis=0,
        )

        attrs = {
            "solver": self.name,
            "re": case.re,
            "u_inlet": case.u_inlet,
            "nu": case.nu,
            "diameter": case.diameter,
            "dt": case.dt,
            "n_steps": case.n_steps,
            "dx": dx,
            "dy": dy,
            "seed": case.seed,
        }
        store_path = write_field_store(
            case.simulation_dir,
            time=time,
            y=y,
            x=x,
            fields={
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "pressure": pressure,
                "vorticity": omega,
            },
            attrs=attrs,
            solid_mask=solid_mask,
        )
        metadata = {
            "sim_id": case.sim_id,
            "re": case.re,
            "split": case.split,
            "path": case.relative_path,
            "u_inlet": case.u_inlet,
            "nu": case.nu,
            "diameter": case.diameter,
            "nx": case.nx,
            "ny": case.ny,
            "dt": case.dt,
            "n_steps": case.n_steps,
            "seed": case.seed,
            "solver": self.name,
        }
        return SimulationResult(case=case, store_path=store_path, metadata=metadata)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import time

from .mac_grid import (
    PeriodicMACGrid,
    coarsen_average,
    divergence,
    interp_xy_for_u,
    interp_xy_for_v,
    interp_x,
    interp_y,
    kinetic_energy,
    roll_x,
    roll_y,
)


SchemeName = Literal["standard_upwind", "kep_skew"]


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 128
    ny: int = 128
    length: float = 1.0
    viscosity: float = 1e-4
    shear_r: float = 80.0
    perturbation_d: float = 0.05
    final_time: float = 1.0
    cfl: float = 0.45
    scheme: SchemeName = "standard_upwind"
    save_stride: int = 1


@dataclass
class SimulationResult:
    grid: PeriodicMACGrid
    config: SimulationConfig
    times: np.ndarray
    kinetic_energy: np.ndarray
    enstrophy: np.ndarray
    div_linf: np.ndarray
    step_walltime: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray


def initial_condition(grid: PeriodicMACGrid, shear_r: float, perturbation_d: float) -> tuple[np.ndarray, np.ndarray]:
    xu, yu = grid.mesh_u()
    xv, yv = grid.mesh_v()

    u = np.where(
        yu <= 0.5,
        np.tanh(shear_r * (yu - 0.25)),
        np.tanh(shear_r * (0.75 - yu)),
    )
    v = perturbation_d * np.sin(2.0 * np.pi * (xv + 0.25))
    return u.astype(np.float64, copy=False), v.astype(np.float64, copy=False)


def _laplacian_u(field: np.ndarray, grid: PeriodicMACGrid) -> np.ndarray:
    return (
        (roll_x(field, -1) - 2.0 * field + roll_x(field, 1)) / (grid.dx * grid.dx)
        + (roll_y(field, -1) - 2.0 * field + roll_y(field, 1)) / (grid.dy * grid.dy)
    )


def _laplacian_v(field: np.ndarray, grid: PeriodicMACGrid) -> np.ndarray:
    return _laplacian_u(field, grid)


def _convective_standard_upwind(
    u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid
) -> tuple[np.ndarray, np.ndarray]:
    u_xface = interp_x(u)
    u_yface = interp_y(u)
    v_xface = interp_x(v)
    v_yface = interp_y(v)

    v_at_u = interp_xy_for_u(v)
    u_at_v = interp_xy_for_v(u)

    upwind_u_x = np.where(u_xface >= 0.0, u, roll_x(u, -1))
    flux_u_x = u_xface * upwind_u_x
    upwind_u_y = np.where(v_at_u >= 0.0, roll_y(u, 1), u)
    flux_u_y = v_at_u * upwind_u_y

    upwind_v_x = np.where(u_at_v >= 0.0, roll_x(v, 1), v)
    flux_v_x = u_at_v * upwind_v_x
    upwind_v_y = np.where(v_yface >= 0.0, v, roll_y(v, -1))
    flux_v_y = v_yface * upwind_v_y

    conv_u = (flux_u_x - roll_x(flux_u_x, 1)) / grid.dx + (flux_u_y - roll_y(flux_u_y, 1)) / grid.dy
    conv_v = (flux_v_x - roll_x(flux_v_x, 1)) / grid.dx + (flux_v_y - roll_y(flux_v_y, 1)) / grid.dy
    return conv_u, conv_v


def _convective_kep_skew(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid) -> tuple[np.ndarray, np.ndarray]:
    u_xface = interp_x(u)
    u_yface = interp_y(u)
    v_xface = interp_x(v)
    v_yface = interp_y(v)

    v_at_u = interp_xy_for_u(v)
    u_at_v = interp_xy_for_v(u)

    dudx = (roll_x(u, -1) - roll_x(u, 1)) / (2.0 * grid.dx)
    dudy = (roll_y(u, -1) - roll_y(u, 1)) / (2.0 * grid.dy)
    dvdx = (roll_x(v, -1) - roll_x(v, 1)) / (2.0 * grid.dx)
    dvdy = (roll_y(v, -1) - roll_y(v, 1)) / (2.0 * grid.dy)

    adv_u = u * dudx + v_at_u * dudy
    adv_v = u_at_v * dvdx + v * dvdy

    fx_u = u_xface * u_xface
    fy_u = v_at_u * u_yface
    fx_v = u_at_v * v_xface
    fy_v = v_yface * v_yface

    div_u = (fx_u - roll_x(fx_u, 1)) / grid.dx + (fy_u - roll_y(fy_u, 1)) / grid.dy
    div_v = (fx_v - roll_x(fx_v, 1)) / grid.dx + (fy_v - roll_y(fy_v, 1)) / grid.dy

    return 0.5 * (adv_u + div_u), 0.5 * (adv_v + div_v)


def _rhs(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid, viscosity: float, scheme: SchemeName) -> tuple[np.ndarray, np.ndarray]:
    if scheme == "standard_upwind":
        conv_u, conv_v = _convective_standard_upwind(u, v, grid)
    elif scheme == "kep_skew":
        conv_u, conv_v = _convective_kep_skew(u, v, grid)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    diff_u = viscosity * _laplacian_u(u, grid)
    diff_v = viscosity * _laplacian_v(v, grid)
    return -(conv_u) + diff_u, -(conv_v) + diff_v


def _build_poisson_solver(grid: PeriodicMACGrid):
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)
    denom = kx[:, None] ** 2 + ky[None, :] ** 2
    denom[0, 0] = 1.0

    def solve(rhs: np.ndarray) -> np.ndarray:
        rhs_hat = np.fft.fftn(rhs)
        rhs_hat[0, 0] = 0.0
        phi_hat = -rhs_hat / denom
        phi_hat[0, 0] = 0.0
        return np.fft.ifftn(phi_hat).real

    return solve


def _project(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid, solve_poisson, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rhs = divergence(u, v, grid) / dt
    phi = solve_poisson(rhs)
    u_corr = u - dt * (phi - roll_x(phi, 1)) / grid.dx
    v_corr = v - dt * (phi - roll_y(phi, 1)) / grid.dy
    return u_corr, v_corr, phi


def _advance_step(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid, viscosity: float, scheme: SchemeName, dt: float, solve_poisson):
    def stage(state_u: np.ndarray, state_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rhs_u, rhs_v = _rhs(state_u, state_v, grid, viscosity, scheme)
        predictor_u = state_u + dt * rhs_u
        predictor_v = state_v + dt * rhs_v
        return _project(predictor_u, predictor_v, grid, solve_poisson, dt)

    u1, v1, _ = stage(u, v)
    u2, v2, _ = stage(0.75 * u + 0.25 * u1, 0.75 * v + 0.25 * v1)
    u3, v3, phi = stage((1.0 / 3.0) * u + (2.0 / 3.0) * u2, (1.0 / 3.0) * v + (2.0 / 3.0) * v2)
    return u3, v3, phi


def _time_step(grid: PeriodicMACGrid, u: np.ndarray, v: np.ndarray, viscosity: float, cfl: float) -> float:
    umax = float(np.max(np.abs(u)))
    vmax = float(np.max(np.abs(v)))
    adv_x = grid.dx / max(umax, 1e-12)
    adv_y = grid.dy / max(vmax, 1e-12)
    adv_dt = cfl * min(adv_x, adv_y)
    diff_dt = 0.25 * min(grid.dx * grid.dx, grid.dy * grid.dy) / max(viscosity, 1e-16)
    return min(adv_dt, diff_dt)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    grid = PeriodicMACGrid(config.nx, config.ny, config.length)
    u, v = initial_condition(grid, config.shear_r, config.perturbation_d)
    solve_poisson = _build_poisson_solver(grid)

    times = [0.0]
    kinetic = [kinetic_energy(u, v)]
    omega0 = (v - roll_x(v, 1)) / grid.dx - (u - roll_y(u, 1)) / grid.dy
    enst = [0.5 * float(np.mean(omega0 * omega0))]
    div_linf = [float(np.max(np.abs(divergence(u, v, grid))))]
    walltime = [0.0]
    wall_clock = time.perf_counter()
    pressure = np.zeros((grid.nx, grid.ny), dtype=np.float64)

    t = 0.0
    while t < config.final_time - 1e-15:
        dt = _time_step(grid, u, v, config.viscosity, config.cfl)
        dt = min(dt, config.final_time - t)
        u, v, pressure = _advance_step(u, v, grid, config.viscosity, config.scheme, dt, solve_poisson)
        t += dt

        times.append(t)
        kinetic.append(kinetic_energy(u, v))
        omega = (v - roll_x(v, 1)) / grid.dx - (u - roll_y(u, 1)) / grid.dy
        enst.append(0.5 * float(np.mean(omega * omega)))
        div_linf.append(float(np.max(np.abs(divergence(u, v, grid)))))
        walltime.append(time.perf_counter() - wall_clock)

    return SimulationResult(
        grid=grid,
        config=config,
        times=np.asarray(times),
        kinetic_energy=np.asarray(kinetic),
        enstrophy=np.asarray(enst),
        div_linf=np.asarray(div_linf),
        step_walltime=np.asarray(walltime),
        u=u,
        v=v,
        pressure=pressure,
    )


def coarsen_result(result: SimulationResult, factor: int) -> tuple[np.ndarray, np.ndarray]:
    return coarsen_average(result.u, factor), coarsen_average(result.v, factor)

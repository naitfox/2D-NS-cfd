from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicMACGrid:
    nx: int
    ny: int
    length: float = 1.0

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.length / self.ny

    @property
    def x_cell(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y_cell(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def x_u(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx

    @property
    def y_u(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def x_v(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y_v(self) -> np.ndarray:
        return np.arange(self.ny) * self.dy

    def mesh_cell(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_cell, self.y_cell, indexing="ij")

    def mesh_u(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_u, self.y_u, indexing="ij")

    def mesh_v(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_v, self.y_v, indexing="ij")


def roll_x(field: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(field, shift, axis=0)


def roll_y(field: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(field, shift, axis=1)


def interp_x(field: np.ndarray) -> np.ndarray:
    return 0.5 * (field + roll_x(field, -1))


def interp_y(field: np.ndarray) -> np.ndarray:
    return 0.5 * (field + roll_y(field, -1))


def interp_xy_for_u(field: np.ndarray) -> np.ndarray:
    return 0.25 * (
        field
        + roll_x(field, 1)
        + roll_y(field, -1)
        + roll_x(roll_y(field, -1), 1)
    )


def interp_xy_for_v(field: np.ndarray) -> np.ndarray:
    return 0.25 * (
        field
        + roll_x(field, -1)
        + roll_y(field, 1)
        + roll_x(roll_y(field, 1), -1)
    )


def cell_center_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (u + roll_y(u, -1)), 0.5 * (v + roll_x(v, -1))


def vorticity(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid) -> np.ndarray:
    dvdx = (v - roll_x(v, 1)) / grid.dx
    dudy = (u - roll_y(u, 1)) / grid.dy
    return dvdx - dudy


def divergence(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid) -> np.ndarray:
    return (roll_x(u, -1) - u) / grid.dx + (roll_y(v, -1) - v) / grid.dy


def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    return 0.5 * (np.mean(u * u) + np.mean(v * v))


def enstrophy(u: np.ndarray, v: np.ndarray, grid: PeriodicMACGrid) -> float:
    omega = vorticity(u, v, grid)
    return 0.5 * np.mean(omega * omega)


def coarsen_average(field: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return field.copy()
    nx, ny = field.shape
    if nx % factor != 0 or ny % factor != 0:
        raise ValueError("Field shape must be divisible by coarsening factor.")
    reshaped = field.reshape(nx // factor, factor, ny // factor, factor)
    return reshaped.mean(axis=(1, 3))

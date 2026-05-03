from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from .mac_grid import coarsen_average, vorticity
from .simulation import SimulationConfig, run_simulation


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_run(result, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    tag = f"{result.config.scheme}_nx{result.config.nx}_r{int(result.config.shear_r)}_d{result.config.perturbation_d:g}"
    path = output_dir / f"{tag}.npz"
    np.savez_compressed(
        path,
        times=result.times,
        kinetic_energy=result.kinetic_energy,
        enstrophy=result.enstrophy,
        div_linf=result.div_linf,
        u=result.u,
        v=result.v,
        pressure=result.pressure,
        config=json.dumps(asdict(result.config)),
    )
    return path


def _plot_timeseries(results, output_dir: Path, title_suffix: str) -> None:
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for result, label in results:
        axes[0, 0].plot(result.times, result.kinetic_energy, label=label)
        axes[0, 1].plot(result.times, result.enstrophy, label=label)
        axes[1, 0].plot(result.times, np.abs(result.kinetic_energy - result.kinetic_energy[0]) / result.kinetic_energy[0], label=label)
        axes[1, 1].plot(result.times, result.div_linf, label=label)

    axes[0, 0].set_title("Kinetic energy")
    axes[0, 1].set_title("Enstrophy")
    axes[1, 0].set_title("Relative kinetic energy error")
    axes[1, 1].set_title("Divergence $L_\infty$")
    for ax in axes.ravel():
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Time evolution diagnostics {title_suffix}")
    fig.savefig(output_dir / f"timeseries{title_suffix}.png", dpi=200)
    plt.close(fig)


def _plot_ke_error(results, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    for result, label in results:
        ax.plot(result.times, np.abs(result.kinetic_energy - result.kinetic_energy[0]) / result.kinetic_energy[0], label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative kinetic energy error")
    ax.set_title("Kinetic energy conservation error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "figure5_ke_conservation_error.png", dpi=220)
    plt.close(fig)


def _plot_ke_and_enstrophy(results, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for result, label in results:
        axes[0].plot(result.times, result.kinetic_energy, label=label)
        axes[1].plot(result.times, result.enstrophy, label=label)
    axes[0].set_title("Time evolution of kinetic energy")
    axes[1].set_title("Time evolution of enstrophy")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(output_dir / "figure2_3_energy_enstrophy.png", dpi=220)
    plt.close(fig)


def _plot_comparison_vorticity(standard, kep, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    for ax, result, title in [
        (axes[0], standard, "Standard upwind"),
        (axes[1], kep, "KEP skew-symmetric"),
    ]:
        grid = result.grid
        omega = vorticity(result.u, result.v, grid)
        X, Y = grid.mesh_cell()
        contour = ax.contourf(X, Y, omega, levels=25, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.9)
    fig.savefig(output_dir / "figure4_standard_vs_kep_vorticity.png", dpi=220)
    plt.close(fig)


def _plot_shear_thickness(thick, thin, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    for ax, result, title in [
        (axes[0], thick, f"Shear thickness r={int(thick.config.shear_r)}"),
        (axes[1], thin, f"Shear thickness r={int(thin.config.shear_r)}"),
    ]:
        grid = result.grid
        omega = vorticity(result.u, result.v, grid)
        X, Y = grid.mesh_cell()
        contour = ax.contourf(X, Y, omega, levels=25, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.9)
    fig.savefig(output_dir / "figure7_vorticity_thickness_compare.png", dpi=220)
    plt.close(fig)


def _plot_vorticity(result, output_dir: Path, name: str, levels: int = 25) -> None:
    grid = result.grid
    omega = vorticity(result.u, result.v, grid)
    X, Y = grid.mesh_cell()
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    contour = ax.contourf(X, Y, omega, levels=levels, cmap="coolwarm")
    fig.colorbar(contour, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(output_dir / f"{name}.png", dpi=220)
    plt.close(fig)


def _plot_validation(convergence_table, output_dir: Path) -> None:
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    for label, hs, errs in convergence_table:
        ax.loglog(hs, errs, "o-", label=label)
        slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        ax.text(hs[-1], errs[-1], f"{slope:.2f}")
    ax.set_xlabel("Grid spacing h")
    ax.set_ylabel("L2 error in vorticity")
    ax.set_title("Grid convergence / validation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "grid_convergence.png", dpi=220)
    plt.close(fig)


def _l2_error(fine_field: np.ndarray, coarse_field: np.ndarray, factor: int) -> float:
    restricted = coarsen_average(fine_field, factor)
    diff = restricted - coarse_field
    return float(np.sqrt(np.mean(diff * diff)))


def run_all_experiments(base_dir: str | Path = "results") -> dict[str, Path]:
    base = Path(base_dir)
    figures = base / "figures"
    data = base / "data"
    _ensure_dir(figures)
    _ensure_dir(data)

    outputs: dict[str, Path] = {}

    reference_cfg = SimulationConfig(nx=256, ny=256, scheme="kep_skew", final_time=1.0)
    validation_64_standard = SimulationConfig(nx=64, ny=64, scheme="standard_upwind", final_time=1.0)
    validation_64_kep = SimulationConfig(nx=64, ny=64, scheme="kep_skew", final_time=1.0)
    standard_cfg = SimulationConfig(nx=128, ny=128, scheme="standard_upwind", final_time=1.0)
    kep_cfg = SimulationConfig(nx=128, ny=128, scheme="kep_skew", final_time=1.0)
    thick_cfg = SimulationConfig(nx=128, ny=128, shear_r=40.0, scheme="kep_skew", final_time=1.0)
    thin_cfg = SimulationConfig(nx=128, ny=128, shear_r=120.0, scheme="kep_skew", final_time=1.0)

    t0 = time.perf_counter()
    ref = run_simulation(reference_cfg)
    outputs["reference"] = _save_run(ref, data)
    standard64 = run_simulation(validation_64_standard)
    outputs["standard64"] = _save_run(standard64, data)
    kep64 = run_simulation(validation_64_kep)
    outputs["kep64"] = _save_run(kep64, data)
    standard = run_simulation(standard_cfg)
    outputs["standard"] = _save_run(standard, data)
    kep = run_simulation(kep_cfg)
    outputs["kep"] = _save_run(kep, data)
    thick = run_simulation(thick_cfg)
    outputs["thick"] = _save_run(thick, data)
    thin = run_simulation(thin_cfg)
    outputs["thin"] = _save_run(thin, data)
    elapsed = time.perf_counter() - t0

    _plot_ke_and_enstrophy([(standard, "Standard upwind"), (kep, "KEP skew-symmetric")], figures)
    _plot_ke_error([(standard, "Standard upwind"), (kep, "KEP skew-symmetric")], figures)
    _plot_comparison_vorticity(standard, kep, figures)
    _plot_shear_thickness(thick, thin, figures)

    _plot_timeseries([(standard, "Standard upwind"), (kep, "KEP skew-symmetric")], figures, "_standard_vs_kep")
    _plot_vorticity(standard, figures, "figure4_standard_vorticity_t1")
    _plot_vorticity(kep, figures, "figure4_kep_vorticity_t1")
    _plot_vorticity(thick, figures, "figure7_thick_shear_vorticity_t1")
    _plot_vorticity(thin, figures, "figure7_thin_shear_vorticity_t1")

    fine_omega = vorticity(ref.u, ref.v, ref.grid)
    conv_table = []
    for label, results in [
        ("Standard upwind", [standard64, standard]),
        ("KEP skew-symmetric", [kep64, kep]),
    ]:
        hs = np.array([result.grid.dx for result in results])
        errs = np.array([_l2_error(fine_omega, vorticity(result.u, result.v, result.grid), ref.grid.nx // result.grid.nx) for result in results])
        conv_table.append((label, hs, errs))

    _plot_validation(conv_table, figures)
    for result, label in [(standard64, "standard64"), (kep64, "kep64"), (standard, "standard128"), (kep, "kep128")]:
        _plot_vorticity(result, figures, f"figure1_validation_{label}")

    summary = {
        "elapsed_seconds": elapsed,
        "standard_total_runtime_seconds": float(standard.step_walltime[-1]),
        "kep_total_runtime_seconds": float(kep.step_walltime[-1]),
        "reference": asdict(reference_cfg),
        "validation_64_standard": asdict(validation_64_standard),
        "validation_64_kep": asdict(validation_64_kep),
        "standard": asdict(standard_cfg),
        "kep": asdict(kep_cfg),
        "thick": asdict(thick_cfg),
        "thin": asdict(thin_cfg),
    }
    with (base / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return outputs

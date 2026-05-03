"""
Generate all 7 figures for the 2D Incompressible Flow Simulation analysis.
Reads data from outputs/ directory and saves plots to figures/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import sys

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Helper for loading data
def load_data(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found. Skipping related plots.")
        return None
    return np.load(filepath)


# Figure 1 & 6: Grid Convergence / Validation
def plot_convergence():
    print("Generating Fig 1 & 6: Grid Convergence...")
    # First check traveling wave
    tw_data = load_data('traveling_wave_test.npz')
    if tw_data is not None:
        grids = tw_data['grids']
        err_std = tw_data['errors_std']
        err_kep = tw_data['errors_kep']
        
        plt.figure(figsize=(8, 6))
        plt.loglog(grids, err_std, 'o-', label='Standard (Div-S2)', linewidth=2)
        plt.loglog(grids, err_kep, 's--', label='KEP (Skew-S2)', linewidth=2)
        
        # Add O(h^2) reference line
        h_vals = 1.0 / grids
        ref_line = err_std[0] * (h_vals / h_vals[0])**2
        plt.loglog(grids, ref_line, 'k:', label='O(h^2) Reference')
        
        plt.xlabel('Grid Size N (NxN)')
        plt.ylabel('L1 Error')
        plt.title('Figure 1 & 6: Grid Convergence (Traveling Wave)')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(FIG_DIR, 'fig1_6_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Fallback to thick shear layer convergence if TW not available
        grids = [64, 128, 256]
        err_std = []
        err_kep = []
        
        ref = load_data('ref_thick_512.npz')
        if ref is None:
            ref = load_data('ref_thick_256.npz')
            grids = [64, 128]
        
        if ref is None: return
        
        # Not implementing full restriction for error calculation here for simplicity,
        # relying on TW test for rigorous convergence plot.


# Figure 2: Time Evolution of Kinetic Energy
def plot_kinetic_energy():
    print("Generating Fig 2: Kinetic Energy Evolution...")
    plt.figure(figsize=(10, 6))
    
    cases = [
        ('thin_standard_128.npz', 'Standard (128x128)', 'b-'),
        ('thin_kep_128.npz', 'KEP (128x128)', 'r--'),
        ('thin_standard_256.npz', 'Standard (256x256)', 'c-'),
        ('thin_kep_256.npz', 'KEP (256x256)', 'm--'),
    ]
    
    for filename, label, fmt in cases:
        data = load_data(filename)
        if data is not None:
            plt.plot(data['times'], data['ke'], fmt, label=label, linewidth=2)
            
    plt.xlabel('Time')
    plt.ylabel('Kinetic Energy')
    plt.title('Figure 2: Time Evolution of Kinetic Energy (Thin Shear Layer, ρ=80)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, 'fig2_kinetic_energy.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Figure 3: Time Evolution of Enstrophy
def plot_enstrophy():
    print("Generating Fig 3: Enstrophy Evolution...")
    plt.figure(figsize=(10, 6))
    
    cases = [
        ('thin_standard_128.npz', 'Standard (128x128)', 'b-'),
        ('thin_kep_128.npz', 'KEP (128x128)', 'r--'),
        ('thin_standard_256.npz', 'Standard (256x256)', 'c-'),
        ('thin_kep_256.npz', 'KEP (256x256)', 'm--'),
    ]
    
    for filename, label, fmt in cases:
        data = load_data(filename)
        if data is not None:
            plt.plot(data['times'], data['enstrophy'], fmt, label=label, linewidth=2)
            
    plt.xlabel('Time')
    plt.ylabel('Enstrophy')
    plt.title('Figure 3: Time Evolution of Enstrophy (Thin Shear Layer, ρ=80)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, 'fig3_enstrophy.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Figure 4: Vorticity Contours (Standard vs KEP)
def plot_vorticity_comparison():
    print("Generating Fig 4: Vorticity Contours Comparison...")
    
    data_std = load_data('thin_standard_128.npz')
    data_kep = load_data('thin_kep_128.npz')
    
    if data_std is None or data_kep is None: return
    
    times_std = data_std['snap_times']
    omega_std = data_std['omega_snaps']
    omega_kep = data_kep['omega_snaps']
    
    # Find indices for t=0.6 and t=1.0
    idx_06 = np.argmin(np.abs(times_std - 0.6))
    idx_10 = np.argmin(np.abs(times_std - 1.0))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # t = 0.6
    im = axes[0, 0].contourf(omega_std[idx_06].T, levels=20, cmap='RdBu_r', vmin=-30, vmax=30)
    axes[0, 0].set_title('Standard (Div-S2) - t=0.6')
    axes[0, 1].contourf(omega_kep[idx_06].T, levels=20, cmap='RdBu_r', vmin=-30, vmax=30)
    axes[0, 1].set_title('KEP (Skew-S2) - t=0.6')
    
    # t = 1.0
    axes[1, 0].contourf(omega_std[idx_10].T, levels=20, cmap='RdBu_r', vmin=-30, vmax=30)
    axes[1, 0].set_title('Standard (Div-S2) - t=1.0')
    axes[1, 1].contourf(omega_kep[idx_10].T, levels=20, cmap='RdBu_r', vmin=-30, vmax=30)
    axes[1, 1].set_title('KEP (Skew-S2) - t=1.0')
    
    for ax in axes.flatten():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vorticity')
    
    fig.suptitle('Figure 4: Spurious Vortex Suppression (128x128, ρ=80)', fontsize=16)
    plt.savefig(os.path.join(FIG_DIR, 'fig4_vorticity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Figure 5: KE Conservation Error
def plot_ke_conservation_error():
    print("Generating Fig 5: KE Conservation Error...")
    plt.figure(figsize=(10, 6))
    
    cases = [
        ('thin_standard_128.npz', 'Standard (128x128)', 'b-'),
        ('thin_kep_128.npz', 'KEP (128x128)', 'r--'),
    ]
    
    for filename, label, fmt in cases:
        data = load_data(filename)
        if data is not None:
            times = data['times']
            ke = data['ke']
            ens = data['enstrophy']
            nu = data['nu']
            
            # Compute conservation error
            # error(t) = KE(t) - KE(0) + 2*nu * integral_0^t Enstrophy(t') dt'
            from scipy.integrate import cumulative_trapezoid
            integral = cumulative_trapezoid(ens, times, initial=0)
            err = ke - ke[0] + 2.0 * nu * integral
            
            plt.plot(times, np.abs(err), fmt, label=label, linewidth=2)
            
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('|KE Conservation Error|')
    plt.title('Figure 5: Kinetic Energy Conservation Error')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(FIG_DIR, 'fig5_ke_error.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Figure 7: Vorticity Contours Varying Rho
def plot_vorticity_varying_rho():
    print("Generating Fig 7: Vorticity Contours (Varying ρ)...")
    
    rhos = [80, 100, 120]
    filenames_std = ['thin_standard_128.npz', 'vthin_standard_rho100.npz', 'vthin_standard_rho120.npz']
    filenames_kep = ['thin_kep_128.npz', 'vthin_kep_rho100.npz', 'vthin_kep_rho120.npz']
    
    # Filter available
    avail_indices = []
    for i in range(len(rhos)):
        if os.path.exists(os.path.join(OUTPUT_DIR, filenames_std[i])) and \
           os.path.exists(os.path.join(OUTPUT_DIR, filenames_kep[i])):
            avail_indices.append(i)
            
    if not avail_indices:
        print("Warning: Missing data for Fig 7")
        return
        
    n_cases = len(avail_indices)
    fig, axes = plt.subplots(n_cases, 2, figsize=(10, 4*n_cases))
    if n_cases == 1:
        axes = np.array([axes])
        
    for i, idx in enumerate(avail_indices):
        data_std = load_data(filenames_std[idx])
        data_kep = load_data(filenames_kep[idx])
        
        times_std = data_std['snap_times']
        idx_10 = np.argmin(np.abs(times_std - 1.0))
        
        im = axes[i, 0].contourf(data_std['omega_snaps'][idx_10].T, levels=20, cmap='RdBu_r', vmin=-30, vmax=30)
        axes[i, 0].set_title(f'Standard - ρ={rhos[idx]}')
        
        axes[i, 1].contourf(data_kep['omega_snaps'][idx_10].T, levels=20, cmap='RdBu_r', vmin=-30, vmax=30)
        axes[i, 1].set_title(f'KEP - ρ={rhos[idx]}')
        
    for ax in axes.flatten():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vorticity')
    
    fig.suptitle('Figure 7: Vorticity at t=1.0 for Varying Shear Layer Thickness', fontsize=16)
    plt.savefig(os.path.join(FIG_DIR, 'fig7_vorticity_varying_rho.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots():
    plot_convergence()
    plot_kinetic_energy()
    plot_enstrophy()
    plot_vorticity_comparison()
    plot_ke_conservation_error()
    plot_vorticity_varying_rho()
    print("\nAll available plots generated in figures/ directory.")


if __name__ == '__main__':
    generate_all_plots()

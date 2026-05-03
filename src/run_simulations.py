"""
Run all simulation cases for the 2D incompressible flow project.
Saves results to outputs/ directory as .npz files.
"""

import numpy as np
import os
import sys
import time as timer

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import solver, diagnostics as diag


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


def save_results(results, filename):
    """Save simulation results to .npz file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Extract snapshot times and vorticity fields
    snap_times = sorted(results['snapshots'].keys())
    omega_snaps = np.array([results['snapshots'][t]['omega'] for t in snap_times])
    u_snaps = np.array([results['snapshots'][t]['u'] for t in snap_times])
    v_snaps = np.array([results['snapshots'][t]['v'] for t in snap_times])
    
    np.savez_compressed(filepath,
                        times=results['times'],
                        ke=results['ke'],
                        enstrophy=results['enstrophy'],
                        u_final=results['u_final'],
                        v_final=results['v_final'],
                        h=results['h'],
                        N=results['N'],
                        method=results['method'],
                        snap_times=np.array(snap_times),
                        omega_snaps=omega_snaps,
                        u_snaps=u_snaps,
                        v_snaps=v_snaps,
                        rho=results['params']['rho'],
                        delta=results['params']['delta'],
                        nu=results['params']['nu'],
                        L=results['params']['L'])
    print(f"  Saved: {filepath}")


def run_case(name, N, rho, delta, nu, t_end, method, cfl=0.5, output_interval=0.05):
    """Run a single simulation case."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  N={N}, rho={rho}, delta={delta}, nu={nu}, method={method}")
    print(f"{'='*60}")
    
    t_start = timer.time()
    results = solver.simulate(N, rho, delta, nu, t_end, method=method,
                              cfl=cfl, output_interval=output_interval)
    elapsed = timer.time() - t_start
    
    print(f"  Wall time: {elapsed:.1f}s")
    results['wall_time'] = elapsed
    
    save_results(results, f"{name}.npz")
    return results


def run_traveling_wave_test():
    """Run convergence test with exact traveling wave solution."""
    print("\n" + "="*60)
    print("TRAVELING WAVE CONVERGENCE TEST")
    print("="*60)
    
    nu = 0.01
    t_end = 0.7
    grids = [16, 32, 64, 128]
    errors_std = []
    errors_kep = []
    
    for N in grids:
        for method in ['standard', 'kep']:
            print(f"\n  N={N}, method={method}")
            u0, v0, h = solver.initial_condition_traveling_wave(N, nu, t=0.0)
            u_exact, v_exact, _ = solver.initial_condition_traveling_wave(N, nu, t=t_end)
            
            poisson_eig = solver._build_poisson_eigenvalues(N, h)
            
            u, v = u0.copy(), v0.copy()
            t = 0.0
            steps = 0
            while t < t_end - 1e-14:
                dt = solver.compute_dt(u, v, h, nu, cfl=0.5)
                if t + dt > t_end:
                    dt = t_end - t
                u, v, _ = solver.step_rk3(u, v, h, nu, dt, poisson_eig, method)
                t += dt
                steps += 1
            
            err = np.mean(np.abs(u - u_exact)) * h * h
            print(f"    L1 error in u: {err:.6e}, steps: {steps}")
            
            if method == 'standard':
                errors_std.append(err)
            else:
                errors_kep.append(err)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR, 'traveling_wave_test.npz'),
             grids=np.array(grids),
             errors_std=np.array(errors_std),
             errors_kep=np.array(errors_kep),
             nu=nu, t_end=t_end)
    
    # Print convergence rates
    for name, errors in [('Standard', errors_std), ('KEP', errors_kep)]:
        print(f"\n  {name} convergence rates:")
        for i in range(1, len(errors)):
            rate = np.log2(errors[i-1] / errors[i])
            print(f"    {grids[i-1]:3d} -> {grids[i]:3d}: rate = {rate:.2f}")


def run_all():
    """Run all simulation cases."""
    
    # 0. Traveling wave test
    run_traveling_wave_test()
    
    # 1. Reference solution (fine grid, thick shear layer)
    run_case('ref_thick_512', N=512, rho=30, delta=0.05, nu=0.002,
             t_end=1.0, method='kep', cfl=0.5, output_interval=0.05)
    
    # 2. Convergence study (thick shear layer, both methods)
    for N in [64, 128, 256]:
        for method in ['standard', 'kep']:
            run_case(f'thick_{method}_{N}', N=N, rho=30, delta=0.05, nu=0.002,
                     t_end=1.0, method=method, cfl=0.5, output_interval=0.05)
    
    # 3. Thin shear layer (spurious vortex study)
    for N in [128, 256]:
        for method in ['standard', 'kep']:
            run_case(f'thin_{method}_{N}', N=N, rho=80, delta=0.05, nu=0.0001,
                     t_end=1.0, method=method, cfl=0.4, output_interval=0.05)
    
    # 4. Very thin shear layers (varying rho)
    for rho, nu_val in [(100, 0.0001), (120, 0.00005)]:
        for method in ['standard', 'kep']:
            run_case(f'vthin_{method}_rho{rho}', N=128, rho=rho, delta=0.05,
                     nu=nu_val, t_end=1.0, method=method, cfl=0.3,
                     output_interval=0.1)
    
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run 2D flow simulations')
    parser.add_argument('--quick', action='store_true',
                        help='Run only essential cases (skip fine grids)')
    parser.add_argument('--case', type=str, default=None,
                        help='Run a specific case only')
    args = parser.parse_args()
    
    if args.case:
        # Parse case specification
        parts = args.case.split('_')
        print(f"Running specific case: {args.case}")
        # Custom handling
    elif args.quick:
        print("QUICK MODE: Running essential cases only")
        run_traveling_wave_test()
        # Smaller reference
        run_case('ref_thick_256', N=256, rho=30, delta=0.05, nu=0.002,
                 t_end=1.0, method='kep', cfl=0.5, output_interval=0.05)
        # Key convergence cases
        for N in [64, 128]:
            for method in ['standard', 'kep']:
                run_case(f'thick_{method}_{N}', N=N, rho=30, delta=0.05, nu=0.002,
                         t_end=1.0, method=method, cfl=0.5, output_interval=0.05)
        # Key thin shear layer case
        for method in ['standard', 'kep']:
            run_case(f'thin_{method}_128', N=128, rho=80, delta=0.05, nu=0.0001,
                     t_end=1.0, method=method, cfl=0.4, output_interval=0.05)
        # One vthin case
        for method in ['standard', 'kep']:
            run_case(f'vthin_{method}_rho100', N=128, rho=100, delta=0.05,
                     nu=0.0001, t_end=1.0, method=method, cfl=0.3,
                     output_interval=0.1)
    else:
        run_all()

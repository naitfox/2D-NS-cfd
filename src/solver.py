"""
Core 2D Incompressible Navier-Stokes Solver on a Staggered (MAC) Grid.

Implements both Standard (Divergence) and Kinetic-Energy-Preserving (Skew-Symmetric)
convective discretizations following Morinishi et al. (1998).

Grid layout (MAC/staggered):
  - u(i,j) stored at east faces:  (x_{i+1/2}, y_j)
  - v(i,j) stored at north faces: (x_i, y_{j+1/2})
  - p(i,j) stored at cell centers: (x_i, y_j)

All arrays are Nx x Ny. Periodic BCs handled via np.roll.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def create_grid(N, L=1.0):
    """Create uniform staggered grid info."""
    h = L / N
    # Cell center coordinates
    xc = np.linspace(h / 2, L - h / 2, N)
    yc = np.linspace(h / 2, L - h / 2, N)
    # Face coordinates (u-points shifted +h/2 in x, v-points shifted +h/2 in y)
    xf = np.linspace(h, L, N)  # wraps: x_N = L = 0
    yf = np.linspace(h, L, N)
    return h, xc, yc, xf, yf


def initial_condition_shear(N, rho, delta, L=1.0):
    """
    Doubly-periodic shear layer initial conditions (Eq. 18, Minion & Brown 1997).

    u = tanh(rho*(y - 0.25))    for y <= 0.5
        tanh(rho*(0.75 - y))    for y > 0.5
    v = delta * sin(2*pi*(x + 0.25))
    """
    h, xc, yc, xf, yf = create_grid(N, L)

    # u is at (xf[i], yc[j]) -- east face centers
    Xu, Yu = np.meshgrid(xf, yc, indexing='ij')
    u = np.where(Yu <= 0.5,
                 np.tanh(rho * (Yu - 0.25)),
                 np.tanh(rho * (0.75 - Yu)))

    # v is at (xc[i], yf[j]) -- north face centers
    Xv, Yv = np.meshgrid(xc, yf, indexing='ij')
    v = delta * np.sin(2.0 * np.pi * (Xv + 0.25))

    return u, v, h


def initial_condition_traveling_wave(N, nu, t=0.0, L=1.0):
    """
    Exact traveling wave solution for convergence testing.
    u(x,y,t) = 1 + 2*cos(2*pi*(x-t))*sin(2*pi*(y-t))*exp(-8*pi^2*nu*t)
    v(x,y,t) = 1 - 2*sin(2*pi*(x-t))*cos(2*pi*(y-t))*exp(-8*pi^2*nu*t)
    """
    h, xc, yc, xf, yf = create_grid(N, L)
    decay = np.exp(-8.0 * np.pi**2 * nu * t)

    Xu, Yu = np.meshgrid(xf, yc, indexing='ij')
    u = 1.0 + 2.0 * np.cos(2*np.pi*(Xu - t)) * np.sin(2*np.pi*(Yu - t)) * decay

    Xv, Yv = np.meshgrid(xc, yf, indexing='ij')
    v = 1.0 - 2.0 * np.sin(2*np.pi*(Xv - t)) * np.cos(2*np.pi*(Yv - t)) * decay

    return u, v, h


# ============================================================
# Interpolation helpers (periodic via np.roll)
# ============================================================

def _avg_x_plus(phi):
    """Average phi in +x direction: (phi[i] + phi[i+1]) / 2"""
    return 0.5 * (phi + np.roll(phi, -1, axis=0))

def _avg_x_minus(phi):
    """Average phi in -x direction: (phi[i] + phi[i-1]) / 2"""
    return 0.5 * (phi + np.roll(phi, 1, axis=0))

def _avg_y_plus(phi):
    """Average phi in +y direction: (phi[j] + phi[j+1]) / 2"""
    return 0.5 * (phi + np.roll(phi, -1, axis=1))

def _avg_y_minus(phi):
    """Average phi in -y direction: (phi[j] + phi[j-1]) / 2"""
    return 0.5 * (phi + np.roll(phi, 1, axis=1))


# ============================================================
# Convective terms
# ============================================================

def convective_standard(u, v, h):
    """
    Standard divergence form (Div-S2) on staggered grid.
    
    For u-momentum:  d(uu)/dx + d(vu)/dy
    For v-momentum:  d(uv)/dx + d(vv)/dy
    
    Fluxes are computed at the correct staggered locations.
    """
    N = u.shape[0]
    
    # --- u-momentum: conv_u = d(uu)/dx + d(vu)/dy ---
    # uu flux at cell centers (x_i, y_j): u averaged to center, times u averaged to center
    # u is at (i+1/2, j). Average two neighbors in x to get u at cell center:
    u_at_center = _avg_x_minus(u)  # u at (i, j)
    uu_flux = u_at_center * u_at_center  # Not quite right for staggered grid
    
    # Correct staggered approach:
    # uu flux needed at (i, j) and (i+1, j) for u-momentum at (i+1/2, j)
    # uu at (i,j) = u(i-1/2,j) * u(i+1/2,j) averaged = interp of u to (i,j) squared... 
    # Actually: flux at x-face of u-control-volume
    # u-CV centered at (i+1/2, j), faces at (i, j) and (i+1, j)
    # uu_right = u at (i+1, j) * u at (i+1, j) 
    # Need U_j at u-faces: interpolate u to cell centers first
    
    # Simpler and correct: use the MAC form directly
    # U at cell centers (for advecting velocity): 
    U_cc = 0.5 * (u + np.roll(u, 1, axis=0))  # u at (i, j)
    
    # uu flux at (i, j): U_cc * u interpolated to (i, j)
    u_interp_x = 0.5 * (u + np.roll(u, 1, axis=0))  # = U_cc
    flux_uu_x = U_cc * u_interp_x  # at (i, j)
    
    # d(uu)/dx at (i+1/2, j):
    duu_dx = (np.roll(flux_uu_x, -1, axis=0) - flux_uu_x) / h
    
    # V at u-points for y-flux: v at (i, j+1/2), need at (i+1/2, j+1/2)
    V_at_u_yface = 0.5 * (v + np.roll(v, -1, axis=0))  # v at (i+1/2, j+1/2)
    # u interpolated to (i+1/2, j+1/2):
    u_interp_y = 0.5 * (u + np.roll(u, -1, axis=1))  # u at (i+1/2, j+1/2)
    flux_vu_y = V_at_u_yface * u_interp_y  # at (i+1/2, j+1/2)
    
    # d(vu)/dy at (i+1/2, j):
    dvu_dy = (flux_vu_y - np.roll(flux_vu_y, 1, axis=1)) / h
    
    conv_u = duu_dx + dvu_dy
    
    # --- v-momentum: conv_v = d(uv)/dx + d(vv)/dy ---
    # U at v-points for x-flux: u at (i+1/2, j), need at (i+1/2, j+1/2)
    U_at_v_xface = 0.5 * (u + np.roll(u, -1, axis=1))  # u at (i+1/2, j+1/2)
    # v interpolated to (i+1/2, j+1/2):
    v_interp_x = 0.5 * (v + np.roll(v, -1, axis=0))  # v at (i+1/2, j+1/2)
    flux_uv_x = U_at_v_xface * v_interp_x  # at (i+1/2, j+1/2)
    
    # d(uv)/dx at (i, j+1/2):
    duv_dx = (flux_uv_x - np.roll(flux_uv_x, 1, axis=0)) / h
    
    # V at cell centers: 
    V_cc = 0.5 * (v + np.roll(v, 1, axis=1))  # v at (i, j)
    v_interp_y = 0.5 * (v + np.roll(v, 1, axis=1))  # = V_cc
    flux_vv_y = V_cc * v_interp_y  # at (i, j)
    
    # d(vv)/dy at (i, j+1/2):
    dvv_dy = (np.roll(flux_vv_y, -1, axis=1) - flux_vv_y) / h
    
    conv_v = duv_dx + dvv_dy
    
    return conv_u, conv_v


def convective_advective(u, v, h):
    """
    Advective form (Adv-S2) on staggered grid.
    (Adv-S2)_i = Ū_j^{1xi} * δ₁Uᵢ / δ₁xⱼ
    """
    # --- u-momentum ---
    # U advecting in x at u-point (i+1/2, j): interpolate U_cc to (i+1/2, j)
    # Actually need U at faces of u-CV
    U_cc = 0.5 * (u + np.roll(u, 1, axis=0))  # u at cell center (i, j)
    U_at_upt = 0.5 * (U_cc + np.roll(U_cc, -1, axis=0))  # U at (i+1/2, j)
    
    # du/dx at (i+1/2, j) using values at (i+3/2, j) and (i-1/2, j)
    du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * h)
    
    # V advecting in y at u-point: need V at (i+1/2, j)
    V_at_upt_top = 0.5 * (v + np.roll(v, -1, axis=0))  # v at (i+1/2, j+1/2)
    V_at_upt = 0.5 * (V_at_upt_top + np.roll(V_at_upt_top, 1, axis=1))  # v at (i+1/2, j)
    
    # du/dy at (i+1/2, j)
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * h)
    
    adv_u = U_at_upt * du_dx + V_at_upt * du_dy
    
    # --- v-momentum ---
    U_at_vpt_right = 0.5 * (u + np.roll(u, -1, axis=1))  # u at (i+1/2, j+1/2)
    U_at_vpt = 0.5 * (U_at_vpt_right + np.roll(U_at_vpt_right, 1, axis=0))  # u at (i, j+1/2)
    
    dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * h)
    
    V_cc = 0.5 * (v + np.roll(v, 1, axis=1))  # v at (i, j)
    V_at_vpt = 0.5 * (V_cc + np.roll(V_cc, -1, axis=1))  # V at (i, j+1/2)
    
    dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * h)
    
    adv_v = U_at_vpt * dv_dx + V_at_vpt * dv_dy
    
    return adv_u, adv_v


def convective_kep(u, v, h):
    """
    Kinetic-Energy-Preserving skew-symmetric form (Skew-S2).
    (Skew-S2) = 0.5*(Div-S2) + 0.5*(Adv-S2)
    
    This form conserves kinetic energy a priori (Morinishi et al. 1998, Table 7).
    """
    div_u, div_v = convective_standard(u, v, h)
    adv_u, adv_v = convective_advective(u, v, h)
    return 0.5 * (div_u + adv_u), 0.5 * (div_v + adv_v)


# ============================================================
# Viscous terms
# ============================================================

def viscous_term(u, v, h, nu):
    """
    Viscous term: nu * Laplacian(u), nu * Laplacian(v)
    Standard 5-point stencil, periodic.
    """
    inv_h2 = 1.0 / (h * h)
    
    lap_u = (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) +
             np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4.0 * u) * inv_h2
    
    lap_v = (np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) +
             np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 4.0 * v) * inv_h2
    
    return nu * lap_u, nu * lap_v


# ============================================================
# Pressure Poisson solver (FFT-based)
# ============================================================

def _build_poisson_eigenvalues(N, h):
    """
    Eigenvalues of the 5-point Laplacian in Fourier space.
    λ = (2*cos(2πk₁h) - 2)/h² + (2*cos(2πk₂h) - 2)/h²
    """
    kx = fftfreq(N, d=h)
    ky = fftfreq(N, d=h)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    eig = (2.0 * np.cos(2.0 * np.pi * KX * h) - 2.0) / (h * h) + \
          (2.0 * np.cos(2.0 * np.pi * KY * h) - 2.0) / (h * h)
    
    # Avoid division by zero for the (0,0) mode
    eig[0, 0] = 1.0
    return eig


def pressure_projection(u, v, h, dt, poisson_eig):
    """
    Project velocity field to be divergence-free using pressure correction.
    
    1. Compute divergence of u*, v*
    2. Solve Poisson equation: Lap(p) = div(u*) / dt
    3. Correct: u = u* - dt * dp/dx,  v = v* - dt * dp/dy
    """
    N = u.shape[0]
    
    # Divergence at cell centers: (u[i+1/2] - u[i-1/2])/h + (v[j+1/2] - v[j-1/2])/h
    # u is stored at (i+1/2, j), so:
    # div at (i,j) = (u[i,j] - u[i-1,j])/h + (v[i,j] - v[i,j-1])/h
    div = (u - np.roll(u, 1, axis=0)) / h + (v - np.roll(v, 1, axis=1)) / h
    
    # RHS of Poisson equation
    rhs = div / dt
    
    # Solve in Fourier space
    rhs_hat = fft2(rhs)
    p_hat = rhs_hat / poisson_eig
    p_hat[0, 0] = 0.0  # Fix mean pressure to zero
    p = np.real(ifft2(p_hat))
    
    # Pressure gradient at velocity locations
    # dp/dx at (i+1/2, j): (p[i+1,j] - p[i,j]) / h
    dpdx = (np.roll(p, -1, axis=0) - p) / h
    # dp/dy at (i, j+1/2): (p[i,j+1] - p[i,j]) / h
    dpdy = (np.roll(p, -1, axis=1) - p) / h
    
    u_new = u - dt * dpdx
    v_new = v - dt * dpdy
    
    return u_new, v_new, p


# ============================================================
# Right-hand side evaluation
# ============================================================

def compute_rhs(u, v, h, nu, method='standard'):
    """
    Compute RHS of momentum equation: -Conv + nu*Lap
    (pressure is handled separately via projection)
    """
    if method == 'kep':
        conv_u, conv_v = convective_kep(u, v, h)
    else:
        conv_u, conv_v = convective_standard(u, v, h)
    
    visc_u, visc_v = viscous_term(u, v, h, nu)
    
    return -conv_u + visc_u, -conv_v + visc_v


# ============================================================
# Time integration: RK3-TVD (Shu-Osher)
# ============================================================

def compute_dt(u, v, h, nu, cfl=0.5):
    """CFL-based adaptive time step."""
    max_vel = np.max(np.abs(u)) + np.max(np.abs(v))
    if max_vel < 1e-14:
        max_vel = 1e-14
    dt = cfl * h / (max_vel + 2.0 * nu / h)
    return dt


def step_rk3(u, v, h, nu, dt, poisson_eig, method='standard'):
    """
    Advance one time step using 3rd-order TVD Runge-Kutta.
    
    Stage 1: u* = u^n + dt * L(u^n);           project
    Stage 2: u* = 3/4*u^n + 1/4*(u1 + dt*L(u1)); project  
    Stage 3: u* = 1/3*u^n + 2/3*(u2 + dt*L(u2)); project
    """
    # Stage 1
    rhs_u, rhs_v = compute_rhs(u, v, h, nu, method)
    u1 = u + dt * rhs_u
    v1 = v + dt * rhs_v
    u1, v1, _ = pressure_projection(u1, v1, h, dt, poisson_eig)
    
    # Stage 2
    rhs_u, rhs_v = compute_rhs(u1, v1, h, nu, method)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs_u)
    v2 = 0.75 * v + 0.25 * (v1 + dt * rhs_v)
    u2, v2, _ = pressure_projection(u2, v2, h, dt, poisson_eig)
    
    # Stage 3
    rhs_u, rhs_v = compute_rhs(u2, v2, h, nu, method)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * rhs_u)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v2 + dt * rhs_v)
    u_new, v_new, p = pressure_projection(u_new, v_new, h, dt, poisson_eig)
    
    return u_new, v_new, p


# ============================================================
# Main simulation driver
# ============================================================

def simulate(N, rho, delta, nu, t_end, method='standard', L=1.0,
             cfl=0.5, output_interval=0.05, verbose=True,
             ic_func=None):
    """
    Run a full simulation and return time histories of diagnostics.
    
    Parameters
    ----------
    N : int - Grid resolution (N x N)
    rho : float - Shear layer width parameter
    delta : float - Perturbation strength
    nu : float - Kinematic viscosity
    t_end : float - Final time
    method : str - 'standard' or 'kep'
    L : float - Domain size
    cfl : float - CFL number
    output_interval : float - Time interval between saved snapshots
    verbose : bool - Print progress
    ic_func : callable - Optional custom initial condition function
    
    Returns
    -------
    results : dict with time histories and final fields
    """
    # Initialize
    if ic_func is not None:
        u, v, h = ic_func(N)
    else:
        u, v, h = initial_condition_shear(N, rho, delta, L)
    
    poisson_eig = _build_poisson_eigenvalues(N, h)
    
    # Storage for diagnostics
    times = [0.0]
    ke_history = []
    enstrophy_history = []
    snapshots = {}
    
    # Compute initial diagnostics
    from . import diagnostics as diag
    ke_history.append(diag.kinetic_energy(u, v, h))
    enstrophy_history.append(diag.enstrophy(u, v, h))
    omega0 = diag.vorticity(u, v, h)
    snapshots[0.0] = {'u': u.copy(), 'v': v.copy(), 'omega': omega0}
    
    t = 0.0
    step_count = 0
    next_output = output_interval
    
    while t < t_end - 1e-14:
        dt = compute_dt(u, v, h, nu, cfl)
        if t + dt > t_end:
            dt = t_end - t
        if t + dt > next_output and next_output <= t_end:
            dt = next_output - t
        
        u, v, _ = step_rk3(u, v, h, nu, dt, poisson_eig, method)
        t += dt
        step_count += 1
        
        # Check divergence periodically
        if step_count % 500 == 0:
            div_max = np.max(np.abs(
                (u - np.roll(u, 1, axis=0)) / h + (v - np.roll(v, 1, axis=1)) / h
            ))
            if verbose:
                print(f"  Step {step_count:6d}, t={t:.6f}, dt={dt:.2e}, "
                      f"max|div|={div_max:.2e}")
        
        # Save output at intervals
        if abs(t - next_output) < 1e-12 or t >= t_end - 1e-14:
            ke = diag.kinetic_energy(u, v, h)
            ens = diag.enstrophy(u, v, h)
            omega = diag.vorticity(u, v, h)
            
            times.append(t)
            ke_history.append(ke)
            enstrophy_history.append(ens)
            snapshots[round(t, 6)] = {
                'u': u.copy(), 'v': v.copy(), 'omega': omega
            }
            
            if verbose:
                print(f"  Output at t={t:.4f}: KE={ke:.8f}, Enstrophy={ens:.4f}")
            
            next_output += output_interval
    
    if verbose:
        print(f"Simulation complete: {step_count} steps, t_final={t:.6f}")
    
    results = {
        'times': np.array(times),
        'ke': np.array(ke_history),
        'enstrophy': np.array(enstrophy_history),
        'snapshots': snapshots,
        'u_final': u,
        'v_final': v,
        'h': h,
        'N': N,
        'method': method,
        'params': {'rho': rho, 'delta': delta, 'nu': nu, 'L': L}
    }
    
    return results

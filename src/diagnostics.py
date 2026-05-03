"""
Diagnostics for 2D incompressible flow simulations.
Computes kinetic energy, enstrophy, vorticity, and conservation errors.
"""

import numpy as np


def kinetic_energy(u, v, h):
    """
    Compute domain-averaged kinetic energy on the staggered grid.
    KE = (1/2) * <u^2 + v^2> * L^2
    
    Since u and v live at different locations, we average each to cell centers
    before computing the squared sum.
    """
    N = u.shape[0]
    L = N * h
    # Interpolate u to cell centers: u_cc[i,j] = 0.5*(u[i,j] + u[i-1,j])
    u_cc = 0.5 * (u + np.roll(u, 1, axis=0))
    # Interpolate v to cell centers: v_cc[i,j] = 0.5*(v[i,j] + v[i,j-1])
    v_cc = 0.5 * (v + np.roll(v, 1, axis=1))
    
    ke = 0.5 * np.sum(u_cc**2 + v_cc**2) * h * h
    return ke


def enstrophy(u, v, h):
    """
    Compute domain-integrated enstrophy: Omega = (1/2) * integral(omega^2) dA
    where omega = dv/dx - du/dy is the vorticity at cell corners.
    """
    omega = vorticity(u, v, h)
    return 0.5 * np.sum(omega**2) * h * h


def vorticity(u, v, h):
    """
    Compute vorticity omega = dv/dx - du/dy at cell corners (i+1/2, j+1/2).
    
    Using the natural staggered-grid finite differences:
    dv/dx at (i+1/2, j+1/2) = (v[i+1, j+1/2] - v[i, j+1/2]) / h
                              = (v[i+1, j] - v[i, j]) / h  (in array indexing)
    du/dy at (i+1/2, j+1/2) = (u[i+1/2, j+1] - u[i+1/2, j]) / h
                              = (u[i, j+1] - u[i, j]) / h  (in array indexing)
    """
    dvdx = (np.roll(v, -1, axis=0) - v) / h
    dudy = (np.roll(u, -1, axis=1) - u) / h
    omega = dvdx - dudy
    return omega


def vorticity_at_centers(u, v, h):
    """
    Compute vorticity at cell centers by averaging corner vorticities.
    Useful for contour plotting.
    """
    omega_corner = vorticity(u, v, h)
    # Average from corners to centers
    omega_cc = 0.25 * (omega_corner + np.roll(omega_corner, 1, axis=0) +
                       np.roll(omega_corner, 1, axis=1) +
                       np.roll(omega_corner, (1, 1), axis=(0, 1)))
    return omega_cc


def ke_conservation_error(times, ke_values, enstrophy_values, nu):
    """
    Compute kinetic energy conservation error.
    
    The exact energy balance is:
    dKE/dt = -2*nu*Enstrophy
    
    So: KE(t) = KE(0) - 2*nu * integral_0^t Enstrophy(t') dt'
    
    The conservation error is:
    error(t) = KE(t) - KE(0) + 2*nu * integral_0^t Enstrophy(t') dt'
    
    For a perfectly energy-conserving spatial scheme, this should be zero
    (up to time integration error).
    """
    nt = len(times)
    error = np.zeros(nt)
    
    for i in range(1, nt):
        # Trapezoidal integration of enstrophy
        dt = times[i] - times[i-1]
        integral = np.trapz(enstrophy_values[:i+1], times[:i+1])
        error[i] = ke_values[i] - ke_values[0] + 2.0 * nu * integral
    
    return error


def divergence_check(u, v, h):
    """Check maximum divergence of the velocity field."""
    div = (u - np.roll(u, 1, axis=0)) / h + (v - np.roll(v, 1, axis=1)) / h
    return np.max(np.abs(div))

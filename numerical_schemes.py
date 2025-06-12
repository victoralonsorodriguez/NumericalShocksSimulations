"""
Implements various numerical schemes for solving the 1D Euler equations.

Includes:
- Lax-Friedrichs scheme
- Lax-Wendroff-Ritchmyer scheme
- Flux-limited scheme using Minmod limiter
This module also provides helper functions for calculating numerical fluxes
at cell interfaces and for flux limiter computations.
"""

import numpy as np
from typing import Tuple

# Assuming these modules are in the same directory
import hydro_utils # For get_primitive_vars, get_physical_fluxes_at_cell_centers
import simulation_constants as constants   # For NUM_GHOST_CELLS, R_DENOMINATOR_FLOOR

#--- Numerical Flux Functions for Interfaces ---#

def calculate_lax_friedrichs_interface_flux(
    u_L: np.ndarray,
    u_R: np.ndarray,
    f_L: np.ndarray,
    f_R: np.ndarray,
    dt: float, dx: float
) -> np.ndarray:
    """
    Calculate the Lax-Friedrichs numerical flux F_LF at an interface.

    The Lax-Friedrichs flux is a first-order accurate numerical flux given by:
    F_LF = 0.5 * (F(U_L) + F(U_R)) - 0.5 * (dx/dt) * (U_R - U_L)

    Parameters
    ----------
    u_L : np.ndarray
        Conservative state vector [rho, rhov, rhoe] at the cell to the left of the interface.
    u_R : np.ndarray
        Conservative state vector [rho, rhov, rhoe] at the cell to the right of the interface.
    f_L : np.ndarray
        Physical flux F(U_L) corresponding to the state u_L.
    f_R : np.ndarray
        Physical flux F(U_R) corresponding to the state u_R.
    dt : float
        Timestep.
    dx : float
        Grid spacing.

    Returns
    -------
    np.ndarray
        The Lax-Friedrichs numerical flux vector at the interface.
    """
    f_interface_lf = 0.5 * (f_L + f_R) - 0.5 * (dx / dt) * (u_R - u_L)
    return f_interface_lf

def calculate_lax_wendroff_interface_flux(
    u_L: np.ndarray,
    u_R: np.ndarray,
    f_L: np.ndarray,
    f_R: np.ndarray,
    dt: float,
    dx: float,
    gamma_gas_scheme: float
) -> np.ndarray:
    """
    Calculate the Lax-Wendroff numerical flux F_LW at an interface.

    This flux is derived from the Richtmyer two-step Lax-Wendroff method.
    U_intermediate = 0.5 * (U_L + U_R) - (0.5 * dt / dx) * (F(U_R) - F(U_L))
    F_LW = F(U_intermediate)

    Parameters
    ----------
    u_L : np.ndarray
        Conservative state vector at the cell to the left of the interface.
    u_R : np.ndarray
        Conservative state vector at the cell to the right of the interface.
    f_L : np.ndarray
        Physical flux F(U_L).
    f_R : np.ndarray
        Physical flux F(U_R).
    dt : float
        Timestep.
    dx : float
        Grid spacing.
    gamma_gas_scheme : float
        Adiabatic index of the gas.

    Returns
    -------
    np.ndarray
        The Lax-Wendroff numerical flux vector at the interface.
    """
    # Step 1: Calculate the intermediate state U_hat at the interface center (j+1/2) and time level (n+1/2).
    u_interface_intermediate = 0.5 * (u_L + u_R) - (0.5 * dt / dx) * (f_R - f_L)

    # Step 2: Convert this intermediate state U_hat to primitive variables.
    rho_interface_intermediate, v_interface_intermediate, P_interface_intermediate = hydro_utils.get_primitive_vars(
        u_interface_intermediate[0], u_interface_intermediate[1], u_interface_intermediate[2],
        gamma=gamma_gas_scheme
    )
    # Step 3: Calculate the physical flux F(U_hat) using these primitive variables. This is the LWR numerical flux.
    f_interface_intermediate = hydro_utils.get_physical_fluxes_at_cell_centers(
        np.array([rho_interface_intermediate]), np.array([v_interface_intermediate]),
        np.array([P_interface_intermediate]), np.array([u_interface_intermediate[2]])
    ).flatten()
    return f_interface_intermediate

#--- Numerical Scheme Steps ---#

def lax_friedrichs_step(
    u_conserved_with_ghosts: np.ndarray,
    dt_sys: float,
    dx_grid: float,
    gamma_gas_scheme: float
) -> np.ndarray:
    """
    Perform one time step using the Lax-Friedrichs numerical scheme.

    The scheme updates the conservative variables U in each active cell i as:
    U_i^{n+1} = U_i^n - (dt/dx) * (F_LF_{i+1/2} - F_LF_{i-1/2})
    where F_LF is the Lax-Friedrichs numerical flux at the cell interfaces.
    """
    num_ghost = constants.NUM_GHOST_CELLS
    N_total = u_conserved_with_ghosts.shape[1]
    N_active = N_total - 2 * num_ghost

    u_current_active = u_conserved_with_ghosts[:, num_ghost:-num_ghost]
    u_next_active = np.zeros_like(u_current_active)

    # Calculate primitive variables and physical fluxes F(U) on the full grid (including ghosts).
    rho_full, v_full, P_full = hydro_utils.get_primitive_vars(
        rho_arr=u_conserved_with_ghosts[0,:],
        rhov_arr=u_conserved_with_ghosts[1,:],
        rhoe_arr=u_conserved_with_ghosts[2,:],
        gamma=gamma_gas_scheme
    )
    f_physical_at_centers_full = hydro_utils.get_physical_fluxes_at_cell_centers(
        rho_arr=rho_full, v_arr=v_full, P_arr=P_full, rhoe_arr=u_conserved_with_ghosts[2,:]
    )

    # Calculate Lax-Friedrichs numerical fluxes at all interfaces relevant to active cells.
    # There are N_active + 1 such interfaces.
    f_numerical_at_interfaces = np.zeros((3, N_active + 1))
    for k_interface_idx in range(N_active + 1):
        # Indices for cells to the left (L) and right (R) of the current interface.
        idx_cell_L = k_interface_idx + num_ghost - 1
        idx_cell_R = idx_cell_L + 1
        u_L = u_conserved_with_ghosts[:, idx_cell_L]
        u_R = u_conserved_with_ghosts[:, idx_cell_R]
        f_L = f_physical_at_centers_full[:, idx_cell_L]
        f_R = f_physical_at_centers_full[:, idx_cell_R]

        f_numerical_at_interfaces[:, k_interface_idx] = calculate_lax_friedrichs_interface_flux(
            u_L, u_R, f_L, f_R, dt_sys, dx_grid
        )

    # Update conservative variables in active cells using the calculated interface fluxes.
    for idx_cell_active in range(N_active):
        f_LF_right_interface = f_numerical_at_interfaces[:, idx_cell_active + 1]
        f_LF_left_interface  = f_numerical_at_interfaces[:, idx_cell_active]
        # U_i^{n+1} = U_i^n - (dt/dx) * (F_{i+1/2} - F_{i-1/2})
        u_next_active[:, idx_cell_active] = (u_current_active[:, idx_cell_active] -
                                            (dt_sys / dx_grid) * (f_LF_right_interface - f_LF_left_interface))
    return u_next_active

def lax_wendroff_ritchmyer_step(
    u_conserved_with_ghosts: np.ndarray,
    dt_sys: float,
    dx_grid: float,
    gamma_gas_scheme: float
) -> np.ndarray:
    """
    Perform one time step using the Lax-Wendroff-Ritchmyer numerical scheme.

    This is a two-step, second-order accurate scheme. The update rule is:
    U_i^{n+1} = U_i^n - (dt/dx) * (F_LW_{i+1/2} - F_LW_{i-1/2})
    where F_LW is the Lax-Wendroff numerical flux at cell interfaces.
    """
    num_ghost = constants.NUM_GHOST_CELLS
    N_total = u_conserved_with_ghosts.shape[1]
    N_active = N_total - 2 * num_ghost

    u_current_active = u_conserved_with_ghosts[:, num_ghost:-num_ghost]
    u_next_active = np.zeros_like(u_current_active)

    # Calculate primitive variables and physical fluxes F(U^n) on the full grid.
    rho_full, v_full, P_full = hydro_utils.get_primitive_vars(
        rho_arr=u_conserved_with_ghosts[0,:],
        rhov_arr=u_conserved_with_ghosts[1,:],
        rhoe_arr=u_conserved_with_ghosts[2,:],
        gamma=gamma_gas_scheme
    )
    f_physical_at_centers_full = hydro_utils.get_physical_fluxes_at_cell_centers(
        rho_arr=rho_full, v_arr=v_full, P_arr=P_full, rhoe_arr=u_conserved_with_ghosts[2,:]
    )

    # Calculate Lax-Wendroff numerical fluxes F_LW at all relevant interfaces.
    f_numerical_at_interfaces = np.zeros((3, N_active + 1))
    for k_interface_idx in range(N_active + 1):
        # Indices for cells to the left (L) and right (R) of the current interface.
        idx_cell_L = k_interface_idx + num_ghost - 1
        idx_cell_R = idx_cell_L + 1
        u_L = u_conserved_with_ghosts[:, idx_cell_L]
        u_R = u_conserved_with_ghosts[:, idx_cell_R]
        f_L = f_physical_at_centers_full[:, idx_cell_L]
        f_R = f_physical_at_centers_full[:, idx_cell_R]

        f_numerical_at_interfaces[:, k_interface_idx] = calculate_lax_wendroff_interface_flux(
            u_L, u_R, f_L, f_R, dt_sys, dx_grid, gamma_gas_scheme
        )

    # Update conservative variables in active cells using these F_LW interface fluxes.
    for idx_cell_active in range(N_active):
        f_LWR_right_interface = f_numerical_at_interfaces[:, idx_cell_active + 1]
        f_LWR_left_interface  = f_numerical_at_interfaces[:, idx_cell_active]
        # U_i^{n+1} = U_i^n - (dt/dx) * (F_{i+1/2} - F_{i-1/2})
        u_next_active[:, idx_cell_active] = (u_current_active[:, idx_cell_active] -
                                            (dt_sys / dx_grid) * (f_LWR_right_interface - f_LWR_left_interface))
    return u_next_active

#--- Flux Limiter Functions ---#

def minmod_limiter(r_val_vec: np.ndarray) -> np.ndarray:
    """
    Apply the Minmod flux limiter function element-wise.

    phi(r) = max(0, min(1, r))
    This limiter is symmetric and ensures Total Variation Diminishing (TVD) properties.
    """
    return np.maximum(0.0, np.minimum(1.0, r_val_vec))

def calculate_relative_gradient_r(
    u_conserved_with_ghosts: np.ndarray,
    idx_cell_current: int
) -> np.ndarray:
    """
    Calculate the relative gradient r for a given cell.
    r_i = (U_i - U_{i-1}) / (U_{i+1} - U_i)
    This ratio compares the gradient on the "upwind" side (U_i - U_{i-1})
    to the gradient on the "downwind" side (U_{i+1} - U_i) of the interface
    for which the limiter is being evaluated.
    Calculated for each conservative variable.
    """
    # Difference with cell to the left (backward difference at cell i)
    delta_u_backward_at_i = u_conserved_with_ghosts[:, idx_cell_current] - u_conserved_with_ghosts[:, idx_cell_current-1]
    # Difference with cell to the right (forward difference at cell i)
    delta_u_forward_at_i  = u_conserved_with_ghosts[:, idx_cell_current+1] - u_conserved_with_ghosts[:, idx_cell_current]

    rel_gradient_vec = np.zeros(u_conserved_with_ghosts.shape[0])

    for var_idx in range(u_conserved_with_ghosts.shape[0]):
        if np.abs(delta_u_forward_at_i[var_idx]) < constants.R_DENOMINATOR_FLOOR:
            # Denominator is close to zero.
            if np.abs(delta_u_backward_at_i[var_idx]) < constants.R_DENOMINATOR_FLOOR:
                # Numerator also close to zero (0/0 case), indicates a smooth region.
                rel_gradient_vec[var_idx] = 1.0 # Treat as r=1, allows for higher-order flux.
            else:
                # Numerator non-zero, denominator zero (extremum or sharp change).
                rel_gradient_vec[var_idx] = np.sign(delta_u_backward_at_i[var_idx]) * np.inf # Effectively a very large r.
        else:
            # Standard case: calculate r.
            rel_gradient_vec[var_idx] = delta_u_backward_at_i[var_idx] / delta_u_forward_at_i[var_idx]
    return rel_gradient_vec

def limiter_scheme_step(
    u_conserved_with_ghosts: np.ndarray,
    dt_sys: float,
    dx_grid: float,
    gamma_gas_scheme: float
) -> np.ndarray:
    """
    Perform one time step using a flux-limited scheme with the Minmod limiter.

    The update rule for an active cell i is:
    U_i^{n+1} = U_i^n - (dt/dx) * (F_LIM_{i+1/2} - F_LIM_{i-1/2})
    The limited flux F_LIM at an interface j+1/2 is a combination of a
    low-order (Lax-Friedrichs) and a high-order (Lax-Wendroff) flux:
    F_LIM_{j+1/2} = F_LF_{j+1/2} + phi(r_j) * (F_LW_{j+1/2} - F_LF_{j+1/2}),
    where phi(r_j) is the Minmod limiter based on the gradient ratio r from cell j.
    """
    num_ghost = constants.NUM_GHOST_CELLS
    N_total = u_conserved_with_ghosts.shape[1]
    N_active = N_total - 2 * num_ghost

    u_current_active = u_conserved_with_ghosts[:, num_ghost:-num_ghost]
    u_next_active = np.zeros_like(u_current_active)

    # Calculate primitive variables and physical fluxes F(U) on the full grid.
    rho_full, v_full, P_full = hydro_utils.get_primitive_vars(
        rho_arr=u_conserved_with_ghosts[0,:],
        rhov_arr=u_conserved_with_ghosts[1,:],
        rhoe_arr=u_conserved_with_ghosts[2,:],
        gamma=gamma_gas_scheme
    )
    f_physical_at_centers_full = hydro_utils.get_physical_fluxes_at_cell_centers(
        rho_arr=rho_full, v_arr=v_full, P_arr=P_full, rhoe_arr=u_conserved_with_ghosts[2,:]
    )

    # Step 1: Calculate the limiter function phi(r_i) for all active cells.
    # phi_for_active_cells[variable_index, active_cell_index]
    phi_for_active_cells = np.zeros((3, N_active))
    for idx_cell_active in range(N_active):
        idx_cell_current_full = idx_cell_active + num_ghost
        r_vec_current_cell = calculate_relative_gradient_r(u_conserved_with_ghosts, idx_cell_current_full)
        phi_for_active_cells[:, idx_cell_active] = minmod_limiter(r_vec_current_cell)

    # Step 2: Calculate the limited numerical fluxes F_LIM at all interfaces relevant to active cells.
    F_limited_at_interfaces = np.zeros((3, N_active + 1))
    for k_interface_idx in range(N_active + 1):
        # Indices for cells to the left (L) and right (R) of the current interface.
        idx_cell_L_of_interface = k_interface_idx + num_ghost - 1
        idx_cell_R_of_interface = idx_cell_L_of_interface + 1
        u_L = u_conserved_with_ghosts[:, idx_cell_L_of_interface]
        u_R = u_conserved_with_ghosts[:, idx_cell_R_of_interface]
        f_L = f_physical_at_centers_full[:, idx_cell_L_of_interface]
        f_R = f_physical_at_centers_full[:, idx_cell_R_of_interface]

        # Calculate low-order (Lax-Friedrichs) and high-order (Lax-Wendroff) fluxes at the interface.
        f_interface_LF = calculate_lax_friedrichs_interface_flux(u_L, u_R, f_L, f_R, dt_sys, dx_grid)
        f_interface_LWR = calculate_lax_wendroff_interface_flux(u_L, u_R, f_L, f_R, dt_sys, dx_grid, gamma_gas_scheme)

        # Determine which phi(r) to use for this interface.
        # The limiter phi(r_j) is typically based on the state in cell j, which is to the "upwind" side
        # or, more commonly for cell-centered schemes, the cell to the left of the interface j+1/2.
        active_idx_of_cell_L = idx_cell_L_of_interface - num_ghost
        phi_to_use_vec = np.zeros(3)
        if active_idx_of_cell_L < 0:
            # Interface is at the left boundary of the active domain. Use phi of the first active cell.
            phi_to_use_vec = phi_for_active_cells[:, 0]
        elif active_idx_of_cell_L >= N_active:
            # Interface is at the right boundary of the active domain. Use phi of the last active cell.
            phi_to_use_vec = phi_for_active_cells[:, N_active - 1]
        else:
            # Standard internal interface. Use phi from the cell to its left.
            phi_to_use_vec = phi_for_active_cells[:, active_idx_of_cell_L]

        F_limited_at_interfaces[:, k_interface_idx] = f_interface_LF + phi_to_use_vec * (f_interface_LWR - f_interface_LF)

    for idx_cell_active in range(N_active):
        F_lim_right_interface = F_limited_at_interfaces[:, idx_cell_active + 1]
        F_lim_left_interface  = F_limited_at_interfaces[:, idx_cell_active]
        u_next_active[:, idx_cell_active] = (u_current_active[:, idx_cell_active] -
                                            (dt_sys / dx_grid) * (F_lim_right_interface - F_lim_left_interface))
    return u_next_active
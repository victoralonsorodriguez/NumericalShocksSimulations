"""
Module for core hydrodynamic utility functions.

Provides functions for:
- Converting between primitive and conservative variables.
- Calculating physical fluxes.
- Calculating the CFL-based timestep.
- Applying boundary conditions.
"""

import numpy as np
from typing import Tuple

import simulation_constants as constants 

def get_primitive_vars(
    rho_arr: np.ndarray,
    rhov_arr: np.ndarray,
    rhoe_arr: np.ndarray,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate primitive variables (density, velocity, pressure) from conservative variables.

    Parameters
    ----------
    rho_arr : np.ndarray
        Array of densities (conservative variable rho).
    rhov_arr : np.ndarray
        Array of momentum densities (conservative variable rho*v).
    rhoe_arr : np.ndarray
        Array of total energy densities (conservative variable rho*e).
    gamma : float
        Adiabatic index of the gas.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - rho_arr (np.ndarray): Density array (same as input, returned for consistency).
        - v_arr (np.ndarray): Velocity array.
        - P_arr (np.ndarray): Pressure array.
    """
    # Ensure density is positive for stable calculations, avoiding division by zero.
    rho_for_calc = np.maximum(rho_arr.copy(), constants.R_DENOMINATOR_FLOOR)

    # Velocity: v = (rho*v) / rho
    v_arr = rhov_arr / rho_for_calc

    # Pressure: P = (gamma - 1) * (rho*e - 0.5 * rho * v^2)
    # rho_u_arr is the internal energy density
    rho_u_arr = rhoe_arr - 0.5 * rho_for_calc * v_arr**2 
    P_arr = (gamma - 1.0) * rho_u_arr
    # Ensure pressure is non-negative.
    P_arr = np.maximum(P_arr, constants.R_DENOMINATOR_FLOOR) 

    return rho_arr, v_arr, P_arr

def get_conservative_vars(
    rho_arr: np.ndarray,
    v_arr: np.ndarray,
    P_arr: np.ndarray,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate conservative variables (density, momentum, total energy) from primitive variables.

    Parameters
    ----------
    rho_arr : np.ndarray
        Array of densities.
    v_arr : np.ndarray
        Array of velocities.
    P_arr : np.ndarray
        Array of pressures.
    gamma : float
        Adiabatic index of the gas.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - rho_arr (np.ndarray): Density array (same as input).
        - rhov_arr (np.ndarray): Momentum density array.
        - rhoe_arr (np.ndarray): Total energy density array.
    """
    # Momentum density: rho*v
    rhov_arr = rho_arr * v_arr

    # Total energy density: rho*e = rho_internal_energy + 0.5 * rho * v^2
    # where rho_internal_energy = P / (gamma - 1)
    # Ensure pressure is non-negative for internal energy calculation.
    safe_P_arr = np.maximum(P_arr, constants.R_DENOMINATOR_FLOOR) 
    rho_internal_energy_arr = safe_P_arr / (gamma - 1.0)
    rhoe_arr = rho_internal_energy_arr + 0.5 * rho_arr * v_arr**2

    return rho_arr, rhov_arr, rhoe_arr

def get_physical_fluxes_at_cell_centers(
    rho_arr: np.ndarray,
    v_arr: np.ndarray,
    P_arr: np.ndarray,
    rhoe_arr: np.ndarray # rhoe_arr is total energy density (rho*e)
) -> np.ndarray:
    """
    Calculate the physical flux vector F(U) at cell centers.

    The flux vector F(U) is defined as:
    F(U) = [rho*v, rho*v^2 + P, (rho*e + P)*v]

    Parameters
    ----------
    rho_arr : np.ndarray
        Array of densities at cell centers.
    v_arr : np.ndarray
        Array of velocities at cell centers.
    P_arr : np.ndarray
        Array of pressures at cell centers.
    rhoe_arr : np.ndarray
        Array of total energy densities (rho*e) at cell centers.

    Returns
    -------
    np.ndarray
        Array of physical fluxes, shape (3, N_cells), where N_cells is the length of input arrays.
        The rows correspond to the components of F(U).
    """
    f0: np.ndarray = rho_arr * v_arr
    f1: np.ndarray = rho_arr * v_arr**2 + P_arr
    f2: np.ndarray = (rhoe_arr + P_arr) * v_arr # Flux of total energy
    return np.array([f0, f1, f2])

def calculate_dt_cfl(
    rho_arr: np.ndarray,
    v_arr: np.ndarray,
    P_arr: np.ndarray,
    gamma: float,
    dx: float,
    CFL_number: float
) -> float:
    """
    Calculate the timestep dt based on the Courant-Friedrichs-Lewy (CFL) condition.

    dt = CFL * dx / max_signal_speed
    where max_signal_speed = max(|v| + c_s) over all cells, and c_s is the sound speed.
    """
    # Ensure density and pressure are positive for stable sound speed calculation.
    safe_rho = np.maximum(rho_arr, constants.R_DENOMINATOR_FLOOR)
    safe_P = np.maximum(P_arr, constants.R_DENOMINATOR_FLOOR)

    # Calculate sound speed c_s = sqrt(gamma * P / rho) for all cells.
    cs_arr = np.sqrt(gamma * safe_P / safe_rho)
    # Determine the maximum characteristic speed in the domain: max(|v| + c_s).
    max_signal_speed = np.max(np.abs(v_arr) + cs_arr)

    # Avoid division by zero or excessively large dt if max_signal_speed is very small.
    if max_signal_speed < constants.R_DENOMINATOR_FLOOR: 
        # Use a small positive value for max_signal_speed if it's effectively zero
        # to prevent dt from becoming infinite or causing division by zero.
        # This might happen in a completely static, uniform medium.
        max_signal_speed = constants.R_DENOMINATOR_FLOOR 

    return CFL_number * dx / max_signal_speed

def apply_boundary_conditions(
    u_conserved_with_ghosts: np.ndarray,
    num_ghost_cells: int,
    bc_type: str
) -> np.ndarray:
    """
    Apply boundary conditions to the array of conservative variables including ghost cells.

    Modifies 'u_conserved_with_ghosts' in-place.

    Parameters
    ----------
    u_conserved_with_ghosts : np.ndarray
        Array of conservative variables (shape: 3 x N_total_cells) including ghost cells.
    num_ghost_cells : int
        Number of ghost cells on each side of the domain.
    bc_type : str
        Type of boundary condition ('periodic' or 'outflow'/'constant_derivative_zero').

    Returns
    -------
    np.ndarray
        The array of conservative variables with boundary conditions applied.
    """
    # The number of active cells can be derived, but not strictly needed here.
    # N_active = u_conserved_with_ghosts.shape[1] - 2 * num_ghost_cells

    if bc_type == 'periodic':
        # For periodic boundary conditions, values from one end of the active domain
        # are copied to the ghost cells at the other end.
        for i_var in range(u_conserved_with_ghosts.shape[0]): # Iterate over rho, rhov, rhoe
            for i_ghost in range(num_ghost_cells):
                # Left ghost cells get values from the right end of the active domain
                u_conserved_with_ghosts[i_var, i_ghost] = u_conserved_with_ghosts[i_var, -2*num_ghost_cells + i_ghost]
                # Right ghost cells get values from the left end of the active domain
                u_conserved_with_ghosts[i_var, -num_ghost_cells + i_ghost] = u_conserved_with_ghosts[i_var, num_ghost_cells + i_ghost]
    
    elif bc_type == 'outflow' or bc_type == 'constant_derivative_zero':
        # For outflow (zero-gradient) conditions, the values in the ghost cells
        # are set to be the same as the nearest active cell.
        for i_var in range(u_conserved_with_ghosts.shape[0]): # Iterate over rho, rhov, rhoe
            for i_ghost in range(num_ghost_cells):
                # Left ghost cells take the value of the first active cell
                u_conserved_with_ghosts[i_var, i_ghost] = u_conserved_with_ghosts[i_var, num_ghost_cells]
                # Right ghost cells take the value of the last active cell
                u_conserved_with_ghosts[i_var, -1-i_ghost] = u_conserved_with_ghosts[i_var, -num_ghost_cells-1]
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    return u_conserved_with_ghosts
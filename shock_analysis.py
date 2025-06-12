"""
Comprehensive analysis script for the Sod shock tube problem.

This script automates two key analyses using the 1D hydrodynamic simulation:
1.  Measures the post-shock pressure (p1) to calculate the Mach number.
2.  Computes the numerical shock speed by tracking the shock front over time.
"""

import numpy as np
import argparse
from typing import Tuple

import problem_setup
import hydro_utils
import numerical_schemes
import simulation_constants as constants
import cli_parser
import config_loader

def find_wave_positions(
    U_state: np.ndarray,
    x_grid: np.ndarray,
    gamma: float,
    x_diaphragm: float 
) -> Tuple[float, float]:
    """
    Finds the positions of the shock front and contact discontinuity.

    This simplified method uses pressure and density gradients. The shock
    is sought in the right half of the domain (max pressure gradient),
    and the contact discontinuity to the left of the shock (min density gradient).
    The 'x_diaphragm' parameter is available for fallback logic if needed.
    """
    rho, _, P = hydro_utils.get_primitive_vars(
        U_state[0, :], U_state[1, :], U_state[2, :], gamma
    )
    pressure_gradient = np.gradient(P)
    density_gradient = np.gradient(rho)

    # Search for shock in the right half of the domain
    mid_point_idx = len(x_grid) // 2
    if mid_point_idx >= len(pressure_gradient): # Handle very small grids
        mid_point_idx = 0
    
    search_region_shock = pressure_gradient[mid_point_idx:]
    if len(search_region_shock) == 0: # Fallback if search region is empty
        shock_idx_absolute = np.argmax(pressure_gradient) if len(pressure_gradient) > 0 else (len(x_grid) -1 if len(x_grid) > 0 else 0)
    else:
        shock_idx_relative = np.argmax(search_region_shock)
        shock_idx_absolute = mid_point_idx + shock_idx_relative
    
    shock_pos = x_grid[shock_idx_absolute]

    # Search for contact discontinuity to the left of the shock
    valid_indices_for_contact_search = np.where(x_grid < shock_pos)[0]
    if len(valid_indices_for_contact_search) > 0:
        density_gradient_in_search_region = density_gradient[valid_indices_for_contact_search]
        if len(density_gradient_in_search_region) > 0:
            contact_idx_relative_to_search = np.argmin(density_gradient_in_search_region)
            contact_idx_absolute = valid_indices_for_contact_search[contact_idx_relative_to_search]
            contact_pos = x_grid[contact_idx_absolute]
        else: # Fallback if density gradient search region is empty
            contact_pos = x_grid[0] if len(x_grid) > 0 else 0.0
    else: # Fallback if no points are found to the left of the shock
        contact_pos = x_grid[0] if len(x_grid) > 0 else 0.0
        # Additional fallback using x_diaphragm if shock is not at the very beginning
        if shock_pos > (x_grid[0] if len(x_grid) > 0 else 0.0): 
             contact_pos = min(x_diaphragm, x_grid[len(x_grid)//3] if len(x_grid) > 0 else 0.0) 

    # Ensure contact_pos is not to the right of shock_pos and within bounds
    if contact_pos >= shock_pos:
        contact_pos = max(x_grid[0] if len(x_grid) > 0 else 0.0, shock_pos - ( (x_grid[1]-x_grid[0]) if len(x_grid)>1 else 0.02) )
    contact_pos = max(x_grid[0] if len(x_grid) > 0 else 0.0, contact_pos)


    return shock_pos, contact_pos


def measure_post_shock_pressure(
    P_active: np.ndarray,
    x_active: np.ndarray,
    shock_pos: float,
    contact_pos: float,
    min_plateau_width_cells: int = 2 
) -> float:
    """
    Measures the average pressure in the post-shock plateau region.
    Uses a simplified buffer strategy to define the plateau.
    """
    region_width = shock_pos - contact_pos
    
    # Define plateau region using a fixed buffer, with fallback for narrow regions
    buffer_abs = 0.02 
    if region_width > 2 * buffer_abs:
        plateau_start = contact_pos + buffer_abs
        plateau_end = shock_pos - buffer_abs
    else: 
        buffer_frac_small = 0.1 # Use a smaller fraction if region is narrow
        buffer = region_width * buffer_frac_small
        plateau_start = contact_pos + buffer
        plateau_end = shock_pos - buffer

    if plateau_start >= plateau_end or region_width <= 1e-9: # Check for invalid or zero-width plateau
        # Fallback: use the entire region between contact and shock
        minimal_region_indices = np.where((x_active > contact_pos) & (x_active < shock_pos))[0]
        if len(minimal_region_indices) >= 1:
            return np.mean(P_active[minimal_region_indices])
        return np.nan # Cannot measure p1 if region is still invalid

    plateau_indices = np.where((x_active >= plateau_start) & (x_active <= plateau_end))[0]
    
    if len(plateau_indices) < min_plateau_width_cells:
        # Fallback if buffered plateau is too small: use entire region
        fallback_indices = np.where((x_active > contact_pos) & (x_active < shock_pos))[0]
        if len(fallback_indices) >= 1:
            return np.mean(P_active[fallback_indices])
        return np.nan # Cannot measure p1 reliably

    p1 = np.mean(P_active[plateau_indices])
    return p1

def run_simulation_to_time(
    U_initial: np.ndarray,
    t_end: float,
    args: argparse.Namespace,
    dx: float,
    gamma: float
) -> Tuple[np.ndarray, float]:
    """
    Runs the 1D hydrodynamic simulation from an initial state until a specified end time.

    This function encapsulates the main simulation loop, advancing the solution
    step-by-step using the selected numerical scheme and CFL-based timestepping.

    Parameters
    ----------
    U_initial : np.ndarray
        Initial state in conservative variables, including ghost cells.
    t_end : float
        The simulation time to stop at.
    args : argparse.Namespace
        Simulation configuration arguments (e.g., cfl_number, scheme).
    dx : float
        Grid spacing.
    gamma : float
        Adiabatic index of the gas.

    Returns
    -------
    Tuple[np.ndarray, float]
        - U_current (np.ndarray): Final state in conservative variables at t_end.
        - current_t (float): The actual simulation time reached (should be close to t_end).
    """
    U_current = U_initial.copy()
    current_t = 0.0
    
    # Select the numerical scheme based on parsed arguments
    if args.scheme == "LWR": scheme_func = numerical_schemes.lax_wendroff_ritchmyer_step
    elif args.scheme == "LF": scheme_func = numerical_schemes.lax_friedrichs_step
    elif args.scheme == "LIM": scheme_func = numerical_schemes.limiter_scheme_step
    else: raise ValueError(f"Unknown scheme specified in args: {args.scheme}")

    #--- Core Simulation Loop (Non-Interactive) ---#
    while current_t < t_end:
        rho_active = U_current[0, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]
        rhov_active = U_current[1, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]
        rhoe_active = U_current[2, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]

        _, v_active, P_active = hydro_utils.get_primitive_vars(
            rho_active, rhov_active, rhoe_active, gamma
        )

        dt = hydro_utils.calculate_dt_cfl(
            rho_active, v_active, P_active, gamma, dx, args.cfl_number
        )
        
        # Adjust timestep to not overshoot the target t_end.
        if current_t + dt > t_end: dt = t_end - current_t
        if dt < 1e-12: # Prevent excessively small timesteps if simulation stagnates.
            # print(f"Warning: Timestep too small ({dt:.2e}) at t={current_t:.4f}. Breaking loop.")
            break

        U_next_active = scheme_func(U_current, dt, dx, gamma)
        U_current[:, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS] = U_next_active
        U_current = hydro_utils.apply_boundary_conditions(
            U_current, constants.NUM_GHOST_CELLS, constants.BC_TYPE_CONFIG
        )
        current_t += dt
        
    return U_current, current_t

if __name__ == "__main__":
    
    print("\n#--- Starting Shock Results Analysis ---#")
    print("-----------------------------------------")
    
    #--- 1. Setup Simulation Parameters for Sod Problem Analysis ---#
    arg_parser_obj = cli_parser.get_parser()
    args = arg_parser_obj.parse_args() 

    # Load general simulation parameters from the config file specified by --sim_config_file
    # This file might contain t_measure1 and t_measure2
    config_sim_params = config_loader.load_simulation_parameters_config(args.sim_config_file)
    if config_sim_params is None:
        config_sim_params = {}

    # Determine final t_measure1 with priority: CMD > Config > Argparse Default
    cmd_t_measure1 = args.t_measure1
    config_t_measure1 = config_sim_params.get('t_measure1')
    default_t_measure1 = arg_parser_obj.get_default('t_measure1')

    if cmd_t_measure1 != default_t_measure1: # CMD line has priority
        final_t_measure1 = cmd_t_measure1
    elif config_t_measure1 is not None: # Config file is next
        final_t_measure1 = config_t_measure1
    else: # Fallback to argparse default
        final_t_measure1 = default_t_measure1

    # Determine final t_measure2 (for t_end_speed_check) with the same priority
    cmd_t_measure2 = args.t_measure2
    config_t_measure2 = config_sim_params.get('t_measure2')
    default_t_measure2 = arg_parser_obj.get_default('t_measure2')

    if cmd_t_measure2 != default_t_measure2: # CMD line has priority
        final_t_measure2 = cmd_t_measure2
    elif config_t_measure2 is not None: # Config file is next
        final_t_measure2 = config_t_measure2
    else: # Fallback to argparse default
        final_t_measure2 = default_t_measure2
    
    ic_params = problem_setup.get_initial_conditions_parameters(args)
    x_active, _, rho_init, v_init, P_init, dx = problem_setup.setup_grid_and_initial_state(args, ic_params)
    
    x_diaphragm_val = ic_params.get('x_diaphragm', 0.5)

    rho_g, rhov_g, rhoe_g = hydro_utils.get_conservative_vars(
        rho_init, v_init, P_init, constants.GAMMA_IDEAL_GAS
    )
    U_initial_state = np.array([rho_g, rhov_g, rhoe_g])
    U_initial_state = hydro_utils.apply_boundary_conditions(
        U_initial_state, constants.NUM_GHOST_CELLS, constants.BC_TYPE_CONFIG
    )
    print("\n#--- Analysis Parameters ---#")
    print(f"    Analysis using IC file: {args.ic_config_file.split('/')[-1]}")
    print(f"    Scheme:  {args.scheme}")
    print(f"    Ncells:  {args.ncells}")
    print(f"    CFL:     {args.cfl_number}")
    print(f"    Epsilon: {args.epsilon}") 
    print(f"    Analysis t_measure1: {final_t_measure1:.4f}")
    print(f"    Analysis t_measure2: {final_t_measure2:.4f}")
    
    print(f"\nRunning simulation to t = {final_t_measure1:.5f} s for analysis...")
    U_at_t_measure, _ = run_simulation_to_time(U_initial_state, final_t_measure1, args, dx, constants.GAMMA_IDEAL_GAS)
    U_active = U_at_t_measure[:, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]
    
    #--- 3. Measure Pressures and Calculate Mach Number ---#
    shock_pos, contact_pos = find_wave_positions(U_active, x_active, constants.GAMMA_IDEAL_GAS, x_diaphragm_val)

    _ , _, P_active_at_t_measure = hydro_utils.get_primitive_vars(U_active[0], U_active[1], U_active[2], constants.GAMMA_IDEAL_GAS)
    
    p0 = ic_params['p_r'] 
    p1 = measure_post_shock_pressure(P_active_at_t_measure, x_active, shock_pos, contact_pos) 

    M0 = np.nan 

    if np.isnan(p1) or np.isnan(shock_pos) or np.isnan(contact_pos): 
        print("Error: Could not reliably measure p1 or wave positions. Aborting Mach number analysis.")
    else:
        if p0 < 1e-9: 
            M0 = np.inf 
        else:
            pressure_ratio_p1_p0 = p1 / p0
            # Note: If p1/p0 < 1, this indicates a non-compressive wave or issues in p1 measurement.
            # The Mach number formula might yield complex numbers or NaN if term2 is negative.
            term1 = (constants.GAMMA_IDEAL_GAS + 1) / (2 * constants.GAMMA_IDEAL_GAS)
            term2 = pressure_ratio_p1_p0 + (constants.GAMMA_IDEAL_GAS - 1) / (constants.GAMMA_IDEAL_GAS + 1)
            if term2 < 0: 
                M0 = np.nan # Result would be complex, indicates non-physical shock or error.
            else:
                M0 = np.sqrt(term1 * term2)
    
    print("\n#--- Mach Number Analysis ---#")
    print(f"Pre-shock pressure (p0)    = {p0:.4f}")
    print(f"Post-shock pressure (p1)   = {p1:.4f} (measured)")
    print(f"Shock Position             = {shock_pos:.4f}") 
    print(f"Contact Position           = {contact_pos:.4f}") 
    print(f"Resulting Mach Number (M0) = {M0:.4f}")

    #--- 4. Calculate Numerical Shock Speed ---#
    t_start_speed_check = final_t_measure1 
    t_end_speed_check_val = final_t_measure2 

    x1, _ = find_wave_positions(U_active, x_active, constants.GAMMA_IDEAL_GAS, x_diaphragm_val)

    print(f"\nRunning simulation to t = {t_end_speed_check_val:.5f} s for shock speed analysis...")
    U_at_t_end, _ = run_simulation_to_time(U_initial_state, t_end_speed_check_val, args, dx, constants.GAMMA_IDEAL_GAS)
    U_active_t2 = U_at_t_end[:, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]
    
    x2, _ = find_wave_positions(U_active_t2, x_active, constants.GAMMA_IDEAL_GAS, x_diaphragm_val)

    shock_speed = np.nan 
    if np.isnan(x1) or np.isnan(x2) :
        print("Error: Could not reliably determine shock positions for speed calculation.")
    elif t_end_speed_check_val - t_start_speed_check < 1e-9: 
        # Time interval too small for reliable speed calculation.
        pass 
    else:
        shock_speed = (x2 - x1) / (t_end_speed_check_val - t_start_speed_check)
    
    print("\n#--- Shock Speed Analysis ---#")
    print(f"Shock position at t={t_start_speed_check:.5f}s: x1={x1:.4f}") 
    print(f"Shock position at t={t_end_speed_check_val:.5f}s: x2={x2:.4f}") 
    print(f"Numerical Shock Speed = {shock_speed:.5f}")
        
    #--- 5. Final Comparison (Theoretical vs. Numerical) ---#
    print("\n#--- Final Comparison ---#")
    
    rho0 = ic_params['rho_r']
    c0 = np.sqrt(constants.GAMMA_IDEAL_GAS * p0 / rho0) if p0 > 0 and rho0 > 0 else np.nan
    
    theoretical_shock_speed = np.nan 
    if not (np.isnan(M0) or np.isnan(c0)):
        theoretical_shock_speed = M0 * c0
    
    relative_error = np.nan 
    if not (np.isnan(shock_speed) or np.isnan(theoretical_shock_speed) or np.abs(theoretical_shock_speed) < 1e-9):
        relative_error = (np.abs(shock_speed - theoretical_shock_speed) / np.abs(theoretical_shock_speed)) * 100
    
    print(f"Sound speed in unperturbed medium (c0)  = {c0:.4f}")
    print(f"Theoretical shock speed (M0 * c0)       = {theoretical_shock_speed:.4f}")
    print(f"Numerical Shock Speed (from simulation) = {shock_speed:.4f}")
    if np.isnan(relative_error):
        print("\nRelative Error: Undetermined (due to issues in prior calculations).")
    else:
        print(f"\nRelative Error: {relative_error:.2f}%")
        if relative_error < 10: 
            print("    Do they agree? Yes, the values are reasonably close.")
        else: 
            print("    Do they agree? No, the error value is somewhat large.")
            
    print("\n    Note: This simplified analysis uses a basic method for wave localization and pressure measurement.")
    print("    Discrepancies can arise from numerical dissipation, the method of shock localization, "
        "and the precision of p1 measurement, especially for sharper schemes.")
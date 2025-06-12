"""
Module for setting up the simulation problem.

This module is responsible for:
- Loading initial condition parameters from a specified configuration file.
- Applying command-line overrides to these initial condition parameters.
- Generating the computational grid (cell centers and spacing).
- Calculating the initial state arrays (density, velocity, pressure) on the grid,
  including smoothing the initial discontinuity.
"""

import numpy as np
import argparse
from typing import Dict, Tuple

import config_loader
import simulation_constants as constants

def get_initial_conditions_parameters(args: argparse.Namespace) -> Dict[str, float]:
    """
    Load specific initial condition parameters from the IC config file 
    and apply command-line overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, which include the path to the IC config file
        and any command-line overrides for ICs.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the final initial condition parameters 
        (e.g., rho_l, p_l, v_l, rho_r, p_r, v_r, x_diaphragm, simulation_name_keyword).
    """
    # Load base initial condition parameters from the specified IC configuration file.
    base_ic_params = config_loader.load_initial_conditions_config(args.ic_config_file)

    if base_ic_params is None:
        # If the config file loading failed, use hardcoded defaults for a standard Sod tube.
        print(f"Warning: Could not load ICs from '{args.ic_config_file}'. "
              "Using hardcoded default ICs for Sod tube.")
        base_ic_params = {
            'rho_l': 1.0, 'p_l': 1.0, 'v_l': 0.0,
            'rho_r': 0.125, 'p_r': 0.1, 'v_r': 0.0,
            'x_diaphragm': 0.5,
            'simulation_name_keyword': 'SodDefault'
        }

    # Apply command-line overrides. Command-line arguments (if not None) take precedence.
    final_ic_params = base_ic_params.copy()
    if args.rho_L is not None: final_ic_params['rho_l'] = args.rho_L
    if args.P_L is not None:   final_ic_params['p_l'] = args.P_L
    if args.v_L is not None:   final_ic_params['v_l'] = args.v_L
    if args.rho_R is not None: final_ic_params['rho_r'] = args.rho_R
    if args.P_R is not None:   final_ic_params['p_r'] = args.P_R
    if args.v_R is not None:   final_ic_params['v_r'] = args.v_R
    if args.x_diaphragm is not None: final_ic_params['x_diaphragm'] = args.x_diaphragm
    # simulation_name_keyword is handled by config_loader, but ensure it's present.
    if 'simulation_name_keyword' not in final_ic_params:
        final_ic_params['simulation_name_keyword'] = "UnknownSim"


    return final_ic_params

def setup_grid_and_initial_state(
    args_merged: argparse.Namespace,
    ic_params_specific: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Set up the computational grid and calculate the initial state arrays.
    Initial conditions are set based on 'ic_params_specific' and smoothed 
    using 'args_merged.epsilon'.

    Parameters
    ----------
    args_merged : argparse.Namespace
        Fully merged command-line arguments and simulation parameters 
        (contains ncells, epsilon).
    ic_params_specific : Dict[str, float]
        Dictionary of specific initial condition parameters (rho_l, p_l, etc.).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        - x_active_grid (np.ndarray): Coordinates of active cell centers.
        - x_full_grid (np.ndarray): Coordinates of all cell centers (including ghosts).
        - rho_initial_full (np.ndarray): Initial density on the full grid.
        - v_initial_full (np.ndarray): Initial velocity on the full grid.
        - P_initial_full (np.ndarray): Initial pressure on the full grid.
        - dx_grid_val (float): Grid spacing.
    """
    # Get grid parameters from merged arguments and constants
    N_active_cells = args_merged.ncells
    num_ghost_cells = constants.NUM_GHOST_CELLS
    x_min = constants.X_MIN
    x_max = constants.X_MAX

    dx_grid_val = (x_max - x_min) / N_active_cells

    # Define grid point locations (cell centers)
    x_active_grid = np.linspace(x_min + dx_grid_val / 2.0, x_max - dx_grid_val / 2.0, N_active_cells)
    x_full_grid = np.linspace(
        x_min - (num_ghost_cells - 0.5) * dx_grid_val,
        x_max + (num_ghost_cells - 0.5) * dx_grid_val,
        N_active_cells + 2 * num_ghost_cells
    )

    # Initialize arrays for the initial state on the full grid
    rho_initial_full = np.zeros_like(x_full_grid)
    v_initial_full = np.zeros_like(x_full_grid)
    P_initial_full = np.zeros_like(x_full_grid)

    # Calculate smoothed initial conditions profile using the tanh function.
    epsilon_smoothing_val = args_merged.epsilon 
    # Effective smoothing width is epsilon_smoothing_val * dx_grid_val
    effective_epsilon_dx = epsilon_smoothing_val * dx_grid_val
    # Prevent division by zero or very small numbers for smoothing width
    if np.abs(effective_epsilon_dx) < constants.R_DENOMINATOR_FLOOR:
        effective_epsilon_dx = constants.R_DENOMINATOR_FLOOR * np.sign(effective_epsilon_dx) if effective_epsilon_dx != 0 else constants.R_DENOMINATOR_FLOOR

    for i, x_val in enumerate(x_full_grid):
        tanh_term = np.tanh((x_val - ic_params_specific['x_diaphragm']) / effective_epsilon_dx)
        rho_initial_full[i] = ic_params_specific['rho_r'] + 0.5 * (1.0 - tanh_term) * (ic_params_specific['rho_l'] - ic_params_specific['rho_r'])
        P_initial_full[i]   = ic_params_specific['p_r']   + 0.5 * (1.0 - tanh_term) * (ic_params_specific['p_l']   - ic_params_specific['p_r'])
        v_initial_full[i]   = ic_params_specific['v_r']   + 0.5 * (1.0 - tanh_term) * (ic_params_specific['v_l']   - ic_params_specific['v_r'])

    return x_active_grid, x_full_grid, rho_initial_full, v_initial_full, P_initial_full, dx_grid_val
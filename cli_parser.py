"""
Module for parsing command-line arguments for the Sod Shock Tube simulation.

Defines arguments for numerical scheme, grid size, CFL number, output settings,
and initial conditions, including the option to specify a configuration file
and override its values via command-line flags.
"""

import argparse
import os # For constructing default paths

def get_parser() -> argparse.ArgumentParser:
    """
    Defines and returns the ArgumentParser object for the simulation.

    Includes arguments for selecting the numerical scheme, grid resolution,
    CFL number, output format and directory, and initial conditions.
    Allows specifying an initial conditions configuration file and overriding
    specific values from the command line.
    """
    parser = argparse.ArgumentParser(
        description="1D Hydrodynamics Sod Shock Tube Simulation with configurable parameters."
    )

    #--- Simulation Core Parameters ---#
    parser.add_argument(
        '-s', '--scheme',
        type=str,
        default='LIM',
        choices=['LF', 'LWR', 'LIM'],
        help=("Numerical scheme to use: LF (Lax-Friedrichs), LWR (Lax-Wendroff-Ritchmyer), "
              "LIM (Minmod Limiter) - Default.")
    )
    parser.add_argument(
        '-n', '--ncells',
        type=int,
        default=1000,
        help="Number of active cells in the domain. Default: 1000."
    )
    parser.add_argument(
        '-cfl', '--cfl_number',
        type=float,
        default=0.9,
        help="CFL number for timestep calculation. Default: 0.9."
    )
    
    parser.add_argument(
        '-eps', '--epsilon',
        type=float,
        default=5.0,
        help="Smoothing factor for the initial conditions (multiples of dx). Default: 5.0."
    )
    parser.add_argument(
        '-dt', '--dt_plot',
        type=float,
        default=0.00001, # Default if not in config and not in cmd line
        help="Time between plot updates in simulation time units. Default: 0.00001"
    )
    parser.add_argument(
        '-dts', '--dt_save',
        type=float,
        default=0.0, # Default to 0.0 (disabled) if not in config or cmd line
        help="Simulation time interval for auto-saving frames. 0 or negative to disable. Default: 0.0"
    )
    parser.add_argument(
        '-tf', '--t_final',
        type=float,
        default=None, # Default to None (continuous animation) if not set
        help="Final simulation time to stop the simulation and save a frame. Default: None (continuous)."
    )

    #--- Analysis-specific Time Parameters (for perform_shock_analysis.py) ---#
    parser.add_argument(
        '-tm1', '--t_measure1',
        type=float,
        default=0.010,
        help="First time point for shock analysis (e.g., post-shock pressure). Default: 0.010"
    )
    parser.add_argument(
        '-tm2', '--t_measure2',
        type=float,
        default=0.014,
        help="Second time point for shock analysis (e.g., for shock speed calculation). Default: 0.014"
    )

    #--- Output Settings ---#
    parser.add_argument(
        '-fmt', '--output_format',
        type=str,
        default='png',
        choices=['pdf', 'png', 'both'],
        help="Output format for saved figures ('pdf', 'png', or 'both'). Default: png."
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=None, # Default will be constructed in main_simulation.py if not provided
        help="Directory to save output frames. Default: {simulation_name_keyword}_frames."
    )

    #--- Simulation Name Keyword ---#
    parser.add_argument(
        '-snk', '--simulation_name_keyword',
        type=str,
        default="NumSche", # Default if not overridden by IC file or command line
        help="Keyword to identify the simulation, used as a base for output filenames and default directory. Default: NumSche."
    )

    #--- Initial Conditions Configuration ---#
    # Renamed from --config_file to --ic_config_file for clarity
    parser.add_argument(
        '-ic','--ic_config_file', # Renamed
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'ic_config_sod.ini'), # Updated default filename
        help="Path to the initial conditions configuration file. Default: ic_config_sod.ini."
    )
    # Argument to specify a configuration file for general simulation parameters
    parser.add_argument(
        '--sim_config_file',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'simulation_config.ini'),
        help="Path to the general simulation parameters configuration file. Default: simulation_config.ini."
    )

    # Arguments to override initial conditions from the command line
    parser.add_argument('--rho_L', type=float, default=None, help="Override: Density in the left state.")
    parser.add_argument('--P_L',   type=float, default=None, help="Override: Pressure in the left state.")
    parser.add_argument('--v_L',   type=float, default=None, help="Override: Velocity in the left state.")
    parser.add_argument('--rho_R', type=float, default=None, help="Override: Density in the right state.")
    parser.add_argument('--P_R',   type=float, default=None, help="Override: Pressure in the right state.")
    parser.add_argument('--v_R',   type=float, default=None, help="Override: Velocity in the right state.")
    parser.add_argument('--x_diaphragm', type=float, default=None, help="Override: Position of the diaphragm.")
    
    return parser

def parse_arguments() -> argparse.Namespace:
    """Define and parse command-line arguments for the simulation."""
    parser = get_parser()
    args = parser.parse_args()
    return args
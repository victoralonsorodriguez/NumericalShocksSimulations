"""
Module for loading simulation configuration from an INI file.

Provides functions to read initial conditions and other parameters
from a specified configuration file.
"""

import configparser
import os
from typing import Dict, Optional, Any

def load_initial_conditions_config(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load initial condition parameters from a specified INI configuration file.

    Reads parameters under the [InitialConditions] section.
    Expected numeric parameters: rho_l, p_l, v_l, rho_r, p_r, v_r, x_diaphragm.
    Expected string parameter: simulation_name_keyword.
    
    Parameters
    ----------
    filepath : str
        The path to the INI configuration file.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the initial condition parameters. Numeric values
        are floats, string values are strings.
        Returns None if the file is not found or the section is missing.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Configuration file not found at '{filepath}'.")
        return None

    config = configparser.ConfigParser()
    try:
        config.read(filepath)
    except configparser.Error as e:
        print(f"Error parsing configuration file '{filepath}': {e}")
        return None

    if 'InitialConditions' not in config:
        print(f"Warning: Section [InitialConditions] not found in '{filepath}'.")
        return None

    ic_params: Dict[str, Any] = {} # Changed to Dict[str, Any] to accommodate mixed types
    expected_numeric_keys = ['rho_l', 'p_l', 'v_l', 'rho_r', 'p_r', 'v_r', 'x_diaphragm']

    for key in expected_numeric_keys:
        try:
            value_str = config.get('InitialConditions', key)
            # Strip inline comments (anything after # or ;)
            if '#' in value_str:
                value_str = value_str.split('#', 1)[0]
            if ';' in value_str:
                value_str = value_str.split(';', 1)[0]
            ic_params[key] = float(value_str.strip()) # Convert to float after stripping
        except (configparser.NoOptionError, ValueError) as e:
            print(f"Warning: Numeric parameter '{key}' not found or invalid in [InitialConditions] in '{filepath}': {e}. Setting to None.")
            # Allow missing parameters; they can be caught later or defaulted by the calling function.
            ic_params[key] = None # Explicitly set to None if missing or invalid

    #--- Load String Parameters ---#
    expected_string_keys = ['simulation_name_keyword']
    for key in expected_string_keys:
        try:
            value_str = config.get('InitialConditions', key)
            if '#' in value_str: value_str = value_str.split('#', 1)[0]
            if ';' in value_str: value_str = value_str.split(';', 1)[0]
            ic_params[key] = value_str.strip()
        except configparser.NoOptionError:
            # Provide a default if the key is missing in the config file
            print(f"Warning: String parameter '{key}' not found in [InitialConditions] in '{filepath}'. Defaulting to 'UnknownSim'.")
            ic_params[key] = "UnknownSim" # Default placeholder

    return ic_params

def load_simulation_parameters_config(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load general simulation parameters from a specified INI configuration file.

    Reads parameters under the [SimulationParameters] section.
    Parameters are read and converted to their expected types (int, float, str).

    Parameters
    ----------
    filepath : str
        The path to the INI configuration file.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the simulation parameters with appropriate types.
        Returns None if the file or section is not found.
    """
    if not os.path.exists(filepath):
        # This warning is generally useful, even if another function also checks.
        # print(f"Info: Simulation parameters config file not found at '{filepath}'. Will rely on defaults/CMD args.")
        return None # Return None if the file itself doesn't exist.

    config = configparser.ConfigParser()
    try:
        config.read(filepath)
    except configparser.Error as e:
        print(f"Error parsing configuration file '{filepath}': {e}")
        return None

    if 'SimulationParameters' not in config:
        # This is a common case if the user only provides an IC file.
        # print(f"Info: Section [SimulationParameters] not found in '{filepath}'. Using argparse defaults for these parameters.")
        return None # Return None if the section is missing.

    sim_params: Dict[str, Any] = {}
    # Define expected keys and their target types for conversion.
    expected_params = {
        'scheme': str, 'ncells': int, 'cfl_number': float,
        'epsilon': float, 'output_format': str, 'output_dir': str,
        'dt_plot': float, 'dt_save': float, 't_final': float,
        't_measure1': float, 't_measure2': float
    }

    for key, param_type in expected_params.items():
        if config.has_option('SimulationParameters', key):
            try:
                value_str = config.get('SimulationParameters', key)
                # Strip inline comments (anything after # or ;)
                if '#' in value_str:
                    value_str = value_str.split('#', 1)[0]
                if ';' in value_str:
                    value_str = value_str.split(';', 1)[0]
                value_str = value_str.strip()

                # Handle empty strings for optional float/int parameters by setting them to None
                if not value_str and param_type in (float, int):
                    sim_params[key] = None
                    # print(f"Info: Optional parameter '{key}' is empty in [SimulationParameters] in '{filepath}'. Setting to None.")
                    continue

                if param_type == int: sim_params[key] = int(value_str)
                elif param_type == float: sim_params[key] = float(value_str)
                else: sim_params[key] = value_str # For str type
            except ValueError as e: # Catch specific error for conversion
                print(f"Warning: Parameter '{key}' has invalid value for type {param_type.__name__} in [SimulationParameters] in '{filepath}': {e}. Setting to None.")
                sim_params[key] = None 
            except configparser.NoOptionError: # Should not happen due to has_option check, but good for safety
                print(f"Warning: Parameter '{key}' unexpectedly not found in [SimulationParameters] in '{filepath}'. Setting to None.")
                sim_params[key] = None
        # If key is not in config, it won't be added to sim_params, allowing defaults elsewhere to take over.
    return sim_params
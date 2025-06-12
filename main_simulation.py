"""
Main script for the 1D Hydrodynamics Sod Shock Tube Simulation.

This script orchestrates the simulation by:
- Parsing command-line arguments and configuration files.
- Setting up the initial problem state and grid.
- Selecting the numerical scheme.
- Running the simulation loop coupled with Matplotlib animation.
- Handling visualization and output of plot frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from typing import Callable, Tuple, Optional, Any
import sys 

# Import all the created modules
import cli_parser
import config_loader
import simulation_constants as constants
import problem_setup
import hydro_utils
import numerical_schemes
import visualization

#--- Global Variables for Simulation State (Managed by Main Script) ---#

_U_current_global: Optional[np.ndarray] = None #: Current conservative state variables (rho, rho*v, rho*e) across the full grid including ghost cells.
_t_current_sim: float = 0.0 #: Current simulation time.
_t_next_plot_update: float = 0.0 #: Simulation time for the next scheduled plot update.
_step_count_sim: int = 0 #: Total number of simulation steps taken.
_t_next_frame_save_main: float = 0.0 #: Simulation time for the next scheduled auto-save of a plot frame.
_dx_grid_val_main: float = 0.0 #: Grid spacing (dx), determined after grid setup.
_args_main: Optional[argparse.Namespace] = None #: Parsed and merged command-line arguments and configuration parameters.

# Scheme-specifics, set after parsing arguments and configuration.
_selected_scheme_func: Optional[Callable] = None #: Function handle for the selected numerical scheme.
_scheme_name_print_main: str = "" #: User-friendly name of the selected numerical scheme for display.

# Grid coordinates for active cells, used in plotting.
_x_active_grid_main: Optional[np.ndarray] = None #: Array of x-coordinates for the centers of active (non-ghost) cells.
# Keyword identifying the specific simulation case, extracted from IC config.
_simulation_name_keyword_main: str = "Sim" #: Keyword from IC config (e.g., "SodStandard") for naming outputs.
_ani_main: Optional[animation.FuncAnimation] = None #: Global reference to the Matplotlib animation object.
_t_final_frame_saved_flag: bool = False #: Flag to indicate if the frame at t_final has already been saved.


def run_simulation_for_plot_frame() -> None:
    """
    Advances the simulation state until the next scheduled plot update time.

    This function contains the core simulation loop that runs between plot
    updates. It iteratively calculates CFL-based timesteps and applies the
    selected numerical scheme to update the conservative state variables.
    It also handles auto-saving of frames if `args.dt_save` is enabled
    and checks if `args.t_final` has been reached.

    Modifies global variables:
        _U_current_global, _t_current_sim, _step_count_sim,
        _t_next_frame_save_main, _t_next_plot_update.
    """
    global _U_current_global, _t_current_sim, _t_next_plot_update
    global _step_count_sim, _dx_grid_val_main, _args_main, _selected_scheme_func 
    global _t_next_frame_save_main, _simulation_name_keyword_main

    if _U_current_global is None or _args_main is None or _selected_scheme_func is None:
        print("Error: Simulation state not properly initialized for run_simulation_for_plot_frame.")
        return

    steps_this_plot_update = 0 # Counter for steps within this plot frame's duration.
    max_steps_per_plot_update = 10000  # Safety break to prevent infinite loops if dt is too small.

    while _t_current_sim < _t_next_plot_update and steps_this_plot_update < max_steps_per_plot_update:
        # Extract active cell data for CFL calculation and scheme update.
        rho_active = _U_current_global[0, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]
        rhov_active = _U_current_global[1, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]
        rhoe_active = _U_current_global[2, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS]

        _, v_active, P_active = hydro_utils.get_primitive_vars(
            rho_active, rhov_active, rhoe_active, constants.GAMMA_IDEAL_GAS
        )

        # Calculate stable timestep based on CFL condition.
        dt_cfl = hydro_utils.calculate_dt_cfl(
            rho_active, v_active, P_active,
            constants.GAMMA_IDEAL_GAS, _dx_grid_val_main, _args_main.cfl_number
        )

        # Determine the actual timestep to advance; do not overshoot the next plot update time.
        dt_to_advance = dt_cfl
        if _t_current_sim + dt_to_advance > _t_next_plot_update:
            dt_to_advance = _t_next_plot_update - _t_current_sim
        
        # If t_final is set, ensure we don't advance past it within this single step.
        if _args_main.t_final is not None and _args_main.t_final > 0:
            if _t_current_sim + dt_to_advance > _args_main.t_final:
                dt_to_advance = _args_main.t_final - _t_current_sim


        if dt_to_advance < 1e-12: # If timestep becomes excessively small, break to avoid stagnation.
            # This can happen if _t_next_plot_update or t_final is very close to _t_current_sim.
            if _t_current_sim >= _t_next_plot_update or \
               (_args_main.t_final is not None and _t_current_sim >= _args_main.t_final):
                break # Exit if we've reached or passed the target time.
            # Otherwise, if dt_to_advance is genuinely tiny but we haven't reached target,
            # it might indicate an issue, but we let it proceed for one more check.
            # If it's still too small and target not reached, the loop condition will handle it.


        # Apply the selected numerical scheme to get the next state of active cells.
        U_next_active = _selected_scheme_func(
            _U_current_global, dt_to_advance, _dx_grid_val_main, constants.GAMMA_IDEAL_GAS
        )

        # Update the active region of the global state array.
        _U_current_global[:, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS] = U_next_active
        # Apply boundary conditions to the ghost cells.
        _U_current_global = hydro_utils.apply_boundary_conditions(
            _U_current_global, constants.NUM_GHOST_CELLS, constants.BC_TYPE_CONFIG 
        )

        # Advance simulation time and counters.
        _t_current_sim += dt_to_advance
        _step_count_sim += 1
        steps_this_plot_update += 1

        # --- Auto-save frame logic ---
        if _args_main.dt_save > 0 and _t_current_sim >= _t_next_frame_save_main:
            if visualization._fig_anim is not None: # Ensure figure object exists for saving.
                print(f"Auto-saving frame for {_simulation_name_keyword_main} at t={_t_current_sim:.4f}...")
                visualization.save_plot_frame(visualization._fig_anim, _t_current_sim, _args_main, _simulation_name_keyword_main)
            _t_next_frame_save_main += _args_main.dt_save # Schedule next auto-save.
                                                        # Ensure this doesn't cause issues if dt_save is very small.
                                                        # It's better to schedule from current time if multiple saves are missed:
                                                        # while _t_next_frame_save_main <= _t_current_sim:
                                                        # _t_next_frame_save_main += _args_main.dt_save

    # --- Check for t_final condition after the loop ---
    if _args_main.t_final is not None and _args_main.t_final > 0 and _t_current_sim >= _args_main.t_final:
        # Ensure current time does not exceed t_final for the state being saved/displayed.
        _t_current_sim = min(_t_current_sim, _args_main.t_final)
        _t_next_plot_update = _t_current_sim # This signals animation_update_manager to stop further simulation.
        return # Exit early, simulation for this frame is done or t_final reached.

    # Normal increment for the next plot update time if t_final not reached.
    _t_next_plot_update = _t_current_sim + _args_main.dt_plot


def animation_update_manager(frame_idx: int) -> Tuple[plt.Line2D, ...]:
    """
    Manages simulation advancement and plot updates for each animation frame.

    This function is called by Matplotlib's `FuncAnimation`. It first checks
    if the final simulation time (`t_final`) has been reached. If so, it stops
    the animation, saves the final frame, and updates the plot to the state
    at `t_final`. Otherwise, it calls `run_simulation_for_plot_frame` to
    advance the simulation state and then updates the plot with the new data.

    Parameters
    ----------
    frame_idx : int
        The current frame index, provided by `FuncAnimation`. Not directly
        used in this function's logic but required by the `FuncAnimation` API.

    Returns
    -------
    Tuple[plt.Line2D, ...]
        A tuple of `Line2D` artists that have been updated. This is required
        by `FuncAnimation` for blitting (though blitting is currently disabled).
    """
    global _U_current_global, _t_current_sim, _args_main, _scheme_name_print_main, _ani_main, _t_final_frame_saved_flag
    global _x_active_grid_main, _simulation_name_keyword_main

    # Check if t_final has been reached and stop animation updates if so.
    if _args_main.t_final is not None and _args_main.t_final > 0 and _t_current_sim >= _args_main.t_final:
        # Ensure we are exactly at t_final if we overshot slightly due to dt_plot.
        _t_current_sim = min(_t_current_sim, _args_main.t_final)

        if not _t_final_frame_saved_flag: # Save frame at t_final only once.
            if _ani_main is not None and _ani_main.event_source is not None:
                _ani_main.event_source.stop() # Stop the animation timer.
            
            print(f"INFO: t_final ({_args_main.t_final:.4f}) reached. Animation stopped. Saving final frame.")
            
            # Ensure the plot displays the state at t_final.
            rho_display_final, v_display_final, P_display_final = hydro_utils.get_primitive_vars(
                _U_current_global[0, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
                _U_current_global[1, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
                _U_current_global[2, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
                constants.GAMMA_IDEAL_GAS
            )
            visualization.update_plot_data(
                _x_active_grid_main, rho_display_final, v_display_final, P_display_final, _t_current_sim,
                _scheme_name_print_main, _simulation_name_keyword_main,
                _args_main.ncells, _args_main.epsilon, _args_main.dt_plot
            )
            if visualization._fig_anim is not None:
                visualization.save_plot_frame(visualization._fig_anim, _t_current_sim, _args_main, _simulation_name_keyword_main)
            print("Please close the plot window manually to exit.")
            _t_final_frame_saved_flag = True # Set flag after stopping and saving.

        return tuple(visualization._lines_anim.values()) if visualization._lines_anim else tuple()

    # Advance the simulation state for the duration of one plot frame.
    # Only run simulation if not already stopped by t_final logic.
    run_simulation_for_plot_frame()


    # Get primitive variables for display.
    rho_display, v_display, P_display = hydro_utils.get_primitive_vars(
        _U_current_global[0, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
        _U_current_global[1, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
        _U_current_global[2, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
        constants.GAMMA_IDEAL_GAS
    )

    # Update the plot data using the visualization module.
    return visualization.update_plot_data(
        _x_active_grid_main, rho_display, v_display, P_display, _t_current_sim,
        _scheme_name_print_main, _simulation_name_keyword_main,
        _args_main.ncells, _args_main.epsilon,
        _args_main.dt_plot
    )


def on_main_animation_close(event: Any) -> None:
    """
    Handles the Matplotlib animation window close event.

    This function is triggered when the user closes the animation window.
    It ensures that the final frame of the simulation is saved, unless
    the simulation reached `t_final` and the frame at `t_final` was
    already saved by `animation_update_manager`.

    Parameters
    ----------
    event : matplotlib.backend_bases.CloseEvent
        The close event object provided by Matplotlib. Not directly used
        in this function's logic but required by the event handler API.
    """
    global _t_current_sim, _args_main, _simulation_name_keyword_main, _t_final_frame_saved_flag

    # Check if t_final was set and if the frame for t_final was already saved.
    was_t_final_reached_and_saved = (_args_main.t_final is not None and 
                                     _args_main.t_final > 0 and 
                                     _t_final_frame_saved_flag and 
                                     np.isclose(_t_current_sim, _args_main.t_final, atol=1e-5))

    if not was_t_final_reached_and_saved:
        # If t_final was not reached, or if it was reached but the flag indicates it wasn't saved via t_final logic
        # (e.g., user closed window before t_final), save the current frame.
        print(f"Main animation window closing. Saving final frame at t={_t_current_sim:.3f}...")
        if visualization._fig_anim is not None and _t_current_sim > 0 and _args_main is not None:
            visualization.save_plot_frame(visualization._fig_anim, _t_current_sim, _args_main, _simulation_name_keyword_main)
            print("Final frame saving process complete.")
        elif _t_current_sim == 0:
            print("Simulation time is 0.0, no frame saved on close.")
    else:
        print(f"Main animation window closing. Frame at t_final={_args_main.t_final:.4f} was already saved.")


if __name__ == "__main__":
    #--- Configuration Setup ---#
    # 1. Get the ArgumentParser object to access defined defaults and parse arguments.
    arg_parser_obj = cli_parser.get_parser()

    # 2. Parse command-line arguments. 
    # args_cmd_line_only will contain values from command line, 
    # or argparse defaults if not provided on command line.
    args_cmd_line_only = arg_parser_obj.parse_args()

    # 3. Load general simulation parameters from the specified simulation_config.ini file.
    config_sim_params = config_loader.load_simulation_parameters_config(args_cmd_line_only.sim_config_file)
    if config_sim_params is None:
        config_sim_params = {} # Use empty dict if file/section not found.

    # 4. Establish final configuration for _args_main by merging sources with defined priority.
    # Priority:
    #   1. Explicit command-line arguments.
    #   2. Values from config file's [SimulationParameters] section (if not None).
    #   3. Argparse defaults.
    
    _args_main = argparse.Namespace()
    cmd_args_raw_list = sys.argv[1:] # Raw list of command line arguments

    # Special handling for output_dir default generation
    output_dir_explicitly_set_on_cmd = False

    for action in arg_parser_obj._actions:
        dest = action.dest
        if dest == 'help': # Skip help action, not a parameter.
            continue
        
        # Start with Priority 3: Argparse default.
        current_value = arg_parser_obj.get_default(dest)

        # Override with Priority 2: Config file value, if key exists in config_sim_params and is not None.
        if dest in config_sim_params and config_sim_params[dest] is not None:
            current_value = config_sim_params[dest]

        # Override with Priority 1: Explicit command-line argument if the flag was present.
        action_flag_was_present_in_cmd = False
        for opt_string in action.option_strings: # e.g., ['-n', '--ncells']
            if opt_string in cmd_args_raw_list:
                action_flag_was_present_in_cmd = True
                if dest == 'output_dir': # Track if output_dir was set via command line
                    output_dir_explicitly_set_on_cmd = True
                break
        
        if action_flag_was_present_in_cmd:
            # If the flag was present, the value from args_cmd_line_only for this 'dest'
            # (which is what argparse determined from the command line) takes precedence.
            current_value = getattr(args_cmd_line_only, dest)
        
        setattr(_args_main, dest, current_value)

    #--- Initial Conditions Setup ---#
    # Get IC parameters. This dict includes 'simulation_name_keyword' from the IC file,
    # or a default from problem_setup (e.g., "SodDefault") if the IC file load failed,
    # or a default from config_loader (e.g., "UnknownSim") if the keyword was missing in a valid IC file.
    ic_params = problem_setup.get_initial_conditions_parameters(_args_main)
    keyword_from_ic_file_or_its_defaults = ic_params.get('simulation_name_keyword')

    # Determine the final _simulation_name_keyword_main to be used as the filename prefix.
    # Priority:
    # 1. Command-line argument -snk/--simulation_name_keyword.
    # 2. Value from the loaded IC file's 'simulation_name_keyword' (if it's not a fallback like "UnknownSim" or "SodDefault").
    # 3. Argparse default for -snk (which is "NumSche").

    # Check if -snk or --simulation_name_keyword was explicitly provided on the command line
    snk_flag_present_in_cmd = False
    for action in arg_parser_obj._actions:
        if action.dest == 'simulation_name_keyword': # The 'dest' of the new argument
            for opt_string in action.option_strings:
                if opt_string in cmd_args_raw_list:
                    snk_flag_present_in_cmd = True
                    break
            if snk_flag_present_in_cmd:
                break
    
    if snk_flag_present_in_cmd:
        # Highest priority: CMD override for the keyword
        _simulation_name_keyword_main = _args_main.simulation_name_keyword 
        print(f"INFO: Simulation name keyword explicitly set via command line to: '{_simulation_name_keyword_main}'")
    elif keyword_from_ic_file_or_its_defaults and \
         keyword_from_ic_file_or_its_defaults != "UnknownSim" and \
         keyword_from_ic_file_or_its_defaults != "SodDefault":
        # Next priority: Meaningful keyword from the IC file
        _simulation_name_keyword_main = keyword_from_ic_file_or_its_defaults
        print(f"INFO: Simulation name keyword set from IC file to: '{_simulation_name_keyword_main}'")
    else:
        # Fallback: Argparse default for -snk (e.g., "NumSche")
        # This case is hit if -snk was not used, AND the keyword from IC file was a fallback name.
        _simulation_name_keyword_main = _args_main.simulation_name_keyword 
        print(f"INFO: Simulation name keyword from IC file was a fallback ('{keyword_from_ic_file_or_its_defaults}'). Defaulting to argparse default for keyword: '{_simulation_name_keyword_main}'")
    
    # _simulation_name_keyword_main is now the definitive prefix for filenames and default output dir.

    #--- Output Directory Finalization ---#
    # If output_dir was NOT explicitly set on the command line,
    # then the default is generated based on simulation_name_keyword.
    # This overrides any output_dir from simulation_config.ini unless output_dir was explicitly set via CMD.
    if not output_dir_explicitly_set_on_cmd:
        generated_dir_from_keyword = f"{_simulation_name_keyword_main}_frames"
        # Check if the current value (which could be from config or argparse default) is different
        if _args_main.output_dir != generated_dir_from_keyword:
            print(f"INFO: Output directory not explicitly set via command line. "
                  f"Using keyword-based default: '{generated_dir_from_keyword}'. "
                  f"(Previous value was: '{_args_main.output_dir}')")
        else:
             print(f"INFO: Output directory not explicitly set via command line. "
                  f"Using keyword-based default: '{generated_dir_from_keyword}'.")
        _args_main.output_dir = generated_dir_from_keyword
    else:
        print(f"INFO: Output directory explicitly set via command line to: '{_args_main.output_dir}'")

    (   _x_active_grid_main,
        x_full_grid_main, # Not directly used as global, but returned by setup_grid_and_initial_state.
        rho_initial_full_main,
        v_initial_full_main,
        P_initial_full_main,
        _dx_grid_val_main
    ) = problem_setup.setup_grid_and_initial_state(_args_main, ic_params)

    #--- Initial State Preparation ---#
    # Convert initial primitive variables to conservative variables.
    rho_g, rhov_g, rhoe_g = hydro_utils.get_conservative_vars(
        rho_initial_full_main, v_initial_full_main, P_initial_full_main, constants.GAMMA_IDEAL_GAS
    )
    _U_current_global = np.array([rho_g, rhov_g, rhoe_g])

    # Apply initial boundary conditions.
    _U_current_global = hydro_utils.apply_boundary_conditions(
        _U_current_global, constants.NUM_GHOST_CELLS, constants.BC_TYPE_CONFIG
    )

    #--- Scheme Selection ---#
    # Select the numerical scheme function and name for printing.
    if _args_main.scheme == "LWR":
        _selected_scheme_func = numerical_schemes.lax_wendroff_ritchmyer_step
        _scheme_name_print_main = "Lax-Wendroff-Ritchmyer"
    elif _args_main.scheme == "LF":
        _selected_scheme_func = numerical_schemes.lax_friedrichs_step
        _scheme_name_print_main = "Lax-Friedrichs"
    elif _args_main.scheme == "LIM":
        _selected_scheme_func = numerical_schemes.limiter_scheme_step
        _scheme_name_print_main = "Minmod Limiter"
    else:
        raise ValueError(f"Unknown scheme: {_args_main.scheme}")

    #--- Simulation Information Printout ---#
    print(f"\n#---Starting 1D Hydro Simulation ({_simulation_name_keyword_main} - {_scheme_name_print_main} - Continuous) ---#")
    
    print("\nSimulation parameters:")
    print(f"    Analysis using IC file: {_args_main.ic_config_file.split('/')[-1]}")
    print(f"    Scheme:  {_args_main.scheme}")
    print(f"    Ncells: {_args_main.ncells}")
    print(f"    CFL: {_args_main.cfl_number}")
    print(f"    Epsilon: {_args_main.epsilon}")
    print(f"    Output format: {_args_main.output_format}")
    print(f"    Output directory: '{_args_main.output_dir}'")
    
    print(f"\nPlot will update every ~{_args_main.dt_plot:.6f} system time units.")

    #--- Plotting Setup ---#
    # Get initial primitive variables for setting up initial plot limits.
    rho_init_display_main, v_init_display_main, P_init_display_main = hydro_utils.get_primitive_vars(
        _U_current_global[0, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
        _U_current_global[1, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
        _U_current_global[2, constants.NUM_GHOST_CELLS:-constants.NUM_GHOST_CELLS],
        constants.GAMMA_IDEAL_GAS
    )

    # Setup plots using the visualization module.
    # This initializes visualization._fig_anim, visualization._axs_anim, etc.
    fig_setup, _, _ = visualization.setup_plots(
        constants.X_MIN, constants.X_MAX,
        initial_rho=rho_init_display_main,
        initial_P=P_init_display_main,
        initial_v=v_init_display_main
    )

    # Connect the close event handler.
    fig_setup.canvas.mpl_connect('close_event', on_main_animation_close)

    #--- Animation Timing Initialization ---#
    # Initialize time for the next plot update.
    _t_next_plot_update = _t_current_sim + _args_main.dt_plot

    # Initialize time for the next auto-save frame (if enabled).
    if _args_main.dt_save > 0:
          _t_next_frame_save_main = _args_main.dt_save # First auto-save scheduled at t = dt_save.
    
    # If t_final is set, print it.
    if _args_main.t_final is not None and _args_main.t_final > 0:
        print(f"Simulation will run until t_final = {_args_main.t_final:.4f} system units.\n")

    # Create the animation object.
    _ani_main = animation.FuncAnimation(
        fig_setup,
        animation_update_manager, # Wrapper function that calls simulation and plot updates.
        frames=None,              # For continuous animation until t_final or window close.
        init_func=visualization.init_animation_plots,
        blit=False,               # blit=False is often more robust, though potentially slower.
        interval=1,               # ms delay between frames (influences drawing speed).
        repeat=False,             # Do not repeat the animation.
        cache_frame_data=False    # Avoids unbounded cache with frames=None.
    )

    # Display the animation.
    plt.show()

    #--- Post-Animation Information ---#
    print(f"\nAnimation stopped. Final simulation time: {_t_current_sim:.4f} system units.")
    print(f"Total simulation steps taken: {_step_count_sim}")
"""
Module for handling simulation visualization and frame saving.
 
Provides functions to:
- Setup Matplotlib figure and axes for plotting.
- Initialize animation elements.
- Update plot data for each frame.
- Save plot frames to files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import argparse # For type hinting args


#--- Module-Level Globals for Plot Elements ---#
# These variables store the Matplotlib figure, axes, and line objects,
# managed internally by this module.
_fig_anim: Optional[plt.Figure] = None
_axs_anim: Optional[np.ndarray] = None 
_lines_anim: Optional[Dict[str, plt.Line2D]] = {} # Initialize as empty dict
_max_reached_plot_limits: Dict[str, Tuple[float, float]] = {} # Initialize as empty dict

def setup_plots(
    x_min_domain: float,
    x_max_domain: float,
    initial_rho: np.ndarray,
    initial_P: np.ndarray,
    initial_v: np.ndarray
) -> Tuple[plt.Figure, np.ndarray, Dict[str, plt.Line2D]]:
    """
    Set up the Matplotlib figure, axes, and line objects for the animation.

    Parameters
    ----------
    x_min_domain : float
        Minimum x-coordinate for the plot x-axis.
    x_max_domain : float
        Maximum x-coordinate for the plot x-axis.
    initial_rho : np.ndarray
        Initial density data to set initial plot limits.
    initial_P : np.ndarray
        Initial pressure data to set initial plot limits.
    initial_v : np.ndarray
        Initial velocity data to set initial plot limits.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray, Dict[str, plt.Line2D]]
        The created figure, array of axes, and dictionary of line artists.
    """
    global _fig_anim, _axs_anim, _lines_anim
    global _max_reached_plot_limits

    _fig_anim, _axs_anim = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    axes_map_setup = {'rho': _axs_anim[0], 'P': _axs_anim[1], 'v': _axs_anim[2]}
    data_map = {'rho': initial_rho, 'P': initial_P, 'v': initial_v}

    for key, data in data_map.items():
        min_val = np.min(data)
        max_val = np.max(data)

        # Store the actual data extremes (unpadded) from the initial state.
        _max_reached_plot_limits[key] = (min_val, max_val)

        # Calculate and apply initial y-axis limits with padding.
        padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 1e-6 else 0.1
        initial_display_limits_with_padding = (min_val - padding, max_val + padding)
        axes_map_setup[key].set_ylim(initial_display_limits_with_padding)

    # Define font sizes
    axis_label_fontsize = 20
    tick_label_fontsize = 18
    legend_fontsize = 18
    offset_text_fontsize = 15

    _lines_anim['rho'], = _axs_anim[0].plot([], [], '-', lw=1, color='b', label="Density ($\\rho$)")
    _axs_anim[0].set_ylabel("Density", fontsize=axis_label_fontsize)
    _axs_anim[0].grid(True, alpha=0.4)
    _axs_anim[0].legend(loc='upper right', fontsize=legend_fontsize)
    _axs_anim[0].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    _axs_anim[0].yaxis.get_offset_text().set_fontsize(offset_text_fontsize)


    _lines_anim['P'], = _axs_anim[1].plot([], [], '-', lw=1, color='g', label="Pressure (P)")
    _axs_anim[1].set_ylabel("Pressure", fontsize=axis_label_fontsize)
    _axs_anim[1].grid(True, alpha=0.4)
    _axs_anim[1].legend(loc='upper right', fontsize=legend_fontsize)
    _axs_anim[1].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    _axs_anim[1].yaxis.get_offset_text().set_fontsize(offset_text_fontsize)


    _lines_anim['v'], = _axs_anim[2].plot([], [], '-', lw=1, color='r', label="Velocity (v)")
    _axs_anim[2].set_ylabel("Velocity", fontsize=axis_label_fontsize)
    _axs_anim[2].set_xlabel("Grid Position", fontsize=axis_label_fontsize)
    _axs_anim[2].grid(True, alpha=0.4)
    _axs_anim[2].legend(loc='upper right', fontsize=legend_fontsize)
    _axs_anim[2].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    _axs_anim[2].yaxis.get_offset_text().set_fontsize(offset_text_fontsize)


    for ax_plot in _axs_anim:
        ax_plot.set_xlim(x_min_domain, x_max_domain)

    return _fig_anim, _axs_anim, _lines_anim

def init_animation_plots() -> Tuple[plt.Line2D, ...]:
    """
    Initialize animation plot elements for FuncAnimation.
    Sets the data for each line to empty.

    Returns
    -------
    Tuple[plt.Line2D, ...]
        A tuple of the Line2D artists.
    """
    if not _lines_anim:
        return tuple()
    for line in _lines_anim.values():
        line.set_data([], [])
    return tuple(_lines_anim.values())

def update_plot_data(
    x_active_grid: np.ndarray,
    rho_display: np.ndarray,
    v_display: np.ndarray,
    P_display: np.ndarray,
    current_time: float,
    scheme_name: str,
    sim_keyword: str,
    n_cells: int,
    epsilon_val: float,
    dt_plot_val: float
) -> Tuple[plt.Line2D, ...]:
    """
    Update the data for the plot lines and the title.

    Parameters
    ----------
    x_active_grid : np.ndarray
        Coordinates of the active grid cell centers.
    rho_display : np.ndarray
        Density data to plot.
    v_display : np.ndarray
        Velocity data to plot.
    P_display : np.ndarray
        Pressure data to plot.
    current_time : float
        Current simulation time.
    scheme_name : str
        Name of the numerical scheme being used.
    sim_keyword : str
        Keyword identifying the simulation case.
    n_cells : int
        Number of active cells.
    epsilon_val : float
        Epsilon value used for smoothing.
    dt_plot_val : float
        Time interval between plot updates.

    Returns
    -------
    Tuple[plt.Line2D, ...]
        A tuple of the updated Line2D artists.
    """
    global _max_reached_plot_limits

    _lines_anim['rho'].set_data(x_active_grid, rho_display)
    _lines_anim['v'].set_data(x_active_grid, v_display)
    _lines_anim['P'].set_data(x_active_grid, P_display)
    
    title_fontsize = 24 
    title_str = rf"{sim_keyword} | {scheme_name} | $N={n_cells}$ | $\epsilon={epsilon_val:.1f}$ | $\Delta t_p={dt_plot_val:.1e}$ | $t={{{current_time:.5f}}}$ s.u."
    _axs_anim[0].set_title(title_str, fontsize=title_fontsize)

    # Dynamically adjust y-limits, ensuring they only expand
    axes_map = {'rho': _axs_anim[0], 'P': _axs_anim[1], 'v': _axs_anim[2]}
    current_data_for_axes = {'rho': rho_display, 'P': P_display, 'v': v_display}

    for key, current_axis_data in current_data_for_axes.items():
        stored_min_data, stored_max_data = _max_reached_plot_limits[key]

        current_frame_min_data = np.min(current_axis_data)
        current_frame_max_data = np.max(current_axis_data)

        new_overall_min_data = min(stored_min_data, current_frame_min_data)
        new_overall_max_data = max(stored_max_data, current_frame_max_data)
        _max_reached_plot_limits[key] = (new_overall_min_data, new_overall_max_data)

        display_range = new_overall_max_data - new_overall_min_data
        padding = display_range * 0.1 if display_range > 1e-9 else 0.1 

        display_min_lim = new_overall_min_data - padding
        display_max_lim = new_overall_max_data + padding

        axes_map[key].set_ylim(display_min_lim, display_max_lim)

    return tuple(_lines_anim.values())

def save_plot_frame(
    fig: plt.Figure,
    time: float,
    args: argparse.Namespace,
    sim_keyword: str
) -> None:
    """
    Saves the current Matplotlib figure to a file.

    The filename is constructed using the provided simulation keyword,
    numerical scheme, number of cells, epsilon value, and current simulation time.
    The output directory is taken from `args.output_dir` and created if it
    does not exist. The output format is determined by `args.output_format`.

    Parameters
    ----------
    fig : plt.Figure
        The Matplotlib figure object to be saved.
    time : float
        The current simulation time, used for naming the output file.
    args : argparse.Namespace
        An object containing parsed command-line arguments and configuration
        settings, including `output_dir`, `scheme`, `ncells`, `epsilon`,
        and `output_format`.
    sim_keyword : str
        The keyword identifying the simulation (e.g., "SodShock", "SNShock"),
        used as the primary prefix for the output filename.
    """
    if fig is None:
        print("Warning: Figure object is None, cannot save frame.")
        return
    if args is None:
        print("Warning: Arguments are None, cannot construct filename for saving.")
        return

    # Ensure the output directory exists.
    output_dir_path = args.output_dir
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            print(f"Created output directory: {output_dir_path}")
        except OSError as e:
            print(f"Error creating output directory {output_dir_path}: {e}")
            return

    # sim_keyword is the final resolved keyword (e.g., "SodShock", "SNShock", or "NumSche")
    # It directly becomes the prefix of the filename.
    filename_prefix = sim_keyword
    
    filename_core_details = f"{args.scheme}_N{args.ncells}_eps{args.epsilon:.1f}_t{time:.4f}"

    def save_single_format(selected_format: str):
        # Construct the full filename.
        frame_filename = f"{filename_prefix}_{filename_core_details}.{selected_format}"
        full_save_path = os.path.join(output_dir_path, frame_filename)
        try:
            fig.savefig(full_save_path, bbox_inches='tight', dpi=200)
            print(f"Frame saved: {full_save_path}")
        except Exception as e:
            print(f"Error saving frame {full_save_path}: {e}")

    if args.output_format == 'both':
        save_single_format('png')
        save_single_format('pdf')
    else:
        save_single_format(args.output_format)
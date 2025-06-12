"""
Defines core physical and numerical constants for the 1D hydrodynamic simulation.

These values are generally fixed for a given set of simulation assumptions
but are centralized here for clarity and ease of modification if needed.
"""

#--- Physical Constants ---#
GAMMA_IDEAL_GAS: float = 1.4  # Adiabatic index for an ideal diatomic gas

#--- Numerical Parameters (Fixed) ---#
# Small floating-point value used as a floor to prevent division by zero
# in calculations like the relative gradient 'r' or when ensuring positive density/pressure.
R_DENOMINATOR_FLOOR: float = 1e-12

# Number of ghost cells on each side of the computational domain.
# These are essential for:
#   1. Applying boundary conditions correctly.
#   2. Providing sufficient stencil points for numerical schemes,
#      especially for higher-order schemes or those involving flux limiters.
NUM_GHOST_CELLS: int = 2

#--- Default Simulation Domain Parameters ---#
# These define the spatial extent of the simulation if not specified elsewhere.
X_MIN: float = 0.0  # Minimum x-coordinate of the domain
X_MAX: float = 1.0  # Maximum x-coordinate of the domain

#--- Default Boundary Condition Type ---#
# Specifies the default boundary condition type to be applied if not
# overridden by other configuration mechanisms (e.g., command-line arguments).
BC_TYPE_CONFIG: str = 'outflow' # Options: 'outflow' (zero-gradient), 'periodic'

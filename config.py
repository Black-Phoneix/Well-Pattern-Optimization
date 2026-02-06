"""
Global configuration constants for the well-field optimization prompt.

All values are in SI units unless otherwise noted.
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Hydraulics defaults (prompt section I)
# -----------------------------------------------------------------------------
M_DOT_TOTAL = 100.0  # kg/s
K_PERM = 5e-14  # m^2
H_THICK = 300.0  # m
D_WELL = 0.41  # m
R_WELL = D_WELL / 2.0
P_INJ = 200e5  # Pa
P_PROD = 150e5  # Pa
T_INJ_C = 40.0  # °C
T_PROD_C = 100.0  # °C
F_TWO_SIDED = 1.0
R_B = 10000.0  # m

# Reservoir/rock
POROSITY = 0.10
RHO_ROCK = 2650.0  # kg/m^3
CP_ROCK = 1000.0  # J/kg/K
RHO_WAT = 1000.0  # kg/m^3
CP_WAT = 4180.0  # J/kg/K
S_WIRR = 0.2
T_RES_C = 150.0  # °C
T_WORKING_C = 80.0  # °C

# -----------------------------------------------------------------------------
# Geometry/optimization bounds and constraints
# -----------------------------------------------------------------------------
R_IN_MIN = 500.0
R_IN_MAX = 1500.0
DELTA_R_MIN = 500.0
R_OUT_MAX = 4000.0
THETA0_MIN = 0.0
THETA0_MAX = 2.0 * np.pi
EPS_MAX = np.deg2rad(25.0)
DELTA_THETA_MIN = np.deg2rad(10.0)
S_MIN = 500.0

# -----------------------------------------------------------------------------
# Objective weights
# -----------------------------------------------------------------------------
W1 = 1.0
W2 = 1.0
W3 = 1.0
W4 = 0.5
TAU_REF_YEARS = 30.0

# -----------------------------------------------------------------------------
# Optimization controls
# -----------------------------------------------------------------------------
DE_POPSIZE = 15
DE_MAXITER = 80
DE_MUTATION = 0.8
DE_RECOMBINATION = 0.7
DE_TOL = 1e-3
DE_SEED = 42

# -----------------------------------------------------------------------------
# Breakthrough/TOF configuration
# -----------------------------------------------------------------------------
N_SEED = 24
R_SEED = 3.0 * R_WELL
R_CAPTURE = 4.0 * R_WELL
TOF_T_MAX = 2.0e9  # s

# Plotting grid
GRID_MARGIN = 500.0
GRID_N = 80

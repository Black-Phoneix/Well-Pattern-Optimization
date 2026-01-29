"""
Centralized configuration for CO2-based closed-loop geothermal well-field optimization.

All constants in one place with exact defaults from the specification.
Units are SI (meters, kg, seconds, Pascals, Kelvin, etc.) unless otherwise noted.

Configuration Groups:
    - Hydraulics: Flow rates, permeability, well dimensions
    - Thermal: Rock/water properties, temperatures
    - Breakthrough/TOF: Streamline parameters
    - Geometry constraints: Minimum spacings, bounds
    - Optimization: Weights, DE parameters
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class Config:
    """
    Centralized configuration with all constants for well-field optimization.
    
    All parameters in SI units unless otherwise noted.
    
    Attributes
    ----------
    Hydraulics (Section I):
        M_DOT_TOTAL : float
            Total mass flow rate [kg/s] (default: 100.0)
        K_PERM : float
            Permeability [m²] (default: 5e-14)
        H_THICK : float
            Reservoir thickness [m] (default: 300.0)
        D_WELL : float
            Well diameter [m] (default: 0.41)
        R_WELL : float
            Well radius [m] (D_WELL/2)
        P_INJ : float
            Injection pressure [Pa] (default: 200e5 = 20 MPa)
        P_PROD : float
            Production pressure [Pa] (default: 150e5 = 15 MPa)
        T_INJ_C : float
            Injection temperature [°C] (default: 40.0)
        T_PROD_C : float
            Production temperature [°C] (default: 100.0)
        F_TWO_SIDED : float
            Dimensionless factor for PDF convention (default: 1.0)
        R_B : float
            Reference/boundary radius [m] (default: 10000.0)
    
    Thermal (Section E):
        POROSITY : float
            Porosity [-] (default: 0.10)
        RHO_ROCK : float
            Rock density [kg/m³] (default: 2300.0)
        CP_ROCK : float
            Rock specific heat capacity [J/(kg·K)] (default: 0.92e3 = 920)
        RHO_WAT : float
            Water density [kg/m³] (default: 1000.0)
        CP_WAT : float
            Water specific heat capacity [J/(kg·K)] (default: 4180.0)
        S_WIRR : float
            Irreducible water saturation [-] (default: 0.20)
        T_RES_C : float
            Reservoir temperature [°C] (default: 120.0)
        T_WORK_C : float
            Working temperature [°C] (default: 40.0)
    
    Breakthrough / TOF (Section D):
        N_SEED : int
            Number of seed points per injector (default: 40)
        R_SEED_FACTOR : float
            Factor for seed radius (r_seed = R_SEED_FACTOR * R_WELL) (default: 3.0)
        R_CAPTURE_FACTOR : float
            Factor for capture radius (r_capture = R_CAPTURE_FACTOR * R_WELL) (default: 3.0)
        T_MAX : float
            Maximum integration time [s] (default: 5e9)
    
    Geometry Constraints (Section B):
        S_MIN : float
            Minimum well spacing [m] (default: 500.0)
        DELTA_R_MIN : float
            Minimum radial gap [m] (default: 500.0)
        DELTA_THETA_MIN_DEG : float
            Minimum angle increment [degrees] (default: 10.0)
        EPS_MAX_DEG : float
            Maximum epsilon deviation [degrees] (default: 25.0)
    
    Bounds (Section B):
        R_IN_MIN : float
            Minimum inner radius [m] (default: 500.0)
        R_IN_MAX : float
            Maximum inner radius [m] (default: 1500.0)
        R_OUT_MAX : float
            Maximum outer radius [m] (default: 4000.0)
    
    Objective Weights (Section F):
        W1 : float
            Weight for CV_inj (default: 1.0)
        W2 : float
            Weight for CV_prod (default: 1.0)
        W3 : float
            Weight for lifetime τ (default: 1.0)
        W4 : float
            Weight for CV_tof (default: 0.5)
        TAU_REF : float
            Reference lifetime for normalization [years] (default: 30.0)
        PENALTY_LAMBDA : float
            Penalty coefficient (default: 1e6)
    
    Optimization (Section G):
        DE_POPSIZE : int
            Population size for differential evolution (default: 15)
        DE_MAXITER : int
            Maximum iterations (default: 100)
        DE_SEED : int
            Random seed for reproducibility (default: 42)
    """
    
    # =========================================================================
    # Hydraulics (Section I of prompt)
    # =========================================================================
    M_DOT_TOTAL: float = 100.0              # kg/s
    K_PERM: float = 5e-14                   # m² (permeability)
    H_THICK: float = 300.0                  # m (reservoir thickness)
    D_WELL: float = 0.41                    # m (well diameter)
    P_INJ: float = 200e5                    # Pa (injection pressure, 20 MPa)
    P_PROD: float = 150e5                   # Pa (production pressure, 15 MPa)
    T_INJ_C: float = 40.0                   # °C (injection temperature)
    T_PROD_C: float = 100.0                 # °C (production temperature)
    F_TWO_SIDED: float = 1.0                # dimensionless factor
    R_B: float = 10000.0                    # m (reference/boundary radius)
    
    # =========================================================================
    # Thermal (Section E of prompt)
    # =========================================================================
    POROSITY: float = 0.10                  # [-] (porosity)
    RHO_ROCK: float = 2300.0                # kg/m³ (rock density)
    CP_ROCK: float = 0.92e3                 # J/(kg·K) (rock heat capacity, 0.92 kJ/kg/°C)
    RHO_WAT: float = 1000.0                 # kg/m³ (water density)
    CP_WAT: float = 4180.0                  # J/(kg·K) (water heat capacity)
    S_WIRR: float = 0.20                    # [-] (irreducible water saturation)
    T_RES_C: float = 120.0                  # °C (reservoir temperature)
    T_WORK_C: float = 40.0                  # °C (working/reference temperature)
    
    # =========================================================================
    # Breakthrough / TOF (Section D of prompt)
    # =========================================================================
    N_SEED: int = 40                        # seeds per injector
    R_SEED_FACTOR: float = 3.0              # r_seed = R_SEED_FACTOR * R_WELL
    R_CAPTURE_FACTOR: float = 3.0           # r_capture = R_CAPTURE_FACTOR * R_WELL
    T_MAX: float = 5e9                      # seconds (maximum integration time)
    
    # =========================================================================
    # Geometry Constraints (Section B of prompt)
    # =========================================================================
    S_MIN: float = 500.0                    # m (minimum well spacing)
    DELTA_R_MIN: float = 500.0              # m (minimum radial gap R_out - R_in)
    DELTA_THETA_MIN_DEG: float = 10.0       # degrees (minimum angle increment)
    EPS_MAX_DEG: float = 25.0               # degrees (max epsilon deviation)
    
    # =========================================================================
    # Bounds (Section B of prompt)
    # =========================================================================
    R_IN_MIN: float = 500.0                 # m
    R_IN_MAX: float = 1500.0                # m
    R_OUT_MAX: float = 4000.0               # m
    
    # =========================================================================
    # Objective Weights (Section F of prompt)
    # =========================================================================
    W1: float = 1.0                         # weight for CV_inj
    W2: float = 1.0                         # weight for CV_prod
    W3: float = 1.0                         # weight for lifetime τ
    W4: float = 0.5                         # weight for CV_tof
    TAU_REF: float = 30.0                   # years (reference lifetime)
    PENALTY_LAMBDA: float = 1e6             # penalty coefficient
    
    # =========================================================================
    # Optimization (Section G of prompt)
    # =========================================================================
    DE_POPSIZE: int = 15                    # differential evolution population size
    DE_MAXITER: int = 100                   # maximum iterations
    DE_SEED: int = 42                       # random seed for reproducibility
    
    # =========================================================================
    # Derived Properties
    # =========================================================================
    @property
    def R_WELL(self) -> float:
        """Well radius [m]."""
        return self.D_WELL / 2.0
    
    @property
    def r_seed(self) -> float:
        """Seed radius for streamlines [m]."""
        return self.R_SEED_FACTOR * self.R_WELL
    
    @property
    def r_capture(self) -> float:
        """Capture radius for producers [m]."""
        return self.R_CAPTURE_FACTOR * self.R_WELL
    
    @property
    def DELTA_THETA_MIN(self) -> float:
        """Minimum angle increment [radians]."""
        return np.deg2rad(self.DELTA_THETA_MIN_DEG)
    
    @property
    def EPS_MAX(self) -> float:
        """Maximum epsilon deviation [radians]."""
        return np.deg2rad(self.EPS_MAX_DEG)
    
    @property
    def T_INJ_K(self) -> float:
        """Injection temperature [K]."""
        return self.T_INJ_C + 273.15
    
    @property
    def T_PROD_K(self) -> float:
        """Production temperature [K]."""
        return self.T_PROD_C + 273.15
    
    @property
    def T_RES_K(self) -> float:
        """Reservoir temperature [K]."""
        return self.T_RES_C + 273.15
    
    @property
    def T_WORK_K(self) -> float:
        """Working temperature [K]."""
        return self.T_WORK_C + 273.15
    
    @property
    def P_MEAN(self) -> float:
        """Mean pressure [Pa]."""
        return 0.5 * (self.P_INJ + self.P_PROD)
    
    @property
    def T_MEAN_K(self) -> float:
        """Mean temperature [K]."""
        return 0.5 * (self.T_INJ_K + self.T_PROD_K)
    
    @property
    def m_dot_inj_each(self) -> float:
        """Mass rate per injector [kg/s] (3 injectors)."""
        return self.M_DOT_TOTAL / 3.0
    
    @property
    def m_dot_prod_each(self) -> float:
        """Mass rate per producer [kg/s] (5 producers)."""
        return self.M_DOT_TOTAL / 5.0
    
    def get_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """
        Get optimization variable bounds.
        
        Variables: [R_in, R_out, θ0, ε1, ε2, ε3]
        
        Returns
        -------
        tuple
            Bounds for each variable as ((min1, max1), (min2, max2), ...)
        """
        return (
            (self.R_IN_MIN, self.R_IN_MAX),                      # R_in
            (self.R_IN_MIN + self.DELTA_R_MIN, self.R_OUT_MAX),  # R_out
            (0.0, 2.0 * np.pi),                                   # θ0
            (-self.EPS_MAX, self.EPS_MAX),                        # ε1
            (-self.EPS_MAX, self.EPS_MAX),                        # ε2
            (-self.EPS_MAX, self.EPS_MAX),                        # ε3
        )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure T_prod > T_inj for positive heat extraction
        if self.T_PROD_C <= self.T_INJ_C:
            raise ValueError(
                f"T_PROD_C ({self.T_PROD_C}°C) must be > T_INJ_C ({self.T_INJ_C}°C) "
                "for positive heat extraction."
            )
        
        # Ensure P_INJ > P_PROD for flow from injector to producer
        if self.P_INJ <= self.P_PROD:
            raise ValueError(
                f"P_INJ ({self.P_INJ/1e6:.1f} MPa) must be > P_PROD ({self.P_PROD/1e6:.1f} MPa) "
                "for flow from injector to producer."
            )


# Default configuration instance
DEFAULT_CONFIG = Config()

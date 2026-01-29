import numpy as np

def calculate_pore_volume(R_inner, R_outer, thickness, porosity):
    """
    Calculate the pore volume of the annular region.
    V = pi * (R_out^2 - R_in^2) * h * phi
    """
    volume_rock = np.pi * (R_outer**2 - R_inner**2) * thickness
    return volume_rock * porosity

# Temperature decline model based on analytical approximations validated by Adams et al. (2020).
def calculate_thermal_power_and_lifetime(
    m_dot,          # Mass flow rate of the well [kg/s]
    V_bulk,         # Pore volume controlled by the well [m3]
    T_inj,          # Injection temperature [C]
    T_res_initial,  # Initial reservoir temperature [C]
    cp_fluid,       # Specific heat capacity of fluid [J/(kg K)]
    rho_fluid,      # Fluid density [kg/m3] (used for heat capacity calculation)
    rho_rock,       # Rock density
    cp_rock,        # Rock specific heat capacity
    porosity,       # Porosity
    operation_time_years=30, # Project duration considered, default 30 years
    cutoff_fraction=0.1 
):
    """
    Calculate for a single well:
    1. Average Thermal Power over 30 years - for maximization
    2. Thermal Lifetime - for constraint/reference
    
    Based on the analytical solution from Master Thesis Eq.(4).
    """
    
    # 0. Basic parameter conversion
    t_end_seconds = operation_time_years * 365 * 24 * 3600
    dT_max = T_res_initial - T_inj
    
    # Calculate effective volumetric heat capacity [J/(m3 K)]
    # (rho * c)_eff = phi * rho_f * c_f + (1-phi) * rho_r * c_r
    rhoc_eff = porosity * rho_fluid * cp_fluid + (1 - porosity) * rho_rock * cp_rock
    
    # 1. Calculate characteristic time tau
    # tau = (V * rhoc_eff) / (m_dot * c_fluid)
    # This is the time scale required for the fluid to completely "sweep" the heat from this volume.
    if m_dot <= 1e-6 or V_bulk <= 0: # Prevent division by zero
        return 0.0, 1e9 
        
    tau = (V_bulk * rhoc_eff) / (m_dot * cp_fluid)
    
    # 2. Calculate Average Thermal Power
    # Integral formula: Average Power = (Integral of P(t) dt) / t_end
    # P(t) = m_dot * cp * (T(t) - T_inj) = m_dot * cp * dT_max * G(t)
    # G(t) = 1 / (t/tau + 1)
    # Integral G(t) dt = tau * ln(t/tau + 1)
    
    integral_G = tau * np.log(t_end_seconds / tau + 1.0)
    
    # Average Power [Watts]
    P_avg = (m_dot * cp_fluid * dT_max) * (integral_G / t_end_seconds)
    
    # 3. Calculate "Thermal Lifetime"
    # Definition: Time when production temperature drops to 10% of initial temperature difference (G(t) = 0.1)
    # 1 / (t_life/tau + 1) = 0.1  =>  t_life/tau + 1 = 10  => t_life = 9 * tau
    cutoff = float(cutoff_fraction)
    cutoff = np.clip(cutoff, 1e-6, 0.999999)
    lifetime_seconds = tau * (1.0 / cutoff - 1.0)
    lifetime_years = lifetime_seconds / (365 * 24 * 3600)
    
    
    return P_avg, lifetime_years

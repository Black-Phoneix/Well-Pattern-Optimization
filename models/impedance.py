import numpy as np
from patterns.geometry import distance_matrix

# ============================================================
# 1) Hydraulics: impedance and admittance
# ============================================================

def impedance_doublet(mu, rho, k, b, L, rw):
    """
    Analytical hydraulic impedance for an injector–producer pair.

    Z(L) = (mu / (rho * k * b)) * (1 / (2*pi)) * ln(L / rw)
    """
    # Numerical protection against L -> 0
    L = np.maximum(L, rw)
    C1 = 1.0 / (2.0 * np.pi)
    return (mu / (rho * k * b)) * C1 * np.log(L / rw)


def admittance_matrix(injectors, producers, mu, rho, k, b, rw):
    """
    Compute admittance matrix Y = 1 / Z.

    Y[j, i] represents the hydraulic conductance
    between injector i and producer j.
    """
    D = distance_matrix(injectors, producers)   # distances [m]
    Z = impedance_doublet(mu, rho, k, b, D, rw)
    Y = 1.0 / Z
    return Y


# ============================================================
# 2) Producer pressures under equal-flow constraint
# ============================================================

def producer_pressures_equal_flow(
    injectors,
    producers,
    P_inj,
    m_total,
    mu,
    rho,
    k,
    b,
    rw,
    thermo_dP=0.0,
):
    """
    Compute required producer pressures assuming equal flow per producer.

    Governing relation:
        m_j = ( (P_inj - P_prod,j) + thermo_dP ) * S_j

    where:
        S_j = sum_i Y[j,i]
        m_j = m_total / n_prod
    """
    Y = admittance_matrix(injectors, producers, mu, rho, k, b, rw)
    S_vec = np.sum(Y, axis=1)

    n_prod = producers.shape[0]
    flow_vec = np.full(n_prod, m_total / n_prod)

    # Avoid division by zero for very small S_j
    eps = 1e-30
    P_prod_vec = P_inj + thermo_dP - flow_vec / np.maximum(S_vec, eps)

    return P_prod_vec, flow_vec, S_vec


def pressure_uniformity_objective(P_prod_vec, mode="variance"):
    """
    Scalar measure of pressure uniformity.
    """
    if mode == "variance":
        return float(np.var(P_prod_vec))
    elif mode == "range":
        return float(np.max(P_prod_vec) - np.min(P_prod_vec))
    else:
        raise ValueError("mode must be 'variance' or 'range'")

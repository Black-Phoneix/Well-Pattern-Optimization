import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Well:
    """
    Simple representation of a well in 2D map view.

    Attributes
    ----------
    x, y : float
        Coordinates in [m].
    kind : str
        "injector" or "producer".
    """
    x: float
    y: float
    kind: str  # "injector" or "producer"


def generate_ring_pattern(
    n_inj: int,
    n_prod: int,
    R_inj: float,
    R_prod: float,
    phi_inj0: float = 0.0,
    phi_prod0: float = 0.0,
) -> Tuple[List[Well], List[Well]]:
    """
    Generate a simple ring pattern:
    - n_inj injectors on a circle of radius R_inj
    - n_prod producers on a circle of radius R_prod

    Angles are measured in radians, starting from phi_inj0 / phi_prod0.

    Returns
    -------
    injectors, producers : list[Well], list[Well]
    """
    injectors: List[Well] = []
    producers: List[Well] = []

    # injectors
    for i in range(n_inj):
        phi = phi_inj0 + 2.0 * np.pi * i / n_inj
        x = R_inj * np.cos(phi)
        y = R_inj * np.sin(phi)
        injectors.append(Well(x=x, y=y, kind="injector"))

    # producers
    for j in range(n_prod):
        phi = phi_prod0 + 2.0 * np.pi * j / n_prod
        x = R_prod * np.cos(phi)
        y = R_prod * np.sin(phi)
        producers.append(Well(x=x, y=y, kind="producer"))

    return injectors, producers


def distance_matrix(injectors: List[Well], producers: List[Well]) -> np.ndarray:
    """
    Compute injector–producer distance matrix.

    D[j, i] = distance between producer j and injector i
    """
    n_inj = len(injectors)
    n_prod = len(producers)
    D = np.zeros((n_prod, n_inj), dtype=float)

    for j, p in enumerate(producers):
        for i, inj in enumerate(injectors):
            dx = p.x - inj.x
            dy = p.y - inj.y
            D[j, i] = np.hypot(dx, dy)

    return D


def minimum_spacing(wells: List[Well]) -> float:
    """
    Compute the minimum pairwise distance between all wells.

    Useful for checking spacing constraints.
    """
    coords = np.array([[w.x, w.y] for w in wells])
    n = len(coords)
    d_min = np.inf

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < d_min:
                d_min = d

    return d_min

# ... (Retain original Well class and imports)

def generate_center_ring_pattern(
    n_inj: int,
    n_prod_outer: int,
    R_inj: float,
    R_prod: float,
    phi_inj0: float = 0.0,
    phi_prod0: float = 0.0,
    center_producer: bool = True
) -> Tuple[List[Well], List[Well]]:
    """
    Generate a 'Center + Ring' hybrid well pattern:
    - n_inj injectors distributed on a circle of radius R_inj
    - 1 producer located at the center (0,0) (optional)
    - n_prod_outer producers distributed on a circle of radius R_prod
    
    This layout allows balancing the impedance between the central well and the peripheral wells 
    by adjusting the ratio of R_inj and R_prod.
    """
    injectors: List[Well] = []
    producers: List[Well] = []

    # 1. Generate Injectors
    for i in range(n_inj):
        phi = phi_inj0 + 2.0 * np.pi * i / n_inj
        x = R_inj * np.cos(phi)
        y = R_inj * np.sin(phi)
        injectors.append(Well(x=x, y=y, kind="injector"))

    # 2. Generate Center Producer
    if center_producer:
        # Center well coordinates fixed at (0,0)
        producers.append(Well(x=0.0, y=0.0, kind="producer"))

    # 3. Generate Outer Ring Producers
    for j in range(n_prod_outer):
        phi = phi_prod0 + 2.0 * np.pi * j / n_prod_outer
        x = R_prod * np.cos(phi)
        y = R_prod * np.sin(phi)
        producers.append(Well(x=x, y=y, kind="producer"))

    return injectors, producers


def validate_well_layout(
    wells: List[Well],
    min_spacing: float = 500.0,
    field_radius: float = 2000.0,
) -> Tuple[bool, str]:
    """
    Validate well layout against constraints.
    
    Parameters
    ----------
    wells : List[Well]
        List of wells to validate
    min_spacing : float, optional
        Minimum inter-well spacing [m] (default: 500)
    field_radius : float, optional
        Maximum field radius [m] (default: 2000)
    
    Returns
    -------
    tuple
        (is_valid, message) where is_valid is bool and message describes any issues
    """
    # Check minimum spacing
    actual_spacing = minimum_spacing(wells)
    if actual_spacing < min_spacing:
        return False, f"Minimum spacing violation: {actual_spacing:.1f} m < {min_spacing:.1f} m"
    
    # Check field boundary
    for well in wells:
        r = np.sqrt(well.x**2 + well.y**2)
        if r > field_radius:
            return False, f"Well at ({well.x:.1f}, {well.y:.1f}) outside field radius {field_radius:.1f} m"
    
    # Check that we have both injectors and producers
    n_inj = sum(1 for w in wells if w.kind == 'injector')
    n_prod = sum(1 for w in wells if w.kind == 'producer')
    
    if n_inj == 0:
        return False, "No injector wells found"
    if n_prod == 0:
        return False, "No producer wells found"
    
    return True, "Layout is valid"


def from_optimization_result(
    positions: np.ndarray,
    n_injectors: int,
) -> List[Well]:
    """
    Convert optimization result to Well list.
    
    Parameters
    ----------
    positions : np.ndarray
        Flattened array [x1, y1, x2, y2, ..., xN, yN]
    n_injectors : int
        Number of injector wells (first n_injectors are injectors, rest are producers)
    
    Returns
    -------
    List[Well]
        List of Well objects
    """
    n_wells = len(positions) // 2
    wells = []
    
    for i in range(n_wells):
        x, y = positions[2*i], positions[2*i+1]
        kind = 'injector' if i < n_injectors else 'producer'
        wells.append(Well(x=x, y=y, kind=kind))
    
    return wells


def to_dict(wells: List[Well]) -> dict:
    """
    Serialize well layout to dictionary.
    
    Parameters
    ----------
    wells : List[Well]
        List of wells
    
    Returns
    -------
    dict
        Dictionary with well data
    """
    return {
        'wells': [
            {'x': w.x, 'y': w.y, 'kind': w.kind}
            for w in wells
        ]
    }


def from_dict(data: dict) -> List[Well]:
    """
    Deserialize well layout from dictionary.
    
    Parameters
    ----------
    data : dict
        Dictionary with well data
    
    Returns
    -------
    List[Well]
        List of Well objects
    """
    return [
        Well(x=w['x'], y=w['y'], kind=w['kind'])
        for w in data['wells']
    ]


def export_to_json(wells: List[Well], filename: str):
    """
    Export well layout to JSON file.
    
    Parameters
    ----------
    wells : List[Well]
        List of wells
    filename : str
        Output file path
    """
    import json
    data = to_dict(wells)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def import_from_json(filename: str) -> List[Well]:
    """
    Import well layout from JSON file.
    
    Parameters
    ----------
    filename : str
        Input file path
    
    Returns
    -------
    List[Well]
        List of Well objects
    """
    import json
    with open(filename, 'r') as f:
        data = json.load(f)
    return from_dict(data)

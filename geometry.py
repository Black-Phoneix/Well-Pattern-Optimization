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

# ... (保留原有的 Well 类和 imports)

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
    生成 '中心 + 环形' 混合井网模式:
    - n_inj 个注入井分布在半径 R_inj 的圆上
    - 1 个生产井位于中心 (0,0) (可选)
    - n_prod_outer 个生产井分布在半径 R_prod 的圆上
    
    这种布局允许通过调整 R_inj 和 R_prod 的比例，来平衡中心井与外围井的阻抗。
    """
    injectors: List[Well] = []
    producers: List[Well] = []

    # 1. 生成注入井 (Injectors)
    for i in range(n_inj):
        phi = phi_inj0 + 2.0 * np.pi * i / n_inj
        x = R_inj * np.cos(phi)
        y = R_inj * np.sin(phi)
        injectors.append(Well(x=x, y=y, kind="injector"))

    # 2. 生成中心生产井 (Center Producer)
    if center_producer:
        # 中心井坐标固定为 (0,0)
        producers.append(Well(x=0.0, y=0.0, kind="producer"))

    # 3. 生成外围生产井环 (Outer Ring Producers)
    for j in range(n_prod_outer):
        phi = phi_prod0 + 2.0 * np.pi * j / n_prod_outer
        x = R_prod * np.cos(phi)
        y = R_prod * np.sin(phi)
        producers.append(Well(x=x, y=y, kind="producer"))

    return injectors, producers
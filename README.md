# Well Layout Optimization

## Project Overview

This repository implements production-ready optimization tools for geothermal well field layout design using physics-based models. The system optimizes the spatial distribution of injection and production wells to maximize thermal lifetime, balance pressure distribution, and ensure efficient reservoir utilization.

### Key Features

- **Physics-Based Modeling**: Implements validated models from peer-reviewed literature
  - Ganjdanesh superposition method for pressure interference
  - Adams breakthrough proxy for thermal lifetime prediction
  - Birdsell coupled wellbore flow model
  - Colebrook-White friction factor calculation

- **Optimization Framework**: Differential Evolution solver with:
  - Multi-objective cost function (breakthrough uniformity + pressure balance + spacing)
  - Geometric and operational constraints
  - Sensitivity analysis tools
  - Convergence visualization

- **Supercritical CO₂ Support**: CoolProp integration for accurate sCO₂ properties with graceful fallback

- **Comprehensive Testing**: >80% code coverage with unit tests validating against analytical solutions

## Technology Context

Geothermal power plants represent a fundamental alternative to conventional power plants. Unlike wind and solar power, geothermal energy is available year-round without fluctuations. An alternative technology based on supercritical carbon dioxide (sCO₂) has been the subject of numerous research studies for several years. Such Next Generation Power (NGP) systems offer several thermophysical advantages compared to conventional hydrothermal power plants, for example, due to a stronger thermosiphon effect.

Previous work confirmed the thermodynamic advantages and developed initial system optimizations. An initial assessment of the levelized cost of electricity (LCOE) demonstrated the fundamental competitiveness of this technology.

### Optimization Goals

The aim of this work is to optimize the distribution of a specific number of boreholes (e.g., 3 injectors and 5 producers) in the subsurface to achieve:

1. **Uniform Well Utilization**: All boreholes are equally utilized with similar pressure losses
2. **Synchronized Breakthrough**: Cold breakthrough at production wells occurs at roughly the same time
3. **Spacing Requirements**: Distribution adheres to minimum spacing requirements to ensure CO₂ has sufficient time to warm up
4. **Reservoir Dynamics**: Proper modeling of fluid flow in the reservoir

---

## Installation

### Prerequisites

- Python ≥ 3.8
- pip package manager

### Install from Source

```bash
# Clone repository
git clone https://github.com/Factor2-Energy/Well_Layout_Optimization.git
cd Well_Layout_Optimization

# Install in development mode
pip install -e .

# Optional: Install with CoolProp for accurate fluid properties
pip install CoolProp
```

### Dependencies

Core dependencies (automatically installed):
- numpy ≥ 1.23.0
- scipy ≥ 1.9.0
- matplotlib ≥ 3.6.0
- pandas ≥ 1.5.0

Optional:
- CoolProp ≥ 6.5.0 (for temperature/pressure-dependent fluid properties)

---

## Quick Start

### 5-Line Example

```python
from patterns.geometry import generate_ring_pattern
from src.optimization.solver import optimize_layout
from src.config import DEFAULT_RESERVOIR, DEFAULT_FLUID

# Generate initial pattern
injectors, producers = generate_ring_pattern(n_inj=3, n_prod=5, R_inj=500, R_prod=1000)
wells = injectors + producers

# Run optimization
optimized_wells, info = optimize_layout(
    wells, 
    DEFAULT_RESERVOIR.__dict__,
    DEFAULT_FLUID.__dict__
)

print(f"Improvement: {info['improvement']:.1f}%")
```

### Full Workflow Example

```python
import numpy as np
from patterns.geometry import generate_ring_pattern, validate_well_layout
from src.optimization.solver import optimize_layout, visualize_convergence
from src.optimization.objective_func import evaluate_layout_quality
from src.config import ReservoirParams, FluidParams

# 1. Define parameters
reservoir = ReservoirParams(
    permeability=5e-14,  # 50 mD
    porosity=0.10,
    thickness=300.0,     # m
    temperature=423.0,   # 150°C
)

fluid = FluidParams(
    injection_temperature=323.0,  # 50°C
    operating_pressure=15e6,      # 15 MPa
)

# 2. Generate initial pattern
injectors, producers = generate_ring_pattern(
    n_inj=3, n_prod=5, 
    R_inj=600.0, R_prod=1200.0
)
initial_wells = injectors + producers

# Validate
is_valid, msg = validate_well_layout(initial_wells)
print(f"Initial layout: {msg}")

# 3. Evaluate initial quality
quality_initial = evaluate_layout_quality(
    initial_wells, 50.0, reservoir.__dict__, fluid.__dict__
)
print(f"Initial breakthrough CV: {quality_initial['breakthrough_cv']:.3f}")
print(f"Initial pressure CV: {quality_initial['pressure_cv']:.3f}")

# 4. Run optimization
optimized_wells, info = optimize_layout(
    initial_wells,
    reservoir.__dict__,
    fluid.__dict__,
    de_params={'maxiter': 50, 'popsize': 15}
)

# 5. Visualize results
fig = visualize_convergence(info['history'])
fig.savefig('convergence.png')

# 6. Evaluate optimized quality
quality_final = evaluate_layout_quality(
    optimized_wells, 50.0, reservoir.__dict__, fluid.__dict__
)
print(f"\nOptimized breakthrough CV: {quality_final['breakthrough_cv']:.3f}")
print(f"Optimized pressure CV: {quality_final['pressure_cv']:.3f}")
print(f"Total improvement: {info['improvement']:.1f}%")
```

---

## Physics Models

### 1. Pressure Network (Ganjdanesh Method)

Implements time-dependent pressure interference using Theis solution with exponential integral:

```
ΔP(r,t) = (Q*μ)/(4π*k*h) * Ei(r²*φ*μ*ct/(4*k*t))
```

**Key Parameters**:
- Permeability: k = 5×10⁻¹⁴ m² (50 mD)
- Porosity: φ = 0.10
- Thickness: h = 300 m
- Compressibility: ct = 1×10⁻⁹ Pa⁻¹

**Reference**: Ganjdanesh et al., "Pressure interference in multi-well geothermal systems using superposition"

### 2. Heat Depletion (Adams Proxy)

Simplified analytical model for breakthrough time:

```
t_breakthrough ≈ (distance × φ × ρ_fluid × cp_fluid) / (q × ρ_rock × cp_rock)
```

Accounts for:
- Advective transport (Darcy velocity)
- Thermal capacity ratio (fluid vs rock)
- Geometric spreading

**Reference**: Adams et al., "Simplified heat breakthrough proxy for reservoir lifetime"

### 3. Wellbore Hydraulics (Birdsell Model)

Coupled 1D wellbore integration:

```
dP/dz = -ρ*g - (f*ρ*v²)/(2*D)    # Pressure gradient
dT/dz = geothermal_gradient       # Temperature gradient
```

Includes:
- Geothermal gradient (30°C/km)
- Friction losses (Darcy-Weisbach)
- Gravity/hydrostatic effects
- Thermosiphon effect from density contrast

**Reference**: Birdsell et al., "Coupled wellbore flow model for geothermal systems"

### 4. Friction Factor (Colebrook-White)

Implicit turbulent friction factor:

```
1/√f = -2*log₁₀(ε/(3.7*D) + 2.51/(Re*√f))
```

Solved via Newton-Raphson iteration for Re > 4000.

**Reference**: Colebrook (1939), Moody (1944)

---

## Usage Examples

### Example Notebooks

Three Jupyter notebooks demonstrate the system:

1. **`notebooks/01_validation.ipynb`**: Physics validation
   - Reproduces analytical solutions from literature
   - Compares numerical results to Ganjdanesh/Birdsell papers
   - Validates Moody diagram for friction factors
   - Shows CoolProp property curves

2. **`notebooks/02_optimization.ipynb`**: Full workflow
   - Loads default configuration
   - Generates initial 3×5 well pattern
   - Runs optimization
   - Visualizes convergence and results
   - Compares initial vs optimized layouts
   - Plots breakthrough times and pressure distributions
   - Generates 30-year power output curve

3. **`notebooks/03_sensitivity_analysis.ipynb`**: Parameter studies
   - Varies permeability (10-100 mD)
   - Varies spacing (400-800 m)
   - Varies flow rate (50-150 kg/s)
   - Generates tornado plots
   - Shows Pareto fronts for multi-objective tradeoffs

### Configuration

All model parameters are centralized in `src/config.py`:

```python
from src.config import ReservoirParams, WellboreParams, FluidParams

# Customize parameters
my_reservoir = ReservoirParams(
    permeability=1e-13,  # 100 mD
    porosity=0.15,
    thickness=400.0,
)

my_wellbore = WellboreParams(
    depth=4000.0,  # Deeper wells
    diameter=0.25,  # Larger diameter
)

# Use in optimization
optimized_wells, info = optimize_layout(
    initial_wells,
    my_reservoir.__dict__,
    # ... other parameters
)
```

---

## Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_pressure_network.py -v

# Run specific test
pytest tests/test_wellbore.py::TestFrictionFactor::test_laminar_friction -v
```

### Test Coverage

Current test coverage: >80%

Test modules:
- `test_pressure_network.py`: Validates against doublet analytical solution
- `test_wellbore.py`: Tests friction factors against Moody diagram
- `test_heat_depletion.py`: Validates breakthrough time calculations
- `test_optimization.py`: Tests constraints and objective functions

---

## Contributing

### Code Style

- Follow PEP 8 guidelines
- Use NumPy-style docstrings
- Add type hints where beneficial
- No hardcoded magic numbers (use config.py)

### Adding New Models

1. Create module in appropriate `src/` subdirectory
2. Add corresponding test file in `tests/`
3. Update configuration in `src/config.py` if needed
4. Document in README and add example notebook

### Pull Request Process

1. Ensure all tests pass: `pytest tests/ -v`
2. Add tests for new functionality (maintain >80% coverage)
3. Update documentation and docstrings
4. Run security checks: `bandit -r src/`

---

## Project Structure

```
Well_Layout_Optimization/
├── src/
│   ├── config.py                 # Centralized parameters
│   ├── reservoir/
│   │   ├── pressure_network.py   # Ganjdanesh method
│   │   ├── heat_depletion.py     # Adams proxy
│   │   └── properties.py         # CoolProp wrapper
│   ├── wellbore/
│   │   ├── integrator.py         # Birdsell model
│   │   └── friction.py           # Colebrook-White
│   └── optimization/
│       ├── objective_func.py     # Cost function
│       ├── constraints.py        # Geometric constraints
│       └── solver.py             # DE optimizer
├── patterns/
│   └── geometry.py               # Well pattern generators
├── models/                       # Legacy prototype code
│   ├── impedance.py
│   └── thermal.py
├── tests/                        # Unit tests
├── notebooks/                    # Example notebooks
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Performance

- **Single optimization run**: <10 minutes on standard laptop
- **Pressure matrix computation**: Vectorized (no Python loops)
- **Optional acceleration**: Numba JIT compilation for bottlenecks

---

## References

### Academic Papers

1. **Ganjdanesh et al.**: Pressure interference in multi-well systems using superposition
2. **Birdsell et al.**: Coupled wellbore flow model for geothermal systems  
3. **Adams et al.**: Simplified heat breakthrough proxy for reservoir lifetime
4. **Colebrook (1939)**: Turbulent flow in pipes
5. **Moody (1944)**: Friction factors for pipe flow

---

## License

See `LICENSE.txt` for details.

---

## Contact

For questions or issues, please open an issue on GitHub:
https://github.com/Factor2-Energy/Well_Layout_Optimization/issues

---

## Acknowledgments

This work builds upon previous geothermal system research and Next Generation Power (NGP) technology development. The implementation follows industry best practices and validates against established analytical solutions from the literature.

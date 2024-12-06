# Genetic Programming Library

A genetic programming library for symbolic regression with focus on computational complexity awareness and exportable solutions. Built with scikit-learn compatibility in mind. This is currently a work in progress and under active development.

## Features

- Scikit-learn compatible API for integration with ML workflows
- Dynamic complexity evaluation modes:
  - Simple Mode: Basic depth and node count evaluation
  - Compute Mode: Full computational complexity analysis including operation costs
  - Hybrid Mode: Adaptive switching between modes based on tree size and operation types
- Automatic Python code generation for discovered solutions
- Multi-threaded evaluation support
- Built-in numerical stability safeguards

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/JonIsHere242/genetic-programming.git
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.19.0
- SciPy ≥ 1.7.0
- scikit-learn ≥ 0.24.0
- tqdm ≥ 4.60.0

## Quick Start

Basic usage example:

```python
from genetic_programming import SymbolicRegressor
import numpy as np

# Generate sample data
X = np.random.randn(100, 5)
y = X[:, 0]**2 + np.sin(X[:, 1]) + X[:, 2]

# Create and fit model
sr = SymbolicRegressor(
    population_size=1000,
    generations=20,
    complexity_mode='hybrid',
    verbose=1
)

sr.fit(X, y)
y_pred = sr.predict(X)
```

## Configuration Options

```python
SymbolicRegressor(
    population_size=1000,    # Size of the population
    generations=20,          # Number of generations to evolve
    tournament_size=20,      # Tournament selection size
    max_depth=4,            # Maximum depth of expression trees
    parsimony_coefficient=0.01,  # Complexity penalty coefficient
    complexity_mode='hybrid',    # Complexity evaluation mode ('simple', 'compute', 'hybrid')
    verbose=1,              # Verbosity level (0=none, 1=progress, 2=detailed with code export)
    n_jobs=1               # Number of parallel jobs
)
```

## Complexity Evaluation Modes

### Simple Mode
- Fast evaluation based on tree structure
- Considers only node count and depth
- Suitable for small expressions or rapid prototyping

### Compute Mode
- Detailed analysis of computational complexity
- Considers operation costs and memory requirements
- Operation-specific cost metrics
- Memory usage estimation
- Better for production use when solution efficiency is critical

### Hybrid Mode (Default)
- Automatically switches between Simple and Compute modes
- Uses Simple mode for small trees (< 8 nodes)
- Switches to Compute mode for larger trees or when expensive operations are detected
- Best balance of evaluation speed and accuracy

## Solution Export

When running with `verbose=2`, the library automatically exports discovered solutions:

- Generates standalone Python files with the solution
- Includes proper feature name mapping
- Adds numerical stability safeguards
- Exports training metrics and performance history
- Solutions can be directly imported and used in other projects

Example exported solution:

```python
import numpy as np

def predict(data: dict) -> np.ndarray:
    """
    Predict using the generated genetic program.
    
    Required features:
    - feature1
    - feature2
    ...
    """
    return np.clip(
        np.sin(data['feature1']) + np.square(data['feature2']),
        -1e6, 1e6
    )
```

## Development Status

This library is currently in active development. Current work includes:
- Expanding operation set
- Improving parallel processing capabilities
- Adding more sophisticated selection methods
- Implementing early stopping criteria
- Adding visualization tools
- Improving documentation

## Author

Jonathan Bellmont (masamunex9000@gmail.com)

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Structure

```
genetic_programming/
├── utils/
│   ├── __init__.py
│   ├── Nodes.py
│   ├── OperationConfig.py
│   └── Verbose.py
├── PopulationMGMT.py
├── Program.py
├── SymbolicRegressor.py
├── TestDataset.py
└── TreeComplexity.py
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{genetic_programming_library,
  author = {Jonathan Bellmont},
  title = {Genetic Programming Library},
  year = {2024},
  url = {https://github.com/JonIsHere242/genetic-programming}
}
```
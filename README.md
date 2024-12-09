# Genetic Programming Library

A sophisticated symbolic regression framework that pushes the boundaries of genetic programming through advanced optimization techniques. At its core, the library combines Monte Carlo Tree Search with innovative complexity analysis to evolve highly efficient mathematical expressions. Built with production-grade features including metaprogrammatic code generation, parallel processing capabilities, and domain-specific customization options, this library represents a significant advancement in automated mathematical modeling.

## Features

- **Monte Carlo Tree Search (MCTS) Optimization**
  - Leverages MCTS algorithms to intelligently explore and optimize elite solutions
  - Dramatically accelerates convergence through guided evolutionary search
  - Refines high-performing expressions with strategic tree modifications
  - Maintains solution diversity while pursuing optimal candidates

- **Advanced Complexity Analysis**
  - **Compute Mode**: Production-grade complexity analysis considering operation costs, memory usage, and computational overhead
  - **Simple Mode**: Rapid evaluation based on structural metrics for fast prototyping
  - **Hybrid Mode**: Dynamic switching between evaluation methods based on expression characteristics
  - Real-time complexity monitoring and optimization during evolution

- **Metaprogrammatic Solution Generation**
  - Automatically exports evolved expressions as production-ready Python code
  - Generates comprehensive documentation including performance metrics and usage examples
  - Implements proper feature name mapping for seamless integration
  - Includes numerical stability safeguards in exported solutions

- **Parallel Processing Architecture**
  - Leverages multiprocessing for evolutionary computations
  - Parallel MCTS optimization of elite programs
  - Scalable design for high-performance computing environments
  - Efficient resource utilization across available cores

- **Domain-Specific Evolution**
  - **Operation Preset System**: Tailored initialization weights for specific domains:
    - Physics: Emphasizes trigonometric and differential operations
    - Finance: Prioritizes exponential and ratio calculations
    - Biology: Focuses on growth and decay patterns
    - Custom: User-defined operation distributions
  - Smart initialization ensuring domain-relevant starting populations
  - Configurable constraints maintaining problem-specific requirements

- **Production-Ready Features**
  - Scikit-learn compatible API for seamless integration
  - Built-in numerical stability safeguards
  - Comprehensive error handling and validation
  - Detailed logging and progress monitoring

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

# Create and fit model with a domain-specific operation preset
sr = SymbolicRegressor(
    population_size=1000,
    generations=20,
    complexity_mode='hybrid',
    verbose=2,
    verbose_output_dir="gp_outputs",
    verbose_output_file="gp_outputs/final_solution.py",
    operation_preset="physics"  # Domain-specific preset influencing initial population
)

sr.fit(X, y)
y_pred = sr.predict(X)
```

## Configuration Options

```python
SymbolicRegressor(
    population_size=1000,           # Size of the population
    generations=20,                 # Number of generations to evolve
    tournament_size=20,             # Tournament selection size
    max_depth=4,                    # Maximum depth of expression trees
    min_depth=2,                    # Minimum depth of expression trees
    parsimony_coefficient=0.01,     # Complexity penalty coefficient
    complexity_ratio_limit=10.0,    # Maximum complexity ratio to maintain diversity
    crossover_probability=0.7,      # Probability of performing crossover
    mutation_probability=0.3,       # Probability of mutating a program
    operation_preset="finance",     # Operation weight preset ('random', 'natural', 'basic', 'physics', 'finance', 'biology', 'custom')
    elite_size=0.05,                # Proportion or number of elite programs to retain
    terminal_probability=0.3,        # Probability of selecting a terminal node during initialization
    constant_range=(-5.0, 5.0),      # Range for sampling constant values
    constant_std=2.0,                # Standard deviation for generating constants
    complexity_mode='hybrid',        # Complexity evaluation mode ('simple', 'compute', 'hybrid')
    use_mcts=True,                   # Enable Monte Carlo Tree Search optimization
    mcts_n_elite=5,                  # Number of elite programs to optimize with MCTS
    mcts_iterations=50,              # Number of MCTS iterations per program
    mcts_exploration_weight=1.414,   # Exploration constant for MCTS
    mcts_max_depth=3,                # Maximum depth for programs during MCTS
    mcts_eval_samples=1000,          # Number of samples for evaluating programs in MCTS
    mcts_threads=None,               # Number of threads for parallel MCTS optimization (None uses all available cores minus one)
    verbose=2,                       # Verbosity level (0=none, 1=progress, 2=detailed with code export)
    verbose_output_dir="gp_outputs", # Directory for verbose output files
    verbose_output_file="gp_outputs/test.py", # Exact file path for exporting final solution and metrics
    n_jobs=-1,                       # Number of parallel jobs (-1 uses all available processors)
    random_state=42                  # Seed for reproducibility
)
```

## Operation Presets

The library offers various operation weight presets tailored to different domains, influencing the distribution of operations in the initial population. This targeted initialization enhances the evolutionary search by embedding domain-relevant operations, leading to more accurate and meaningful symbolic expressions.

Available presets:
- `random`: Uniform distribution of all operations.
- `natural`: Mimics natural mathematical expressions with a balanced mix of operations.
- `basic`: Focuses on fundamental arithmetic operations with minimal complexity.
- `physics`: Emphasizes operations common in physical laws and formulas, such as trigonometric functions and conditional operations.
- `finance`: Prioritizes operations relevant to financial models, including ratios and exponential functions.
- `biology`: Tailors operations for biological modeling, incorporating growth and decay processes.
- `custom`: Allows users to define custom operation weight distributions.

### Example of Custom Preset

```python
from genetic_programming import SymbolicRegressor, OperationWeightPreset

custom_weights = {
    'add': 1.0,
    'subtract': 0.8,
    'multiply': 0.6,
    'divide': 0.5,
    'conditional': 0.7,
    'safe_divide': 0.9,
    'abs': 0.4,
    'square': 0.5,
    'sqrt': 0.3,
    'sin': 0.2,
    'exp': 0.8
}

sr = SymbolicRegressor(
    population_size=1000,
    generations=20,
    operation_preset=OperationWeightPreset.CUSTOM,
    custom_operation_weights=custom_weights,  # Define custom weights
    verbose=2,
    verbose_output_dir="gp_outputs",
    verbose_output_file="gp_outputs/final_solution.py",
    # ... other parameters ...
)
```

## Complexity Evaluation Modes

### Simple Mode
- Fast Evaluation: Based on node count and tree depth.
- Low Overhead: Suitable for small expressions or rapid prototyping.
- Limited Insight: Does not account for operation-specific costs or memory usage.

### Compute Mode
- Detailed Analysis: Considers computational and memory costs associated with each operation.
- Operation-Specific Metrics: Uses predefined operation costs and memory requirements for accurate complexity assessment.
- Production-Ready: Ideal for scenarios where solution efficiency and resource usage are critical.

### Hybrid Mode (Default)
- Adaptive Switching: Automatically switches between Simple and Compute modes based on tree size and presence of expensive operations.
- Balanced Approach: Provides a good balance between evaluation speed and accuracy.
- Optimal for Most Use Cases: Suitable for both small and large expressions, adapting to the complexity of the evolved programs.

## Solution Export

When running with `verbose=2`, the library automatically exports discovered solutions with advanced metaprogrammatic verbosity:

- Standalone Python Files: Generates executable Python files containing the evolved symbolic expressions.
- Feature Name Mapping: Includes proper mapping of feature names to ensure consistency and usability.
- Numerical Stability Safeguards: Incorporates safeguards such as np.clip and conditional operations to handle numerical stability issues like division by zero and overflow.
- Training Metrics and Performance History: Embeds detailed metrics and performance history within the exported solution for comprehensive analysis.
- Easy Integration: Solutions can be directly imported and utilized in other projects without modification.

### Example Exported Solution

```python
import numpy as np

'''
Genetic Programming Solution
Generated on: 2024-12-08 18:04:22

Final Test Results:
R^2 Score:          -0.0000
Mean Squared Error: 9.2872
Mean Abs Error:     1.8076
'''


def predict(data: dict) -> np.ndarray:
  """
  Generated prediction function.

  Required features:
  - x3
  - x6
  - x9
  """
  return np.clip(
    data['x3'] * np.sin(data['x6']) + np.log1p(data['x9']),
    -1e6, 1e6
  )
```


## Development Status

This library is currently in active development. Ongoing work includes:

### Expanding Operation Set
- Adding more operations such as logical functions (e.g., OR, AND, NOT, XOR) and advanced mathematical functions (e.g., MAX, MIN, COS, TAN, LOG, CEIL, FLOOR, ROUND, SIGN).

### Improving Parallel Processing Capabilities
- Enhancing multiprocessing support for more efficient evolution and optimization processes.

### Adding More Sophisticated Selection Methods
- Implementing advanced selection strategies to improve population diversity and convergence rates.

### Implementing Early Stopping Criteria
- Introducing mechanisms to halt evolution early based on convergence metrics or performance thresholds.

### Adding Visualization Tools
- Developing tools for visualizing evolutionary progress, expression trees, and complexity metrics.

### Enhancing Documentation
- Expanding comprehensive documentation, including detailed guides, API references, and usage examples.

## Author
Jonathan Bellmont (masamunex9000@gmail.com)

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests to enhance the library's functionality, performance, and documentation. When contributing, please ensure that your code adheres to the project's coding standards and includes appropriate tests and documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@software{genetic_programming_library,
  author = {Jonathan Bellmont},
  title = {Genetic Programming Library},
  year = {2024},
  url = {https://github.com/JonIsHere242/genetic-programming}
}
```

## Acknowledgements
Special thanks to contributors and the open-source community for their invaluable support and contributions to the development of this library.
# Genetic Programming Library

A symbolic regression framework that combines traditional genetic programming with advanced optimization techniques. The library focuses on evolving efficient mathematical expressions through standard genetic operations, Monte Carlo Tree Search, and complexity-aware evolution. Built to be practical and scalable, it includes features for code generation, parallel processing, and domain-specific customization through population operation initalization presets, roughly matching the observed operations of the feild.

## Features

- **Core Genetic Programming**
  - Standard genetic operations (crossover, mutation, reproduction)
  - Tournament selection with configurable pressure
  - Population management with elitism
  - Customizable fitness functions
  - Configurable tree depth and size constraints
  - Multi-generational evolution with adaptive parameters

- **Monte Carlo Tree Search Integration**
  - Uses MCTS to optimize elite solutions
  - Guides the search through promising evolutionary paths
  - Refines high-performing expressions
  - Helps maintain population diversity

- **Complexity Analysis**
  - Compute Mode: Full analysis of operation costs and memory usage
  - Simple Mode: Quick evaluation using tree structure metrics
  - Hybrid Mode: Switches between modes based on expression size
  - Tracks and manages solution complexity during evolution

- **Code Generation**
  - Exports evolved expressions as Python code
  - Generates documentation and usage examples
  - Maps feature names for easy integration
  - Adds numerical stability checks

- **Parallel Processing**
  - Parallel evolution and fitness evaluation
  - Multi-threaded MCTS optimization
  - Scales with available computing resources

- **Domain-Specific Evolution**
  - Operation presets for different domains (Physics, Finance, Biology)
  - Custom operation weight configuration
  - Problem-specific initialization options

- **Production Features**
  - Scikit-learn compatible API
  - Numerical safeguards
  - Progress monitoring
  - Error handling

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

The library implements a sophisticated complexity evaluation system that considers both computational costs and memory usage of evolved expressions. This system helps guide evolution toward efficient solutions while maintaining performance.

### Simple Mode
- Quick evaluation based on node count and tree depth
- Suitable for:
  - Initial prototyping
  - Small expressions
  - Rapid evolution cycles
- Doesn't consider operation-specific costs
- Lowest overhead for computation

### Compute Mode
- Detailed analysis of computational and memory costs
- Tracks costs for each operation:
  ```python
  # Example operation costs (scale: 1.0 = basic operation)
  OPERATION_COSTS = {
      'add': 1.0,          # Basic arithmetic
      'multiply': 2.0,     # More complex than addition
      'divide': 3.0,       # Complex with safety checks
      'sqrt': 4.0,         # Computationally expensive
      'exp': 3.0,         # Complex mathematical operation
      'conditional': 2.0   # Includes branching logic
  }
  
  # Memory usage tracking (scale: 1.0 = single array)
  MEMORY_COSTS = {
      'add': 1.0,          # Single output array
      'divide': 2.0,       # Needs mask arrays for zero checks
      'sqrt': 2.0,         # Intermediate arrays for safety
      'conditional': 2.0   # Arrays for both branches
  }
  ```
- Includes safety mechanisms:
  - Value clipping for numerical stability
  - Domain checks for mathematical operations
  - Proper handling of edge cases (division by zero, etc.)
- Best for:
  - Production environments
  - Resource-constrained systems
  - Performance-critical applications

### Hybrid Mode (Default)
- Dynamically switches between Simple and Compute modes
- Triggers for switching:
  - Expression size exceeds threshold
  - Presence of expensive operations (cost ≥ 3.0)
  - Memory usage concerns
- Balances evaluation speed with accuracy
- Adapts to expression complexity:
  ```python
  def is_expensive_operation(operation: str) -> bool:
      """Check if operation requires detailed cost analysis"""
      return OPERATION_COSTS.get(operation, 0.0) >= 3.0
  ```

### Implementation Details

The system uses vectorized NumPy operations for performance while maintaining safety:

```python
OPERATIONS = {
    'add': lambda x, y: np.clip(x + y, -1e6, 1e6),
    'divide': lambda x, y: np.divide(
        np.clip(x, -1e6, 1e6),
        np.clip(y, -1e6, 1e6),
        out=np.zeros_like(x),
        where=np.abs(y) > 1e-10
    ),
    'exp': lambda x: np.clip(
        np.exp(-np.abs(np.clip(x, -50, 50))), 
        0, 1e6
    )
}
```

### Benefits of Complexity Management

1. **Optimization Guidance**
   - Steers evolution toward efficient solutions
   - Prevents bloat in expressions
   - Balances accuracy with computational cost

2. **Resource Control**
   - Manages memory usage
   - Prevents excessive computation
   - Ensures stable production deployment

3. **Performance Predictability**
   - Known costs for operations
   - Predictable memory usage
   - Reliable execution times





## Solution Export and Code Generation

The library's code generation system is designed with a "production-first" mindset. Instead of generating abstract formula representations that require translation, it directly produces production-ready Python code that can be immediately used in any environment.

### Key Features of Code Generation

- **Production-Ready Python Functions**
  - Generates complete, standalone Python functions
  - Uses proper typing hints and docstrings
  - Includes all necessary imports
  - Ready for immediate integration into production systems

- **Comprehensive Documentation**
  - Documents required input features
  - Includes performance metrics from training
  - Provides timestamp and generation metadata
  - Lists all dependencies and requirements

- **Built-in Safety Features**
  - Automatic addition of numerical stability checks
  - Proper handling of edge cases (division by zero, etc.)
  - Input validation
  - Value clipping for output stability

### Example Generated Solution

```python
import numpy as np

'''
Genetic Programming Solution
Generated on: 2024-12-08 18:04:22

Performance Metrics:
R^2 Score:          0.897
Mean Squared Error: 0.142
Mean Abs Error:     0.276

Feature Importance:
- x1: 37.2%
- x3: 42.8%
- x4: 20.0%
'''

def predict(data: dict) -> np.ndarray:
    """
    Predicts target values using evolved mathematical expression.
    
    Required features:
    - x1: First predictor variable
    - x3: Second predictor variable
    - x4: Third predictor variable
    
    Returns:
    numpy.ndarray: Predicted values
    
    Note: All inputs are automatically clipped to ensure numerical stability
    """
    return np.clip(
        np.sin(data['x1']) * np.exp(data['x3']) + np.sqrt(np.abs(data['x4'])),
        -1e6, 1e6
    )
```

### Direct Usage Example

Generated solutions can be used immediately:

```python
# Direct usage in production
from solution import predict

# Your production data
production_data = {
    'x1': np.array([1.2, 2.3, 3.4]),
    'x3': np.array([0.1, 0.2, 0.3]),
    'x4': np.array([5.0, 6.0, 7.0])
}

# Get predictions
predictions = predict(production_data)
```

### Benefits of Direct Code Generation

1. **No Translation Layer**
   - Solutions are immediately usable without any conversion steps
   - No need for formula interpreters or parsers
   - Reduces potential points of failure in production

2. **Easy Testing and Validation**
   - Generated code can be directly unit tested
   - Simple to validate behavior with different inputs
   - Easy to profile performance

3. **Simple Integration**
   - Copy-paste deployment ready
   - Works with standard Python tooling
   - Compatible with any deployment environment

4. **Maintainable Solutions**
   - Clear, readable code format
   - Self-documenting structure
   - Easy to modify or debug if needed


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
This project is licensed under the MEME License - see the LICENSE file for details.

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
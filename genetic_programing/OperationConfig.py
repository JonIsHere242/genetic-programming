from typing import Dict, Set, ClassVar
import numpy as np
from enum import Enum





class OperationWeightPreset(Enum):
    RANDOM = "random"
    NATURAL = "natural"
    BASIC = "basic"
    PHYSICS = "physics"
    FINANCE = "finance"
    BIOLOGY = "biology"
    CUSTOM = "custom"



class OperationConfig:
    """
    OperationConfig class provides configuration for genetic programming operations.
    Attributes:
        OPERATIONS (dict): Dictionary of available operations and their implementations.
        BINARY_OPS (Set[str]): Set of binary operations.
        UNARY_OPS (Set[str]): Set of unary operations.
        OPERATION_PRESETS (ClassVar[Dict[OperationWeightPreset, Dict[str, float]]]): Presets for operation weights based on different contexts.
    Methods:
        get_operation_weights(preset: OperationWeightPreset, custom_weights: Dict[str, float] = None) -> Dict[str, float]:
            Returns operation weights for a given preset. Raises ValueError if CUSTOM preset is used without providing custom weights.
    Notes:
        - You can add more operations such as random number generators, logical operations (e.g., OR, AND, NOT, XOR), and mathematical functions (e.g., MAX, MIN, COS, TAN, LOG, CEIL, FLOOR, ROUND, SIGN).
        - Ensure to update BINARY_OPS or UNARY_OPS sets, OPERATION_PRESETS dictionary, OPERATION_COSTS and MEMORY_COSTS dictionaries, and OPERATION_WEIGHTS dictionary in the get_operation_weights method when adding new operations.
    """
    """Configuration for genetic programming operations"""
    

    OPERATIONS = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y!=0),
        'abs': lambda x: np.abs(x),
        'square': lambda x: x * x,
        'sqrt': lambda x: np.sqrt(np.abs(x)),
        'sin': lambda x: np.sin(x),
        'exp': lambda x: np.exp(-np.abs(x)),
        'conditional': lambda x, y: np.where(x > 0, y, -y),
        'safe_divide': lambda x, y: np.divide(x, 1 + np.abs(y))
    }
    
    # Sets of operation types
    BINARY_OPS: Set[str] = {'add', 'subtract', 'multiply', 'divide', 'conditional', 'safe_divide'}
    UNARY_OPS: Set[str] = {'abs', 'square', 'sqrt', 'sin', 'exp'}
    
    # Operation weight presets
    #the following distrubutions where just ripped from openai-o1 and pasted here there is no real imperical data backing this but it looks about right
    #for the most part
    #if you are doing a specific thing in each feild feel free to change but as far as im aware inital starting pop formula should be kinda important

    OPERATION_PRESETS: ClassVar[Dict[OperationWeightPreset, Dict[str, float]]] = {
        OperationWeightPreset.RANDOM: {
            'add': 1.0, 'subtract': 1.0, 'multiply': 1.0, 'divide': 1.0,
            'conditional': 1.0, 'safe_divide': 1.0, 'abs': 1.0, 'square': 1.0,
            'sqrt': 1.0, 'sin': 1.0, 'exp': 1.0
        },
        OperationWeightPreset.NATURAL: {
            'add': 1.0,        # Common simple operation
            'subtract': 0.9,   # Slightly less than add
            'multiply': 0.9,   # Common but slightly less than add
            'divide': 0.5,     # Less frequent in natural settings
            'conditional': 0.2,
            'safe_divide': 0.4,
            'abs': 0.3,
            'square': 0.5,     # Moderate likelihood of squared terms
            'sqrt': 0.4,
            'sin': 0.2,        # Occasional wave/periodic patterns in nature
            'exp': 0.2         # Occasional growth/decay phenomena
        },
        OperationWeightPreset.BASIC: {
            'add': 1.0,
            'subtract': 1.0,
            'multiply': 0.8,
            'divide': 0.4,
            'conditional': 0.1, 
            'safe_divide': 0.3,
            'abs': 0.2,
            'square': 0.3,
            'sqrt': 0.2,
            'sin': 0.1,
            'exp': 0.1
        },
        OperationWeightPreset.PHYSICS: {
            'add': 1.0,
            'subtract': 1.0,
            'multiply': 1.0,
            'divide': 0.7,      # Common in formulae for ratios (F=ma -> a=F/m)
            'conditional': 0.2, # Occasionally used in piecewise models
            'safe_divide': 0.5, # Common for avoiding division by zero in simulations
            'abs': 0.3,
            'square': 0.8,      # Common in energy, variance, and field equations
            'sqrt': 0.5,        # Appears in root-mean-square, velocities, etc.
            'sin': 0.8,         # Very common due to wave functions and oscillations
            'exp': 0.4          # Decay processes, certain wave functions in QM
        },
        OperationWeightPreset.FINANCE: {
            'add': 1.0,
            'subtract': 1.0,
            'multiply': 0.8,
            'divide': 1.0,       # Ratios, returns, price-to-earnings, etc.
            'conditional': 0.7,  # Option payoff conditions, piecewise models
            'safe_divide': 1.0,  # Avoiding division by zero in ratio calculations
            'abs': 0.8,          # Volatility, absolute returns
            'square': 0.4,       # Variance calculations, but less common than direct abs
            'sqrt': 0.3,         # Volatility (std dev), but not as frequent as division
            'sin': 0.1,          # Rare in finance formulas
            'exp': 0.7           # Compound interest, discount factors, exponential smoothing
        },
        OperationWeightPreset.BIOLOGY: {
            'add': 1.0,
            'subtract': 0.8,
            'multiply': 0.7,
            'divide': 0.7,       # Ratios for per-capita rates, resource allocation
            'conditional': 0.6,  # Threshold-based conditions in population dynamics
            'safe_divide': 0.7,  # Ensuring stable computations with small populations
            'abs': 0.3,
            'square': 0.5,       # Occasionally used in statistical models (variances)
            'sqrt': 0.5,         # Growth rates, metabolic scaling laws
            'sin': 0.3,          # Seasonal cycles, rhythmic patterns in ecology
            'exp': 0.9           # Common in growth/decay processes, logistic models
        }
    }

    
    @staticmethod
    def get_operation_weights(preset: OperationWeightPreset, 
                            custom_weights: Dict[str, float] = None) -> Dict[str, float]:
        """Get operation weights for given preset"""
        if preset == OperationWeightPreset.CUSTOM:
            if not custom_weights:
                raise ValueError("Custom weights must be provided when using CUSTOM preset")
            return custom_weights
        return OperationConfig.OPERATION_PRESETS[preset]





    # Dictionary of available operations and their implementations
    # Note: Using NumPy operations for vectorized performance; careful handling of domains and zero divisions.
    OPERATIONS = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y!=0),  # Safe handling of division by zero
        'abs': lambda x: np.abs(x),
        'square': lambda x: x * x,
        'sqrt': lambda x: np.sqrt(np.abs(x)),   # sqrt is only defined for non-negative inputs
        'sin': lambda x: np.sin(x),
        'exp': lambda x: np.exp(-np.abs(x)),    # Exponential with negative absolute for controlled growth
        'conditional': lambda x, y: np.where(x > 0, y, -y),  # Simple conditional: choose y or -y based on sign of x
        'safe_divide': lambda x, y: np.divide(x, y+1e-12)    # Example implementation: add epsilon to avoid zero division
    }
    
    # Approximate computational costs for each operation
    # Scale: 1.0 ~ minimal overhead, higher values reflect more expensive operations.
    OPERATION_COSTS = {
        'add': 1.0,          # Very cheap, basic arithmetic
        'subtract': 1.0,     # Same scale as add
        'multiply': 2.0,     # Slightly more costly than add/subtract, but still fast
        'divide': 3.0,       # Division is generally more expensive and includes checks
        'abs': 1.0,          # Simple sign check, very cheap
        'square': 2.0,       # Equivalent to a multiply, so treat similarly
        'sqrt': 4.0,         # sqrt is more expensive due to more complex math
        'sin': 3.0,          # Trigonometric functions are relatively costly
        'exp': 3.0,          # Exponential is also relatively costly
        'conditional': 2.0,  # Branching logic, potentially evaluating multiple paths
        'safe_divide': 2.0   # Similar to divide but possibly simpler if using a fixed epsilon
    }
    
    # Approximate memory costs for each operation
    # Reflects how many temporary arrays or intermediate values might be needed.
    MEMORY_COSTS = {
        'add': 1.0,          # One output array, minimal overhead
        'subtract': 1.0,     # Similar to add
        'multiply': 1.0,     # Single temporary for result
        'divide': 2.0,       # May need mask arrays or intermediate storage for zero checks
        'abs': 1.0,          # Single pass, straightforward
        'square': 1.0,       # Simple result array, like multiply
        'sqrt': 2.0,         # Might need intermediate array for abs() and then sqrt()
        'sin': 1.0,          # One-to-one mapping, minimal overhead
        'exp': 2.0,          # Additional intermediate steps (abs, negation) before exp
        'conditional': 2.0,  # Could need separate arrays for true/false branches
        'safe_divide': 2.0   # Similar complexity to divide due to masking or epsilon handling
    }

    
    # Helper methods to identify operation types
    @classmethod
    def is_binary_operation(cls, operation: str) -> bool:
        """Check if operation requires two operands"""
        return operation in ['add', 'subtract', 'multiply', 'divide']
    
    @classmethod
    def is_unary_operation(cls, operation: str) -> bool:
        """Check if operation requires one operand"""
        return operation in ['abs', 'square', 'sqrt']
    
    @classmethod
    def is_expensive_operation(cls, operation: str) -> bool:
        """Check if operation is computationally expensive"""
        return cls.OPERATION_COSTS.get(operation, 0.0) >= 3.0
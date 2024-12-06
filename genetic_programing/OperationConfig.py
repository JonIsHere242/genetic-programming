from typing import Dict
import numpy as np

class OperationConfig:
    """Configuration for genetic programming operations"""
    
    # Dictionary of available operations and their implementations
    OPERATIONS = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y!=0),
        'abs': lambda x: np.abs(x),
        'square': lambda x: x * x,
        'sqrt': lambda x: np.sqrt(np.abs(x)),
        'sin': lambda x: np.sin(x),
        'exp': lambda x: np.exp(-np.abs(x)),  #simple exponential function
        'conditional': lambda x, y: np.where(x > 0, y, -y),  # Simple conditional
    }
    
    # Computational costs for each operation
    OPERATION_COSTS = {
        'add': 1.0,            # Basic operation
        'subtract': 1.0,       # Basic operation
        'multiply': 2.0,       # More expensive than addition
        'divide': 3.0,         # Most expensive basic operation (includes division-by-zero check)
        'abs': 1.0,            # Simple comparison and potential negation
        'square': 2.0,         # Specialized multiplication
        'sqrt': 4.0,           # Most expensive unary operation
        'sin': 3.0,            # Trigonometric functions are computationally expensive
        'exp': 3.0,            # Exponential function is computationally expensive
        'conditional': 2.0,    # Conditional operation
        'safe_divide': 2.0     # Safe division operation
    }
    
    # Memory costs for each operation
    MEMORY_COSTS = {
        'add': 1.0,         # Needs one temporary result
        'subtract': 1.0,    # Needs one temporary result
        'multiply': 1.0,    # Needs one temporary result
        'divide': 2.0,      # Needs temporary and division check
        'abs': 1.0,         # Needs one temporary comparison
        'square': 1.0,      # Needs one temporary result
        'sqrt': 2.0,        # Needs temporary and absolute value
        'sin': 1.0,         # Needs one temporary result
        'exp': 2.0,         # Needs temporary and absolute value
        'conditional': 2.0, # Needs two temporary results
        'safe_divide': 2.0  # Needs two temporary results
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
        return cls.OPERATION_COSTS.get(operation, 0.0) >= 2.0
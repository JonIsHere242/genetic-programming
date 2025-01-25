from typing import List, Union, Optional, Dict, ClassVar
import numpy as np
from enum import Enum
from numba import jit, vectorize
from array import array
from functools import lru_cache

class NodeType(Enum):
    """Enum to define different types of nodes"""
    OPERATOR = "operator"
    FEATURE = "feature"
    CONSTANT = "constant"

# Pre-compile common numerical operations with Numba
@vectorize(['float64(float64, float64)'])
def safe_divide(x, y):
    return x / (1.0 + abs(y)) if abs(y) > 1e-10 else 0.0

@vectorize(['float64(float64)'])
def safe_exp(x):
    return np.exp(-abs(np.minimum(np.maximum(x, -50), 50)))

@vectorize(['float64(float64)'])
def safe_sqrt(x):
    return np.sqrt(abs(x))

@vectorize(['float64(float64, float64)'])
def conditional_op(x, y):
    return y if x > 0 else -y

@vectorize(['float64(float64, float64)'])
def safe_divide_op(x, y):
    return x / y if abs(y) > 1e-10 else 0.0


# Constants for performance
CLIP_MIN = -1e6
CLIP_MAX = 1e6
TRIG_CLIP = 50.0

class Node:
    """Optimized node implementation for expression trees"""
    
    # Class-level operation cache
    OPERATIONS: ClassVar[Dict] = {
        'add': np.add,
        'subtract': np.subtract,
        'multiply': np.multiply,
        'divide': safe_divide_op,  # Use our new safe divide
        'abs': np.abs,
        'square': np.square,
        'sqrt': safe_sqrt,
        'sin': np.sin,
        'exp': safe_exp,
        'safe_divide': safe_divide_op,
        'conditional': conditional_op
    }

    # Cache binary/unary operation types
    BINARY_OPS = frozenset(['add', 'subtract', 'multiply', 'divide', 'safe_divide', 'conditional'])
    UNARY_OPS = frozenset(['abs', 'square', 'sqrt', 'sin', 'exp'])

    __slots__ = ('node_type', 'value', 'children', '_cached_op', '_zeros_cache', 
                 '_data_shape', '_left_result', '_right_result')

    def __init__(self, 
                 node_type: NodeType,
                 value: Union[str, float] = None,
                 children: List['Node'] = None):
        """Initialize node with performance optimizations"""
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        
        # Performance optimization attributes
        self._cached_op = None  # Cache for operator lookup
        self._zeros_cache = None  # Cache for zero array of current shape
        self._data_shape = None  # Cache for current data shape
        self._left_result = None  # Cache for left child evaluation
        self._right_result = None  # Cache for right child evaluation
        
        # Validate and cache operation if needed
        if self.node_type == NodeType.OPERATOR:
            self._cached_op = self.OPERATIONS.get(self.value)
            if not self._cached_op:
                raise ValueError(f"Unknown operation: {self.value}")
            
            # Validate children count
            required = 2 if self.value in self.BINARY_OPS else 1
            if len(self.children) != required:
                raise ValueError(f"Operation {self.value} requires {required} children")
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def _get_zeros(shape: tuple) -> np.ndarray:
        """Cached creation of zero arrays"""
        return np.zeros(shape)

    def _update_caches(self, data: dict) -> None:
        """Update cached arrays if data shape changes"""
        first_data = next(iter(data.values()))
        shape = first_data.shape
        
        if self._data_shape != shape:
            self._data_shape = shape
            self._zeros_cache = self._get_zeros(shape)
            self._left_result = np.empty(shape) if self.children else None
            self._right_result = (np.empty(shape) 
                                if len(self.children) > 1 else None)

    @staticmethod
    def _clip_array(arr: np.ndarray) -> np.ndarray:
        """Optimized array clipping"""
        return np.clip(arr, CLIP_MIN, CLIP_MAX, out=arr)




    def evaluate(self, data: dict) -> np.ndarray:
        """Optimized evaluation with minimal allocations and safer division"""
        try:
            # Update caches if needed
            self._update_caches(data)
            
            if self.node_type == NodeType.CONSTANT:
                return self.value + self._zeros_cache
            
            elif self.node_type == NodeType.FEATURE:
                if self.value not in data:
                    return self._zeros_cache
                return self._clip_array(data[self.value])
            
            elif self.node_type == NodeType.OPERATOR:
                if len(self.children) == 1:
                    # Unary operation
                    result = self._cached_op(
                        self.children[0].evaluate(data)
                    )
                else:
                    # Binary operation
                    np.copyto(self._left_result, 
                             self.children[0].evaluate(data))
                    np.copyto(self._right_result, 
                             self.children[1].evaluate(data))
                    
                    if self.value in ('divide', 'safe_divide'):
                        # Handle division with explicit zero check
                        mask = np.abs(self._right_result) > 1e-10
                        np.divide(
                            self._left_result,
                            np.where(mask, self._right_result, 1.0),
                            out=self._left_result,
                            where=mask
                        )
                        self._left_result[~mask] = 0.0
                        result = self._left_result
                    elif self.value == 'conditional':
                        result = np.where(
                            self._left_result > 0,
                            self._right_result,
                            -self._right_result
                        )
                    else:
                        # Other binary operations
                        result = self._cached_op(
                            self._left_result,
                            self._right_result
                        )
                
                return self._clip_array(result)
                
        except Exception:
            return self._zeros_cache




    def copy(self) -> 'Node':
        """Optimized deep copy implementation"""
        try:
            # Copy children with validation
            if self.node_type == NodeType.OPERATOR:
                required = 2 if self.value in self.BINARY_OPS else 1
                children = [child.copy() for child in self.children[:required]]
                if len(children) < required:
                    children.extend([Node(NodeType.CONSTANT, 1.0)] * 
                                 (required - len(children)))
            else:
                children = []
            
            # Create new node
            return Node(self.node_type, self.value, children)
            
        except Exception:
            # Fallback for error cases
            return Node(NodeType.CONSTANT, 1.0)

    def __str__(self) -> str:
        """String representation with minimal allocations"""
        if self.node_type == NodeType.CONSTANT:
            return str(self.value)
        elif self.node_type == NodeType.FEATURE:
            return f"Feature({self.value})"
        else:
            if len(self.children) == 1:
                return f"{self.value}({self.children[0]})"
            return f"({self.children[0]} {self.value} {self.children[1]})"
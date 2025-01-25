from typing import List, Optional, Dict, Set, Tuple
import numpy as np
import random
from Nodes import Node, NodeType
from OperationConfig import OperationConfig, OperationWeightPreset
from TreeComplexity import TreeComplexity, ComplexityMode, ComplexityMetrics
from functools import lru_cache

class Program:
    """Symbolic expression tree for genetic programming"""
    
    # Class constants
    TERMINAL_PROB = 0.3
    
    __slots__ = ('max_depth', 'min_depth', 'available_features', 'root', 
                 'fitness', 'operation_weights', '_weighted_ops')
    
    def __init__(
                 self, 
                 max_depth: int = 4,
                 min_depth: int = 2,
                 available_features: List[str] = None,
                 weight_preset: OperationWeightPreset = OperationWeightPreset.NATURAL,
                 custom_weights: Optional[Dict[str, float]] = None):
        """Initialize a program instance"""
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.available_features = available_features or []
        self.root: Optional[Node] = None
        self.fitness: Optional[float] = None
        
        # Pre-compute operation weights
        self.operation_weights = OperationConfig.get_operation_weights(
            weight_preset, 
            frozenset(custom_weights.items()) if custom_weights else None
        )
        self._init_weighted_operations()

    @lru_cache(maxsize=32)
    def _should_terminate(self, depth: int, full: bool, seed: int) -> bool:
        """Cached decision for tree termination"""
        random.seed(seed)  # Use seed for reproducibility while allowing caching
        must_expand = depth < self.min_depth
        must_terminate = depth >= self.max_depth
        
        if full:
            return depth >= self.max_depth
        return (not must_expand) and (must_terminate or random.random() < self.TERMINAL_PROB)

    def generate_random_tree(self, depth: int = 0, full: bool = False) -> Node:
        """Generate a random expression tree"""
        # Use current timestamp as seed for termination decision
        seed = int(depth * 1000 + (1 if full else 0))
        is_terminal = self._should_terminate(depth, full, seed)
        
        if is_terminal:
            return self._generate_terminal()
        return self._generate_operator(depth, full)

    def _init_weighted_operations(self) -> None:
        """Initialize weighted operation selection"""
        self._weighted_ops = tuple(zip(*self.operation_weights.items()))

    def _generate_operator(self, depth: int, full: bool) -> Node:
        """Generate an operator node"""
        operation = random.choices(*self._weighted_ops)[0]
        
        # Pre-allocate children list for better memory efficiency
        if operation in OperationConfig.BINARY_OPS:
            children = [None] * 2
            children[0] = self.generate_random_tree(depth + 1, full)
            children[1] = self.generate_random_tree(depth + 1, full)
        else:
            children = [self.generate_random_tree(depth + 1, full)]
            
        return Node(NodeType.OPERATOR, operation, children)

    def _generate_terminal(self) -> Node:
        """Generate a terminal node"""
        if self.available_features and random.random() < self.TERMINAL_PROB:
            return Node(NodeType.FEATURE, random.choice(self.available_features))
        return Node(NodeType.CONSTANT, random.gauss(0, 2))
    
    def create_initial_program(self, method: str = 'grow') -> None:
        """Create initial program tree"""
        if method == 'full':
            self.root = self.generate_random_tree(depth=0, full=True)
        elif method == 'grow':
            self.root = self.generate_random_tree(depth=0, full=False)
        elif method == 'ramped':
            self.root = self.generate_random_tree(depth=0, full=bool(random.getrandbits(1)))
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate program with input data"""
        if self.root is None:
            raise ValueError("Program not initialized")
        return self.root.evaluate(data)
    
    def copy(self) -> 'Program':
        """Create deep copy of program"""
        new_program = Program(
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            available_features=self.available_features,
            weight_preset=OperationWeightPreset.NATURAL
        )
        if self.root:
            new_program.root = self.root.copy()
        new_program.fitness = self.fitness
        return new_program

    def __str__(self) -> str:
        """Get string representation"""
        return str(self.root) if self.root else "Empty Program"
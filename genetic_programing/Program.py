from typing import List, Optional, Dict, Set, ClassVar
import numpy as np
import random
from Nodes import Node, NodeType
from OperationConfig import OperationConfig
from TreeComplexity import TreeComplexity, ComplexityMode, ComplexityMetrics
from OperationConfig import OperationConfig, OperationWeightPreset

class Program:
    """
    Represents a complete expression tree for genetic programming.
    Handles creation, evaluation, and manipulation of node trees.
    """
    
    # Class constant for terminal probability
    TERMINAL_PROB = 0.3
    
    def __init__(self, 
                 max_depth: int = 4,
                 min_depth: int = 2,
                 available_features: List[str] = None,
                 weight_preset: OperationWeightPreset = OperationWeightPreset.NATURAL,
                 custom_weights: Optional[Dict[str, float]] = None):
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.available_features = available_features or []
        self.root: Optional[Node] = None
        self.fitness: Optional[float] = None
        
        # Get operation weights and pre-compute weighted choices
        self.operation_weights = OperationConfig.get_operation_weights(weight_preset, custom_weights)
        self._init_weighted_operations()

    def generate_random_tree(self, depth: int = 0, full: bool = False) -> Node:
        """
        Generate a random expression tree.
        
        Args:
            depth: Current depth in the tree
            full: If True, creates a full tree to max_depth, if False allows variable depth
            
        Returns:
            Node: Root node of the generated tree
        """
        # Must create non-terminal if we haven't reached minimum depth
        must_expand = depth < self.min_depth
        
        # Must create terminal if we've reached maximum depth
        must_terminate = depth >= self.max_depth
        
        # For full trees, expand until max_depth then terminate
        if full:
            is_terminal = depth >= self.max_depth
        else:
            # For grow method, randomly decide to terminate if within valid depth range
            is_terminal = (not must_expand) and (
                must_terminate or random.random() < self.TERMINAL_PROB
            )
        
        if is_terminal:
            return self._generate_terminal()
        else:
            return self._generate_operator(depth, full)

    def _init_weighted_operations(self):
        """Initialize weighted operations for random selection"""
        ops, weights = zip(*self.operation_weights.items())
        self._weighted_ops = (ops, weights)

    def set_weight_preset(self, preset: OperationWeightPreset, custom_weights: Optional[Dict[str, float]] = None):
        """Change the operation weight preset"""
        self.operation_weights = OperationConfig.get_operation_weights(preset, custom_weights)
        self._init_weighted_operations()

    def _generate_operator(self, depth: int, full: bool) -> Node:
        """Generate operator node using weighted random selection"""
        operation = random.choices(self._weighted_ops[0], weights=self._weighted_ops[1])[0]
        
        if operation in OperationConfig.BINARY_OPS:
            children = [
                self.generate_random_tree(depth + 1, full),
                self.generate_random_tree(depth + 1, full)
            ]
        else:
            children = [self.generate_random_tree(depth + 1, full)]
            
        return Node(NodeType.OPERATOR, operation, children)

    def _generate_terminal(self) -> Node:
        """Generate terminal node with caching for repeated calls."""
        if random.random() < self.TERMINAL_PROB and self.available_features:
            feature = random.choice(self.available_features)
            return Node(NodeType.FEATURE, feature)
        else:
            # Use a more focused range for constants based on feature statistics
            value = random.gauss(0, 2)  # Normal distribution for more natural constants
            return Node(NodeType.CONSTANT, value)
    
    def create_initial_program(self, method: str = 'grow') -> None:
        """Create initial program with method selection."""
        methods = {
            'full': lambda: self.generate_random_tree(depth=0, full=True),
            'grow': lambda: self.generate_random_tree(depth=0, full=False),
            'ramped': lambda: self.generate_random_tree(
                depth=0,
                full=bool(random.getrandbits(1))  # Faster than random.choice
            )
        }
        
        if method not in methods:
            raise ValueError(f"Unknown initialization method: {method}")
            
        self.root = methods[method]()
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate program with validation."""
        if self.root is None:
            raise ValueError("Program has not been initialized")
        return self.root.evaluate(data)
    
    def copy(self) -> 'Program':
        """Create efficient copy of program."""
        new_program = Program(
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            available_features=self.available_features,
            weight_preset=OperationWeightPreset.NATURAL  # Use default preset for copy
        )
        if self.root:
            new_program.root = self.root.copy()
        new_program.fitness = self.fitness
        return new_program

    def __str__(self) -> str:
        """String representation of the program."""
        return str(self.root) if self.root else "Empty Program"
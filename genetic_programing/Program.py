from typing import List, Optional, Dict
import numpy as np
import random
from Nodes import Node, NodeType
from OperationConfig import OperationConfig
from TreeComplexity import TreeComplexity, ComplexityMode, ComplexityMetrics

class Program:
    """
    Represents a complete expression tree for genetic programming.
    Handles creation, evaluation, and manipulation of node trees.
    """
    
    def __init__(self, 
                 max_depth: int = 4,
                 min_depth: int = 2,
                 available_features: List[str] = None):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.available_features = available_features or []
        self.root: Optional[Node] = None
        self.fitness: Optional[float] = None

    def generate_random_tree(self, depth: int = 0, full: bool = False) -> Node:
        if depth >= self.max_depth:
            return self._generate_terminal()
            
        if depth < self.min_depth or (full and depth < self.max_depth):
            return self._generate_operator(depth, full)
        else:
            if random.random() < 0.7:
                return self._generate_operator(depth, full)
            else:
                return self._generate_terminal()
    
    def _generate_operator(self, depth: int, full: bool) -> Node:
        operation = random.choice(list(Node.OPERATIONS.keys()))
        
        # Check if operation is binary or unary
        if operation in ['add', 'subtract', 'multiply', 'divide', 'conditional', 'safe_divide']:
            children = [
                self.generate_random_tree(depth + 1, full),
                self.generate_random_tree(depth + 1, full)
            ]
        else:  # unary operations: abs, square, sqrt, sin, exp
            children = [self.generate_random_tree(depth + 1, full)]
            
        return Node(NodeType.OPERATOR, operation, children)
    
    def _generate_terminal(self) -> Node:
        if random.random() < 0.7 and self.available_features:
            feature = random.choice(self.available_features)
            return Node(NodeType.FEATURE, feature)
        else:
            value = random.uniform(-5, 5)
            return Node(NodeType.CONSTANT, value)
    
    def create_initial_program(self, method: str = 'grow') -> None:
        if method == 'full':
            self.root = self.generate_random_tree(full=True)
        elif method == 'grow':
            self.root = self.generate_random_tree(full=False)
        elif method == 'ramped':
            self.root = self.generate_random_tree(full=random.choice([True, False]))
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.root is None:
            raise ValueError("Program has not been initialized")
        return self.root.evaluate(data)
    
    def get_complexity(self, mode: ComplexityMode = ComplexityMode.HYBRID) -> ComplexityMetrics:
        return TreeComplexity.analyze(self.root, mode)
    
    def get_complexity_score(self, mode: ComplexityMode = ComplexityMode.HYBRID) -> float:
        metrics = self.get_complexity(mode)
        return TreeComplexity.get_complexity_score(metrics)
    
    def __str__(self) -> str:
        return "Empty Program" if self.root is None else str(self.root)
    
    def copy(self) -> 'Program':
        new_program = Program(
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            available_features=self.available_features.copy()
        )
        if self.root:
            new_program.root = self.root.copy()
        new_program.fitness = self.fitness
        return new_program
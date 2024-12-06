from typing import List, Union, Optional, Callable
import numpy as np
from enum import Enum

class NodeType(Enum):
    """Enum to define different types of nodes"""
    OPERATOR = "operator"    # Mathematical operations like +, -, *, /
    FEATURE = "feature"      # Input features/data columns
    CONSTANT = "constant"    # Numerical constants
    


class Node:
    """
    A node in the expression tree that can represent operations, features, or constants.
    
    Attributes:
        node_type (NodeType): Type of the node (operator, feature, or constant)
        value: The value/operation stored in the node
        children (List[Node]): Child nodes
    """
    
    # Dictionary of available operations and their implementations
    OPERATIONS = {
        # Basic arithmetic with safeguards
        'add': lambda x, y: np.clip(x + y, -1e6, 1e6),
        'subtract': lambda x, y: np.clip(x - y, -1e6, 1e6),
        'multiply': lambda x, y: np.clip(x * y, -1e6, 1e6),
        'divide': lambda x, y: np.divide(
            x, 
            y, 
            out=np.zeros_like(x), 
            where=np.abs(y) > 1e-10
        ),
        
        # Unary operations with bounds
        'abs': lambda x: np.abs(np.clip(x, -1e6, 1e6)),
        'square': lambda x: np.minimum(x * x, 1e6),
        'sqrt': lambda x: np.sqrt(np.abs(np.clip(x, 0, 1e6))),
        
        # New operations with numerical stability
        'sin': lambda x: np.sin(np.clip(x, -50, 50)),  # Prevent extreme inputs
        'exp': lambda x: np.clip(
            np.exp(-np.abs(np.clip(x, -50, 50))),  # Bounded exponential
            0,
            1e6
        ),
        'conditional': lambda x, y: np.where(
            np.clip(x, -1e6, 1e6) > 0,
            np.clip(y, -1e6, 1e6),
            -np.clip(y, -1e6, 1e6)
        ),
        'safe_divide': lambda x, y: np.divide(
            np.clip(x, -1e6, 1e6),
            1 + np.abs(np.clip(y, -1e6, 1e6))
        )
    }
    
    def __init__(self, 
                 node_type: NodeType,
                 value: Union[str, float, Callable] = None,
                 children: List['Node'] = None):
        """
        Initialize a new Node.
        
        Args:
            node_type: Type of the node (operator, feature, or constant)
            value: The value or operation of the node
            children: List of child nodes (optional)
        """
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        
        # Validate the node configuration
        self._validate_node()
    
    def _validate_node(self):
        """Validate the node's configuration based on its type."""
        if self.node_type == NodeType.OPERATOR:
            if self.value not in self.OPERATIONS:
                raise ValueError(f"Unknown operation: {self.value}")
            # Verify correct number of children for the operation
            if self.value in ['add', 'subtract', 'multiply', 'divide', 'conditional', 'safe_divide']:
                required_children = 2
            else:
                required_children = 1
            if len(self.children) != required_children:
                raise ValueError(f"Operation {self.value} requires {required_children} children, got {len(self.children)}")
                
        elif self.node_type == NodeType.FEATURE:
            if not isinstance(self.value, str):
                raise ValueError("Feature nodes must have a string value (column name)")
            if self.children:
                raise ValueError("Feature nodes cannot have children")
                
        elif self.node_type == NodeType.CONSTANT:
            if not isinstance(self.value, (int, float)):
                raise ValueError("Constant nodes must have a numeric value")
            if self.children:
                raise ValueError("Constant nodes cannot have children")
    
    def evaluate(self, data: dict) -> np.ndarray:
        """
        Evaluate the node and its subtree with numerical stability safeguards.
        
        Args:
            data: Dictionary mapping feature names to their values
            
        Returns:
            numpy.ndarray: Result of the evaluation
        """
        try:
            if self.node_type == NodeType.CONSTANT:
                return np.full_like(list(data.values())[0], self.value)
                
            elif self.node_type == NodeType.FEATURE:
                if self.value not in data:
                    raise KeyError(f"Feature '{self.value}' not found in input data")
                # Clip feature values for stability
                return np.clip(data[self.value], -1e6, 1e6)
                
            elif self.node_type == NodeType.OPERATOR:
                operation = self.OPERATIONS[self.value]
                if len(self.children) == 1:
                    result = operation(self.children[0].evaluate(data))
                else:
                    left = self.children[0].evaluate(data)
                    right = self.children[1].evaluate(data)
                    result = operation(left, right)
                
                # Final safety clip
                return np.clip(result, -1e6, 1e6)
                
        except Exception as e:
            # Return zeros on any numerical error
            return np.zeros_like(list(data.values())[0])
    
    def __str__(self) -> str:
        """Return a string representation of the node."""
        if self.node_type == NodeType.CONSTANT:
            return str(self.value)
        elif self.node_type == NodeType.FEATURE:
            return f"Feature({self.value})"
        else:
            if len(self.children) == 1:
                return f"{self.value}({self.children[0]})"
            else:
                return f"({self.children[0]} {self.value} {self.children[1]})"



    def copy(self) -> 'Node':
        """Create a deep copy of the node and its subtree with validation."""
        try:
            # First validate the current node before copying
            if self.node_type == NodeType.OPERATOR:
                # Check if operation requires 1 or 2 children
                required_children = 2 if self.value in ['add', 'subtract', 'multiply', 'divide', 'conditional', 'safe_divide'] else 1
                # If children count doesn't match, fix it before copying
                if len(self.children) != required_children:
                    if required_children == 1:
                        # For unary operations, just take the first child if multiple exist
                        self.children = self.children[:1]
                    else:
                        # For binary operations, ensure two children exist
                        while len(self.children) < 2:
                            # Add constant node as filler if needed
                            self.children.append(Node(NodeType.CONSTANT, 1.0))
            
            # Now do the deep copy with validated children
            new_children = [child.copy() for child in self.children]
            new_node = Node(self.node_type, self.value, new_children)
            
            # Validate the new node before returning
            new_node._validate_node()
            return new_node
            
        except Exception as e:
            # If something goes wrong, return a safe default node
            if self.node_type == NodeType.OPERATOR:
                return Node(NodeType.CONSTANT, 1.0)
            return Node(self.node_type, self.value, [])
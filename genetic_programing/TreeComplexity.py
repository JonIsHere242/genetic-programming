from enum import Enum
from dataclasses import dataclass
import numpy as np
from typing import Tuple
from Nodes import Node, NodeType
from OperationConfig import OperationConfig

class ComplexityMode(Enum):
    """Different modes for complexity evaluation"""
    SIMPLE = "simple"           # Just node count and depth
    COMPUTE = "compute"         # Full computational complexity analysis
    HYBRID = "hybrid"          # Adaptive based on tree size

@dataclass
class ComplexityMetrics:
    """Stores different complexity metrics for a program"""
    node_count: int = 0
    depth: int = 0
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    mode_used: ComplexityMode = ComplexityMode.SIMPLE

class TreeComplexity:
    """Handles complexity analysis of expression trees"""
    
    COMPUTE_THRESHOLD = 8  # Node count threshold for switching to compute mode

    @staticmethod
    def analyze(root: Node, mode: ComplexityMode = ComplexityMode.HYBRID) -> ComplexityMetrics:
        """Analyze tree complexity using specified mode"""
        if mode == ComplexityMode.HYBRID:
            quick_count = TreeComplexity._quick_node_count(root)
            has_expensive_ops = TreeComplexity._has_expensive_operations(root)
            use_compute = (quick_count > TreeComplexity.COMPUTE_THRESHOLD or 
                         (quick_count > 5 and has_expensive_ops))
            mode = ComplexityMode.COMPUTE if use_compute else ComplexityMode.SIMPLE
        
        return (TreeComplexity._compute_complexity(root) 
                if mode == ComplexityMode.COMPUTE 
                else TreeComplexity._simple_complexity(root))

    @staticmethod
    def _quick_node_count(node: Node) -> int:
        """Fast node counting"""
        if node is None:
            return 0
        return 1 + sum(TreeComplexity._quick_node_count(child) for child in node.children)

    @staticmethod
    def _has_expensive_operations(node: Node) -> bool:
        """Check if tree contains expensive operations"""
        if node is None:
            return False
        
        if (node.node_type == NodeType.OPERATOR and 
            OperationConfig.is_expensive_operation(node.value)):
            return True
            
        return any(TreeComplexity._has_expensive_operations(child) 
                  for child in node.children)

    @staticmethod
    def _simple_complexity(node: Node) -> ComplexityMetrics:
        """Calculate basic complexity metrics"""
        def analyze_simple(node: Node) -> Tuple[int, int]:
            if node is None:
                return 0, 0
            counts = [analyze_simple(child) for child in node.children]
            return (
                1 + sum(c[0] for c in counts),  # node count
                1 + max((c[1] for c in counts), default=0)  # depth
            )
        
        nodes, depth = analyze_simple(node)
        return ComplexityMetrics(
            node_count=nodes,
            depth=depth,
            mode_used=ComplexityMode.SIMPLE
        )

    @staticmethod
    def _compute_complexity(node: Node) -> ComplexityMetrics:
        """Calculate detailed complexity metrics"""
        def analyze_compute(node: Node) -> Tuple[int, int, float, float]:
            if node is None:
                return 0, 0, 0.0, 0.0
                
            if node.node_type in [NodeType.CONSTANT, NodeType.FEATURE]:
                return 1, 1, 1.0, 1.0
                
            children_metrics = [analyze_compute(child) for child in node.children]
            
            node_count = 1 + sum(m[0] for m in children_metrics)
            depth = 1 + max((m[1] for m in children_metrics), default=0)
            
            op_cost = OperationConfig.OPERATION_COSTS.get(node.value, 1.0)
            compute_cost = op_cost * (1 + sum(m[2] for m in children_metrics))
            
            memory_cost = OperationConfig.MEMORY_COSTS.get(node.value, 1.0)
            total_memory = memory_cost + max((m[3] for m in children_metrics), default=0)
            
            depth_penalty = np.log2(depth + 1)
            compute_cost *= depth_penalty
            
            return node_count, depth, compute_cost, total_memory
            
        nodes, depth, compute, memory = analyze_compute(node)
        return ComplexityMetrics(
            node_count=nodes,
            depth=depth,
            compute_cost=compute,
            memory_cost=memory,
            mode_used=ComplexityMode.COMPUTE
        )

    @staticmethod
    def get_complexity_score(metrics: ComplexityMetrics) -> float:
        """Calculate single complexity score from metrics"""
        if metrics.mode_used == ComplexityMode.SIMPLE:
            return 0.7 * metrics.node_count + 0.3 * metrics.depth
        else:
            normalized_compute = np.log2(1 + metrics.compute_cost)
            normalized_memory = np.log2(1 + metrics.memory_cost)
            normalized_depth = metrics.depth / 10
            
            return (0.4 * normalized_compute + 
                   0.3 * normalized_memory + 
                   0.2 * normalized_depth + 
                   0.1 * np.log2(1 + metrics.node_count))
        


        
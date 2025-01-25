from enum import Enum
from typing import Tuple, Dict, Set
import numpy as np
from Nodes import Node, NodeType
from OperationConfig import OperationConfig
from functools import lru_cache
import array

class ComplexityMode(Enum):
    """Different modes for complexity evaluation"""
    SIMPLE = "simple"
    COMPUTE = "compute"
    HYBRID = "hybrid"

class ComplexityMetrics:
    """Stores different complexity metrics for a program"""
    __slots__ = ['node_count', 'depth', 'compute_cost', 'memory_cost', 'mode_used']
    
    def __init__(self, 
                 node_count: int = 0, 
                 depth: int = 0, 
                 compute_cost: float = 0.0, 
                 memory_cost: float = 0.0, 
                 mode_used: ComplexityMode = ComplexityMode.SIMPLE):
        """Initialize complexity metrics with default values"""
        self.node_count = node_count
        self.depth = depth
        self.compute_cost = compute_cost
        self.memory_cost = memory_cost
        self.mode_used = mode_used

class TreeComplexity:
    """Optimized complexity analysis of expression trees"""
    
    # Class-level constants
    COMPUTE_THRESHOLD = 8
    
    # Pre-compute operation sets for faster lookups
    EXPENSIVE_OPS: Set[str] = frozenset(
        op for op, cost in zip(
            OperationConfig.OPERATIONS.keys(),
            OperationConfig.OPERATION_COSTS_ARRAY
        ) if cost >= 3.0
    )
    
    # Cache for operation costs
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_op_costs(op_name: str) -> Tuple[float, float]:
        """Cached lookup for operation costs"""
        op_idx = OperationConfig.OP_INDICES[op_name]
        return (
            OperationConfig.OPERATION_COSTS_ARRAY[op_idx],
            OperationConfig.MEMORY_COSTS_ARRAY[op_idx]
        )
    
    @staticmethod
    def analyze(root: Node, mode: ComplexityMode = ComplexityMode.HYBRID) -> ComplexityMetrics:
        """Analyze tree complexity using specified mode with optimized dispatch"""
        if root is None:
            return ComplexityMetrics()
            
        if mode == ComplexityMode.HYBRID:
            # Use array-based quick count for better performance
            quick_count = TreeComplexity._quick_node_count_array(root)
            mode = (ComplexityMode.COMPUTE 
                   if quick_count > TreeComplexity.COMPUTE_THRESHOLD 
                   or (quick_count > 5 and TreeComplexity._has_expensive_ops_fast(root))
                   else ComplexityMode.SIMPLE)
        
        analyzer = (TreeComplexity._compute_complexity_vectorized 
                   if mode == ComplexityMode.COMPUTE 
                   else TreeComplexity._simple_complexity_array)
        
        return analyzer(root)
    
    @staticmethod
    def _quick_node_count_array(node: Node) -> int:
        """Fast node counting using array-based stack"""
        if node is None:
            return 0
            
        count = 0
        stack = [node]  # Use list for better compatibility
        
        while stack:
            current = stack.pop()
            count += 1
            stack.extend(child for child in current.children)
            
        return count
    
    @staticmethod
    def _has_expensive_ops_fast(node: Node) -> bool:
        """Optimized check for expensive operations using pre-computed set"""
        if node is None:
            return False
            
        # Use pre-computed set for O(1) lookup
        if (node.node_type == NodeType.OPERATOR and 
            node.value in TreeComplexity.EXPENSIVE_OPS):
            return True
            
        return any(TreeComplexity._has_expensive_ops_fast(child) 
                  for child in node.children)
    
    @staticmethod
    def _simple_complexity_array(node: Node) -> ComplexityMetrics:
        """Calculate basic complexity metrics using array-based approach"""
        if node is None:
            return ComplexityMetrics()
            
        nodes = 0
        max_depth = 0
        stack = [(node, 1)]  # (node, depth)
        
        while stack:
            current, depth = stack.pop()
            nodes += 1
            max_depth = max(max_depth, depth)
            stack.extend((child, depth + 1) for child in current.children)
        
        return ComplexityMetrics(
            node_count=nodes,
            depth=max_depth,
            mode_used=ComplexityMode.SIMPLE
        )
    
    @staticmethod
    def _compute_complexity_vectorized(node: Node) -> ComplexityMetrics:
        """Vectorized computation of detailed complexity metrics"""
        if node is None:
            return ComplexityMetrics()
            
        # Use numpy arrays for vectorized operations
        metrics = np.zeros(4)  # [nodes, depth, compute, memory]
        stack = [(node, 1, None)]  # (node, depth, parent_metrics)
        
        while stack:
            current, depth, parent_metrics = stack.pop()
            
            # Update base metrics
            metrics[0] += 1  # node count
            metrics[1] = max(metrics[1], depth)  # max depth
            
            # Get operation costs using cached lookup
            if current.node_type == NodeType.OPERATOR:
                op_compute, op_memory = TreeComplexity._get_op_costs(current.value)
            else:
                op_compute = op_memory = 1.0
                
            # Calculate node metrics
            if parent_metrics is not None:
                node_metrics = np.array([
                    1,  # node count
                    depth,  # depth
                    op_compute * (1.0 + parent_metrics[2]),  # compute cost
                    op_memory + parent_metrics[3]  # memory cost
                ])
                metrics = np.maximum(metrics, node_metrics)
            
            # Add children to stack with current metrics
            for child in current.children:
                stack.append((child, depth + 1, metrics.copy()))
        
        # Apply depth penalty using vectorized operations
        depth_penalty = np.log2(metrics[1] + 1)
        metrics[2] *= depth_penalty
        
        return ComplexityMetrics(
            node_count=int(metrics[0]),
            depth=int(metrics[1]),
            compute_cost=metrics[2],
            memory_cost=metrics[3],
            mode_used=ComplexityMode.COMPUTE
        )
    
    @staticmethod
    def get_complexity_score(metrics: ComplexityMetrics) -> float:
        """Calculate complexity score using vectorized operations"""
        if metrics.mode_used == ComplexityMode.SIMPLE:
            weights = np.array([0.7, 0.3, 0.0, 0.0])
            values = np.array([
                metrics.node_count,
                metrics.depth,
                0.0,
                0.0
            ])
        else:
            weights = np.array([0.1, 0.2, 0.4, 0.3])
            values = np.array([
                np.log2(1 + metrics.node_count),
                metrics.depth / 10,
                np.log2(1 + metrics.compute_cost),
                np.log2(1 + metrics.memory_cost)
            ])
            
        return float(np.dot(weights, values))
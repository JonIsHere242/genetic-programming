from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
from Program import Program
from Nodes import Node, NodeType
import random
import math
from copy import deepcopy
from multiprocessing import cpu_count
from OperationConfig import OperationConfig

@dataclass
class MCTSNode:
    """Node in the MCTS tree (not to be confused with expression tree nodes)"""
    node: Node  # Reference to actual expression tree node
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    value: float = 0.0
    untried_actions: Set[str] = None  # Available modifications not yet tried
    
    def __post_init__(self):
        self.children = [] if self.children is None else self.children
        if self.untried_actions is None:
            self.untried_actions = self._get_valid_actions()
    
    def _get_valid_actions(self) -> Set[str]:
        """Get valid actions for this node based on its type"""
        actions = {'modify_value'}  # Always can modify value
        
        if self.node.node_type == NodeType.OPERATOR:
            actions.update({'change_operator', 'simplify', 'delete'})
        elif self.node.node_type == NodeType.CONSTANT:
            actions.update({'perturb', 'optimize'})
            
        if len(self.node.children) < 2:  # Can add child if not binary
            actions.add('add_child')
            
        return actions
    
    def ucb_score(self, exploration_weight: float) -> float:
        """Calculate UCB score for this node"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

class MCTSOptimizer:
    """Monte Carlo Tree Search optimizer for fine-tuning expressions"""
    
    def __init__(self,
                 exploration_weight: float = 1.414,
                 max_iterations: int = 100,
                 evaluation_samples: int = 1000,
                 ucb_constant: float = 2.0,
                 n_elite: int = 5,
                 n_threads: Optional[int] = None):
        
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.evaluation_samples = evaluation_samples
        self.ucb_constant = ucb_constant
        self.n_elite = n_elite
        self.n_threads = n_threads or max(1, cpu_count() - 1)
        self.best_score = float('inf')
        self.best_program = None
        
    def optimize_program(self, 
                        program: Program,
                        data: Dict[str, np.ndarray],
                        y_true: np.ndarray) -> Program:
        """Optimize program using MCTS with UCT"""
        self.data = data
        self.y_true = y_true
        self.best_program = program.copy()
        self.best_score = self._evaluate(self.best_program)
        
        # Create root MCTS node
        root = MCTSNode(program.root)
        
        # Run MCTS iterations
        for _ in range(self.max_iterations):
            node = self._select(root)
            if node.untried_actions:
                child = self._expand(node)
                reward = self._simulate(child)
                self._backpropagate(child, reward)
            else:
                reward = self._simulate(node)
                self._backpropagate(node, reward)
                
        return self.best_program
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to explore using UCT"""
        while not node.untried_actions and node.children:
            node = max(node.children,
                      key=lambda n: n.ucb_score(self.exploration_weight))
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying a new action"""
        action = random.choice(list(node.untried_actions))
        node.untried_actions.remove(action)
        
        # Create new program with modification
        new_program = self._apply_action(node, action)
        child_node = MCTSNode(
            node=new_program.root,
            parent=node
        )
        node.children.append(child_node)
        return child_node
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate the value of a node through random playouts"""
        program = self._create_program_from_node(node)
        score = self._evaluate(program)
        
        # Update best program if improvement found
        if score < self.best_score:
            self.best_score = score
            self.best_program = program.copy()
            
        # Convert score to reward (lower is better, so invert)
        return 1.0 / (1.0 + score)
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate results through the tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _evaluate(self, program: Program) -> float:
        """Evaluate program using subset of data"""
        try:
            # Sample subset of data
            indices = np.random.choice(
                len(self.y_true),
                min(self.evaluation_samples, len(self.y_true)),
                replace=False
            )
            
            sampled_data = {
                key: values[indices] for key, values in self.data.items()
            }
            sampled_y = self.y_true[indices]
            
            # Get predictions and calculate MSE
            y_pred = program.evaluate(sampled_data)
            mse = np.mean((sampled_y - y_pred) ** 2)
            
            # Add small penalty for complexity
            complexity_penalty = 0.01 * len(self._get_all_nodes(program.root))
            return mse + complexity_penalty
            
        except Exception:
            return float('inf')
    
    def _apply_action(self, node: MCTSNode, action: str) -> Program:
        """Apply modification action to create new program"""
        program = self._create_program_from_node(node)
        target_node = program.root

        if action == 'modify_value':
            if target_node.node_type == NodeType.CONSTANT:
                # Perturb constant value
                target_node.value *= (1 + random.gauss(0, 0.1))
            elif target_node.node_type == NodeType.OPERATOR:
                # Change to similar operator maintaining arity
                current_arity = len(target_node.children)
                # Use OperationConfig instead of accessing from Node
                available_ops = [
                    op for op in OperationConfig.OPERATIONS.keys()
                    if (current_arity == 2 and op in OperationConfig.BINARY_OPS) or
                       (current_arity == 1 and op in OperationConfig.UNARY_OPS)
                ]
                if available_ops:
                    target_node.value = random.choice(available_ops)

        elif action == 'simplify':
            self._try_simplify(target_node)

        elif action == 'add_child':
            if len(target_node.children) < 2:
                new_child = Node(
                    NodeType.CONSTANT,
                    value=random.gauss(0, 1)
                )
                target_node.children.append(new_child)

        elif action == 'delete' and target_node.children:
            # Replace with first child
            child = target_node.children[0]
            target_node.value = child.value
            target_node.node_type = child.node_type
            target_node.children = child.children

        return program
    
    def _get_valid_actions(self) -> Set[str]:
        """Get valid actions for this node based on its type"""
        actions = {'modify_value'}  # Always can modify value

        if self.node.node_type == NodeType.OPERATOR:
            actions.update({'change_operator', 'simplify', 'delete'})
        elif self.node.node_type == NodeType.CONSTANT:
            actions.update({'perturb', 'optimize'})

        # Check against OperationConfig instead of Node
        if len(self.node.children) < 2:  # Can add child if not binary
            actions.add('add_child')

        return actions




    def _try_simplify(self, node: Node) -> None:
        """Try to simplify expression at node"""
        if node.node_type != NodeType.OPERATOR:
            return
            
        if node.value in ['add', 'subtract'] and len(node.children) == 2:
            if (node.children[1].node_type == NodeType.CONSTANT and 
                abs(node.children[1].value) < 1e-10):
                # Simplify x+0 or x-0 to x
                child = node.children[0]
                node.value = child.value
                node.node_type = child.node_type
                node.children = child.children
                
        elif node.value == 'multiply' and len(node.children) == 2:
            if (node.children[1].node_type == NodeType.CONSTANT):
                if abs(node.children[1].value) < 1e-10:
                    # Simplify x*0 to 0
                    node.value = 0
                    node.node_type = NodeType.CONSTANT
                    node.children = []
                elif abs(node.children[1].value - 1) < 1e-10:
                    # Simplify x*1 to x
                    child = node.children[0]
                    node.value = child.value
                    node.node_type = child.node_type
                    node.children = child.children
    
    def _create_program_from_node(self, mcts_node: MCTSNode) -> Program:
        """Create a new program with the MCTS node's expression tree"""
        program = Program(
            max_depth=10,  # Large enough to handle modifications
            min_depth=1,
            available_features=list(self.data.keys())
        )
        program.root = deepcopy(mcts_node.node)
        return program
    
    def _get_all_nodes(self, node: Node) -> List[Node]:
        """Get all nodes in expression tree"""
        if not node:
            return []
        return [node] + sum((self._get_all_nodes(child) for child in node.children), [])
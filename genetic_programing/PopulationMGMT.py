from typing import List, Callable, Dict, Union, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, mutual_info_score
import warnings
from Program import Program
from Nodes import Node, NodeType
from OperationConfig import OperationConfig, OperationWeightPreset
from TreeComplexity import TreeComplexity, ComplexityMode, ComplexityMetrics
import random
from dataclasses import dataclass
from time import time
import math
from multiprocessing import Pool, cpu_count
from Verbose import VerboseHandler, GenerationMetrics
import tqdm as tqdm
from GuidedEvolution import MCTSOptimizer
from numba import jit, prange
from functools import lru_cache

@dataclass 
class FitnessMetrics:
    """Stores various fitness metrics for a program"""
    mae: float = float('inf')
    mse: float = float('inf')
    rmse: float = float('inf')
    ic: float = 0.0  # Information Coefficient (Spearman)
    mic: float = 0.0  # Mutual Information Coefficient
    complexity_metrics: ComplexityMetrics = None
    final_score: float = float('inf')



@dataclass
class BatchResult:
    """Results from processing a batch of programs"""
    predictions: np.ndarray
    fitnesses: List[float]
    metrics: List[FitnessMetrics]





class GeneticOperators:
    """Handles genetic operations with configurable parameters"""
    
    @staticmethod
    def crossover(parent1: Program, parent2: Program, 
                 crossover_prob: float) -> Tuple[Program, Program]:
        """Perform crossover with configurable probability"""
        if random.random() > crossover_prob:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()
        
        nodes1 = GeneticOperators._get_all_nodes(child1.root)
        nodes2 = GeneticOperators._get_all_nodes(child2.root)

        if not nodes1 or not nodes2:
            return child1, child2
            
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        
        GeneticOperators._swap_nodes(child1.root, child2.root, node1, node2)
        
        return child1, child2
    
    @staticmethod
    def mutate(program: Program, 
               mutation_prob: float,
               mutation_ops: Dict[str, float]) -> Program:
        """Apply mutation with configurable operations and probabilities"""
        if random.random() > mutation_prob:
            return program.copy()
            
        mutated = program.copy()
        nodes = GeneticOperators._get_all_nodes(mutated.root)
        
        if not nodes:
            return mutated
            
        node = random.choice(nodes)
        
        # Select mutation operation based on configured weights
        op_type = random.choices(
            list(mutation_ops.keys()),
            weights=list(mutation_ops.values())
        )[0]
        
        mutation_funcs = {
            'point': GeneticOperators._point_mutation,
            'insert': GeneticOperators._insert_node,
            'delete': GeneticOperators._delete_node,
            'swap': GeneticOperators._swap_siblings
        }
        
        mutation_funcs[op_type](mutated.root, node)
        
        return mutated
    



    @staticmethod
    def _get_all_nodes(root: Node) -> List[Node]:
        """Get all nodes in the tree"""
        if not root:
            return []
            
        nodes = [root]
        for child in root.children:
            nodes.extend(GeneticOperators._get_all_nodes(child))
        return nodes
    
    @staticmethod
    def _swap_nodes(root1: Node, root2: Node, node1: Node, node2: Node) -> None:
        """Swap two nodes in different trees"""
        # Find parents of nodes
        parent1 = GeneticOperators._find_parent(root1, node1)
        parent2 = GeneticOperators._find_parent(root2, node2)
        
        if parent1:
            idx1 = parent1.children.index(node1)
            parent1.children[idx1] = node2
        
        if parent2:
            idx2 = parent2.children.index(node2)
            parent2.children[idx2] = node1
    
    @staticmethod
    def _find_parent(root: Node, target: Node) -> Optional[Node]:
        """Find parent of a target node"""
        if not root or not root.children:
            return None
            
        if target in root.children:
            return root
            
        for child in root.children:
            parent = GeneticOperators._find_parent(child, target)
            if parent:
                return parent
        return None
    
    @staticmethod
    def _point_mutation(root: Node, node: Node) -> None:
        """Modify a single node's value"""
        if node.node_type == NodeType.OPERATOR:
            if OperationConfig.is_binary_operation(node.value):
                node.value = random.choice([op for op in OperationConfig.OPERATIONS 
                                          if OperationConfig.is_binary_operation(op)])
            else:
                node.value = random.choice([op for op in OperationConfig.OPERATIONS 
                                          if OperationConfig.is_unary_operation(op)])
        elif node.node_type == NodeType.CONSTANT:
            node.value = random.uniform(-5, 5)
            
    @staticmethod
    def _insert_node(root: Node, node: Node) -> None:
        """Insert a new node as parent of selected node"""
        parent = GeneticOperators._find_parent(root, node)
        if not parent:
            return
            
        idx = parent.children.index(node)
        new_node = Node(
            NodeType.OPERATOR,
            random.choice([op for op in OperationConfig.OPERATIONS 
                          if OperationConfig.is_unary_operation(op)]),
            [node]
        )
        parent.children[idx] = new_node
    
    @staticmethod
    def _delete_node(root: Node, node: Node) -> None:
        """Delete a node, connecting its parent to its child"""
        if node.node_type != NodeType.OPERATOR or not node.children:
            return
            
        parent = GeneticOperators._find_parent(root, node)
        if not parent:
            return
            
        idx = parent.children.index(node)
        parent.children[idx] = node.children[0]
    
    @staticmethod
    def _swap_siblings(root: Node, node: Node) -> None:
        """Swap node with its sibling if it has one"""
        parent = GeneticOperators._find_parent(root, node)
        if not parent or len(parent.children) < 2:
            return
            
        idx = parent.children.index(node)
        other_idx = (idx + 1) % len(parent.children)
        parent.children[idx], parent.children[other_idx] = \
            parent.children[other_idx], parent.children[idx]



class PopulationManager:
    """Manages a population of genetic programs with flexible fitness evaluation"""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_depth: int = 4,
                 features: List[str] = None,
                 fitness_function: Optional[Callable] = None,
                 tournament_size: int = 5,
                 parsimony_coefficient: float = 0.01,
                 complexity_mode: ComplexityMode = ComplexityMode.HYBRID,
                 complexity_ratio_limit: float = 100.0,
                 # Add genetic operation parameters
                 crossover_probability: float = 0.7,
                 mutation_probability: float = 0.3,
                 elitism_ratio: float = 0.05,  # Percentage of population to preserve
                 mutation_operations: Optional[Dict[str, float]] = None,
                 # Existing parameters
                 verbose: int = 1,
                 output_prefix: str = "",
                 use_progress_bar: bool = True,
                 total_generations: int = 10,
                 operation_preset: OperationWeightPreset = OperationWeightPreset.RANDOM):
        """Initialize Population Manager with configurable genetic parameters"""
        # Existing initialization
        self.population_size = population_size
        self.max_depth = max_depth
        self.features = features or []
        self.custom_fitness = fitness_function
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.complexity_mode = complexity_mode
        self.complexity_ratio_limit = complexity_ratio_limit
        
        # New genetic parameters
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_ratio = elitism_ratio
        
        # Default mutation operation weights if not provided
        self.mutation_operations = mutation_operations or {
            'point': 0.4,    # Modify node value
            'insert': 0.2,   # Insert new node
            'delete': 0.2,   # Delete node
            'swap': 0.2      # Swap siblings
        }
        
        # Normalize mutation operation weights
        total_weight = sum(self.mutation_operations.values())
        self.mutation_operations = {
            k: v/total_weight for k, v in self.mutation_operations.items()
        }
        
        # Rest of initialization
        self.population: List[Program] = []
        self.best_program: Optional[Program] = None
        self.best_complexity: Optional[float] = None
        self.generation = 0
        self.operation_preset = operation_preset
        
        # Verbose handler initialization
        output_dir = f"{output_prefix}gp_outputs" if output_prefix else "gp_outputs"
        self.verbose_handler = VerboseHandler(
            level=verbose,
            use_progress_bar=use_progress_bar,
            output_dir=output_dir,
            feature_names=features,
            total_generations=total_generations
        )
    
    def initialize_population(self) -> None:
        """Initialize population with random programs using specified operation preset"""
        self.population = []
        methods = ['full', 'grow', 'ramped']
        
        for i in range(self.population_size):
            method = methods[i % len(methods)]
            program = Program(
                max_depth=self.max_depth,
                min_depth=2,
                available_features=self.features,
                weight_preset=self.operation_preset  # Pass the preset to Program
            )
            program.create_initial_program(method=method)
            self.population.append(program)


    @staticmethod
    @jit(nopython=True)
    def _tournament_select_fast(fitnesses: np.ndarray, tournament_size: int) -> int:
        """Fast tournament selection using pre-computed fitness array"""
        tournament_idx = np.random.choice(len(fitnesses), tournament_size, replace=False)
        return tournament_idx[np.argmin(fitnesses[tournament_idx])]


    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_fitness_calc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Optimized calculation of basic fitness metrics"""
        diff = y_true - y_pred
        abs_diff = np.abs(diff)
        
        mae = np.mean(abs_diff)
        mse = np.mean(diff * diff)
        rmse = np.sqrt(mse)
        
        return mae, mse, rmse
    


    def evaluate_fitness(self, y_true: np.ndarray, data: Dict[str, np.ndarray]) -> None:
        """Optimized fitness evaluation"""
        best_fitness = float('inf')
        
        # Pre-allocate arrays for vectorized operations
        n_samples = len(y_true)
        predictions = np.zeros((len(self.population), n_samples))
        
        # Batch evaluate programs
        for i, program in enumerate(self.population):
            try:
                predictions[i] = program.evaluate(data)
            except Exception:
                predictions[i] = np.full(n_samples, np.nan)
        
        # Mask invalid predictions
        mask = np.isfinite(predictions)
        
        # Vectorized metrics calculation
        for i, program in enumerate(self.population):
            try:
                prog_mask = mask[i]
                if not np.any(prog_mask):
                    program.fitness = float('inf')
                    continue
                    
                # Fast metrics calculation
                mae, mse, rmse = self._fast_fitness_calc(
                    y_true[prog_mask], 
                    predictions[i, prog_mask]
                )
                
                # Get complexity metrics once
                complexity_metrics = TreeComplexity.analyze(program.root, self.complexity_mode)
                complexity_score = TreeComplexity.get_complexity_score(complexity_metrics)
                
                # Calculate final score efficiently
                if self.best_complexity and self.best_complexity > 0:
                    complexity_ratio = complexity_metrics.compute_cost / self.best_complexity
                    complexity_penalty = np.exp(
                        np.clip(complexity_ratio / self.complexity_ratio_limit - 1, 0, 10)
                    ) - 1
                else:
                    complexity_penalty = 0
                
                final_score = (
                    0.35 * mae +
                    0.25 * rmse +
                    0.15 * self.parsimony_coefficient * complexity_score
                ) * (1 + complexity_penalty)
                
                program.fitness = final_score
                
                # Update best program
                if final_score < best_fitness:
                    best_fitness = final_score
                    self.best_program = program.copy()
                    self.best_complexity = complexity_metrics.compute_cost
                    
            except Exception:
                program.fitness = float('inf')
                




    def evaluate_single_program(self, program: Program, y_true: np.ndarray, 
                              data: Dict[str, np.ndarray]) -> None:
        """
        Evaluate a single program and set its fitness with robust error handling.
        
        Args:
            program: Program to evaluate
            y_true: True target values
            data: Dictionary of feature arrays
        """
        try:
            # Get program predictions with timeout and error catching
            try:
                y_pred = program.evaluate(data)
                
                # Check for invalid predictions
                if y_pred is None or not isinstance(y_pred, np.ndarray):
                    program.fitness = float('inf')
                    return
                    
                # Check for NaN or Inf values
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    program.fitness = float('inf')
                    return
                    
                # Ensure matching shapes
                if y_pred.shape != y_true.shape:
                    program.fitness = float('inf')
                    return
                    
            except Exception as e:
                program.fitness = float('inf')
                return
    
            # Calculate metrics with error handling
            try:
                metrics = self._calculate_metrics(y_true, y_pred, program)
                if metrics is None:
                    program.fitness = float('inf')
                    return
            except Exception as e:
                program.fitness = float('inf')
                return
    
            # Handle complexity ratio calculation safely
            try:
                if self.best_complexity is not None and self.best_complexity > 0:
                    current_complexity = metrics.complexity_metrics.compute_cost
                    
                    # Ensure both values are positive and non-zero
                    if current_complexity > 0:
                        complexity_ratio = current_complexity / self.best_complexity
                        
                        # Apply penalty if ratio exceeds limit
                        if complexity_ratio > self.complexity_ratio_limit:
                            # Use log scaling to prevent excessive penalties
                            penalty = np.log1p(complexity_ratio / self.complexity_ratio_limit)
                            metrics.final_score *= (1.0 + penalty)
            except Exception as e:
                # Don't fail completely on complexity calculation error
                pass
            
            # Set final fitness
            try:
                if self.custom_fitness:
                    program.fitness = self.custom_fitness(y_true, y_pred, metrics)
                else:
                    program.fitness = metrics.final_score
    
                # Final sanity check on fitness value
                if not np.isfinite(program.fitness):
                    program.fitness = float('inf')
                    
            except Exception as e:
                program.fitness = float('inf')
    
        except Exception as e:
            # Catch any remaining unhandled exceptions
            program.fitness = float('inf')
            
        # Final validation
        if not hasattr(program, 'fitness') or program.fitness is None:
            program.fitness = float('inf')
    
    

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          program: Program) -> FitnessMetrics:
        """Calculate various fitness metrics with robust error handling"""
        # Handle potential NaN/Inf values
        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) < 2:
            return FitnessMetrics()

        try:
            # Calculate basic error metrics
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)

            # Calculate IC (Spearman correlation)
            ic = self._calculate_ic(y_true_clean, y_pred_clean)

            # Calculate MIC (Mutual Information)
            mic = self._calculate_mic(y_true_clean, y_pred_clean)

            # Calculate complexity metrics using TreeComplexity
            complexity_metrics = TreeComplexity.analyze(program.root, self.complexity_mode)
            complexity_score = TreeComplexity.get_complexity_score(complexity_metrics)

            metrics = FitnessMetrics(
                mae=mae,
                mse=mse,
                rmse=rmse,
                ic=ic,
                mic=mic,
                complexity_metrics=complexity_metrics
            )

            # Calculate final score
            metrics.final_score = self._default_fitness_function(metrics, complexity_score)

            return metrics
        
        except Exception as e:
            warnings.warn(f"Error calculating metrics: {e}")
            return FitnessMetrics()
    
    def _calculate_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Safely calculate Information Coefficient (Spearman correlation)"""
        try:
            # Check if either array is constant (all values the same)
            if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
                return 0.0

            # Additional check for near-zero variance
            if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
                return 0.0

            # Suppress spearmanr warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                ic = abs(spearmanr(y_true, y_pred)[0])

            return 0.0 if np.isnan(ic) else ic
        except:
            return 0.0

    def _calculate_mic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Safely calculate Mutual Information Coefficient"""
        try:
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                return 0.0
                
            # Normalize data
            y_true_norm = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-10)
            y_pred_norm = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-10)
            
            # Discretize and calculate MI
            y_true_disc = np.digitize(y_true_norm, np.linspace(min(y_true_norm), max(y_true_norm), 10))
            y_pred_disc = np.digitize(y_pred_norm, np.linspace(min(y_pred_norm), max(y_pred_norm), 10))
            
            mic = mutual_info_score(y_true_disc, y_pred_disc)
            return 0.0 if np.isnan(mic) else mic
        except:
            return 0.0
    
    def _default_fitness_function(self, metrics: FitnessMetrics, complexity_score: float) -> float:
        """Default fitness function combining multiple metrics with safety checks"""
        # Error components
        mae_component = 0.35 * metrics.mae if np.isfinite(metrics.mae) else float('inf')
        rmse_component = 0.25 * metrics.rmse if np.isfinite(metrics.rmse) else float('inf')
        
        # Correlation components
        ic_component = 0.15 * (1 - metrics.ic) if np.isfinite(metrics.ic) else 0.15
        mic_component = 0.10 * (1 - metrics.mic) if np.isfinite(metrics.mic) else 0.10
        
        # Complexity component (weighted by parsimony coefficient)
        complexity_component = 0.15 * self.parsimony_coefficient * complexity_score

        fitness = (mae_component + 
                  rmse_component + 
                  ic_component + 
                  mic_component + 
                  complexity_component)
    
        return float('inf') if np.isnan(fitness) else fitness
    
    
    def get_population_stats(self) -> Dict[str, Union[float, ComplexityMetrics]]:
        """Get statistical information about the population"""
        fitnesses = [p.fitness for p in self.population]
        complexity_metrics = [TreeComplexity.analyze(p.root, self.complexity_mode) 
                            for p in self.population]
        
        return {
            'best_fitness': min(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'avg_node_count': np.mean([m.node_count for m in complexity_metrics]),
            'avg_depth': np.mean([m.depth for m in complexity_metrics]),
            'avg_compute_cost': np.mean([m.compute_cost for m in complexity_metrics]),
            'avg_memory_cost': np.mean([m.memory_cost for m in complexity_metrics]),
            'population_size': len(self.population),
            'generation': self.generation
        }
    

    def evolve_population(self):
        """Create next generation using configured genetic parameters"""
        population_size = len(self.population)
        elite_size = max(1, int(population_size * self.elitism_ratio))
        
        # Pre-compute fitness array for fast selection
        fitnesses = np.array([p.fitness for p in self.population])
        
        # Sort once for elitism
        elite_indices = np.argpartition(fitnesses, elite_size)[:elite_size]
        new_population = [self.population[i].copy() for i in elite_indices]
        
        # Pre-allocate parent arrays
        parent_indices = np.zeros((2, population_size - elite_size), dtype=np.int32)
        
        # Fast tournament selection for all parents at once
        for i in range(parent_indices.shape[1]):
            for j in range(2):
                parent_indices[j, i] = self._tournament_select_fast(
                    fitnesses, self.tournament_size
                )
        
        # Batch process genetic operations
        for i in range(0, len(parent_indices[0]), 2):
            parent1 = self.population[parent_indices[0, i]]
            parent2 = self.population[parent_indices[1, i]]
            
            # Crossover with configured probability
            if random.random() < self.crossover_probability:
                child1, child2 = GeneticOperators.crossover(
                    parent1, parent2, 
                    self.crossover_probability
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation with configured probability and operations
            if random.random() < self.mutation_probability:
                child1 = GeneticOperators.mutate(
                    child1,
                    self.mutation_probability,
                    self.mutation_operations
                )
            if random.random() < self.mutation_probability:
                child2 = GeneticOperators.mutate(
                    child2,
                    self.mutation_probability,
                    self.mutation_operations
                )
            
            new_population.extend([child1, child2])
            
            # Ensure population size
            if len(new_population) >= population_size:
                break
        
        self.population = new_population[:population_size]




    def run_generation(self, generation: int, y_true: np.ndarray, data: Dict[str, np.ndarray], 
                      use_mcts: bool = False,
                      mcts_params: Optional[dict] = None) -> None:
        """Run a single generation with optional MCTS optimization"""
        start_time = time()

        # Evaluate fitness
        self.evaluate_fitness(y_true, data)

        # Evolve population
        self.evolve_population()

        # Apply MCTS optimization if enabled
        if use_mcts:
            # Get top programs
            sorted_pop = sorted(self.population, key=lambda x: x.fitness)
            mcts_params = mcts_params or {}

            # Create optimizer with provided or default parameters
            optimizer = MCTSOptimizer(
                exploration_weight=mcts_params.get('exploration_weight', 1.414),
                max_iterations=mcts_params.get('max_iterations', 50),
                evaluation_samples=mcts_params.get('evaluation_samples', 1000),
                ucb_constant=mcts_params.get('ucb_constant', 2.0),
                n_elite=mcts_params.get('n_elite', 5),
                n_threads=mcts_params.get('n_threads', None)
            )

            # Temporarily disable verbose output for MCTS optimization
            original_verbose = self.verbose_handler.level
            self.verbose_handler.level = 0

            # Get elite programs to optimize
            elite_programs = sorted_pop[:optimizer.n_elite]

            # Run parallel optimization
            with Pool(optimizer.n_threads) as pool:
                optimized_programs = pool.starmap(
                    optimizer.optimize_program,
                    [(prog, data, y_true) for prog in elite_programs]
                )

            # Replace original programs with optimized versions if they're better
            for i, (orig, opt) in enumerate(zip(elite_programs, optimized_programs)):
                if opt.fitness < orig.fitness:
                    self.population[i] = opt

            # Restore original verbose level
            self.verbose_handler.level = original_verbose

            # Re-evaluate population
            self.evaluate_fitness(y_true, data)

        # Calculate generation metrics
        stats = self.get_population_stats()
        best_program = self.get_best_program()

        if best_program:
            y_pred = best_program.evaluate(data)
            mae = mean_absolute_error(y_true, y_pred)

            metrics = GenerationMetrics(
                generation=generation,
                best_fitness=stats['best_fitness'],
                avg_fitness=stats['avg_fitness'],
                std_fitness=stats['fitness_std'],
                best_expression=str(best_program),
                mae=mae,
                compute_cost=stats['avg_compute_cost'],
                memory_cost=stats['avg_memory_cost'],
                runtime=time() - start_time
            )

            # Print generation stats after all optimization is complete
            self.verbose_handler.print_generation_stats(metrics)

            # Export solution if it's the last generation
            if generation == self.generation + 1:
                self.verbose_handler.export_solution(best_program)

        self.generation += 1
        # After completing all generations and MCTS steps
 



    
    def get_best_program(self) -> Optional[Program]:
        """Get the best program found so far"""
        return self.best_program
    



class ParallelPopulationManager:
    """Manages a large population of genetic programs with parallel processing"""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_depth: int = 4,
                 features: List[str] = None,
                 fitness_function: Optional[Callable] = None,
                 tournament_size: int = 5,
                 parsimony_coefficient: float = 0.01,
                 complexity_mode: ComplexityMode = ComplexityMode.HYBRID,
                 complexity_ratio_limit: float = 100.0,
                 n_workers: int = None,
                 batch_size: int = 1000):
        """
        Initialize Population Manager with parallel processing capabilities.
        
        Args:
            population_size: Number of programs in population
            max_depth: Maximum depth of program trees
            features: List of available feature names
            fitness_function: Custom fitness function (optional)
            tournament_size: Size of tournament for selection
            parsimony_coefficient: Weight for complexity penalty
            complexity_mode: Mode for calculating program complexity
            complexity_ratio_limit: Upper limit for complexity based on current best solution as a ratio ie 100x
            n_workers: Number of CPU cores to use (default: all available - 1)
            batch_size: Number of programs to evaluate per batch
        """
        self.population_size = population_size
        self.max_depth = max_depth
        self.features = features or []
        self.custom_fitness = fitness_function
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.complexity_mode = complexity_mode
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.batch_size = batch_size

        self.complexity_ratio_limit = complexity_ratio_limit
        self.best_complexity = None
        
        self.population: List[Program] = []
        self.best_program: Optional[Program] = None
        self.generation = 0
        
        print(f"Initializing with {self.n_workers} workers")
        
    def initialize_population(self) -> None:
        """Initialize population with random programs in parallel"""
        n_batches = math.ceil(self.population_size / self.batch_size)
        
        print(f"Creating initial population in {n_batches} batches...")
        start_time = time()
        
        with Pool(self.n_workers) as pool:
            # Create batches of programs
            batch_sizes = [self.batch_size] * (n_batches - 1)
            batch_sizes.append(self.population_size - sum(batch_sizes))
            
            batch_args = [(size, self.max_depth, self.features) 
                         for size in batch_sizes]
            
            # Initialize batches in parallel
            results = pool.starmap(self._initialize_batch, batch_args)
            
            # Combine results
            self.population = [prog for batch in results for prog in batch]
        
        print(f"Population initialized in {time() - start_time:.2f} seconds")
    
    @staticmethod
    def _initialize_batch(batch_size: int, max_depth: int, features: List[str]) -> List[Program]:
        """Initialize a batch of programs"""
        methods = ['full', 'grow', 'ramped']
        programs = []
        
        for i in range(batch_size):
            method = methods[i % len(methods)]
            program = Program(
                max_depth=max_depth,
                min_depth=2,
                available_features=features
            )
            program.create_initial_program(method=method)
            programs.append(program)
            
        return programs
    
    def evaluate_fitness(self, y_true: np.ndarray, data: Dict[str, np.ndarray]) -> None:
        """Evaluate fitness with complexity constraints"""
        best_fitness = float('inf')

        for program in self.population:
            try:
                # Get program predictions
                y_pred = program.evaluate(data)

                # Calculate metrics
                metrics = self._calculate_metrics(y_true, y_pred, program)

                # Check complexity ratio if we have a reference
                if self.best_complexity is not None:
                    current_complexity = metrics.complexity_metrics.compute_cost
                    complexity_ratio = current_complexity / self.best_complexity

                    # If complexity is too high relative to best known solution,
                    # penalize the fitness severely
                    if complexity_ratio > self.complexity_ratio_limit:
                        metrics.final_score *= (complexity_ratio / self.complexity_ratio_limit)

                # Use custom fitness function if provided, otherwise use default
                if self.custom_fitness:
                    program.fitness = self.custom_fitness(y_true, y_pred, metrics)
                else:
                    program.fitness = metrics.final_score

                # Update best program and complexity reference
                if program.fitness < best_fitness:
                    best_fitness = program.fitness
                    self.best_program = program.copy()
                    self.best_complexity = metrics.complexity_metrics.compute_cost

            except Exception as e:
                warnings.warn(f"Error evaluating program: {e}")
                program.fitness = float('inf')
    
    @staticmethod
    def _evaluate_batch(programs: List[Program], y_true: np.ndarray, 
                       data: Dict[str, np.ndarray], custom_fitness: Optional[Callable],
                       complexity_mode: ComplexityMode, 
                       parsimony_coefficient: float) -> BatchResult:
        """Evaluate a batch of programs"""
        predictions = []
        fitnesses = []
        metrics_list = []
        
        for program in programs:
            try:
                # Get predictions
                y_pred = program.evaluate(data)
                
                # Calculate metrics
                metrics = PopulationManager._calculate_metrics_static(
                    y_true, y_pred, program, complexity_mode, parsimony_coefficient
                )
                
                # Calculate fitness
                if custom_fitness:
                    fitness = custom_fitness(y_true, y_pred, metrics)
                else:
                    fitness = metrics.final_score
                
                predictions.append(y_pred)
                fitnesses.append(fitness)
                metrics_list.append(metrics)
                
            except Exception as e:
                warnings.warn(f"Error evaluating program: {e}")
                predictions.append(np.full_like(y_true, np.nan))
                fitnesses.append(float('inf'))
                metrics_list.append(FitnessMetrics())
        
        return BatchResult(
            predictions=np.array(predictions),
            fitnesses=fitnesses,
            metrics=metrics_list
        )
    

    def evolve_population(self):
        """Create next generation through selection and genetic operators"""
        new_population = []
        elite_size = max(1, self.population_size // 20)  # Keep top 5%

        # Sort population by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)

        # Elitism - keep best programs
        new_population.extend(p.copy() for p in sorted_pop[:elite_size])

        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            # Crossover
            child1, child2 = GeneticOperators.crossover(parent1, parent2)

            # Mutation
            child1 = GeneticOperators.mutate(child1)
            child2 = GeneticOperators.mutate(child2)

            # Check complexity before adding to population
            if self.best_complexity is not None:
                for child in [child1, child2]:
                    complexity = TreeComplexity.analyze(child.root, self.complexity_mode)
                    ratio = complexity.compute_cost / self.best_complexity

                    # If too complex, try to simplify by regenerating
                    attempts = 0
                    while ratio > self.complexity_ratio_limit and attempts < 3:
                        child.create_initial_program(method='grow')
                        complexity = TreeComplexity.analyze(child.root, self.complexity_mode)
                        ratio = complexity.compute_cost / self.best_complexity
                        attempts += 1

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        
    @staticmethod
    def _evolve_batch(batch_size: int, sorted_pop: List[Program], 
                      tournament_size: int) -> List[Program]:
        """Create a batch of offspring through tournament selection and genetic operators"""
        offspring = []
        
        while len(offspring) < batch_size:
            # Select parents
            parent1 = PopulationManager._tournament_select_static(sorted_pop, tournament_size)
            parent2 = PopulationManager._tournament_select_static(sorted_pop, tournament_size)
            
            # Create offspring
            child1, child2 = GeneticOperators.crossover(parent1, parent2)
            child1 = GeneticOperators.mutate(child1)
            child2 = GeneticOperators.mutate(child2)
            
            offspring.append(child1)
            if len(offspring) < batch_size:
                offspring.append(child2)
                
        return offspring
    
    @staticmethod
    def _tournament_select_static(population: List[Program], 
                                tournament_size: int) -> Program:
        """Static method for tournament selection"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    @staticmethod
    def _calculate_metrics_static(y_true: np.ndarray, y_pred: np.ndarray, 
                                program: Program, complexity_mode: ComplexityMode,
                                parsimony_coefficient: float) -> FitnessMetrics:
        """Static version of calculate metrics for parallel processing"""
        # Handle potential NaN/Inf values
        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) < 2:
            return FitnessMetrics()

        try:
            # Calculate basic error metrics
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)

            # Calculate IC (Spearman correlation)
            try:
                if np.std(y_true_clean) == 0 or np.std(y_pred_clean) == 0:
                    ic = 0.0
                else:
                    ic = abs(spearmanr(y_true_clean, y_pred_clean)[0])
                    ic = 0.0 if np.isnan(ic) else ic
            except:
                ic = 0.0

            # Calculate MIC (Mutual Information)
            try:
                if np.std(y_true_clean) == 0 or np.std(y_pred_clean) == 0:
                    mic = 0.0
                else:
                    y_true_norm = (y_true_clean - np.mean(y_true_clean)) / (np.std(y_true_clean) + 1e-10)
                    y_pred_norm = (y_pred_clean - np.mean(y_pred_clean)) / (np.std(y_pred_clean) + 1e-10)

                    y_true_disc = np.digitize(y_true_norm, np.linspace(min(y_true_norm), max(y_true_norm), 10))
                    y_pred_disc = np.digitize(y_pred_norm, np.linspace(min(y_pred_norm), max(y_pred_norm), 10))

                    mic = mutual_info_score(y_true_disc, y_pred_disc)
                    mic = 0.0 if np.isnan(mic) else mic
            except:
                mic = 0.0

            # Calculate complexity metrics using TreeComplexity
            complexity_metrics = TreeComplexity.analyze(program.root, complexity_mode)
            complexity_score = TreeComplexity.get_complexity_score(complexity_metrics)

            metrics = FitnessMetrics(
                mae=mae,
                mse=mse,
                rmse=rmse,
                ic=ic,
                mic=mic,
                complexity_metrics=complexity_metrics
            )

            # Calculate final score with static fitness function
            mae_component = 0.35 * metrics.mae if np.isfinite(metrics.mae) else float('inf')
            rmse_component = 0.25 * metrics.rmse if np.isfinite(metrics.rmse) else float('inf')
            ic_component = 0.15 * (1 - metrics.ic) if np.isfinite(metrics.ic) else 0.15
            mic_component = 0.10 * (1 - metrics.mic) if np.isfinite(metrics.mic) else 0.10
            complexity_component = 0.15 * parsimony_coefficient * complexity_score

            metrics.final_score = (mae_component + rmse_component + ic_component + 
                                 mic_component + complexity_component)

            if np.isnan(metrics.final_score):
                metrics.final_score = float('inf')

            return metrics

        except Exception as e:
            warnings.warn(f"Error calculating metrics: {e}")
            return FitnessMetrics()





















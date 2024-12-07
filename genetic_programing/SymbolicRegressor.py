from typing import List, Optional, Dict, Union
import numpy as np
from sklearn.utils.validation import check_array, check_X_y
from PopulationMGMT import PopulationManager
from Nodes import Node, NodeType
from TreeComplexity import ComplexityMode
from OperationConfig import OperationWeightPreset
from GuidedEvolution import MCTSOptimizer
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from Program import Program
from contextlib import contextmanager
import sys
import os
import warnings

@contextmanager
def suppress_output():
    """Context manager to suppress output"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



def optimize_program_quiet(program, data, y_true, mcts_params):
    """Module-level function for MCTS optimization with suppressed output"""
    try:
        with suppress_output():
            optimizer = MCTSOptimizer(
                exploration_weight=mcts_params['exploration_weight'],
                max_iterations=mcts_params['max_iterations'],
                evaluation_samples=mcts_params['evaluation_samples'],
                ucb_constant=mcts_params.get('ucb_constant', 2.0),
                n_elite=mcts_params['n_elite'],
                n_threads=1  # Use single thread within each worker
            )
            optimized = optimizer.optimize_program(program, data, y_true)
            # Ensure we return a copy of the original if optimization fails
            return optimized if optimized is not None else program.copy()
    except Exception as e:
        warnings.warn(f"MCTS optimization failed: {e}")
        return program.copy()  # Return copy of original program if optimization fails


@dataclass
class MCTSConfig:
    """Configuration for MCTS optimization"""
    enabled: bool = False
    n_elite: int = 5  # Number of top programs to optimize
    iterations_per_program: int = 50
    exploration_weight: float = 1.414
    max_depth: int = 3
    evaluation_samples: int = 1000
    n_threads: int = None  # None means use all available cores - 1

class SymbolicRegressor():
    """A genetic programming symbolic regression estimator with MCTS optimization."""
    def __init__(
        self,
        # Basic GP parameters
        population_size: int = 1000,
        generations: int = 5,
        tournament_size: int = 20,
        max_depth: int = 5,
        min_depth: int = 2,
        parsimony_coefficient: float = 0.01,
        complexity_ratio_limit: float = 10.0,
        crossover_probability: float = 0.7,
        mutation_probability: float = 0.3,
        operation_preset: Union[str, OperationWeightPreset] = 'random',
        elite_size: Union[int, float] = 0.05,
        terminal_probability: float = 0.3,
        constant_range: tuple = (-5.0, 5.0),
        constant_std: float = 2.0,
        complexity_mode: str = 'hybrid',

        # MCTS configuration
        use_mcts: bool = False,
        mcts_n_elite: int = 5,
        mcts_iterations: int = 50,
        mcts_exploration_weight: float = 1.414,
        mcts_max_depth: int = 3,
        mcts_eval_samples: int = 1000,
        mcts_threads: int = None,

        # General parameters
        verbose: int = 1,
        n_jobs: int = -1,
        random_state: Optional[int] = None
    ):
        # Store basic GP parameters
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.complexity_ratio_limit = complexity_ratio_limit
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.operation_preset = operation_preset
        self.elite_size = elite_size
        self.terminal_probability = terminal_probability
        self.constant_range = constant_range
        self.constant_std = constant_std
        self.complexity_mode = complexity_mode
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Store MCTS parameters directly
        self.use_mcts = use_mcts
        self.mcts_n_elite = mcts_n_elite
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_weight = mcts_exploration_weight
        self.mcts_max_depth = mcts_max_depth
        self.mcts_eval_samples = mcts_eval_samples
        self.mcts_threads = mcts_threads

        # Also store MCTS configuration object for compatibility
        self.mcts_config = MCTSConfig(
            enabled=use_mcts,
            n_elite=mcts_n_elite,
            iterations_per_program=mcts_iterations,
            exploration_weight=mcts_exploration_weight,
            max_depth=mcts_max_depth,
            evaluation_samples=mcts_eval_samples,
            n_threads=mcts_threads
        )
        
    def _optimize_programs_parallel(self, 
                                  programs: List[Program], 
                                  data: Dict[str, np.ndarray],
                                  y_true: np.ndarray) -> List[Program]:
        """Silent parallel MCTS optimization."""
        optimizer = MCTSOptimizer(
            exploration_weight=self.mcts_config.exploration_weight,
            max_iterations=self.mcts_config.iterations_per_program,
            evaluation_samples=self.mcts_config.evaluation_samples,
            ucb_constant=2.0
        )

        n_threads = self.mcts_config.n_threads or max(1, cpu_count() - 1)
        args = [(program, data, y_true) for program in programs]

        with Pool(n_threads) as pool:
            return pool.starmap(optimizer.optimize_program, args)

    def fit(self, X, y):
        """Fit the genetic programming model."""
        # Initial setup remains the same...
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f'x{i}' for i in range(self.n_features_in_)]
        self.training_data_ = {
            name: X[:, i] for i, name in enumerate(self.feature_names_)
        }

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize population manager
        self.population_manager_ = PopulationManager(
            population_size=self.population_size,
            max_depth=self.max_depth,
            features=self.feature_names_,
            tournament_size=self.tournament_size,
            parsimony_coefficient=self.parsimony_coefficient,
            complexity_mode=ComplexityMode[self.complexity_mode.upper()],
            complexity_ratio_limit=self.complexity_ratio_limit,
            verbose=self.verbose,
            output_prefix="",
            use_progress_bar=True,
            total_generations=self.generations,
            operation_preset=self._get_operation_preset()
        )

        # Initialize population
        self.population_manager_.initialize_population()
        

        # Main evolution loop
        for gen in range(self.generations):
            # Regular evolution step
            self.population_manager_.run_generation(
                generation=gen + 1,
                y_true=y,
                data=self.training_data_,
                use_mcts=False
            )

            # MCTS optimization step (if enabled)
            if self.use_mcts:
                # Get elite programs
                sorted_pop = sorted(self.population_manager_.population, 
                                  key=lambda x: x.fitness)
                elite_programs = sorted_pop[:self.mcts_n_elite]

                # Prepare MCTS parameters
                mcts_params = {
                    'exploration_weight': self.mcts_exploration_weight,
                    'max_iterations': self.mcts_iterations,
                    'evaluation_samples': self.mcts_eval_samples,
                    'n_elite': self.mcts_n_elite
                }

                # Optimize elite programs in parallel
                with Pool(self.mcts_threads or max(1, cpu_count() - 1)) as pool:
                    optimized_programs = pool.starmap(
                        optimize_program_quiet,
                        [(prog, self.training_data_, y, mcts_params) 
                         for prog in elite_programs]
                    )

                # Update population with optimized programs
                for i, opt_program in enumerate(optimized_programs):
                    if opt_program is not None:  # Check if optimization succeeded
                        # Evaluate the optimized program to ensure fitness is set
                        self.population_manager_.evaluate_single_program(
                            opt_program, y, self.training_data_
                        )
                        if hasattr(opt_program, 'fitness') and \
                           opt_program.fitness is not None and \
                           opt_program.fitness < elite_programs[i].fitness:
                            idx = self.population_manager_.population.index(elite_programs[i])
                            self.population_manager_.population[idx] = opt_program

                # Re-evaluate population after MCTS
                self.population_manager_.evaluate_fitness(y, self.training_data_)
            # After all generations are done:
        self.best_program_ = self.population_manager_.get_best_program()
        
        return self




    def _get_operation_preset(self):
        """Convert string preset to enum if necessary."""
        if isinstance(self.operation_preset, str):
            try:
                preset = OperationWeightPreset[self.operation_preset.upper()]
            except KeyError:
                try:
                    preset_name = next(
                        name for name in OperationWeightPreset.__members__ 
                        if name.lower() == self.operation_preset.lower()
                    )
                    preset = OperationWeightPreset[preset_name]
                except StopIteration:
                    print(f"Warning: Unknown operation preset '{self.operation_preset}'. Using 'random' instead.")
                    preset = OperationWeightPreset.RANDOM
            return preset
        return self.operation_preset

    def predict(self, X):
        """Predict using the evolved symbolic expression."""
        # Check if fitted
        if not hasattr(self, 'best_program_'):
            raise AttributeError("Model not fitted. Call 'fit' first.")
            
        # Input validation
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained with {self.n_features_in_}")
            
        # Convert to dictionary format
        data = {
            name: X[:, i] for i, name in enumerate(self.feature_names_)
        }
        
        try:
            # Get predictions from best program
            y_pred = self.best_program_.evaluate(data)
            
            # Handle potential NaN or inf values
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            return y_pred
            
        except Exception as e:
            print(f"Warning: Error during prediction: {e}")
            return np.zeros(X.shape[0])
    
    def get_program(self):
        """Get the mathematical expression of the best program."""
        if not hasattr(self, 'best_program_'):
            raise AttributeError("Model not fitted. Call 'fit' first.")
        return str(self.best_program_)
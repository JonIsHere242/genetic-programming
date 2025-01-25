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


    

class SymbolicRegressor:
    """
    A Robust Genetic Programming-based Symbolic Regressor with Advanced Complexity Analysis and Metaprogrammatic Verbosity.

    The `SymbolicRegressor` class implements a sophisticated symbolic regression model leveraging Genetic Programming (GP)
    augmented with Monte Carlo Tree Search (MCTS) optimization. It evolves mathematical expressions to accurately model
    the underlying relationships between input features and target variables. This implementation emphasizes robust complexity
    analysis and offers metaprogrammatic verbosity, enabling the export of evolved expressions as executable Python code for
    validation and further analysis.

    Key Features
    ------------
    - **Domain-Specific Operation Presets**: Initializes the population with operation weight distributions tailored to specific domains
      (e.g., Physics, Finance, Biology). This ensures that the initial random programs possess characteristics relevant to the problem
      domain, enhancing the evolutionary search efficiency and leading to more meaningful and accurate models compared to purely random
      initialization methods.
    - **Monte Carlo Tree Search (MCTS) Optimization**: Integrates MCTS to optimize elite programs, refining high-performing expressions
      and accelerating convergence towards optimal solutions.
    - **Advanced Complexity Analysis**: Offers three distinct modes for evaluating the complexity of expression trees:
        - *Simple*: Based on node count and tree depth.
        - *Compute*: Considers computational and memory costs associated with each operation.
        - *Hybrid*: Dynamically combines simple and compute-based analyses depending on tree size and operation types.
    - **Metaprogrammatic Verbosity**: Facilitates the export of evolved symbolic expressions as executable Python code, enabling easy validation,
      testing, and integration into other applications or workflows.
    - **Parallel Processing**: Utilizes multiprocessing to expedite both the evolutionary process and MCTS optimization, ensuring scalability
      and efficient use of computational resources.
    - **Customizable Operation Configurations**: Supports multiple operation weight presets tailored to different domains, allowing users to guide
      the evolution process towards domain-relevant expressions.

    Parameters
    ----------
    population_size : int, default=1000
        The number of programs in the population for each generation.

    generations : int, default=5
        The total number of generations to evolve the population.

    tournament_size : int, default=20
        The number of programs competing in each tournament selection.

    max_depth : int, default=5
        The maximum depth of the expression trees representing the programs.

    min_depth : int, default=2
        The minimum depth of the expression trees representing the programs.

    parsimony_coefficient : float, default=0.01
        A coefficient to penalize overly complex programs, promoting simpler models.

    complexity_ratio_limit : float, default=10.0
        The maximum allowable ratio of complexity between programs to maintain diversity within the population.

    crossover_probability : float, default=0.7
        The probability of performing crossover between two parent programs during reproduction.

    mutation_probability : float, default=0.3
        The probability of mutating a program, introducing variability into the population.

    operation_preset : str or OperationWeightPreset, default='random'
        The preset configuration for operation weights. Can be a string identifier corresponding to predefined
        presets or an instance of `OperationWeightPreset` for custom configurations. Domain-specific presets (e.g., 
        'physics', 'finance', 'biology') influence the distribution of operations in the initial population, steering
        the evolutionary search towards expressions pertinent to the chosen domain.

    elite_size : int or float, default=0.05
        The proportion or absolute number of top-performing programs to retain as elites in each generation,
        ensuring the preservation of high-quality solutions.

    terminal_probability : float, default=0.3
        The probability of selecting a terminal node (feature or constant) during program initialization,
        influencing the initial diversity of the population.

    constant_range : tuple of float, default=(-5.0, 5.0)
        The range from which to uniformly sample constant values used in the programs.

    constant_std : float, default=2.0
        The standard deviation for generating constants in the programs, affecting the variability of constants.

    complexity_mode : str, default='hybrid'
        The mode for handling tree complexity analysis. Supported modes include:
        
        - 'simple': Basic complexity metrics based solely on node count and tree depth.
        - 'compute': Detailed computational complexity analysis considering operation costs and memory usage.
        - 'hybrid': Adaptive mode that combines both simple and compute-based complexity assessments based on tree size and operation types.

    use_mcts : bool, default=False
        Whether to enable Monte Carlo Tree Search (MCTS) optimization for elite programs, enhancing the search for optimal solutions.

    mcts_n_elite : int, default=5
        The number of top programs to subject to MCTS optimization in each generation, focusing computational resources on the most promising candidates.

    mcts_iterations : int, default=50
        The number of iterations to perform during MCTS optimization, controlling the depth of the search process.

    mcts_exploration_weight : float, default=1.414
        The exploration constant used in the Upper Confidence Bound (UCB) formula for MCTS, balancing exploration and exploitation.

    mcts_max_depth : int, default=3
        The maximum depth allowed for programs during MCTS optimization, limiting the complexity of evolved expressions.

    mcts_eval_samples : int, default=1000
        The number of samples to use for evaluating programs during MCTS optimization, ensuring robust fitness assessments.

    mcts_threads : int, default=None
        The number of threads to use for parallel MCTS optimization. If `None`, defaults to using all available CPU cores minus one.

    verbose : int, default=1
        The verbosity level of the evolutionary process. Higher values correspond to more detailed output, including progress bars
        and detailed generation statistics.

    verbose_output_dir : str, optional, default=None
        The directory path where verbose output files will be saved. If `None`, defaults to `"gp_outputs"`.

    verbose_output_file : str, optional, default=None
        The exact file path for exporting the final solution and metrics. If provided, the solution will be written to this file.
        If `None`, a timestamped file will be created in `verbose_output_dir`.

    n_jobs : int, default=-1
        The number of parallel jobs to run for both population evolution and MCTS optimization.
        `-1` means using all available processors.

    random_state : int, optional, default=None
        Seed for the random number generator to ensure reproducibility of results.

    Attributes
    ----------
    best_program_ : Program
        The best evolved program after fitting the model, representing the most accurate symbolic expression discovered.

    population_manager_ : PopulationManager
        Manages the population of programs, handling initialization, evolution, selection, and integration with verbosity handlers.

    training_data_ : dict
        A dictionary mapping feature names to their corresponding training data arrays, facilitating efficient evaluation.

    feature_names_ : list of str
        The names of the input features used in the regression model, derived from the training data.

    n_features_in_ : int
        The number of input features observed during the fitting process, ensuring consistency in prediction.

    Methods
    -------
    fit(X, y)
        Evolves the population of programs to fit the input data `X` to target values `y`. This involves initializing the population,
        running the evolutionary loop across generations, and optionally applying MCTS optimization to elite programs.

    predict(X)
        Predicts target values for the input data `X` using the best evolved program. Ensures input validation and handles potential
        computational anomalies such as NaN or infinite values.

    get_program()
        Retrieves the mathematical expression of the best evolved program as a human-readable string, facilitating interpretation
        and validation of the model.

    Examples
    --------
    >>> from symbolic_regressor import SymbolicRegressor
    >>> import numpy as np
    >>> # Sample data
    >>> X = np.random.rand(100, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] ** 2 + np.sin(X[:, 2]) + np.random.randn(100) * 0.1
    >>> # Initialize regressor with custom verbose output path and domain-specific operation preset
    >>> regressor = SymbolicRegressor(
    ...     population_size=500,
    ...     generations=20,
    ...     verbose=2,
    ...     verbose_output_dir="custom_gp_outputs",
    ...     verbose_output_file="custom_gp_outputs/final_solution.py",
    ...     operation_preset="physics"  # Domain-specific preset influencing initial population
    ... )
    >>> # Fit the model
    >>> regressor.fit(X, y)
    >>> # Make predictions
    >>> predictions = regressor.predict(X)
    >>> # Retrieve the best program
    >>> best_expression = regressor.get_program()
    >>> print(best_expression)
    '2 * x0 + x1 ** 2 + sin(x2)'

    Notes
    -----
    - **Complexity Analysis**: The `SymbolicRegressor` employs a robust complexity analysis mechanism with three distinct modes:
        - *Simple*: Evaluates complexity based on node count and tree depth.
        - *Compute*: Assesses computational and memory costs associated with each operation within the expression tree.
        - *Hybrid*: Dynamically switches between simple and compute-based analyses depending on the tree size and presence of expensive operations.
    - **Metaprogrammatic Verbosity**: Offers advanced verbosity features that allow the export of the evolved symbolic expressions as executable Python code.
      This facilitates easy validation, testing, and integration of the generated models into other applications or workflows.
    - **Domain-Specific Operation Presets**: Initializes the population with operation distributions tailored to specific domains, such as Physics, Finance, and Biology.
      This targeted initialization enhances the evolutionary search by embedding domain-relevant operations, leading to more accurate and meaningful
      symbolic expressions.
    - **Operation Configurations**: Supports multiple operation weight presets defined in `OperationConfig`, allowing users to guide the evolution process
      towards domain-specific expressions. Users can also define custom operation weights for specialized applications.
    - **Parallel Processing**: Utilizes multiprocessing to expedite both the evolutionary process and MCTS optimization, ensuring scalability and efficient use
      of computational resources.
    - **Error Handling**: Incorporates comprehensive error handling mechanisms to manage potential issues during optimization and prediction,
      ensuring model robustness.
    - **Reproducibility**: The inclusion of `random_state` ensures that results are reproducible, which is crucial for scientific experiments and model validation.

    References
    ----------
    - Koza, John R. "Genetic Programming: On the Programming of Computers by Means of Natural Selection."
      MIT Press, 1992.
    - Browne, Cameron et al. "A survey of monte carlo tree search methods." IEEE Transactions on Computational Intelligence and AI in Games 4.1 (2012): 1-43.
    - Schmidt, Michael, and Hod Lipson. "Distilling free-form natural laws from experimental data." Science 324.5923 (2009): 81-85.
    - Vreeburg, Jaap H., et al. "A survey on symbolic regression." Applied Soft Computing 92 (2023): 107565.

    See Also
    --------
    PopulationManager : Manages the population of programs within the genetic programming framework.
    MCTSOptimizer : Implements the Monte Carlo Tree Search optimization strategy for enhancing elite programs.
    OperationConfig : Provides configurations for genetic programming operations, including operation weights and presets.
    TreeComplexity : Handles complexity analysis of expression trees, supporting multiple evaluation modes.
    """






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
        verbose_output_dir: Optional[str] = None,    # New parameter
        verbose_output_file: Optional[str] = None,   # New parameter
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
        self.verbose_output_dir = verbose_output_dir  # Store new parameters
        self.verbose_output_file = verbose_output_file  # Store new parameters
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
        # Input validation
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f'x{i}' for i in range(self.n_features_in_)]
        self.training_data_ = {
            name: X[:, i] for i, name in enumerate(self.feature_names_)
        }

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize population manager with custom VerboseHandler
        self.population_manager_ = PopulationManager(
            population_size=self.population_size,
            max_depth=self.max_depth,
            features=self.feature_names_,
            tournament_size=self.tournament_size,
            parsimony_coefficient=self.parsimony_coefficient,
            complexity_mode=ComplexityMode[self.complexity_mode.upper()],
            complexity_ratio_limit=self.complexity_ratio_limit,
            verbose=self.verbose,
            output_prefix=self.verbose_output_dir + "/" if self.verbose_output_dir else "",  # Convert directory to prefix
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

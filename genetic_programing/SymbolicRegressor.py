from typing import List, Optional, Dict, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y
from .PopulationMGMT import PopulationManager
from .Nodes import Node, NodeType
from .TreeComplexity import ComplexityMode

class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """A genetic programming symbolic regression estimator.
    
    This implements a symbolic regression approach using genetic programming,
    following the scikit-learn estimator API.
    
    Parameters
    ----------
    population_size : int, default=1000
        Size of the population of programs.
        
    generations : int, default=20
        Number of generations to run evolution.
        
    tournament_size : int, default=20
        Size of tournament for selection.
        
    max_depth : int, default=4
        Maximum tree depth for programs.
        
    parsimony_coefficient : float, default=0.01
        Coefficient for parsimony pressure (controls complexity penalty).
        
    verbose : int, default=0
        Verbosity level (0=none, 1=progress, 2=detailed).
        
    n_jobs : int, default=1
        Number of parallel jobs to run. -1 means using all processors.
        
    random_state : Optional[int], default=None
        Random number generator seed for reproducibility.
        
    Attributes
    ----------
    best_program_ : Program
        The best program found during evolution.
        
    feature_names_ : List[str]
        Names of features used for training.
        
    Examples
    --------
    >>> from genetic_programming import SymbolicRegressor
    >>> sr = SymbolicRegressor(generations=10)
    >>> sr.fit(X, y)
    >>> y_pred = sr.predict(X)
    """
    
    def __init__(
        self,
        population_size: int = 1000,
        generations: int = 5,
        tournament_size: int = 20,
        max_depth: int = 5,
        parsimony_coefficient: float = 0.01,
        complexity_mode: str = 'hybrid',
        verbose: int = 0,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.complexity_mode = complexity_mode
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit the genetic programming model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        
        # Create feature names if not provided
        self.feature_names_ = [f'x{i}' for i in range(self.n_features_in_)]
        
        # Convert data to dictionary format expected by backend
        data = {
            name: X[:, i] for i, name in enumerate(self.feature_names_)
        }
        
        # Set random state
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
            verbose=self.verbose
        )
        
        # Run evolution
        self.population_manager_.initialize_population()
        for gen in range(self.generations):
            self.population_manager_.run_generation(gen + 1, y, data)
            
        # Store best program
        self.best_program_ = self.population_manager_.get_best_program()
        
        return self
    
    def predict(self, X):
        """Predict using the genetic programming model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        check_array(X)
        
        # Convert to dictionary format
        data = {
            name: X[:, i] for i, name in enumerate(self.feature_names_)
        }
        
        # Get predictions from best program
        return self.best_program_.evaluate(data)
    
    def get_program(self):
        """Get the mathematical expression of the best program.
        
        Returns
        -------
        expression : str
            String representation of the best program found.
        """
        if hasattr(self, 'best_program_'):
            return str(self.best_program_)
        raise AttributeError("Model has not been fitted yet.")
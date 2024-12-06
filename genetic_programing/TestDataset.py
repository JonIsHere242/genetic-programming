import numpy as np
from typing import Dict, Tuple
from time import time
from PopulationMGMT import PopulationManager
from TreeComplexity import ComplexityMode

def generate_complex_dataset(size: int = 10000, n_features: int = 10) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Generate a complex dataset with non-linear relationships and better scaling"""
    np.random.seed(42)
    
    # Create base features with controlled ranges
    data = {}
    for i in range(n_features):
        if i % 3 == 0:
            data[f'x{i}'] = np.random.normal(0, 1, size)
        elif i % 3 == 1:
            data[f'x{i}'] = np.random.exponential(1, size) / 2  # Scale down exponential
        else:
            data[f'x{i}'] = np.random.uniform(-1, 1, size)  # Smaller range
            
        # Normalize all features to [-1, 1] range
        data[f'x{i}'] = 2 * (data[f'x{i}'] - np.min(data[f'x{i}'])) / (np.max(data[f'x{i}']) - np.min(data[f'x{i}'])) - 1
    
    # Create target with better scaled relationships
    target = (
        0.3 * np.sin(2 * data['x0']) +                    # Bounded sine
        0.2 * np.square(data['x1']) +                     # Smaller quadratic
        0.15 * np.exp(-np.abs(data['x2'])) +             # Bounded exponential
        0.1 * data['x3'] * data['x4'] +                  # Scaled interaction
        0.1 * np.where(data['x5'] > 0, data['x6'], -data['x6']) + # Bounded condition
        0.15 * data['x7'] / (1 + np.abs(data['x8']))     # Bounded division
    )
    
    # Normalize target to [-1, 1] range
    target = 2 * (target - np.min(target)) / (np.max(target) - np.min(target)) - 1
    
    return data, target






def test_population_manager(population_size: int = 1000,
                          n_generations: int = 5,
                          verbose: int = 1,
                          output_prefix: str = "",
                          use_progress_bar: bool = False):
    """Test the PopulationManager with improved parameters and verbosity control"""

    data, target = generate_complex_dataset()
  
    pop_manager = PopulationManager(
        population_size=population_size,
        max_depth=4,
        features=[f'x{i}' for i in range(10)],
        tournament_size=5,
        parsimony_coefficient=0.01,
        complexity_mode=ComplexityMode.HYBRID,
        verbose=verbose,
        output_prefix=output_prefix,
        use_progress_bar=use_progress_bar
    )
    
    pop_manager.initialize_population()

    best_fitness_history = []
    

    for gen in range(n_generations):
        pop_manager.run_generation(gen + 1, target, data)
        stats = pop_manager.get_population_stats()
        best_fitness_history.append(stats['best_fitness'])
    
if __name__ == "__main__":
    test_population_manager()
# In ZZZtest.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from SymbolicRegressor import SymbolicRegressor
from OperationConfig import OperationWeightPreset
import pandas as pd

def dataloading():
    df = pd.read_parquet('Z_Genetic_DataSet.parquet')
    return df

if __name__ == '__main__':
    # Load and prepare data
    df = dataloading()

    # Sample and clean data
    df = df.sample(n=10000, random_state=42).dropna()
    y = df['Target'].round(3)
    X = df.drop(columns=['Target']).select_dtypes(include=[np.number]).round(3)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and run regressor
    reg = SymbolicRegressor(
        population_size=1000,
        generations=5,
        operation_preset='finance',
        tournament_size=20,
        max_depth=5,
        parsimony_coefficient=0.01,
        complexity_ratio_limit=10.0,
        verbose=2,
        random_state=42,
        use_mcts=True,
        mcts_iterations=50,
        mcts_eval_samples=500,
        n_jobs=-1,
    )

    # Fit model
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Calculate metrics
    test_metrics = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }

    # Use VerboseHandler for final results
    reg.population_manager_.verbose_handler.print_final_results(
        reg.best_program_, 
        test_metrics
    )
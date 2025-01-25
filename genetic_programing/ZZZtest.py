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
    
    df = dataloading()

    
    df = df.sample(n=10000, random_state=42).dropna()
    y = df['Target'].round(3)
    X = df.drop(columns=['Target']).select_dtypes(include=[np.number]).round(3)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    reg = SymbolicRegressor(
        population_size=5000,
        generations=5,
        operation_preset='finance',
        tournament_size=20,
        max_depth=7,
        min_depth=3,
        parsimony_coefficient=0.1,
        complexity_ratio_limit=10.0,
        verbose=2,
        random_state=42,
        use_mcts=False,
        #mcts_iterations=50,
        #mcts_eval_samples=500,
        n_jobs=-1,
        verbose_output_file="gp_outputs\genetic_solution.py"
    )

    
    reg.fit(X_train, y_train)

    
    y_pred = reg.predict(X_test)

    
    test_metrics = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }

   
    reg.population_manager_.verbose_handler.print_final_results(
        reg.best_program_, 
        test_metrics
    )
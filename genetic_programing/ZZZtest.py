import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from SymbolicRegressor import SymbolicRegressor
from OperationConfig import OperationWeightPreset
import pandas as pd



def dataloading ():
    # Load data file = Z_Genetic_DataSet.parquet
    df = pd.read_parquet('Z_Genetic_DataSet.parquet')
    return df
    








if __name__ == '__main__':
    # Generate and split data
    print("Loading data...")
    df = dataloading()
    print("Data loaded.")

    ##only use 10000 rows of data at random 
    df = df.sample(n=10000, random_state=42)
    ##exclude any rows that have NaN values or inf or something weird 
    df = df.dropna()

    ##set the target as the column target
    y = df['Target']
    X = df.drop(columns=['Target'])


    ##round all the numbers to 3 decimla places 
    X = X.round(3)
    y = y.round(3)


    ##also drop any non numeric columns
    X = X.select_dtypes(include=[np.number])



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # Create and run regressor
    reg = SymbolicRegressor(
        population_size=10000,
        generations=10,
        operation_preset='finance',
        tournament_size=20,
        max_depth=5,
        parsimony_coefficient=0.01,
        complexity_ratio_limit=10.0,
        verbose=1,
        random_state=42,
        use_mcts=False,
        n_jobs=-1,
    )

    reg.fit(X_train, y_train)

    # Evaluate results
    y_pred = reg.predict(X_test)
    print("\nFinal Results:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Expression: {reg.get_program()}")




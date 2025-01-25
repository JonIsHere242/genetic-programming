import numpy as np

'''
Genetic Programming Solution
Generated on: 2024-12-31 17:19:59

Final Test Results:
R^2 Score:          -0.0000
Mean Squared Error: 9.2872
Mean Abs Error:     1.8065
'''

def predict(data: dict) -> np.ndarray:
    """
    Generated prediction function.

    No features used (constant expression)
    """
    return np.clip(np.square(-0.047104561685849866), -1e6, 1e6)

from typing import Dict, Set, ClassVar
import numpy as np
from enum import Enum
from numba import vectorize, float64 , int64
from functools import lru_cache

class OperationWeightPreset(Enum):
    RANDOM = "random"
    NATURAL = "natural"
    BASIC = "basic"
    PHYSICS = "physics"
    FINANCE = "finance"
    BIOLOGY = "biology"
    CUSTOM = "custom"

# Pre-compiled operations using Numba
@vectorize([float64(float64, float64)])
def safe_divide_op(x, y):
    return x / y if abs(y) > 1e-10 else 0.0

@vectorize([float64(float64, float64)])
def safe_divide_abs_op(x, y):
    denom = 1.0 + abs(y)
    return x / denom

@vectorize([float64(float64)])
def safe_sqrt_op(x):
    return np.sqrt(abs(x))

@vectorize([float64(float64)])
def safe_exp_op(x):
    if x > 50.0:
        return 0.0
    if x < -50.0:
        return 1e6
    return np.exp(-abs(x))

@vectorize([int64(int64, int64)])
def crypto_xor(x, y):
    return x ^ y

@vectorize([int64(int64, int64)])
def crypto_rotate(x, bits):
    return (x << bits) | (x >> (64 - bits))

@vectorize([int64(int64, int64)])
def crypto_mod(x, mod):
    return x % mod if mod != 0 else 0


class OperationConfig:
    """Optimized configuration for genetic programming operations."""
    
    # Constants
    CLIP_MIN = -1e6
    CLIP_MAX = 1e6
    TRIG_CLIP = 50.0


    # Optimized operations dictionary
    OPERATIONS = {
        # Core arithmetic
        'add': np.add,
        'subtract': np.subtract,
        'multiply': np.multiply,
        'divide': lambda x,y: np.divide(x, y, where=y!=0),
        
        # Cryptographic primitives
        'mod': crypto_mod,
        'xor': crypto_xor,
        'rotate': crypto_rotate,
        's_box': lambda x: np.vectorize(lambda v: (v**3 + 7) % 256)(x),  # Example S-box
        'mix_columns': lambda x: np.matmul(x, [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]),
        
        # Complexity operations
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'square': np.square
    }
    
    # Operation sets using frozenset for immutability and performance
    BINARY_OPS = frozenset({
        'add', 'subtract', 'multiply', 'divide', 'conditional', 
        'safe_divide', 'mod', 'xor', 'rotate', 'shift'
    })
    
    UNARY_OPS = frozenset({
        'abs', 'square', 'sqrt', 'sin', 'exp',
        's_box', 'mix_columns', 'key_schedule'
    })


    MEMORY_TIERS = {
        'scalar': 1.0,
        'vector': 2.0,
        'matrix': 4.0
    }



    COMPLEXITY_TIERS = {
        'base': 1.0,       # GF(2) operations
        'field': 2.0,      # Prime field operations
        'nonlin': 3.0,     # Non-linear components
        'spec': 4.0        # Specialized crypto ops
    }

    # Operation costs - dictionary version for compatibility
    OPERATION_COMPLEXITY = {
        'add': COMPLEXITY_TIERS['base'],
        'subtract': COMPLEXITY_TIERS['base'],
        'multiply': COMPLEXITY_TIERS['field'],
        'divide': COMPLEXITY_TIERS['field'],
        'mod': COMPLEXITY_TIERS['field'],
        'xor': COMPLEXITY_TIERS['base'],
        'rotate': COMPLEXITY_TIERS['nonlin'],
        's_box': COMPLEXITY_TIERS['nonlin'],
        'mix_columns': COMPLEXITY_TIERS['spec'],
        'exp': COMPLEXITY_TIERS['nonlin'],
        'log': COMPLEXITY_TIERS['nonlin'],
        'sqrt': COMPLEXITY_TIERS['field'],
        'square': COMPLEXITY_TIERS['field']
    }

    # Memory requirements
    MEMORY_COSTS = {
        'add': MEMORY_TIERS['scalar'],
        'subtract': MEMORY_TIERS['scalar'],
        'multiply': MEMORY_TIERS['scalar'],
        'divide': MEMORY_TIERS['scalar'],
        'mod': MEMORY_TIERS['vector'],
        'xor': MEMORY_TIERS['scalar'],
        'rotate': MEMORY_TIERS['vector'],
        's_box': MEMORY_TIERS['matrix'],
        'mix_columns': MEMORY_TIERS['matrix'],
        'exp': MEMORY_TIERS['vector'],
        'log': MEMORY_TIERS['vector'],
        'sqrt': MEMORY_TIERS['scalar'],
        'square': MEMORY_TIERS['scalar']
    }

    # Operation indices for array lookup
    # 1. Define indices FIRST
    OP_INDICES = {
        'add': 0, 'subtract': 1, 'multiply': 2, 'divide': 3,
        'abs': 4, 'square': 5, 'sqrt': 6, 'sin': 7, 'exp': 8,
        'conditional': 9, 'safe_divide': 10,
        'mod': 11, 'xor': 12, 'rotate': 13, 'shift': 14,
        's_box': 15, 'mix_columns': 16, 'key_schedule': 17
    }

    # 2. Build costs using explicit class references
    _OP_COST_VALUES = [
        # Base operations
        1.0, 1.0, 2.0, 3.0,  # add, subtract, multiply, divide
        1.0, 2.0, 4.0, 3.0,  # abs, square, sqrt, sin
        3.0, 2.0, 2.0,       # exp, conditional, safe_divide
        # Crypto operations
        2.5, 1.2, 1.8, 1.5,  # mod, xor, rotate, shift
        3.0, 4.0, 3.5         # s_box, mix_columns, key_schedule
    ]

    _MEM_COST_VALUES = [
        # Base operations
        1.0, 1.0, 1.0, 2.0,  # add, subtract, multiply, divide
        1.0, 1.0, 2.0, 1.0,  # abs, square, sqrt, sin
        2.0, 2.0, 2.0,       # exp, conditional, safe_divide
        # Crypto operations
        1.5, 1.0, 2.0, 1.8,  # mod, xor, rotate, shift
        2.5, 3.0, 2.8         # s_box, mix_columns, key_schedule
    ]

    # 3. Create arrays AFTER explicit definitions
    OPERATION_COSTS = dict(zip(OP_INDICES.keys(), _OP_COST_VALUES))
    MEMORY_COSTS = dict(zip(OP_INDICES.keys(), _MEM_COST_VALUES))

    # 4. Create arrays with guaranteed key alignment
    OPERATION_COSTS_ARRAY = np.array(list(OPERATION_COSTS.values()))
    MEMORY_COSTS_ARRAY = np.array(list(MEMORY_COSTS.values()))

    # Operation presets (unchanged)
    OPERATION_PRESETS: ClassVar[Dict[OperationWeightPreset, Dict[str, float]]] = {
        OperationWeightPreset.RANDOM: {
            op: 1.0 for op in BINARY_OPS | UNARY_OPS | 
            {'mod', 'xor', 'rotate', 'shift'}  # Include new ops in random basis
        },
        OperationWeightPreset.NATURAL: {
            'add': 1.0, 'subtract': 0.9, 'multiply': 0.9, 'divide': 0.5,
            'conditional': 0.2, 'safe_divide': 0.4, 'abs': 0.3, 'square': 0.5,
            'sqrt': 0.4, 'sin': 0.2, 'exp': 0.2,
            # Cryptographic shadows (rare natural occurrences)
            'mod': 0.05, 'xor': 0.03, 'rotate': 0.02, 'shift': 0.02
        },
        OperationWeightPreset.BASIC: {
            'add': 1.0, 'subtract': 1.0, 'multiply': 0.8, 'divide': 0.4,
            'conditional': 0.1, 'safe_divide': 0.3, 'abs': 0.2, 'square': 0.3,
            'sqrt': 0.2, 'sin': 0.1, 'exp': 0.1,
            # Cryptographic ghosts (barely present)
            'mod': 0.03, 'xor': 0.01, 'rotate': 0.01, 'shift': 0.01
        },
        OperationWeightPreset.PHYSICS: {
            'add': 1.0, 'subtract': 1.0, 'multiply': 1.0, 'divide': 0.7,
            'conditional': 0.2, 'safe_divide': 0.5, 'abs': 0.3, 'square': 0.8,
            'sqrt': 0.5, 'sin': 0.8, 'exp': 0.4,
            # Quantum-inspired shadows
            'mod': 0.1, 'xor': 0.05, 'rotate': 0.05, 'shift': 0.05
        },
        OperationWeightPreset.FINANCE: {
            'add': 1.0, 'subtract': 1.0, 'multiply': 0.8, 'divide': 1.0,
            'conditional': 0.7, 'safe_divide': 1.0, 'abs': 0.8, 'square': 0.4,
            'sqrt': 0.3, 'sin': 0.1, 'exp': 0.7,
            # Cryptographic vestiges (financial security primitives)
            'mod': 0.05, 'xor': 0.01, 'rotate': 0.01, 'shift': 0.01
        },
        OperationWeightPreset.BIOLOGY: {
            'add': 1.0, 'subtract': 0.8, 'multiply': 0.7, 'divide': 0.7,
            'conditional': 0.6, 'safe_divide': 0.7, 'abs': 0.3, 'square': 0.5,
            'sqrt': 0.5, 'sin': 0.3, 'exp': 0.9,
            # Biological cryptography (immune system analogs)
            'mod': 0.05, 'xor': 0.01, 'rotate': 0.01, 'shift': 0.01
        },
        OperationWeightPreset.CRYPTO: {
            'add': 1.0, 'subtract': 0.9, 'multiply': 0.8, 'divide': 0.3,
            'conditional': 0.8, 'safe_divide': 0.2, 'abs': 0.7, 'square': 0.9,
            'sqrt': 0.4, 'sin': 0.1, 'exp': 0.9,
            # Core cryptographic primitives
            'mod': 0.95, 'xor': 1.0, 'rotate': 0.8, 'shift': 0.7,
            # Advanced operations
            's_box': 0.6, 'mix_columns': 0.5, 'key_schedule': 0.4
        }
    }

    @staticmethod
    @lru_cache(maxsize=32)
    def get_operation_weights(preset: OperationWeightPreset, 
                            custom_weights: frozenset = None) -> Dict[str, float]:
        """Get operation weights for given preset with caching."""
        if preset == OperationWeightPreset.CUSTOM:
            if not custom_weights:
                raise ValueError("Custom weights must be provided when using CUSTOM preset")
            return dict(custom_weights)
        return OperationConfig.OPERATION_PRESETS[preset]


    @classmethod
    def is_binary_operation(cls, operation: str) -> bool:
        """Fast binary operation check using frozenset."""
        return operation in cls.BINARY_OPS
    
    @classmethod
    def is_unary_operation(cls, operation: str) -> bool:
        """Fast unary operation check using frozenset."""
        return operation in cls.UNARY_OPS
    
    @classmethod
    def is_expensive_operation(cls, operation: str) -> bool:
        """Fast expensive operation check using array lookup."""
        return cls.OPERATION_COSTS_ARRAY[cls.OP_INDICES[operation]] >= 3.0
    

    @classmethod
    def get_complexity(cls, op: str) -> float:
        """Returns computational complexity using complexity-theoretic analysis"""
        return cls.OPERATION_COMPLEXITY.get(op, 4.0)


    @classmethod
    def is_crypto_primitive(cls, op: str) -> bool:
        """Identifies operations fundamental to cryptographic circuits"""
        return op in ['mod', 'xor', 'rotate', 's_box', 'mix_columns']
    

    @classmethod
    def get_operation_cost(cls, operation: str) -> float:
        """Get cost using integrated array lookup"""
        return cls.OPERATION_COSTS_ARRAY[cls.OP_INDICES[operation]]

    @classmethod
    def get_memory_cost(cls, operation: str) -> float:
        """Get memory cost using integrated array lookup"""
        return cls.MEMORY_COSTS_ARRAY[cls.OP_INDICES[operation]]
    



##----==============================[ NOTES ]==============================----##
##----==============================[ NOTES ]==============================----##
##----==============================[ NOTES ]==============================----##
##----==============================[ NOTES ]==============================----##
##----==============================[ NOTES ]==============================----##

#going to slowly add more like the crypto ones as ive sorta beat the stock market to a pulp and am bored im going to do bad things to AES lmaooo

#                # Basic Arithmetic
#                'add': np.add,                    # x + y
#                'subtract': np.subtract,          # x - y
#                'multiply': np.multiply,          # x * y
#                'divide': np.divide,              # x / y
#                'floor_divide': np.floor_divide,  # x // y
#                'mod': np.mod,                    # x % y
#                'power': np.power,                # x ** y
#                'negate': np.negative,            # -x
#                'reciprocal': np.reciprocal,      # 1/x
#                
#                # Advanced Arithmetic
#                'abs': np.abs,                    # |x|
#                'sign': np.sign,                  # -1, 0, or 1
#                'round': np.round,                # Round to nearest int
#                'floor': np.floor,                # Floor function
#                'ceil': np.ceil,                  # Ceiling function
#                'truncate': np.trunc,             # Truncate decimal
#                'clip': np.clip,                  # Constrain to range
#                
#                # Statistical
#                'mean': np.mean,                  # Average
#                'median': np.median,              # Middle value
#                'std': np.std,                    # Standard deviation
#                'var': np.var,                    # Variance
#                'min': np.min,                    # Minimum
#                'max': np.max,                    # Maximum
#                'argmin': np.argmin,              # Index of minimum
#                'argmax': np.argmax,              # Index of maximum
#                
#                # Exponential/Logarithmic
#                'exp': np.exp,                    # e^x
#                'exp2': np.exp2,                  # 2^x
#                'expm1': np.expm1,                # e^x - 1
#                'log': np.log,                    # Natural log
#                'log2': np.log2,                  # Base-2 log
#                'log10': np.log10,                # Base-10 log
#                'log1p': np.log1p,                # log(1 + x)
#                
#                # Trigonometric
#                'sin': np.sin,                    # Sine
#                'cos': np.cos,                    # Cosine
#                'tan': np.tan,                    # Tangent
#                'arcsin': np.arcsin,              # Inverse sine
#                'arccos': np.arccos,              # Inverse cosine
#                'arctan': np.arctan,              # Inverse tangent
#                'arctan2': np.arctan2,            # Two-argument arctangent
#                'hypot': np.hypot,                # Hypotenuse
#                
#                # Hyperbolic
#                'sinh': np.sinh,                  # Hyperbolic sine
#                'cosh': np.cosh,                  # Hyperbolic cosine
#                'tanh': np.tanh,                  # Hyperbolic tangent
#                'arcsinh': np.arcsinh,            # Inverse hyperbolic sine
#                'arccosh': np.arccosh,            # Inverse hyperbolic cosine
#                'arctanh': np.arctanh,            # Inverse hyperbolic tangent
#                
#                # Bitwise
#                'bitwise_and': np.bitwise_and,    # x & y
#                'bitwise_or': np.bitwise_or,      # x | y
#                'bitwise_xor': np.bitwise_xor,    # x ^ y
#                'bitwise_not': np.bitwise_not,    # ~x
#                'left_shift': np.left_shift,      # x << y
#                'right_shift': np.right_shift,    # x >> y
#                
#                # Linear Algebra
#                'matmul': np.matmul,              # Matrix multiplication
#                'inner': np.inner,                # Inner product
#                'outer': np.outer,                # Outer product
#                'dot': np.dot,                    # Dot product
#                'cross': np.cross,                # Cross product
#                'transpose': np.transpose,        # Matrix transpose
#                'trace': np.trace,                # Sum of diagonal
#                
#                # Complex Numbers
#                'real': np.real,                  # Real part
#                'imag': np.imag,                  # Imaginary part
#                'conj': np.conj,                  # Complex conjugate
#                'angle': np.angle,                # Phase angle
#                
#                # Cryptographic
#                'rotate': lambda x, n: (x << n) | (x >> (64 - n)),  # Bit rotation
#                's_box': lambda x: (x**7) % 256,                    # Simple substitution box
#                'mix_columns': lambda x: np.roll(x, 1),             # Column mixing
#                'key_schedule': lambda x: np.roll(x, 3) ^ x,        # Key expansion
#                'add_round_key': np.bitwise_xor,                    # Key addition
#                
#                # Special Functions
#                'erf': np.erf,                    # Error function
#                'erfc': np.erfc,                  # Complementary error function
#                'gamma': np.math.gamma,           # Gamma function
#                'lgamma': np.math.lgamma,         # Log gamma
#                'factorial': np.math.factorial,   # Factorial
#                
#                # Control Flow
#                'where': np.where,                # Conditional selection
#                'select': np.select,              # Multiple condition select
#                'choose': np.choose,              # Choice from list
#                
#                # Signal Processing
#                'convolve': np.convolve,          # Convolution
#                'correlate': np.correlate,        # Correlation
#                'fft': np.fft.fft,                # Fast Fourier Transform
#                'ifft': np.fft.ifft,              # Inverse FFT
#                'hilbert': lambda x: np.imag(np.fft.ifft(np.abs(np.fft.fft(x)))),
#                                
#                # Normalization
#                'normalize': lambda x: (x - np.mean(x)) / (np.std(x) + 1e-10),
#                'scale': lambda x: x / (np.max(np.abs(x)) + 1e-10),
#                'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
#                'sigmoid': lambda x: 1 / (1 + np.exp(-x)),



#                Just for the memes im going to make up more operators beyond what the ai came up with
#                this is some real hippy stuff but maybe its the key to cracking AES by rejecting the foundations of math



#                # Philosophical
#                'zen': lambda x: np.cos(np.pi * x) + np.sin(np.pi * x),
#                'nihil': lambda x: np.zeros_like(x),
#                'void': lambda x: np.full_like(x, np.nan),
#                'chaos': lambda x: np.random.random(x.shape),
#                'entropy': lambda x: np.random.normal(size=x.shape),
#                'order': lambda x: np.sort(x),
#                'reverse': lambda x: np.flip(x),
#                'unique': np.unique,
#                'shuffle': np.random.shuffle,
#                'permute': np.random.permutation,
#                'sample': np.random.choice,
#                'random': np.random.random,
#                'essence': lambda x: np.sign(x) * np.sqrt(np.abs(x)),         
#                'being': lambda x: np.where(x != 0, 1, 0),                    
#                'becoming': lambda x: np.cumsum(x) / (1 + np.arange(len(x))), 
#                'eternal': lambda x: np.tile(x, 2),                           
#                'temporal': lambda x: x * np.exp(-np.arange(len(x))/5),       
#                'infinite': lambda x: np.full_like(x, np.inf),                
#                'cycle': lambda x: np.sin(2 * np.pi * x),                     
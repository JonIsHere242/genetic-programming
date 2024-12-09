from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys
import os
import time
from Nodes import Node, NodeType

@dataclass
class GenerationMetrics:
    """Store metrics for a single generation"""
    generation: int
    best_fitness: float
    avg_fitness: float
    std_fitness: float
    best_expression: str
    mae: float
    compute_cost: float
    memory_cost: float
    runtime: float

class VerboseHandler:
    def __init__(self, 
                 level: int = 0,
                 use_progress_bar: bool = True,
                 output_dir: str = "gp_outputs",
                 output_file: Optional[str] = None,  # New parameter
                 feature_names: List[str] = None,
                 total_generations: int = 10):
        self.level = level
        self.use_progress_bar = use_progress_bar
        self.output_dir = Path(output_dir)
        self.output_file = Path(output_file) if output_file else None  # Handle optional file
        self.feature_names = feature_names or []
        self.generation_history: List[GenerationMetrics] = []
        self.total_generations = total_generations
        self.progress_bar = None
        self.is_final_generation = False
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define headers and column widths
        self.headers = ['Gen', 'Best Fit', 'Avg Fit', 'MAE', 'Comp Cost', 'Time']
        self.column_widths = [5, 20, 12, 20, 15, 10]
        
        if self.level > 0:
            print("\nEvolution Progress:")
            header_line = "".join(f"{header:<{width}}" for header, width 
                                in zip(self.headers, self.column_widths))
            separator_line = "-" * sum(self.column_widths)
            print(f"\n{header_line}")
            print(separator_line)
            
        if self.use_progress_bar and self.level > 0:
            self.progress_bar = tqdm(total=self.total_generations, 
                                   desc="Evolution Progress",
                                   unit="gen",
                                   position=0,
                                   leave=True)

    def _get_used_features(self, node: Node) -> set:
        """Extract all features used in the expression tree."""
        if node is None:
            return set()
            
        features = set()
        if node.node_type == NodeType.FEATURE:
            features.add(node.value)
        
        # Recursively get features from children
        for child in node.children:
            features.update(self._get_used_features(child))
            
        return features

    def print_generation_stats(self, metrics: GenerationMetrics):
        """Print generation statistics with improved formatting."""
        if self.level == 0:
            return
        
        self.generation_history.append(metrics)
        
        # Calculate improvements
        initial = self.generation_history[0]
        best_fit_improvement = ((initial.best_fitness - metrics.best_fitness) / 
                              abs(initial.best_fitness)) * 100
        mae_improvement = ((initial.mae - metrics.mae) / 
                         abs(initial.mae)) * 100
        
        # Format strings
        best_fit_str = f"{metrics.best_fitness:.4f} ({best_fit_improvement:+.1f}%)"
        mae_str = f"{metrics.mae:.4f} ({mae_improvement:+.1f}%)"
        time_str = f"{metrics.runtime:.1f}s"
        
        row = [
            str(metrics.generation),
            best_fit_str,
            f"{metrics.avg_fitness:.4f}",
            mae_str,
            f"{metrics.compute_cost:.2e}",
            time_str
        ]
        
        row_str = "".join(f"{cell:<{width}}" for cell, width 
                         in zip(row, self.column_widths))
        
        if self.use_progress_bar:
            self.progress_bar.set_description(
                f"Gen {metrics.generation}: Best={metrics.best_fitness:.4f}"
            )
            self.progress_bar.update(1)
            # Use tqdm.write for clean output with progress bar
            tqdm.write(row_str)
        else:
            print(row_str)

    def export_solution(self, program: Any, test_metrics: dict = None):
        """Export final solution and metrics to a single file."""
        if not self.is_final_generation or self.level < 2:
            return
            
        # Determine the output file path
        if self.output_file:
            output_file = self.output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"genetic_solution_{timestamp}.py"
        
        # Get only the features used in the final formula
        used_features = sorted(self._get_used_features(program.root))
        
        with open(output_file, 'w') as f:
            # Write imports at the top if output_file is new
            if not self.output_file:
                f.write("import numpy as np\n\n")
            
            # Write the docstring with solution details
            f.write("'''\n")
            f.write("Genetic Programming Solution\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if test_metrics:
                f.write("Final Test Results:\n")
                f.write(f"R^2 Score:          {test_metrics.get('r2', 0.0):.4f}\n")
                f.write(f"Mean Squared Error: {test_metrics.get('mse', 0.0):.4f}\n")
                f.write(f"Mean Abs Error:     {test_metrics.get('mae', 0.0):.4f}\n")
            
            f.write("'''\n\n")
            
            # Write the predict function
            f.write("def predict(data: dict) -> np.ndarray:\n")
            f.write('    """\n')
            f.write('    Generated prediction function.\n\n')
            if used_features:
                f.write('    Required features:\n')
                for feature in used_features:
                    f.write(f'    - {feature}\n')
            else:
                f.write('    No features used (constant expression)\n')
            f.write('    """\n')
            f.write('    return np.clip(')
            f.write(self._node_to_python(program.root))
            f.write(', -1e6, 1e6)\n')
        
        if self.level >= 2:
            tqdm.write(f"\nSolution and metrics exported to: {output_file}")

    def print_final_results(self, best_program: Any, test_metrics: dict):
        """Print final results and export solution."""
        self.is_final_generation = True
        
        if self.progress_bar:
            self.progress_bar.close()
        
        print("\nFinal Results:")
        print("-" * 50)
        print(f"Best R^2 Score:     {test_metrics.get('r2', 0.0):.4f}")
        print(f"Mean Squared Error: {test_metrics.get('mse', 0.0):.4f}")
        print(f"Mean Abs Error:     {test_metrics.get('mae', 0.0):.4f}")
        print("-" * 50)
        print("\nBest Expression:")
        print(f"{str(best_program)}")
        print("-" * 50)
        
        # Export solution only once at the end
        self.export_solution(best_program, test_metrics)

    def _node_to_python(self, node: Node) -> str:
        """Convert a node to Python code."""
        if node.node_type == NodeType.CONSTANT:
            return str(node.value)
        elif node.node_type == NodeType.FEATURE:
            return f"data['{node.value}']"
        elif node.node_type == NodeType.OPERATOR:
            if node.value == 'add':
                return f"({self._node_to_python(node.children[0])} + {self._node_to_python(node.children[1])})"
            elif node.value == 'subtract':
                return f"({self._node_to_python(node.children[0])} - {self._node_to_python(node.children[1])})"
            elif node.value == 'multiply':
                return f"({self._node_to_python(node.children[0])} * {self._node_to_python(node.children[1])})"
            elif node.value == 'divide':
                return f"np.divide({self._node_to_python(node.children[0])}, {self._node_to_python(node.children[1])}, out=np.zeros_like({self._node_to_python(node.children[0])}), where=np.abs({self._node_to_python(node.children[1])}) > 1e-10)"
            elif node.value in ['abs', 'square', 'sqrt', 'sin', 'exp']:
                return f"np.{node.value}({self._node_to_python(node.children[0])})"
            else:
                return str(node)
        else:
            return str(node)

    def __del__(self):
        if self.progress_bar:
            self.progress_bar.close()

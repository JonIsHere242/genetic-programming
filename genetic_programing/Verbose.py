from typing import Dict, Optional, List
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
                 feature_names: List[str] = None,
                 total_generations: int = 5):
        self.level = level
        self.use_progress_bar = use_progress_bar
        self.output_dir = Path(output_dir)
        self.feature_names = feature_names or []
        self.generation_history: List[GenerationMetrics] = []
        self.total_generations = total_generations
        
        # Define headers and column widths
        self.headers = ['Gen', 'Best Fit', 'Avg Fit', 'MAE', 'Comp Cost', 'Mem Cost']
        self.column_widths = [5, 20, 12, 20, 15, 12]
        
        # Print header
        if self.level > 0:
            print("Generation Progress:\n")
            header_line = "".join(f"{header:<{width}}" for header, width in zip(self.headers, self.column_widths))
            separator_line = "".join("-" * width for width in self.column_widths)
            print(header_line)
            print(separator_line)
        
        # Initialize progress bar
        self.progress_bar = None
        if self.use_progress_bar and total_generations > 0:
            self.progress_bar = tqdm(total=total_generations, 
                                     desc="Evolution Progress",
                                     unit="gen",
                                     leave=True)
    
    def print_generation_stats(self, metrics: GenerationMetrics):
        """Print a single line for this generation, with improvements."""
        if self.level == 0:
            return
        
        self.generation_history.append(metrics)
        
        # Calculate improvements relative to the first generation
        initial = self.generation_history[0]
        best_fit_improvement = ((initial.best_fitness - metrics.best_fitness) / abs(initial.best_fitness)) * 100
        mae_improvement = ((initial.mae - metrics.mae) / abs(initial.mae)) * 100
        
        best_fit_str = f"{metrics.best_fitness:.4f} ({best_fit_improvement:+.1f}%)"
        mae_str = f"{metrics.mae:.4f} ({mae_improvement:+.1f}%)"
        
        # Format the row with fixed widths
        row = [
            str(metrics.generation),
            best_fit_str,
            f"{metrics.avg_fitness:.4f}",
            mae_str,
            f"{metrics.compute_cost:.2e}",
            f"{metrics.memory_cost:.2e}"
        ]
        
        row_str = "".join(f"{cell:<{width}}" for cell, width in zip(row, self.column_widths))
        
        # Print the row above the progress bar
        if self.use_progress_bar:
            tqdm.write(row_str)
            time.sleep(0.05)  # Small delay to stabilize
        else:
            print(row_str)
        
        if self.progress_bar is not None:
            self.progress_bar.update(1)
    
    def print_final_summary(self):
        """Print final summary statistics."""
        if self.level == 0 or not self.generation_history:
            return
        
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        
        initial = self.generation_history[0]
        final = self.generation_history[-1]
        
        improvement = (initial.best_fitness - final.best_fitness) / abs(initial.best_fitness) * 100
        mae_improvement = (initial.mae - final.mae) / abs(initial.mae) * 100
        
        # Define summary headers and widths
        summary_headers = ["Metric", "Improvement"]
        summary_column_widths = [30, 15]
        
        # Print Final Summary
        print("\nFinal Summary:")
        summary_header_str = "".join(f"{header:<{width}}" for header, width in zip(summary_headers, summary_column_widths))
        summary_separator_str = "".join("-" * width for width in summary_column_widths)
        print(summary_header_str)
        print(summary_separator_str)
        
        summary_rows = [
            ["Best Fitness Improvement", f"{improvement:+.1f}%"],
            ["MAE Improvement", f"{mae_improvement:+.1f}%"]
        ]
        
        for row in summary_rows:
            row_str = "".join(f"{cell:<{width}}" for cell, width in zip(row, summary_column_widths))
            print(row_str)
        print()
    
    def __del__(self):
        if self.progress_bar is not None:
            self.progress_bar.close()

    def _node_to_python(self, node) -> str:
        """Convert a node to actual Python code using real feature names"""
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
            elif node.value == 'abs':
                return f"np.abs({self._node_to_python(node.children[0])})"
            elif node.value == 'square':
                return f"np.square({self._node_to_python(node.children[0])})"
            elif node.value == 'sqrt':
                return f"np.sqrt(np.abs({self._node_to_python(node.children[0])}))"
            elif node.value == 'sin':
                return f"np.sin(np.clip({self._node_to_python(node.children[0])}, -50, 50))"
            elif node.value == 'exp':
                return f"np.clip(np.exp(-np.abs({self._node_to_python(node.children[0])})), 0, 1e6)"
            elif node.value == 'conditional':
                return f"np.where({self._node_to_python(node.children[0])} > 0, {self._node_to_python(node.children[1])}, -{self._node_to_python(node.children[1])})"
            elif node.value == 'safe_divide':
                return f"np.divide({self._node_to_python(node.children[0])}, 1 + np.abs({self._node_to_python(node.children[1])}))"
    
    def export_solution(self, program, timestamp: str = None):
        if self.level < 2:
            return
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        code_file = self.output_dir / f"genetic_solution_{timestamp}.py"
        metrics_file = self.output_dir / f"training_metrics_{timestamp}.txt"
        
        # Generate Python solution file
        with open(code_file, 'w') as f:
            f.write("import numpy as np\n\n")
            f.write('def predict(data: dict) -> np.ndarray:\n')
            f.write('    """\n')
            f.write('    Predict using the generated genetic program.\n\n')
            f.write('    Required features:\n')
            for feature in self.feature_names:
                f.write(f'    - {feature}\n')
            f.write('\n')
            f.write('    Args:\n')
            f.write('        data: Dictionary mapping feature names to numpy arrays\n')
            f.write('    Returns:\n')
            f.write('        numpy.ndarray: Predictions\n')
            f.write('    """\n')
            f.write('    # Clip inputs for numerical stability\n')
            f.write('    ')
            f.write(f"return np.clip({self._node_to_python(program.root)}, -1e6, 1e6)")
        
        if self.generation_history and self.level >= 2:
            with open(metrics_file, 'w') as f:
                f.write("Training Metrics:\n\n")
                for m in self.generation_history:
                    f.write(f"Generation {m.generation}:\n")
                    f.write(f"  Best Fitness: {m.best_fitness:.4f}\n")
                    f.write(f"  Avg Fitness: {m.avg_fitness:.4f}\n")
                    f.write(f"  MAE: {m.mae:.4f}\n")
                    f.write(f"  Runtime: {m.runtime:.2f}s\n\n")
                
        print(f"\nSolution exported to: {code_file}")

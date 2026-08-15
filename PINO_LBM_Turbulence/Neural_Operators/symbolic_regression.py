import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import sympy as sp

class SymbolicRegression:
   
    def __init__(self, candidate_functions=None):
        if candidate_functions is None:
            self.candidate_functions = [
                ('constant', lambda: 1.0),
                ('I2S', lambda x: x['I2S']),
                ('I2S^0.25', lambda x: x['I2S']**0.25),
                ('I2S^0.5', lambda x: x['I2S']**0.5),
                ('I2Omega', lambda x: x['I2Omega']),
                ('I2Omega^0.25', lambda x: x['I2Omega']**0.25),
                ('I2Omega^0.5', lambda x: x['I2Omega']**0.5),
                ('I2S*I2Omega', lambda x: x['I2S'] * x['I2Omega']),
                ('IV_SOmega', lambda x: x['IV_SOmega']),
                ('IV_SOmega^0.5', lambda x: x['IV_SOmega']**0.5),
                ('IS', lambda x: x['IS']),
                ('IS^0.5', lambda x: x['IS']**0.5),
                ('-I2S', lambda x: -x['I2S']),
                ('(-I2S)^0.5', lambda x: (-x['I2S'])**0.5 if x['I2S'] < 0 else 0.0),
            ]
    
    def discover_closure(self, data, max_terms=6, population_size=100, generations=50):
        """
        Discover symbolic expression using genetic programming.
        
        Args:
            data: Dict of flow invariants
            max_terms: Maximum number of terms in expression
            population_size: Population size for genetic algorithm
            generations: Number of generations
        
        Returns:
            best_expression: Symbolic expression as a function
        """
        # Simple implementation using genetic programming
        # For production, consider using libraries like DEAP or PySR
        
        import random
        
        def random_expression(depth=0):
            """Generate a random expression tree."""
            if depth > 3 or random.random() < 0.3:
                # Terminal
                return random.choice(self.candidate_functions)
            else:
                # Operator
                op = random.choice(['+', '*'])
                left = random_expression(depth + 1)
                right = random_expression(depth + 1)
                return (op, left, right)
        
        def evaluate_expression(expr, x):
            """Evaluate expression given input dictionary."""
            if isinstance(expr, tuple):
                if len(expr) == 2:
                    # Terminal
                    return expr[1](x)
                else:
                    # Operator
                    op = expr[0]
                    left = evaluate_expression(expr[1], x)
                    right = evaluate_expression(expr[2], x)
                    if op == '+':
                        return left + right
                    elif op == '*':
                        return left * right
                    elif op == '-':
                        return left - right
                    elif op == '/':
                        return left / (right + 1e-8)
            else:
                return evaluate_expression(expr, x)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            expr = random_expression()
            population.append((expr, 0))
        
        # Evolve
        best_expr = None
        best_fitness = float('inf')
        
        for gen in range(generations):
            # Evaluate fitness
            for i, (expr, _) in enumerate(population):
                fitness = 0.0
                for d in data:
                    pred = evaluate_expression(expr, d)
                    target = d['tau_turb']
                    fitness += (pred - target) ** 2
                population[i] = (expr, fitness)
            
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            if population[0][1] < best_fitness:
                best_expr = population[0][0]
                best_fitness = population[0][1]
            
            # Selection and crossover
            new_population = population[:population_size // 2]
            while len(new_population) < population_size:
                parent1 = random.choice(population[:population_size // 4])
                parent2 = random.choice(population[:population_size // 4])
                child = self._crossover(parent1[0], parent2[0])
                new_population.append((child, 0))
            
            # Mutation
            for i in range(len(new_population)):
                if random.random() < 0.1:
                    new_population[i] = (self._mutate(new_population[i][0]), 0)
            
            population = new_population
        
        return self._expression_to_function(best_expr)
    
    def _crossover(self, expr1, expr2):
        """Crossover operation for genetic programming."""
        import copy
        import random
        
        def get_subtree(expr):
            if random.random() < 0.3:
                return expr
            if isinstance(expr, tuple) and len(expr) > 2:
                child_idx = random.randint(1, len(expr) - 1)
                return get_subtree(expr[child_idx])
            return expr
        
        child = copy.deepcopy(expr1)
        # Replace a random subtree in child with a subtree from parent2
        # Simplified implementation
        return child
    
    def _mutate(self, expr):
        """Mutation operation for genetic programming."""
        import copy
        import random
        
        def random_terminal():
            return random.choice(self.candidate_functions)
        
        def mutate_subtree(expr, depth=0):
            if depth > 3 or random.random() < 0.2:
                return random_terminal()
            if isinstance(expr, tuple) and len(expr) > 2:
                child_idx = random.randint(1, len(expr) - 1)
                new_child = mutate_subtree(expr[child_idx], depth + 1)
                new_expr = list(expr)
                new_expr[child_idx] = new_child
                return tuple(new_expr)
            return expr
        
        return mutate_subtree(copy.deepcopy(expr))
    
    def _expression_to_function(self, expr):
        """Convert expression tree to a callable function."""
        def eval_expr(expr, x):
            if isinstance(expr, tuple):
                if len(expr) == 2:
                    return expr[1](x)
                else:
                    op = expr[0]
                    left = eval_expr(expr[1], x)
                    right = eval_expr(expr[2], x)
                    if op == '+':
                        return left + right
                    elif op == '*':
                        return left * right
                    elif op == '-':
                        return left - right
                    elif op == '/':
                        return left / (right + 1e-8)
            else:
                return eval_expr(expr, x)
        
        def func(x):
            return eval_expr(expr, x)
        
        return func
    
    def get_symbolic_expression(self, expr):
        """Convert expression tree to sympy expression."""
        def to_sympy(expr):
            if isinstance(expr, tuple):
                if len(expr) == 2:
                    name = expr[0]
                    if name.startswith('I2S'):
                        return sp.Symbol('I2S') ** 0.25
                    elif name.startswith('I2Omega'):
                        return sp.Symbol('I2Omega')
                    elif name.startswith('IV_SOmega'):
                        return sp.Symbol('IV_SOmega')
                    elif name.startswith('IS'):
                        return sp.Symbol('IS')
                    else:
                        return sp.Symbol(name)
                else:
                    op = expr[0]
                    left = to_sympy(expr[1])
                    right = to_sympy(expr[2])
                    if op == '+':
                        return left + right
                    elif op == '*':
                        return left * right
                    elif op == '-':
                        return left - right
                    elif op == '/':
                        return left / right
            else:
                return sp.Symbol(str(expr))
        
        return to_sympy(expr)

class InterpretableClosure(nn.Module):
    """
    Neural module that combines symbolic regression with neural refinement.
    """
    def __init__(self, symbolic_func, hidden_dim=32):
        super().__init__()
        self.symbolic_func = symbolic_func
        self.refinement = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, invariants):
        """
        Compute effective relaxation time from invariants.
        
        Args:
            invariants: Dict of flow invariants (IS, I2S, I2Omega, IV_SOmega)
        
        Returns:
            tau_eff: Effective relaxation time
        """
        # Symbolic prediction
        tau_symbolic = self.symbolic_func(invariants)
        
        # Neural refinement
        inv_tensor = torch.stack([
            invariants['IS'],
            invariants['I2S'],
            invariants['I2Omega'],
            invariants['IV_SOmega'],
            invariants.get('IS', torch.zeros_like(invariants['IS'])),
            invariants.get('IS', torch.zeros_like(invariants['IS']))
        ], dim=-1)
        
        delta = self.refinement(inv_tensor).squeeze(-1)
        
        # Combine with physical constraints
        tau_eff = tau_symbolic + delta
        tau_eff = torch.clamp(tau_eff, 0.5, 1.0)  # Stability bound
        
        return tau_eff

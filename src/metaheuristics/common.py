"""Shared tools for all metaheuristics: a budget-limited objective that records the search history."""
import numpy as np

class BudgetExhausted(Exception):
    """Raised when the evaluation budget is used up (stops any algorithm cleanly)."""

class Objective:
    """Wraps a fitness function f(x), x in [0,1]^D, minimised.
    - counts evaluations and stops at `budget`
    - remembers the best solution found so far
    - records the history needed for the convergence curves
    """

    def __init__(self, func, budget):
        self.func = func
        self.budget = budget
        self.n_evals = 0
        self.best_f = np.inf
        self.best_x = None
        self.best_history = []
        self.values = []
        self.phase = []
        self.current_phase = ''

    @property
    def left(self):
        return self.budget - self.n_evals

    def __call__(self, x):
        if self.n_evals >= self.budget:
            raise BudgetExhausted
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        f = float(self.func(x))
        self.n_evals += 1
        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()
        self.best_history.append(self.best_f)
        self.values.append(f)
        self.phase.append(self.current_phase)
        return f

def evaluate_all(obj, X):
    """Evaluate the rows of X until the budget runs out; returns the fitness of the rows evaluated."""
    fit = []
    for x in X:
        if obj.left == 0:
            break
        fit.append(obj(x))
    return np.array(fit)

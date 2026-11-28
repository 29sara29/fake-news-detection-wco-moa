"""World Cup Optimization (WCO)
Used here as the EXPLORATION phase.

Mapping of the paper to this implementation (interpretation documented in the report):
  team                 -> one candidate solution x in [0,1]^D
  continents           -> groups of teams; each starts in its own region with its own spread
  continent score      -> Rank = (beta * std + mean) / 2 of the members' fitness   (paper eq. 10)
  preliminary round    -> the 2 best teams of each continent qualify (X_Rank)       (eq. 13)
  champion             -> best team of the cup                                       (eq. 14)
  next cup population  -> Pop = [X_Best, X_Rand] split by the cross point CP        (eq. 15, 19)
       X_Rand : new random teams anywhere in the bounds        (exploration)
       X_Best : teams generated near qualified teams, within a radius set by the accuracy
                coefficient ac; better-ranked continents produce more of them   (eq. 16-18)
  the champion is kept (elitism) so the best solution is never lost.
"""
import numpy as np
from .common import BudgetExhausted, evaluate_all

def wco(obj, dim, rng, pop_size=15, n_continents=5, beta=0.5, cp=0.5, ac=0.2):
    """Run WCO until obj's budget is used. Returns the final population (X, fit), sorted best first."""
    obj.current_phase = 'WCO'
    sizes = np.full(n_continents, pop_size // n_continents)
    sizes[: pop_size % n_continents] += 1
    labels = np.repeat(np.arange(n_continents), sizes)

    spreads = np.linspace(0.10, 0.40, n_continents)
    X = np.vstack([np.clip(rng.normal(rng.random(dim), spreads[c], (sizes[c], dim)), 0, 1)
                   for c in range(n_continents)])

    try:
        fit = evaluate_all(obj, X)
        if len(fit) < len(X):
            raise BudgetExhausted
        while obj.left > 0:
            scores = np.array([(beta * fit[labels == c].std(ddof=1) + fit[labels == c].mean()) / 2
                               for c in range(n_continents)])
            order = np.argsort(scores)

            qualified, weights = [], []
            for rank_pos, c in enumerate(order):
                members = np.where(labels == c)[0]
                top2 = members[np.argsort(fit[members])[:2]]
                qualified.extend(top2)
                weights.extend([n_continents - rank_pos] * len(top2))
            qualified = np.array(qualified)
            weights = np.array(weights, dtype=float) / np.sum(weights)
            champ = int(np.argmin(fit))

            n_new = pop_size - 1
            n_rand = int(round(cp * n_new))
            n_best = n_new - n_rand
            parents = rng.choice(qualified, size=n_best, p=weights)
            radius = ac / 2
            X_best = np.clip(X[parents] + rng.uniform(-radius, radius, (n_best, dim)), 0, 1)
            X_rand = rng.random((n_rand, dim))
            X_new = np.vstack([X_best, X_rand])

            fit_new = evaluate_all(obj, X_new)
            X = np.vstack([X[champ], X_new[: len(fit_new)]])
            fit = np.concatenate([[fit[champ]], fit_new])
            if len(fit) < pop_size:
                break

    except BudgetExhausted:
        pass

    order = np.argsort(fit)
    return X[: len(fit)][order], fit[order]

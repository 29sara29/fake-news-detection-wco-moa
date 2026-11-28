"""Hybrid WCO -> MOA: WCO explores for the first part of the budget, then MOA exploits,
starting from WCO's best solutions."""
from .wco import wco
from .moa import moa

def wco_moa(obj, dim, rng, switch=0.5, wco_kwargs=None, moa_kwargs=None):
    """Spend `switch` x budget on WCO, the rest on MOA seeded with WCO's best agents."""
    total = obj.budget
    obj.budget = int(round(switch * total))
    X, fit = wco(obj, dim, rng, **(wco_kwargs or {}))
    switch_eval = obj.n_evals
    obj.budget = total
    X, fit, stats = moa(obj, dim, rng, init_X=X, init_fit=fit, **(moa_kwargs or {}))
    stats['switch_eval'] = switch_eval
    return X, fit, stats

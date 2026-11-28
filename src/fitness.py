import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

SPACE = {
    'n_estimators':     (100, 600, 'lin', True),
    'learning_rate':    (0.01, 0.3, 'log', False),
    'max_depth':        (3, 10, 'lin', True),
    'min_child_weight': (1, 10, 'lin', False),
    'subsample':        (0.5, 1.0, 'lin', False),
    'colsample_bytree': (0.3, 1.0, 'lin', False),
    'gamma':            (0, 5, 'lin', False),
    'reg_alpha':        (0, 5, 'lin', False),
    'reg_lambda':       (0, 5, 'lin', False),
}
DIM = len(SPACE)

def decode(x):
    """[0,1]^9 vector -> dict of XGBoost hyperparameters."""
    params = {}
    for v, (name, (lo, hi, scale, is_int)) in zip(np.clip(x, 0, 1), SPACE.items()):
        val = 10 ** (np.log10(lo) + v * (np.log10(hi) - np.log10(lo))) if scale == 'log' else lo + v * (hi - lo)
        params[name] = int(round(val)) if is_int else float(val)
    return params

def make_model(params, device='cuda', seed=42):
    return XGBClassifier(**params, tree_method='hist', device=device, eval_metric='logloss',
                         random_state=seed, n_jobs=4, verbosity=0)

def make_fitness(X, y, device='cuda', n_splits=3, seed=42):
    """Returns f(x) = 1 - mean macro-F1 over fixed stratified folds (lower is better)."""
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    def fitness(x):
        scores = []
        for tr, va in folds:
            model = make_model(decode(x), device, seed)
            model.fit(X[tr], y[tr])
            scores.append(f1_score(y[va], model.predict(X[va]), average='macro'))
        return 1.0 - float(np.mean(scores))
    return fitness

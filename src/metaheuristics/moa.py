"""Morabaraba Optimization Algorithm (MOA)
Used here as the EXPLOITATION phase.

Game -> optimisation mapping (paper section 3):
  cow (agent)            -> one candidate solution x
  two teams Ta / Tb      -> population split in two competing halves
  cow count c_i          -> per-agent state (starts at 12) that selects the phase: (energy)
                              Placing  c >= 9      : Levy pull toward the team leader        (eq. 4)
                              Moving   3 < c < 9   : mill check + 3 candidate moves          (eq. 5-9)
                              Flying   c == 3      : Levy jump around the global best        (eq. 10)
                              c <= 2               : cow respawned at a random position
  mill                   -> 3 teammates aligned on a board line (a triple of dimensions)   (eq. 5)
  cow shooting           -> after a mill, the worst opposing cow loses one cow and is
                            dragged toward the mill on its worst-aligned dimension       (eq. 6)
  mill roster            -> the K last mills, used as a multi-leader pool (roulette)     (eq. 12)
  mood, velocity, retreat-> x += eta*v - zeta*sign(m)*(x_worst - x)                       (eq. 11, 13)

Implementation choices
  - velocity / retreat are applied to each proposal BEFORE it is evaluated (no extra evaluation)
  - the mill check runs for every cow after its move (flowchart), at most one mill per cow per turn
  - a mill's roster entry is the position of the best cow of the triple
  - teams are formed by alternating the ranked population (both teams get strong cows)
"""
import numpy as np
from .common import BudgetExhausted, evaluate_all

def _board_mills():
    mills = []
    for s in range(3):
        b = 8 * s
        mills += [(b + 0, b + 1, b + 2), (b + 2, b + 3, b + 4), (b + 4, b + 5, b + 6), (b + 6, b + 7, b + 0)]
    for p in (1, 3, 5, 7):
        mills.append((p, 8 + p, 16 + p))
    for p in (0, 2, 4, 6):
        mills.append((p, 8 + p, 16 + p))
    return mills

def board_lines(dim):
    """Map the 20 board mills to triples of distinct decision variables (round-robin point -> dimension)."""
    lines = []
    for m in _board_mills():
        dims = tuple(sorted({p % dim for p in m}))
        if len(dims) == 3 and dims not in lines:
            lines.append(dims)
    return lines

def levy(rng, size, beta=1.5):
    """Levy-flight steps (Mantegna's algorithm): mostly small steps, occasionally very long jumps."""
    from math import gamma, sin, pi
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, size)
    v = rng.normal(0, 1, size)
    return u / np.abs(v) ** (1 / beta)

def tent_obl_init(obj, n, dim, rng, mu=1.999):
    """Chaotic Tent-map initialisation + Opposition-Based Learning: evaluate 2n points, keep the best n."""
    z = rng.random(dim)
    X = np.empty((n, dim))
    for i in range(n):
        z = np.where(z < 0.5, mu * z, mu * (1 - z))
        z = np.clip(z, 1e-6, 1 - 1e-6)
        X[i] = z
    X_all = np.vstack([X, 1.0 - X])
    fit = evaluate_all(obj, X_all)
    X_all = X_all[: len(fit)]
    order = np.argsort(fit)[:n]
    return X_all[order], fit[order]

def moa(obj, dim, rng, pop_size=10, init_X=None, init_fit=None,
        alpha=0.05, tau=0.05, gamma_drag=0.30, F=0.8, b=1.0, eta=0.40, zeta=0.20, K=5, c0=12):

    """Run MOA until obj's budget is used. Optional init_X/init_fit = seed population (e.g. from WCO).
    Returns the final population (X, fit) sorted best first, and phase statistics."""
    obj.current_phase = 'MOA'
    stats = {'placing': 0, 'moving': 0, 'flying': 0, 'mills': 0, 'respawns': 0}
    try:
        if init_X is None:
            X, fit = tent_obl_init(obj, pop_size, dim, rng)
        else:
            X, fit = np.array(init_X[:pop_size], dtype=float), np.array(init_fit[:pop_size], dtype=float)
        n = len(X)
        order = np.argsort(fit)
        X, fit = X[order], fit[order]
        team = np.arange(n) % 2
        cows = np.full(n, c0)
        vel = np.zeros((n, dim))
        roster = []
        lines = board_lines(dim)

        def propose(i, x_new, m):
            """Velocity persistence + retreat from the worst teammate, then stay in bounds (eq. 13)."""
            mates = np.where(team == team[i])[0]
            x_worst = X[mates[np.argmax(fit[mates])]]
            return np.clip(x_new + eta * vel[i] - zeta * np.sign(m) * (x_worst - X[i]), 0, 1)

        while obj.left > 0:
            for i in range(n):
                if obj.left == 0:
                    break
                mates = np.where(team == team[i])[0]
                lead = X[mates[np.argmin(fit[mates])]]
                m = 2 * rng.random() - 1
                c = cows[i]

                if c <= 2:
                    stats['respawns'] += 1
                    x_new = rng.random(dim)
                    f_new = obj(x_new)
                    cows[i] = c0
                elif c >= 9:
                    stats['placing'] += 1
                    x_new = propose(i, lead + alpha * levy(rng, dim) * (lead - X[i]), m)
                    f_new = obj(x_new)
                elif c == 3:
                    stats['flying'] += 1
                    best = obj.best_x
                    x_new = propose(i, best + rng.normal(0, 1, dim) * levy(rng, dim) * np.abs(best - X[i]), m)
                    f_new = obj(x_new)
                else:
                    stats['moving'] += 1
                    if roster:
                        q = np.array([r[1] for r in roster]); q = q / q.sum()
                        lead_m = roster[rng.choice(len(roster), p=q)][0]
                    else:
                        lead_m = lead
                    r1, r2 = rng.choice(mates, 2, replace=len(mates) < 2)
                    l = rng.uniform(-1, 1)
                    A = X[i] + rng.random() * (lead_m - X[i])
                    B = lead_m + F * (X[r1] - X[r2])
                    C = np.abs(lead_m - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + lead_m
                    cands = [propose(i, z, m) for z in (A, B, C)]
                    f_c = [obj(z) for z in cands]
                    k = int(np.argmin(f_c))
                    x_new, f_new = cands[k], f_c[k]

                if f_new < fit[i]:
                    d = x_new - X[i]
                    vel[i] = d / (np.linalg.norm(d) + 1e-12)
                X[i], fit[i] = x_new, f_new

                mates = np.where(team == team[i])[0]
                others = [j for j in mates[np.argsort(fit[mates])] if j != i][:2]
                if len(others) == 2:
                    trio = X[[i] + others]
                    for L in lines:
                        vals = trio[:, L]
                        spread = vals.max(axis=0) - vals.min(axis=0)
                        Q = 1 - np.max(spread / np.maximum(np.abs(vals).max(axis=0), 1e-12))
                        if Q >= 1 - tau:
                            stats['mills'] += 1
                            opp = np.where(team != team[i])[0]
                            w = opp[np.argmax(fit[opp])]
                            cows[w] -= 1
                            mu_L = vals.mean(axis=0)
                            k = int(np.argmax(np.abs(X[w, list(L)] - mu_L)))
                            X[w, L[k]] = (1 - gamma_drag) * X[w, L[k]] + gamma_drag * mu_L[k]
                            roster.append((trio[np.argmin(fit[[i] + others])].copy(), Q))
                            roster = roster[-K:]
                            break
    except BudgetExhausted:
        pass

    order = np.argsort(fit)
    return X[order], fit[order], stats

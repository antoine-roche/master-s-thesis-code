"""
Simulations for "Finite-sample behaviour of the MLE in Poisson regression
under Gaussian design".

Three things are computed here.

1. EXISTENCE.  The MLE fails to exist iff some b != 0 "Poisson separates"
   the data (eq. (22) of the thesis):
         <X_i, b> = 0   for every i with Y_i > 0,
         <X_i, b> <= 0  for every i with Y_i = 0.
   This is decided exactly by a small linear program -- see `mle_exists`.

2. THE MLE ITSELF, by damped Newton with backtracking.  The Poisson loss is
   self-concordant, so Newton behaves very well once damped.

3. THE EXCESS RISK, in closed form.  Under X ~ N(0, I_d) the population risk
   is available analytically (see `excess_risk`), so the excess risk is
   computed exactly -- there is no Monte-Carlo error on the y-axis of the
   risk plots, only the randomness of the sample itself.

Author: Antoine Roche
"""

from __future__ import annotations

import numpy as np
# (null spaces are obtained from a d x d eigendecomposition; no SVD needed)
from scipy.optimize import linprog


# ----------------------------------------------------------------------
# 1. Sampling
# ----------------------------------------------------------------------

def sample_data(n, d, beta_star, rng):
    """Draw n i.i.d. observations from the well-specified Poisson model.

    X_i ~ N(0, I_d),  Y_i | X_i ~ Poisson(exp(<beta_star, X_i>)).

    Returns (X, Y) with shapes (n, d) and (n,).
    """
    X = rng.standard_normal((n, d))
    eta = X @ beta_star
    # exp overflows for large <beta_star, X_i>; clip the *rate*, not eta,
    # and warn the caller through a finite cap.  With ||beta_star|| <= 4 and
    # n <= 1e6 this cap is essentially never active.
    lam = np.exp(np.clip(eta, -700.0, 30.0))
    Y = rng.poisson(lam)
    return X, Y


def make_beta_star(d, norm, rng=None, direction=None):
    """A parameter vector of prescribed Euclidean norm.

    By rotational invariance the direction is irrelevant, so e_1 is the
    default; pass `rng` to randomise it instead.
    """
    if direction is None:
        if rng is None:
            direction = np.zeros(d)
            direction[0] = 1.0
        else:
            direction = rng.standard_normal(d)
    direction = np.asarray(direction, dtype=float)
    return norm * direction / np.linalg.norm(direction)


# ----------------------------------------------------------------------
# 2. Existence of the MLE
# ----------------------------------------------------------------------

def mle_exists(X, Y, tol=1e-7, return_certificate=False):
    """Decide whether the Poisson MLE exists for this sample.

    The MLE fails to exist iff there is b != 0 with
        <X_i, b> = 0    for i in P = {Y_i > 0},
        <X_i, b> <= 0   for i in Z = {Y_i = 0}.

    Implementation.  The equality constraints say exactly that b lies in
    N = ker(X_P).  We take an orthonormal basis V of N (via SVD) and write
    b = V c, so the question becomes: is there c != 0 with M c <= 0, where
    M = X_Z V?  Two shortcuts settle most cases immediately:

      * dim N = 0  (i.e. |P| >= d, generically)  =>  b = 0 only  =>  exists.
      * Z empty and dim N > 0                    =>  any c works =>  fails.

    Otherwise we solve the LP
        max t   s.t.  M c + t <= 0,  |c|_inf <= 1,  0 <= t <= 1,
    after normalising the rows of M to unit length (so that `tol` is scale
    free).  t* > 0 certifies a strictly separating direction.

    NOTE ON THE BOUNDARY CASE.  The LP detects a c with M c < 0 strictly.
    A cone {c : Mc <= 0} that is non-trivial but has empty interior would be
    missed; this requires the rows of M to positively span a proper subspace,
    a measure-zero coincidence for a continuous design.  So the test is
    correct almost surely, which is all the theory asks for.

    Returns bool, or (bool, b) if return_certificate=True, where b is a
    separating direction when the MLE fails and None when it exists.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    n, d = X.shape

    P = Y > 0
    Zmask = ~P
    nP = int(P.sum())

    # --- basis of the null space of X_P ---
    # Never form the |P| x |P| factor of a full SVD: we only ever need the
    # right singular vectors, so we eigendecompose the d x d Gram matrix
    # X_P^T X_P, whose kernel is ker(X_P).  Cost O(|P| d^2 + d^3).
    if nP == 0:
        V = np.eye(d)
    else:
        XP = X[P]
        G = XP.T @ XP                              # (d, d)
        evals, evecs = np.linalg.eigh(G)           # ascending
        lam_max = evals[-1] if evals.size else 0.0
        rank_tol = max(nP, d) * np.finfo(float).eps * max(lam_max, 1.0)
        V = evecs[:, evals <= rank_tol]            # (d, m)
    m = V.shape[1]

    if m == 0:
        # only b = 0 satisfies the equality constraints
        return (True, None) if return_certificate else True

    nZ = int(Zmask.sum())
    if nZ == 0:
        b = V[:, 0]
        return (False, b) if return_certificate else False

    M = X[Zmask] @ V                          # (|Z|, m)
    rownorm = np.linalg.norm(M, axis=1, keepdims=True)
    rownorm[rownorm == 0] = 1.0
    M = M / rownorm

    # variables (c, t); maximise t  <=>  minimise -t
    cost = np.zeros(m + 1)
    cost[-1] = -1.0
    A_ub = np.hstack([M, np.ones((M.shape[0], 1))])
    b_ub = np.zeros(M.shape[0])
    bounds = [(-1.0, 1.0)] * m + [(0.0, 1.0)]

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"existence LP failed: {res.message}")

    t = -res.fun
    fails = t > tol
    if return_certificate:
        cert = V @ res.x[:m] if fails else None
        return (not fails), cert
    return not fails


# ----------------------------------------------------------------------
# 3. Fitting the MLE
# ----------------------------------------------------------------------

def _loss_grad_hess(beta, X, Y, need_hess=True):
    eta = X @ beta
    # cap at 300 rather than 700: e^700 overflows once summed by the matmul
    # below, which pollutes the line search with inf/nan instead of simply
    # rejecting a bad step.
    eta = np.clip(eta, -700.0, 300.0)
    w = np.exp(eta)                              # weights e^{<beta,X_i>}
    n = X.shape[0]
    loss = (w.sum() - Y @ eta) / n
    grad = X.T @ (w - Y) / n
    if not need_hess:
        return loss, grad, None
    hess = (X * w[:, None]).T @ X / n
    return loss, grad, hess


def fit_mle(X, Y, beta0=None, max_iter=200, tol=1e-10, verbose=False):
    """Minimise the empirical Poisson risk by damped Newton + backtracking.

    Returns (beta_hat, info).  `info['converged']` is True when the Newton
    decrement criterion was met.

    IMPORTANT.  Always call `mle_exists` FIRST.  When the MLE does not exist
    the loss is flat along the separating ray, so the gradient underflows
    while the iterate drifts off to infinity, and this routine then reports
    `converged=True` at a large, meaningless beta.  Convergence of the
    optimiser is therefore NOT a valid test of existence; the LP is.
    Watch `info['beta_norm']` if you want a diagnostic.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    beta = np.zeros(d) if beta0 is None else np.array(beta0, dtype=float)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        loss, grad, hess = _loss_grad_hess(beta, X, Y)

        # solve H step = -grad, with a small ridge if H is near-singular
        ridge = 0.0
        for _ in range(8):
            try:
                step = np.linalg.solve(hess + ridge * np.eye(d), -grad)
                break
            except np.linalg.LinAlgError:
                ridge = max(1e-12, ridge * 10)
        else:
            step = -grad

        # Newton decrement squared: lambda^2 = -grad . step
        lam2 = float(-grad @ step)
        if lam2 <= tol:
            converged = True
            break

        # backtracking line search (Armijo)
        t = 1.0
        for _ in range(60):
            new_loss, _, _ = _loss_grad_hess(beta + t * step, X, Y, need_hess=False)
            if np.isfinite(new_loss) and new_loss <= loss - 0.25 * t * lam2:
                break
            t *= 0.5
        else:
            break

        beta = beta + t * step
        if verbose:
            print(f"  it {it:3d}  loss {new_loss: .8f}  lam2 {lam2:.3e}  t {t:.3g}")

    grad_norm = float(np.linalg.norm(_loss_grad_hess(beta, X, Y, need_hess=False)[1]))
    return beta, {"converged": converged, "n_iter": it, "grad_norm": grad_norm,
                  "beta_norm": float(np.linalg.norm(beta))}


# ----------------------------------------------------------------------
# 4. Risk, in closed form
# ----------------------------------------------------------------------

def population_risk(beta, beta_star):
    """L(beta) = E[e^{<beta,X>}] - E[Y <beta,X>] for X ~ N(0, I_d).

    Both expectations are Gaussian integrals:
        E[e^{<beta,X>}]   = e^{||beta||^2 / 2},
        E[Y <beta,X>]     = E[e^{<beta_star,X>} <beta,X>]
                          = e^{||beta_star||^2 / 2} <beta, beta_star>,
    the second by exponential tilting (Appendix B.1).
    """
    beta = np.asarray(beta, float)
    beta_star = np.asarray(beta_star, float)
    return np.exp(beta @ beta / 2) - np.exp(beta_star @ beta_star / 2) * (beta @ beta_star)


def excess_risk(beta, beta_star):
    """L(beta) - L(beta_star), computed stably.

    Writing delta = beta - beta_star and s = <delta, beta_star> + ||delta||^2/2,
        L(beta) - L(beta_star)
            = e^{||beta_star||^2/2} [ (e^s - s - 1) + ||delta||^2 / 2 ].
    Both bracketed terms are non-negative, so this form avoids the
    catastrophic cancellation of subtracting two large risks, and manifestly
    returns a non-negative number.
    """
    beta = np.asarray(beta, float)
    beta_star = np.asarray(beta_star, float)
    delta = beta - beta_star
    dd = float(delta @ delta)
    s = float(delta @ beta_star) + dd / 2
    return float(np.exp(beta_star @ beta_star / 2) * ((np.expm1(s) - s) + dd / 2))


# ----------------------------------------------------------------------
# 5. Experiment drivers
# ----------------------------------------------------------------------

def existence_probability(n, d, beta_norm, n_rep, rng):
    """Fraction of samples for which the MLE exists, plus the mean of |Z|/n."""
    beta_star = make_beta_star(d, beta_norm)
    hits = 0
    zfrac = 0.0
    for _ in range(n_rep):
        X, Y = sample_data(n, d, beta_star, rng)
        hits += bool(mle_exists(X, Y))
        zfrac += float((Y == 0).mean())
    return hits / n_rep, zfrac / n_rep


def existence_curve(d, beta_norm, ratios, n_rep, rng):
    """P(MLE exists) as a function of n/d, at fixed d and ||beta_star||."""
    out = []
    for r in ratios:
        n = int(round(r * d))
        p, zf = existence_probability(n, d, beta_norm, n_rep, rng)
        out.append({"ratio": r, "n": n, "d": d, "beta_norm": beta_norm,
                    "p_exists": p, "zero_frac": zf})
    return out


def risk_experiment(n, d, beta_norm, n_rep, rng, require_existence=True):
    """Excess risk of the MLE over n_rep samples.

    Returns a dict with the mean and quantiles of the *scaled* excess risk
        n * (L(beta_hat) - L(beta_star)) / d,
    which Wilks' theorem predicts should settle at 1/2 once n is large
    enough, since 2n * excess -> chi^2(d).
    """
    beta_star = make_beta_star(d, beta_norm)
    scaled, dists, n_ok, n_fail = [], [], 0, 0

    for _ in range(n_rep):
        X, Y = sample_data(n, d, beta_star, rng)
        if require_existence and not mle_exists(X, Y):
            n_fail += 1
            continue
        beta_hat, info = fit_mle(X, Y, beta0=np.zeros(d))
        if not info["converged"]:
            n_fail += 1
            continue
        n_ok += 1
        er = excess_risk(beta_hat, beta_star)
        scaled.append(n * er / d)
        dists.append(float(np.linalg.norm(beta_hat - beta_star)))

    scaled = np.array(scaled)
    dists = np.array(dists)
    res = {"n": n, "d": d, "beta_norm": beta_norm,
           "n_ok": n_ok, "n_fail": n_fail}
    if scaled.size:
        res.update({
            "mean_scaled": float(scaled.mean()),
            "median_scaled": float(np.median(scaled)),
            "q90_scaled": float(np.quantile(scaled, 0.90)),
            "mean_dist": float(dists.mean()),
            "q90_dist": float(np.quantile(dists, 0.90)),
        })
    else:
        res.update({k: np.nan for k in
                    ["mean_scaled", "median_scaled", "q90_scaled",
                     "mean_dist", "q90_dist"]})
    return res


def wilks_sample(n, d, beta_norm, n_rep, rng):
    """Samples of 2n(L(beta_hat) - L(beta_star)), to compare with chi^2(d)."""
    beta_star = make_beta_star(d, beta_norm)
    vals = []
    for _ in range(n_rep):
        X, Y = sample_data(n, d, beta_star, rng)
        if not mle_exists(X, Y):
            continue
        beta_hat, info = fit_mle(X, Y, beta0=np.zeros(d))
        if info["converged"]:
            vals.append(2 * n * excess_risk(beta_hat, beta_star))
    return np.array(vals)

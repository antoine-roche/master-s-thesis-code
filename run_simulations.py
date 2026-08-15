"""
Produces the figures for the simulation section.

    python run_simulations.py            # full run  (a few minutes)
    python run_simulations.py --quick    # coarse run (well under a minute)

Writes fig_existence.pdf, fig_risk.pdf, fig_threshold.pdf, fig_wilks.pdf
into ./figures, plus a numerical log to stdout.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from poisson_sim import (existence_probability, risk_experiment,
                         wilks_sample, make_beta_star)

OUT = "figures"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140,
})


# ----------------------------------------------------------------------

def fig_existence(rng, quick):
    """P(MLE exists) as a function of n/d, for several d and ||beta_star||."""
    ratios = np.array([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    dims = [10, 40] if quick else [10, 40, 100]
    norms = [0.0, 2.0, 4.0]
    n_rep = 60 if quick else 300

    fig, axes = plt.subplots(1, len(dims), figsize=(3.1 * len(dims), 2.7),
                             sharey=True)
    print("\n=== Figure 1: existence ===")
    for ax, d in zip(np.atleast_1d(axes), dims):
        for bn in norms:
            ps = []
            for r in ratios:
                p, zfrac = existence_probability(max(int(round(r * d)), 1),
                                                 d, bn, n_rep, rng)
                ps.append(p)
            ax.plot(ratios, ps, marker="o", ms=3, lw=1.2,
                    label=r"$\|\beta_*\|=%.0f$" % bn)
            # predicted location: n(1-p) = d  =>  n/d = 1/(1-p)
            ax.axvline(1.0 / (1.0 - zfrac), ls=":", lw=0.8, alpha=0.5,
                       color=ax.lines[-1].get_color())
            print(f"  d={d:4d} |b*|={bn:.0f}  1/(1-p)={1/(1-zfrac):.2f}  "
                  + " ".join(f"{x:.2f}" for x in ps))
        ax.set_title(f"$d={d}$")
        ax.set_xlabel("$n/d$")
        ax.grid(alpha=0.25, lw=0.5)
    np.atleast_1d(axes)[0].set_ylabel(r"$\mathbb{P}(\mathrm{MLE\ exists})$")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="lower right")
    fig.suptitle(r"Existence transition sits at $n\asymp d$, essentially "
                 r"independent of $\|\beta_*\|$", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_existence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_risk(rng, quick):
    """Scaled excess risk n*(L(bhat)-L(b*))/d against n; Wilks level = 1/2."""
    d = 5
    norms = [0.0, 1.0, 2.0, 2.5] if quick else [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
    ns = ([50, 200, 800, 3200, 12800] if quick else
          [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200])
    reps = (lambda n: 60) if quick else (
        lambda n: 300 if n <= 2000 else (150 if n <= 20000 else 60))

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    print("\n=== Figure 2: excess risk ===")
    for bn in norms:
        ys = [risk_experiment(n, d, bn, reps(n), rng)["mean_scaled"] for n in ns]
        ax.plot(ns, ys, marker="o", ms=3, lw=1.2,
                label=r"$\|\beta_*\|=%.1f$" % bn)
        print(f"  |b*|={bn:.1f}  " + " ".join(f"{y:6.2f}" for y in ys))
    ax.axhline(0.5, color="k", ls="--", lw=1,
               label=r"Wilks: $\mathbb{E}\,\chi^2(d)/(2d)=1/2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$n\,(L(\hat\beta_n)-L(\beta_*))/d$")
    ax.set_title("The asymptotic rate is reached later for stronger signals",
                 fontsize=9)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_risk.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_threshold(rng, quick, level=0.60):
    """Locate n*(B) = least n with scaled risk <= `level`, and fit log n* ~ B^2."""
    d = 5
    norms = [1.0, 1.5, 2.0, 2.5] if quick else [1.0, 1.5, 2.0, 2.5, 3.0]
    reps = (lambda n: 80) if quick else (
        lambda n: 400 if n <= 2000 else (200 if n <= 20000 else 80))

    print("\n=== Figure 3: threshold scaling ===")
    nstars = []
    for bn in norms:
        n, prev_n, prev_v = 4 * d, None, None
        while n < 4_000_000:
            v = risk_experiment(n, d, bn, reps(n), rng)["mean_scaled"]
            if np.isfinite(v) and v <= level:
                if prev_n is None:
                    ns = float(n)
                else:                        # interpolate in log n
                    f = (prev_v - level) / (prev_v - v)
                    ns = float(np.exp(np.log(prev_n) + f * (np.log(n) - np.log(prev_n))))
                break
            prev_n, prev_v = n, v
            n = int(n * 1.6)
        else:
            ns = np.nan
        nstars.append(ns)
        print(f"  |b*|={bn:.1f}  n* = {ns:10.0f}")

    B2 = np.array([b ** 2 for b in norms])
    y = np.log(np.array(nstars))
    A = np.vstack([np.ones_like(B2), B2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  fit: log n* = {coef[0]:.3f} + {coef[1]:.3f} B^2   (R^2={r2:.4f})")

    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    ax.plot(B2, y, "o", ms=5, label="simulated $n_*$")
    grid = np.linspace(0, B2.max() * 1.05, 50)
    ax.plot(grid, coef[0] + coef[1] * grid, "-", lw=1.3,
            label=r"fit: slope $%.2f$ ($R^2=%.3f$)" % (coef[1], r2))
    ax.plot(grid, y[0] + 0.5 * (grid - B2[0]), "--", lw=1,
            label=r"slope $1/2$   ($e^{B^2/2}$, Hessian lower bd.)")
    ax.plot(grid, y[0] + 1.0 * (grid - B2[0]), ":", lw=1,
            label=r"slope $1$   ($e^{B^2}$, Hessian upper bd.)")
    ax.set_xlabel(r"$B^2=\|\beta_*\|^2$")
    ax.set_ylabel(r"$\log n_*$")
    ax.set_title(r"Threshold is exponential in $B^2$", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_threshold.pdf", bbox_inches="tight")
    plt.close(fig)
    return coef, r2


def fig_wilks(rng, quick):
    """Empirical law of 2n(L(bhat)-L(b*)) against chi^2(d)."""
    settings = ([(5, 1.0, 4000), (5, 2.0, 20000)] if quick else
                [(5, 1.0, 4000), (5, 2.0, 20000), (10, 1.0, 8000)])
    n_rep = 200 if quick else 800

    print("\n=== Figure 4: Wilks ===")
    fig, axes = plt.subplots(1, len(settings), figsize=(3.0 * len(settings), 2.6))
    for ax, (d, bn, n) in zip(np.atleast_1d(axes), settings):
        v = wilks_sample(n, d, bn, n_rep, rng)
        ks = stats.kstest(v, "chi2", args=(d,))
        ax.hist(v, bins=30, density=True, alpha=0.55, edgecolor="none")
        xs = np.linspace(1e-3, max(v.max(), 3 * d), 300)
        ax.plot(xs, stats.chi2.pdf(xs, d), "k-", lw=1.3, label=r"$\chi^2(%d)$" % d)
        ax.set_title(r"$d=%d,\ \|\beta_*\|=%.1f,\ n=%d$" % (d, bn, n), fontsize=8)
        ax.set_xlabel(r"$2n(L(\hat\beta_n)-L(\beta_*))$")
        ax.legend(frameon=False)
        print(f"  d={d} |b*|={bn} n={n}: mean {v.mean():.2f} (vs {d}), "
              f"KS p={ks.pvalue:.3f}")
    np.atleast_1d(axes)[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_wilks.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    fig_existence(rng, args.quick)
    fig_risk(rng, args.quick)
    fig_threshold(rng, args.quick)
    fig_wilks(rng, args.quick)

    print(f"\nDone in {time.time() - t0:.1f}s. Figures in ./{OUT}/")


if __name__ == "__main__":
    main()

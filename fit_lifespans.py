"""
fit_lifespans.py
================

Fit exponential, power-law, lognormal, and truncated-power-law distributions
to polity lifespans in the Cliopatria dataset. Replicates the analysis
described in README.md.

Usage
-----
Run from the directory containing this file, after cloning and unzipping
Cliopatria into a sibling `cliopatria/` directory:

    git clone https://github.com/Seshat-Global-History-Databank/cliopatria.git
    cd cliopatria && unzip -o cliopatria.geojson.zip && cd ..
    python fit_lifespans.py

Outputs go to ./results/ and ./figures/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from scipy import stats
from scipy.stats import lognorm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
GEOJSON = HERE / "cliopatria" / "cliopatria_polities_only.geojson"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: load and aggregate to one row per polity
# ---------------------------------------------------------------------------
def load_lifespans(geojson_path: Path) -> pd.DataFrame:
    """Aggregate Cliopatria's per-time-slice rows to one row per polity.

    Cliopatria stores polity geometries as multiple rows when the geometry
    changes across time. We collapse on `Name` and take the full temporal
    envelope: lifespan = max(ToYear) - min(FromYear) + 1 (inclusive).
    """
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"Cliopatria GeoJSON not found at {geojson_path}.\n"
            "Run:\n"
            "  git clone https://github.com/Seshat-Global-History-Databank/cliopatria.git\n"
            "  cd cliopatria && unzip -o cliopatria.geojson.zip && cd .."
        )

    print(f"Loading {geojson_path.name} ...")
    gdf = gpd.read_file(geojson_path)
    print(f"  {len(gdf):,} rows, {gdf['Name'].nunique():,} unique polity names")

    agg = (
        gdf.groupby("Name")
        .agg(
            from_year=("FromYear", "min"),
            to_year=("ToYear", "max"),
            n_rows=("Name", "size"),
        )
        .reset_index()
    )
    agg["lifespan"] = agg["to_year"] - agg["from_year"] + 1
    assert (agg["lifespan"] > 0).all(), "negative or zero lifespan detected"
    return agg


# ---------------------------------------------------------------------------
# Step 2: fit distributions
# ---------------------------------------------------------------------------
def fit_all(x: np.ndarray) -> dict:
    """Fit exponential, pure PL, truncated PL, and lognormal; compare them."""
    n = len(x)
    out: dict = {
        "n_polities": int(n),
        "mean_lifespan": float(x.mean()),
        "median_lifespan": float(np.median(x)),
        "max_lifespan": int(x.max()),
    }

    # --- Exponential on the full distribution (Arbesman's choice) -----------
    lam_full = float(x.mean())
    ll_exp_full = float(np.sum(stats.expon.logpdf(x, scale=lam_full)))
    ks_full = stats.kstest(x, "expon", args=(0, lam_full))
    out["exponential_full"] = {
        "scale_lambda": lam_full,
        "log_likelihood": ll_exp_full,
        "AIC": 2 * 1 - 2 * ll_exp_full,
        "BIC": 1 * np.log(n) - 2 * ll_exp_full,
        "KS_statistic": float(ks_full.statistic),
        "KS_pvalue": float(ks_full.pvalue),
    }

    # --- powerlaw package: discrete fits with KS-minimized x_min ------------
    fit = powerlaw.Fit(x.astype(int), discrete=True, verbose=False)
    xmin = float(fit.xmin)
    alpha = float(fit.alpha)
    n_tail = int((x >= xmin).sum())

    out["pure_power_law"] = {
        "xmin": xmin,
        "alpha": alpha,
        "n_tail": n_tail,
        "KS_statistic": float(fit.D),
    }

    out["lognormal"] = {
        "mu": float(fit.lognormal.mu),
        "sigma": float(fit.lognormal.sigma),
        "median_implied": float(np.exp(fit.lognormal.mu)),
        "mean_implied": float(np.exp(fit.lognormal.mu + 0.5 * fit.lognormal.sigma**2)),
    }

    out["truncated_power_law"] = {
        "alpha": float(fit.truncated_power_law.alpha),
        "Lambda": float(fit.truncated_power_law.Lambda),
        "cutoff_scale": float(1 / fit.truncated_power_law.Lambda),
    }

    # --- Shifted exponential on tail (apples-to-apples vs PL on tail) -------
    tail = x[x >= xmin]
    lam_tail = float(tail.mean() - xmin)
    ll_exp_tail = float(np.sum(-np.log(lam_tail) - (tail - xmin) / lam_tail))
    out["exponential_tail"] = {
        "xmin": xmin,
        "scale_lambda": lam_tail,
        "log_likelihood_tail": ll_exp_tail,
    }

    # --- Vuong-style LR comparisons ----------------------------------------
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_tpl, p_tpl = fit.distribution_compare("power_law", "truncated_power_law", normalized_ratio=True)
    out["likelihood_ratio_tests"] = {
        "PL_vs_exponential": {"R": float(R_exp), "p": float(p_exp)},
        "PL_vs_lognormal": {"R": float(R_ln), "p": float(p_ln)},
        "PL_vs_truncated_PL": {"R": float(R_tpl), "p": float(p_tpl)},
        "note": "R > 0 favors the first model (power law). p < 0.05 is significant.",
    }

    return out, fit


# ---------------------------------------------------------------------------
# Step 3: bootstrap goodness-of-fit (Clauset et al. 2009)
# ---------------------------------------------------------------------------
def bootstrap_gof_powerlaw(x: np.ndarray, fit: powerlaw.Fit, n_boot: int = 500,
                           seed: int = 42) -> float:
    """Parametric bootstrap p-value for the fitted power law.

    Generate synthetic datasets matching the empirical body below x_min and
    a fresh power-law tail above. Refit each, record KS distance, return
    the fraction with KS >= observed.
    """
    rng = np.random.default_rng(seed)
    xmin = fit.xmin
    alpha = fit.alpha
    n_total = len(x)
    n_tail = int((x >= xmin).sum())
    p_tail = n_tail / n_total
    body = x[x < xmin]
    D_obs = fit.D

    D_sims = []
    for _ in range(n_boot):
        n1 = rng.binomial(n_total, p_tail)
        n2 = n_total - n1
        u = rng.uniform(size=n1)
        pl_samples = xmin * (1 - u) ** (-1 / (alpha - 1))
        body_samples = (
            rng.choice(body, size=n2, replace=True) if len(body) else np.array([])
        )
        sim = np.concatenate([pl_samples, body_samples])
        try:
            f2 = powerlaw.Fit(sim, discrete=False, verbose=False)
            D_sims.append(f2.D)
        except Exception:
            continue
    return float(np.mean(np.array(D_sims) >= D_obs))


def bootstrap_gof_exponential_tail(x: np.ndarray, xmin: float, n_boot: int = 500,
                                   seed: int = 42) -> float:
    """Parametric bootstrap p-value for shifted exponential on the tail."""
    rng = np.random.default_rng(seed)
    tail = x[x >= xmin]
    n_t = len(tail)
    lam = tail.mean() - xmin
    emp = np.sort(tail)
    ecdf = np.arange(1, n_t + 1) / n_t
    tcdf = 1 - np.exp(-(emp - xmin) / lam)
    D_obs = float(np.max(np.abs(ecdf - tcdf)))

    D_sims = []
    for _ in range(n_boot):
        sim_t = xmin + rng.exponential(scale=lam, size=n_t)
        lam_s = sim_t.mean() - xmin
        emp_s = np.sort(sim_t)
        tcdf_s = 1 - np.exp(-(emp_s - xmin) / lam_s)
        D_sims.append(np.max(np.abs(ecdf - tcdf_s)))
    return float(np.mean(np.array(D_sims) >= D_obs))


# ---------------------------------------------------------------------------
# Step 4: diagnostic plots
# ---------------------------------------------------------------------------
def make_plots(x: np.ndarray, fit: powerlaw.Fit, out_path: Path) -> None:
    sx = np.sort(x)
    ccdf = 1 - np.arange(len(sx)) / len(sx)
    xmin = fit.xmin
    alpha = fit.alpha
    n_tail = (x >= xmin).sum()
    norm = n_tail / len(x)
    lam_full = x.mean()
    mu_ln, sig_ln = fit.lognormal.mu, fit.lognormal.sigma
    tpla, tpll = fit.truncated_power_law.alpha, fit.truncated_power_law.Lambda
    tail = x[x >= xmin]
    lam_t = tail.mean() - xmin

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # log-log CCDF
    ax = axes[0]
    ax.loglog(sx, ccdf, "o", ms=3, color="steelblue", alpha=0.5, label="Empirical CCDF")

    xt = np.logspace(np.log10(xmin), np.log10(sx.max()), 200)
    ax.loglog(xt, (xt / xmin) ** (1 - alpha) * norm, "r-", lw=2,
              label=f"Power law (α={alpha:.2f}, x_min={int(xmin)})")

    xt_tpl = np.unique(x[x >= xmin]).astype(float)
    ax.loglog(xt_tpl, fit.truncated_power_law.ccdf(xt_tpl) * norm, "g--", lw=2,
              label=f"Truncated PL (α={tpla:.2f}, cutoff~{1/tpll:.0f})")

    xt_ln = np.logspace(0, np.log10(sx.max()), 300)
    ccdf_ln = 1 - lognorm.cdf(xt_ln, sig_ln, scale=np.exp(mu_ln))
    ax.loglog(xt_ln, ccdf_ln, "m:", lw=2,
              label=f"Lognormal (μ={mu_ln:.2f}, σ={sig_ln:.2f})")

    ax.loglog(xt_ln, np.exp(-xt_ln / lam_full), color="orange", lw=2, ls="-.",
              label=f"Exp full (λ={lam_full:.0f})")

    ax.set_xlabel("Lifespan (years)")
    ax.set_ylabel("CCDF P(X ≥ x)")
    ax.set_title("Cliopatria polity lifespans: log-log survival")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    # semi-log: straight line ⇔ exponential
    ax = axes[1]
    ax.semilogy(sx, ccdf, "o", ms=3, alpha=0.5, color="steelblue", label="Empirical")
    ax.semilogy(sx, np.exp(-sx / lam_full), "r-", lw=2,
                label=f"Exp full (λ={lam_full:.0f})")
    xt3 = np.linspace(xmin, sx.max(), 200)
    ax.semilogy(xt3, np.exp(-(xt3 - xmin) / lam_t) * norm, "r--", lw=2,
                label=f"Exp on tail (λ={lam_t:.0f})")
    ax.set_xlabel("Lifespan (years)")
    ax.set_ylabel("CCDF (log)")
    ax.set_title("Semi-log: straight line ⇔ exponential")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_lifespans(GEOJSON)
    df.to_csv(RESULTS / "lifespans.csv", index=False)
    print(f"  wrote {RESULTS / 'lifespans.csv'}")

    x = df["lifespan"].values.astype(int)
    print(f"\nFitting distributions to {len(x):,} polity lifespans ...")
    summary, fit = fit_all(x)

    print("\nBootstrapping GoF p-values (n_boot=500, ~30s) ...")
    p_pl = bootstrap_gof_powerlaw(x, fit, n_boot=500)
    p_exp = bootstrap_gof_exponential_tail(x, fit.xmin, n_boot=500)
    summary["bootstrap_gof"] = {
        "pure_power_law_p": p_pl,
        "shifted_exp_on_tail_p": p_exp,
        "note": "p > 0.1 ⇒ model is plausible (Clauset, Shalizi, Newman 2009).",
    }

    with open(RESULTS / "fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {RESULTS / 'fit_summary.json'}")

    print("\nMaking plots ...")
    make_plots(x, fit, FIGURES / "cliopatria_lifespan_fits.png")
    print(f"  wrote {FIGURES / 'cliopatria_lifespan_fits.png'}")

    # ---- short stdout summary ------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"N = {summary['n_polities']:,} polities, "
          f"mean = {summary['mean_lifespan']:.1f} yr, "
          f"median = {summary['median_lifespan']:.1f} yr, "
          f"max = {summary['max_lifespan']} yr")
    print()
    e = summary["exponential_full"]
    print(f"Exponential, full distribution:  λ = {e['scale_lambda']:.1f} yr, "
          f"KS = {e['KS_statistic']:.4f}, p = {e['KS_pvalue']:.2e}")
    p = summary["pure_power_law"]
    print(f"Pure power law on tail:          x_min = {p['xmin']:.0f}, "
          f"α = {p['alpha']:.3f}, n_tail = {p['n_tail']}, KS = {p['KS_statistic']:.4f}")
    t = summary["truncated_power_law"]
    print(f"Truncated power law:             α = {t['alpha']:.3f}, "
          f"cutoff scale = {t['cutoff_scale']:.1f} yr")
    L = summary["lognormal"]
    print(f"Lognormal:                       μ = {L['mu']:.3f}, σ = {L['sigma']:.3f}, "
          f"median_implied = {L['median_implied']:.1f} yr")
    print()
    lr = summary["likelihood_ratio_tests"]
    for name, d in lr.items():
        if isinstance(d, dict):
            print(f"  {name:25s}: R = {d['R']:+.3f}, p = {d['p']:.4f}")
    print()
    g = summary["bootstrap_gof"]
    print(f"Bootstrap GoF, pure power law:   p = {g['pure_power_law_p']:.3f}")
    print(f"Bootstrap GoF, exp on tail:      p = {g['shifted_exp_on_tail_p']:.3f}")


if __name__ == "__main__":
    main()

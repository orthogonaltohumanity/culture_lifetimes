# Cliopatria polity-lifespan distribution fits

Replication code for fitting exponential, power-law, lognormal, and truncated
power-law distributions to polity lifespans in the
[Cliopatria](https://github.com/Seshat-Global-History-Databank/cliopatria)
geospatial dataset (Seshat Global History Databank, 1,618 polities,
3400 BCE – 2024 CE).

This is a re-examination of Arbesman (2011), "The Life-Spans of Empires,"
*Historical Methods* 44(3), which fitted an exponential distribution to
N=41 large empires from Taagepera's dataset and concluded that empire
lifetimes are memoryless. With N=1,618 we can distinguish models that are
indistinguishable at N=41.

## Headline result

| Comparison | normalized LR | p | Winner |
|---|---|---|---|
| Power law vs exponential (tail) | −0.06 | 0.95 | tie |
| Power law vs lognormal | −2.63 | 0.009 | **lognormal** |
| Power law vs truncated power law | −2.84 | 0.005 | **truncated PL** |

Pure exponential (Arbesman's choice) is decisively rejected on the full
distribution: KS = 0.092, p ≈ 2 × 10⁻¹². Lognormal and truncated power
law both beat the pure power law that competes with it on the tail.
Tail-only exponential and tail-only power law are statistically
indistinguishable, which means Arbesman's claim is observationally
equivalent to a power law on his sample — a known small-N artifact
(Mitzenmacher 2004).

## Setup

```bash
# 1. Clone Cliopatria into this directory
git clone https://github.com/Seshat-Global-History-Databank/cliopatria.git

# 2. Unzip the GeoJSON
cd cliopatria && unzip -o cliopatria.geojson.zip && cd ..

# 3. Install dependencies (or use the provided venv setup)
pip install -r requirements.txt
```

## Run

```bash
python fit_lifespans.py
```

Outputs:
- `results/lifespans.csv` — one row per polity (Name, FromYear, ToYear, lifespan)
- `results/fit_summary.json` — all parameter estimates, KS stats, LR tests
- `figures/cliopatria_lifespan_fits.png` — log-log CCDF + semi-log diagnostic

Run takes ~30 s on a laptop (most of it loading the 207 MB GeoJSON).

Reference outputs from a clean run are checked in under `results_reference/`
and `figures_reference/` — your run should produce numerically identical
files (the bootstrap p-values use a fixed seed of 42).

## Method notes

1. **Aggregation.** Cliopatria stores one row per (polity, time-slice) — geometry
   changes are recorded as separate rows. We collapse to one row per `Name`,
   defining `lifespan = max(ToYear) - min(FromYear) + 1` (inclusive years).

2. **Discrete data.** Years are integers; we pass `discrete=True` to the
   `powerlaw` package's MLE. This matters for the likelihood values but
   not for the qualitative ranking of the four candidates.

3. **`x_min` selection.** Following Clauset, Shalizi & Newman (2009),
   we choose the lower bound of the power-law regime by minimizing the
   KS distance between empirical and theoretical CDF on data above `x_min`.

4. **Bootstrap GoF.** The parametric bootstrap p-value is computed by
   simulating 500 datasets from the fitted model (mixing the fitted PL
   above `x_min` with resampled body data below) and asking how often the
   simulated KS distance exceeds the observed. p > 0.1 is the conventional
   threshold to consider the model plausible.

5. **Likelihood-ratio comparison.** Vuong-style normalized LR test. R is
   the sum of pointwise log-likelihood differences; positive R favors the
   first model. Two-sided p-value via z = R/σ.

## Caveats

- **Right-censoring at 2024 CE.** ~200 polities are still extant; their
  lifespans are lower bounds. Proper treatment would use censored MLE
  (e.g. via `lifelines`). This typically *strengthens* heavy-tail evidence.
- **Left-censoring at 3400 BCE.** Sumerian City-States and Elam are
  truncated at the dataset's start year.
- **Polity individuation.** Cliopatria's `Name` field has some
  near-duplicates (e.g. "Kingdom of Italy" vs "(Kingdom of Italy)").
  Splitting/merging events also make "lifespan of an entity" ontologically
  slippery — the Roman Empire → ERE/WRE transition is a concrete example.

## References

- Arbesman, S. (2011). The Life-Spans of Empires. *Historical Methods* 44(3).
- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
  distributions in empirical data. *SIAM Review* 51(4).
- Mitzenmacher, M. (2004). A brief history of generative models for power
  law and lognormal distributions. *Internet Mathematics* 1(2).
- Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: A Python package
  for analysis of heavy-tailed distributions. *PLoS ONE* 9(1).
- Cliopatria: Seshat Global History Databank.
  https://github.com/Seshat-Global-History-Databank/cliopatria

# Improved R-hat and ESS (Vehtari et al. 2021) and how `tree_ensembles.bart` uses them

Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021). "Rank-normalization, folding, and
localization: An improved R̂ for assessing convergence of MCMC". *Bayesian Analysis* 16(2),
667–718. Stan and ArviZ use the same definitions; `tree_ensembles.bart.diagnostics` follows
them too (cross-checked against ArviZ: R̂ equal to 3 decimals, ESS within about 1%).

## The problem with the classic R̂

Classic R̂ (Gelman and Rubin 1992) runs several chains and compares the variance **between**
chains with the variance **within** them; if all chains explore the same distribution, R̂ ≈ 1.
Because it only looks at means and variances, it misses real problems:

- chains with the **same mean but a different spread** (one chain stuck in the center, another
  wandering in the tails): the means agree, so R̂ ≈ 1;
- **heavy tails** (e.g. Cauchy-like posteriors): with a huge or infinite variance the
  variance comparison means nothing;
- a chain that is **still drifting** can have an average that looks fine;
- the old threshold, R̂ < 1.1, is far too loose.

## The paper's fixes

1. **Split each chain in half**, so a drift within a chain shows up as a difference between
   halves (already in the textbook "split R̂").
2. **Rank-normalize**: pool all draws, replace each by its rank r, then by a normal score
   z = Φ⁻¹((r − 3/8) / (S + 1/4)), with S the total number of draws. R̂ on these z is robust to
   heavy tails and the same for θ or log θ. This is the **bulk R̂**.
3. **Fold**: compute the same on |θ − median(θ)| (rank-normalized), which compares how
   **spread out** the chains are. The final **R̂ = max(bulk R̂, folded R̂)**.
4. **Two effective sample sizes (ESS)**:
   - **bulk ESS**: ESS of the rank-normalized split chains. How precisely the mean / median
     are estimated.
   - **tail ESS**: the smaller ESS of the indicators I(θ ≤ 5% quantile) and
     I(θ ≤ 95% quantile). How precisely the tails, and so the interval endpoints, are
     estimated. This is the one that matters for credible intervals.

   Both combine the chains' autocorrelations into one estimate and sum them with Geyer's
   initial monotone sequence (stop when pairs of autocorrelations turn negative, and force
   them to decrease), which cuts off the noisy long lags.
5. **Stricter thresholds**: R̂ < 1.01, and bulk and tail ESS > 400 (e.g. 4 chains × 100).
6. **Rank plots** instead of trace plots: for each chain, a histogram of its draws' ranks
   among all draws. Mixing chains give flat histograms; a stuck chain stands out.
7. **Localization**: ESS and Monte Carlo standard error (MCSE) for any quantile, and
   efficiency plots ("localization" in the title).

## Where each piece lives in `tree_ensembles.bart`

| Paper | Code |
|---|---|
| Split chains | `diagnostics._split_chains` |
| Rank normalization, (r − 3/8)/(S + 1/4) | `diagnostics._rank_normalize` |
| Between / within-chain variance (classic R̂ formula) | `diagnostics._rhat` |
| R̂ = max(bulk, folded) | `diagnostics.split_rhat` |
| Multi-chain ESS, autocorrelation via FFT, Geyer's monotone sequence | `diagnostics._ess` |
| Bulk ESS | `diagnostics.ess_bulk` |
| Tail ESS (5% and 95% indicators) | `diagnostics.ess_tail` |
| R̂ < 1.01, ESS > 400 | `summarize_draws(..., rhat_max=1.01, ess_min=400)` → column `ok` |
| Rank plots | `plots.plot_rank` |
| Localization (quantile ESS / MCSE, efficiency plots) | not implemented |

What `diagnostics()` adds for BART: the paper applies to any scalar MCMC quantity, and the
estimators check these:

- `sigma` (regressor), the noise sd;
- `mean_tree_leaves`, `mean_tree_depth`: tree size per draw, which mixes slowly in BART and so
  is a sensitive warning sign;
- `f(x_probe[i])`: the prediction at 20 training rows (or rows you pass as `X_probe`), which
  checks that the **predictions and their intervals** have converged, not only global
  parameters.
  By default they are summarized in two rows, `f(x_probe): median` and `f(x_probe): worst`
  (median and largest R̂, median and smallest ESS), plus the share of points passing;
  `diagnostics(per_point=True)` shows every point. f(x) at single rows mixes far more slowly
  than σ in BART, so this is where large models usually fail; more trees help (on
  cooking-time, 50 → 200 → 800 trees moved the median R̂ of f(x) from 1.56 to 1.24 to 1.10).

It also reports the average acceptance rate (`accept_rate`: share of trees with an accepted
grow / prune move per iteration) and `leaf_fill` (mean leaves per tree out of the
2^(maxdepth − 1) allowed, bartz's "leaves 2.6/32", with a warning above 0.25). These describe
the sampler and the model and are not checked for convergence.

On the cooking-time data (20k rows, 4 chains), sigma was nearly fine but tree size and f(x)
had R̂ up to about 1.5 and bulk ESS below 20, even with longer chains. The predictions were
still accurate, so the practical check was the coverage of the predictive intervals on
held-out data.

Small differences from the paper:

- `ess_min=400` is a total, whatever the number of chains; the paper's 400 assumes about four
  chains (100 per chain). Pass `ess_min` to adjust.
- The Geyer cut-off is a simplified version of Stan's; ESS stays within about 1% of ArviZ.

## Why it matters: classic vs improved R̂

Two failure cases from the paper and one healthy case (4 chains × 1,000 draws each):

| Case | Classic split R̂ | Rank-normalized R̂ | Bulk ESS | Tail ESS |
|---|---|---|---|---|
| Same mean, one chain 3× wider | 1.000 | **1.152** | 3,900 | **32** |
| One chain stuck in a tail (Cauchy) | 1.001 | **1.331** | **10** | 213 |
| All chains fine (normal) | 1.001 | 1.001 | 4,051 | 3,929 |

Classic split R̂ passes both broken cases even at the strict 1.01; the improved R̂ and the
tail ESS catch them, and all agree on the healthy chains. To reproduce:

```python
import numpy as np
from tree_ensembles.bart.diagnostics import _rhat, _split_chains, split_rhat, ess_bulk, ess_tail

rng = np.random.default_rng(0)
cases = {
    "same mean, one chain 3x wider": np.vstack(
        [rng.normal(0, 1, (3, 1000)), rng.normal(0, 3, (1, 1000))]
    ),
    "one chain stuck in a tail": np.vstack(
        [rng.standard_cauchy((3, 1000)), 5 + np.abs(rng.standard_cauchy((1, 1000)))]
    ),
    "all chains fine": rng.normal(size=(4, 1000)),
}
for name, x in cases.items():
    classic = _rhat(_split_chains(x))  # = numpyro.diagnostics.split_gelman_rubin
    print(name, round(classic, 3), round(split_rhat(x), 3), round(ess_bulk(x)), round(ess_tail(x)))
```

## Does NumPyro implement this?

Not the 2021 version (checked in the NumPyro 0.22.0 source, `numpyro/diagnostics.py`):

| | NumPyro | Vehtari et al. 2021 / Stan / ArviZ / `tree_ensembles.bart` |
|---|---|---|
| R̂ | `split_gelman_rubin`: split chains, **no** rank normalization, **no** folding | rank-normalized, max of bulk and folded |
| ESS | `effective_sample_size`: multi-chain, Geyer's sequence, on the **raw** draws, not split | **bulk** ESS on rank-normalized split chains |
| Tail ESS | **no** | yes (5% / 95% indicators) |
| Quantile ESS / MCSE | no | in ArviZ; not in `tree_ensembles.bart` |
| Summary | `summary` / `print_summary`: mean, sd, median, **90%** HPDI, `n_eff`, `r_hat` | `posterior_summary`: mean, sd, **95%** HPDI, `rhat`, `ess_bulk`, `ess_tail`, `ok` |

So `numpyro.diagnostics.split_gelman_rubin` is the "classic split R̂" column above. To get the
2021 diagnostics from NumPyro samples, pass them to ArviZ (`arviz.rhat`, `arviz.ess` with
`method="bulk"` / `"tail"`) or to `tree_ensembles.bart.diagnostics` (`split_rhat`, `ess_bulk`,
`ess_tail`, `summarize_draws`), which take draws shaped (chain, draw).

## References

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of
  MCMC. *Bayesian Analysis* 16(2), 667–718.
- Gelman, A. and Rubin, D. B. (1992). Inference from iterative simulation using multiple
  sequences. *Statistical Science* 7(4), 457–472.
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science* 7(4), 473–483.

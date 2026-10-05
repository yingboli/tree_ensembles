"""BART (Bayesian additive regression trees) via bartz, with posterior tools.

Needs the optional extra: pip install "tree_ensembles[bart]".
"""

from tree_ensembles.bart.diagnostics import ess_bulk, ess_tail, split_rhat, summarize_draws
from tree_ensembles.bart.estimators import BartClassifier, BartRegressor
from tree_ensembles.bart.intervals import hpdi, quantile_interval
from tree_ensembles.bart.missing import MissingValueImputer

__all__ = [
    "BartClassifier",
    "BartRegressor",
    "MissingValueImputer",
    "ess_bulk",
    "ess_tail",
    "hpdi",
    "quantile_interval",
    "split_rhat",
    "summarize_draws",
]

"""BART regressor/classifier: scikit-learn wrappers around bartz with posterior tools.

Beyond fit / predict / predict_proba, each fitted model gives:
    predict_samples    posterior draws of f(x) (or p(x)), or of new outcomes
    predict_dist       posterior mean, sd of f(x), and posterior predictive sd
    predict_interval   HPDI (or equal-tailed) credible / predictive intervals
    posterior_summary  mean, sd, HPDI, R-hat and ESS of the model's parameters
    diagnostics        MCMC convergence checks, including f(x) at probe points
    forest_summary     depth, leaves and variable usage of the sampled trees

Notes:
    - The classifier is probit BART: p(x) = Phi(sum of trees).
    - bartz has no sample weights or per-row offsets (not supported here yet).
    - dump/load use bartz's dump, a pickle tied to the bartz/jax/equinox versions:
      fine for caching, not for long-term archival.
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import pandas as pd
from bartz import Bart
from sklearn.base import ClassifierMixin, RegressorMixin

from tree_ensembles._base import KwargsEstimator
from tree_ensembles.bart import trees
from tree_ensembles.bart.diagnostics import summarize_draws
from tree_ensembles.bart.intervals import hpdi, quantile_interval
from tree_ensembles.bart.missing import MissingValueImputer
from tree_ensembles.bart.trees import trees_to_dataframe as _trees_table

ArrayLike = pd.DataFrame | np.ndarray


def _row_chunks(X: ArrayLike, batch_size: int) -> list[ArrayLike]:
    """Split X into consecutive batches of at most `batch_size` rows.

    Only to limit memory: per-row results never depend on the other rows in a batch.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    starts = range(0, len(X), batch_size)
    if isinstance(X, pd.DataFrame):
        return [X.iloc[i : i + batch_size] for i in starts]
    return [X[i : i + batch_size] for i in starts]


class _BartBase(KwargsEstimator):
    """Shared logic of BartRegressor and BartClassifier (use those classes).

    Parameters
    ----------
    num_trees : int, default 200
        Number of trees in the sum. Each tree is kept small by the prior (see `power`, `base`).
    n_save : int, default 1000
        Posterior draws kept per chain, after burn-in. The total is num_chains * n_save.
    n_burn : int, default 1000
        MCMC iterations discarded at the start of every chain.
    n_skip : int, default 1
        Thinning: keep every n_skip-th iteration after burn-in. Each chain runs
        n_burn + n_skip * n_save iterations and still keeps n_save draws; e.g. n_skip=3 with
        n_save=1000 keeps 1,000 draws taken from 3,000 iterations after burn-in: less
        autocorrelated draws, but the post-burn-in part takes three times as long.
    num_chains : int, default 4
        Independent MCMC chains. At least 2 are needed for a meaningful R-hat. On CPU, call
        jax.config.update("jax_num_cpu_devices", num_chains) before fitting to run them in
        parallel.
    maxdepth : int, default 6
        Maximum tree depth, counting levels of nodes from 1 (bartz's convention): with 6,
        nodes sit at depths 0 to 5, so a path has at most 5 splits and a tree at most 32
        leaves. XGBoost's max_depth counts levels of splits instead, so bartz maxdepth =
        XGBoost max_depth + 1. The defaults are not equivalent: XGBoost's max_depth=6
        matches bartz maxdepth=7, and BART's default maxdepth=6 matches XGBoost
        max_depth=5. (Depths reported by forest_summary() count levels of splits, like
        XGBoost.)
    power, base : float, default 2.0 and 0.95
        Tree prior: a node at depth d (root d = 0) splits with probability
        base / (1 + d) ** power. Smaller `base` or larger `power` give smaller trees.
    random_state : int, default 0
        Seed of the MCMC, of the 20 probe rows used by diagnostics(), and of the outcome draws
        in predict_samples(kind="predictive").
    show_progress : bool, default True
        Show bartz's progress bar while fitting.
    impute_strategy : {"median", "mean"} or None, default "median"
        How NaN in X are filled before bartz sees the data (see missing.MissingValueImputer).
        Fill values are learned on the training rows and reused for new rows. None passes
        NaN to bartz unchanged, which bins them poorly (see trees.py; fit then warns).
    missing_indicator_threshold : float, default 0.5
        A feature whose training missing rate is above this gets an extra 0/1 column
        `<name>_missing`, so the trees can split on whether it is missing. 0 adds one for
        every feature with NaN. Ignored when impute_strategy is None.
    **bartz_params
        Any other argument of bartz.Bart, e.g. k=2.0, sigma_df=3.0 or
        sparse=SparseConfig(...). They work with get_params, set_params, clone and
        GridSearchCV like the named parameters. `outcome_type` (set by the class), `seed`
        (use random_state) and `printevery` / `pbar` (use show_progress) raise a ValueError.

    Attributes
    ----------
    bart_ : bartz.Bart
        The fitted bartz model.
    feature_names_in_ : list of str
        Column names of the training DataFrame, or x0, x1, ... for an array.
    n_features_in_ : int
        Number of features in X.
    imputer_ : MissingValueImputer or None
        The fitted missing-value step (None when impute_strategy is None).
    model_feature_names_ : list of str
        Names of the columns bartz is fit on: feature_names_in_ plus the `<name>_missing`
        indicators. The tree summaries (variable_usage, split_points, format_tree) use these.
    fitted_with_names_ : bool
        Whether fit got a DataFrame; if so, later DataFrames must have the same columns in
        the same order.
    X_probe_ : ndarray of shape (n_probe, p)
        A copy of n_probe random training rows (fit argument, 20 by default; as given, before
        imputation) where diagnostics() checks f(x). They are not held out: the model is fit
        on all training rows.
    classes_ : ndarray of shape (2,)
        Classifier only: the two class labels; predict_proba columns follow this order.

    Notes
    -----
    Uses bartz (https://github.com/bartz-org/bartz). Not supported yet: sample weights and
    per-row offsets. NaN in X are imputed by default (impute_strategy), a workaround for how
    bartz bins NaN (see trees.py).
    """

    _kwargs_attr = "bartz_params"  # where KwargsEstimator keeps the **bartz_params
    _outcome_type: str  # bartz outcome_type, set by each subclass

    def __init__(
        self,
        num_trees: int = 200,
        n_save: int = 1000,
        n_burn: int = 1000,
        n_skip: int = 1,
        num_chains: int = 4,
        maxdepth: int = 6,
        power: float = 2.0,
        base: float = 0.95,
        random_state: int = 0,
        show_progress: bool = True,
        impute_strategy: Literal["median", "mean"] | None = "median",
        missing_indicator_threshold: float = 0.5,
        **bartz_params: Any,
    ) -> None:
        self.num_trees = num_trees
        self.n_save = n_save
        self.n_burn = n_burn
        self.n_skip = n_skip
        self.num_chains = num_chains
        self.maxdepth = maxdepth
        self.power = power
        self.base = base
        self.random_state = random_state
        self.show_progress = show_progress
        self.impute_strategy = impute_strategy
        self.missing_indicator_threshold = missing_indicator_threshold
        self.bartz_params = bartz_params

    # --- hooks implemented by the subclasses ---------------------------------

    def _prepare_y(self, y: Any) -> np.ndarray:
        """Check and encode the training target as float32 (0/1 for the classifier)."""
        raise NotImplementedError

    def _predictive_var(self, mean_draws: np.ndarray) -> np.ndarray:
        """Posterior predictive variance of a new outcome, from draws of its mean."""
        raise NotImplementedError

    def _parameter_draws(self) -> dict[str, np.ndarray]:
        """Model-specific scalar parameters, each shaped (chain, draw)."""
        return {}

    # --- input handling ------------------------------------------------------

    def _check_X(self, X: ArrayLike, fitting: bool = False) -> np.ndarray:
        """X as float32 (n, p).

        When fitting, remember the feature names (x0, x1, ... for arrays). Afterwards, a
        DataFrame must have the same columns in the same order as the training DataFrame.
        """
        columns = [str(c) for c in X.columns] if isinstance(X, pd.DataFrame) else None
        if fitting:
            self.fitted_with_names_ = columns is not None
            self.feature_names_in_ = columns or [f"x{j}" for j in range(np.shape(X)[1])]
            self.n_features_in_ = len(self.feature_names_in_)
        elif columns is not None and self.fitted_with_names_:
            if columns != self.feature_names_in_:
                missing = sorted(set(self.feature_names_in_) - set(columns))
                extra = sorted(set(columns) - set(self.feature_names_in_))
                raise ValueError(
                    "X columns must match the training columns in the same order; "
                    f"missing: {missing}, unexpected: {extra}"
                    + ("" if missing or extra else " (same names, different order)")
                )
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_}), got {X_arr.shape}")
        return X_arr

    def _model_X(self, X_arr: np.ndarray) -> np.ndarray:
        """The columns bartz sees: X with NaN imputed and missing indicators appended."""
        return X_arr if self.imputer_ is None else self.imputer_.transform(X_arr)

    # --- fitting -------------------------------------------------------------

    def fit(self, X: ArrayLike, y: Any, n_probe: int = 20) -> "_BartBase":
        """Run the MCMC on all of (X, y).

        Parameters
        ----------
        X : DataFrame or array of shape (n, p)
            Training features (numeric; encode categories as numbers or dummies).
        y : array-like of shape (n,)
            Training target; two classes for the classifier.
        n_probe : int, default 20
            Number of random training rows to keep a copy of (X_probe_), where
            diagnostics() checks the convergence of f(x). They are not held out: the model
            is fit on all rows.

        Returns
        -------
        self

        Notes
        -----
        JAX runs asynchronously, so fit can return before the MCMC has finished; the first
        prediction then waits for it.
        """
        X_arr = self._check_X(X, fitting=True)
        y_arr = self._prepare_y(y)
        # Arguments of bartz.Bart this class sets that are not named arguments above.
        extra = dict(self.bartz_params)
        reserved = {"outcome_type", "seed", "printevery", "pbar"} & extra.keys()
        if reserved:
            raise ValueError(
                f"{sorted(reserved)} are set by this class and cannot be passed "
                "(use random_state for seed, show_progress for printevery / pbar)"
            )

        # Workaround for bartz's NaN binning (see missing.py): impute + indicators.
        self.imputer_: MissingValueImputer | None = None
        if self.impute_strategy is not None:
            self.imputer_ = MissingValueImputer(
                self.impute_strategy, self.missing_indicator_threshold
            ).fit(X_arr)
            self.model_feature_names_ = self.imputer_.get_feature_names_out(self.feature_names_in_)
        else:
            self.model_feature_names_ = list(self.feature_names_in_)
            if np.isnan(X_arr).any():
                warnings.warn(
                    "X contains NaN and impute_strategy=None: bartz bins NaN poorly (every NaN "
                    "counts as a distinct value, crowding out real cutpoints; see trees.py).",
                    stacklevel=2,
                )

        self.bart_ = Bart(
            self._model_X(X_arr).T,  # bartz wants (p, n)
            y_arr,
            outcome_type=self._outcome_type,
            num_trees=self.num_trees,
            n_save=self.n_save,
            n_burn=self.n_burn,
            n_skip=self.n_skip,
            num_chains=self.num_chains,
            maxdepth=self.maxdepth,
            power=self.power,
            base=self.base,
            seed=self.random_state,
            printevery=100 if self.show_progress else None,
            **extra,
        )
        # Keep a copy of n_probe random training rows (as given) for diagnostics(). They are
        # not held out: bartz above was fit on all rows; these only say where to check f(x).
        rng = np.random.default_rng(self.random_state)
        rows = rng.choice(len(X_arr), size=min(n_probe, len(X_arr)), replace=False)
        self.X_probe_ = X_arr[np.sort(rows)]
        return self

    # --- posterior draws -----------------------------------------------------

    def _draws(self, X: ArrayLike, kind: str, seed: int | None = None) -> np.ndarray:
        """bartz predictions for X, reshaped to (chain, draw, n).

        bartz returns all chains concatenated along the draw axis, chain 0 first, so a plain
        reshape restores the chain axis. `kind` is a bartz PredictKind ("mean_samples",
        "outcome_samples" or "latent_samples"); `seed` is only used for "outcome_samples".
        """
        X_arr = self._model_X(self._check_X(X))
        key = jax.random.key(self.random_state + 1 if seed is None else seed)
        out = self.bart_.predict(X_arr.T, kind=kind, key=key if kind == "outcome_samples" else None)
        return np.asarray(out).reshape(self.num_chains, self.bart_.n_save, len(X_arr))

    def predict_samples(
        self,
        X: ArrayLike,
        kind: Literal["mean", "predictive"] = "mean",
        seed: int | None = None,
    ) -> np.ndarray:
        """Posterior draws for every row of X, chains pooled.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        kind : {"mean", "predictive"}, default "mean"
            "mean": draws of f(x), the mean of y at x (p(x) = P(y = 1 | x) for the
            classifier). "predictive": draws of a new observation y at x, i.e. f(x) plus
            noise (0/1 draws for the classifier).
        seed : int, optional
            Seed for the noise in kind="predictive"; default random_state + 1.

        Returns
        -------
        ndarray of shape (num_chains * n_save, m)
            One row per posterior draw (chain 0's draws first), one column per row of X.
            Memory is draws x m floats; split very large X into batches.
        """
        bartz_kind = {"mean": "mean_samples", "predictive": "outcome_samples"}[kind]
        draws = self._draws(X, bartz_kind, seed)
        return draws.reshape(-1, draws.shape[-1])

    def predict_dist(self, X: ArrayLike, batch_size: int = 2000) -> pd.DataFrame:
        """Posterior mean and uncertainty for every row of X.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        batch_size : int, default 2000
            Rows processed at a time; memory is about n_draws x batch_size floats. Results
            do not depend on it.

        Returns
        -------
        DataFrame with one row per row of X (same index for a DataFrame) and columns:
            mean           posterior mean of f(x) (of p(x) for the classifier)
            sd             posterior sd of f(x): uncertainty about the mean
            predictive_sd  sd of a new observation y at x: sqrt(sd^2 + E[sigma^2]) (law of
                           total variance); for the classifier sqrt(p (1 - p)) of the 0/1 y

        Notes
        -----
        Rows are processed in batches only to bound memory; each row's numbers come from
        that row's own draws.
        """
        parts = []
        for chunk in _row_chunks(X, batch_size):
            mean_draws = self.predict_samples(chunk, kind="mean")
            parts.append(
                pd.DataFrame(
                    {
                        "mean": mean_draws.mean(axis=0),
                        "sd": mean_draws.std(axis=0, ddof=1),
                        "predictive_sd": np.sqrt(self._predictive_var(mean_draws)),
                    }
                )
            )
        result = pd.concat(parts, ignore_index=True)
        if isinstance(X, pd.DataFrame):
            result.index = X.index
        return result

    def predict_interval(
        self,
        X: ArrayLike,
        prob: float = 0.95,
        kind: Literal["mean", "predictive"] = "mean",
        method: Literal["hpdi", "quantile"] = "hpdi",
        batch_size: int = 2000,
    ) -> pd.DataFrame:
        """Interval holding `prob` of the posterior draws, for every row of X.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        prob : float, default 0.95
            Probability inside the interval.
        kind : {"mean", "predictive"}, default "mean"
            "mean": credible interval of f(x), the mean of y at x; it only reflects the
            uncertainty about the mean, so it is narrow. "predictive": interval for a new
            observation y at x; it also includes the noise sigma, is much wider, and should
            cover about `prob` of future observations.
        method : {"hpdi", "quantile"}, default "hpdi"
            "hpdi": highest posterior density interval (the narrowest one). "quantile":
            equal-tailed interval between the (1 - prob)/2 and (1 + prob)/2 quantiles.
        batch_size : int, default 2000
            Rows processed at a time; memory is about n_draws x batch_size floats. Results
            do not depend on it.

        Returns
        -------
        DataFrame with columns lower and upper, one row per row of X (same index for a
        DataFrame).

        Notes
        -----
        Each row's interval comes from that row's own draws. Rows are processed in batches
        only so that the (draws x rows) array never has to fit in memory for all rows.
        """
        interval = {"hpdi": hpdi, "quantile": quantile_interval}[method]
        parts = []
        for chunk in _row_chunks(X, batch_size):
            lower, upper = interval(self.predict_samples(chunk, kind=kind), prob, axis=0)
            parts.append(pd.DataFrame({"lower": lower, "upper": upper}))
        result = pd.concat(parts, ignore_index=True)
        if isinstance(X, pd.DataFrame):
            result.index = X.index
        return result

    # --- parameters, convergence and trees -----------------------------------

    def _tree_size_draws(self) -> dict[str, np.ndarray]:
        """Mean leaves and mean depth over the trees of each draw, each shaped (chain, draw)."""
        sizes = self.forest_summary().tree_sizes()
        shape = (self.num_chains, self.bart_.n_save)
        return {
            "mean_tree_leaves": sizes["mean_leaves"].to_numpy().reshape(shape),
            "mean_tree_depth": sizes["mean_depth"].to_numpy().reshape(shape),
        }

    def parameter_draws(self) -> dict[str, np.ndarray]:
        """Posterior draws of the model's scalar parameters.

        Returns
        -------
        dict of name -> ndarray of shape (num_chains, n_save), with:
            sigma               noise sd (regressor only)
            mean_tree_leaves    average number of leaves per tree in each draw
            mean_tree_depth     average tree depth in each draw, in levels of splits like
                                XGBoost (0 = single leaf); see forest_summary()
            varprob[<feature>]  probability of splitting on each feature, only with the
                                sparse prior (sparse=SparseConfig(...))
        Pass any of them to plots.plot_trace / plots.plot_rank.
        """
        draws = {**self._parameter_draws(), **self._tree_size_draws()}
        if self.bart_._main_trace.varprob is not None:  # sparse (variable selection) prior
            shape = (self.num_chains, self.bart_.n_save, len(self.model_feature_names_))
            varprob = np.asarray(self.bart_.varprob).reshape(shape)
            for j, name in enumerate(self.model_feature_names_):
                draws[f"varprob[{name}]"] = varprob[..., j]
        return draws

    def posterior_summary(
        self, prob: float = 0.95, rhat_max: float = 1.01, ess_min: float = 400
    ) -> pd.DataFrame:
        """Summary of every parameter in parameter_draws().

        Parameters
        ----------
        prob : float, default 0.95
            Probability inside the HPDI.
        rhat_max, ess_min : float, default 1.01 and 400
            Thresholds for the `ok` column (Vehtari et al. 2021).

        Returns
        -------
        DataFrame indexed by parameter, with columns mean, sd (posterior sd; its square is
        the posterior variance), hpdi_<prob>_low, hpdi_<prob>_high, rhat, ess_bulk,
        ess_tail and ok (R-hat < rhat_max and both ESS > ess_min).
        """
        return summarize_draws(self.parameter_draws(), prob, rhat_max=rhat_max, ess_min=ess_min)

    def diagnostics(
        self,
        X_probe: ArrayLike | None = None,
        prob: float = 0.95,
        rhat_max: float = 1.01,
        ess_min: float = 400,
    ) -> pd.DataFrame:
        """MCMC convergence checks.

        Covers the parameters (as in posterior_summary), the share of trees with an accepted
        grow/prune move per iteration, and f(x) at the rows of `X_probe`. Checking f(x)
        tells whether the predictions themselves have converged, not only global numbers.

        Parameters
        ----------
        X_probe : DataFrame or array of shape (k, p), optional
            Rows where f(x) is checked, e.g. some test rows. Default: X_probe_, a copy of
            random training rows kept when fitting (n_probe in fit, 20 by default; the model
            itself is fit on all rows).
        prob : float, default 0.95
            Probability inside the HPDI columns.
        rhat_max, ess_min : float, default 1.01 and 400
            A row is flagged when R-hat >= rhat_max or bulk/tail ESS <= ess_min.

        Returns
        -------
        DataFrame in the same format as posterior_summary, with extra rows accept_rate and
        f(x_probe[i]).

        Warns
        -----
        UserWarning
            Listing the rows that fail R-hat < rhat_max or ESS > ess_min (Vehtari et al.
            2021). If so, run longer (n_burn / n_save, n_skip) or more chains.
        """
        n_save = self.bart_.n_save
        accept = np.asarray(self.bart_.accept).reshape(self.num_chains, -1)[:, -n_save:]
        draws = {**self.parameter_draws(), "accept_rate": accept}
        X_check = self.X_probe_ if X_probe is None else X_probe
        mean_draws = self._draws(X_check, "mean_samples")
        for i in range(mean_draws.shape[-1]):
            draws[f"f(x_probe[{i}])"] = mean_draws[..., i]
        table = summarize_draws(draws, prob, rhat_max=rhat_max, ess_min=ess_min)
        failed = table.index[~table["ok"]].tolist()
        if failed:
            warnings.warn(f"Possible lack of convergence for: {failed}", stacklevel=2)
        return table

    def forest_summary(self) -> trees.ForestSummary:
        """Structure of every sampled tree: depth, leaves, splits and variable usage.

        Returns
        -------
        trees.ForestSummary
            Arrays shaped (chain, draw, tree) plus helpers: tree_sizes(),
            depth_distribution(), leaves_distribution() and variable_usage().

        Notes
        -----
        The depth reported by tree_ensembles.bart follows the XGBoost meaning:
        forest_summary().depth, depth_distribution(), mean_tree_depth and plot_tree_sizes all
        report levels of splits (0 for a tree without a split). So a BART depth of 3 means the
        same as an XGBoost depth of 3, and only the bartz `maxdepth` setting is shifted by one
        (bartz maxdepth = XGBoost max_depth + 1).
        """
        return trees.forest_summary(self.bart_, self.model_feature_names_)

    def split_points(self) -> pd.DataFrame:
        """How often each cut value is used, over all trees and draws, most used first.

        Returns
        -------
        DataFrame with columns feature, cutpoint (a row goes right if x > cutpoint) and
        count. A NaN cutpoint marks a split caused by NaN in X: useless (every row goes
        left) or an "is missing" split, depending on the feature (see trees.py).
        """
        return trees.split_points(self.bart_, self.model_feature_names_)

    def trees_to_dataframe(
        self,
        chains: int | list[int] | None = None,
        draws: int | list[int] | None = None,
        trees: int | list[int] | None = None,
        X: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """All nodes of the selected sampled trees as a table, like XGBoost's.

        Parameters
        ----------
        chains, draws, trees : int, list of int or None, default None (all)
            Which trees to include; negative ints count from the end (draws=-1 is the
            last draw). All draws of all trees can be millions of rows: select a few.
        X : DataFrame or array of shape (n, p), optional
            Same columns as in fit. If given, a `num_rows` column counts the rows of X
            reaching each node (after the same NaN imputation as in fit).

        Returns
        -------
        DataFrame with one row per node: chain, draw, tree, node, depth, is_leaf,
        feature, cutpoint, condition, left, right, missing, leaf_value (and num_rows). See
        trees.trees_to_dataframe for the column meanings. Plot it with
        tree_ensembles.tree_plot.plot_tree / plot_trees.

        When NaN are imputed before fitting (impute_strategy, the default), no NaN ever
        reaches the trees, so `missing` is None: a missing direction would mean nothing.
        """
        X_model = None if X is None else self._model_X(self._check_X(X))
        # `trees` (the argument) hides the trees module here, hence the _trees_table alias
        table = _trees_table(self.bart_, self.model_feature_names_, chains, draws, trees, X_model)
        if self.imputer_ is not None:  # NaN were imputed: no missing direction to report
            table["missing"] = None
        return table

    def format_tree(self, chain: int = 0, draw: int = 0, tree: int = 0) -> str:
        """One sampled tree as readable text, with feature names, cut values and leaves.

        Parameters
        ----------
        chain, draw, tree : int, default 0
            Which tree: chain in [0, num_chains), draw in [0, n_save), tree in
            [0, num_trees).

        Returns
        -------
        str
            Nested if/else rules. Leaf values are contributions to f(x) (to the probit
            latent value for the classifier); f(x) is the sum over all trees plus an offset.
        """
        return trees.format_tree(self.bart_, self.model_feature_names_, chain, draw, tree)

    # --- dump / load ---------------------------------------------------------

    def dump(self, path: str | Path) -> None:
        """Write the fitted model to a folder.

        The folder gets bart.pkl (bartz's own dump of the MCMC trace) and estimator.pkl
        (everything else). Reload with `load`.

        Parameters
        ----------
        path : str or Path
            Folder to write; created if needed.

        Notes
        -----
        bartz's format depends on the bartz, JAX and Equinox versions: fine for caching,
        not for long-term archival. Files are large with many trees and draws (about 4 GB
        for 10,000 trees x 1,000 draws).
        """
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        self.bart_.dump(folder / "bart.pkl")
        state = {k: v for k, v in self.__dict__.items() if k != "bart_"}
        with (folder / "estimator.pkl").open("wb") as f:
            pickle.dump((type(self), state), f)

    @staticmethod
    def load(path: str | Path) -> "_BartBase":
        """Load a model written with `dump`.

        Parameters
        ----------
        path : str or Path
            Folder written by dump.

        Returns
        -------
        BartRegressor or BartClassifier
            The same class that was dumped.
        """
        folder = Path(path)
        with (folder / "estimator.pkl").open("rb") as f:
            cls, state = pickle.load(f)
        model = cls.__new__(cls)
        model.__dict__.update(state)
        model.bart_ = Bart.load(folder / "bart.pkl")
        return model


def _shared_doc(cls: type) -> str:
    """The class docstring without its first (summary) line, to reuse in subclasses."""
    return (cls.__doc__ or "").split("\n", 1)[1]


class BartRegressor(RegressorMixin, _BartBase):
    __doc__ = """Bayesian additive regression trees for a continuous target.

    Model: y = f(x) + e, e ~ N(0, sigma^2), where f is a sum of `num_trees` small trees.
    The posterior over f and sigma is sampled by MCMC (bartz), so every prediction comes
    with uncertainty.

    Examples
    --------
    >>> bart = BartRegressor(num_chains=4).fit(X_train, y_train)
    >>> bart.predict(X_test)                                  # posterior mean of f(x)
    >>> bart.predict_interval(X_test, kind="predictive")      # 95% HPDI for a new y
    >>> bart.posterior_summary()                              # sigma, tree size
    >>> bart.diagnostics()                                    # R-hat, ESS
    """ + _shared_doc(_BartBase)  # its Parameters and Attributes

    _outcome_type = "continuous"

    def _prepare_y(self, y: Any) -> np.ndarray:
        """y as float32."""
        return np.asarray(y, dtype=np.float32)

    def _sigma_draws(self) -> np.ndarray:
        """Noise sd draws, shaped (chain, draw)."""
        sdev = np.asarray(self.bart_.get_error_sdev())
        return sdev.reshape(self.num_chains, self.bart_.n_save)

    def _parameter_draws(self) -> dict[str, np.ndarray]:
        """The regressor's own parameter: sigma."""
        return {"sigma": self._sigma_draws()}

    def _predictive_var(self, mean_draws: np.ndarray) -> np.ndarray:
        """Var[y_new] = Var[f(x)] + E[sigma^2], from draws of f(x) shaped (draws, m)."""
        return mean_draws.var(axis=0, ddof=1) + np.mean(self._sigma_draws() ** 2)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Posterior mean of f(x) for every row of X.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.

        Returns
        -------
        ndarray of shape (m,)
        """
        return np.asarray(self.bart_.predict(self._model_X(self._check_X(X)).T, kind="mean"))


class BartClassifier(ClassifierMixin, _BartBase):
    __doc__ = """Bayesian additive regression trees for a binary target (probit link).

    Model: P(y = 1 | x) = Phi(f(x)), where Phi is the standard normal CDF and f a sum of
    `num_trees` small trees. There is no noise sd sigma. Labels can be any two values;
    they are encoded to 0/1 (in sorted order) and decoded back in predict.

    Examples
    --------
    >>> clf = BartClassifier(num_chains=4).fit(X_train, y_train)
    >>> clf.predict_proba(X_test)[:, 1]                       # posterior mean of p(x)
    >>> clf.predict_interval(X_test)                          # 95% HPDI of p(x)
    """ + _shared_doc(_BartBase)  # its Parameters and Attributes

    _outcome_type = "binary"

    def _prepare_y(self, y: Any) -> np.ndarray:
        """Encode the two labels to 0/1 (in sorted order) and remember them in classes_."""
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary targets are supported, got {len(self.classes_)} classes")
        return y_encoded.astype(np.float32)

    def _predictive_var(self, mean_draws: np.ndarray) -> np.ndarray:
        """Var of a new 0/1 outcome, p (1 - p) with p the posterior mean probability."""
        p = mean_draws.mean(axis=0)
        return p * (1 - p)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Posterior mean probability of each class for every row of X.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.

        Returns
        -------
        ndarray of shape (m, 2)
            Columns P(classes_[0]) and P(classes_[1]); the second is the posterior mean
            of p(x) = Phi(f(x)).
        """
        p = np.asarray(self.bart_.predict(self._model_X(self._check_X(X)).T, kind="mean"))
        return np.column_stack([1 - p, p])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predicted class for every row of X: classes_[1] where P >= 0.5.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.

        Returns
        -------
        ndarray of shape (m,) with values from classes_.
        """
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def predict_interval(
        self,
        X: ArrayLike,
        prob: float = 0.95,
        kind: Literal["mean", "predictive"] = "mean",
        method: Literal["hpdi", "quantile"] = "hpdi",
        batch_size: int = 2000,
    ) -> pd.DataFrame:
        """Credible interval of p(x) = P(y = 1 | x) for every row of X.

        Same as the base method, but only kind="mean" is allowed: an interval for a single
        0/1 outcome is not useful (it is {0}, {1} or [0, 1]).

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        prob : float, default 0.95
            Probability inside the interval.
        kind : {"mean"}, default "mean"
            Anything else raises a ValueError.
        method : {"hpdi", "quantile"}, default "hpdi"
            Narrowest interval, or equal-tailed quantiles.
        batch_size : int, default 2000
            Rows processed at a time, to bound memory.

        Returns
        -------
        DataFrame with columns lower and upper (probabilities), one row per row of X.
        """
        if kind != "mean":
            raise ValueError("BartClassifier only supports kind='mean' (an interval for p(x))")
        return super().predict_interval(
            X, prob=prob, kind=kind, method=method, batch_size=batch_size
        )

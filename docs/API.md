# API and migration notes for 0.2.6

## Graph analysis

`backend.id(Y: set, X: set, G)` returns `(equation, identifiable)` for
`p(Y | do(X))`. Pass whole variable names, for example `{"Treatment"}`, not
`set("Treatment")`. `G` is the result of `grapl.dsl.GraplDSL().readgrapl(...)`.
A non-identifiable effect returns `(None, False)`.

`workflows.general_cb_analysis(causal_graph, effect_var_name, cause_var_name,
info_print=True)` returns `(builder, weight_expression)`. It returns
`(None, None)` if identification or automatic weighting is unsupported.
`weight_expression.tostr()` produces text. The returned builder is called with
`dist_map`, `N`, `kernel`, and `cause_intv_name_map` and returns
`(weight_function, weight_expression)`.

## Data and distribution maps

Observational data are a nonempty dictionary of row-aligned arrays `(N, d)`;
low-level resampling also accepts `(N,)` and retains that rank in its output.
Use `(N, 1)` for scalar variables when fitting densities and building maps.

`cause_intv_name_map={"Y": "intv_Y"}` distinguishes observations from assigned
interventions. A distribution-map key is a comma-separated set of arguments,
for example `"intv_Y,Z"`. The corresponding callable has named arguments
`lambda intv_Y, Z: ...` and returns `(N,)` or `(N, 1)` values. Primed symbolic
names use `_prime` in callable argument names. Missing primed data are resolved
from the corresponding base observed variable; explicit primed data take
precedence. Do not supply conflicting aliases.

`intv_dict={"intv_Y": 1}` specifies one scalar intervention. A one-dimensional
vector specifies one multivariate intervention, **not** a batch of distinct
interventions; call the functions separately for different intervention values.
The dictionary is not mutated.

For back-door weighting with discrete causes, the default kernel is the
Kronecker delta. Continuous causes require an appropriate kernel, for example
a Gaussian with an explicitly chosen bandwidth. Arbitrary density-flooring,
weight clipping and empirical support choices change the approximation and
are not performed automatically.

## Weight evaluation

```python
weights = backend.weight_compute(w_func, data, intv_dict, nan_policy="raise")
```

Returns `(N,)` finite nonnegative weights. The compiled formula includes `1/N`;
bootstrap probabilities are proportional to weights regardless of their common
scale. Wrong output length, NaN, infinity and negative values raise ValueError.
`nan_policy="min"` explicitly enables legacy NaN replacement with a warning;
all-NaN arrays remain an error. Infinite and negative weights always raise.

## Resampling

```python
sample = backend.cw_bootstrapper(
    data, weights, intv_dict, n_sample,
    sampling_mode="fast", random_state=42, return_original_idx=True,
)
```

- `n_sample`: nonnegative integer; zero returns correctly shaped empty arrays.
- `weights`: one finite nonnegative value per input row, with positive total mass.
- `sampling_mode`: `"fast"` uses weighted choice; `"robust"` uses Gumbel-max.
- `random_state`: integer seed or None. Both modes use a local random stream.
- `return_original_idx`: adds `sample["original_idx"]`, shape `(n_sample, 1)`.
  The function always returns **one dictionary**, not a tuple.
- Observed keys historically strip apostrophes in the output. Conflicting arrays
  that would overwrite one another now raise. Intervention keys must differ
  from observed output keys. `original_idx` is reserved when requested.

Original variables and source indices stay aligned. Assigned intervention
labels are separate from original labels. Gumbel-max avoids explicit probability
normalization, but does not repair statistical weight concentration.

## Convenience workflows

`frontdoor_intv`, `backdoor_intv`, and `general_causal_bootstrapping_intv`
resample each observed cause value using its original frequency. Their output
is grouped by intervention, not shuffled. Multiple groups now use different
seeded random streams instead of restarting the identical stream in every group.

`frontdoor_cf`, `backdoor_cf`, and `general_causal_bootstrapping_cf` accept a
specified `intv_dict` and `n_sample`. The front/back-door wrappers warn and rename
an intervention key that equals the observed cause (e.g. `Y` to `intv_Y`).
Both the symbolic map and output name are updated consistently.

## Distribution estimators

`MultivarContiDistributionEstimator(data_fit)` expects a finite nonempty
`(N, d)` array. `fit_multinorm`, `fit_kde(bandwidth=None)`,
`fit_gmm(n_components)`, `fit_kmeans(k)` and `fit_histogram(n_bins)` each return
a vectorized evaluator. Query arrays are `(M, d)` or lists of column arrays.
For `d=1`, a 1D array is a batch of scalar observations. `plot` requires the
`plot` extra and a two-dimensional fitted distribution.

`fit_histogram` returns **bin probability masses**, not bin-width-normalized
continuous densities. `n_bins=0` per dimension uses the number of observed
unique values as the bin count; it is not a general categorical encoder.
Use compact category coding and consistent bin choices when estimating ratios.

`MultivarDiscDistributionEstimator(data_fit, data_est=None)` retains the legacy
second argument (unused). `fit_categorical()` requires one-hot encoded rows
and returns a frozen SciPy discrete distribution. `fit_multinomial()` requires
integer count rows with the same positive total and returns a frozen multinomial
distribution. These methods fix previously invalid SciPy constructor calls.

## Changes to account for when upgrading

- Metadata now has a single authoritative version/Python requirement in
  `pyproject.toml`; `setup.py` is only a compatibility shim.
- Core requirements admit compatible NumPy 1.x and 2.x versions; scikit-learn
  is declared, and Matplotlib is optional for plotting.
- `robust` sampling and grouped workflows have corrected seeded sequences.
  Reproducibility within this release does not imply identical samples to 0.2.5.
- Default NaN handling now raises rather than silently modifying weights.
- Multi-character cause/effect names and repeated symbolic-factor cancellation
  are handled correctly; weight-expression rendering includes the `1/N` scale.
- `general_causal_bootstrapping_simple` is not a current interface. Use
  `general_causal_bootstrapping_intv`; the README has been corrected.
- `causal-sampler==0.0.6` as previously prepared pins `causalbootstrapping==0.2.5`.
  Publishing this package alone does not change that downstream pin. Update and
  rebuild the downstream package before expecting it to install 0.2.6.

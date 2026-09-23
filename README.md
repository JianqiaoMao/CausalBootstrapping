# CausalBootstrapping

CausalBootstrapping identifies causal effects from a specified causal graph and
constructs weights for empirical resampling towards an estimated interventional
distribution. It includes graph analysis, distribution estimators, and front-door,
back-door and general weighting workflows. Validity depends on the causal
assumptions, available support and quality of the estimated distributions.

## Installation

Python **3.9 or later** is declared in the package metadata. See
[validation notes](docs/VALIDATION.md) for the versions actually tested.

Install the prepared wheel locally:

```bash
python -m pip install causalbootstrapping-0.2.6-py3-none-any.whl
```

After the maintainer publishes 0.2.6 to PyPI:

```bash
python -m pip install "causalbootstrapping==0.2.6"
```

For a source checkout:

```bash
python -m pip install -e ".[dev,demo]"
python -m unittest discover -s tests -v
python examples/quickstart.py
```

The import name is lowercase:

```python
import causalbootstrapping
from causalbootstrapping import backend, workflows
print(causalbootstrapping.__version__)
```

Core installation includes NumPy, SciPy, GRAPL, Graphviz's Python package,
SymPy and scikit-learn. `plot` adds Matplotlib; `demo` adds notebook and data-frame
requirements; `dev` adds build, distribution validation and notebook-validation
tools. Graph rendering additionally requires the Graphviz system executable
(`dot`); identification and resampling do not require rendering.

## Self-contained front-door example

This example fits empirical distributions from generated observations. It needs
no CSV files or external downloads. `Y` is the cause, `Z` is a mediator, and `X`
is the effect in `Y -> Z -> X, Y <-> X`.

```python
import numpy as np
from causalbootstrapping import backend as be, workflows as wf
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator

rng = np.random.default_rng(42)
N = 600
U = rng.integers(0, 2, size=(N, 1))
Y = (rng.random((N, 1)) < 0.2 + 0.6 * U).astype(int)
Z = (rng.random((N, 1)) < 0.2 + 0.6 * Y).astype(int)
X = 2 * Z + U + rng.normal(size=(N, 1))
data = {"Y": Y, "Z": Z, "X": X}

builder, weight_expr = wf.general_cb_analysis(
    causal_graph="Y; Z; X; Y -> Z; Z -> X; Y <-> X;",
    effect_var_name="X", cause_var_name="Y", info_print=True,
)

p_y = MultivarContiDistributionEstimator(Y).fit_histogram([0])
p_yz = MultivarContiDistributionEstimator(np.hstack([Y, Z])).fit_histogram([0, 0])
dist_map = {
    "intv_Y,Z": lambda intv_Y, Z: p_yz([intv_Y, Z]),
    "intv_Y": lambda intv_Y: p_y(intv_Y),
    "Y',Z": lambda Y_prime, Z: p_yz([Y_prime, Z]),
    "Y'": lambda Y_prime: p_y(Y_prime),
}
w_func, _ = builder(
    dist_map=dist_map, N=N, kernel=None,
    cause_intv_name_map={"Y": "intv_Y"},
)
weights = be.weight_compute(w_func, data, {"intv_Y": 1})
sample = be.cw_bootstrapper(
    data=data, weights=weights, intv_dict={"intv_Y": 1},
    n_sample=200, sampling_mode="fast", random_state=42,
    return_original_idx=True,
)
print(sample["X"].shape)             # (200, 1)
print(sample["intv_Y"].shape)        # (200, 1): assigned intervention labels
print(sample["original_idx"].shape)  # (200, 1): positions in input data
np.testing.assert_array_equal(
    sample["X"], X[sample["original_idx"].ravel()]
)
```

The symbolic dummy variable `Y'` is mapped to its observed `Y` values when
needed. Explicit historical `{"Y'": Y, ...}` dictionaries are still supported.
Distribution callables use valid Python argument names such as `Y_prime`, while
the distribution-map key retains the apostrophe. Each callable returns one
probability/density per input row.

`sample["Y"]` records the original sampled diagnosis/category; `sample["intv_Y"]`
records the requested intervention. They need not coincide in a front-door
resample. Source indices refer to the supplied array order, not a DataFrame's
index labels.

## Public interfaces

| Interface | Purpose |
| --- | --- |
| `backend.id(Y, X, G)` | Identify p(Y \| do(X)); returns `(equation, identifiable)`. |
| `workflows.general_cb_analysis(...)` | Analyze a graph; return a weight builder and `weightExpr`. |
| `backend.build_weight_function(...)` | Compile a supported ID equation using a distribution map. |
| `backend.weight_compute(...)` | Evaluate one causal weight per observational row. |
| `backend.cw_bootstrapper(...)` | Resample rows and optionally retain source positions. |
| `workflows.general_causal_bootstrapping_intv(...)` | Use observed cause values and their frequencies. |
| `workflows.general_causal_bootstrapping_cf(...)` | Sample a specified intervention and requested count. |
| `workflows.backdoor_intv / backdoor_cf` | Convenience back-door workflows. |
| `workflows.frontdoor_intv / frontdoor_cf` | Convenience front-door workflows. |

The historical `_cf` name means sampling under a specified intervention here;
it does not implement individual-level counterfactual inference.

See [API notes](docs/API.md) for array shapes, distribution-map conventions,
weight validation, kernels and return values. Automatic weight construction
supports only the identification expressions accepted by `weight_func_parse`,
not every identifiable expression.

## Numerical behaviour and reproducibility

Both `fast` (weighted sampling with replacement) and `robust` (Gumbel-max)
respect `random_state`. Robust sampling uses log weights and preserves exact
zero support; it does **not** flatten the distribution or cure low effective
sample size. Fast normalization is scaled to avoid overflow when summing large
finite weights. Neither mode changes the relative intended weights.

Invalid weights fail explicitly. `weight_compute` no longer silently replaces
NaNs by default. For legacy behaviour, `nan_policy="min"` opts into replacement
by the smallest finite nonnegative weight with a warning; investigate density
support before using it. An all-zero vector cannot be resampled.

Split participants/observational rows into train and test partitions before
estimating training distributions or resampling. Preserve source IDs when
assessing overlap; independent bootstrap draws from the same input pool are
not independent held-out data.

## Tutorials and release contents

- [Tutorial0](Tutorials/Tutorial0-Quickstart.ipynb): self-contained introduction,
  source indices, reproducibility, and high-level workflow examples.
- [Tutorials 1--3](Tutorials/README.md): historical back-door, front-door and
  general-graph studies. Their external CSV data are **not shipped** in this
  release. Set `CAUSALBOOTSTRAPPING_DATA_ROOT` to your local `test_data` directory.
- `examples/quickstart.py` and `tests/` need no external data.
- The wheel contains the importable library and license. The source distribution
  additionally contains documentation, notebooks, examples and tests.
- [CHANGELOG.md](CHANGELOG.md), [RELEASE.md](RELEASE.md), and
  [validation notes](docs/VALIDATION.md) describe changes and publishing steps.

## Citation

```bibtex
@article{little2019causal,
  title={Causal bootstrapping},
  author={Little, Max A and Badawy, Reham},
  journal={arXiv preprint arXiv:1910.09648},
  year={2019}
}
```

Author: Jianqiao Mao. The original GPL license is retained in `LICENSE`.

# Validation of 0.2.6

All checks below were run on Linux. The declared Python requirement is `>=3.9`.

## Environment matrix

| Python | NumPy | SciPy | scikit-learn | Self-contained tests |
| --- | --- | --- | --- | --- |
| 3.9.25 | 2.0.2 | 1.13.1 | 1.6.1 | 30 passed |
| 3.12.14 | 2.5.3 | 1.18.1 | 1.9.1 | 30 passed |
| 3.13.15 | 2.5.3 | 1.18.1 | 1.9.1 | 30 passed |

These environments used GRAPL 1.6.1, Graphviz's Python package 0.21, and
SymPy 1.14.0. Dependency consistency checks passed in all three environments.

The same 30 tests also passed in a separate Python 3.12 environment installed
from the built wheel. Its import path was checked to be in `site-packages`.

## Compatibility with the causal-sampler package

The 45 tests of the previously prepared causal-sampler 0.0.6 passed while
loading this new causalbootstrapping source via an explicit PYTHONPATH override
in Python 3.10.21 (NumPy 1.25.2, SciPy 1.13.1, SymPy 1.13.1, scikit-learn 1.6.1,
CPU PyTorch 2.5.1). This verifies runtime compatibility for the exercised
interfaces, including all five samplers, but is **not** a dependency-resolution
claim: that downstream package still pins causalbootstrapping 0.2.5 and must
be updated/rebuilt separately.

## Other checks

- All README Python examples executed successfully.
- The self-contained quickstart and exact Tutorial0 code cells executed
  sequentially in a headless Python process.
- All four notebooks passed schema, unique-cell-ID and code-syntax validation;
  saved outputs/execution counts are cleared.
- Eight historical test functions passed using CSV files available in the
  supplied archive, outside the release directory, on Python 3.10/NumPy 1.25.
  Those CSVs are intentionally excluded from the deliverable.
- A source distribution and a wheel built from that source distribution passed
  `twine check`. Both declare version 0.2.6 and `Requires-Python: >=3.9`.
- Archive inventories were checked to exclude old 0.2.5 distributions, test
  data, bytecode/cache directories and model artifacts.

Coverage includes an analytic front-door weight comparison, back-door/front-door
workflows, unsupported identification, multi-character names, repeated-factor
cancellation, one-row and vectorized density evaluation, categorical/multinomial
PMFs, unconditional numerical integration, deterministic robust sampling,
source-index alignment, zero support, huge finite weights, empty outputs,
input non-mutation and explicit invalid-weight rejection.

## Limits

Python 3.11, Windows, macOS and Python versions newer than 3.13 were not executed
locally. The included CI matrix covers Python 3.9--3.13, but hosted CI was not
run during this preparation. Notebook UI/kernel execution and the full
historical notebook experiments were not validated; headless Tutorial0 and
historical test-script execution are the checks reported above. GRAPL may emit
upstream syntax/resource warnings; they did not prevent the tested operations.

The tests do not prove causal identification correctness for every graph or
scientific validity of every density estimate. The ID recursion is retained
from the supplied code. Automatic weighting still has its existing symbolic
eligibility restrictions.

This release was packaged and checked, not uploaded to PyPI.

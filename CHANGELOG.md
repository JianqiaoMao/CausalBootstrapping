# Changelog

## 0.2.6

### Packaging and documentation
- Single-source version 0.2.6 and Python >=3.9 metadata; remove conflicting
  version 0.2.2 / Python <3.11 declarations from legacy setup.py.
- Broaden numerical dependency ranges for modern Python and NumPy 2; declare
  scikit-learn and optional plot/demo/dev dependencies.
- Add an accurate README, API/migration notes, independent quickstart, release
  instructions and validation report; refresh tutorial paths and clear outputs.
- Exclude test data, old distributions, caches and model artifacts; include
  source documentation, examples, notebooks and regression tests in the sdist.

### Runtime fixes
- Make robust Gumbel-max sampling honor the local random seed, and protect fast
  normalization from overflow when summing large finite weights.
- Validate weights/counts/row alignment, preserve zero support and input
  dictionaries, and return aligned original indices when requested.
- Raise on invalid computed weights by default; retain explicit nan_policy='min'.
- Resolve ID dummy-variable aliases, handle whole multi-character variable names,
  repair symbolic cancellation and include N in the rendered weight formula.
- Keep renamed interventions synchronized with their symbolic mappings and use
  separate random streams for different intervention groups.
- Preserve array rank when combining bootstrap groups and single-row density
  outputs; fix categorical/multinomial fitting and unconditional integration.

See docs/API.md for behaviour changes. Identification theory and the core ID
recursion are retained; this release does not claim support for every possible
identified expression.

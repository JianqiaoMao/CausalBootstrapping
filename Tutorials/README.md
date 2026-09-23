# Tutorials

Start with **Tutorial0-Quickstart.ipynb**: it generates its own data and exercises
identification, density estimation, bootstrap indices and reproducible sampling.

Tutorials 1--3 retain the historical back-door, front-door and general-graph
studies. Install the `demo` extra, then set `CAUSALBOOTSTRAPPING_DATA_ROOT` to a
separately supplied CSV directory containing:

- `frontdoor_discY_contZ_contX_discU/`
- `backdoor_contY_contX_contU/`
- `complex_scenario/`

The release contains no test CSVs. Notebook outputs are cleared so historical
results are not presented as results of the current runtime.

"""Self-contained front-door example; no CSV files or downloads required."""
import numpy as np
from causalbootstrapping import backend as be, workflows as wf
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator


def run(seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    N = 600
    U = rng.integers(0, 2, size=(N, 1))
    Y = (rng.random((N, 1)) < 0.2 + 0.6 * U).astype(int)
    Z = (rng.random((N, 1)) < 0.2 + 0.6 * Y).astype(int)
    X = 2.0 * Z + U + rng.normal(size=(N, 1))
    data = {"Y": Y, "Z": Z, "X": X}
    graph = "Y; Z; X; Y -> Z; Z -> X; Y <-> X;"
    builder, expression = wf.general_cb_analysis(graph, "X", "Y", info_print=verbose)
    p_y = MultivarContiDistributionEstimator(Y).fit_histogram([0])
    p_yz = MultivarContiDistributionEstimator(np.hstack([Y, Z])).fit_histogram([0, 0])
    dist_map = {
        "intv_Y,Z": lambda intv_Y, Z: p_yz([intv_Y, Z]),
        "intv_Y": lambda intv_Y: p_y(intv_Y),
        "Y',Z": lambda Y_prime, Z: p_yz([Y_prime, Z]),
        "Y'": lambda Y_prime: p_y(Y_prime),
    }
    w_func, _ = builder(dist_map=dist_map, N=N, kernel=None,
                        cause_intv_name_map={"Y": "intv_Y"})
    weights = be.weight_compute(w_func, data, {"intv_Y": 1})
    sample = be.cw_bootstrapper(data, weights, {"intv_Y": 1}, n_sample=200,
                               random_state=seed, return_original_idx=True)
    np.testing.assert_array_equal(sample["X"], X[sample["original_idx"].ravel()])
    if verbose:
        print("Sample shapes:", {key: value.shape for key, value in sample.items()})
    return data, dist_map, builder, w_func, weights, sample


if __name__ == "__main__":
    run()

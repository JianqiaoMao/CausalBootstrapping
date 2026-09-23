import copy
import importlib.util
from pathlib import Path
import unittest
import warnings
import numpy as np
from grapl import dsl
from causalbootstrapping import backend as be, workflows as wf
from causalbootstrapping.expr_extend import weightExpr
from causalbootstrapping.distEst_lib import (
    MultivarContiDistributionEstimator as Continuous,
    MultivarDiscDistributionEstimator as Discrete,
    user_defined_func_obj,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cb_example", ROOT / "examples/quickstart.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.data = {"Y": np.arange(4).reshape(-1, 1), "X": np.arange(8).reshape(4, 2)}
        self.weights = np.array([0., 1., 2., 3.])

    def test_indices_and_reproducibility(self):
        for mode in ("fast", "robust"):
            with self.subTest(mode=mode):
                a = be.cw_bootstrapper(self.data, self.weights, {"intv_Y": 2}, 30, mode, 42, True)
                b = be.cw_bootstrapper(self.data, self.weights, {"intv_Y": 2}, 30, mode, 42, True)
                np.testing.assert_array_equal(a["original_idx"], b["original_idx"])
                np.testing.assert_array_equal(a["X"], self.data["X"][a["original_idx"].ravel()])
                self.assertTrue(np.all(a["original_idx"] != 0))
                self.assertTrue(np.all(a["intv_Y"] == 2))

    def test_robust_distribution(self):
        out = be.cw_bootstrapper(self.data, self.weights, {"intv_Y": 0}, 6000, "robust", 3, True)
        freq = np.bincount(out["original_idx"].ravel(), minlength=4) / 6000
        np.testing.assert_allclose(freq, self.weights / 6, atol=0.025)

    def test_global_rng_not_mutated(self):
        np.random.seed(87)
        before = np.random.get_state()
        be.cw_bootstrapper(self.data, self.weights, {"intv_Y": 0}, 10, "robust", 42)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_weights(self):
        for weights in ([0]*4, [1, -1, 1, 1], [np.nan]*4, [np.inf]*4, [1, 2]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                be.cw_bootstrapper(self.data, weights, {}, 3)

    def test_extreme_weights(self):
        for mode in ("fast", "robust"):
            out = be.cw_bootstrapper(self.data, [1e308]*4, {}, 10, mode, 1)
            self.assertEqual(out["X"].shape, (10, 2))

    def test_empty_output(self):
        for mode in ("fast", "robust"):
            out = be.cw_bootstrapper(self.data, self.weights, {"intv_Y": [1, 2]}, 0, mode, 42, True)
            self.assertEqual(out["X"].shape, (0, 2))
            self.assertEqual(out["intv_Y"].shape, (0, 2))
            self.assertEqual(out["original_idx"].shape, (0, 1))

    def test_inputs_not_mutated(self):
        interventions = {"intv_Y": np.array([1, 2])}
        before = copy.deepcopy(interventions)
        be.cw_bootstrapper(self.data, self.weights, interventions, 3)
        np.testing.assert_array_equal(interventions["intv_Y"], before["intv_Y"])
        be.weight_compute(lambda **kw: np.ones(4), self.data, interventions)
        np.testing.assert_array_equal(interventions["intv_Y"], before["intv_Y"])

    def test_invalid_count(self):
        for n in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                be.cw_bootstrapper(self.data, self.weights, {}, n)

    def test_conflicting_names(self):
        for data, intv in ((self.data, {"Y": 1}),
                          ({"Y": np.zeros(4), "Y'": np.ones(4)}, {})):
            with self.assertRaises(ValueError):
                be.cw_bootstrapper(data, self.weights, intv, 2)

    def test_invalid_data_lengths(self):
        with self.assertRaises(ValueError):
            be.cw_bootstrapper({"X": np.ones(2), "Y": np.ones(3)}, self.weights, {}, 2)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            be.cw_bootstrapper(self.data, self.weights, {}, 2, "typo")

    def test_weight_validation(self):
        for w in ([1, 2], [np.nan]*4, [np.inf]*4, [-1]*4):
            with self.assertRaises(ValueError):
                be.weight_compute(lambda **kw: np.array(w), self.data, {})

    def test_explicit_nan_policy(self):
        with self.assertWarns(RuntimeWarning):
            w = be.weight_compute(lambda **kw: np.array([1, np.nan, 3, 4]), self.data, {}, nan_policy="min")
        np.testing.assert_array_equal(w, [1, 1, 3, 4])
        with self.assertRaises(ValueError):
            be.weight_compute(lambda **kw: np.full(4, np.nan), self.data, {}, nan_policy="min")


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.dist, cls.builder, cls.wfunc, cls.weights, cls.sample = example.run(verbose=False)

    def test_frontdoor_weights(self):
        y, z = self.data["Y"].ravel(), self.data["Z"].ravel()
        expected = np.array([np.mean(z[y == 1] == zi) / np.mean(z[y == yi] == zi)
                             for yi, zi in zip(y, z)]) / len(y)
        np.testing.assert_allclose(self.weights, expected)

    def test_primed_aliases(self):
        data = dict(self.data)
        data["Y'"] = data.pop("Y")
        w = be.weight_compute(type(self).wfunc, data, {"intv_Y": 1})
        np.testing.assert_allclose(w, self.weights)

    def test_frontdoor_wrappers(self):
        args = dict(cause_data={"Y": self.data["Y"]}, effect_data={"X": self.data["X"]},
                    mediator_data={"Z": self.data["Z"]}, dist_map=self.dist, random_state=42)
        a = wf.frontdoor_cf(**args, intv_dict={"intv_Y": 1}, n_sample=15)
        with self.assertWarns(UserWarning):
            b = wf.frontdoor_cf(**args, intv_dict={"Y": 1}, n_sample=15)
        np.testing.assert_array_equal(a["X"], b["X"])
        c = wf.frontdoor_intv(**args, cause_intv_name_map={"Y": "intv_Y"})
        self.assertEqual(len(c["X"]), len(self.data["X"]))
        np.testing.assert_array_equal(np.unique(c["intv_Y"], return_counts=True)[1],
                                      np.unique(self.data["Y"], return_counts=True)[1])

    def test_backdoor_weights_and_rename(self):
        y = np.array([0, 0, 1, 1]).reshape(-1, 1)
        data = {"Y": y, "U": np.array([0, 1, 0, 1]).reshape(-1, 1), "X": y + 1}
        dist = {"U": lambda U: np.full(len(U), .5),
                "intv_Y,U": lambda intv_Y, U: np.full(len(U), .25)}
        args = dict(cause_data={"Y": y}, effect_data={"X": data["X"]},
                    confounder_data={"U": data["U"]}, dist_map=dist, random_state=42)
        a = wf.backdoor_cf(**args, intv_dict={"intv_Y": 1}, n_sample=20)
        self.assertTrue(np.all(a["Y"] == 1))
        with self.assertWarns(UserWarning):
            b = wf.backdoor_cf(**args, intv_dict={"Y": 1}, n_sample=20)
        np.testing.assert_array_equal(a["X"], b["X"])
        c = wf.backdoor_intv(**args, cause_intv_name_map={"Y": "intv_Y"})
        self.assertEqual(c["X"].shape, (4, 1))

    def test_multichar_names(self):
        graph = "Treatment; Outcome; Confounder; Treatment -> Outcome; Confounder -> Outcome; Confounder -> Treatment;"
        builder, expression = wf.general_cb_analysis(graph, "Outcome", "Treatment", info_print=False)
        self.assertTrue(callable(builder))
        self.assertIn("treatment", expression.tostr())
        self.assertNotIn("P(t_{i}", expression.tostr())

    def test_unidentifiable(self):
        graph = dsl.GraplDSL().readgrapl("Y; X; Y -> X; Y <-> X;")
        equation, ok = be.id(Y={"X"}, X={"Y"}, G=graph)
        self.assertFalse(ok)
        self.assertIsNone(equation)

    def test_one_row_weight_evaluation(self):
        data = {k: v[:1] for k, v in self.data.items()}
        wfunc, _ = type(self).builder(dist_map=self.dist, N=1, kernel=None,
                                      cause_intv_name_map={"Y": "intv_Y"})
        self.assertEqual(be.weight_compute(wfunc, data, {"intv_Y": 1}).shape, (1,))

    def test_expression_cancellation(self):
        e = weightExpr(w_nom=[{"Y"}, {"Y"}], w_denom=[{"Y"}, {"Y"}], cause_var="Treatment")
        self.assertEqual(e.w_nom, [])
        self.assertEqual(e.w_denom, [])
        self.assertEqual(e.cause_var, {"Treatment"})


class DensityTests(unittest.TestCase):
    def test_continuous_methods(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(80, 2))
        for method, kwargs in (("fit_multinorm", {}), ("fit_kde", {}),
                               ("fit_kmeans", {"k": 2}), ("fit_gmm", {"n_components": 2}),
                               ("fit_histogram", {"n_bins": 3})):
            with self.subTest(method=method):
                f = getattr(Continuous(x), method)(**kwargs)
                values = f(x[:5])
                self.assertEqual(values.shape, (5,))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values >= 0).all())
                self.assertEqual(f(x[:1]).shape, (1,))

    def test_one_dimensional_queries(self):
        x = np.linspace(-2, 2, 20).reshape(-1, 1)
        f = Continuous(x).fit_multinorm()
        self.assertEqual(f(np.array([0., 1.])).shape, (2,))

    def test_histogram_edges(self):
        f = Continuous(np.array([[0.], [1.]])).fit_histogram(2)
        np.testing.assert_allclose(f(np.array([0., 1.])), [.5, .5])
        with self.assertWarns(UserWarning):
            np.testing.assert_array_equal(f(np.array([-1., 2.])), [0., 0.])

    def test_categorical(self):
        distribution = Discrete(np.array([[1, 0], [0, 1], [1, 0]])).fit_categorical()
        np.testing.assert_allclose(distribution.pmf([0, 1]), [2/3, 1/3])

    def test_multinomial(self):
        distribution = Discrete(np.array([[1, 1], [2, 0]])).fit_multinomial()
        self.assertAlmostEqual(distribution.pmf([1, 1]), 2 * .75 * .25)

    def test_unconditional_integration(self):
        f = user_defined_func_obj(lambda x: x, ranges="specified")
        self.assertAlmostEqual(f.integral_over_specified_var({"x": (0, 1)}, {}), .5)

    def test_invalid_fit_data(self):
        for x in (np.empty((0, 1)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                Continuous(x)


if __name__ == "__main__":
    unittest.main()

from causalbootstrapping.expr_extend import weightExpr, DOExpr
from causalbootstrapping.utils import gumbel_max, remove_incoming
import copy as cp
import grapl.eqn as eqn
import grapl.expr as expr
import numpy as np
import inspect
import warnings
import numbers
from causalbootstrapping.cb_type_defs import (
    DataDict,
    IntvDict,
    DistFunc,
    DistMap,
    WeightFunc,
    IdExpr
)
from typing import Dict, Sequence, Tuple, Union, Optional, Literal, Any


def id(Y:set, X:set, G: Any) -> Tuple[Optional[IdExpr], bool]:
    """
    Identify the causal effect P(Y|do(X)) in a given causal graph G using the ID algorithm.
    
    Parameters:
        Y (set): A set of outcome variable names.
        X (set): A set of intervention variable names.
        G (grapl.admg object): The causal graph represented as a grapl Graph object.
    
    Returns:
        id_formula (grapl.eqn object): The identified causal effect formula as a grapl Eqn object.
        identifiable (bool): A boolean indicating whether the causal effect is identifiable.
    
    """
    
    Z = set()
    P_star = DOExpr()  
    P_star.addvars(num = G.nodes())
    
    def sid_recur(Y, X, Z, P_star, G):
        
        V = G.nodes()
        
        # Line 1
        if len(X) == 0:
            VdY = V.difference(Y)
            P_star_copy = cp.deepcopy(P_star)
            P_star_copy.addvars(mrg = VdY.union(P_star_copy.mrg))
            P_star_copy.simplify()
            return P_star_copy, True

        Y_an_G = G.an(Y)
        VdY_an_G = V.difference(Y_an_G)
        #Line 2
        if len(VdY_an_G) > 0:
            P_star_copy = cp.deepcopy(P_star)
            P_star_copy.addvars(mrg = VdY_an_G.union(P_star_copy.mrg))
            G_sub_Y_an_G = cp.deepcopy(G)
            G_sub_Y_an_G = G_sub_Y_an_G.sub(Y_an_G)
            return sid_recur(Y, X.intersection(Y_an_G), Z, P_star_copy, G_sub_Y_an_G)
        
        VdX = V.difference(X)
        G_doX = cp.deepcopy(G)
        for x in X:
            G_doX = remove_incoming(G_doX, x)
        Y_an_G_doX = G_doX.an(Y)
        W = VdX.difference(Y_an_G_doX)
        # Line 3
        if len(W) > 0:
            return sid_recur(Y, X.union(W), Z, P_star, G)
        
        G_sub_dX = cp.deepcopy(G)
        G_sub_dX = G_sub_dX.sub(VdX)
        C = G_sub_dX.districts()
            
        # Line 4
        if len(C) > 1:
            I_star_exprs = []
            for c in C:
                P_star_copy = cp.deepcopy(P_star)
                I_star_expr_c, c_sidfixable = sid_recur(c, V.difference(c), Z, P_star_copy, G)
                if c_sidfixable:
                    I_star_mrg = list(I_star_expr_c.mrg)
                    for mrg_var in I_star_mrg:
                        new_var = mrg_var + chr(39)     # Avoid variable name clashes
                        I_star_expr_c.subsvar(mrg_var, new_var)
                    I_star_expr_c.simplify()
                    I_star_exprs.append(I_star_expr_c)
                else:
                    return None, False
            I_star_expr = DOExpr() 
            I_star_expr.addvars(mrg = V.difference(Y.union(X)))
            I_star_expr.combine(tuple(I_star_exprs))
            # I_star_expr.simplify()
            return I_star_expr, True

        # Line 5
        if len(C) == 1:
            # Line 6
            if len(G.districts()) > 1: 
                # Line 7
                if C[0] in G.districts():
                    mrg_vars = C[0].difference(Y)
                    expr_c0 = DOExpr() 
                    expr_c0.addvars(mrg = mrg_vars)
                    topo_ordering = G.topsort()
                    for v_i in C[0]:
                        vi_topo_index = topo_ordering.index(v_i)
                        vi_topo_procd = set(topo_ordering[: vi_topo_index])
                        den_term = vi_topo_procd
                        num_term = {v_i}.union(den_term)
                        expr_c0.addvars(num = num_term, den = den_term)
                        expr_c0.simplify()
                    expr_c = DOExpr()
                    expr_c.addvars(mrg = mrg_vars)
                    for num in expr_c0.num:
                        mrg_vars_num = V.difference(num)
                        P_star_copy = cp.deepcopy(P_star)
                        P_star_copy.addvars(mrg = mrg_vars_num)
                        P_star_copy.simplify()
                        expr_c.combine((P_star_copy,))
                    for den in expr_c0.den:
                        mrg_vars_den = V.difference(den)
                        P_star_copy = cp.deepcopy(P_star)
                        P_star_copy.addvars(mrg = mrg_vars_den)
                        P_star_copy.simplify()
                        P_star_copy_inv = DOExpr(num=P_star_copy.den, den=P_star_copy.num, mrg=P_star_copy.mrg)
                        P_star_copy_inv.simplify()
                        expr_c.combine((P_star_copy_inv,))
                    expr_c.simplify()
                    return expr_c, True
                
                c0_in_c_flag = False
                for c in G.districts():
                    if C[0].issubset(c):
                        c0_in_c_flag = True
                        c_prime = c
                        break
                # Line 8
                if c0_in_c_flag:
                    c_expr = DOExpr()
                    topo_ordering = G.topsort()
                    for v_i in c_prime:
                        vi_topo_index = topo_ordering.index(v_i)
                        vi_topo_procd = set(topo_ordering[: vi_topo_index])
                        vi_topo_procd_and_c_prime = vi_topo_procd.intersection(c_prime)
                        vidc_prime = vi_topo_procd.difference(c_prime)
                        den_term = vi_topo_procd_and_c_prime.union(vidc_prime)
                        num_term = {v_i}.union(den_term)
                        c_expr.addvars(num = num_term, den = den_term)
                    # Paper: Z = C'/X, but the implementation seems shows Z = X/C'
                    G_sub_c_prime = cp.deepcopy(G)
                    G_sub_c_prime = G_sub_c_prime.sub(c_prime)
                    return sid_recur(Y, X.intersection(c_prime), X.difference(c_prime), c_expr, G_sub_c_prime)            
            
            else:
                return None, False    
    rhs, identifiable = sid_recur(Y, X, Z, P_star, G)
    if not identifiable:
        print("Not identifiable.")
        return None, identifiable
    rhs.simplify()
    lhs = expr.Expr()
    lhs.addvars(num=Y, dov=X)
    id_formula = eqn.Eqn(lhs, rhs)
    return id_formula, identifiable

def _data_arrays(data):
    if not isinstance(data, dict) or not data:
        raise ValueError("data must be a nonempty dictionary of arrays.")
    arrays = {name: np.asarray(value) for name, value in data.items()}
    if any(value.ndim not in (1, 2) for value in arrays.values()):
        raise ValueError("Each data array must have shape (N,) or (N, d).")
    N = len(next(iter(arrays.values())))
    if N == 0 or any(len(value) != N for value in arrays.values()):
        raise ValueError("All data arrays must have the same positive row count.")
    return arrays, N


def _expand_interventions(intv_dict, N):
    expanded = {}
    for name, value in intv_dict.items():
        vector = np.asarray(value)
        if vector.ndim == 0:
            vector = vector.reshape(1)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("An intervention must be a scalar or nonempty 1D vector.")
        expanded[name] = np.broadcast_to(vector, (N, vector.size)).copy()
    return expanded


def weight_compute(w_func: WeightFunc, data: DataDict, intv_dict: IntvDict,
                   *, nan_policy: Literal["raise", "min"] = "raise") -> np.ndarray:
    """Compute one finite nonnegative weight per observational row.

    Intervention values are a scalar or a vector for a single intervention,
    broadcast across all rows. Inputs are not mutated. By default undefined
    weights raise ValueError. ``nan_policy='min'`` explicitly opts into the
    historical replacement of NaNs by the minimum finite nonnegative weight;
    it does not repair infinite/negative weights or an all-NaN vector.
    """
    if nan_policy not in ("raise", "min"):
        raise ValueError("nan_policy must be 'raise' or 'min'.")
    arrays, N = _data_arrays(data)
    weights = np.asarray(w_func(**{**arrays, **_expand_interventions(intv_dict, N)}),
                         dtype=float).reshape(-1).copy()
    if weights.size != N:
        raise ValueError("Weight function must return exactly one weight per data row.")
    missing = np.isnan(weights)
    if missing.any() and nan_policy == "min":
        valid = np.isfinite(weights) & (weights >= 0)
        if not valid.any():
            raise ValueError("No finite nonnegative weight is available for NaN replacement.")
        warnings.warn(f"Replacing {int(missing.sum())} NaN weights with the minimum "
                      "finite nonnegative weight.", RuntimeWarning, stacklevel=2)
        weights[missing] = weights[valid].min()
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights contain NaN, infinity or negative values. "
                         "Check estimated densities and denominator support.")
    return weights


def build_weight_function(
    intv_prob: IdExpr,
    dist_map: DistMap,
    N: int,
    cause_intv_name_map: Dict[str, str],
    kernel: Optional[DistFunc] = None,
) -> Tuple[WeightFunc, weightExpr]:
    """
    Generate the causal bootstrapping weight function using the identified interventional probability and 
    corresponding distribution functions.

    Parameters:
        intv_prob (grapl.eqn object): The identified interventional probability expression.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        cause_intv_name_map (dict): A dictionary mapping cause variable names to their corresponding intervention variable names.
        N (int): The number of data points in the dataset.
        kernel (function, optional): The kernel function to be used in the weight computation. Defaults to None.

    Returns:
        function: The corresponding causal bootstrapping weight function.
        weightExpr: The weight expression object representing the weight computation.
    """

    def divide_functions(**funcs):
        def division(**kwargs):
            kwargs = {key.replace("'","_prime"): value for key, value in kwargs.items()}
            # ID dummy variables reuse the corresponding observational rows.
            for key in list(kwargs):
                if "_prime" in key:
                    kwargs.setdefault(key.split("_prime")[0], kwargs[key])
            def resolve(name):
                if name in kwargs:
                    return kwargs[name]
                if "_prime" in name and name.split("_prime")[0] in kwargs:
                    return kwargs[name.split("_prime")[0]]
                raise KeyError(f"Missing data for distribution argument {name!r}.")
            result = np.ones(N, dtype=float)
            for nom_i in w_nom_mapped:
                func_key = ",".join(nom_i)
                param_names = inspect.signature(funcs[func_key]).parameters
                param = {key: resolve(key) for key in param_names}
                result *= funcs[func_key](**param).reshape(-1)
            for denom_i in w_denom_mapped:
                func_key = ",".join(denom_i)
                param_names = inspect.signature(funcs[func_key]).parameters
                param = {key: resolve(key) for key in param_names}
                result /= funcs[func_key](**param).reshape(-1)
            if cause_kernel_flag:
                param_names = inspect.signature(funcs["kernel"]).parameters
                param = {key : kwargs[key] for key in param_names}
                result *= funcs["kernel"](**param).reshape(-1)
            result *= (lambda n: 1/n)(N)
            return result
        return division    

    if isinstance(N, bool) or not isinstance(N, numbers.Integral) or N <= 0:
        raise ValueError("N must be a positive integer.")
    from causalbootstrapping.utils import weight_func_parse
    if intv_prob is None or not weight_func_parse(intv_prob)[3]:
        raise ValueError("Identification expression is not supported for automatic weighting.")
    if set(cause_intv_name_map) != set(intv_prob.lhs.dov):
        raise ValueError("cause_intv_name_map must cover the intervention variables.")
    if any(k == v for k, v in cause_intv_name_map.items()):
        raise ValueError("Use distinct names for observed and intervention variables.")
    dist_map_sep = ","
    dist_map_sorted = {}
    
    for key ,value in dist_map.items():
        sorted_key = dist_map_sep.join(sorted(key.split(","))).replace(" ","")
        dist_map_sorted[sorted_key] = value
    
    cause_var = intv_prob.lhs.dov.copy()
    eff_var = intv_prob.lhs.num[0].copy()
    w_denom = intv_prob.rhs.den.copy()
    w_denom = [sorted(w_denom[i]) for i in range(len(w_denom))]
    w_nom = intv_prob.rhs.num.copy()
    w_nom = [sorted(w_nom[i]) for i in range(len(w_nom))]
    epsilo = intv_prob.rhs.mrg.copy()
    pa_var = sorted(epsilo.union(eff_var).union(cause_var))
    if pa_var in w_nom:
        w_nom.remove(pa_var)
        cause_kernel_flag = True
        if kernel is None:
            raise ValueError("Kernel function is required but not provided. E.g., kernel = lambda Y, intv_Y: np.equal(Y, intv_Y)")
        kernel_func = {"kernel": kernel}
        funcs = {**dist_map_sorted, **kernel_func}
    else:
        pa_var = set(pa_var)
        pa_var = pa_var.difference(cause_var)
        pa_var = sorted(pa_var)
        w_nom.remove(list(pa_var))
        cause_kernel_flag = False
        funcs = dist_map_sorted

    w_nom_mapped = [sorted([cause_intv_name_map.get(name, name) for name in w_nom_i]) for w_nom_i in w_nom]
    w_denom_mapped = [sorted([cause_intv_name_map.get(name, name) for name in w_denom_i]) for w_denom_i in w_denom]

    weight_func = divide_functions(**funcs)
    weight_expr = weightExpr(w_nom = w_nom, w_denom = w_denom, cause_var = list(cause_var)[0], kernel_flag = cause_kernel_flag)

    return weight_func, weight_expr

def cw_bootstrapper(
    data: DataDict,
    weights: np.ndarray,
    intv_dict: IntvDict,
    n_sample: int,
    sampling_mode: Literal["fast", "robust"] = "fast",
    random_state: Optional[int] = None,
    return_original_idx: bool = False
) -> Dict[str, np.ndarray]:
    """
    Perform causal bootstrapping on the input observational data using the provided weight function and 
    designated intervention values.

    Parameters:
        data (dict): A dictionary containing variable names as keys and their corresponding data arrays as values.
        weights (numpy.ndarray): An array containing the computed causal bootstrapping weights for each data point (N,1).
        intv_dict (dict): key: str, value: int/list(len: M)/ndarray(M,), a dictionary containing the intervention variable names and their corresponding values.
        n_sample (int): The number of samples to be generated through bootstrapping.
        sampling_mode (str, optional): The mode for bootstrapping. Options: 'fast' or 'robust'. Defaults to 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        return_original_idx (bool, optional): Whether to return the original indices of the samples. Defaults to False.

    Returns:
        bootstrap_data (dict): A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
        When return_original_idx=True, the returned dictionary also contains
        'original_idx', an integer array of shape (n_sample, 1). No tuple is returned.
    """
    data, N = _data_arrays(data)
    if isinstance(n_sample, bool) or not isinstance(n_sample, numbers.Integral) or n_sample < 0:
        raise ValueError("n_sample must be a nonnegative integer.")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if (weights.size != N or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()):
        raise ValueError("weights must match the data length, be finite and nonnegative, "
                         "and contain positive mass.")
    if sampling_mode not in ("fast", "robust"):
        raise ValueError("Invalid mode. Choose either 'fast' or 'robust'.")
    expanded = _expand_interventions(intv_dict, int(n_sample))
    # Retain the historical unprimed output names without silent overwrites.
    canonical = {}
    for name, value in data.items():
        name = name.replace("'", "")
        if name in canonical and not np.array_equal(canonical[name], value):
            raise ValueError(f"Conflicting arrays collapse to output name {name!r}.")
        canonical[name] = value
    if set(canonical).intersection(expanded):
        raise ValueError("Intervention names must differ from observed output names.")
    if return_original_idx and "original_idx" in (set(canonical) | set(expanded)):
        raise ValueError("original_idx is reserved when return_original_idx=True.")
    rng = np.random.RandomState(random_state)
    if sampling_mode == "fast":
        scaled = weights / weights.max()
        sample_indices = rng.choice(N, p=scaled/scaled.sum(), size=n_sample, replace=True)
    else:
        sample_indices = np.asarray([gumbel_max(weights, rng=rng)
                                     for _ in range(n_sample)], dtype=np.int64)
    bootstrap_data = {name: value[sample_indices] for name, value in canonical.items()}
    bootstrap_data.update(expanded)
    if return_original_idx:
        bootstrap_data["original_idx"] = sample_indices.reshape(-1, 1)
    return bootstrap_data

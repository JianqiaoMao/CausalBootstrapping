from causalbootstrapping.expr_extend import weightExpr, DOExpr
from causalbootstrapping.utils import gumbel_max, remove_incoming
import copy as cp
import grapl.eqn as eqn
import grapl.expr as expr
import numpy as np
import inspect
import warnings
from causalbootstrapping.cb_type_defs import (
    DataDict,
    IntvDict,
    DistFunc,
    DistMap,
    WeightFunc,
    IdExpr
)
from typing import Dict, Sequence, Tuple, Union, Optional, Literal


def id(Y, X, G):
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

def weight_compute(
    w_func: WeightFunc,
    data: DataDict,
    intv_dict: IntvDict,
) -> np.ndarray:
    """
    Compute causal bootstrapping weights for the given weight function and input observational data.
    
    Parameters:
        w_func (function): The causal bootstrapping weight function to be used.
        data (dict): A dictionary containing variable names as keys and their corresponding ndarray as values.
        intv_dict (dict): key: str, value: float/list(len: M)/ndarray(M,), a dictionary containing the intervention variable names and their corresponding values.
    Returns:
        numpy.ndarray: An array containing the computed causal bootstrapping weights for each data point.
    """
    N = data[list(data.keys())[0]].shape[0]
    intv_dict_expand = {}
    for intv_var in intv_dict.keys():
        if np.isscalar(intv_dict[intv_var]):
            intv_dict[intv_var] = [intv_dict[intv_var]]
        if isinstance(intv_dict[intv_var], np.ndarray):
            if intv_dict[intv_var].ndim >= 2:
                raise ValueError("intv_dict value should be 1-dimensional if numpy.ndarray.")
            intv_dict[intv_var] = intv_dict[intv_var].flatten().tolist()

        intv_dict_expand[intv_var] = np.array([intv_dict[intv_var] for i in range(N)]).reshape(N, len(intv_dict[intv_var]))
    data_for_weight_compute = {**data, **intv_dict_expand}
    weights = w_func(**data_for_weight_compute).reshape(-1)
    if np.any(np.isnan(weights)):
        number_nan = np.sum(np.isnan(weights))
        warnings.warn(f"{number_nan} NaN values found in weights. Replacing NaNs with the minimum non-NaN weight.")
    weights[np.isnan(weights)] = weights[~np.isnan(weights)].min()
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
        intv_prob (grapl.expr object): The identified interventional probability expression.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        cause_intv_name_map (dict): A dictionary mapping cause variable names to their corresponding intervention variable names.
        N (int): The number of data points in the dataset.
        kernel (function, optional): The kernel function to be used in the weight computation. Defaults to None.

    Returns:
        function: The corresponding causal bootstrapping weight function.
    """

    def divide_functions(**funcs):
        def division(**kwargs):
            kwargs = {key.replace("'","_prime"): value for key, value in kwargs.items()}
            result = 1
            for nom_i in w_nom_mapped:
                func_key = ",".join(nom_i)
                param_names = inspect.signature(funcs[func_key]).parameters
                param = {key : kwargs[key] if kwargs[key].shape[0] != 1 else kwargs[key][0] for key in param_names}
                result *= funcs[func_key](**param).reshape(-1)
            for denom_i in w_denom_mapped:
                func_key = ",".join(denom_i)
                param_names = inspect.signature(funcs[func_key]).parameters
                param = {key : kwargs[key] if kwargs[key].shape[0] != 1 else kwargs[key][0] for key in param_names}
                result /= funcs[func_key](**param).reshape(-1)
            if cause_kernel_flag:
                param_names = inspect.signature(funcs["kernel"]).parameters
                param = {key : kwargs[key] for key in param_names}
                result *= funcs["kernel"](**param).reshape(-1)
            result *= (lambda n: 1/n)(N)
            return result
        return division    

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

def bootstrapper(
    data: DataDict,
    w_func: WeightFunc,
    intv_var_name_in_data: Union[str, Sequence[str]],
    intv_var_name: Union[str, Sequence[str]],
    mode: Literal["fast", "robust"] = "fast",
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Perform causal bootstrapping on the input observational data using the provided weight function and given observations.
    
    Parameters:
        data (dict): A dictionary containing variable names as keys and their corresponding data arrays as values.
        w_func (function): The causal bootstrapping weight function.
        intv_var_name_in_data (str or list of str): A list of strings representing the variable names of observational data used as the interventional values.
        intv_var_name (str or list of str): The variable name for the interventional variable.
        mode (str, optional): The mode for bootstrapping. Options: 'fast' or 'robust'. Defaults to 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
    
    Returns:
        bootstrap_data (dict): A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
        weights (numpy.ndarray): An array containing the computed causal bootstrapping weights for each data point.
    """
    if isinstance(intv_var_name_in_data, str):
        intv_var_name_in_data = [intv_var_name_in_data]
    if isinstance(intv_var_name, str):
        intv_var_name = [intv_var_name]
    rng = np.random.RandomState(random_state)
    
    var_names = list(data.keys())
    N = data[var_names[0]].shape[0]

    stack_intv_data = np.hstack([data[key] if data[key].ndim == 2 else data[key].reshape(-1,1) for key in intv_var_name_in_data])
    intv_var_values_unique, counts = np.unique(stack_intv_data, axis=0, return_counts=True)
    intv_dims = [data[name].shape[1] if data[name].ndim == 2 else 1
             for name in intv_var_name_in_data]
    intv_dim_offsets = np.cumsum([0] + intv_dims)
    intv_var_slices = {
        name: slice(intv_dim_offsets[i], intv_dim_offsets[i+1])
        for i, name in enumerate(intv_var_name)
    }    
    
    causal_weights = np.zeros((N, intv_var_values_unique.shape[0]))

    bootstrap_data = {}
    for var in var_names:
        bootstrap_data[var] = np.zeros((N, data[var].shape[1]))
    for i, var in enumerate(intv_var_name):
        bootstrap_data[var] = np.zeros((N, intv_dims[i]))

    ind_pos = 0
    for i, (intv_, n) in enumerate(zip(intv_var_values_unique, counts)):
        intv_i_dict = {name: intv_[intv_var_slices[name]] for name in intv_var_name}
        weight_intv = weight_compute(w_func, data, intv_i_dict)
        causal_weights[:, i] = weight_intv
        if mode == "fast":
            sample_indices = rng.choice(range(N), p=weight_intv/np.sum(weight_intv), size=n, replace=True)
        elif mode == "robust":
            sample_indices = [gumbel_max(weight_intv) for _ in range(n)]
        else:
            raise ValueError("Invalid mode. Choose either 'fast' or 'robust'.")
        
        for var in var_names:
            bootstrap_data[var][ind_pos:ind_pos+n] = data[var][sample_indices]
        for var in intv_var_name:
            bootstrap_data[var][ind_pos:ind_pos+n] = np.array([intv_[intv_var_slices[var]] for _ in range(n)])
        ind_pos += n
        
    return bootstrap_data, causal_weights

def simu_bootstrapper(
    data: DataDict,
    w_func: WeightFunc,
    intv_dict: IntvDict,
    n_sample: int,
    mode: Literal["fast", "robust"] = "fast",
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Perform simulational causal bootstrapping on the input observational data using the provided weight function and 
    designated intervention values.

    Parameters:
        data (dict): A dictionary containing variable names as keys and their corresponding data arrays as values.
        w_func (function): The causal bootstrapping weight function.
        intv_dict (dict): key: str, value: int/list(len: M)/ndarray(M,), a dictionary containing the intervention variable names and their corresponding values.
        n_sample (int): The number of samples to be generated through bootstrapping.
        mode (str, optional): The mode for bootstrapping. Options: 'fast' or 'robust'. Defaults to 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.

    Returns:
        bootstrap_data (dict): A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
        weights (numpy.ndarray): An array containing the computed causal bootstrapping weights for each data point.
    """
    rng = np.random.RandomState(random_state)
    
    causal_weights = weight_compute(w_func, data, intv_dict)
    
    var_names = list(data.keys())
    N = data[var_names[0]].shape[0]
    bootstrap_data = {}
    if mode == "fast":
        sample_indices = rng.choice(range(N), p=causal_weights/np.sum(causal_weights), size=n_sample, replace=True)
    elif mode == "robust":
        sample_indices = [gumbel_max(causal_weights) for _ in range(n_sample)]
    else:
        raise ValueError("Invalid mode. Choose either 'fast' or 'robust'.")
    
    for var in var_names:
        bootstrap_data[var.replace("'","")] = data[var][sample_indices]
    for intv_var in intv_dict.keys():
        if isinstance(intv_dict[intv_var], int):
            intv_dict[intv_var] = [intv_dict[intv_var]]
        if isinstance(intv_dict[intv_var], np.ndarray):
            if intv_dict[intv_var].ndim >= 2:
                raise ValueError("intv_dict value should be 1-dimensional if numpy.ndarray.")
            intv_dict[intv_var] = intv_dict[intv_var].flatten().tolist()
        bootstrap_data[intv_var] = np.array([intv_dict[intv_var] for _ in range(n_sample)]).reshape(n_sample, len(intv_dict[intv_var]))

    return bootstrap_data, causal_weights
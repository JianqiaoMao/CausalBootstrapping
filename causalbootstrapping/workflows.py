
from causalbootstrapping.backend import *
from causalbootstrapping.expr_extend import weightExpr
from causalbootstrapping.utils import weight_func_parse
import grapl.algorithms as algs
import grapl.dsl as dsl
import numpy as np
import warnings
from causalbootstrapping.cb_type_defs import (
    DataDict,
    DistMap,
    IntvDict,
    WeightBuilder,
    CauseIntvMap,
    KernelFunc,
)
from typing import Optional, Tuple, Dict, Literal

def general_cb_analysis(
    causal_graph: str, 
    effect_var_name: str, 
    cause_var_name: str, 
    info_print: bool = True
) -> Tuple[WeightBuilder, weightExpr]:

    """
    Perform pre-analysis for the given causal graph with intended cause(intenventional) and effect variables to 
    formulate the weight function.

    This function generates causal bootstrapping weight function and required distributions
    for a given causal graph and a specified intervention.

    Parameters:
    causal_graph (str): A causal graph representing the relationships between variables.
    effect_var_name (str): The name of the effect variable for the causal intervention.
    cause_var_name (str): The name of the cause variable responsible for the intervention.
    info_print (bool, optional): A boolean indicating whether to print the weight function and required distributions. Default is True.

    Returns:
    weight_func_lam (function): A lambda function to calculate causal bootstrapping weights.
    weight_func_str (str): The string representation of the causal bootstrapping weights function.

    Usage Instructions:
    1. Call this function with the causal graph and variable names.
    2. If the intervention is identifiable, it will return a lambda function and the weights function string.
    3. Prepare inputs for the lambda function:
       - dist_map: A dictionary containing required probability distributions.
       - N: Sample size for the intervention.
       - kernel: (Optional) A kernel function if needed.
    4. Call weight_func_lam(dist_map, N, kernel) to estimate the causal effect.
    """    
    
    grapl_obj = dsl.GraplDSL()
    G = grapl_obj.readgrapl(causal_graph)
    id_formular, identifiable = id(Y = set(effect_var_name),
                                  X = set(cause_var_name),
                                  G = G)

    if not identifiable:
        print("Not identifiable")
        return None, None
    
    cause_var = id_formular.lhs.dov
    eff_var = id_formular.lhs.num[0]
    
    w_nom, w_denom, kernel_flag, valid_id_eqn = weight_func_parse(id_formular)
    
    if not valid_id_eqn:
        warnings.warn("No automatic process available for current version of causalBootstrapping lib. Derive manually.")
        return None, None

    id_formular_str = id_formular.tocondstr()
    weight_func_lam = lambda dist_map, N, kernel, cause_intv_name_map: build_weight_function(id_formular, 
                                                                                             dist_map, 
                                                                                             N, 
                                                                                             cause_intv_name_map, 
                                                                                             kernel)
    weight_func_expr = weightExpr(
        w_nom=w_nom,
        w_denom=w_denom,
        cause_var=cause_var,
        kernel_flag=kernel_flag
    )
    weight_func_str = weight_func_expr.tostr()
    wanted_dist = weight_func_expr.build_p_term(w_nom + w_denom)
    kernel_func = weight_func_expr.build_kernel_term()
    # print out interventional probability expression and required distributions
    if info_print:
        print("Interventional prob.:{}".format(id_formular_str))    
        print("Causal bootstrapping weights function: {}".format(weight_func_str))
        print("Required distributions:")
        for i, dist in enumerate(wanted_dist):
            print("{}: {}".format(i+1, dist))
        if kernel_flag:
            print("Kernel function required:")
            for i, kernel_i in enumerate(kernel_func):
                print("{}: {}".format(i+1, kernel_i))

    return weight_func_lam, weight_func_expr

def general_causal_bootstrapping_intv(
    weight_func_lam: WeightBuilder, 
    dist_map: DistMap, 
    data: DataDict, 
    intv_var_name_in_data: str, 
    cause_intv_name_map: CauseIntvMap,
    kernel: Optional[KernelFunc] = None, 
    sampling_mode: Literal["fast", "robust"] = "fast", 
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform causal bootstrapping for a general causal graph if it is identifiable using a specified weight function. 
    This function is designed to create a bootstrapped dataset based on the specified intervention variables and the 
    weighting function. The function allows for an optional kernel to be specified.

    Parameters:
        weight_func_lam (function): A lambda or function that calculates weights. It should take 'dist_map', 'N', and optionally 'kernel' as arguments.
        dist_map (dict): An object representing a distance map, which is used by 'weight_func_lam' to calculate weights.
        data (dict): A dictionary containing the dataset. Keys are variable names and values are corresponding data arrays.
        intv_var_name_in_data (str): The name of the intervention variable in the 'data' dictionary. The order should match intv_var_name.
        cause_intv_name_map (dict): A dictionary mapping cause variable names to their corresponding intervention variable names.
        kernel (function, optional): An optional kernel function that can be used in weight computation. Default is None.
        sampling_mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """
    N = list(data.values())[0].shape[0]
    intv_var_name = list(cause_intv_name_map.values())[0]
    w_func, _ = weight_func_lam(dist_map = dist_map, 
                                N = N, 
                                kernel = kernel, 
                                cause_intv_name_map = cause_intv_name_map)
    
    if isinstance(intv_var_name_in_data, str):
        intv_var_name_in_data = [intv_var_name_in_data]
    if isinstance(intv_var_name, str):
        intv_var_name = [intv_var_name]    

    stack_intv_data = np.hstack([data[key] if data[key].ndim == 2 else data[key].reshape(-1,1) for key in intv_var_name_in_data])
    intv_var_values_unique, counts = np.unique(stack_intv_data, axis=0, return_counts=True)
    intv_dims = [data[name].shape[1] if data[name].ndim == 2 else 1
             for name in intv_var_name_in_data]
    intv_dim_offsets = np.cumsum([0] + intv_dims)
    intv_var_slices = {
        name: slice(intv_dim_offsets[i], intv_dim_offsets[i+1])
        for i, name in enumerate(intv_var_name)
    }    

    bootstrapped_data = {}
    for i, (intv_, n) in enumerate(zip(intv_var_values_unique, counts)):
        intv_i_dict = {name: intv_[intv_var_slices[name]] for name in intv_var_name}
        weight_intv = weight_compute(w_func, data, intv_i_dict)
        bootstrapped_data_i = cw_bootstrapper(data = data, 
                                            weights = weight_intv, 
                                            intv_dict = intv_i_dict, 
                                            n_sample = n, 
                                            sampling_mode = sampling_mode, 
                                            random_state = random_state)
        for key in bootstrapped_data_i:
            if key not in bootstrapped_data:
                bootstrapped_data[key] = bootstrapped_data_i[key]
            else:
                bootstrapped_data[key] = np.vstack((bootstrapped_data[key], bootstrapped_data_i[key]))
    return bootstrapped_data
    

def general_causal_bootstrapping_cf(
    weight_func_lam: WeightBuilder, 
    dist_map: DistMap, 
    data: DataDict, 
    cause_intv_name_map: CauseIntvMap, 
    intv_dict: IntvDict, 
    n_sample: int, 
    kernel: Optional[KernelFunc] = None, 
    sampling_mode: Literal["fast", "robust"] = "fast", 
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform simulational causal bootstrapping for a general causal graph if it is identifiable using a specified weight function. 
    This function is designed to create a bootstrapped dataset based on the specified intervention {variable:values} pair and the weighting function.
    The function allows for an optional kernel to be specified.

    Parameters:
        weight_func_lam (function): A lambda or function that calculates weights. It should take 'dist_map', 'N', and optionally 'kernel' as arguments.
        dist_map (dict): An object representing a distance map, which is used by 'weight_func_lam' to calculate weights.
        data (dict): A dictionary containing the dataset. Keys are variable names and values are corresponding data arrays.
        cause_intv_name_map (dict): A dictionary mapping cause variable names to their corresponding intervention variable names.
        intv_dict (dict): key: str, value: int/list(len: M)/ndarray(M,), a dictionary containing the intervention variable names and their corresponding values.
        n_sample (int): The number of samples to be generated through bootstrapping.
        kernel (function, optional): An optional kernel function that can be used in weight computation. Default is None.
        sampling_mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """    
    N = list(data.values())[0].shape[0]
    w_func, _ = weight_func_lam(dist_map = dist_map, 
                                N = N, 
                                kernel = kernel, 
                                cause_intv_name_map = cause_intv_name_map)
    causal_weights = weight_compute(w_func = w_func, 
                                    data = data,
                                    intv_dict = intv_dict)    
    bootstrapped_data = cw_bootstrapper(data = data, 
                                        weights = causal_weights, 
                                        intv_dict = intv_dict, 
                                        n_sample = n_sample, 
                                        sampling_mode = sampling_mode, 
                                        random_state = random_state)
        
    
    return bootstrapped_data

def backdoor_intv(
    cause_data: DataDict, 
    effect_data: DataDict, 
    confounder_data: DataDict, 
    dist_map: DistMap, 
    cause_intv_name_map: CauseIntvMap,
    kernel_intv: Optional[KernelFunc] = None, 
    sampling_mode: Literal["fast", "robust"] = "fast", 
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform backdoor causal bootstrapping to de-confound the causal effect using the provided observational 
    data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        confounder_data (dict): A dictionary containing the confounder variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        cause_intv_name_map (dict): A dictionary mapping cause variable names to their corresponding intervention variable names.
        kernel_intv (function, optional): The kernel function to be used in the backdoor bootstrapping for the cause variable. Defaults to None, use discrete Kronecker delta.
        sampling_mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """
    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    confounder_var_name = list(confounder_data.keys())[0]
    intv_var_name = cause_intv_name_map[cause_var_name]
    
    data = cause_data.copy()
    data.update(effect_data)
    data.update(confounder_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + confounder_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + cause_var_name + "; \n"
    
    weight_func_lam, _ = general_cb_analysis(causal_graph = causal_graph, 
                                             effect_var_name = effect_var_name, 
                                             cause_var_name = cause_var_name, 
                                             info_print= False)
    
    if kernel_intv is None:
        _kernel_intv = eval(f"lambda {intv_var_name},{cause_var_name}: np.equal({intv_var_name},{cause_var_name})",
                            {"np": np})
    else:
        _kernel_intv = eval(f"lambda {intv_var_name},{cause_var_name}: kernel_intv({intv_var_name},{cause_var_name})",
                            {"kernel_intv": kernel_intv})
    cb_data = general_causal_bootstrapping_intv(weight_func_lam = weight_func_lam, 
                                                dist_map = dist_map, 
                                                data = data, 
                                                intv_var_name_in_data = cause_var_name, 
                                                cause_intv_name_map = cause_intv_name_map,
                                                kernel = _kernel_intv, 
                                                sampling_mode = sampling_mode, 
                                                random_state = random_state)

    return cb_data

def backdoor_cf(
    cause_data: DataDict, 
    effect_data: DataDict, 
    confounder_data: DataDict, 
    dist_map: DistMap, 
    intv_dict: IntvDict, 
    n_sample: int, 
    kernel_intv: Optional[KernelFunc] = None,
    sampling_mode: Literal["fast", "robust"] = "fast", 
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform simulational backdoor causal bootstrapping to de-confound the causal effect using the provided 
    observational data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        confounder_data (dict): A dictionary containing the confounder variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        intv_dict (dict): key: str, value: int/list(len: M)/ndarray(M,), a dictionary containing the intervention variable names and their corresponding values.
        n_sample (int): The number of samples to be generated through bootstrapping.
        kernel_intv (function, optional): The kernel function to be used in the backdoor adjustment for the cause variable. Defaults to None.
        sampling_mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """
 	
    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    confounder_var_name = list(confounder_data.keys())[0]
    intv_var_name = list(intv_dict.keys())[0]
    cause_intv_name_map = {cause_var_name: intv_var_name}
    
    if cause_var_name == intv_var_name:
        warnings.warn("Intervention variable in intv_dict should be different from the cause variable name in cause_data. \
                      Automatically rewrite interventional variable as 'intv_{}'.".format(cause_var_name))
        intv_dict = {"intv_{}".format(cause_var_name): intv_dict[intv_var_name]}
        intv_var_name = "intv_{}".format(cause_var_name)

    data = cause_data.copy()
    data.update(effect_data)
    data.update(confounder_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + confounder_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + cause_var_name + "; \n"
    
    weight_func_lam, _ = general_cb_analysis(causal_graph = causal_graph, 
                                             effect_var_name = effect_var_name, 
                                             cause_var_name = cause_var_name, info_print= False)
    
    if kernel_intv is None:
        _kernel_intv = eval(f"lambda {cause_var_name}, {intv_var_name}: np.equal({cause_var_name}, {intv_var_name})",
                            {"np": np})
    else:
        _kernel_intv = eval(f"lambda {cause_var_name}, {intv_var_name}: kernel_intv({cause_var_name}, {intv_var_name})",
                            {"kernel_intv": kernel_intv})
    
    cb_data = general_causal_bootstrapping_cf(weight_func_lam = weight_func_lam, 
                                              dist_map = dist_map, 
                                              data = data, 
                                              cause_intv_name_map = cause_intv_name_map,
                                              intv_dict = intv_dict, 
                                              n_sample = n_sample, 
                                              kernel = _kernel_intv, 
                                              sampling_mode = sampling_mode, 
                                              random_state = random_state)
    return cb_data

def frontdoor_intv(
    cause_data: DataDict, 
    mediator_data: DataDict, 
    effect_data: DataDict, 
    dist_map: DistMap, 
    cause_intv_name_map: CauseIntvMap,
    sampling_mode: Literal["fast", "robust"] = "fast", 
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform frontdoor causal bootstrapping to de-confound the causal effect using the provided observational 
    data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        mediator_data (dict): A dictionary containing the mediator variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        cause_intv_name_map (dict): A dictionary mapping cause variable names to their corresponding intervention variable names.
        sampling_mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """
    intv_var_name_in_data = list(cause_data.keys())[0]
    cause_var_name = intv_var_name_in_data.replace("'", "")
    effect_var_name = list(effect_data.keys())[0]
    mediator_var_name = list(mediator_data.keys())[0]

    data = cause_data.copy()
    data.update(effect_data)
    data.update(mediator_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + mediator_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + mediator_var_name + "; \n"
    causal_graph = causal_graph + mediator_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "<->" + effect_var_name + "; \n"
    
    weight_func_lam, _ = general_cb_analysis(causal_graph = causal_graph, 
                                                           effect_var_name = effect_var_name, 
                                                           cause_var_name = cause_var_name, 
                                                           info_print= False)
    cb_data = general_causal_bootstrapping_intv(weight_func_lam = weight_func_lam, 
                                                dist_map = dist_map, 
                                                data = data, 
                                                intv_var_name_in_data = intv_var_name_in_data, 
                                                cause_intv_name_map = cause_intv_name_map,
                                                sampling_mode = sampling_mode, 
                                                random_state = random_state)

    return cb_data

def frontdoor_cf(
    cause_data: DataDict, 
    mediator_data: DataDict, 
    effect_data: DataDict, 
    dist_map: DistMap, 
    intv_dict: IntvDict, 
    n_sample: int,
    sampling_mode: Literal["fast", "robust"] = "fast", 
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform simulational frontdoor causal bootstrapping to de-confound the causal effect using the provided 
    observational data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        mediator_data (dict): A dictionary containing the mediator variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        intv_dict (dict): key: str, value: int/list/ndarray(M,), a dictionary containing the intervention variable names and their corresponding values.
        n_sample (int): The number of samples to be generated through bootstrapping.
        sampling_mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """

    cause_var_name = list(cause_data.keys())[0].replace("'", "")
    effect_var_name = list(effect_data.keys())[0]
    mediator_var_name = list(mediator_data.keys())[0]
    intv_var_name = list(intv_dict.keys())[0]
    cause_intv_name_map = {cause_var_name: intv_var_name}
    
    if cause_var_name == intv_var_name:
        warnings.warn("Intervention variable in intv_dict should be different from the cause variable name in cause_data. \
                      Automatically rewrite interventional variable as 'intv_{}'.".format(cause_var_name))
        intv_dict = {"intv_{}".format(cause_var_name): intv_dict[intv_var_name]}
    
    data = cause_data.copy()
    data.update(effect_data)
    data.update(mediator_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + mediator_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + mediator_var_name + "; \n"
    causal_graph = causal_graph + mediator_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "<->" + effect_var_name + "; \n"
    
    weight_func_lam, _ = general_cb_analysis(causal_graph = causal_graph, 
                                                           effect_var_name = effect_var_name, 
                                                           cause_var_name = cause_var_name, info_print= False)
    cb_data = general_causal_bootstrapping_cf(weight_func_lam = weight_func_lam, 
                                              dist_map = dist_map, 
                                              data = data, 
                                              cause_intv_name_map = cause_intv_name_map,
                                              intv_dict = intv_dict, 
                                              n_sample = n_sample, 
                                              sampling_mode = sampling_mode, 
                                              random_state = random_state)
    return cb_data
# %%

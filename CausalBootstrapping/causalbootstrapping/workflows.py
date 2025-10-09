
from causalbootstrapping.backend import *
from causalbootstrapping.expr import Expr
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
    info_print: bool = True, 
    idmode: str = "all", 
    idgreedy: bool = True
) -> Tuple[WeightBuilder, Expr]:

    """
    Perform pre-analysis for the given causal graph with intended cause(intenventional) and effect variables to 
    formulate the weight function.

    This function generates causal bootstrapping weight function and required distributions
    for a given causal graph and a specified intervention.

    Parameters:
    - causal_graph (str): A causal graph representing the relationships between variables.
    - effect_var_name (str): The name of the effect variable for the causal intervention.
    - cause_var_name (str): The name of the cause variable responsible for the intervention.
    - info_print (bool, optional): A boolean indicating whether to print the weight function and required distributions. Default is True.
    - idmode (str, optional): A string indicating the mode for identifying the interventional probability. It can be either 'all', 'shortest', 'mostmrg' or 'random'. Default is 'all'.
    - idgreedy (bool, optional): A boolean indicating whether to use the greedy strategy for identifying the interventional probability. Default is True.

    Returns:
    - weight_func_lam (function): A lambda function to calculate causal bootstrapping weights.
    - weight_func_str (str): The string representation of the causal bootstrapping weights function.

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
    id_str_all, id_eqn_all, identifiable = algs.idfixall(G = G,
                                                         X = set(cause_var_name), 
                                                         Y = set(effect_var_name), 
                                                         mode = idmode, 
                                                         greedy = idgreedy)

    if not identifiable:
        print("Not identifiable")
        return None, None

    if idmode != "all":
        id_str_all = [id_str_all]
        id_eqn_all = [id_eqn_all]
        
    simplicity_score = 0
    best_simp_score = 0
    
    cause_var = id_eqn_all[0].lhs.dov
    eff_var = id_eqn_all[0].lhs.num[0]
    
    valid_eqn_flag = False
    for i,id_eqn in enumerate(id_eqn_all):
        
        w_denom = id_eqn.rhs.den.copy()
        w_nom = id_eqn.rhs.num.copy()
        
        eff_var_cnt_nom = len([0 for w_nom_i in w_nom if eff_var.issubset(w_nom_i)])
        eff_var_cnt_denom = len([0 for w_denom_i in w_denom if eff_var.issubset(w_denom_i)])
        eff_var_cnt = eff_var_cnt_nom + eff_var_cnt_denom
        
        # Check if only one effect variable in the expression
        valid_cond1 = eff_var_cnt<=1
        # Check if interventional variable in the expression
        valid_cond2 = (any(cause_var.issubset(w_nom_i) for w_nom_i in w_nom) or any(cause_var.issubset(w_denom_i) for w_denom_i in w_denom))
        # Check if Pa(effect_var) or Pa(effect_var)\cause_var is in the nominator
        epsilo = id_eqn.rhs.mrg
        YuPa= epsilo.union(eff_var)
        XuYuPa = YuPa.union(cause_var)
        # if Y is in the Pa(X)
        YuPa_in_nom = YuPa in w_nom
        # if Y is not in the Pa(X) 
        XuYuPa_in_nom = XuYuPa in w_nom
        
        valid_cond3 = YuPa_in_nom or XuYuPa_in_nom
        # Check if it is a valid id_eqn
        valid_id_eqn = valid_cond1 and valid_cond2 and valid_cond3
        # Criteria to select the simplest id_eqn
        simplicity_score = 1/(len(w_nom)+len(w_denom))
        
        if valid_id_eqn and simplicity_score > best_simp_score:
            best_simp_score = simplicity_score
            
            if YuPa_in_nom:
                kernel_flag = False
                w_nom.remove(YuPa)
            if XuYuPa_in_nom:
                kernel_flag = True
                w_nom.remove(XuYuPa)
            
            best_w_nom = w_nom
            best_w_denom = w_denom
            best_id_str = id_str_all[i]
            best_id_eqn = id_eqn
            
            valid_eqn_flag = True
    
    if not valid_eqn_flag:
        warnings.warn("No automatic process available for current version of causalBootstrapping lib. Derive manually.")
        return None, None

    weight_func_lam = lambda dist_map, N, kernel, cause_intv_name_map: build_weight_function(best_id_eqn, 
                                                                                             dist_map, 
                                                                                             N, 
                                                                                             cause_intv_name_map, 
                                                                                             kernel)
    weight_func_expr = Expr(
        w_nom=best_w_nom,
        w_denom=best_w_denom,
        cause_var=cause_var,
        kernel_flag=kernel_flag
    )
    weight_func_str = weight_func_expr.tostr()
    wanted_dist = weight_func_expr.build_p_term(best_w_nom + best_w_denom)
    kernel_func = weight_func_expr.build_kernel_term()
    # print out interventional probability expression and required distributions
    if info_print:
        print("Interventional prob.:{}".format(best_id_str))    
        print("Causal bootstrapping weights function: {}".format(weight_func_str))
        print("Required distributions:")
        for i, dist in enumerate(wanted_dist):
            print("{}: {}".format(i+1, dist))
        if kernel_flag:
            print("Kernel function required:")
            for i, kernel_i in enumerate(kernel_func):
                print("{}: {}".format(i+1, kernel_i))

    return weight_func_lam, weight_func_expr

def general_causal_bootstrapping_simple(
    weight_func_lam: WeightBuilder, 
    dist_map: DistMap, 
    data: DataDict, 
    intv_var_name_in_data: str, 
    cause_intv_name_map: CauseIntvMap,
    kernel: Optional[KernelFunc] = None, 
    mode: Literal["fast", "robust"] = "fast", 
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
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
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
    bootstrapped_data, weights = bootstrapper(data = data, w_func = w_func, 
                                              intv_var_name_in_data = intv_var_name_in_data, 
                                              intv_var_name = intv_var_name,
                                              mode = mode,
                                              random_state = random_state)
    
    return bootstrapped_data

def general_causal_bootstrapping_simu(
    weight_func_lam: WeightBuilder, 
    dist_map: DistMap, 
    data: DataDict, 
    cause_intv_name_map: CauseIntvMap, 
    intv_dict: IntvDict, 
    n_sample: int, 
    kernel: Optional[KernelFunc] = None, 
    mode: Literal["fast", "robust"] = "fast", 
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
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
        random_state (int, optional): The random state for the bootstrapping. Defaults to None.
        
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """    
    N = list(data.values())[0].shape[0]
    w_func, _ = weight_func_lam(dist_map = dist_map, 
                                N = N, 
                                kernel = kernel, 
                                cause_intv_name_map = cause_intv_name_map)
    bootstrapped_data, weights = simu_bootstrapper(data = data, 
                                                   w_func = w_func, 
                                                   intv_dict = intv_dict, 
                                                   n_sample = n_sample, 
                                                   mode = mode, 
                                                   random_state = random_state)
    
    return bootstrapped_data

def backdoor_simple(
    cause_data: DataDict, 
    effect_data: DataDict, 
    confounder_data: DataDict, 
    dist_map: DistMap, 
    cause_intv_name_map: CauseIntvMap,
    kernel_intv: Optional[KernelFunc] = None, 
    mode: Literal["fast", "robust"] = "fast", 
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
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
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
    cb_data = general_causal_bootstrapping_simple(weight_func_lam = weight_func_lam, 
                                                  dist_map = dist_map, 
                                                  data = data, 
                                                  intv_var_name_in_data = cause_var_name, 
                                                  cause_intv_name_map = cause_intv_name_map,
                                                  kernel = _kernel_intv, 
                                                  mode = mode, 
                                                  random_state = random_state)

    return cb_data

def backdoor_simu(
    cause_data: DataDict, 
    effect_data: DataDict, 
    confounder_data: DataDict, 
    dist_map: DistMap, 
    intv_dict: IntvDict, 
    n_sample: int, 
    kernel_intv: Optional[KernelFunc] = None,
    mode: Literal["fast", "robust"] = "fast", 
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
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
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
    
    cb_data = general_causal_bootstrapping_simu(weight_func_lam = weight_func_lam, 
                                                dist_map = dist_map, 
                                                data = data, 
                                                cause_intv_name_map = cause_intv_name_map,
                                                intv_dict = intv_dict, 
                                                n_sample = n_sample, 
                                                kernel = _kernel_intv, 
                                                mode = mode, 
                                                random_state = random_state)
    return cb_data

def frontdoor_simple(
    cause_data: DataDict, 
    mediator_data: DataDict, 
    effect_data: DataDict, 
    dist_map: DistMap, 
    cause_intv_name_map: CauseIntvMap,
    mode: Literal["fast", "robust"] = "fast", 
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
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
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
    cb_data = general_causal_bootstrapping_simple(weight_func_lam = weight_func_lam, 
                                                  dist_map = dist_map, 
                                                  data = data, 
                                                  intv_var_name_in_data = intv_var_name_in_data, 
                                                  cause_intv_name_map = cause_intv_name_map,
                                                  mode = mode, 
                                                  random_state = random_state)

    return cb_data

def frontdoor_simu(
    cause_data: DataDict, 
    mediator_data: DataDict, 
    effect_data: DataDict, 
    dist_map: DistMap, 
    intv_dict: IntvDict, 
    n_sample: int,
    mode: Literal["fast", "robust"] = "fast", 
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
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.
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
    cb_data = general_causal_bootstrapping_simu(weight_func_lam = weight_func_lam, 
                                                dist_map = dist_map, 
                                                data = data, 
                                                cause_intv_name_map = cause_intv_name_map,
                                                intv_dict = intv_dict, 
                                                n_sample = n_sample, mode = mode, 
                                                random_state = random_state)
    return cb_data
# %%

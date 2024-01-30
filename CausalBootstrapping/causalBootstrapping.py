
import grapl.algorithms as algs
import grapl.dsl as dsl
import inspect
import numpy as np
import random

def weight_compute(weight_func, data, intv_var):
    """
    Compute causal bootstrapping weights for the given weight function and input observational data.
    
    Parameters:
        weight_func (function): The causal bootstrapping weight function to be used.
        data (dict): A dictionary containing variable names as keys and their corresponding data arrays as values.
        intv_var (dict): A dictionary containing the intervention variable name as a key and its data array as the value.
    
    Returns:
        numpy.ndarray: An array containing the computed causal bootstrapping weights for each data point.
    """

    kwargs = {key: 0 for key in data}
    N = list(data.values())[0].shape[0]
    weights = np.zeros((N))
    for i in range(N):
        for key, value in data.items():
            if key in kwargs:
                value_i =  np.array(value[i])
                if value_i.ndim == 0:
                    value_i = value_i.reshape(-1)
                kwargs[key] = np.array(value[i])
        for key, value in intv_var.items():
            value_i =  np.array(value[i])
            if value_i.ndim == 0:
                value_i = value_i.reshape(-1)
            kwargs[key] = value_i
        weights[i] = weight_func(**kwargs)
    return weights

def weight_func(intv_prob, dist_map, N, kernel = None):
    """
    Generate the causal bootstrapping weight function using the identified interventional probability and 
    corresponding distribution maps.

    Parameters:
        intv_prob (grapl.expr object): The identified interventional probability expression.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        N (int): The number of data points in the dataset.
        kernel (function, optional): The kernel function to be used in the weight computation. Defaults to None.

    Returns:
        function: The corresponding causal bootstrapping weight function.
    """
    
    def divide_functions(*funcs):
        def division(**kwargs):
            param_names = inspect.signature(funcs[0]).parameters
            kwargs = {key.replace("'","_prime"): value for key, value in kwargs.items()}
            param = {key : kwargs[key] for key in param_names}
            result = funcs[0](**param)
            if len(w_nom) > 1:
                for nom_i in range(1,len(w_nom)):
                    param_names = inspect.signature(funcs[nom_i]).parameters
                    param = {key : kwargs[key] for key in param_names}
                    result *= funcs[nom_i](**param)
            for denom_i in range(len(w_nom),len(w_denom)+len(w_nom)):
                param_names = inspect.signature(funcs[denom_i]).parameters
                param = {key : kwargs[key] for key in param_names}
                result /= funcs[denom_i](**param)
            if cause_kernel_flag:
                param_names = inspect.signature(funcs[-1]).parameters
                param = {key : kwargs[key] for key in param_names}
                result *= funcs[-1](**param)
            result *= (lambda n: 1/n)(N)
            return result
        return division    
    
    cause_var = intv_prob.lhs.dov
    eff_var = intv_prob.lhs.num[0]
    w_denom = intv_prob.rhs.den
    w_denom = [sorted(w_denom[i]) for i in range(len(w_denom))]
    w_nom = intv_prob.rhs.num
    w_nom = [sorted(w_nom[i]) for i in range(len(w_nom))]
    epsilo = intv_prob.rhs.mrg
    pa_var = sorted(epsilo.union(eff_var).union(cause_var))
    if pa_var in w_nom:
        w_nom.remove(pa_var)
        cause_kernel_flag = True
        funcs = [dist_map[tuple(w_nom_i)] for w_nom_i in w_nom] + [dist_map[tuple(w_denom_i)] for w_denom_i in w_denom] + [kernel]
    else:
        pa_var = set(pa_var)
        pa_var -= cause_var
        pa_var = sorted(list(pa_var))
        w_nom.remove(list(pa_var))
        cause_kernel_flag = False
        funcs = [dist_map[tuple(w_nom_i)] for w_nom_i in w_nom] + [dist_map[tuple(w_denom_i)] for w_denom_i in w_denom]
    
    weight_func = divide_functions(*funcs)

    return weight_func

def bootstrapper(data, weights, intv_var_name_in_data, mode = "fast"):
    """
    Perform bootstrapping on the input observational data using the provided weights.
    
    Parameters:
        data (dict): A dictionary containing variable names as keys and their corresponding data arrays as values.
        weights (numpy.ndarray): An array containing the weights computed for each data point given the interventional values as all possible observational values.
        intv_var_name_in_data (list): A list of variable names of observational data used as the interventional values.
        mode (str, optional): The mode for bootstrapping. Options: 'fast' or 'robust'. Defaults to 'fast'.
    
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """
    
    if mode == "robust":
         weights = np.log(weights+np.e)
        
    var_names = list(data.keys())
    N = data[var_names[0]].shape[0]
    bootstrap_data_keys = var_names + ["intv_"+intv_var_name_in_data[i] for i in range(len(intv_var_name_in_data))]
    bootstrap_data = dict(zip(bootstrap_data_keys, [[] for i in range(len(bootstrap_data_keys))]))
    intv_values = np.array([data[v].reshape(-1) for v in intv_var_name_in_data])
    intv_values_combined = list(set(zip(*intv_values)))

    for i, v in enumerate(intv_values_combined):
        weight_intv = weights[:, i]
        n_sample = np.where(np.all(intv_values[:len(v), :].T == np.array(v), axis=1))[0].shape[0]
        sample_indices = random.choices([n for n in range(N)], weights = weight_intv, k = n_sample)
        
        for var in var_names:
            bootstrap_data[var].append(data[var][sample_indices])
            
        for intv_var in intv_var_name_in_data:
            bootstrap_data["intv_"+intv_var].append(np.array([v for i in range(len(sample_indices))]))
    
    for var in bootstrap_data_keys:
        bootstrap_data[var] = np.vstack(bootstrap_data[var])
        bootstrap_data[var.replace("'", "")] = bootstrap_data.pop(var)
    return bootstrap_data

def simu_bootstrapper(data, weight_func, intv_var_value, n_sample, mode = "fast"):
    """
    Perform simulational bootstrapping on the input observational data using the provided weight function and 
    designated intervion values.

    Parameters:
        data (dict): A dictionary containing variable names as keys and their corresponding data arrays as values.
        weight_func (function): The causal bootstrapping weight function.
        intv_var_value (dict): A dictionary containing the intervention variable name as a key and its data array as the value.
        n_sample (int): The number of samples to be generated through bootstrapping.
        mode (str, optional): The mode for bootstrapping. Options: 'fast' or 'robust'. Defaults to 'fast'.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """
    
    weights = weight_compute(weight_func, data, intv_var_value)
    if mode == "robust":
         weights = np.log(weights+np.e)
         
    intv_var_name = list(intv_var_value.keys())
    var_names = list(data.keys())
    N = data[var_names[0]].shape[0]
    # var_names = [item for item in var_names if (item in var_names) and (item.replace("'","") not in intv_var_name)]
    bootstrap_data = {}
    sample_indices = random.choices([n for n in range(N)], weights = weights, k = n_sample)
    
    for var in var_names:
        bootstrap_data[var.replace("'","")] = data[var][sample_indices]
    for intv_var in intv_var_name:
        intv_value_i= np.array(intv_var_value[intv_var])
        if intv_value_i.ndim == 1:
            intv_value_i = intv_value_i.reshape(-1,1)
        bootstrap_data["intv_"+intv_var if intv_var in bootstrap_data.keys() else intv_var] = intv_value_i[sample_indices]
    return bootstrap_data, weights

def general_cb_analysis(causal_graph, effect_var_name, cause_var_name, info_print = True):
    
    """
    Perform pre-analysis for the given causal graph with intended cause(intenventional) and effect variables to 
    formulate the weight function.

    This function generates causal bootstrapping weight function and required distributions
    for a given causal graph and a specified intervention.

    Parameters:
    - causal_graph (grapl.admg.ADMG object): A causal graph representing the relationships between variables.
    - effect_var_name (str): The name of the effect variable for the causal intervention.
    - cause_var_name (str): The name of the cause variable responsible for the intervention.

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
    id_str_all, id_eqn_all, identifiable = algs.idfixall(G, set(cause_var_name), set(effect_var_name), mode = "all")

    if not identifiable:
        print("Not identifiable")
        return None, None

    simplicity_score = 0
    best_simp_score = 0
    
    cause_var = id_eqn_all[0].lhs.dov
    eff_var = id_eqn_all[0].lhs.num[0]
    for i,id_eqn in enumerate(id_eqn_all):
        
        w_denom = id_eqn.rhs.den
        w_denom = [sorted(w_denom[i]) for i in range(len(w_denom))]
        w_nom = id_eqn.rhs.num
        w_nom = [sorted(w_nom[i]) for i in range(len(w_nom))]
        
        eff_var_cnt_nom = len([0 for w_nom_i in w_nom for element in w_nom_i if list(eff_var)[0] in element])
        eff_var_cnt_denom = len([0 for w_denom_i in w_denom for element in w_denom_i if list(eff_var)[0] in element])
        eff_var_cnt = eff_var_cnt_nom + eff_var_cnt_denom
        
        # Check if only one effect variable in the expression
        valid_cond1 = eff_var_cnt<=1
        # Check if interventional variable in the expression
        valid_cond2 = ((any(list(cause_var)[0] in w_nom_i for w_nom_i in w_nom)) or (any(list(cause_var)[0] in w_denom_i for w_denom_i in w_denom)))
        # Check if Pa(effect_var) or Pa(effect_var)\cause_var is in the nominator
        epsilo = id_eqn.rhs.mrg
        YuPa= sorted(epsilo.union(eff_var))
        XuYuPa = sorted(set(YuPa).union(cause_var))
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
            
    weight_func_lam = lambda dist_map, N, kernel: weight_func(best_id_eqn, dist_map, N, kernel)        
    
    wanted_dist = []
    w_nom_str = ""
    for nom in best_w_nom:
        joint_var_str = ""
        for i in range(len(nom)):
            joint_var_str += nom[i]
            if i != len(nom)-1:
                joint_var_str += ","
        w_nom_str += "P(" + joint_var_str + ")"
        wanted_dist.append("P(" + joint_var_str + ")")
    if kernel_flag:
        kernel_func_str = "K("+list(cause_var)[0]+","+list(cause_var)[0]+"')"
        w_nom_str += kernel_func_str
    
    w_denom_str = ""
    for denom in best_w_denom:
        joint_var_str = ""
        for i in range(len(denom)):
            joint_var_str += denom[i]
            if i != len(denom)-1:
                joint_var_str += ","
        w_denom_str += "P(" + joint_var_str + ")"
        wanted_dist.append("P(" + joint_var_str + ")")
        
    weight_func_str = "["+w_nom_str+"]"+"/N*[" +w_denom_str + "]"
    # print out interventional probability expression and required distributions
    if info_print:
        print("Interventional prob.:{}".format(best_id_str))    
        print("Causal bootstrapping weights function: {}".format(weight_func_str))
        print("Required distributions:")
        i = 0
        for dist in wanted_dist:
            i += 1
            print("{}: {}".format(i, dist))
        if kernel_flag:
            print("Kernel function required: {}".format(kernel_func_str))
    
    return weight_func_lam, weight_func_str

def general_causal_bootstrapping_simple(weight_func_lam, dist_map, data, intv_var_name, kernel = None, mode = "fast"):
    """
    Perform causal bootstrapping for a general causal graph if it is identifiable using a specified weight function. 
    This function is designed to create a bootstrapped dataset based on the specified intervention variables and the 
    weighting function. The function allows for an optional kernel to be specified.

    Parameters:
        weight_func_lam (function): A lambda or function that calculates weights. It should take 'dist_map', 'N', and optionally 'kernel' as arguments.
        dist_map (dict): An object representing a distance map, which is used by 'weight_func_lam' to calculate weights.
        data (dict): A dictionary containing the dataset. Keys are variable names and values are corresponding data arrays.
        intv_var_name (str): The name of the intervention variable. This variable should be a key in the 'data' dictionary.
        kernel (function, optional): An optional kernel function that can be used in weight computation. Default is None.
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """
    cause_data = [value for key, value in data.items() if intv_var_name in key][0]
    intv_var_name_in_data = [key for key, value in data.items() if intv_var_name in key][0]
    N = cause_data.shape[0]
    if kernel is not None:
        intv_var_name = [param for param in inspect.signature(kernel).parameters if param != intv_var_name_in_data][0]
    
    w_func = weight_func_lam(dist_map = dist_map, N = N, kernel = kernel)
    cause_unique = set(cause_data.reshape(-1))
    weights = np.zeros((N, len(cause_unique)))
    for i, y in enumerate(cause_unique):
        weights[:,i]=weight_compute(weight_func = w_func, data = data ,intv_var = {intv_var_name: [y for i in range(N)]})
        
    bootstrapped_data = bootstrapper(data = data, weights = weights, intv_var_name_in_data = [intv_var_name_in_data], mode = mode)
    
    return bootstrapped_data

def general_causal_bootstrapping_simu(weight_func_lam, dist_map, data, intv_var_value, n_sample, kernel = None, mode = "fast"):
    """
    Perform simulational causal bootstrapping for a general causal graph if it is identifiable using a specified weight function. 
    This function is designed to create a bootstrapped dataset based on the specified intervention {variable:values} pair and the weighting function.
    The function allows for an optional kernel to be specified.

    Parameters:
        weight_func_lam (function): A lambda or function that calculates weights. It should take 'dist_map', 'N', and optionally 'kernel' as arguments.
        dist_map (dict): An object representing a distance map, which is used by 'weight_func_lam' to calculate weights.
        data (dict): A dictionary containing the dataset. Keys are variable names and values are corresponding data arrays.
        intv_var_value (dict): A dictionary containing the intervention variable name as a key and its data array as the value.
        n_sample (int): The number of samples to be generated through bootstrapping.
        kernel (function, optional): An optional kernel function that can be used in weight computation. Default is None.
        mode (str, optional): A string indicating the bootstrapping mode. It can be either 'fast' or 'robust' depending on the implementation. Default is 'fast'.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding bootstrapped data arrays as values.
    """    
    N = list(data.values())[0].shape[0]
    w_func = weight_func_lam(dist_map = dist_map, N = N, kernel = kernel)
    bootstrapped_data, weights = simu_bootstrapper(data = data, weight_func = w_func, intv_var_value = intv_var_value, n_sample = n_sample, mode = mode)
    
    return bootstrapped_data

def backdoor_simple(cause_data, effect_data, confounder_data, dist_map, kernel_intv = None):
    """
    Perform backdoor causal bootstrapping to de-confound the causal effect using the provided observational 
    data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        confounder_data (dict): A dictionary containing the confounder variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        kernel_intv (function, optional): The kernel function to be used in the backdoor bootstrapping for the cause variable. Defaults to None.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """
    
    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    confounder_var_name = list(confounder_data.keys())[0]
    
    data = cause_data.copy()
    data.update(effect_data)
    data.update(confounder_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + confounder_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + cause_var_name + "; \n"
    
    weight_func_lam, weight_func_str = general_cb_analysis(causal_graph = causal_graph, 
                                                           effect_var_name = effect_var_name, 
                                                           cause_var_name = cause_var_name, info_print= False)
    
    if kernel_intv is None:
        intv_var_name = "intv_" + cause_var_name
        kernel_intv = eval("lambda "+intv_var_name+","+cause_var_name+": 1 if "+intv_var_name+"==" + cause_var_name + " else 0")
    else:
        kernel_params = set(inspect.signature(kernel_intv).parameters.keys())
        intv_var_name = list(kernel_params - set(data.keys()))
        if len(intv_var_name) == 0:
            intv_var_name = "intv_" + cause_var_name
        else:
            intv_var_name = intv_var_name[0]
    cb_data = general_causal_bootstrapping_simple(weight_func_lam = weight_func_lam, 
                                                  dist_map = dist_map, data = data, 
                                                  intv_var_name = cause_var_name, kernel = kernel_intv)

    return cb_data

def backdoor_simu(cause_data, effect_data, confounder_data, dist_map, intv_value, n_sample, kernel_intv = None):
    """
    Perform simulational backdoor causal bootstrapping to de-confound the causal effect using the provided 
    observational data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        confounder_data (dict): A dictionary containing the confounder variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        intv_value (list): A list containing the interventional value.
        n_sample (int): The number of samples to be generated through bootstrapping.
        kernel_intv (function, optional): The kernel function to be used in the backdoor adjustment for the cause variable. Defaults to None.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """
 	
    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    confounder_var_name = list(confounder_data.keys())[0]
    
    data = cause_data.copy()
    data.update(effect_data)
    data.update(confounder_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + confounder_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + confounder_var_name + "->" + cause_var_name + "; \n"
    
    weight_func_lam, weight_func_str = general_cb_analysis(causal_graph = causal_graph, 
                                                           effect_var_name = effect_var_name, 
                                                           cause_var_name = cause_var_name, info_print= False)
    
    if kernel_intv is None:
        intv_var_name = "intv_" + cause_var_name
        kernel_intv = eval("lambda "+intv_var_name+","+cause_var_name+": 1 if "+intv_var_name+"==" + cause_var_name + " else 0")
    else:
        kernel_params = set(inspect.signature(kernel_intv).parameters.keys())
        intv_var_name = list(kernel_params - set(data.keys()))
        if len(intv_var_name) == 0:
            intv_var_name = "intv_" + cause_var_name
        else:
            intv_var_name = intv_var_name[0]
    cb_data = general_causal_bootstrapping_simu(weight_func_lam = weight_func_lam, 
                                                dist_map = dist_map, data = data, 
                                                intv_var_value = {intv_var_name: intv_value}, 
                                                n_sample = n_sample, kernel = kernel_intv)
    return cb_data

def frontdoor_simple(cause_data, mediator_data, effect_data, dist_map):
    """
    Perform fontdoor causal bootstrapping to de-confound the causal effect using the provided observational 
    data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        mediator_data (dict): A dictionary containing the mediator variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """

    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    mediator_var_name = list(mediator_data.keys())[0]
    
    data = {cause_var_name+"'": list(cause_data.values())[0]}
    data.update(effect_data)
    data.update(mediator_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + mediator_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + mediator_var_name + "; \n"
    causal_graph = causal_graph + mediator_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "<->" + effect_var_name + "; \n"
    
    weight_func_lam, weight_func_str = general_cb_analysis(causal_graph = causal_graph, 
                                                           effect_var_name = effect_var_name, 
                                                           cause_var_name = cause_var_name, info_print= False)
    cb_data = general_causal_bootstrapping_simple(weight_func_lam = weight_func_lam, 
                                                  dist_map = dist_map, data = data, 
                                                  intv_var_name = cause_var_name)

    return cb_data

def frontdoor_simu(cause_data, mediator_data, effect_data, dist_map, intv_value, n_sample):
    """
    Perform simulational frontdoor causal bootstrapping to de-confound the causal effect using the provided 
    observational data and distribution maps.

    Parameters:
        cause_data (dict): A dictionary containing the cause variable name as a key and its data array as the value.
        mediator_data (dict): A dictionary containing the mediator variable name as a key and its data array as the value.
        effect_data (dict): A dictionary containing the effect variable name as a key and its data array as the value.
        dist_map (dict): A dictionary mapping tuples of variable combinations to their corresponding distribution functions.
        intv_value (list): A list containing the interventional value.
        n_sample (int): The number of samples to be generated through bootstrapping.

    Returns:
        dict: A dictionary containing variable names as keys and their corresponding de-confounded data arrays as values.
    """
    
    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    mediator_var_name = list(mediator_data.keys())[0]
    
    data = {cause_var_name+"'": list(cause_data.values())[0]}
    data.update(effect_data)
    data.update(mediator_data)
    
    causal_graph = cause_var_name + ";" + effect_var_name + ";" + mediator_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "->" + mediator_var_name + "; \n"
    causal_graph = causal_graph + mediator_var_name + "->" + effect_var_name + "; \n"
    causal_graph = causal_graph + cause_var_name + "<->" + effect_var_name + "; \n"
    
    weight_func_lam, weight_func_str = general_cb_analysis(causal_graph = causal_graph, 
                                                           effect_var_name = effect_var_name, 
                                                           cause_var_name = cause_var_name, info_print= False)
    intv_var_name = cause_var_name
    cb_data = general_causal_bootstrapping_simu(weight_func_lam = weight_func_lam, 
                                                dist_map = dist_map, data = data, 
                                                intv_var_value = {intv_var_name: intv_value}, 
                                                n_sample = n_sample)
    return cb_data
import numpy as np

def remove_outgoing(G, node):
    """Remove all outgoing edges (directed and bidirected) from the node.
       This modifies the graph in-place.
    """
    # Remove this node from its children's parents
    for child in G.ch({node}):
        G.vars[child].parents = G.vars[child].parents.difference({node})
    G.vars[node].children = set()
    # for bidirect in G.bi({node}):
    #     G.vars[bidirect].bidirects = G.vars[bidirect].bidirects.difference({node})
    # G.vars[node].bidirects = set()
    return G

def remove_incoming(G, node):
    for parent in G.pa({node}):
        G.vars[parent].children = G.vars[parent].children.difference({node})
    G.vars[node].parents = set()
    for bidirect in G.bi({node}):
        G.vars[bidirect].bidirects = G.vars[bidirect].bidirects.difference({node})
    G.vars[node].bidirects = set()
    return G

def gumbel_max(weights: np.ndarray) -> int:
    """
    Apply the Gumbel-max trick to sample indices based on the provided weights.
    
    Parameters:
        weights (numpy.ndarray): The weights for sampling.
    
    Returns:
        int: The index of the sampled weight.
    """
    gumbel_noise = -np.log(-np.log(np.random.rand(*weights.shape)))
    return np.argmax(np.log(weights) + gumbel_noise)

def weight_func_parse(id_formular):
    
        cause_var = id_formular.lhs.dov
        eff_var = id_formular.lhs.num[0]
    
        w_denom = id_formular.rhs.den.copy()
        w_nom = id_formular.rhs.num.copy()
        
        eff_var_cnt_nom = len([0 for w_nom_i in w_nom if eff_var.issubset(w_nom_i)])
        eff_var_cnt_denom = len([0 for w_denom_i in w_denom if eff_var.issubset(w_denom_i)])
        eff_var_cnt = eff_var_cnt_nom + eff_var_cnt_denom
        
        # Check if only one effect variable in the expression
        valid_cond1 = eff_var_cnt<=1
        # Check if interventional variable in the expression
        valid_cond2 = (any(cause_var.issubset(w_nom_i) for w_nom_i in w_nom) or any(cause_var.issubset(w_denom_i) for w_denom_i in w_denom))
        # Check if Pa(effect_var) or Pa(effect_var)\cause_var is in the nominator
        epsilo = id_formular.rhs.mrg
        YuPa= epsilo.union(eff_var)
        XuYuPa = YuPa.union(cause_var)
        # if Y is in the Pa(X)
        YuPa_in_nom = YuPa in w_nom
        # if Y is not in the Pa(X) 
        XuYuPa_in_nom = XuYuPa in w_nom
        
        valid_cond3 = YuPa_in_nom or XuYuPa_in_nom
        # Check if it is a valid id_eqn
        valid_id_eqn = valid_cond1 and valid_cond2 and valid_cond3
        
        if valid_id_eqn:
            
            if YuPa_in_nom:
                kernel_flag = False
                w_nom.remove(YuPa)
            if XuYuPa_in_nom:
                kernel_flag = True
                w_nom.remove(XuYuPa)
            
            return w_nom, w_denom, kernel_flag, valid_id_eqn
        
        else:
            return None, None, None, valid_id_eqn
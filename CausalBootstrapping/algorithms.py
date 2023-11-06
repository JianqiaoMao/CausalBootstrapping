# -*- coding: utf-8 -*-
"""
Implements algorithms for processing of structural causal models represented
by acyclic directed mixed graphs (ADMGs).

(CC BY-SA 4.0) 2021. If you use this code, please cite: M.A. Little, R. Badawy,
2019, "Causal bootstrapping", arXiv:1910.09648
"""

import grapl.expr as expr
import grapl.eqn as eqn
import grapl.condind as condind
from idfixtree import IDFixSeqTree
import copy as cp
import numpy as np
from itertools import product

def befintervvar(G, X = {}, Y = {}):
    # Collate variable names
    V = G.nodes()
    # Handle effects
    if not Y:
        Y = V.difference(X)
    # Find Y* = an_G(V)_{V\A}(Y)    
    A = X
    VdA = V.difference(A)
    G_sub_VdA = G.sub(VdA)
    Y_star = G_sub_VdA.an(Y)
    
    return Y_star

def fixd(G, Y_star):
    # Find subgraph G(V)_Y*
    G_sub_Ys = G.sub(Y_star)
    # Find corresponding districts
    D_y = G_sub_Ys.districts()
    
    return D_y

def fixdseq(G, Y_star, D, X = {}, Y = {}):
    # Initialise fixing sequence searcher
    seq_searcher = IDFixSeqTree(G = G, Y_star = Y_star, X = X, Y = Y)
    # Find all possible fixing sequences and their fixed graphs
    fix_node_seqs, fix_graph_seqs, identifiable = seq_searcher.searchdfixseq(D)
    
    if identifiable:
        return fix_node_seqs, fix_graph_seqs, identifiable
    else:
        return None, None, identifiable

def dexprconstru(G, fix_node_seq, fix_graph_seq):
    # Start with the joint distribution for all nodes
    D_expr = expr.Expr(num=[G.nodes()])
    graph_seq = [G] + fix_graph_seq[:-1]
    for i in range(len(fix_node_seq)):
        fix = fix_node_seq[i]
        G_fixed = graph_seq[i]
        # Fix or marginalize (Corollary 33) the accumulated expression
        dispa_fix = G_fixed.dispa(fix)
        fix_ch = G_fixed.ch({fix})
        if len(fix_ch) == 0:
            D_expr.fixmarginal(fix)
        else:
            D_expr.fix(fix, dispa_fix)

    D_expr.simplify()
    D_expr_str = D_expr.tostr()
    D_num = D_expr.num
    D_den = D_expr.den
    return D_expr, D_expr_str, D_num, D_den

def ideqnconstru(Y_star, X, Y, D_exprs):
    Y_mrg = Y_star.difference(Y)
    id_expr = expr.Expr(mrg=Y_mrg)
    
    for ex in D_exprs:
        clash_vars = id_expr.mrg.difference(id_expr.mrg.difference(ex.mrg))
        for old_var in clash_vars:
            new_var = old_var + chr(39)
            ex.subsvar(old_var, new_var)
        id_expr.num = id_expr.num + ex.num
        id_expr.den = id_expr.den + ex.den
        id_expr.mrg = id_expr.mrg.union(ex.mrg)

    id_expr.simplify()
    lhs = expr.Expr()
    lhs.addvars(num=Y, dov=X)
    id_eqn = eqn.Eqn(lhs, id_expr)
    id_str = id_eqn.tocondstr()
    
    return id_str, id_eqn

def idfixall(G, X = {}, Y ={}, mode = "shortest"):
    """
    Search for interventional probability expressions for the given causal graph
    and retrieve results based on the selected mode.
    A general implementation of Richardson et al.'s Theorem 60 for causal effect 
    identification from given ADMG. See: T.S. Richardson, J.M. Robins, I. Shpitser, 2012:
    "Nested Markov properties for acyclic directed mixed graphs", UAI'12.
    
    Parameters:
    G (ADMG)           -- A mixed causal graph object
    X (Set of Strings) -- Set of interventional variables, where each string is a random variable name (must not be empty)
    Y (Set of Strings) -- Set of variables on which interventional expression is represented, where each string is a random
                          variable name (if empty, then all variables in G except for those in X)
    mode (String)      -- The mode for selecting interventional equations {'shortest', 'mostmrg', 'random', or 'all'}, default 'shortest'.
    
    Returns:
    Depending on the selected 'mode', it returns different results:
    - 'shortest': a Latex string of identified equation with the least number of distributions, its corresponding Eqn object, tracking information of fixing, and a boolean indicating if the causal effect is identifiable.
    - 'mostmrg': a Latex string of identified equation with the most number marginalized variables, its corresponding Eqn object, tracking information of fixing, and a boolean indicating if the causal effect is identifiable.
    - 'random': A randomly selected Latex string of identified equation, its corresponding Eqn object, tracking information of fixing, and a boolean indicating if the causal effect is identifiable.
    - 'all': A list of all Latex string of identified equations, a list of their corresponding Eqn objects, tracking information of fixing, and a boolean indicating if the causal effect is identifiable.
    If the causal effect is not identifiable: returns '', None, None, False
    """
    
    Y_star = befintervvar(G, X, Y)
    D_y = fixd(G, Y_star)
    D_exprs = []
    for D in D_y:
        fix_node_seqs, fix_graph_seqs, identifiable = fixdseq(G, Y_star, D, X, Y)
        if not identifiable:
            return None, None, False
        D_exprs.append([])
        D_expr_num = []
        D_expr_den = []
        for i in range(len(fix_node_seqs)):
            D_expr, D_expr_str, D_num, D_den = dexprconstru(G, fix_node_seqs[i], fix_graph_seqs[i])
            if (D_num not in D_expr_num) and (D_den not in D_expr_den):
                D_exprs[-1].append(D_expr)
                D_expr_num.append(D_num)
                D_expr_den.append(D_den)

    D_exprs_cartprod = list(product(*D_exprs))

    id_str_all =[]
    id_eqn_all = []
    for exprs in D_exprs_cartprod:
        id_str, id_eqn = ideqnconstru(Y_star, X, Y, exprs)
        id_str_all.append(id_str)
        id_eqn_all.append(id_eqn)
    
    if mode == "shortest":
        shortest_dist_cnt = np.inf
        shortest_id_eqn = None
        shortest_id_str = None
        for i,id_eqn in enumerate(id_eqn_all):
            w_denom = id_eqn.rhs.den
            w_nom = id_eqn.rhs.num
            dist_cnt = len(w_denom) + len(w_nom)
            if dist_cnt < shortest_dist_cnt:
                shortest_dist_cnt = dist_cnt
                shortest_id_eqn = id_eqn
                shortest_id_str = id_str_all[i]
        return shortest_id_str, shortest_id_eqn, identifiable
    
    elif mode == "mostmrg":
        most_mrg_cnt = -1
        most_mrg_id_eqn = None
        most_mrg_id_str = None
        most_mrg_track_info = None
        for i, id_eqn in enumerate(id_eqn_all):
            mrg_cnt = len(id_eqn.rhs.mrg)
            if mrg_cnt > most_mrg_cnt:
                most_mrg_cnt = mrg_cnt
                most_mrg_id_eqn = id_eqn
                most_mrg_id_str = id_str_all[i]
        return most_mrg_id_str, most_mrg_id_eqn, identifiable
    
    elif mode == "random":
        rd_id_idx = np.random.randint(0, len(id_eqn_all))
        rd_id_eqn = id_eqn_all[rd_id_idx]
        rd_id_str = id_str_all[rd_id_idx]
        return rd_id_str, rd_id_eqn, identifiable
    
    elif mode == "all":
        return id_str_all, id_eqn_all, identifiable
    
    else:
        raise ValueError("'{}' is not a valid value for mode. Valid values are: 'shortest', 'mostmrg', 'random', 'all'. Default: 'shortest'. ".format(mode))

def idfixing(G, X={}, Y={}):
    """A special case for idfixall function with mode = 'random', which returns a randomly selected Latex string of 
       identified equation, its corresponding Eqn object, and a boolean indicating if the causal effect is identifiable.

       Parameters:
       G (ADMG)           -- A mixed causal graph object
       X (Set of Strings) -- Set of interventional variables, where each string is a random variable name (must not be empty)
       Y (Set of Strings) -- Set of variables on which interventional expression is represented, where each string is a random
                             variable name (if empty, then all variables in G except for those in X)

       Returns: (String, Eqn, Boolean)
       If interventional distribution is identifiable, returns a randomly selected identified 
       equation as a Latex String, with the corresponding Eqn object, and True. Otherwise, 
       returns '', None, False
    """
    id_str, id_eqn, identifiable = idfixall(G, X, Y, mode = "random")
    if identifiable:
        return id_str, id_eqn, identifiable
    else:
        return "", None, identifiable

def dagfactor(G, Y={}, simplify=True):
    """Factorized distribution for DAGs (ADMGs with no bidirects).

       Parameters:
       G (ADMG)           -- DAG representing the causal graph (must not have bidirects)
       Y (Set of Strings) -- Set of effect variables (if empty, all variables in G)
       simplify (Boolean) -- If True, final expression is explicitly simplified

       Returns: (String, Expr, Boolean)
       If G is a DAG, factored, Latex-format equation String, corresponding Eqn object, and True.
       If G is not a DAG then '', None, False.
    """

    # Check to make sure there are no latent variables
    if not G.isdag():
        return "", None, False

    # Collate variable names
    V = G.nodes()

    # Handle marginals
    if not Y:
        Y = V
    Y_mrg = V.difference(Y)
    fac_expr = expr.Expr(mrg=Y_mrg)

    # Construct factorized expression over all nodes
    for node in V:
        fac_expr.num = fac_expr.num + [G.pa({node}).union({node})]
        fac_expr.den = fac_expr.den + [G.pa({node})]

    # Final expression simplification
    if simplify:
        fac_expr.simplify()

    # Construct Latex expression
    lhs = expr.Expr()
    lhs.addvars(num=Y)
    fac_eqn = eqn.Eqn(lhs, fac_expr)
    dist_str = fac_eqn.tocondstr()

    return dist_str, fac_eqn, True
    

def truncfactor(G, X={}, Y={}, prefactor=True):
    """Truncated factorization ("g-formula") for DAGs (ADMGs with no bidirects).

       Parameters:
       G (ADMG) -- DAG object representing the causal graph (must not have bidirects)
       X (Set of Strings)  -- Interventional variables where each string is a random variable name (must not be empty)
       Y (Set of Strings)  -- Effect variables, each sring is a random variable name (if empty, all variables in G other than the set X)
       prefactor (Boolean) -- If True, joint distribution is chain factored before fixing

       Returns: (String, Expr, Boolean)
       If G is a DAG, factored, Latex-formated interventonal distribution string, corresponding Eqn object,
       and True. Otherwise, returns '', None, False.
    """

    # Must supply the set of interventional variables
    if not X:
        return '', None, False

    # Check there are no latent variables
    if not G.isdag():
        return '', None, False

    # Collate variable names
    V = G.nodes()

    # Initialize accumulated DAG and interventional expression, p(V)
    G_fixed = cp.deepcopy(G)
    if prefactor:
        dummy, D_eqn, dummy_bool = dagfactor(G)        # Start with factorized joint distribution
        D_expr = cp.copy(D_eqn.rhs)
        # dummy, D_expr, dummy_bool = dagfactor(G)        # Start with factorized joint distribution
    else:
        D_expr = expr.Expr(num=[V])                     # Start with the unfactored joint distribution

    # Apply fixing sequence
    nodes_fix = cp.deepcopy(X)
    while (len(nodes_fix) > 0):
        
        # Select next fixing node in sequence
        fix = nodes_fix.pop()
        
        # Fix or marginalize (Corollary 33) the accumulated expression
        dispa_fix = G_fixed.dispa(fix)
        fix_ch = G_fixed.ch({fix})
        if len(fix_ch) == 0:
            D_expr.fixmarginal(fix)
        else:
            D_expr.fix(fix, dispa_fix)

        # Fix node (remove all its incoming edges) in accumulated DAG,
        # remove from remaining fixing sequence
        G_fixed.fix(fix)
        nodes_fix = nodes_fix.difference({fix})

    # Simplify final expression
    D_expr.simplify()

    # Handle marginals
    if not Y:
        Y = V.difference(X)
    Y_mrg = (V.difference(Y)).difference(X)
    id_expr = expr.Expr(mrg=Y_mrg)

    # Avoid naming clashes with dummy marginal variables
    clash_vars = id_expr.mrg.difference(id_expr.mrg.difference(D_expr.mrg))
    for old_var in clash_vars:
        new_var = old_var + chr(39)
        D_expr.subsvar(old_var, new_var)
    id_expr.num = id_expr.num + D_expr.num
    id_expr.den = id_expr.den + D_expr.den
    id_expr.mrg = id_expr.mrg.union(D_expr.mrg)

    # Final simplification
    id_expr.simplify()

    # Construct Latex expression
    lhs = expr.Expr()
    lhs.addvars(num=Y, dov=X)
    id_eqn = eqn.Eqn(lhs, id_expr)
    id_str = id_eqn.tocondstr()

    return id_str, id_eqn, True


def admgfactor(G, Y={}):
    """Factorized distribution for ADMGs.

       Parameters:
       G (ADMG)           -- A mixed causal graph
       Y (Set of Strings) -- Set of variables on which expression is represented, where each string is a random variable name
                             (if empty, then all variables in G)

       Returns: (String, Eqn)
       Factored Latex-format distribution equation string, corresponding Eqn object.
    """

    # Collate variable names
    V = G.nodes()

    # Find a topological ordering of the observed nodes
    sort_nodes = G.topsort()

    # Handle marginals
    if not Y:
        Y = V
    Y_mrg = V.difference(Y)
    fac_expr = expr.Expr(mrg=Y_mrg)

    # Factor distribution for each observed node in turn
    for node in V:

        # Set of all nodes preceding and including this one in topological order
        nodes_prec = set(sort_nodes[:sort_nodes.index(node) + 1])

        # Subgraph of all such ordered nodes
        G_prec = G.sub(nodes_prec)

        # District for node in G_prec
        D_node = G_prec.district(node)

        # Restrict ordered nodes to those within the district
        D_node_prec = nodes_prec.intersection(D_node)

        # Find (inclusive) parents of all such nodes
        D_parents = G.pa(D_node_prec).union(D_node_prec)

        # Remove node to find 'shielding' nodes
        shield = D_parents.difference({node})

        fac_expr.num = fac_expr.num + [shield.union({node})]
        fac_expr.den = fac_expr.den + [shield]

    # Final expression simplification
    fac_expr.simplify()

    # Construct Latex expression
    lhs = expr.Expr()
    lhs.addvars(num=Y)
    fac_eqn = eqn.Eqn(lhs, fac_expr)
    dist_str = fac_eqn.tocondstr()

    return dist_str, fac_eqn


def localmarkov(G):
    """Compute all local Markov independences for DAGs (ADMGs with no bidirects).

       Parameters:
       G (ADMG) -- DAG object representing the causal graph (must not have bidirects)

       Returns: (Set of Strings, Boolean)
       If G is a DAG, set of Unicode-format strings representing conditional independences,
       a CondIndSet object containing these, and True. Otherwise returns empty string,
       empty set and False.
    """

    # Check to make sure there are no latent variables
    if not G.isdag():
        return "", set(), False

    # Collate variable names
    nodes = set(G.vars)

    cis = condind.CondIndSet()
    for node in nodes:
        parents = G.pa({node})
        non_descend = G.nd({node})
        conds = non_descend.difference({node}).difference(parents)
        if conds:
            ci = condind.CondInd({node},conds,parents)
            cis.ciset.add(ci)

    return cis, cis.tostr(),  True


def dseparate(G,X,Y,Z={}):
    """For DAGs, test whether a set of source nodes are conditionally independent from a set of targets,
       conditioned on another set of nodes.

       Parameters:
       G (ADMG)            -- DAG object representing the causal graph (must not have bidirects)
       X (Set of Strings)  -- Source variables where each string is a random variable name
       Y (Set of Strings)  -- Target variables, each string is a random variable name
       Z (Set of Strings)  -- Conditioned variables, each string a random variable name

       Returns: (Boolean, String, Boolean)
       If G is a DAG, last parameter is True, otherwise False. If X is d-separated from Y given Z,
       first parameter is True, second parameter is a CondInd object representing the conditional
       independence, third parameter contains the corresponding Unicode-format string.
    """

    # Check to make sure there are no latent variables
    if not G.isdag():
        return None, None, "", False

    dsep = True
    for nodeX in X:
        for nodeY in Y:
            dsep = dsep and G.isdsep(nodeX,nodeY,Z)

    if dsep:
        conds = Y.difference(X)
        ci = condind.CondInd(X,conds,Z)
        return True, ci, ci.tostr(), True
    else:
        return False, None, "", True




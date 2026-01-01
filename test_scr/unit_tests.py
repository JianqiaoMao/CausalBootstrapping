#%%
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causalbootstrapping import workflows as wf
from causalbootstrapping import backend as be
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator
from grapl import dsl
from grapl import algorithms as algs
import pandas as pd
import numpy as np
from scipy.stats import norm
#%%
def run_all():
    
    test_weight_compute()
    test_bootstrappers()
    test_general_cb_analysis()
    test_general_causal_bootstrapping()
    test_backdoor_intv_clas()
    test_backdoor_cf_reg()
    test_frontdoor_intv()
    test_frontdoor_cf()
    
    print("All tests passed!")
    
def test_weight_compute():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"
    n_bins_yu = [0, 0]
    n_bins_u = [0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv").values
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv").values
    U_train = pd.read_csv(testdata_dir + "U_train.csv").values
    data = {'Y': Y_train,
            'U': U_train,
            'X': X_train}
    joint_yu_data = np.concatenate((Y_train, U_train), axis = 1)
   
    N = Y_train.shape[0]
    
    dist_estimator_yu = MultivarContiDistributionEstimator(data_fit=joint_yu_data)
    pdf_yu = dist_estimator_yu.fit_histogram(n_bins = n_bins_yu)
    dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_train)
    pdf_u = dist_estimator_u.fit_histogram(n_bins = n_bins_u)

    dist_map = {"intv_Y,U": lambda intv_Y, U: pdf_yu([intv_Y,U]),
                "U": lambda U: pdf_u(U)}

    causal_graph = '"back-door test example"; \
                 Y; X; U; \
                 Y -> X; \
                 U -> X; \
                 U -> Y;'

    grapl_obj = dsl.GraplDSL()
    G = grapl_obj.readgrapl(causal_graph)
    id_str, id_eqn, isident = algs.idfixing(G, {'Y'}, {'X'})
    
    w_func, _ = be.build_weight_function(intv_prob = id_eqn, 
                                         dist_map = dist_map, 
                                         N = N, 
                                         cause_intv_name_map = {"Y": "intv_Y"},
                                         kernel = lambda Y, intv_Y: np.equal(Y, intv_Y).astype(np.int8))
    Y_unique = np.unique(Y_train)
    weights = np.zeros((N, Y_unique.shape[0]))
    for i, y in enumerate(Y_unique):
        intv_dict = {"intv_Y": y}
        weight_i = be.weight_compute(w_func, data, intv_dict)
        weights[:, i] = weight_i.reshape(-1)
    
    assert callable(w_func)
    assert weights.shape == (N, Y_unique.shape[0])
    assert not (weights<0).any()
    assert not (weights==0).all()

def test_bootstrappers():
    N = 100
    simu_n_sample = 50
    
    data = {"Y": np.array([np.random.randint(0,2) for i in range(N)]).reshape(-1,1),
            "X": np.array([np.random.normal(0,1) for i in range(N)]).reshape(-1,1)}
    w_func = lambda X, Y, intv_Y: (X**2+0.5)+Y+intv_Y
    causal_weights = be.weight_compute(w_func = w_func,
                                       data = data,
                                       intv_dict = {"intv_Y": 1})
    btstrp_data_cf = be.cw_bootstrapper(data = data, 
                                          weights = causal_weights, 
                                          intv_dict = {"intv_Y": 1}, 
                                          n_sample = simu_n_sample, 
                                          sampling_mode = 'fast')
    
    assert btstrp_data_cf["intv_Y"].shape == (simu_n_sample, 1)
    
def test_backdoor_intv_clas():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"

    n_bins_yu = [0, 0]
    n_bins_u = [0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv").values
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv", dtype = int).values
    U_train = pd.read_csv(testdata_dir + "U_train.csv").values
    joint_yu_data = np.concatenate((Y_train, U_train), axis = 1)

    dist_estimator_yu = MultivarContiDistributionEstimator(data_fit=joint_yu_data)
    pdf_yu = dist_estimator_yu.fit_histogram(n_bins = n_bins_yu)
    dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_train)
    pdf_u = dist_estimator_u.fit_histogram(n_bins = n_bins_u)

    dist_map = {"intv_Y,U": lambda intv_Y, U: pdf_yu([intv_Y,U]),
                "U": lambda U: pdf_u(U)}
    
    cause_data = {"Y": Y_train}
    effect_data = {"X": X_train}
    confounder_data = {"U": U_train}

    cb_data = wf.backdoor_intv(cause_data = cause_data, 
                                 effect_data = effect_data, 
                                 confounder_data = confounder_data, 
                                 dist_map = dist_map,
                                 cause_intv_name_map = {"Y": "intv_Y"})

    assert cb_data["intv_Y"].shape == Y_train.shape

def test_backdoor_cf_reg():
    testdata_dir = "../test_data/backdoor_contY_contX_contU/"

    intv_intval_num = 100
    width = 1

    X_train = pd.read_csv(testdata_dir + "X_train.csv").values
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv").values
    U_train = pd.read_csv(testdata_dir + "U_train.csv").values
    joint_yu_data = np.concatenate((Y_train, U_train), axis = 1)

    Y_interv_values = np.linspace(np.min(Y_train), np.max(Y_train), intv_intval_num)
    N = Y_train.shape[0]
    

    dist_estimator_yu = MultivarContiDistributionEstimator(data_fit=joint_yu_data)
    pdf_yu = dist_estimator_yu.fit_multinorm()
    dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_train)
    pdf_u = dist_estimator_u.fit_multinorm()

    dist_map = {"intv_Y,U": lambda intv_Y, U: pdf_yu([intv_Y,U]),
                "U": lambda U: pdf_u(U)}
    
    cause_data = {"Y": Y_train}
    effect_data = {"X": X_train}
    confounder_data = {"U": U_train}
    
    cb_data_cf = {}
    for i, interv_value in enumerate(Y_interv_values):
        cb_data = wf.backdoor_cf(cause_data = cause_data, 
                                   effect_data = effect_data, 
                                   confounder_data = confounder_data, 
                                   dist_map = dist_map, 
                                   intv_dict = {"intv_Y": interv_value}, 
                                   n_sample = int(N/intv_intval_num), 
                                   kernel_intv = lambda Y, intv_Y: norm.pdf(Y-intv_Y, 0, width))
        
        for key in cb_data:
            if i == 0:
                cb_data_cf[key] = cb_data[key]
            else:
                cb_data_cf[key] = np.vstack((cb_data_cf[key], cb_data[key]))
                
    assert cb_data_cf["intv_Y"].shape[0] == int(N/intv_intval_num)*intv_intval_num
    
def test_frontdoor_intv():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"

    n_bins_yz = [0,20]
    n_bins_y = [0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv").values
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv").values
    Z_train = pd.read_csv(testdata_dir + "Z_train.csv").values
    joint_yz_data = np.concatenate((Y_train, Z_train), axis = 1)

    dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data)
    pdf_yz = dist_estimator_yz.fit_histogram(n_bins = n_bins_yz)
    dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y_train)
    pdf_y = dist_estimator_y.fit_histogram(n_bins = n_bins_y)
    
    cause_data = {"Y'": Y_train}
    mediator_data = {"Z": Z_train}
    effect_data = {"X": X_train}
    dist_map = {"intv_Y,Z": lambda intv_Y, Z: pdf_yz([intv_Y,Z]),
                "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
                "intv_Y": lambda intv_Y: pdf_y(intv_Y),
                "Y'": lambda Y_prime: pdf_y(Y_prime)}
    cb_data = wf.frontdoor_intv(cause_data, 
                                  mediator_data, 
                                  effect_data, 
                                  dist_map, 
                                  cause_intv_name_map = {"Y": "intv_Y"})
    
    assert cb_data["intv_Y"].shape == Y_train.shape

def test_frontdoor_cf():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"
    n_sample = 1000
    n_bins_yz = [0,20]
    n_bins_y = [0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv").values
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv").values
    Z_train = pd.read_csv(testdata_dir + "Z_train.csv").values
    joint_yz_data = np.concatenate((Y_train, Z_train), axis = 1)

    dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data)
    pdf_yz = dist_estimator_yz.fit_histogram(n_bins = n_bins_yz)
    dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y_train)
    pdf_y = dist_estimator_y.fit_histogram(n_bins = n_bins_y)
    
    cause_data = {"Y'": Y_train}
    mediator_data = {"Z": Z_train}
    effect_data = {"X": X_train}
    dist_map = {"intv_Y,Z": lambda intv_Y, Z: pdf_yz([intv_Y,Z]),
                "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
                "intv_Y": lambda intv_Y: pdf_y(intv_Y),
                "Y'": lambda Y_prime: pdf_y(Y_prime)}
    
    cb_data_cf_intv1 = wf.frontdoor_cf(cause_data = cause_data, 
                                           mediator_data = mediator_data, 
                                           effect_data = effect_data,
                                           dist_map = dist_map, 
                                           intv_dict = {"intv_Y": 1}, 
                                           n_sample = n_sample)
    cb_data_cf_intv2 = wf.frontdoor_cf(cause_data = cause_data, 
                                           mediator_data = mediator_data, 
                                           effect_data = effect_data,
                                           dist_map = dist_map, 
                                           intv_dict = {"intv_Y": 2}, 
                                           n_sample = n_sample)

    assert cb_data_cf_intv1["intv_Y"].shape[0] == n_sample
    assert cb_data_cf_intv2["intv_Y"].shape[0] == n_sample
    assert (cb_data_cf_intv1["intv_Y"]==1).all()
    assert (cb_data_cf_intv2["intv_Y"]==2).all()
    
def test_general_cb_analysis():
    causal_graph = '"Complex case"; \
                    Y; X; U; Z; \
                    U -> Y; \
                    Y -> Z; \
                    U -> Z; \
                    Z -> X; \
                    X <-> Y;'
    weight_func_lam, weight_func_expr = wf.general_cb_analysis(causal_graph = causal_graph, 
                                                               effect_var_name = 'X', 
                                                               cause_var_name = 'Y',
                                                               info_print = False)
    
    assert callable(weight_func_lam)
    assert weight_func_expr is not None
    
def test_general_causal_bootstrapping():
    testdata_dir = "../test_data/complex_scenario/"
    causal_graph = '"Complex case"; \
                    Y; X; U; Z; \
                    U -> Y; \
                    Y -> Z; \
                    U -> Z; \
                    Z -> X; \
                    X <-> Y;'
    n_bins_uyz = [0,0,0,0]
    n_bins_uy = [0,0]
    n_sample = 1000
                    
    X_train = pd.read_csv(testdata_dir + "X_train.csv").values
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv").values
    Z_train = pd.read_csv(testdata_dir + "Z_train.csv").values
    U_train = pd.read_csv(testdata_dir + "U_train.csv").values
    data_uyz = np.concatenate((U_train, Y_train, Z_train), axis = 1)
    data_uy = np.concatenate((U_train, Y_train), axis = 1)
    data = {"Y'": Y_train,
            "X": X_train,
            "Z": Z_train,
            "U": U_train}
    
    dist_estimator_uyz = MultivarContiDistributionEstimator(data_fit=data_uyz)
    pdf_uyz = dist_estimator_uyz.fit_histogram(n_bins = n_bins_uyz)
    dist_estimator_uy = MultivarContiDistributionEstimator(data_fit=data_uy)
    pdf_uy = dist_estimator_uy.fit_histogram(n_bins = n_bins_uy)
    dist_map = {"U,intv_Y,Z": lambda U, intv_Y, Z: pdf_uyz([U, intv_Y, Z]),
                "U,Y',Z": lambda U, Y_prime, Z: pdf_uyz([U, Y_prime, Z]),
                "U,Y'": lambda U, Y_prime: pdf_uy([U,Y_prime]),
                "U,intv_Y": lambda U, intv_Y: pdf_uy([U, intv_Y])}
    
    weight_func_lam, weight_func_expr = wf.general_cb_analysis(causal_graph = causal_graph, 
                                                               effect_var_name = 'X', 
                                                               cause_var_name = 'Y',
                                                               info_print = False)
    
    cb_data = wf.general_causal_bootstrapping_intv(weight_func_lam = weight_func_lam, 
                                                   dist_map = dist_map, 
                                                   data = data, 
                                                   intv_var_name_in_data = "Y'",
                                                   cause_intv_name_map = {"Y": "intv_Y"}, 
                                                   kernel = None)

    cb_data_intv1= wf.general_causal_bootstrapping_cf(weight_func_lam = weight_func_lam, 
                                                      dist_map = dist_map, 
                                                      data = data, 
                                                      cause_intv_name_map = {"Y": "intv_Y"},
                                                      intv_dict = {"intv_Y": 1}, 
                                                      n_sample = n_sample)
    
    assert cb_data["intv_Y"].shape == Y_train.shape
    assert np.sum(cb_data["intv_Y"] == 1) == np.sum(Y_train == 1)
    assert cb_data_intv1["intv_Y"].shape == (n_sample, 1)

run_all()  

# %%

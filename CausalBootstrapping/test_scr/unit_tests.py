import causalBootstrapping as cb
from distEst_lib import MultivarContiDistributionEstimator
from grapl import dsl
from grapl import algorithms as algs
import pandas as pd
import numpy as np
from scipy.stats import norm

def run_all():
    
    test_weight_compute()
    test_bootstrappers()
    test_general_cb_analysis()
    test_general_causal_bootstrapping()
    test_backdoor_simple_clas()
    test_backdoor_simu_reg()
    test_frontdoor_simple()
    test_frontdoor_simu()
    
    print("All tests passed!")
    
def test_weight_compute():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"
    n_bins_yu = [0, 0, 0]
    n_bins_u = [0, 0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv")
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv")
    U_train = pd.read_csv(testdata_dir + "U_train.csv")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    U_train = np.array(U_train)
    data = {'Y': Y_train,
            'U': U_train,
            'X': X_train}
    joint_yu_data = np.concatenate((Y_train.reshape(-1,1), U_train), axis = 1)
   
    N = Y_train.shape[0]
    
    dist_estimator_yu = MultivarContiDistributionEstimator(data_fit=joint_yu_data, n_bins = n_bins_yu)
    pdf_yu, pyu = dist_estimator_yu.fit_histogram()
    dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_train, n_bins = n_bins_u)
    pdf_u, pu = dist_estimator_u.fit_histogram()

    dist_map = {"Y,U": lambda Y, U: pdf_yu([Y,U]),
                "U": lambda U: pdf_u(U)}

    causal_graph = '"back-door test example"; \
                 Y; X; U; \
                 Y -> X; \
                 U -> X; \
                 U -> Y;'

    grapl_obj = dsl.GraplDSL()
    G = grapl_obj.readgrapl(causal_graph)
    id_str, id_eqn, isident = algs.idfixing(G, {'Y'}, {'X'})
    
    w_func = cb.weight_func(intv_prob = id_eqn, dist_map = dist_map, N = N, kernel = lambda intv_Y, Y: 1 if intv_Y == Y else 0)
    weights = []
    for y in set(Y_train.reshape(-1)):
        weights.append(cb.weight_compute(weight_func = w_func, data = data ,intv_var = {"intv_Y": [y for i in range(Y_train.shape[0])]}))
    weights = np.array(weights).T
    
    assert callable(w_func)
    assert weights.shape == (N, len(set(Y_train.reshape(-1))))
    assert not (weights<0).any()
    assert not (weights==0).all()

def test_bootstrappers():
    N = 100
    simu_n_sample = 50
    
    data = {"Y": np.array([np.random.randint(0,2) for i in range(N)]).reshape(-1,1),
            "X": np.array([np.random.normal(0,1) for i in range(N)]).reshape(-1,1)}
    weights = np.array([[np.random.uniform(0,1) for i in range(N)],
                        [np.random.uniform(0,1) for i in range(N)]]).T
    btstrp_data = cb.bootstrapper(data = data, weights = weights, 
                                  intv_var_name_in_data = ['Y'], mode = 'fast')
    
    w_func = lambda X, Y, intv_Y: (X**2)*Y*intv_Y
    btstrp_data_simu, weights_simu = cb.simu_bootstrapper(data = data, weight_func = w_func, 
                                                          intv_var_value = {"intv_Y": [1 for i in range(N)]}, 
                                                          n_sample = simu_n_sample, mode = 'fast')
    
    assert btstrp_data["intv_Y"].shape == (N, 1)
    assert btstrp_data_simu["intv_Y"].shape == (simu_n_sample, 1)
    
    
    
def test_backdoor_simple_clas():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"
    n_bins_yu = [0, 0, 0]
    n_bins_u = [0, 0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv")
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv")
    U_train = pd.read_csv(testdata_dir + "U_train.csv")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    U_train = np.array(U_train)
    joint_yu_data = np.concatenate((Y_train.reshape(-1,1), U_train), axis = 1)

    dist_estimator_yu = MultivarContiDistributionEstimator(data_fit=joint_yu_data, n_bins = n_bins_yu)
    pdf_yu, pyu = dist_estimator_yu.fit_histogram()
    dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_train, n_bins = n_bins_u)
    pdf_u, pu = dist_estimator_u.fit_histogram()

    dist_map = {"Y, U": lambda Y, U: pdf_yu([Y,U]),
                "U": lambda U: pdf_u(U)}
    
    cause_data = {"Y": Y_train}
    effect_data = {"X": X_train}
    confounder_data = {"U": U_train}
    
    cb_data = cb.backdoor_simple(cause_data, effect_data, confounder_data, dist_map)
    
    assert cb_data["intv_Y"].shape == Y_train.shape

def test_backdoor_simu_reg():
    testdata_dir = "../test_data/backdoor_contY_contX_contU/"
    dist_est_bin_num = 20
    intv_intval_num = 10
    width = 0.25
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv")
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv")
    U_train = pd.read_csv(testdata_dir + "U_train.csv")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    U_train = np.array(U_train)
    joint_yu_data = np.concatenate((Y_train.reshape(-1,1), U_train), axis = 1)

    Y_interv_values = np.linspace(np.min(Y_train), np.max(Y_train), intv_intval_num)
    N = Y_train.shape[0]
    
    n_bins_yu = [dist_est_bin_num,dist_est_bin_num]
    n_bins_u = [dist_est_bin_num]
    dist_estimator_yu = MultivarContiDistributionEstimator(data_fit=joint_yu_data, n_bins = n_bins_yu)
    pdf_yu, pyu = dist_estimator_yu.fit_histogram()
    dist_estimator_u = MultivarContiDistributionEstimator(data_fit=U_train, n_bins = n_bins_u)
    pdf_u, pu = dist_estimator_u.fit_histogram()
    
    dist_map = {"Y,U": lambda Y, U: pdf_yu([Y,U]),
                "U": lambda U: pdf_u(U)}
    
    cause_data = {"Y": Y_train}
    effect_data = {"X": X_train}
    confounder_data = {"U": U_train}
    
    cb_data_simu = {}
    for i, interv_value in enumerate(Y_interv_values):
        cb_data = cb.backdoor_simu(cause_data = cause_data, 
                                   effect_data = effect_data, 
                                   confounder_data = confounder_data, 
                                   dist_map = dist_map, 
                                   intv_value = [interv_value for i in range(N)], 
                                   n_sample = int(N/intv_intval_num), 
                                   kernel_intv = lambda Y, intv_Y: norm.pdf(Y-intv_Y, 0, width))
        for key in cb_data:
            if i == 0:
                cb_data_simu[key] = cb_data[key]
            else:
                cb_data_simu[key] = np.vstack((cb_data_simu[key], cb_data[key]))
                
    assert cb_data_simu["intv_Y"].shape[0] == int(N/intv_intval_num)*intv_intval_num

def test_frontdoor_simple():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"
    n_bins_yz = [0,20]
    n_bins_y = [0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv")
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv")
    Z_train = pd.read_csv(testdata_dir + "Z_train.csv")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Z_train = np.array(Z_train)
    joint_yz_data = np.concatenate((Y_train.reshape(-1,1), Z_train), axis = 1)

    dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data, n_bins = n_bins_yz)
    pdf_yz, pyz = dist_estimator_yz.fit_histogram()
    dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y_train, n_bins = n_bins_y)
    pdf_y, py = dist_estimator_y.fit_histogram()
    
    cause_data = {"Y": Y_train}
    mediator_data = {"Z": Z_train}
    effect_data = {"X": X_train}
    dist_map = {"Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
                "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
                "Y": lambda Y: pdf_y(Y),
                "Y'": lambda Y_prime: pdf_y(Y_prime)}
    cb_data = cb.frontdoor_simple(cause_data, mediator_data, effect_data, dist_map)
    
    assert cb_data["intv_Y"].shape == Y_train.shape

def test_frontdoor_simu():
    testdata_dir = "../test_data/frontdoor_discY_contZ_contX_discU/"
    n_sample = 1000
    n_bins_yz = [0,20]
    n_bins_y = [0]
    
    X_train = pd.read_csv(testdata_dir + "X_train.csv")
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv")
    Z_train = pd.read_csv(testdata_dir + "Z_train.csv")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Z_train = np.array(Z_train)
    joint_yz_data = np.concatenate((Y_train.reshape(-1,1), Z_train), axis = 1)
    
    dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data, n_bins = n_bins_yz)
    pdf_yz, pyz = dist_estimator_yz.fit_histogram()
    dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y_train, n_bins = n_bins_y)
    pdf_y, py = dist_estimator_y.fit_histogram()
    
    cause_data = {"Y": Y_train}
    mediator_data = {"Z": Z_train}
    effect_data = {"X": X_train}
    dist_map = {"Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
                "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
                "Y": lambda Y: pdf_y(Y),
                "Y'": lambda Y_prime: pdf_y(Y_prime)}
    
    
    cb_data_simu_intv1 = cb.frontdoor_simu(cause_data = cause_data, mediator_data = mediator_data, effect_data = effect_data,
                                           dist_map = dist_map, intv_value = [1 for i in range(Y_train.shape[0])], n_sample = n_sample)
    cb_data_simu_intv2 = cb.frontdoor_simu(cause_data = cause_data, mediator_data = mediator_data, effect_data = effect_data,
                                           dist_map = dist_map, intv_value = [2 for i in range(Y_train.shape[0])], n_sample = n_sample)

    assert cb_data_simu_intv1["intv_Y"].shape[0] == n_sample
    assert cb_data_simu_intv2["intv_Y"].shape[0] == n_sample
    assert (cb_data_simu_intv1["intv_Y"]==1).all()
    assert (cb_data_simu_intv2["intv_Y"]==2).all()
    
def test_general_cb_analysis():
    causal_graph = '"Complex case"; \
                    Y; X; U; Z; \
                    U -> Y; \
                    Y -> Z; \
                    U -> Z; \
                    Z -> X; \
                    X <-> Y;'
    weight_func_lam, weight_func_str = cb.general_cb_analysis(causal_graph = causal_graph, 
                                                              effect_var_name = 'X', 
                                                              cause_var_name = 'Y',
                                                              info_print = False)
    
    assert callable(weight_func_lam)
    assert weight_func_str is not None
    
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
                    
    X_train = pd.read_csv(testdata_dir + "X_train.csv")
    Y_train = pd.read_csv(testdata_dir + "Y_train.csv")
    Z_train = pd.read_csv(testdata_dir + "Z_train.csv")
    U_train = pd.read_csv(testdata_dir + "U_train.csv")
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    Z_train = np.array(Z_train)
    U_train = np.array(U_train)
    data_uyz = np.concatenate((U_train, Y_train, Z_train), axis = 1)
    data_uy = np.concatenate((U_train, Y_train), axis = 1)
    data = {"Y'": Y_train,
            "X": X_train,
            "Z": Z_train,
            "U": U_train}
    
    dist_estimator_uyz = MultivarContiDistributionEstimator(data_fit=data_uyz, n_bins = n_bins_uyz)
    pdf_uyz, puyz = dist_estimator_uyz.fit_histogram()
    dist_estimator_uy = MultivarContiDistributionEstimator(data_fit=data_uy, n_bins = n_bins_uy)
    pdf_uy, puy = dist_estimator_uy.fit_histogram()
    dist_map = {"U,Y,Z": lambda U, Y, Z: pdf_uyz([U, Y, Z]),
                "U,Y',Z": lambda U, Y_prime, Z: pdf_uyz([U, Y_prime, Z]),
                "U,Y'": lambda U, Y_prime: pdf_uy([U,Y_prime]),
                "U,Y": lambda U, Y: pdf_uy([U, Y])}
    
    weight_func_lam, weight_func_str = cb.general_cb_analysis(causal_graph = causal_graph, 
                                                              effect_var_name = 'X', 
                                                              cause_var_name = 'Y',
                                                              info_print = False)
    
    cb_data = cb.general_causal_bootstrapping_simple(weight_func_lam = weight_func_lam, 
                                                     dist_map = dist_map, data = data, 
                                                     intv_var_name = "Y", kernel = None)
    
    N = Y_train.shape[0]
    cb_data_intv1= cb.general_causal_bootstrapping_simu(weight_func_lam = weight_func_lam, 
                                                        dist_map = dist_map, data = data, 
                                                        intv_var_value = {"Y": [1 for i in range(N)]}, 
                                                        n_sample = n_sample, kernel = lambda Y_prime, Y: 1 if Y_prime == Y else 0)
    
    assert cb_data["intv_Y"].shape == Y_train.shape
    assert cb_data_intv1["intv_Y"].shape == (n_sample, 1)

run_all()  

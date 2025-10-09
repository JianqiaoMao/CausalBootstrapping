# CausalBootstrapping [![DOI](https://sandbox.zenodo.org/badge/715233323.svg)](https://sandbox.zenodo.org/doi/10.5072/zenodo.24049)

CausalBootstrapping is an easy-access implementation and extention of causal bootstrapping (CB) technique for causal analysis. With certain input of observational data, causal graph and variable distributions, CB resamples the data by adjusting the variable distributions which follow intended causal effects, so an appropriate and unbiased causal effects between the cause variable and effect variable can be captured.

### Confounding

<div style="text-align: center;">
  <img src="https://github.com/JianqiaoMao/CausalBootstrapping/blob/main/Images/backdoor_graph.png" width="300">
</div>

In a backdoor setting, an existing confounder may lead to so-called "selection bias". And thus a machine leanring model which is blind to the backend causal relationships between variables is exposed to risks of learning biased and unreliable associations between the predicting target and the features. A simple and intuitive example is as below:

<div style="text-align: center;">
  <img src="https://github.com/JianqiaoMao/CausalBootstrapping/blob/main/Images/demo_graph_backdoor.png" width="500">
</div>

In the figure, the model trained on confounded dataset (for example, the observational data collected from uncontrolled experiments) is biased due to the existence of the confounder. [Causal Bootstrapping](https://arxiv.org/abs/1910.09648) can aid this challenge by adjusting the observational data's distribution, and thus the model is supposed to learn from the data given the generative distribution of $P(X|do(Y))$ instead of $P(X|Y)$. That is, the model trained on de-confounded dataset by performing backdoor causal bootstrapping shows a proper behavior eliminating the influence imposed by the confounder $U$ as expected (the de-confounded decision boundary is closer to the true class boundary).

### Citing

Please use one of the following to cite the code of this repository.

```
@article{little2019causal,
  title={Causal bootstrapping},
  author={Little, Max A and Badawy, Reham},
  journal={arXiv preprint arXiv:1910.09648},
  year={2019}
}
```


## Installation and getting started

We currently offer seamless installation with `pip`. 

Simply:
```
pip install CausalBootstrapping
```

Alternatively, download the current distribution of the package, and run:
```
pip install .
```
in the root directory of the decompressed package.

To import the package:
```python
import causalBootstrapping as cb
```

## Example Demo.

Please refer to [Tutorials](https://github.com/JianqiaoMao/CausalBootstrapping/tree/main/CausalBootstrapping/Tutorials) for more instructions and examples.

1. Import causalBootstrapping lib and other libs for demo.
```python
from causalbootstrapping import workflows
from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
```

2. Define a causal graph
```python
causal_graph = '"General Causal Graph"; \
                Y; X; U; Z; \
                U -> Y; \
                Y -> Z; \
                U -> Z; \
                Z -> X; \
                X <-> Y;'
```
The above causal graph is equivalent to:

<img src="https://github.com/JianqiaoMao/CausalBootstrapping/blob/main/Images/complex_graph.png" width="300">

3. Analyse the causal graph and output the weights function expression and required distributions
```python
weight_func_lam, weight_func_str = workflows.general_cb_analysis(causal_graph = causal_graph, 
                                                                 effect_var_name = 'X', 
                                                                 cause_var_name = 'Y',
                                                                 info_print = True)
```

This code is expected to output as below:
```output
Interventional prob.:p_{Y}(X)=\sum_{U,Z,Y'}[p(X|U,Z,Y')p(Z|U,Y)p(U,Y')]
Causal bootstrapping weights function: [P(U,Y')P(U,Y,Z)]/N*[P(U,Y',Z)P(U,Y)]
Required distributions:
1: P(U,Y')
2: P(U,Y,Z)
3: P(U,Y',Z)
4: P(U,Y)
```

4. Read the demo. data for causal bootstrapping bootstraping
```python
# Read demo data
testdata_dir = "../test_data/complex_scenario/"
X_train = pd.read_csv(testdata_dir + "X_train.csv").values
Y_train = pd.read_csv(testdata_dir + "Y_train.csv").values
Z_train = pd.read_csv(testdata_dir + "Z_train.csv").values
U_train = pd.read_csv(testdata_dir + "U_train.csv").values
data = {"Y'": Y_train,
        "X": X_train,
        "Z": Z_train,
        "U": U_train}
```

5. Estimate the desired distributions (as shown in previous output of `general_cb_analysis()`). User is also encourged to define the distribution functions if certain domain knowledge has been obtained.
```python
#Set number of the bins for histogram becasue all variables follow discrete distributions.
n_bins_uyz = [0,0,0,0]
n_bins_uy = [0,0]
data_uyz = np.concatenate((U_train, Y_train, Z_train), axis = 1)
data_uy = np.concatenate((U_train, Y_train), axis = 1)

dist_estimator_uyz = MultivarContiDistributionEstimator(data_fit=data_uyz)
pdf_uyz = dist_estimator_uyz.fit_histogram(n_bins = n_bins_uyz)
dist_estimator_uy = MultivarContiDistributionEstimator(data_fit=data_uy)
pdf_uy = dist_estimator_uy.fit_histogram(n_bins = n_bins_uy)
```

6. Construct the distribution mapping dict
```python
dist_map = {"U,intv_Y,Z": lambda U, intv_Y, Z: pdf_uyz([U, intv_Y, Z]),
            "U,Y',Z": lambda U, Y_prime, Z: pdf_uyz([U, Y_prime, Z]),
            "U,Y'": lambda U, Y_prime: pdf_uy([U,Y_prime]),
            "U,intv_Y": lambda U, intv_Y: pdf_uy([U, intv_Y])}
```

7. bootstrap the dataset given the weight function expression
```python
cb_data = workflows.general_causal_bootstrapping_simple(weight_func_lam = weight_func_lam, 
                                                        dist_map = dist_map, 
                                                        data = data, 
                                                        intv_var_name_in_data = "Y'",
                                                        cause_intv_name_map = {"Y": "intv_Y"}, 
                                                        kernel = None)
```

8. Train two linear support vector machines using confounded and de-confounded datasets
```python
clf_conf = svm.SVC(kernel = 'linear', C=2)
clf_conf.fit(X_train, Y_train.reshape(-1))

clf_cb = svm.SVC(kernel = 'linear', C=2)
clf_cb.fit(cb_data['X'], cb_data["intv_Y"].reshape(-1))
```

9. Compare their performance on an un-confounded test set
```python
X_test = pd.read_csv(testdata_dir +  "X_test.csv")
Y_test = pd.read_csv(testdata_dir +  "Y_test.csv")
X_test = np.array(X_test)
Y_test = np.array(Y_test)

y_pred_conf = clf_conf.predict(X_test)
print("Report of confonded model:")
print(classification_report(Y_test, y_pred_conf))

y_pred_deconf = clf_cb.predict(X_test)
print("Report of de-confonded model:")
print(classification_report(Y_test, y_pred_deconf))
```
The expected output should be similar to:
```output
Report of confonded model:
              precision    recall  f1-score   support

           1       0.56      0.88      0.68       865
           2       0.84      0.46      0.60      1135

    accuracy                           0.65      2000
   macro avg       0.70      0.67      0.64      2000
weighted avg       0.72      0.65      0.63      2000

Report of de-confonded model:
              precision    recall  f1-score   support

           1       0.63      0.84      0.72       865
           2       0.84      0.63      0.72      1135

    accuracy                           0.72      2000
   macro avg       0.73      0.73      0.72      2000
weighted avg       0.75      0.72      0.72      2000
```

10. Compare models' decision boundaries
```python
#confounding boundary
conf_x2, conf_x3 = np.meshgrid(np.linspace(-6, 6, 20), np.linspace(-6, 6, 20))
conf_x1 = np.zeros((20,20))
# real boundary
real_x1, real_x2 = np.meshgrid(np.linspace(-6, 6, 20), np.linspace(-6, 6, 20))
real_x3 = np.full_like(real_x1, 0)

# confounded svm boundary
xx1, xx2= np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
xx_conf = (-clf_conf.intercept_[0] - clf_conf.coef_[0][0] * xx1 - clf_conf.coef_[0][1] * xx2) / clf_conf.coef_[0][2]

# deconfounded svm boundary
xx1, xx2= np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
xx_cb = (-clf_cb.intercept_[0] - clf_cb.coef_[0][0] * xx1 - clf_cb.coef_[0][1] * xx2) / clf_cb.coef_[0][2]

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test[:,0],X_test[:,1],X_test[:,2],c=Y_test, s = 5, alpha = 0.5)
surf1 = ax.plot_surface(conf_x1, conf_x2, conf_x3, alpha=0.5, rstride=100, cstride=100, color = "yellow", label = "confounding boundary")
surf2 = ax.plot_surface(real_x1, real_x2, real_x3, alpha=0.5, rstride=100, cstride=100, color = "green", label = "real boundary")
surf3 = ax.plot_surface(xx1, xx2, xx_conf, color='red', alpha=0.5, rstride=100, cstride=100, label = "confounded decision boundary")
surf4 = ax.plot_surface(xx1, xx2, xx_cb, color='blue', alpha=0.5, rstride=100, cstride=100, label = "confounded decision boundary")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
surf1._facecolors2d=surf1._facecolors
surf1._edgecolors2d=surf1._edgecolors
surf2._facecolors2d=surf2._facecolors
surf2._edgecolors2d=surf2._edgecolors
surf3._facecolors2d=surf3._facecolors
surf3._edgecolors2d=surf3._edgecolors
surf4._facecolors2d=surf4._facecolors
surf4._edgecolors2d=surf4._edgecolors
ax.legend(["Unconfounded test data", "confounding boundary", "real boundary", "confounded decision boundary", "deconfounded decision boundary"])
plt.title('Decision boundary comparison')
plt.tight_layout()
plt.show()
```
The expected output of the image should be similar to:

<img src="https://github.com/JianqiaoMao/CausalBootstrapping/blob/main/Images/demo_graph_general_case.png" width="600">





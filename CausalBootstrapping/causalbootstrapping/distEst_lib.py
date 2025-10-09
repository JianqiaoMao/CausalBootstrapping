# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:50:12 2023

@author: NickMao
"""
import numpy as np
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.stats import rv_discrete
from scipy.cluster.vq import kmeans2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import sympy as sy
from scipy import integrate
import warnings

class MultivarContiDistributionEstimator:
    def __init__(self, data_fit):
        self.data_fit = np.asarray(data_fit, dtype=float)
        if self.data_fit.ndim != 2:
            raise ValueError(f"Expected 2D array for data_fit in shape (N,d), got {self.data_fit.ndim}D array.")
        self.d = self.data_fit.shape[1]
        self.pdf = None
        
    def _ensure_2d_points(self, X):
        if isinstance(X, (list, tuple)):
            X = np.column_stack([np.asarray(col, dtype=float) for col in X])
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                # d==1 时你也可以改成 X.reshape(-1, 1)
                X = X.reshape(1, -1)
        if X.shape[1] != self.d:
            raise ValueError(f"X.shape[1] ({X.shape[1]}) != d ({self.d})")
        return X
    
    def fit_multinorm(self):
        mu = np.mean(self.data_fit, axis=0)
        cov = np.cov(self.data_fit, rowvar=False, ddof=1)
        cov = cov + 1e-9 * np.eye(self.d)
        mvn = multivariate_normal(mean=mu, cov=cov, allow_singular=False)

        def pdf_eval(X):
            X = self._ensure_2d_points(X)
            return mvn.pdf(X)
        
        self.pdf = pdf_eval
        
        return pdf_eval
    
    def fit_kmeans(self, k):
        centroids, labels = kmeans2(self.data_fit, k, minit='points')
        comps = []
        weights = []

        for i in range(k):
            cluster = self.data_fit[labels == i]
            n_i = cluster.shape[0]
            if n_i == 0:
                continue 
            mu = cluster.mean(axis=0)
            if n_i < 2:
                cov = 1e-6 * np.eye(self.d)
            else:
                cov = np.cov(cluster, rowvar=False, ddof=1)
                cov = cov + 1e-9 * np.eye(self.d)
            comps.append(multivariate_normal(mean=mu, cov=cov, allow_singular=False))
            weights.append(n_i)

        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum() if weights.size else np.array([1.0])

        def pdf_eval(X):
            X = self._ensure_2d_points(X)
            if not comps:
                return np.zeros(X.shape[0], dtype=float)
            prob = np.zeros(X.shape[0], dtype=float)
            for w, comp in zip(weights, comps):
                prob += w * comp.pdf(X)
            return prob

        self.pdf = pdf_eval
        
        return pdf_eval
    
    def fit_gmm(self, n_components):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=1e-6)
        gmm.fit(self.data_fit)

        def pdf_eval(X):
            X = self._ensure_2d_points(X)
            prob = np.exp(gmm.score_samples(X))
            return prob

        self.pdf = pdf_eval

        return pdf_eval
    
    def fit_kde(self, bandwidth = None):
        kde = gaussian_kde(self.data_fit.T, bw_method=bandwidth)
        
        def pdf_eval(X):
            X = self._ensure_2d_points(X)
            prob = kde(X.T)
            return prob

        self.pdf = pdf_eval
        
        return pdf_eval

    def fit_histogram(self, n_bins, zero_outside = True):
        """
        Fit a histogram to the data.

        Parameters:
            n_bins (int or list of int): The number of bins for each dimension.
            zero_outside (bool): Whether to set the PDF values to 0 for points outside the histogram range.

        Returns:
            function: A PDF evaluation function for the fitted histogram.
        """
        if isinstance(n_bins, int):
            n_bins = [n_bins] * self.d
        else:
            n_bins = list(n_bins)
            if len(n_bins) != self.d:
                raise ValueError("Length of n_bins must match number of dimensions (d).")
        n_bins = [len(set(self.data_fit[:,i])) if n_bins[i]==0 else n_bins[i] for i in range(self.d)]
        
        hist, edges = np.histogramdd(sample=self.data_fit, bins=n_bins, density=False)
        est_pmf = hist / np.sum(hist)
        e0 = np.array([e[0]   for e in edges], dtype=float)
        e1 = np.array([e[-1]  for e in edges], dtype=float)

        def pdf_eval(X):
            X = self._ensure_2d_points(X)
            N = X.shape[0]

            inds = np.empty((N, self.d), dtype=int)
            if zero_outside:
                outside = ((X < e0) | (X > e1)).any(axis=1)

            for j, e in enumerate(edges):
                idx = np.searchsorted(e, X[:, j], side='right') - 1
                idx = np.clip(idx, 0, len(e) - 2)
                inds[:, j] = idx

            prob = est_pmf[tuple(inds.T)].astype(float, copy=False)
            if zero_outside:
                if np.any(outside):
                    warnings.warn(f"{int(outside.sum())} points are outside the histogram range and their PDF values are set to 0.")
                    prob[outside] = 0.0
            return prob
        
        self.pdf = pdf_eval

        return pdf_eval


    def evaluate_density(self, n_bins, lower_bound = None, upper_bound = None):
        if self.pdf is None or not callable(self.pdf):
            raise RuntimeError(
                "No density function set. Call a fit_* method and assign it to self.pdf, "
                "e.g., `self.pdf = self.fit_kde(...)` before evaluate_density()."
            )        
        if isinstance(n_bins, int):
            n_bins = [n_bins] * self.d
        else:
            n_bins = list(n_bins)
            if len(n_bins) != self.d:
                raise ValueError("Length of n_bins must match number of dimensions (d).")
        n_bins = [len(set(self.data_fit[:,i])) if n_bins[i]==0 else n_bins[i] for i in range(self.d)]
        
        if lower_bound is None:
            lower_bound = np.min(self.data_fit, axis=0)
        if upper_bound is None:
            upper_bound = np.max(self.data_fit, axis=0)

        axes = [np.linspace(float(lower_bound[i]), float(upper_bound[i]), n_bins[i]) for i in range(self.d)]
        grid= np.meshgrid(*axes, indexing='ij')
        self.stacked_grid = np.stack(grid, axis=-1)
        flat_grid = self.stacked_grid.reshape(-1, self.d)
        pdf_eval = self.pdf(flat_grid).reshape(-1)
        pdf_eval_grid = pdf_eval.reshape(*n_bins)
        
        return pdf_eval_grid

    def plot(self, n_bins, title=None):
        
        if self.pdf is None:
            raise RuntimeError(
                "No density function set. Call a fit_* method and assign it to self.pdf, "
                "e.g., `self.pdf = self.fit_kde(...)` before evaluate_density()."
            )        
        if self.d != 2:
            raise ValueError("plot() method is only valid for a 2D distribution")
        prob_grid = self.evaluate_density(n_bins)
        plt.figure()
        plt.contourf(self.stacked_grid[:,:,0], self.stacked_grid[:,:,1], prob_grid)
        plt.xlabel("x1")
        plt.ylabel("x2")
        clb = plt.colorbar()
        clb.ax.set_title('Prob. Density')
        plt.title(title)  
        plt.show()

class MultivarDiscDistributionEstimator:
    def __init__(self, data_fit, data_est):
        self.data_fit = data_fit
    
    def fit_multinomial(self):
        n = len(self.data_fit)
        k = self.data_fit.shape[1]
        counts = np.sum(self.data_fit, axis=0)
        p = counts / n
        return rv_discrete(name='multinomial', values=(np.arange(k + 1),), args=(n, p))
    
    def fit_categorical(self):
        k = self.data_fit.shape[1]
        p = np.mean(self.data_fit, axis=0)
        return rv_discrete(name='categorical', values=(np.arange(k),), args=(p,))

class user_defined_func_obj:
    
    def __init__(self, func, data = None, ranges = 'auto'):
        self.func = func
        self.data = data
        if ranges == 'auto':
            if data is None:
                raise ValueError("data must be specified when using 'auto' range mode.")
            self.ranges = [(data[:,i].min(), data[:,i].max()) for i in range(data.shape[1])]
        if ranges == 'specified':
            if data is not None:
                warnings.warn("Integral's lower and upper limit does not fit the input data and will be requried if used.")
            self.ranges = None
        
    def integral_over_specified_var(self, int_var: dict, holding_var: dict):
        """
        Parameters
        ----------
        holding_var : dict
            Holding variables in the form {str 'name': float 'value'}
        int_var : dict
            Integral variables in the form if:
                i)  in the 'auto' mode: {str 'name': int 'column index corresponding to the fitted data'};
                ii) in the 'specified' mode: {str 'name': tuple '(lower limit, upper limit)'}

        Returns
        -------
        result: float
            Integral of user-defined function over specified integral and holding variables

        """
        int_var_name = list(int_var.keys())
        
        if self.ranges is None:
            ranges_list = [int_var[var] for var in int_var_name]
        else:
            ranges_list = []
            for i in range(len(int_var_name)):
                ranges_list.append(self.ranges[int_var[int_var_name[i]]])
        
        def integrand(*args):
            variables = {}
            variables.update(holding_var_val)
            variables.update(dict(zip(int_var_name, args)))
            return self.func(**variables)
        
        if isinstance(list(holding_var.values())[0], list) or isinstance(list(holding_var.values())[0], np.ndarray):
            results = []
            for vals in zip(*list(holding_var.values())):
                holding_var_val = dict(zip(holding_var.keys(), vals))
                result, _ = integrate.nquad(integrand, ranges_list)
                results.append(result)
            return results
        else:
            holding_var_val = holding_var
            result, _ = integrate.nquad(integrand, ranges_list)
            return result







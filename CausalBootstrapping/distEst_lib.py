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


def ndgrid_gen(data, n_bins, grid_type='grid'):
    min_dims = np.min(data, axis=0)
    max_dims = np.max(data, axis=0)
    ranges = max_dims - min_dims
    bin_sizes = ranges / n_bins

    if grid_type == 'flatten':
        pos = None
        sub_args = [slice(min_dims[d], max_dims[d], bin_sizes[d]) for d in range(data.shape[1])]
        pos = np.mgrid[tuple(sub_args)].reshape(data.shape[1], -1)
        return pos
    elif grid_type == 'grid':
        args = [np.linspace(min_dims[d], max_dims[d], n_bins[d]) for d in range(data.shape[1])]
        grid = np.stack(np.meshgrid(*args, indexing='ij'), axis=data.shape[1])
        return grid
    else:
        raise ValueError("Invalid value for 'grid_type', should be 'flatten' or 'grid'.")

class MultivarContiDistributionEstimator:
    def __init__(self, data_fit, n_bins, data_est = None):
        self.data_fit = data_fit
        self.n_bins = [len(set(data_fit[:,i])) if n_bins[i]==0 else n_bins[i] for i in range(data_fit.shape[1])]
        self.data_est = data_fit if data_est is None else data_est
        self.est_dist = None
        self.est_pdf = None
    
    def fit_multinorm(self, data_est = None):
        # data_unmean = self.data_fit - np.mean(self.data_fit, axis = 0)
        # cov_ = np.dot(data_unmean.T, data_unmean)/(self.data_fit.shape[0]-1)
        est_dist = multivariate_normal(mean=np.mean(self.data_fit , axis=0), cov=np.cov(self.data_fit.T))
        self.pos = ndgrid_gen(self.data_est, self.n_bins)
        est_pdf = est_dist.pdf(self.pos)
        # est_pdf = est_pdf/np.sum(est_pdf)
        
        self.est_dist = est_dist
        self.est_pdf = est_pdf
        
        return self.est_dist, self.est_pdf
    
    def fit_kmeans(self, k, data_est = None):
        centroids, labels = kmeans2(self.data_fit, k)
        means = []
        covs = []
        for i in range(k):
            cluster_data = self.data_fit[labels==i]
            means.append(np.mean(cluster_data, axis=0))
            covs.append(np.cov(cluster_data.T))
        est_dist = [multivariate_normal(mean=mean, cov=cov) for mean, cov in zip(means, covs)]
        self.pos = ndgrid_gen(self.data_est, self.n_bins)
        est_pdf = sum([est_dist[i].pdf(self.pos) for i in range(k)])
        est_pdf = est_pdf/np.sum(est_pdf)
        
        self.est_dist = est_dist
        self.est_pdf = est_pdf
        
        return self.est_dist, self.est_pdf
    
    def fit_gmm(self, n_components, data_est = None):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(self.data_fit)
        est_dist = [multivariate_normal(mean=mean, cov=cov) for mean, cov in zip(gmm.means_, gmm.covariances_)]
        self.pos = ndgrid_gen(self.data_est, self.n_bins)
        est_pdf = sum([est_dist[i].pdf(self.pos) for i in range(n_components)])
        
        self.est_dist = est_dist
        self.est_pdf = est_pdf
        
        return self.est_dist, self.est_pdf
    
    def fit_kde(self, bandwidth = None, data_est = None):
        self.est_dist = gaussian_kde(self.data_fit.T, bw_method=bandwidth)
        pos = ndgrid_gen(self.data_est, self.n_bins, 'flatten')
        est_pdf = self.est_dist(pos).reshape(self.n_bins)
        self.est_pdf = est_pdf/np.sum(est_pdf)
        self.pos = ndgrid_gen(self.data_est, self.n_bins)
        
        return self.est_dist, self.est_pdf
    
    def fit_histogram(self, data_est = None):
        hist, edges = np.histogramdd(sample = self.data_fit, bins=self.n_bins, density=False)
        self.est_pdf = hist/np.sum(hist)
        self.pos = ndgrid_gen(self.data_est, self.n_bins)
        
        def est_dist(x):
            if len(x) > 1:
                try:
                    x = np.concatenate(x)
                except:
                    x = x.reshape(-1)
            else:
                x = np.array(x).reshape(-1,1)
            indices = np.array([np.digitize(x[i], edges[i][:-1]) for i in range(len(edges))]) - 1
            prob= self.est_pdf[tuple(indices.T)]
            return prob
        # self.est_dist = rv_histogram(hist_tuple, density=True)
        
        return est_dist, self.est_pdf
    
    def plot(self, title=None):
        
        if self.est_dist is None:
            raise RuntimeError("The distribution is not fitted.")
        if self.data_fit.shape[1] != 2:
            raise ValueError("plot() method is only valid for a 2D distribution")
        plt.figure()
        plt.contourf(self.pos[:,:,0], self.pos[:,:,1], self.est_pdf)
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







from scipy.stats import rv_continuous
from scipy.special import  gammainc, gammaincinv, gamma, gammaln
import numpy as np

class pandel_gen(rv_continuous):
    """Pandel distribution
    
    Distribution Parameters:
    -----------
    
    t : variable of distribution (time)
    d : distance parameter (length)
    
    Instantiation parameters (optional):
    ------------------------------------

    tau : tau parameter (time)
    lambda_s : scattering length (length)
    lambda_a : absorption length (length)
    v : velocity (length/time)
    
    """
    
    def __init__(self, tau=557, lambda_s=33.3, lambda_a=98, v=0.3/1.3, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.lambda_s = lambda_s
        self.lambda_a = lambda_a
        self.v = v
        self.lam = self.v/self.lambda_a
        self.beta = 1/self.tau
        self.lb = self.lam + self.beta
    
    def _pdf(self, t, d):
        alpha = d/self.lambda_s
        x = t*self.lb
        return self.lb/gamma(alpha) * x ** (alpha-1) * np.exp(-x)
    
    def _logpdf(self, t, d):
        alpha = d/self.lambda_s
        x = t*self.lb
        return np.log(self.lb) - gammaln(alpha) + (alpha-1)*np.log(x) - x
    
    def _cdf(self, t, d):
        alpha = d/self.lambda_s
        return gammainc(alpha, t*self.lb)
    
    def _ppf(self, q, d):
        alpha = d/self.lambda_s
        return gammaincinv(alpha, q)/self.lb

    
pandel = pandel_gen(a=0.0, name='pandel')
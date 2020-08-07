import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import numba


# speed of light
c = 0.3 # m / ns

# Helper functions

# path length from src_x, src_b to sensor at position x
# (r_src - r_sensor)^2
def r2(x, x_src, b_src):
    # include finite size of sensor to avoid divide by 0 when scanning (x, b)
    return (x_src-x)**2 + b_src**2 + 0.05**2

# expected light arrival time at position x given a source at t_src, (x_src, b_src)
def arrival_time(x, t_src, x_src, b_src, v=c):
    return t_src + np.sqrt(r2(x, x_src, b_src))/c

# Lambda_d (expected charge) for detector at position x given source at (x_src, b_src) with "energy" N_src
def lambda_d(x, x_src, b_src, N_src):
    return N_src/r2(x, x_src, b_src)


class toy_experiment():

    def __init__(self, detector_xs = np.linspace(-5, 5, 11), t_std = 1):
        self.detector_xs = detector_xs
        self.t_std = t_std
        # time PDF factory
        self.time_dist = lambda t: stats.norm(loc=t, scale=self.t_std)

    # PDFs for event generation and likelihood calculation
    # numba decorators will be used for functions that are destined to be used in llh evaluations

    # returns function p_d(t) for detector at position x given source at t_src, (x_src, b_src)
    def get_p_d(self, x, t_src, x_src, b_src):
        return self.time_dist(arrival_time(x, t_src, x_src, b_src))

    def get_p_dists(self, t_src, x_src, b_src):
        return [self.get_p_d(x, t_src, x_src, b_src) for x in self.detector_xs]

    def get_lambda_ds(self, x_src, b_src, N_src):
        return lambda_d(self.detector_xs, x_src, b_src, N_src) 
            

    # define separate functions for DOM formulation hit term, DOM charge term, 
    # total charge formulation hit term, and total charge term
    # all functions will be log(L(event | x, b, t, N))

    # manual logpdf of the time distribution to avoid repeated construction of scipy.stats objects
    def log_p_d_t(self, det_x, t, hypo_x, hypo_b, hypo_t):
        # drop terms that are constant wrt the hypothesis
        t_exp = arrival_time(det_x, hypo_t, hypo_x, hypo_b)
        # t_std could in principle vary with the hypothesis, even though it doesn't in this example
        return -(t-t_exp)**2/(2*self.t_std**2) + np.log(self.t_std)

    # dom formulation
    def dom_hit_term(self, hit_times, hit_time_inds, x, b, t):
        llh = 0
        for hit_time, sensor_ind in zip(hit_times, hit_time_inds):
            llh += self.log_p_d_t(self.detector_xs[sensor_ind], hit_time, x, b, t)
        return llh

    def dom_charge_term(self, Ns, x, b, N_src):
        lambda_ds = lambda_d(self.detector_xs, x, b, N_src)
        return (Ns*np.log(lambda_ds) - lambda_ds).sum() 

    # total charge formulation
    def total_charge_hit_term(self, hit_times, hit_time_inds, x, b, t, N_src):
        llh = 0
        lambda_ds = lambda_d(self.detector_xs, x, b, N_src)
        p_sensor = lambda_ds/lambda_ds.sum()
        for hit_time, sensor_ind in zip(hit_times, hit_time_inds):
            llh += self.log_p_d_t(self.detector_xs[sensor_ind], hit_time, x, b, t) + np.log(p_sensor[sensor_ind])
        return llh

    def total_charge_term(self, Ns, x, b, N_src):
        lambda_tot = lambda_d(self.detector_xs, x, b, N_src).sum()
        N_tot = np.sum(Ns)
        return N_tot*np.log(lambda_tot) - lambda_tot


    def generate_event(self, x_src, t_src=0, N_src=10, b=1):
        '''
        generates one event
        
        Parameters:
        
        x_src : float
            Source position
        t_src : float
            Source time
        N_src : int
            Amount of photons sent out
        b : float
            perpendicaulr distance off of sensor line
            
        Returns:
        
        Ns : list
            observed number of photons per detector
        Ns_sensor_idx : list
            according index of sensor
        ts : list
            observed photon times
        ts_sensor_idx : list
            according index of sensor
        '''
        Ns = []
        Ns_sensor_idx = []
        ts = []
        ts_sensor_idx = []
        lambda_ds = lambda_d(self.detector_xs, x_src, b, N_src)
        for i, x in enumerate(self.detector_xs):
            N_exp = lambda_ds[i]
            N_obs = stats.poisson(mu=N_exp).rvs()
            Ns.append(N_obs)
            Ns_sensor_idx.append(i)
            if N_obs > 0:
                pulse_times = self.get_p_d(x, t_src, x_src, b).rvs(size=N_obs)
                ts.extend(pulse_times)
                ts_sensor_idx.extend([i]*N_obs)
        return Ns, Ns_sensor_idx, ts, ts_sensor_idx


    def generate_events(self, N):
        pass

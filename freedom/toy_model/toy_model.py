import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#import numba


# speed of light
c = 0.3 # m / ns

# Helper functions

def r2(x, x_src, b_src):
    '''
    path length from src_x, src_b to sensor at position x
    (r_src - r_sensor)^2
    includeis finite size of sensor to avoid divide by 0 when scanning (x, b)
    '''
    return (x_src-x)**2 + b_src**2 + 0.05**2

def arrival_time(x, t_src, x_src, b_src, v=c):
    '''
    expected light arrival time at position x given a source at t_src, (x_src, b_src)
    '''
    return t_src + np.sqrt(r2(x, x_src, b_src))/c

def lambda_d(x, x_src, b_src, N_src):
    '''
    Lambda_d (expected charge) for detector at position x given source at (x_src, b_src) with "energy" N_src
    '''
    return N_src/r2(x, x_src, b_src)


class toy_experiment():

    def __init__(self, detector_xs = np.linspace(-5, 5, 11), t_std = 1):
        self.detector_xs = detector_xs
        self.t_std = t_std
        # time PDF factory
        self.time_dist = lambda t: stats.norm(loc=t, scale=self.t_std)

    # PDFs for event generation and likelihood calculation

    def get_p_d(self, x, t_src, x_src, b_src):
        '''
        returns function p_d(t) for detector at position x given source at t_src, (x_src, b_src)
        '''
        return self.time_dist(arrival_time(x, t_src, x_src, b_src))

    def get_p_dists(self, t_src, x_src, b_src):
        '''returns funstions p_d(t) for each detector position'''
        return [self.get_p_d(x, t_src, x_src, b_src) for x in self.detector_xs]

    def get_lambda_ds(self, x_src, b_src, N_src):
        ''' returns expected charge at each detector'''
        return lambda_d(self.detector_xs, x_src, b_src, N_src) 
            

    # define separate functions for DOM formulation hit term, DOM charge term, 
    # total charge formulation hit term, and total charge term
    # all functions will be log(L(event | x, b, t, N))

    def log_p_d_t(self, det_x, t, hypo_x, hypo_b, hypo_t):
        '''
        manual logpdf of the time distribution to avoid repeated construction of scipy.stats objects
        drops terms that are constant wrt the hypothesis

        t_std could in principle vary with the hypothesis, even though it doesn't in this example
        '''
        t_exp = arrival_time(det_x, hypo_t, hypo_x, hypo_b)
        return -(t-t_exp)**2/(2*self.t_std**2) + np.log(self.t_std)

    # dom formulation
    def dom_hit_term(self, hit_times, x, b, t):
        '''hit llh for each / sensor'''
        return np.sum(self.log_p_d_t(hit_times[:,1], hit_times[:,0], x, b, t))

    def dom_charge_term(self, Ns, x, b, N_src):
        '''charge llh for each sensor'''
        lambda_ds = lambda_d(Ns[:,1], x, b, N_src)
        return np.sum(Ns[:,0]*np.log(lambda_ds) - lambda_ds)

    # total charge formulation
    def total_charge_hit_term(self, hit_times, x, b, t, N_src):
        '''hit llh for each / sensor reweighted by charge factor'''
        lambda_ds = lambda_d(self.detector_xs, x, b, N_src)
        lambda_hits = lambda_ds[hit_times[:,2].astype(np.int32)]
        p_sensor = lambda_hits/np.sum(lambda_ds)
        return np.sum(self.log_p_d_t(hit_times[:,1], hit_times[:,0], x, b, t) + np.log(p_sensor))

    def total_charge_term(self, Ns, x, b, N_src):
        '''total charge llh'''
        lambda_tot = np.sum(lambda_d(Ns[:,1], x, b, N_src))
        N_tot = np.sum(Ns[:,0])
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
        
        Ns : arrayy
            observed number of photons per detector, x-position of detectors, indices of detectors
        ts : list
            observed photon times, x-position of detectors, indices of detectors
        '''
        Ns = []
        Ns_sensor_idx = []
        Ns_sensor_x = []
        ts = []
        ts_sensor_idx = []
        ts_sensor_x = []
        lambda_ds = lambda_d(self.detector_xs, x_src, b, N_src)
        for i, x in enumerate(self.detector_xs):
            N_exp = lambda_ds[i]
            N_obs = stats.poisson(mu=N_exp).rvs()
            Ns.append(N_obs)
            Ns_sensor_x.append(x)
            Ns_sensor_idx.append(i)
            if N_obs > 0:
                pulse_times = self.get_p_d(x, t_src, x_src, b).rvs(size=N_obs)
                ts.extend(pulse_times)
                ts_sensor_idx.extend([i]*N_obs)
                ts_sensor_x.extend([x]*N_obs)
        return np.array([Ns, Ns_sensor_x, Ns_sensor_idx]).T, np.array([ts, ts_sensor_x, ts_sensor_idx]).T


    def generate_events(self, N_events, xlims=(-5, 5), blims=(-2,2), N_lims=(1,20)):
        '''
        sample source parameters from uniform distribution of x, b, and N
        and generate events using those.

        N_events : int
            number of desired events

        *_lims : tuple
            lower and upper bount of the uniform to sample from

        Returns:

        events : list of generated events
        truth : true parameters

        '''

        # truth array x, b, N

        x = np.random.uniform(*xlims, N_events)
        b = np.random.uniform(*blims, N_events)
        N = np.random.uniform(*N_lims, N_events)

        truth = np.vstack([x, b, N]).T

        events = []

        for i in range(N_events):
            events.append(self.generate_event(x[i], b=b[i], N_src=N[i]))

        return events, truth

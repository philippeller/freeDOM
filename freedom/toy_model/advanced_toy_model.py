import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Helper function
def r2(pos, pos_src):
    r2 = 0.05**2
    for i in range(len(pos_src)):
        r2 += (pos[i]-pos_src[i])**2
    return r2

# Pandel time distribution class
class pandel():
    def __init__(self, T):
        self.T = T
    def pdf(self, t):
        tr = t - self.T
        a = 1./np.clip(self.T, 0.2, 10)
        return np.where((tr <= 0), 1e-100, a**2 * tr * np.exp(-a*tr))
    def logpdf(self, t):
        tr = t - self.T
        a = 1./np.clip(self.T, 0.2, 10)
        return np.where((tr <= 0), np.log(1e-100), np.log(a**2 * tr) - a*tr)
    def rvs(self, size):
        out = []
        while len(out) < size:
            P = np.random.rand(size-len(out))*100 + self.T
            p = self.pdf(P)
            w = np.random.rand(size-len(out))*1.8
            out.extend(P[w<p])
        return np.array(out)


class advanced_toy_experiment():
    def __init__(self, detectors, time_dist=None, charge_dist=None, isotrop=True):
        self.detectors = detectors.T.reshape(len(detectors[0]), len(detectors))
        
        # time PDF factory
        if time_dist == None:
            self.time_dist = lambda t: stats.norm(loc=t, scale=1)
        else:
            self.time_dist = time_dist
        
        # charge PDF factory
        if charge_dist == None:
            self.charge_dist = lambda c: stats.poisson(mu=c)
        else:
            self.charge_dist = charge_dist
            
        self.isotrop = isotrop
    
    
    # time expectation
    def arrival_time(self, pos, t_src, pos_src, angle):
        r_2 = r2(pos, pos_src)
        if self.isotrop:
            A = 1
        else:
            A = (np.cos(angle)*(pos[0]-pos_src[0]) + np.sin(angle)*(pos[1]-pos_src[1]))/np.sqrt(r_2)
            A = 3 - (A+1)
        return t_src + A*np.sqrt(r_2)/3 #v=3
        
    # charge expectation
    def lambda_d(self, pos, pos_src, N_src, angle):
        r_2 = r2(pos, pos_src)
        if self.isotrop:
            A = 1
        else:
            A = (np.cos(angle)*(pos[0]-pos_src[0]) + np.sin(angle)*(pos[1]-pos_src[1]))/np.sqrt(r_2)
            A = (A+1)#/2
        return np.clip(A * N_src/r_2, 0, 100)
    
    
    # PDFs for event generation and likelihood calculation

    def get_p_d_t(self, pos, t_src, pos_src, ang_src):
        return self.time_dist(self.arrival_time(pos, t_src, pos_src, ang_src))
    
    def log_p_d_t(self, pos, t, hypo_pos, hypo_t, hypo_ang):
        tdist = self.get_p_d_t(pos, hypo_t, hypo_pos, hypo_ang)
        return tdist.logpdf(t)
    
    def get_p_d_c(self, pos, pos_src, N_src, ang_src):
        return self.charge_dist(np.sum(self.lambda_d(pos, pos_src, N_src, ang_src)))
    
    def log_p_d_c(self, pos, N, hypo_pos, hypo_N, hypo_ang):
        cdist = self.get_p_d_c(pos, hypo_pos, hypo_N, hypo_ang)
        return cdist.logpmf(N)
    
    
    # LLH terms
    
    def hit_term(self, hit_times, pos_src, t_src, N_src, ang_src):
        '''hit llh for each / sensor reweighted by charge factor'''
        lambda_ds = self.lambda_d(self.detectors, pos_src, N_src, ang_src)
        lambda_hits = lambda_ds[hit_times[:,3].astype(np.int32)]
        p_sensor = lambda_hits/np.sum(lambda_ds)
        
        hpos = hit_times[:,1:3].T.reshape(2, len(hit_times))
        return np.sum(self.log_p_d_t(hpos, hit_times[:,0], pos_src, t_src, ang_src) + np.log(p_sensor))

    def charge_term(self, Ns, pos_src, N_src, ang_src):
        '''total charge llh'''
        lambda_tot = np.sum(self.lambda_d(self.detectors, pos_src, N_src, ang_src))
        N_tot = Ns[0]
        return N_tot*np.log(lambda_tot) - lambda_tot #self.log_p_d_c(self.detectors, N_tot, pos_src, N_src, ang_src)
    
    
    # derivations
    
    def charge_E_E(self, Ns, pos_src, N_src, ang_src):
        return Ns[0]/N_src**2
    def charge_x_x(self, Ns, pos_src, N_src, ang_src):
        R2 = r2(self.detectors, pos_src)
        x, y = self.detectors[0] - pos_src[0], self.detectors[1] - pos_src[1]
        return np.sum(-2*(-Ns[:,0]*x**4 + Ns[:,0]*y**4 + 3*N_src*x**2 - N_src*y**2)/R2**3)
    def charge_y_y(self, Ns, pos_src, N_src, ang_src):
        R2 = r2(self.detectors, pos_src)
        x, y = self.detectors[0] - pos_src[0], self.detectors[1] - pos_src[1]
        return np.sum(-2*(-Ns[:,0]*y**4 + Ns[:,0]*x**4 + 3*N_src*y**2 - N_src*x**2)/R2**3)
    '''
    def hit_t_t(self, hit_times, pos_src, t_src, N_src, ang_src):
        return len(hit_times)
    def hit_x_x(self, hit_times, pos_src, t_src, N_src, ang_src):
        
    def hit_y_y(self, hit_times, pos_src, t_src, N_src, ang_src):
        
    '''
    
    #events

    def generate_event(self, pos_src, t_src=0, N_src=10, ang_src=0):
        ts, ts_sensor_idx, ts_sensor_x, ts_sensor_b = [], [], [], []
        
        lambda_ds = self.lambda_d(self.detectors, pos_src, N_src, ang_src)
        N_obs = np.clip(self.charge_dist(lambda_ds).rvs(), 0, 100)
        for i, n in enumerate(N_obs):
            if n > 0:
                pos = np.array([self.detectors[0][i], self.detectors[1][i]])
                pulse_times = self.get_p_d_t(pos, t_src, pos_src, ang_src).rvs(size=n)
                ts.extend(pulse_times)
                ts_sensor_idx.extend([i]*n)
                ts_sensor_x.extend([pos[0]]*n)
                ts_sensor_b.extend([pos[1]]*n)
        return np.array([np.sum(N_obs), np.sum(N_obs!=0)]), np.array([ts, ts_sensor_x, ts_sensor_b,  ts_sensor_idx]).T


    def generate_events(self, N_events, xlims=(-5, 5), blims=(-2,2), N_lims=(1,20)):
        x = np.random.uniform(*xlims, N_events)
        b = np.random.uniform(*blims, N_events)
        N = np.random.uniform(*N_lims, N_events)
        a = np.random.uniform(*(0,2*np.pi), N_events)

        truth = np.vstack([x, b, N, a]).T

        events = []

        for i in range(N_events):
            events.append(self.generate_event(np.array([x[i], b[i]]), N_src=N[i], ang_src=a[i]))

        return np.array(events), truth

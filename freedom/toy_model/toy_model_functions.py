import numpy as np
from collections import namedtuple
from freedom.utils.pandel import pandel_gen, cpandel_gen
from scipy.spatial import distance
from scipy import stats
from tqdm import tqdm


const_tuple = namedtuple('const', ['lambda_a', 'lambda_s', 'tau', 'n_ref', 'c', 's', 'trck_e_to_l', 'ns_per_trck_m', 'ns_per_cscd_gev', 'noise_level', 'r_pmt', 'q_eff', 'track_step'])

std_consts = const_tuple(lambda_a=100, # Absorbtion length
                     lambda_s=30, # Scattering length
                     tau=500, # some other pandel parameter that i have no idea of
                     n_ref=1.3, # refractive index
                     c=0.3, # speed of light
                     s=10, # time smearing width
                     trck_e_to_l=4.5, # track energy to length factor
                     ns_per_trck_m=2451, # photons per meter track
                     ns_per_cscd_gev = 12819, # photons per GeV cascade
                     noise_level=1e-10, # noise floor
                     r_pmt=0.1, # PMT radius
                     q_eff=0.3, # PMT efficiency
                     track_step=1, # track sampling step length
                    )

class toy_model():
    
    def __init__(self, detector, consts=std_consts):
        
        self.detector = detector
        self.consts = consts
        self.pandel = cpandel_gen(tau=self.consts.tau, lambda_a=self.consts.lambda_a, lambda_s=self.consts.lambda_s, v=self.consts.c/self.consts.n_ref, s=self.consts.s, name='pandel')
  
    def model(self, x, y, z, t, az, zen, e_cscd, e_trck):
        """Simple event model

        Paramaters:
        -----------
        x, y, z, t, az, zen, e_cscd, e_trck : truth parameters

        Returns:
        --------
        collection of emmitters shape (n, 5), each segment with (x, y, z, t, n)
        """    
        length = e_trck * self.consts.trck_e_to_l
        segments = np.arange(0, 1 +np.ceil(length/self.consts.track_step))

        zen = np.pi - zen
        az = az + np.pi

        dx = np.sin(zen) * np.cos(az)
        dy = np.sin(zen) * np.sin(az)
        dz = np.cos(zen)

        X = np.empty((len(segments), 5))

        X[:,0] = x + segments * dx * self.consts.track_step
        X[:,1] = y + segments * dy * self.consts.track_step
        X[:,2] = z + segments * dz * self.consts.track_step
        X[:,3] = t + segments / self.consts.c * self.consts.track_step
        X[:,4] = self.consts.ns_per_trck_m * length / len(segments)
        X[0,4] += self.consts.ns_per_cscd_gev * e_cscd

        return X

    def survival(self, d):
        return np.exp(-d/self.consts.lambda_a) / (d + self.consts.r_pmt)**2 * (self.consts.r_pmt)**2 / 4 * self.consts.q_eff

    def generate_event(self, truth):
        # generate events
        segments = self.model(*truth)

        # x, y, z, d, t, n, idx
        sensors = np.zeros((self.detector.shape[0], 7))
        sensors[:, :3] = self.detector
        sensors[:, 6] = np.arange(self.detector.shape[0])

        sensors_x_segments = np.repeat(sensors[:,np.newaxis,:], segments.shape[0], axis=1)
        sensors_x_segments[:,:, 4] = distance.cdist(self.detector, segments[:,:3])

        n_exp = self.survival(sensors_x_segments[:,:, 4]) * segments[np.newaxis,:,4]
        n_obs = stats.poisson.rvs(mu=n_exp)
        sensors[:, 5] = np.sum(n_obs, axis=1)

        sensors_x_segments[:, :, 3] = segments[np.newaxis,:,3] + sensors_x_segments[:,:, 4]*self.consts.n_ref/self.consts.c
        sensors_x_segments = sensors_x_segments.reshape(-1, 7)

        # unused charge
        sensors_x_segments[:, 5] = 1
        hits = np.repeat(sensors_x_segments, n_obs.flatten(), axis=0)
        hits[:, 3] += self.pandel.rvs(d=hits[:,4])

        return np.delete(hits, 4, axis=1), sensors[:, 5]
    
    @staticmethod
    def sample_sphere(center=np.zeros(3), radius=1):
        x, y, z = np.random.randn(3)
        r = np.sqrt(x**2 + y**2 + z**2)
        u = np.random.rand()
        return (np.array([x, y, z])/r*u**(1/3) * radius) + center
    
    def endpoint(self, x, y, z, t, az, zen, e_cscd, e_trck):
        '''calculate track enpoint'''
        length = e_trck * self.consts.trck_e_to_l

        zen = np.pi - zen
        az = az + np.pi
        d = np.array([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)])
        
        return np.array([x, y, z]) + d * length
    
    def generate_event_sphere(self, n, e_lim=(1,20), inelast_lim=(0,1), t_width=100, N_min=0, center=np.zeros(3), radius=30, contained=True):
        """ Generete events inside a sphere
        
        n : int
            numbr of events
        e_lim : tuple
            limits for energy distribution
        inelast_lim : tuple
            lmits for inelasticity distribution
        t_width : float
            width of time distribution
        N_min : int
            minimum number of pulses
        center : array
            center of generation sphere
        radius : float
            radisu of generation sphere
        contained : bool
            track enpoint must be contained within generation volume
        
        """
        
        events = []
        truths = []
        
        for i in tqdm(range(n)):
            while True:
                while True:
                    x, y, z = self.sample_sphere(center, radius)
                    t = np.random.randn() * t_width
                    az = np.random.uniform(*(0,2*np.pi))
                    zen = np.arccos(np.random.uniform(*(-1,1)))
                    E = np.random.uniform(*e_lim)
                    inelast = np.random.uniform(*inelast_lim)
                    Ecscd = E * (1 - inelast)
                    Etrck = E * inelast

                    truth = np.array([x, y, z, t, az, zen, Ecscd, Etrck])

                    if contained:
                        endpoint = self.endpoint(*truth)
                        if np.sum(endpoint - center) <= radius**2:
                            break
                    else:
                        break

                hits, n_obs = self.generate_event(truth)
                if np.sum(n_obs) >= N_min:
                    break

            truths.append(truth)
            events.append([hits, n_obs])
            
        events = np.array(events, dtype='O')
        truths = np.array(truths)
        
        return events, truths
            
    def generate_event_box(self, n, e_lim=(1,20), N_min=0, x_lim=(-10,10), y_lim=(-10,10), z_lim=(-10,10), coszen_lim=(-1,1), inelast_lim=(0,1), t_width=100, contained=True):
        """ Generete events inside a box
        
        n : int
            numbr of events
        e_lim : tuple
            limits for energy distribution
        inelast_lim : tuple
            lmits for inelasticity distribution
        t_width : float
            width of time distribution
        N_min : int
            minimum number of pulses
        x_lim, y_lim, z_lim : tuples
            limits for vertex distributions
        contained : bool
            track enpoint must be contained within generation volume
        
        """
        events = []
        truths = []
        
        for i in tqdm(range(n)):
        
            while True:
                while True:
                    x = np.random.uniform(*x_lim)
                    y = np.random.uniform(*y_lim)
                    z = np.random.uniform(*z_lim)
                    t = np.random.randn() * t_width
                    az = np.random.uniform(*(0,2*np.pi))
                    zen = np.arccos(np.random.uniform(*coszen_lim))
                    E = np.random.uniform(*e_lim)
                    inelast = np.random.uniform(*inelast_lim)
                    Ecscd = E * (1 - inelast)
                    Etrck = E * inelast

                    truth = np.array([x, y, z, t, az, zen, Ecscd, Etrck])

                    if contained:
                        endpoint = self.endpoint(*truth)
                        if (endpoint[0] >= x_lim[0]) & (endpoint[0] <= x_lim[1]) & (endpoint[1] >= y_lim[0]) & (endpoint[1] <= y_lim[1]) & (endpoint[2] >= z_lim[0]) & (endpoint[2] <= z_lim[1]):
                            break
                    else:
                        break

                hits, n_obs = self.generate_event(truth)
                if np.sum(n_obs) >= N_min:
                    break
        
            truths.append(truth)
            events.append([hits, n_obs])
            
        events = np.array(events, dtype='O')
        truths = np.array(truths)
        
        return events, truths
    
    def p_terms(self, segments, hits):
        hits_x_segments = np.repeat(hits[:,np.newaxis,:], segments.shape[0], axis=1)
        hits_x_segments[:,:, 4] = distance.cdist(hits[:,:3], segments[:,:3])
        n_exp = self.survival(hits_x_segments[:,:, 4]) * segments[np.newaxis,:,4]
        # hit time - true time - direct line-of-sight time
        d_t = hits_x_segments[:,:,3] - segments[np.newaxis,:,3] - hits_x_segments[:,:, 4]*self.consts.n_ref/self.consts.c
        # Mixture of pdfs of all segments for hit in sensor, weighted by n_exp
        ps = np.clip(np.nan_to_num(self.pandel.pdf(d_t, d=hits_x_segments[:,:,4]) ), a_min=0, a_max=None) * n_exp
        # summing up mixture + pedestal (e.g. noise rate) to avoid log(0)
        return np.sum(ps, axis=1) / np.sum(n_exp, axis=1) + self.consts.noise_level   


    def N_exp(self, segments):
        sensors = np.zeros((self.detector.shape[0], 6))
        sensors[:, :3] = self.detector
        sensors_x_segments = np.repeat(sensors[:,np.newaxis,:], segments.shape[0], axis=1)
        sensors_x_segments[:,:, 4] = distance.cdist(self.detector, segments[:,:3])
        return np.sum(self.survival(sensors_x_segments[:,:, 4]) * segments[:,4], axis=1)


    def nllh_p_term_dom(self, segments, hits):
        ps = self.p_terms(segments, hits)
        return -np.sum(np.log(ps))

    def nllh_N_term_dom(self, segments, n_obs):
        N = self.N_exp(segments)
        nllh = N - n_obs * np.log(N)
        return np.sum(nllh)

    def nllh_p_term_tot(self, segments, hits):
        N = self.N_exp(segments)
        N_tot = np.sum(N)

        f = N[hits[:, -1].astype(int)]/N_tot
        ps = self.p_terms(segments, hits) * f

        return -np.sum(np.log(ps))


    def nllh_N_term_tot(self, segments, n_obs):
        N_exp_tot = np.sum(self.N_exp(segments))
        N_obs_tot = np.sum(n_obs)
        return N_exp_tot - N_obs_tot * np.log(N_exp_tot)


    def nllh(self, params, hits, n_obs):
        ''' DOM charge Formulation '''
        segments = self.model(*params)

        # total charge part:
        nllh_N = self.nllh_N_term_dom(segments, n_obs)

        # hit part:
        nllh_p = self.nllh_p_term_dom(segments, hits)

        # putting both together into extended llh
        return nllh_N + nllh_p

    def nllh_formulation2(self, params, hits, n_obs):
        ''' Total charge Formulation '''
        segments = self.model(*params)

        # total charge part:
        nllh_N = self.nllh_N_term_tot(segments, n_obs)

        # hit part:
        nllh_p = self.nllh_p_term_tot(segments, hits)

        # putting both together into extended llh
        return nllh_N + nllh_p
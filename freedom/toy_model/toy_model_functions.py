import numpy as np
from freedom.utils.pandel import pandel_gen, cpandel_gen
from scipy.spatial import distance
from scipy import stats
from tqdm import tqdm
import awkward as ak
import pyarrow.parquet as pq
import json
import getpass
import os

std_config = {}
std_config['lambda_a'] = 100 # Absorbtion length
std_config['lambda_s'] = 30 # Scattering length
std_config['tau'] = 500 # some other pandel parameter that i have no idea of
std_config['n_ref'] = 1.3 # refractive index
std_config['c'] = 0.3 # speed of light
std_config['s'] = 10 # time smearing width
std_config['trck_e_to_l'] = 4.5 # track energy to length factor
std_config['ns_per_trck_m'] = 2451 # photons per meter track
std_config['ns_per_cscd_gev'] = 12819 # photons per GeV cascade
std_config['noise_level'] = 1e-10 # noise floor
std_config['r_pmt'] = 0.1 # PMT radius
std_config['q_eff'] = 0.3 # PMT efficiency
std_config['track_step'] = 1 # track sampling step length


class toy_model():
    
    def __init__(self, detector, config=std_config):
        self.detector = detector
        self.config = config
        self.pandel = cpandel_gen(tau=self.config['tau'], lambda_a=self.config['lambda_a'], lambda_s=self.config['lambda_s'], v=self.config['c']/self.config['n_ref'], s=self.config['s'], name='pandel')
        self.params = ['x', 'y', 'z', 't', 'az', 'zen', 'energy', 'inelast']
  
    def model(self, x, y, z, t, az, zen, energy, inelast):
        """Simple event model

        Paramaters:
        -----------
        x, y, z, t, az, zen, energy, inelast : truth parameters

        Returns:
        --------
        collection of emmitters shape (n, 5), each segment with (x, y, z, t, n)
        """    
        
        e_trck = energy * (1 - inelast)
        e_cscd = energy * inelast
        
        length = e_trck * self.config['trck_e_to_l']
        segments = np.arange(0, 1 +np.ceil(length/self.config['track_step']))

        zen = np.pi - zen
        az = az + np.pi

        dx = np.sin(zen) * np.cos(az)
        dy = np.sin(zen) * np.sin(az)
        dz = np.cos(zen)

        X = np.empty((len(segments), 5))

        X[:,0] = x + segments * dx * self.config['track_step']
        X[:,1] = y + segments * dy * self.config['track_step']
        X[:,2] = z + segments * dz * self.config['track_step']
        X[:,3] = t + segments / self.config['c'] * self.config['track_step']
        X[0,4] = self.config['ns_per_cscd_gev'] * e_cscd + self.config['ns_per_trck_m'] * np.clip(length, 0, self.config['track_step'])
        X[1:,4] = self.config['ns_per_trck_m'] * self.config['track_step']
        X[-1,4] *= (length%self.config['track_step'])/self.config['track_step']

        return X

    def survival(self, d):
        return np.exp(-d/self.config['lambda_a']) / (d + self.config['r_pmt'])**2 * (self.config['r_pmt'])**2 / 4 * self.config['q_eff']

    def generate_event(self, truth, rand=np.random.RandomState()):
        # generate events
        segments = self.model(*truth)

        # x, y, z, d, t, n, idx
        sensors = np.zeros((self.detector.shape[0], 7))
        sensors[:, :3] = self.detector
        sensors[:, 6] = np.arange(self.detector.shape[0])

        sensors_x_segments = np.repeat(sensors[:,np.newaxis,:], segments.shape[0], axis=1)
        sensors_x_segments[:,:, 4] = distance.cdist(self.detector, segments[:,:3])

        n_exp = self.survival(sensors_x_segments[:,:, 4]) * segments[np.newaxis,:,4]
        n_obs = stats.poisson.rvs(mu=n_exp, random_state=rand)
        sensors[:, 5] = np.sum(n_obs, axis=1)

        sensors_x_segments[:, :, 3] = segments[np.newaxis,:,3] + sensors_x_segments[:,:, 4]*self.config['n_ref']/self.config['c']
        sensors_x_segments = sensors_x_segments.reshape(-1, 7)

        # unused charge
        sensors_x_segments[:, 5] = 1
        hits = np.repeat(sensors_x_segments, n_obs.flatten(), axis=0)
        hits[:, 3] += self.pandel.rvs(d=hits[:,4], random_state=rand)

        return np.delete(hits, 4, axis=1), sensors[:, 5]
    
    @staticmethod
    def sample_sphere(center=np.zeros(3), radius=1, rand=np.random.RandomState()):
        x, y, z = rand.randn(3)
        r = np.sqrt(x**2 + y**2 + z**2)
        u = rand.rand()
        return (np.array([x, y, z])/r*u**(1/3) * radius) + center
    
    def endpoint(self, x, y, z, t, az, zen, energy, inelast):
        '''calculate track enpoint'''
        
        e_trck = energy * (1 - inelast)
        length = e_trck * self.config['trck_e_to_l']

        zen = np.pi - zen
        az = az + np.pi
        d = np.array([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)])
        
        return np.array([x, y, z]) + d * length
    
    def limits(self, idx, fraction=0.2):
        low = np.min(self.detector[:, idx])
        high = np.max(self.detector[:, idx])
        ext = high - low
        return (low - fraction * ext, high + fraction * ext)
    
    def generate_events(self, n, outfile=None, gamma=-2, gen_volume="box", e_lim=(1,20), N_min=0, coszen_lim=(-1,1), inelast_lim=(0,1), t_width=100, contained=True, rand=0, **kwargs):
        """ Generete events inside a box
        
        n : int
            numbr of events
        outfile : str
            *.parquet file to write events to
        gamma : negative int
            gamma for energy distribution (if == 0 : uniform)
        gen_volume : str
            "sphere":
                add optional kwargs radius and center, or padding
            or "box":
                add optional kwargs x_lim, y_lim, z_lim, or padding
        e_lim : tuple
            limits for energy distribution
        inelast_lim : tuple
            lmits for inelasticity distribution
        t_width : float
            width of time distribution
        N_min : int
            minimum number of pulses
        contained : bool
            track enpoint must be contained within generation volume
        rand : int or RandomState
        """
        
        if isinstance(rand, int):
            random_state = rand
            rand = np.random.RandomState(rand)
        else:
            if rand is None:
                rand = np.random.RandomState()                        
            if not isinstance(rand, np.random.RandomState):
                state = rand
                rand = np.random.RandomState()        
                rand.set_state(state)
            random_state = rand.get_state()[1]
        
        ak_array = []

        padding = kwargs.get('padding', 0.2)
        
        if gen_volume == "box":
            x_lim = kwargs.get('x_lim', self.limits(0, padding))
            y_lim = kwargs.get('y_lim', self.limits(1, padding))
            z_lim = kwargs.get('z_lim', self.limits(2, padding))
        elif gen_volume == "sphere":
            center = kwargs.get('center', np.mean(self.detector, axis=1))
            radius = kwargs.get('radius', np.max(np.sum(np.square(self.detector - center), axis=1)) * (1 + padding))
        else:
            raise Exception("unknown generation volume %s, must be ['box', 'sphere']"%gen_volume)
        
        for i in tqdm(range(n)):
        
            while True:
                while True:
                    if gen_volume == "box":
                        x = rand.uniform(*x_lim)
                        y = rand.uniform(*y_lim)
                        z = rand.uniform(*z_lim)
                    else:
                        x, y, z = self.sample_sphere(center, radius, rand=rand)

                    t = rand.randn() * t_width
                    az = rand.uniform(*(0,2*np.pi))
                    zen = np.arccos(rand.uniform(*coszen_lim))
                    if gamma == 0:
                        energy = rand.uniform(*e_lim)
                    else:
                        energy = np.inf
                        while energy > e_lim[1]:
                            energy = stats.pareto.rvs(b=-gamma, scale=e_lim[0], random_state=rand)
                    inelast = rand.uniform(*inelast_lim)

                    params = np.array([x, y, z, t, az, zen, energy, inelast])

                    if contained:
                        endpoint = self.endpoint(*params)
                        if gen_volume == "box":
                            if (endpoint[0] >= x_lim[0]) & (endpoint[0] <= x_lim[1]) & (endpoint[1] >= y_lim[0]) & (endpoint[1] <= y_lim[1]) & (endpoint[2] >= z_lim[0]) & (endpoint[2] <= z_lim[1]):
                                break
                        else:
                            if np.sum(np.square(endpoint - center)) <= radius**2:
                                break
                    else:
                        break

                hits, n_obs = self.generate_event(params, rand=rand)
                
                if np.sum(n_obs) >= N_min:
                    break
        
            ak_array.append({'event_idx':i, 'MC_truth': {'x':x, 'y':y, 'z':z, 't':t, 'az':az, 'zen':zen, 'energy':energy, 'inelast':inelast}, 'photons' : {'x':hits[:,0], 'y':hits[:,1], 'z':hits[:,2], 't':hits[:,3], 'sensor_idx':hits[:,5]}, 'n_obs':n_obs})

        
        ak_array = ak.from_iter(ak_array)
        
        meta = {}
        meta['rand'] = random_state
        meta['gen_volume'] = gen_volume
        if gen_volume == "box":
            meta['x_lim'] = x_lim
            meta['y_lim'] = y_lim
            meta['z_lim'] = z_lim
        elif gen_volume == "sphere":
            meta['center'] = center
            meta['radius'] = radius
        meta['n'] = n
        meta['gamma'] = gamma
        meta['e_lim'] = e_lim
        meta['inelast_lim'] = inelast_lim
        meta['t_width'] = t_width
        meta['N_min'] = N_min
        meta['contained'] = contained
        meta['user'] = getpass.getuser()
        meta['system'] = ' '.join(os.uname())
        
        if outfile is not None:
            ak.to_parquet(ak_array, outfile)
            t = pq.read_table(outfile)
            m = t.schema.metadata
            m = m if not m is None else {}
            m['gen_config'] = json.dumps(meta)
            m['exp_config'] = json.dumps(self.config)
            m['detector'] = json.dumps(self.detector.tolist())
            t = t.replace_schema_metadata(m)
            pq.write_table(t, outfile)
        
        return ak_array, meta
    
    def p_terms(self, segments, hits):
        hits_x_segments = np.repeat(hits[:,np.newaxis,:], segments.shape[0], axis=1)
        d = distance.cdist(hits[:,:3], segments[:,:3])
        n_exp = self.survival(d) * segments[np.newaxis,:,4]
        # hit time - true time - direct line-of-sight time
        d_t = hits_x_segments[:,:,3] - segments[np.newaxis,:,3] - d*self.config['n_ref']/self.config['c']
        # Mixture of pdfs of all segments for hit in sensor, weighted by n_exp
        ps = np.clip(np.nan_to_num(self.pandel.pdf(d_t, d=d)), a_min=0, a_max=None) * n_exp
        # summing up mixture + pedestal (e.g. noise rate) to avoid log(0)
        return np.sum(ps, axis=1) / np.sum(n_exp, axis=1) + self.config['noise_level']

    def N_exp(self, segments):
        d = distance.cdist(self.detector, segments[:,:3])
        n_exp = self.survival(d) * segments[np.newaxis,:,4]
        return np.sum(n_exp, axis=1)

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


    def nllh(self, params, hits, n_obs, form='per_dom'):
        segments = self.model(*params)

        if form == 'per_dom':
            # charge part:
            nllh_N = self.nllh_N_term_dom(segments, n_obs)
            # hit part:
            nllh_p = self.nllh_p_term_dom(segments, hits)
        
        elif form == 'all_dom':
            # charge part:
            nllh_N = self.nllh_N_term_tot(segments, n_obs)
            # hit part:
            nllh_p = self.nllh_p_term_tot(segments, hits)

        else:
            raise NameError("Formulation must be one of ['per_dom', 'all_dom'], not "+form)

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
    
    def calc_analytic_llhs(self, g, event, truth):
        '''
        Perform an LLH scan in n-dim

        g : dama.GridData
        event : tuple
            [hits, n_obs]
        truth : value of parameters
        '''

        g['per_dom_hit_term'] = np.empty(g.shape)
        g['per_dom_charge_terms'] = np.empty(g.shape)
        g['all_dom_charge_hit_terms'] = np.empty(g.shape)
        g['all_dom_charge_terms'] = np.empty(g.shape)

        p = np.copy(truth)

        for idx in tqdm(np.ndindex(g.shape)):
            for i in range(g.ndim):
                var = g.grid.vars[i]
                p[self.params.index(var)] = g[var][idx]
            segments = self.model(*p)
            g['per_dom_hit_term'][idx] = self.nllh_p_term_dom(segments, event[0])
            g['per_dom_charge_terms'][idx] = self.nllh_N_term_dom(segments, event[1])
            g['all_dom_charge_hit_terms'][idx] = self.nllh_p_term_tot(segments, event[0])
            g['all_dom_charge_terms'][idx] = self.nllh_N_term_tot(segments, event[1])   
        g['per_dom_hit_term'] -= g['per_dom_hit_term'].min()
        g['per_dom_charge_terms'] -= g['per_dom_charge_terms'].min()
        g['per_dom_llh'] = g['per_dom_hit_term'] + g['per_dom_charge_terms']
        g['all_dom_charge_hit_terms'] -= g['all_dom_charge_hit_terms'].min()
        g['all_dom_charge_terms'] -= g['all_dom_charge_terms'].min()
        g['all_dom_charge_llh'] = g['all_dom_charge_hit_terms'] + g['all_dom_charge_terms']
        g['per_dom_llh'] -= g['per_dom_llh'].min()
        g['all_dom_charge_llh'] -= g['all_dom_charge_llh'].min()

        return g
    

    def calc_NN_llhs(self, g, event, truth, chargenet, hitnet):
        '''
        Perform an LLH scan in n-dim

        g : dama.GridData
        event : tuple
            [hits, n_obs]
        truth : value of parameters
        chargenet : all_dom chargenet
        hitnet : all_dom hitnet
        toy_experiment : toy_experiment
        '''

        xxs = np.tile([np.sum(event[1]), np.sum(event[1]>0)], np.prod(g.shape))
        xxs = xxs.reshape(-1, 2)
        tts = np.repeat(truth[np.newaxis, :], np.prod(g.shape), axis=0)

        for i in range(g.ndim):
            var = g.grid.vars[i]
            tts[:, self.params.index(var)] = g.get_array(var, flat=True)

        llhs = np.nan_to_num(-chargenet.predict((xxs, tts), batch_size=4096))

        g.chargenet_llh = llhs.reshape(g.shape)
        g.chargenet_llh -= g.chargenet_llh.min()

        xxs = np.repeat(event[0][:, :4][np.newaxis, :], np.prod(g.shape), axis=0)
        xxs = xxs.reshape(-1, 4)


        tts = np.repeat(tts, len(event[0]), axis=0)
        llhs = -hitnet.predict((xxs, tts), batch_size=4096)

        llhs = np.nan_to_num(np.sum(llhs.reshape(-1, len(event[0])), axis=1))

        g.hitnet_llh = llhs.reshape(g.shape)

        g.hitnet_llh -= g.hitnet_llh.min()

        g.freedom_llh = g.hitnet_llh + g.chargenet_llh
        g.freedom_llh -= g.freedom_llh.min()

        return g
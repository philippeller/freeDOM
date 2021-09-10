import numpy as np
from scipy import stats
from scipy.special import gamma, hyp1f1


# Helper function
def r2(pos, pos_src):
    r2 = 0.1**2
    for i in range(len(pos_src)):
        r2 += (pos[i]-pos_src[i])**2
    return r2


# Pandel time distribution class convolved with Gauss
class pandel():
    def __init__(self, T, t_hypo=0):
        self.T = T
        self.t_hypo = t_hypo
        self.a = 1/(self.T - self.t_hypo)
        self.s = 1
    
    def pdf(self, t):
        tr = t - self.T
        tr = np.clip(tr, -3, 1e3)
        
        rho = np.clip(self.a*self.s - tr/self.s, -31, 7)
        A = self.a**2*self.s*np.exp(-tr**2/(2*self.s**2)) / 2**(3/2)
        return np.where(rho < -30, 
                        self.a**2 * tr * np.exp(-self.a*tr),
                        A * (hyp1f1(1,0.5,0.5*rho**2)/gamma(1.5) - np.sqrt(2)*rho*hyp1f1(1.5,1.5,0.5*rho**2))
                       )
    
    def logpdf(self, t):
        return np.log(self.pdf(t))
    
    def rvs(self, size):
        max_y = self.pdf(2.48/self.a) + 0.01
        max_x = -np.log(max_y*1e-5 / self.a**2) / self.a + 3
        
        out = []
        while len(out) < size:
            P = np.random.rand(size-len(out))*max_x + self.T - 2
            p = self.pdf(P)
            w = np.random.rand(size-len(out))*max_y
            out.extend(P[w<=p])
        
        return np.array(out)


class advanced_toy_experiment():
    def __init__(self, detectors, time_dist=None, charge_dist=None):
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
            
        self.speed_of_light = 1
    
    
    def ClosestApproachCalc(self, pos, pos_src, angles):
        pos = np.array(pos).T
        theta  = angles[1]
        phi    = angles[0]

        e_x = -np.sin(theta)*np.cos(phi)
        e_y = -np.sin(theta)*np.sin(phi)
        e_z = -np.cos(theta)

        h_x = pos[:, 0] - pos_src[0]
        h_y = pos[:, 1] - pos_src[1]
        h_z = pos[:, 2] - pos_src[2]

        s = e_x*h_x + e_y*h_y + e_z*h_z
        
        pos2_x = pos_src[0] + s*e_x
        pos2_y = pos_src[1] + s*e_y
        pos2_z = pos_src[2] + s*e_z

        appos = np.stack([pos2_x, pos2_y, pos2_z], axis=1)
        apdist = np.linalg.norm(pos-appos, axis=1)
        Dir = np.array([e_x, e_y, e_z])

        return appos, apdist, s, Dir
    
    # time expectation
    def arrival_time(self, pos, t_src, pos_src, angles, trackE, Index=1.33):
        changle = np.arccos(1/Index)
        length = trackE

        appos, apdist, s, Dir = self.ClosestApproachCalc(pos, pos_src, angles)
        a = s - apdist/np.tan(changle)
        
        t = np.where(a <= 0, np.linalg.norm(pos.T-pos_src, axis=1) * Index/self.speed_of_light,
                     np.where(a <= length, (a + apdist/np.sin(changle)*Index) / self.speed_of_light,
                              (length + np.linalg.norm(pos.T-(pos_src + length*Dir), axis=1)*Index) / self.speed_of_light
                             )
                    )
        
        return t_src + t
    
    # charge expectation
    def lambda_d(self, pos, pos_src, angles, Ecscd, Etrck):
        C_cscd = Ecscd/r2(pos, pos_src)
        
        if Etrck == 0:
            return np.clip(C_cscd, 0, 50)

        length = Etrck
        appos, apdist, s, Dir = self.ClosestApproachCalc(pos, pos_src, angles)
        
        C1 = 1/(1+np.exp(20*(np.abs(np.clip(s/length,-3,3)-0.5)-0.5))) * 0.5*Etrck/(apdist**2+0.1**2)
        C2 = 0.25*Etrck/r2(pos, pos_src + length*Dir)
        C_trck = C1 + C2
        
        return np.clip(C_cscd + C_trck, 0, 50)
    
    
    # PDFs for event generation and likelihood calculation

    def get_p_d_t(self, pos, t_src, pos_src, ang_src, trackE):
        return self.time_dist(self.arrival_time(pos, t_src, pos_src, ang_src, trackE), t_src)
    
    def log_p_d_t(self, pos, t, hypo_pos, hypo_t, hypo_ang, hypo_trackE):
        tdist = self.get_p_d_t(pos, hypo_t, hypo_pos, hypo_ang, hypo_trackE)
        return tdist.logpdf(t)
    
    def get_p_d_c(self, pos, pos_src, angles, Ecscd, Etrck):
        return self.charge_dist(np.sum(self.lambda_d(pos, pos_src, angles, Ecscd, Etrck)))
    
    def log_p_d_c(self, pos, N, hypo_pos, hypo_ang, hypo_Ecscd, hypo_Etrck):
        cdist = self.get_p_d_c(pos, hypo_pos, hypo_ang, hypo_Ecscd, hypo_Etrck)
        return cdist.logpmf(N)
    
    
    # LLH terms
    
    def hit_term(self, hit_times, pos_src, t_src, ang_src, Ecscd, Etrck):
        '''hit llh for each / sensor reweighted by charge factor'''
        lambda_ds = self.lambda_d(self.detectors, pos_src, ang_src, Ecscd, Etrck)
        lambda_hits = lambda_ds[hit_times[:,4].astype(np.int32)]
        p_sensor = lambda_hits/np.sum(lambda_ds)
        
        hpos = hit_times[:,1:4].T.reshape(3, len(hit_times))
        return np.sum(self.log_p_d_t(hpos, hit_times[:,0], pos_src, t_src, ang_src, Etrck) + np.log(p_sensor))

    def charge_term(self, Ns, pos_src, ang_src, Ecscd, Etrck):
        '''total charge llh'''
        lambda_tot = np.sum(self.lambda_d(self.detectors, pos_src, ang_src, Ecscd, Etrck))
        N_tot = Ns[0]
        return N_tot*np.log(lambda_tot) - lambda_tot #self.log_p_d_c(self.detectors, N_tot, pos_src, N_src, ang_src)
    
    
    #events

    def generate_event(self, pos_src, ang_src, Ecscd, Etrck):
        t_src = 0
        ts, sensor_idx, sensor_x, sensor_y, sensor_z = [], [], [], [], []
        
        lambda_ds = self.lambda_d(self.detectors, pos_src, ang_src, Ecscd, Etrck)
        N_obs = np.clip(self.charge_dist(lambda_ds).rvs(), 0, 50)
        for i, n in enumerate(N_obs):
            if n > 0:
                pos = np.array([self.detectors[0][i], self.detectors[1][i], self.detectors[2][i]])
                pulse_times = self.get_p_d_t(np.array([pos]).T, t_src, pos_src, ang_src, Etrck).rvs(size=n)
                ts.extend(pulse_times)
                sensor_idx.extend([i]*n)
                sensor_x.extend([pos[0]]*n)
                sensor_y.extend([pos[1]]*n)
                sensor_z.extend([pos[2]]*n)
        return np.array([np.sum(N_obs), np.sum(N_obs!=0)]), np.array([ts, sensor_x, sensor_y, sensor_z, sensor_idx]).T


    def generate_events(self, N_events):
        x = np.random.uniform(*(-2, 12), N_events)
        y = np.random.uniform(*(-12, 4), N_events)
        z = np.random.uniform(*(-18, 18), N_events)
        azi = np.random.uniform(*(0,2*np.pi), N_events)
        zen = np.arccos(np.random.uniform(*(-1,1), N_events))
        Ecscd = np.random.uniform(*(1, 30), N_events)
        Etrck = np.random.uniform(*(0, 30), N_events)
        
        Etrck[np.random.choice(range(N_events), int(N_events/3.))] = 0

        truth = np.vstack([x, y, z, np.zeros(N_events), azi, zen, Ecscd, Etrck]).T

        events = []

        for i in range(N_events):
            pos = np.array([x[i], y[i], z[i]])
            ang = np.array([azi[i], zen[i]])
            events.append(self.generate_event(pos, ang, Ecscd[i], Etrck[i]))

        return np.array(events), truth

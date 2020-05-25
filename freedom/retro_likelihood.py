import sys
if 'pytest' in sys.argv[0]:
    sys.exit()
import numpy as np
from retro import init_obj
from retro.retro_types import EVT_DOM_INFO_T, EVT_HIT_INFO_T, FitStatus
from retro.tables.pexp_5d import generate_pexp_and_llh_functions
from retro.const import SRC_CKV_BETA1 
from retro.retro_types import SRC_T

class retroLLH():
    def __init__(self,
                 dom_tables_kind="ckv_templ_compr", 
                 dom_tables_fname_proto="/home/iwsatlas1/peller/work/retro_tables/SpiceLea/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}", 
                 gcd="/home/iwsatlas1/peller/retro/retro_data/GeoCalibDetectorStatus_IC86.55697_corrected_V2.pkl",
                 template_library="/home/iwsatlas1/peller/work/retro_tables/SpiceLea/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy",
                 cascade_kernel='aligned_one_dim',
                 track_kernel='table_energy_loss',
                 track_time_step=1,
                 ):

        self.dom_tables = init_obj.setup_dom_tables(dom_tables_kind=dom_tables_kind, 
                                       dom_tables_fname_proto=dom_tables_fname_proto, 
                                       gcd=gcd,
                                       template_library=template_library,
                                      )

        pexp, self.get_llh, _ = generate_pexp_and_llh_functions(dom_tables=self.dom_tables)

        self.hypo_handler = init_obj.setup_discrete_hypo(cascade_kernel=cascade_kernel,
                                            track_kernel=track_kernel,
                                            track_time_step=track_time_step)
        
        
    def __call__(self, event, params):
        """Evaluate LLH for a given event + params

        event : dict containing:
            'total_charge' : float
            'hits' : array shape (n_hits, 5)
                each row is (x, y, z) DOM poitions, time, charge
        params : ndarray
            shape (n_likelihood_points, len(labels)) 
        """
        if params.ndim == 1:
            params = np.array([params])
        n_points = params.shape[0]   
        
        # create all the crazy arrays needed by retro:
        event_hit_info = np.zeros(shape=event['hits'].shape[0], dtype=EVT_HIT_INFO_T)
        for i in range(event['hits'].shape[0]):
            event_hit_info[i]['event_dom_idx'] = event['hits'][i][7] + 86 * event['hits'][i][8]
            event_hit_info[i]['time'] = event['hits'][i][3]
            event_hit_info[i]['charge'] = event['hits'][i][4]

        dom_info = self.dom_tables.dom_info
        num_operational_doms = np.sum(dom_info["operational"]) 
        sd_idx_table_indexer = self.dom_tables.sd_idx_table_indexer
        event_dom_info = np.zeros(shape=num_operational_doms, dtype=EVT_DOM_INFO_T)

        copy_fields = [
                    "sd_idx",
                    "x",
                    "y",
                    "z",
                    "quantum_efficiency",
                    "noise_rate_per_ns",]
        
        # Fill `event_{hit,dom}_info` arrays only for operational DOMs                                                                                                                                                         
        for dom_idx, this_dom_info in enumerate(dom_info[dom_info["operational"]]):                                                                                                                               
            this_event_dom_info = event_dom_info[dom_idx : dom_idx + 1]
            this_event_dom_info[copy_fields] = this_dom_info[copy_fields]
            sd_idx = this_dom_info["sd_idx"]
            this_event_dom_info["table_idx"] = sd_idx_table_indexer[sd_idx]
            
            # Copy any hit info from `hits_indexer` and total charge from
            # `hits` into `event_hit_info` and `event_dom_info` arrays
            hit_indices = event_hit_info['event_dom_idx'] == sd_idx
            if np.any(hit_indices):
                w = np.where(hit_indices)[0]
                this_event_dom_info["hits_start_idx"] = w[0]
                this_event_dom_info["hits_stop_idx"] = w[-1] + 1
                this_event_dom_info["total_observed_charge"] = np.sum(event_hit_info['charge'][hit_indices])
                
        llhs = np.zeros(n_points, dtype=np.float32)
        for i in range(n_points):
            hypo = {}
            hypo['x'] = params[i, 0]
            hypo['y'] = params[i, 1]
            hypo['z'] = params[i, 2]
            hypo['time'] = params[i, 3]
            hypo['track_azimuth'] = params[i, 4]
            hypo['track_zenith'] = params[i, 5]
            hypo['cascade_energy'] = params[i, 6]
            hypo['track_energy'] = params[i, 7]

        
            generic_sources = self.hypo_handler.get_generic_sources(hypo)
            pegleg_sources = self.hypo_handler.get_pegleg_sources(hypo)
            scaling_sources = self.hypo_handler.get_scaling_sources(hypo)     

            get_llh_retval = self.get_llh(
                            generic_sources=generic_sources,
                            pegleg_sources=pegleg_sources,
                            scaling_sources=scaling_sources,
                            event_hit_info=event_hit_info,
                            event_dom_info=event_dom_info,
                            pegleg_stepsize=1,
                            )

            llh, pegleg_idx, scalefactor = get_llh_retval[:3]
            llhs[i] = -llh
        return llhs

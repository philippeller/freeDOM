'''Module to read in i3cols format data
i3cols is available here: https://github.com/jllanfranchi/i3cols
'''
import os
import pkg_resources
import numpy as np
from scipy.stats import iqr

def get_energies(mcprimary, mctree, mctree_idx, dtype=np.float32):
    '''Get energies per event'''
    
    neutrino_energy = mcprimary['energy']
    track_energy = np.zeros_like(neutrino_energy, dtype=dtype)
    invisible_energy = np.zeros_like(neutrino_energy, dtype=dtype)
    
    for i in range(len(mctree_idx)):
        this_idx = mctree_idx[i]
        this_mctree = mctree[this_idx[0] : this_idx[1]]
        pdg = this_mctree['particle']['pdg_encoding']
        en = this_mctree['particle']['energy']
        parent = pdg[this_mctree['parent_idx']]
        #parent[0] = 0
    
        muon_mask = np.abs(pdg) == 13
        if np.any(muon_mask):
            track_energy[i] = np.max(en[muon_mask])

        invisible_mask = (np.abs(pdg) == 12) | (np.abs(pdg) == 14) | (np.abs(pdg) == 16) 
        # exclude primary:
        invisible_mask[0] = False
        if np.any(invisible_mask):
            # we'll make the bold assumptions that none of the neutrinos re-interact ;)
            invisible_energy[i] = np.sum(en[(invisible_mask) & (np.abs(parent)!=13)])

    cascade_energy = neutrino_energy - track_energy - invisible_energy
    return neutrino_energy, track_energy, cascade_energy

def get_params(labels, mcprimary, mctree, mctree_idx, dtype=np.float32):
    '''construct params array

    retruns:
    params : array shape (len(mcprimary), len(labels))
    '''

    neutrino_energy, track_energy, cascade_energy = get_energies(mcprimary, mctree, mctree_idx, dtype=dtype)
    
    params = np.empty(mcprimary.shape + (len(labels), ), dtype=dtype)

    for i, label in enumerate(labels):
        if label == 'x': params[:, i] = mcprimary['pos']['x']
        elif label == 'y': params[:, i] = mcprimary['pos']['y']
        elif label == 'z': params[:, i] = mcprimary['pos']['z']
        elif label == 'time': params[:, i] = mcprimary['time']
        elif label == 'azimuth': params[:, i] = mcprimary['dir']['azimuth']
        elif label == 'zenith': params[:, i] = mcprimary['dir']['zenith']
        elif label == 'neutrino_energy': params[:, i] = neutrino_energy
        elif label == 'energy': params[:, i] = track_energy + cascade_energy
        elif label == 'cascade_energy': params[:, i] = cascade_energy
        elif label == 'track_energy': params[:, i] = track_energy

    return params



def load_charges(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 dtype=np.float32,
                 pulses=['SRTTWOfflinePulsesDC'],
                 geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                 data=False):
    """
    Create training data for chargenet
    
    Returns:
    --------
    total_charge : ndarray
        shape (N_events, 2)
    params : ndarray
        shape (N_events, len(labels))
    labels
    """
    
    #geo = np.load(geo)
    if not type(pulses) == list:
        pulses = [pulses]
    N_events = len(np.load(os.path.join(dir, 'I3EventHeader/data.npy')))
    
    if not data:
        mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
        mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
        if os.path.exists(os.path.join(dir, 'MCInIcePrimary/data.npy')):
            mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))
        else:
            mcprimary = mctree['particle'][mctree['level'] == 0]

    # Get charge per event
    total_charge = np.zeros((N_events, 2*len(pulses)), dtype=dtype) #4 8
    for j in range(len(pulses)):
        hits_idx = np.load(os.path.join(dir, pulses[j]+'/index.npy'))
        hits = np.load(os.path.join(dir, pulses[j]+'/data.npy'))
        
        for i in range(len(hits_idx)):
            this_idx = hits_idx[i]
            this_hits = hits[this_idx['start'] : this_idx['stop']]

            total_charge[i][2*j] = np.sum(this_hits['pulse']['charge'])
            total_charge[i][2*j+1] = len(np.unique(this_hits['key']))
            #total_charge[i][2] = len(np.unique(this_hits['key']['string']*113 + this_hits['key']['om']))
            
    if not data:
        params = get_params(labels, mcprimary, mctree, mctree_idx)
    else:
        params = None
    return total_charge, params, labels

def load_doms(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
              labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
              geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
              dtype=np.float32,
              reduced=False,
              include_params=True):
    """
    Create training data for domnet, if reduced is True just uses DOMs in resources/allowed_DOMs.npy
    
    Returns:
    --------
    dom_charges : ndarray
        shape (N_events*N_usedDOMs, 4)
    repeated_params : ndarray
        shape (N_events*N_usedDOMs, len(labels))
    labels
    """
    
    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    geo = np.load(geo)
    if reduced:
        allowed_DOMs = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_DOMs.npy'))
    else:
        allowed_DOMs = np.arange(5160)
    
    # construct strings array
    # shape N x N_usedDOMs x (x, y, z, q)
    
    doms = np.zeros(hits_idx.shape + (len(allowed_DOMs), 4,), dtype=dtype)
    doms[:, :, 0:3] = geo.reshape((5160, 3))[allowed_DOMs]

    # Get charge per event and DOM
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        this_hits = hits[this_idx['start'] : this_idx['stop']]
        for hit in this_hits:
            idx = (hit['key']['string'] - 1) * 60 + hit['key']['om'] - 1
            idx = np.where(allowed_DOMs == idx)[0]
            if len(idx) == 1:
                doms[i, idx[0], 3] += hit['pulse']['charge']

    doms = doms.reshape(-1, 4)
    
    if include_params:
        params = get_params(labels, mcprimary, mctree, mctree_idx)

        repeated_params = np.repeat(params, repeats=len(allowed_DOMs), axis=0)

        return doms, repeated_params, labels
    
    else:
        return doms, labels

def load_strings(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
                 labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                 geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                 dtype=np.float32,
                 reduced=False):
    """
    Create training data for stringnet
    
    Returns:
    --------
    string_charges : ndarray
        shape (N_events*86, 5)
    repeated_params : ndarray
        shape (N_events*86, len(labels))
    labels
    """
    
    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    geo = np.load(geo)
    if reduced:
        allowed_strings = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_strings.npy'))
    else:
        allowed_strings = np.arange(86)
    
    # construct strings array
    # shape N x N_usedStrings x (x, y, min(z), q, nChannels)
    
    strings = np.zeros(hits_idx.shape + (len(allowed_strings), 5,), dtype=dtype)
    strings[:, :, 0:3] = geo[np.newaxis, :, -1][0][allowed_strings]

    # Get charge per event and string
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        this_hits = hits[this_idx['start'] : this_idx['stop']]
        for j, hit in enumerate(this_hits):
            s_idx = hit['key']['string'] - 1
            s_idx = np.where(allowed_strings == s_idx)[0]
            if len(s_idx) == 1:
                strings[i, s_idx[0], 3] += hit['pulse']['charge']
                if j == 0 or hit['key'] != this_hits[j-1]['key']: # assuming hits are sorted by DOMs
                    strings[i, s_idx[0], 4] += 1

    strings = strings.reshape(-1, 5)
    
    params = get_params(labels, mcprimary, mctree, mctree_idx)

    repeated_params = np.repeat(params, repeats=len(allowed_strings), axis=0)
    
    return strings, repeated_params, labels

def load_layers(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
                labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                dtype=np.float32,
                n_layers=60,
                reduced=False):
    """
    Create training data for layernet
    
    Returns:
    --------
    layer_charges : ndarray
        shape (N_events*n_layers, 4)
    repeated_params : ndarray
        shape (N_events*n_layers, len(labels))
    labels
    """
    
    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    hits = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/data.npy'))
    mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
    mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
    mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))

    geo = np.load(geo)
    if reduced:
        assert n_layers == 60, 'At the moment reduction will only work properly with 60 layers'
        allowed_layers = np.load(pkg_resources.resource_filename('freedom', 'resources/allowed_layers.npy'))
    else:
        allowed_layers = np.arange(n_layers)
    min_z, max_z = np.min(geo[:, : , 2]), np.max(geo[:, : , 2])
    z_edges = np.linspace(min_z, max_z+1e-3, n_layers+1)
    
    # construct layers array
    # shape N x N_usedLayers x (nDOMs, z, q, nChannels)
    
    layers = np.zeros(hits_idx.shape + (len(allowed_layers), 4,), dtype=dtype)
    layers[:, :, 0] = np.histogram(geo[:, : , 2].flatten(), z_edges)[0][allowed_layers]
    layers[:, :, 1] = ((z_edges[:-1]+z_edges[1:])/2)[allowed_layers]

    # Get charge per event and layer
    for i in range(len(hits_idx)):
        this_idx = hits_idx[i]
        this_hits = hits[this_idx['start'] : this_idx['stop']]
        for j, hit in enumerate(this_hits):
            z = geo[hit['key'][0]-1, hit['key'][1]-1, 2]
            b = np.digitize(z, z_edges)-1
            b = np.where(allowed_layers == b)[0]
            if len(b) == 1:
                layers[i, b, 2] += hit['pulse']['charge']
                if j == 0 or hit['key'] != this_hits[j-1]['key']: # assuming hits are sorted by DOMs
                    layers[i, b, 3] += 1

    layers = layers.reshape(-1, 4)
    
    params = get_params(labels, mcprimary, mctree, mctree_idx)

    repeated_params = np.repeat(params, repeats=len(allowed_layers), axis=0)
    
    return layers, repeated_params, labels


def load_hits(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
              labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
              geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
              dtype=np.float32,
              pulses='SRTTWOfflinePulsesDC',
              data=False):
    """
    Create training data for hitnet
    
    Returns:
    --------
    single_hits : ndarray
        shape (N_hits, 8)
    repeated_params : ndarray
        shape (N_hits, len(labels))
    labels
    """
    
    hits_idx = np.load(os.path.join(dir, pulses+'/index.npy'))
    hits = np.load(os.path.join(dir, pulses+'/data.npy'))
    if not data:
        mctree_idx = np.load(os.path.join(dir, 'I3MCTree/index.npy'))
        mctree = np.load(os.path.join(dir, 'I3MCTree/data.npy'))
        if os.path.exists(os.path.join(dir, 'MCInIcePrimary/data.npy')):
            mcprimary = np.load(os.path.join(dir, 'MCInIcePrimary/data.npy'))
        else:
            mcprimary = mctree['particle'][mctree['level'] == 0]

    geo = np.load(geo)
    if 'mDOM' in pulses:
        pmt_dirs = np.load(pkg_resources.resource_filename('freedom', 'resources/mdom_directions.npy'))
    elif 'DEgg' in pulses:
        pmt_dirs =np.array([[0, np.pi], [np.pi, np.pi]]) #[zen, azi], zen: 0=down
    else:
        pmt_dirs = np.array([[0, np.pi]])
    
    # construct hits array
    
    # shape N x (x, y, z, t, q, flags, pmt dirs, idx)
    single_hits = np.empty(hits.shape + (10,), dtype=dtype)
    om_idx = hits['key']['om'] - 1
    pmt_idx = hits['key']['pmt']
    if 'mDOM' in pulses or 'DEgg' in pulses or 'PDOM' in pulses:
        string_idx = hits['key']['string'] - 87
        idx = string_idx * 7 + om_idx * 24 + pmt_idx
    else:
        string_idx = hits['key']['string'] - 1
        idx = string_idx * 60 + om_idx

    single_hits[:, 0:3] = geo[string_idx, om_idx]
    single_hits[:, 3] = hits['pulse']['time']
    single_hits[:, 4] = hits['pulse']['charge']
    single_hits[:, 5] = hits['pulse']['flags'] & 1 # is LC or not?
    single_hits[:, 6] = (hits['pulse']['flags'] & 2) / 2 # has ATWD or not?
    single_hits[:, 7:9] = pmt_dirs[pmt_idx] 
    single_hits[:, 9] = idx # for inDOM shuffling
    
    if not data:
        params = get_params(labels, mcprimary, mctree, mctree_idx)

        repeats = (hits_idx['stop'] - hits_idx['start']).astype(np.int64)
        repeated_params = np.repeat(params, repeats=repeats, axis=0)
    else:
        repeated_params = None

    return single_hits, repeated_params, labels


def load_events(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
               labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
               geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
               recos = {},
               dtype=np.float32,
               include_doms=False,
               data=False):
    """
    Create event=by=event data for hit and charge net
    
    Returns:
    --------
    list of:
        single_hits : ndarray
            shape (N_hits, 9)
        total_charge : float
        params : ndarray
            shape (len(labels)) (if not data)
    labels
    """
    
    hits_idx = np.load(os.path.join(dir, 'SRTTWOfflinePulsesDC/index.npy'))
    
    single_hits, repeated_params, labels = load_hits(dir=dir, labels=labels, geo=geo, dtype=dtype, data=data)

    total_charge, params, labels = load_charges(dir=dir, labels=labels, dtype=dtype, data=data)

    #string_charges, _, _ = load_strings(dir=dir, labels=labels, geo=geo, dtype=dtype)
    #string_charges = string_charges.reshape(len(total_charge), 86, -1)
    
    #layer_charges, _, _ = load_layers(dir=dir, labels=labels, geo=geo, dtype=dtype, n_layers=60)
    #layer_charges = layer_charges.reshape(len(total_charge), 60, -1)
    
    if include_doms:
        dom_charges, _ = load_doms(dir=dir, labels=labels, 
                                   geo=geo, dtype=dtype, include_params=False)
        dom_charges = dom_charges.reshape(len(total_charge), 5160, -1)
        
    reco_params = {}
    for r,f in recos.items():
        reco = np.load(os.path.join(dir, f, 'data.npy'))
        reco_params[r] = np.zeros((len(total_charge), len(labels))) #np.zeros_like(params)
        for i, label in enumerate(labels):
            if label == 'x': reco_params[r][:, i] = reco['pos']['x']
            elif label == 'y': reco_params[r][:, i] = reco['pos']['y']
            elif label == 'z': reco_params[r][:, i] = reco['pos']['z']
            elif label == 'time': reco_params[r][:, i] = reco['time']
            elif label == 'azimuth': reco_params[r][:, i] = reco['dir']['azimuth']
            elif label == 'zenith': reco_params[r][:, i] = reco['dir']['zenith']
            elif label == 'energy': reco_params[r][:, i] = reco['energy']
        
        # for retro unfortunately these are in different keys...
        if 'track_energy' in labels and f.strip('/').endswith('__neutrino'):
            reco_track = np.load(os.path.join(dir, f.replace('__neutrino','__track'), 'data.npy'))
            idx = labels.index('track_energy')
            reco_params[r][:, idx] = reco_track['energy']

        if 'cascade_energy' in labels and f.strip('/').endswith('__neutrino'):
            reco_track = np.load(os.path.join(dir, f.replace('__neutrino','__cascade'), 'data.npy'))
            idx = labels.index('cascade_energy')
            reco_params[r][:, idx] = reco_track['energy']
            

            
    events = []
    
    for i in range(len(total_charge)):
        event = {}
        event['total_charge'] = total_charge[i]
        event['hits'] = single_hits[hits_idx[i]['start'] : hits_idx[i]['stop']]
        if not data:
            event['params'] = params[i]
        #event['strings'] = string_charges[i]
        #event['layers'] = layer_charges[i]
        if include_doms:
            event['doms'] = dom_charges[i]
        for r in recos.keys():
            event[r] = reco_params[r][i]
        events.append(event)
    return events, labels


def load_upgrade_events(dir='/home/iwsatlas1/peller/work/oscNext/level7_v01.04/140000_i3cols',
                       labels=['x', 'y', 'z', 'time', 'azimuth','zenith', 'cascade_energy', 'track_energy'],
                       geo=pkg_resources.resource_filename('freedom', 'resources/geo_array.npy'),
                       geo_upgarde=pkg_resources.resource_filename('freedom', 'resources/geo_array_upgrade.npy'),
                       DOM_key = 'IceCubePulsesTWSRT',
                       mDOM_key = 'mDOMPulsesTWSRT',
                       DEgg_key = 'DEggPulsesTWSRT',
                       dtype=np.float32):
    """
    Create event=by=event data for hit and charge net
    
    Returns:
    --------
    list of:
        single_hits : ndarray
            shape (N_hits, 9)
        total_charge : float
        params : ndarray
            shape (len(labels))
    labels
    """
    
    hits_idx_DOM = np.load(os.path.join(dir, DOM_key+'/index.npy'))
    hits_idx_mDOM = np.load(os.path.join(dir, mDOM_key+'/index.npy'))
    hits_idx_DEgg = np.load(os.path.join(dir, DEgg_key+'/index.npy'))
    
    single_hits_DOM, _, _ = load_hits(dir=dir, labels=labels, geo=geo, dtype=dtype, pulses=DOM_key)
    single_hits_mDOM, _, _ = load_hits(dir=dir, labels=labels, geo=geo_upgarde, dtype=dtype, pulses=mDOM_key)
    single_hits_DEgg, _, _ = load_hits(dir=dir, labels=labels, geo=geo_upgarde, dtype=dtype, pulses=DEgg_key)

    total_charge_DOM, params, _ = load_charges(dir=dir, labels=labels, dtype=dtype, pulses=DOM_key)
    total_charge_mDOM, _, _ = load_charges(dir=dir, labels=labels, dtype=dtype, pulses=mDOM_key)
    total_charge_DEgg, _, _ = load_charges(dir=dir, labels=labels, dtype=dtype, pulses=DEgg_key)

            
    events = []
    for i in range(len(total_charge_DOM)):
        if np.sum(total_charge_DOM[i]) + np.sum(total_charge_mDOM[i]) + np.sum(total_charge_DEgg[i]) == 0:
            continue
        
        event = {}
        event['total_charge_DOM'] = total_charge_DOM[i]
        event['total_charge_mDOM'] = total_charge_mDOM[i]
        event['total_charge_DEgg'] = total_charge_DEgg[i]
        event['hits_DOM'] = single_hits_DOM[hits_idx_DOM[i]['start'] : hits_idx_DOM[i]['stop']]
        event['hits_mDOM'] = single_hits_mDOM[hits_idx_mDOM[i]['start'] : hits_idx_mDOM[i]['stop']]
        event['hits_DEgg'] = single_hits_DEgg[hits_idx_DEgg[i]['start'] : hits_idx_DEgg[i]['stop']]
        event['params'] = params[i]
        events.append(event)
    
    return events, labels

"""Provides functions for extracting event data from i3frames"""
import numpy as np

DEFAULT_LABELS = [
    "x",
    "y",
    "z",
    "time",
    "azimuth",
    "zenith",
    "cascade_energy",
    "track_energy",
]


def _is_invisible(particle, nu_ids=[12, 14, 16]):
    abs_id = abs(particle.pdg_encoding)
    return any(abs_id == nu_id for nu_id in nu_ids)


def _is_muon(particle, muon_id=13):
    return abs(particle.pdg_encoding) == muon_id


def _calc_energies(mc_primary, mc_tree):
    """returns [cascade_energy, track_energy]"""
    nu_E = mc_primary.energy

    track_E = max((p.energy for p in mc_tree if _is_muon(p)), default=0)
    # skip mc_tree[0] to exclude the primary from the calculation of invis_E
    invis_E = sum(p.energy for p in mc_tree[1:] if _is_invisible(p))

    return [nu_E - invis_E - track_E, track_E]


_param_getters = dict(
    x=lambda p: p.pos.x,
    y=lambda p: p.pos.y,
    z=lambda p: p.pos.z,
    time=lambda p: p.time,
    azimuth=lambda p: p.dir.azimuth,
    zenith=lambda p: p.dir.zenith,
)


def load_params(
    frame, labels=DEFAULT_LABELS,
):
    """extract truth parameters from an i3frame"""
    try:
        mc_primary = frame["MCInIcePrimary"]
        mc_tree = frame["I3MCTree"]
    except KeyError:
        # no MC info
        return None

    casc_E, track_E = _calc_energies(mc_primary, mc_tree)
    p_getters = _param_getters.copy()
    p_getters["cascade_energy"] = lambda p: casc_E
    p_getters["track_energy"] = lambda p: track_E

    return [p_getters[label](mc_primary) for label in labels]


def load_total_charge(pulses):
    """extract total charge parameters from a I3RecoPulseSeriesMap"""
    oms = set()
    total_charge = 0
    for omkey, om_pulses in pulses:
        oms.add(omkey)
        total_charge += sum(p.charge for p in om_pulses)

    return [total_charge, len(oms)]


def load_hits(pulses, geo):
    """extract hit parameters from a I3RecoPulseSeriesMap"""
    n_pulses = sum(len(pulse_vec) for pulse_vec in pulses.values())

    hits = np.empty((n_pulses, 10))
    hits_view = hits
    for omkey, om_pulses in pulses.items():
        om_idx = omkey.om - 1
        pmt_idx = omkey.pmt
        string_idx = omkey.string - 1
        flat_idx = string_idx * 60 + om_idx

        n_dom_pulses = len(om_pulses)
        hits_view[:n_dom_pulses, :3] = geo[string_idx, om_idx]
        for i, pulse in enumerate(om_pulses):
            hits_view[i, 3] = pulse.time
            hits_view[i, 4] = pulse.charge
            hits_view[i, 5] = pulse.flags & 1
            hits_view[i, 6] = (pulse.flags & 2) >> 1
            hits_view[i, 7:9] = [0, np.pi]  # handle Gen 1 DOMs for now
            hits_view[i, 9] = flat_idx

        hits_view = hits_view[n_dom_pulses:]

    return hits


def load_event(frame, geo, reco_pulse_series_name):
    """extract an event from an i3frame"""
    pulses = frame[reco_pulse_series_name]
    try:
        pulses = pulses.apply(frame)
    except AttributeError:
        # pulses is not a MapMask, no need to call apply
        pass

    hits = load_hits(pulses, geo)
    total_charge = load_total_charge(pulses)
    params = load_params(frame)

    return dict(hit_data=[hits], evt_data=[total_charge], params=params)

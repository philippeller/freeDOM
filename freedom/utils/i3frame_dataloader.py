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

def _is_tau(particle, tau_id=15):
    return abs(particle.pdg_encoding) == tau_id


def _calc_energies(mc_primary, mc_tree):
    """returns [cascade_energy, track_energy]"""
    nu_E = mc_primary.energy

    track_E = max((p.energy for p in mc_tree if _is_muon(p)), default=0)
    tau_E = max((p.energy for p in mc_tree if _is_tau(p)), default=0)
    # skip mc_tree[0] to exclude the primary from the calculation of invis_E
    invis_E = sum(p.energy for p in mc_tree[1:] if _is_invisible(p))

    return [nu_E - invis_E - track_E , track_E] #- 0.5*tau_E


_param_getters = dict(
    x=lambda p: p.pos.x,
    y=lambda p: p.pos.y,
    z=lambda p: p.pos.z,
    time=lambda p: p.time,
    azimuth=lambda p: p.dir.azimuth,
    zenith=lambda p: p.dir.zenith,
)


def load_params(
    frame,
    labels=DEFAULT_LABELS,
):
    """extract truth parameters from an i3frame"""
    try:
        mc_tree = frame["I3MCTree"]
        mc_primary = mc_tree[0]
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
        if len(om_pulses) > 0:
            oms.add(omkey)
            total_charge += sum(p.charge for p in om_pulses)

    return [total_charge, len(oms)]


def load_hits(pulses, geo, pmt_directions):
    """extract hit parameters from a I3RecoPulseSeriesMap"""
    n_pulses = sum(len(pulse_vec) for pulse_vec in pulses.values())

    hits = np.empty((n_pulses, 10))
    hits_view = hits
    for omkey, om_pulses in pulses.items():
        om_idx = omkey.om - 1
        pmt_idx = omkey.pmt
        string_idx = omkey.string - 1
        flat_idx = string_idx * 2712 + om_idx * 24 + pmt_idx

        n_dom_pulses = len(om_pulses)
        hits_view[:n_dom_pulses, :3] = geo[string_idx % 86, om_idx]
        for i, pulse in enumerate(om_pulses):
            hits_view[i, 3] = pulse.time
            hits_view[i, 4] = pulse.charge
            hits_view[i, 5] = pulse.flags & 1
            hits_view[i, 6] = (pulse.flags & 2) >> 1
            hits_view[i, 7:9] = pmt_directions[pmt_idx]
            hits_view[i, 9] = flat_idx

        hits_view = hits_view[n_dom_pulses:]

    return hits


def load_reco_series(frame, geo, series_name, ug_geo=None, mdom_directions=None):
    """load hits and total charge from a single pulse series

    handles OM-type-dependent behavior
    """
    pulses = frame[series_name]
    try:
        pulses = pulses.apply(frame)
    except AttributeError:
        # pulses is not a MapMask, no need to call apply
        pass

    if "mDOM" in series_name:
        om_geo = ug_geo
        pmt_directions = mdom_directions
    elif "DEgg" in series_name:
        om_geo = ug_geo
        pmt_directions = [[0, np.pi], [np.pi, np.pi]]
    elif "PDOM" in series_name:
        om_geo = ug_geo
        pmt_directions = [[0, np.pi]]
    else:
        om_geo = geo
        pmt_directions = [[0, np.pi]]

    if om_geo is None or pmt_directions is None:
        raise ValueError(
            "The `ug_geo` and `mdom_directions` arguments are required when running Upgrade reco!"
        )

    return dict(
        hits=load_hits(pulses, om_geo, pmt_directions),
        total_charge=load_total_charge(pulses),
    )


def load_event(frame, geo, reco_pulse_series_names, ug_geo=None, mdom_directions=None):
    """extract an event from an i3frame"""
    if isinstance(reco_pulse_series_names, str):
        reco_pulse_series_names = [reco_pulse_series_names]

    hits = []
    total_charge = []
    for series in reco_pulse_series_names:
        series_data = load_reco_series(frame, geo, series, ug_geo, mdom_directions)
        hits.append(series_data["hits"])
        total_charge.append(series_data["total_charge"])

    params = load_params(frame)

    return dict(hit_data=hits, evt_data=total_charge, params=params)

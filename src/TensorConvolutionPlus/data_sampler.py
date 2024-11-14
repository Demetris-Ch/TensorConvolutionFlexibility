#!/usr/bin/env python
import itertools
import numpy as np
import time
from utils import kumaraswamymontecarlo
import pandas as pd


__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def profile_creation(no_samples, net, distribution, keep_mp, services='DG Only', flexible_loads=[], flexible_dg=[],
                     non_lin_dgs=[]):
    """ Creation of {no_samples} of new [P, Q] for each FSP. These new [P,Q] are
    a flexibility activation [ΔP, ΔQ] applied on the initial output [P_0, Q_0] of each FSP.
    Based on the services (i.e. DG, Load, or Both), the more specific functions for sample creation are called.

    :param no_samples: Amount of shifts that the algorithm will create and run.
    :type no_samples: int

    :param net: PandaPower network in which the simulations are performed.
    :type net: pandapower.network

    :param distribution: Type of distribution by which the new [P,Q] samples are obtained.
    :type distribution: str

    :param keep_mp: Boolean whether the DG FSP shifts only concern the output power factor (keeping maximum output
                    power), or can also reduce the DG FSP power output/consumption.
    :type keep_mp: bool

    :param services: If FSPs are DG, Loads or both.
    :type services: str

    :param flexible_loads: Which Loads are considered FSPs.
    :type flexible_loads: list[int]

    :param flexible_dg: Which DG are considered FSPs.
    :type flexible_dg: list[int]

    :param non_lin_dgs: Which FSP DGs are considered non-linear.
    :type non_lin_dgs: list[int]

    :return: The new active and reactive power for each FSP, for all power flow simulations, the duration in seconds
             for the sample creations.
    :rtype: list, float
    """
    rng = np.random.RandomState(21)
    if services == 'DG only':
        profiles, dur_samples = create_samples(no_samples, net.sgen.iloc[flexible_dg, :], distribution=distribution,
                                               keep_mp=keep_mp, rng=rng, non_lin_fsps=non_lin_dgs)
    elif services == 'All':
        if len(flexible_loads) == 0:
            pq_profiles, dur_samples = create_samples(no_samples, net.sgen.iloc[flexible_dg, :],
                                                      distribution=distribution, keep_mp=keep_mp, rng=rng)
            pq_load_profiles, dur_load_samples = create_load_samples(no_samples, net, distribution='No_change',
                                                                     flex_loads=[], rng=rng)
            profiles = np.concatenate((pq_profiles, pq_load_profiles), axis=1).tolist()
            dur_samples += dur_load_samples
        elif len(flexible_dg) == 0:
            pq_profiles, dur_samples = create_samples(no_samples, net.sgen,
                                                      distribution='No_change',
                                                      keep_mp=keep_mp, rng=rng)
            pq_load_profiles, dur_load_samples = create_load_samples(no_samples, net, distribution=distribution,
                                                                     flex_loads=flexible_loads, rng=rng)
            profiles = np.concatenate((pq_profiles, pq_load_profiles), axis=1).tolist()
            dur_samples += dur_load_samples
        else:
            pq_profiles, dur_samples = create_samples(int(no_samples), net.sgen.iloc[flexible_dg, :],
                                                      distribution=distribution, keep_mp=keep_mp, rng=rng,
                                                      non_lin_fsps=non_lin_dgs)
            pq_load_profiles, dur_load_samples = create_load_samples(int(no_samples), net, distribution=distribution,
                                                                     flex_loads=flexible_loads, rng=rng)
            profiles = np.concatenate((pq_profiles, pq_load_profiles), axis=1).tolist()
            dur_samples += dur_load_samples
    elif services == 'Load only':
        pq_profiles, dur_samples = create_samples(no_samples, net.sgen.iloc[flexible_dg, :], distribution='No_change',
                                                  keep_mp=keep_mp, rng=rng)
        pq_load_profiles, dur_load_samples = create_load_samples(no_samples, net, distribution=distribution,
                                                                 flex_loads=flexible_loads, rng=rng)
        profiles = np.concatenate((pq_profiles, pq_load_profiles), axis=1).tolist()
        dur_samples += dur_load_samples
    else:
        assert False, 'Error: Choose FSPs from {All, Load only, DG only}'

    return profiles, dur_samples


def profile_creation_bf(dp, dq, net, services='DG Only', flexible_loads=[], flexible_dgs=[], non_linear_dgs=[]):
    """ Creation of {no_samples} of new [P, Q] for each FSP for the Brute Force case. These new [P,Q] are
    a flexibility activation [ΔP, ΔQ] applied on the initial output [P_0, Q_0] of each FSP.
    Based on the services (i.e. DG, Load, or Both), the more specific functions for sample creation are called.

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :param net: PandaPower network in which the simulations are performed.
    :type net: pandapower.network

    :param services: If FSPs are DG, Loads or both.
    :type services: str

    :param flexible_loads: Which Loads are considered FSPs.
    :type flexible_loads: list[int]

    :param flexible_dgs: Which DG are considered FSPs.
    :type flexible_dgs: list[int]

    :param non_linear_dgs: Which FSP DGs are considered non-linear.
    :type non_linear_dgs: list[int]

    :return: The new active and reactive power for each FSP, for all power flow simulations, the active and reactive
             power for each small FSP, the duration in seconds for the sample creations, the names of non-linear DGs,
             the number of samples.
    :rtype: list, list, float, list, int
    """
    fsps_dg, key_order_dg = get_fsps_with_idx(net.sgen, flexible_dgs)
    fsps_load, key_order_load = get_fsps_with_idx(net.load, flexible_loads)
    non_linear_dg_names = get_non_linear_comp(net.sgen, non_linear_dgs)
    if services == 'DG only':
        fsps = fsps_dg
    elif services == 'All':
        fsps = pd.concat([fsps_dg, fsps_load])
    elif services == 'Load only':
        fsps = fsps_load
    else:
        assert False, 'Error: Choose FSPs from {All, Load only, DG only}'
    profiles, small_fsp_prof, dur_samples = create_all_fsp_samples(dp, dq, fsps, non_linear_dg_names)
    samps = [len(profiles[key]) for key in profiles]
    return combine_profiles(profiles, key_order_dg, key_order_load), small_fsp_prof, dur_samples, non_linear_dg_names,\
        np.prod(samps)


def combine_profiles(profiles, key_order_dg, key_order_load):
    """ Combine profiles from DGs and Loads.

    :param profiles: profiles generated for all (large) DGs.
    :type profiles: list

    :param key_order_dg: order of DG names in profile.
    :type key_order_dg: list

    :param key_order_load: order of load names in profile.
    :type key_order_load: list

    :return: Combined profiles from all FSPs.
    :rtype: list
    """
    list_for_product = [profiles[key] for key in key_order_dg+key_order_load]
    return itertools.product(*list_for_product)


def conv_profile_creation(dp, dq, net, services='DG Only', flexible_loads=[], flexible_dgs=[], non_linear_dgs=[]):
    """ Create profiles for convolution simulations.

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :param net: PandaPower network in which the simulations are performed.
    :type net: pandapower.network

    :param services: If FSPs are DG, Loads or both.
    :type services: str

    :param flexible_loads: Which Loads are considered FSPs.
    :type flexible_loads: list[int]

    :param flexible_dgs: Which DG are considered FSPs.
    :type flexible_dgs: list[int]

    :param non_linear_dgs: Which FSP DGs are considered non-linear.
    :type non_linear_dgs: list[int]

    :return: The new active and reactive power for each FSP, for all power flow simulations, the active and reactive
             power for each small FSP, the duration in seconds for the sample creations, the names of non-linear DGs.
    :rtype: list, list, float, list
    """
    fsps_dg = get_fsps(net.sgen, flexible_dgs)
    fsps_load = get_fsps(net.load, flexible_loads)
    non_linear_dg_names = get_non_linear_comp(net.sgen, non_linear_dgs)
    if services == 'DG only':
        fsps = fsps_dg
    elif services == 'All':
        fsps = pd.concat([fsps_dg, fsps_load])
    elif services == 'Load only':
        fsps = fsps_load
    else:
        assert False, 'Error: Choose FSPs from {All, Load only, DG only}'
    profiles, small_fsp_prof, dur_samples = create_all_fsp_samples(dp, dq, fsps, non_linear_dg_names)
    return profiles, small_fsp_prof, dur_samples, non_linear_dg_names


def conv_profile_creation_sq(dp, dq, net, services='DG Only', flexible_loads=[], flexible_dgs=[], non_linear_dgs=[]):
    """ Create square shaped profiles for convolution simulations.

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :param net: PandaPower network in which the simulations are performed.
    :type net: pandapower.network

    :param services: If FSPs are DG, Loads or both.
    :type services: str

    :param flexible_loads: Which Loads are considered FSPs.
    :type flexible_loads: list[int]

    :param flexible_dgs: Which DG are considered FSPs.
    :type flexible_dgs: list[int]

    :param non_linear_dgs: Which FSP DGs are considered non-linear.
    :type non_linear_dgs: list[int]

    :return: The new active and reactive power for each FSP, for all power flow simulations, the active and reactive
             power for each small FSP, the duration in seconds for the sample creations, the names of non-linear DGs.
    :rtype: list, list, float, list
    """
    fsps_dg = get_fsps(net.sgen, flexible_dgs)
    fsps_load = get_fsps(net.load, flexible_loads)
    non_linear_dg_names = get_non_linear_comp(net.sgen, non_linear_dgs)
    if services == 'DG only':
        fsps = fsps_dg
    elif services == 'All':
        fsps = pd.concat([fsps_dg, fsps_load])
    elif services == 'Load only':
        fsps = fsps_load
    else:
        assert False, 'Error: Choose FSPs from {All, Load only, DG only}'
    profiles, small_fsp_prof, dur_samples = create_all_fsp_samples_sq(dp, dq, fsps, non_linear_dg_names)
    return profiles, small_fsp_prof, dur_samples, non_linear_dg_names


def get_fsps(comps, comp_no_list):
    """ Get important information for FSPs.

    :param comps: network components.
    :type comps: pandas.dataframe (pandapower)

    :param comp_no_list: list of FSPs numbers in the network.
    :type comp_no_list: list[int]

    :return: relevant FSP information.
    :rtype: pandas.dataframe
    """
    fsps = []
    for fsp in comps.iterrows():
        if fsp[0] in comp_no_list:
            fsps.append([fsp[1]['name'], fsp[1]['p_mw'], fsp[1]['q_mvar'], fsp[1]['sn_mva']])
    fsps = pd.DataFrame(fsps, columns=['name', 'p_mw', 'q_mvar', 'sn_mva'])
    return fsps


def get_oltcs(comps, comp_no_list):
    """ Get important information for FSPs.

    :param comps: network components.
    :type comps: pandas.dataframe (pandapower)

    :param comp_no_list: list of FSPs numbers in the network.
    :type comp_no_list: list[int]

    :return: relevant FSP information.
    :rtype: pandas.dataframe
    """
    poses = {}
    for fsp in comps.iterrows():
        if fsp[0] in comp_no_list:
            #poses[fsp[1]['name']] = [fsp[1]['tap_pos']] + np.arange(fsp[1]['tap_min']+1, fsp[1]['tap_max']-1+0.5).tolist()
            poses[fsp[1]['name']] = [fsp[1]['tap_pos'], -2, -1, 0, 1, 2]
    return poses


def get_fsps_with_idx(comps, comp_no_list):
    """Get important information for FSPs.

    :param comps: network components.
    :type comps: pandas.dataframe (pandapower)

    :param comp_no_list: list of FSPs numbers in the network.
    :type comp_no_list: list[int]

    :return: relevant FSP information, names of ordered FSPs.
    :rtype: pandas.dataframe, list
    """
    fsps = []
    fsp_order = []
    for fsp in comps.iterrows():
        if fsp[0] in comp_no_list:
            fsps.append([fsp[1]['name'], fsp[1]['p_mw'], fsp[1]['q_mvar'], fsp[1]['sn_mva'], fsp[0]])
            fsp_order.append(fsp[1]['name'])
    fsps = pd.DataFrame(fsps, columns=['name', 'p_mw', 'q_mvar', 'sn_mva', 'idx'])
    return fsps, fsp_order


def get_non_linear_comp(comps, non_linear_no):
    """ Get names of not linear FSPs.

    :param comps: network components.
    :type comps: pandas.dataframe (pandapower)

    :param non_linear_no: number of not linear FSPs.
    :type non_linear_no: list[int]

    :return: names of not linear FSPs.
    :rtype: list
    """
    non_lins = []
    for fsp in comps.iterrows():
        if fsp[0] in non_linear_no:
            non_lins.append(fsp[1]['name'])
    return non_lins


def create_load_samples(no_samples, net, distribution, flex_loads, rng):
    """ Creation of {no_samples} of new [P, Q] for each load FSP. These new [P,Q] are
    a flexibility activation [ΔP, ΔQ] applied on the initial output [P_0, Q_0] of each load FSP.

    :param no_samples: Amount of shifts that the function will create.
    :type no_samples: int

    :param net: PandaPower network in which the simulations are performed.
    :type net: pandapower.network

    :param distribution: Type of distribution by which the new [P,Q] samples are obtained.
    :type distribution: str

    :param flex_loads: Which Loads are considered FSPs.
    :type flex_loads: list[int]

    :param rng: Function by which the random numbers are generated.
    :type rng: np.random

    :return: The new active and reactive power for each load FSP, for the {no_samples}, the duration in seconds for the
             sample creations [float].
    :rtype: list, float
    """
    if len(flex_loads) == 0:
        flex_load_len = len(net.load)
    else:
        flex_load_len = len(flex_loads)
    t_start_create_mc_samples = time.time()
    random_p = sample_from_rng(distribution, no_samples*flex_load_len, 2, rng)
    pq_profiles = sample_new_load_point(net.load, random_p, no_samples, flex_loads)
    t_stop_create_mc_samples = time.time()
    return pq_profiles, t_stop_create_mc_samples - t_start_create_mc_samples


def create_samples(no_samples, net_sgen, distribution, keep_mp, rng, non_lin_fsps = []):
    """ Creation of {no_samples} of new [P, Q] for each DG FSP. These new [P,Q] are
    a flexibility activation [ΔP, ΔQ] applied on the initial output [P_0, Q_0] of each DG FSP.

    :param no_samples: Amount of shifts that the function will create, [int].
    :type no_samples: int

    :param net_sgen: PandaPower network sgen.
    :type net_sgen: pandas.dataframe (pandapower)

    :param distribution: Type of distribution by which the new [P,Q] samples are obtained.
    :type distribution: str

    :param keep_mp: Boolean whether the DG FSP shifts only concern the output power factor
                    (keeping maximum output power), or can also reduce the DG FSP power output/consumption.
    :type keep_mp: bool

    :param rng: Function by which the random numbers are generated.
    :type rng: np.random

    :param non_lin_fsps: FSPs that only offer 2 setpoint options (full or no curtailment).
    :type non_lin_fsps: list[int]

    :return: The new active and reactive power for each DG FSP, for the {no_samples}, the duration in seconds for the
             sample creations [float].
    :rtype: list, float
    """
    t_start_create_mc_samples = time.time()
    if keep_mp:
        random_p = sample_from_rng(distribution, no_samples * (len(net_sgen) - len(non_lin_fsps)), 1, rng)
        pq_profiles = sample_new_point(net_sgen, random_p, no_samples, non_lin_fsps)
    else:
        random_p = sample_from_rng(distribution, no_samples * (len(net_sgen) - len(non_lin_fsps)), 2, rng)
        pq_profiles = sample_new_non_mp_point(net_sgen, random_p, no_samples, non_lin_fsps)
    t_stop_create_mc_samples = time.time()
    return pq_profiles, t_stop_create_mc_samples - t_start_create_mc_samples


def create_all_fsp_samples(dp, dq, fsps, non_linear_dg_names):
    """ Create all FSP samples for the Convolution based simulations.

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :param fsps: dataframe of relevant data for the fsps.
    :type fsps: pandas.dataframe

    :param non_linear_dg_names: names of not linear DGs.
    :type non_linear_dg_names: list[str]

    :return: profiles for FSP, profiles for small FSP, duration for sample generation.
    :rtype: list, list, float
    """
    t_start_create_mc_samples = time.time()
    pq_fsp = {}
    small_fsp_pq = {}
    for fsp in fsps.iterrows():
        if fsp[1]['name'] not in non_linear_dg_names:
            pq_fsp[fsp[1]['name']] = sample_fsp_thorough_points(fsp[1], dp, dq)
            if len(pq_fsp[fsp[1]['name']]) <= 1:
                tmp = sample_fsp_thorough_points(fsp[1], dp/10, dq/10)
                if len(tmp) > 1:
                    small_fsp_pq[fsp[1]['name']] = tmp
        else:
            pq_fsp[fsp[1]['name']] = sample_non_lin_fsp(fsp[1], dp, dq)
    t_stop_create_mc_samples = time.time()
    return pq_fsp, small_fsp_pq, t_stop_create_mc_samples - t_start_create_mc_samples


def create_all_fsp_samples_sq(dp, dq, fsps, non_linear_dg_names):
    """ Create all FSP samples for the Convolution based simulations.

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :param fsps: dataframe of relevant data for the fsps.
    :type fsps: pandas.dataframe

    :param non_linear_dg_names: names of not linear DGs.
    :type non_linear_dg_names: list[str]

    :return: profiles for FSP, profiles for small FSP, duration for sample generation.
    :rtype: list, list, float
    """
    t_start_create_mc_samples = time.time()
    pq_fsp = {}
    small_fsp_pq = {}
    for fsp in fsps.iterrows():
        if fsp[1]['name'] not in non_linear_dg_names:
            pq_fsp[fsp[1]['name']] = sample_fsp_thorough_points_sq(fsp[1], dp, dq)
            if len(pq_fsp[fsp[1]['name']]) <= 1:
                tmp = sample_fsp_thorough_points_sq(fsp[1], dp/10, dq/10)
                if len(tmp) > 1:
                    small_fsp_pq[fsp[1]['name']] = tmp
        else:
            pq_fsp[fsp[1]['name']] = sample_non_lin_fsp(fsp[1], dp, dq)
    t_stop_create_mc_samples = time.time()
    return pq_fsp, small_fsp_pq, t_stop_create_mc_samples - t_start_create_mc_samples


def sample_from_rng(distribution, no_samples_dg, dof, rng):
    """ Based on the type of distribution, this function generates {no_samples_dg} for the flexibility activations which
    do not yet concern for the limits of each FSP and will later be applied on each FSP.

    :param distribution: Type of distribution used for the data generation.
    :type distribution: str

    :param no_samples_dg: Number of samples to be generated.
    :type no_samples_dg: int

    :param dof: Generated based on the 'keep_mp' variable. If dof=1, the random number generated for the
           active power P also defines the shift for the reactive power Q to keep S constant.
    :type dof: int (1 or 2)

    :param rng: RNG function to be used to generate the data, is defined outside of the function to return the
           same results at each simulation (through a seed) but to avoid returning the same numbers if it is called
           multiple times in one simulation.
    :type rng: numpy.random

    :return: array of generated random values to be used for the FSP activations.
    :rtype: np.array
    """
    if distribution == 'Normal':
        random_p = rng.normal(0.5, 1, [no_samples_dg, dof])
    elif distribution == 'Kumaraswamy':
        if dof == 1:
            random_p = kumaraswamymontecarlo(2, 2, 0, np.zeros(1), np.ones(1), no_samples_dg, rng)
        elif dof == 2:
            ranp = kumaraswamymontecarlo(2, 2, 0, np.zeros(1), np.ones(1), no_samples_dg, rng)[0]
            ranq = kumaraswamymontecarlo(2, 2, 0, np.zeros(1), np.ones(1), no_samples_dg, rng)[0]
            random_p = np.array([[float(ranp[i]), float(ranq[i])] for i in range(no_samples_dg)])
    elif distribution == 'Uniform':
        random_p = rng.uniform(0, 1, [no_samples_dg, dof])
    elif distribution == 'Normal_Limits_Oriented':
        if dof == 1:
            random_p = rng.normal(1, 1, [no_samples_dg, dof])
        elif dof == 2:
            # this scenario generates:
            #   25% of the samples for high shifts in P and high shifts in Q
            #   25% of the samples for high shifts in P and low shifts in Q
            #   25% of the samples for high shifts in Q and low shifts in P
            #   25% of the samples for medium shifts in P and medium shifts in Q
            random_p1 = rng.normal(0, 1, [1*int(no_samples_dg/4), 2])
            random_p2 = np.array(np.concatenate((rng.normal(0, 1, [int(no_samples_dg/4), 1]),
                                                 rng.normal(1, 1, [int(no_samples_dg/4), 1])), axis=1))
            random_p3 = np.array(np.concatenate((rng.normal(1, 1, [int(no_samples_dg/4), 1]),
                                                 rng.normal(0, 1, [int(no_samples_dg/4), 1])), axis=1))
            random_p4 = np.array(np.concatenate((rng.normal(0.5, 1, [no_samples_dg-int(no_samples_dg/4), 1]),
                                                 rng.normal(0.5, 1, [no_samples_dg-int(no_samples_dg/4), 1])), axis=1))
            random_p = np.array(np.concatenate((random_p1, random_p2, random_p3, random_p4)))
    elif distribution == 'No_change':
        random_p = np.ones((no_samples_dg, dof))
    else:
        assert False, f"Please specify a viable sampling distribution, i.e. 'Kumaraswamy', 'Uniform', " \
                      f"'Normal_Limits_Oriented' or 'No_change'. Not {distribution}"
    return random_p


def sample_new_point(sgen, random_p, no_samples, non_lin_fsps=[]):
    """ This function is called when the keep_mp is true for DG FSPs.
    Based on the random values generated for active power shifts,
    it applies the shifts on P, and applies shifts is Q which will keep the S of the DG FSP same as the initial.

    :param sgen: DG FSP.
    :type sgen: pandas.dataframe

    :param random_p: Loop values for the P shift.
    :type random_p: numpy array

    :param no_samples: Number of samples for the applied shifts.
    :type no_samples: int

    :param non_lin_fsps: Number of not linear FSPs.
    :type non_lin_fsps: list

    :return: {no_samples} of new P and Q values for the DG FSPs.
    :rtype: list
    """
    pq_profiles = []
    for j in range(0, no_samples):
        sample = []
        m = 0
        for n, i in enumerate(list(sgen.index)):
            if i not in non_lin_fsps:
                p_perc = random_p[(len(sgen)-len(non_lin_fsps))*j + m][0]
                p_new = sgen['sn_mva'][i]*p_perc
                # s^2 = p^2 + q^2 -> q^2 = s^2-p^2
                # iteratively make q positive or negative
                q_new = (-1)**j * np.sqrt(sgen['sn_mva'][i]**2 - p_new**2)
                m += 1
            else:
                if j % 2 == 0:
                    p_new = 0
                    q_new = 0
                else:
                    p_new = sgen['p_mw'][i]
                    q_new = sgen['q_mvar'][i]
            sample.append([p_new, q_new])
        pq_profiles.append(sample)
    return pq_profiles


def sample_new_non_mp_point(sgen, random_p, no_samples, non_lin_fsps=[]):
    """ This function is called when keep_mp is false for DG FSPs.
    Based on the random values generated for active and reactive power shifts, it checks that:
    (1) The active power shift percentages are between [0%, 100%], to avoid negative or increased generation values.
    (2) The reactive power shifts will not cause abs(S new) > abs(S initial).

    :param sgen: DG FSP.
    :type sgen: pandas.dataframe

    :param random_p: Loop values for the P shift.
    :type random_p: numpy array

    :param no_samples: Number of samples for the applied shifts.
    :type no_samples: int

    :param non_lin_fsps: Number of not linear FSPs.
    :type non_lin_fsps: list

    :return: no_samples of new P and Q values for the DG FSPs.
    :rtype: list
    """
    pq_profiles = []
    for j in range(0, no_samples):
        sample = []
        m = 0
        for n, i in enumerate(list(sgen.index)):
            if i not in non_lin_fsps:
                p_perc = random_p[(len(sgen)-len(non_lin_fsps))*j + m][0]
                if p_perc >= 1:
                    p_new = sgen['sn_mva'][i]
                elif p_perc <= 0:
                    p_new = 0
                else:
                    p_new = sgen['sn_mva'][i]*p_perc
                # s^2 = p^2 + q^2 -> q^2 = s^2-p^2
                # iteratively make q positive or negative
                q_max = np.sqrt(sgen['sn_mva'][i]**2 - p_new**2)
                twh = 0
                while sgen['sn_mva'][i]**2 < p_new**2+q_max**2:
                    twh += 1
                    if q_max > 0:
                        q_max += -sgen['sn_mva'][i]/100
                    else:
                        q_max += sgen['sn_mva'][i]/100
                    if twh >= 10000:
                        assert False, "Error!"
                if q_max >= abs(sgen['sn_mva'][i]*(random_p[(len(sgen)-len(non_lin_fsps))*j + m][1])):
                    q_new = (-1)**j*sgen['sn_mva'][i]*random_p[(len(sgen)-len(non_lin_fsps))*j + m][1]
                else:
                    q_new = (-1)**j*q_max
                m += 1
            else:
                if j % 2 == 1:
                    p_new = 0
                    q_new = 0
                else:
                    p_new = sgen['p_mw'][i]
                    q_new = sgen['q_mvar'][i]
            twh = 0
            while sgen['sn_mva'][i] ** 2 < p_new ** 2 + q_new ** 2:
                twh += 1
                if q_new > 0:
                    q_new += -sgen['sn_mva'][i] / 100
                else:
                    q_new += sgen['sn_mva'][i] / 100
                if twh >= 10000:
                    assert False, "Error!"
            sample.append([p_new, q_new])
        pq_profiles.append(sample)
    return pq_profiles


def sample_fsp_thorough_points(fsp, dp, dq):
    """ Sample FSP shifts for all spectrum in dp, dq resolution.

    :param fsp: dataframe of relevant data for the fsps.
    :type fsp: pandas.dataframe

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :return: p,q values as setpoints for the FSP shifts.
    :rtype: list
    """
    max_s = fsp['sn_mva']
    if np.isnan(max_s):
        max_s = np.sqrt(fsp['p_mw']**2+fsp['q_mvar']**2)
    pq_profiles = [[fsp['p_mw'], fsp['q_mvar']]]
    no_samples = int(max_s/dp)
    init_pq = [fsp['p_mw'], fsp['q_mvar']]
    if max_s <= dp:
        return pq_profiles
    else:
        seen_0 = False
        for j in range(0, no_samples+1):
            p_new = max(0., init_pq[0] - dp*j)
            q_max = np.sqrt(max_s ** 2 - p_new ** 2)
            q = np.inf
            if q_max == 0:
                pq_profiles = append_margin_profiles(pq_profiles, max_s, p_new, dp, dq, no_samples)
            for q in np.arange(-q_max, q_max, dq):
                q_new = q
                pq_profiles.append([p_new, q_new])
            if q < q_max:
                pq_profiles.append([p_new, q_max])
            if p_new == 0.:
                seen_0 = True
        if not seen_0:
            for q in np.arange(-max_s, max_s, dq):
                q_new = q
                pq_profiles.append([0, q_new])
            if q < q_max:
                pq_profiles.append([0, max_s])
    return pq_profiles


def sample_fsp_thorough_points_sq(fsp, dp, dq):
    """ Sample FSP shifts for all spectrum in dp, dq resolution.

    :param fsp: dataframe of relevant data for the fsps.
    :type fsp: pandas.dataframe

    :param dp: Resolution in active power shifts that the algorithm will create and run.
    :type dp: float

    :param dq: Resolution in reactive power shifts that the algorithm will create and run.
    :type dq: float

    :return: p,q values as setpoints for the FSP shifts.
    :rtype: list
    """
    max_s = fsp['sn_mva']
    if np.isnan(max_s):
        max_s = np.sqrt(fsp['p_mw']**2+fsp['q_mvar']**2)
    pq_profiles_in = [[fsp['p_mw'], fsp['q_mvar']]]
    if max_s <= dp:
        return pq_profiles_in
    else:
        p_s = np.arange(0, max_s+dp, dp)
        q_s = np.arange(-max_s, max_s+dq, dq)
        pq_profiles = list(itertools.product(p_s, q_s))

        # If you want C as a list of lists instead of a list of tuples
        pq_profiles = [list(item) for item in pq_profiles]
        pq_profiles.insert(0, pq_profiles_in[0])

    return pq_profiles


def check_margins(max_s, p, dp, dq, no_samples):
    """ Check if new vertical shifts should be given due to the descritization.

    :param max_s: maximum apparent power of component.
    :type max_s: float

    :param p: active power set-point.
    :type p: float

    :param dp: resolution in active power shifts.
    :type dp: float

    :param dq: resolution in reactive power shifts.
    :type dq: float

    :param no_samples: number of samples.
    :type no_samples: int

    :return: how many additional vertical shift samples are needed.
    :rtype: int
    """
    for i in range(0, no_samples):
        if abs(max_s**2 - p**2 - (i*dq)**2) > dq**2 + dp**2:
            return i-1
    return 0


def append_margin_profiles(pq_profiles, max_s, p, dp, dq, no_samples):
    """ Add profiles for vertical shifts needed due to discretization.

    :param pq_profiles: p,q values for the shifts.
    :type pq_profiles: list

    :param max_s: maximum apparent power of component.
    :type max_s: float

    :param p: active power set-point.
    :type p: float

    :param dp: resolution in active power shifts.
    :type dp: float

    :param dq: resolution in reactive power shifts.
    :type dq: float

    :param no_samples: number of samples.
    :type no_samples: int

    :return: new p,q profiles.
    :rtype: list
    """
    i = check_margins(max_s, p, dp, dq, no_samples)
    pq_profiles.append([p, 0])
    for j in np.arange(-i, i+0.5, 1):
        if j != 0:
            pq_profiles.append([p, j*dq])
    return pq_profiles


def sample_non_lin_fsp(fsp, dp, dq):
    """ Get the shifts for the non linear FSPs (this can be modified to also offer different scenarios for non-linear FSPs).

    :param fsp: relevant information for not linear FSPs.
    :type fsp:  pandas.dataframe

    :param dp: resolution in active power shifts.
    :type dp: float

    :param dq: resolution in reactive power shifts.
    :type dq: float

    :return: list of new setpoints (profiles) for not linear FSPs.
    :rtype: list
    """
    return [[fsp['p_mw'], fsp['q_mvar']], [fsp['p_mw'] % dp, fsp['q_mvar'] % dq]]


def sample_new_load_point(loads, random_p, no_samples, flex_loads=[]):
    """ This function is called to apply the generated random shifts on Load FSPs.
    Based on the random values generated for active and reactive power shifts, it checks that:
    (1) The active power shift percentages are between [0%, 100%], to avoid negative or increased consumption values
    (2) The reactive power shifts will not cause abs(S new) > abs(S initial).

    :param loads: Load FSP.
    :type loads:  pandas.dataframe

    :param random_p: Loop values for the P Q shifts.
    :type random_p:  numpy array

    :param no_samples: Number of samples for the applied shifts.
    :type no_samples:  int

    :param flex_loads: Which loads of flexible out of all the network loads. If empty, it is assumed that all loads are
                       flexible (since this function is called when at least 1 load is flexible.
    :type flex_loads:  list

    :return: no_samples of new P and Q values for the Load FSPs.
    :rtype: list
    """
    pq_profiles = []
    load_change = 1
    if len(flex_loads) == 0:
        for j in range(0, no_samples):
            sample = []
            for i in range(0, len(loads)):
                p_perc = random_p[len(loads)*j + i][0]
                if p_perc >= 1:
                    p_new = loads['sn_mva'][i]
                elif p_perc <= 0:
                    p_new = 0
                else:
                    p_new = loads['sn_mva'][i]*p_perc

                # s^2 = p^2 + q^2 -> q^2 = s^2-p^2
                # iteratively make q positive or negative
                p_new = load_change*p_new
                q_max = np.sqrt(load_change*load_change*(loads['sn_mva'][i]**2) - p_new**2)
                twh = 0
                while loads['sn_mva'][i]**2 < p_new**2+q_max**2:
                    twh += 1
                    if q_max > 0:
                        q_max += -loads['sn_mva'][i]/100
                    else:
                        q_max += loads['sn_mva'][i]/100
                    if twh > 10000:
                        assert False, "Error!!!"
                if q_max >= abs(loads['sn_mva'][i]*(random_p[len(loads) * j + i][1])):
                    q_new = (-1)**j*loads['sn_mva'][i]*(random_p[len(loads) * j + i][1])
                else:
                    q_new = (-1)**j*q_max
                if np.isnan(q_new):
                    assert False, f"Pnew = {p_new}, Q_max = {q_max}, Sn={loads['sn_mva'][i]}"
                sample.append([p_new, q_new])
            pq_profiles.append(sample)
    else:
        for j in range(0, no_samples):
            sample = []
            for idx, i in enumerate(flex_loads):
                p_perc = random_p[len(flex_loads) * j + idx][0]
                if p_perc >= 1:
                    p_new = loads['sn_mva'][i]
                elif p_perc <= 0:
                    p_new = 0
                else:
                    p_new = loads['sn_mva'][i]*p_perc
                # s^2 = p^2 + q^2 -> q^2 = s^2-p^2
                # iteratively make q positive or negative
                p_new = load_change * p_new
                q_max = np.sqrt(loads['sn_mva'][i]**2 - p_new**2)
                twh = 0
                while loads['sn_mva'][i]**2 < p_new**2+q_max**2:
                    twh += 1
                    if q_max > 0:
                        q_max += -loads['sn_mva'][i]/100
                    else:
                        q_max += loads['sn_mva'][i]/100
                    if twh > 10000:
                        assert False, "Error!!!"
                if q_max >= abs(loads['sn_mva'][i]*(random_p[len(flex_loads) * j + idx][1])):
                    q_new = (-1)**j*loads['sn_mva'][i]*(random_p[len(flex_loads) * j + idx][1])
                else:
                    q_new = (-1)**j*q_max
                if np.isnan(q_new):
                    assert False, f"Pnew = {p_new}, Q_max = {q_max}, Sn={loads['sn_mva'][i]}"
                sample.append([p_new, q_new])
            pq_profiles.append(sample)
    return pq_profiles


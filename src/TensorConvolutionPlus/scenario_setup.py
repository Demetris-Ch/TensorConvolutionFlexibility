#!/usr/bin/env python
import pandapower.networks as pn
import numpy as np
import pandapower as pp
from .utils import update_pqs, fix_net, check_limits_bool
import logging
logging.getLogger("pandapower").setLevel(logging.ERROR)


__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def get_network(settings):
    """Gen the network specified in settings. Currently, only "CIGRE MV" with der "pv_wind" is implemented

    :param settings: json file input data
    :type settings: object

    :return: network model
    :rtype: pandapower.network
    """
    if settings.net_name == 'CIGRE MV':
        net = pn.create_cigre_network_mv(with_der="pv_wind")
        net.ext_grid["s_sc_max_mva"] = 5e6
        net.ext_grid["s_sc_min_mva"] = 5e6
    elif settings.net_name == 'Four bus':
        net = pn.simple_four_bus_system()
    elif settings.net_name == 'MV Open Ring':
        net = pn.simple_mv_open_ring_net()
    elif settings.net_name == "MV Closed Ring":
        net = pn.simple_mv_open_ring_net()
        net.switch['closed'][6] = True
    elif settings.net_name == "MV Oberrhein":
        net = pn.mv_oberrhein(scenario='generation')
        net.load['sn_mva'] = list(net.load['p_mw'].pow(2).add(net.load['q_mvar'].pow(2)).pow(0.5))
    elif settings.net_name == "MV Oberrhein0":
        net, net_tmp = pn.mv_oberrhein(separation_by_sub=True)
        net.load['sn_mva'] = list(net.load['p_mw'].pow(2).add(net.load['q_mvar'].pow(2)).pow(0.5))
        net.load['scaling'] = [1 for i in range(len(net.load))]
        net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
    elif settings.net_name == "MV Oberrhein1":
        net_tmp, net = pn.mv_oberrhein(separation_by_sub=True)
        net.load['sn_mva'] = list(net.load['p_mw'].pow(2).add(net.load['q_mvar'].pow(2)).pow(0.5))
        net.load['scaling'] = [1 for i in range(len(net.load))]
        net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
    else:
        assert AssertionError, "Error: Known Network Names are: 'CIGRE MV', 'Four bus', 'MV Open Ring', " \
                                 "'MV Closed Ring', 'MV Oberrhein', 'MV Oberrhein0', 'MV Oberrhein1'"
    return fix_net(net)


def update_settings(settings):
    """Update settings by changing observable and unobservable bus and line information based on the loaded network

    :param settings: json file input data
    :type settings: object

    :return: updated settings object
    :rtype: object
    """
    settings.net = get_network(settings)
    settings = get_observable_lines_buses(settings)
    settings = get_unobservable_lines_buses_indices(settings)
    settings = get_fsp(settings)
    return settings


def get_fsp(settings):
    """Get indices of FSPs if any of them are specified as -1.
    This value , -1, is used to say 'I do not know the indices but all components of type x are assumed FSPs

    :param settings: json file input data
    :type settings: object

    :return: updated settings object
    :rtype: object
    """
    net = settings.net
    if settings.fsp_wt:
        if settings.fsp_wt[0] == -1:
            settings.fsp_wt = [len(net.sgen)-1]
    if settings.fsp_pv:
        if settings.fsp_pv[0] == -1:
            settings.fsp_pv = np.arange(0, len(net.sgen))
    if settings.fsp_load:
        if settings.fsp_load[0] == -1:
            settings.fsp_load = np.arange(0, len(net.load))
    if settings.fsp_dg:
        if settings.fsp_dg[0] == -1:
            settings.fsp_dg = np.arange(0, len(net.sgen))
    return settings


def get_unobservable_lines_buses_indices(settings):
    """ Gen indices of network unobservable lines (all lines not specified as observable)

    :param settings: json file input data
    :type settings: object

    :return: updated settings object
    :rtype: object
    """
    net = settings.net
    u_line_indices = []
    for i in range(0, len(net.line)):
        if i not in settings.observ_lines:
            u_line_indices.append(i)
    u_bus_indices = []
    for i in range(0, len(net.bus)):
        if i not in settings.observ_buses:
            u_bus_indices.append(i)
    settings.non_observ_lines = u_line_indices
    settings.non_observ_buses = u_bus_indices
    return settings


def get_observable_lines_buses(settings):
    """ Get indices of observable lines and buses if any of them are specified as -1.
    This value , -1, is used to say 'I do not know the indices but all components of type x are assumed observable

    :param settings: json file input data
    :type settings: object

    :return: updated settings object
    :rtype: object
    """
    net = settings.net
    if settings.observ_lines[0] == -1:
        settings.observ_lines = np.arange(0, len(net.line))
    if settings.observ_buses[0] == -1:
        settings.observ_buses = np.arange(0, len(net.bus))
    return settings


def get_operating_point(settings):
    """ Get network PCC P,Q (y)

    :param settings: json file input data
    :type settings: object

    :return: Network PCC P,Q (y)
    :rtype: list[float, float]
    """
    net = get_network(settings)
    net = update_pqs(net, scale_pv=settings.scale_pv, scale_w=settings.scale_wt)  # scale DG generation times higher
    pp.runpp(net)
    if len(net.ext_grid) > 1:
        egid = 1
    else:
        egid = 0
    return [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]


def apply_uss(net, settings):
    """Apply USS scenario shifts on the network

    :param net: Network model for scenario
    :type net: pandapower.network

    :param settings: json file input data
    :type settings: object

    :return: network updated for the USS# case, network new PCC P,Q (y)
    :rtype: pandapower.network, list[float, float]
    """
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    max_curr = settings.max_curr
    scenario = settings.scenario_type_dict['no.']
    p234 = [0.3, 1]
    p7 = [0.414, 0.045]
    p13 = [0.7275, 1.06]
    q7 = [0.276, 0.492]
    q13 = [0.3, 1]
    q14 = [0.3, 1.431]

    net.load['p_mw'][2] = net.load['p_mw'][2] - 0.43165
    net.load['p_mw'][3] = net.load['p_mw'][3] - 0.7275
    net.load['p_mw'][4] = net.load['p_mw'][4] - 0.54805
    net.load['p_mw'][14] = net.load['p_mw'][14] + 0.54805

    net.load['q_mvar'][2] = net.load['q_mvar'][2] - p234[scenario-1]
    net.load['q_mvar'][3] = net.load['q_mvar'][3] - p234[scenario-1]
    net.load['q_mvar'][4] = net.load['q_mvar'][4] - p234[scenario-1]
    net.load['p_mw'][7] = net.load['p_mw'][7] + p7[scenario-1]
    net.load['p_mw'][13] = net.load['p_mw'][13] + p13[scenario-1]
    net.load['q_mvar'][7] = net.load['q_mvar'][7] + q7[scenario-1]
    net.load['q_mvar'][13] = net.load['q_mvar'][13] + q13[scenario-1]
    net.load['q_mvar'][14] = net.load['q_mvar'][14] + q14[scenario-1]
    pp.runpp(net)
    return net, [net.res_ext_grid['p_mw'].iloc[0], net.res_ext_grid['q_mvar'].iloc[0]]


def apply_tss(net, settings):
    """Apply TSS scenario shifts on the network

    :param net: Network model for scenario
    :type net: pandapower.network

    :param settings: json file input data
    :type settings: object

    :return: network updated for the USS# case, network new PCC P,Q (y)
    :rtype: pandapower.network, list[float, float]
    """
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    max_curr = settings.max_curr
    scenario = settings.scenario_type_dict['no.']
    if scenario == 1:
        net.load['q_mvar'][11] = net.load['q_mvar'][11] - 0.067
        net.load['p_mw'][11] = net.load['p_mw'][11] - 0.031
        net.switch['closed'][2] = True
        net.switch['closed'][1] = True
        net.line['in_service'][6] = False  # 8 - 9
        net.line['in_service'][2] = False  # 3-4
    elif scenario == 2:
        net.load['q_mvar'][11] = net.load['q_mvar'][11] - 0.03
        net.switch['closed'][2] = True
        net.switch['closed'][1] = True
        net.line['in_service'][6] = False  # 8 - 9
        net.line['in_service'][4] = False  # 5-6
    elif scenario == 3:
        net.load['q_mvar'][11] = net.load['q_mvar'][11] + 1.111
        net.load['p_mw'][11] = net.load['p_mw'][11] + 2.14
        net.load['q_mvar'][7] = net.load['q_mvar'][7] - 1.115
        net.load['p_mw'][7] = net.load['p_mw'][7] - 2.13
        net.switch['closed'][2] = True
        net.switch['closed'][1] = True
        net.line['in_service'][9] = False  # 3 - 8
        net.line['in_service'][2] = False  # 3-4
        net.switch['closed'][4] = True
    pp.runpp(net)
    return net, [net.res_ext_grid['p_mw'].iloc[0], net.res_ext_grid['q_mvar'].iloc[0]]


def apply_cs(net, settings):
    """Apply CS scenario shifts on the network

    :param net: Network model for scenario
    :type net: pandapower.network

    :param settings: json file input data
    :type settings: object

    :return: network updated for the USS# case, network new PCC P,Q (y)
    :rtype: pandapower.network, list[float, float]
    """
    scenario = settings.scenario_type_dict.get('no.', 0)
    std_cap_l = settings.scenario_type_dict.get('std_cap_ld', 0)
    std_pf_l = settings.scenario_type_dict.get('std_pf_ld', 0)
    std_cap_dg = settings.scenario_type_dict.get('std_cap_dg', 0)
    std_pf_dg = settings.scenario_type_dict.get('std_pf_dg', 0)
    if scenario == 4:
        net.switch['closed'].iloc[4] = True
    elif scenario == 101:
        net.load.at[13, 'p_mw'] = net.load.iloc[13]['sn_mva']
        net.load.at[13, 'q_mvar'] = 0
    elif scenario == 102:
        net.load.at[0, 'q_mvar'] = -net.load.at[0, 'q_mvar']
        net.load.at[7, 'q_mvar'] = -net.load.at[7, 'q_mvar']
        net.load.at[8, 'q_mvar'] = -net.load.at[8, 'q_mvar']
        net.load.at[12, 'q_mvar'] = -net.load.at[12, 'q_mvar']
        net.load.at[13, 'p_mw'] = net.load.iloc[13]['sn_mva']
        net.load.at[13, 'q_mvar'] = 0
        net.load.at[15, 'q_mvar'] = -net.load.at[15, 'q_mvar']
    elif scenario == 103:
        net.switch['closed'] = [True for i in range(len(net.switch))]
        net.load['scaling'] = [1 for i in range(len(net.load))]
        net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
        rng = np.random.RandomState(103)
        new_net, rng = rand_resample(net.deepcopy(), settings.fsp_load, settings.fsp_dg, rng, std_cap_l, std_pf_l,
                                     std_cap_dg, std_pf_dg)
        i = 0
        while not check_limits_bool(new_net, settings):
            new_net, rng = rand_resample(net.deepcopy(), settings.fsp_load, settings.fsp_dg, rng, std_cap_l, std_pf_l,
                                     std_cap_dg, std_pf_dg)
            if i == 10000:
                assert False, "Cannot Find Feasible Operating Conditions"
            i += 1
        print("Updated Scenario")
        net = new_net
    elif scenario == 104:
        net.load['scaling'] = [1 for i in range(len(net.load))]
        net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
    else:
        net.switch['closed'] = [True for i in range(len(net.switch))]
        net.load['scaling'] = [1 for i in range(len(net.load))]
        net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
        if len(net.ext_grid) > 1:
            net.ext_grid['in_service'] = [False, True] + [False for i in range(len(net.ext_grid)-2)]
    if len(net.ext_grid) > 1:
        egid = 1
    else:
        egid = 0
    pp.runpp(net)
    return net, [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]


def rand_resample(net, no_change_loads, no_change_dgs, rng, std_cap_l, std_pf_l, std_cap_dg, std_pf_dg):
    """ Randomly sample Operating Condition shift for adaptability case study

    :param net: Network model for scenario
    :type net: pandapower.network

    :param no_change_loads: loads not to change (FSPs)
    :type no_change_loads: list

    :param no_change_dgs: DGS not to change (FSPs)
    :type no_change_dgs: list

    :param rng: object to sample random values from
    :type rng: numpy.random

    :param std_cap_l: standard deviation for load capacities
    :type std_cap_l: float

    :param std_pf_l: standard deviation for load power factor
    :type std_pf_l: float

    :param std_cap_dg: standard deviation for DG capacities
    :type std_cap_dg: float

    :param std_pf_dg: standard deviation for DG power factor
    :type std_pf_dg: float

    :return: network and used random function
    :rtype: pandapower.network, numpy.random
    """
    load_cap_shifts = rng.normal(1, std_cap_l, len(net.load))
    gen_cap_shifts = rng.normal(1, std_cap_dg, len(net.sgen))
    load_pf_shifts = rng.normal(1, std_pf_l, len(net.load))
    gen_pf_shifts = rng.normal(1, std_pf_dg, len(net.sgen))
    for i in range(0, len(load_cap_shifts)):
        if i not in no_change_loads:
            old_pf = net.load.iloc[i]['p_mw']/net.load.iloc[i]['sn_mva']
            if net.load.iloc[i]['q_mvar'] > 0:
                sgn = 1
            else:
                sgn = -1
            new_s = net.load.iloc[i]['sn_mva']*max(0, load_cap_shifts[i])
            new_pf = max(0, old_pf*load_pf_shifts[i])
            if new_pf > 1:
                new_pf = 2-new_pf
                sgn = -sgn
            net.load.at[i, 'p_mw'] = new_pf*new_s
            net.load.at[i, 'q_mvar'] = sgn*(new_s**2 - (new_pf*new_s)**2)**0.5
    for i in range(0, len(gen_cap_shifts)):
        if i not in no_change_dgs:
            old_pf = net.sgen.iloc[i]['p_mw']/net.sgen.iloc[i]['sn_mva']
            if net.sgen.iloc[i]['q_mvar'] > 0:
                sgn = 1
            else:
                sgn = -1
            new_s = net.sgen.iloc[i]['sn_mva']*max(0, gen_cap_shifts[i])
            new_pf = max(0, old_pf*gen_pf_shifts[i])
            if new_pf > 1:
                new_pf = 2-new_pf
                sgn = -sgn
            net.sgen.at[i, 'p_mw'] = new_pf*new_s
            net.sgen.at[i, 'q_mvar'] = sgn*(new_s**2 - (new_pf*new_s)**2)**0.5
    return net, rng


#!/usr/bin/env python
import time
from utils import update_pqs2, check_voltage_limits, check_line_current_limits, update_pqs_wl2, \
    check_trafo_current_limits
import pandapower as pp
from tqdm import tqdm
import logging
logging.getLogger("pandapower").setLevel(logging.ERROR)


__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def all_pf_simulations(settings, net, pq_profiles):
    """main function running all power flow simulations based on which FSP types will be used.

    :param settings: Information of the json file.
    :type settings: object

    :param net: Network on which the simulations will be performed.
    :type net: pandapower.network

    :param pq_profiles: P and Q values for each FSP for each iteration in the Monte Carlo simulation.
    :type pq_profiles: list

    :return: feasible P, feasible Q, infeasible P, infeasible Q, duration of simulations [s],
             FSP PQ values for the feasible y, FSP PQ values for the infeasible y.
    :rtype: list, list, list, list, float, list, list
    """
    if settings.fsps == 'DG only':
        return run_all_samples(settings, net, pq_profiles)
    elif settings.fsps == 'All' or settings.fsps == 'Load only':
        return run_all_samples_wl(settings, net, pq_profiles)
    else:
        assert False, 'Error: Choose FSPs from {All, Load only, DG only}'


def run_all_samples(settings, net, pq_profiles):
    """ Run all power flows for scenarios where only DG are used as FSP.

    :param settings: information of the json file.
    :type settings: object

    :param net: Network on which the simulations will be performed.
    :type net: pandapower.network

    :param pq_profiles: P and Q values for each FSP for each iteration in the Monte Carlo simulation.
    :type pq_profiles: list

    :return: feasible P, feasible Q, infeasible P, infeasible Q, duration of simulations [s],
             FSP PQ values for the feasible y, FSP PQ values for the infeasible y.
    :rtype: list, list, list, list, float, list, list
    """
    max_curr = settings.max_curr
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    fsp_dg = settings.fsp_dg
    x_flexible = []
    y_flexible = []
    x_non_flexible = []
    y_non_flexible = []
    t_start_run_mc_pf = time.time()
    prof_flexible = []
    prof_non_flexible = []
    if len(net.ext_grid) > 1:
        egid = 1
    else:
        egid = 0
    for profile in tqdm(pq_profiles, desc="Power flows Completed:"):
        net = update_pqs2(net, flex_dg=fsp_dg, profile=profile)
        try:
            pp.runpp(net, numba=False)
            pq_value = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
            if check_voltage_limits(net.res_bus['vm_pu'], max_volt, min_volt) and \
                    check_line_current_limits(net, max_curr) and check_trafo_current_limits(net, max_curr):
                x_flexible.append(pq_value[0])
                y_flexible.append(pq_value[1])
                prof_flexible.append(profile)
            else:
                x_non_flexible.append(pq_value[0])
                y_non_flexible.append(pq_value[1])
                prof_non_flexible.append(profile)
        except:
            print(f"Power flow did not converge for profile {profile}")
    t_stop_run_mc_pf = time.time()
    print(f"{settings.no_samples} MC Power flows needed {t_stop_run_mc_pf - t_start_run_mc_pf} seconds")
    print(f"Pf run {len(y_flexible)+len(y_non_flexible)}")
    return x_flexible, y_flexible, x_non_flexible, y_non_flexible, t_stop_run_mc_pf - t_start_run_mc_pf, \
           prof_flexible, prof_non_flexible


def run_all_samples_wl(settings, net, pq_profiles):
    """ Run all power flows for scenarios where loads are included in the FSP.

    :param settings: information of the json file.
    :type settings: object

    :param net: network on which the simulations will be performed.
    :type net: pandapower.network

    :param pq_profiles: P and Q values for each FSP for each iteration in the Monte Carlo simulation.
    :type pq_profiles: list

    :return: feasible P, feasible Q, infeasible P, infeasible Q, duration of simulations [s],
             FSP PQ values for the feasible y, FSP PQ values for the infeasible y.
    :rtype: list, list, list, list, float, list, list
    """
    fsp_dg = settings.fsp_dg
    fsp_load = settings.fsp_load
    max_curr = settings.max_curr
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    x_flexible = []
    y_flexible = []
    x_non_flexible = []
    y_non_flexible = []
    t_start_run_mc_pf = time.time()
    prof_flexible = []
    prof_non_flexible = []
    if len(net.ext_grid) > 1:
        egid = 1
    else:
        egid = 0
    for profile in tqdm(pq_profiles, desc=f"Running Power Flows:"):
        net = update_pqs_wl2(net, profile=profile, load_ind=fsp_load, dg_ind=fsp_dg).deepcopy()

        try:
            pp.runpp(net, numba=False)
            pq_value = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
            if check_voltage_limits(net.res_bus['vm_pu'], max_volt, min_volt) and \
                    check_line_current_limits(net, max_curr) and check_trafo_current_limits(net, max_curr):
                x_flexible.append(pq_value[0])
                y_flexible.append(pq_value[1])
                prof_flexible.append(profile)
            else:
                x_non_flexible.append(pq_value[0])
                y_non_flexible.append(pq_value[1])
                prof_non_flexible.append(profile)
        except:
            print(f"Power flow did not converge for profile {profile}")
    t_stop_run_mc_pf = time.time()
    return x_flexible, y_flexible, x_non_flexible, y_non_flexible, t_stop_run_mc_pf - t_start_run_mc_pf, \
           prof_flexible, prof_non_flexible


def run_uc6(settings, net, pq_profiles):
    """ Run all power flows for scenarios where loads are included in the FSP.

    :param settings: information of the json file.
    :type settings: object

    :param net: network on which the simulations will be performed.
    :type net: pandapower.network

    :param pq_profiles: P and Q values for each FSP for each iteration in the Monte Carlo simulation.
    :type pq_profiles: list

    :return: feasible P, feasible Q, infeasible P, infeasible Q, duration of simulations [s],
             FSP PQ values for the feasible y, FSP PQ values for the infeasible y.
    :rtype: list, list, list, list, float, list, list
    """
    fsp_dg = settings.fsp_dg
    fsp_load = settings.fsp_load
    max_curr = settings.max_curr
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    dp = settings.dp
    dq = settings.dq
    x_flexible = []
    y_flexible = []
    x_non_flexible = []
    y_non_flexible = []
    t_start_run_mc_pf = time.time()
    prof_flexible = []
    prof_non_flexible = []
    init_pq_fsp_load = [[net.load['p_mw'][x], net.load['q_mvar'][x]] for x in fsp_load]
    init_pq_fsp_dg = [[net.sgen['p_mw'][x], net.sgen['q_mvar'][x]] for x in fsp_dg]
    x_flex_minv = []
    x_flex_maxv = []
    x_flex_maxload = []
    x_nflex_maxload = []

    x_nflex_minv = []
    x_nflex_maxv = []

    pricesq = [40, 50, 60]
    pricesp = [40, 50, 60]


    change_per_flex_fsp = []
    change_per_nflex_fsp = []
    used_fsps_flex = []
    used_fsps_nflex = []

    print(init_pq_fsp_dg, init_pq_fsp_load, net.res_ext_grid['p_mw'].iloc[0], net.res_ext_grid['q_mvar'].iloc[0])
    v_plot = {'a': [], 'b': [], 'z': []}
    if len(net.ext_grid) > 1:
        egid = 1
    else:
        egid = 0
    for profile in tqdm(pq_profiles, desc=f"Running Power Flows:"):
        net = update_pqs_wl2(net, profile=profile, load_ind=fsp_load, dg_ind=fsp_dg).deepcopy()
        try:
            pp.runpp(net, numba=False)
            pq_value = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
            if check_voltage_limits(net.res_bus['vm_pu'], max_volt, min_volt) and \
                    check_line_current_limits(net, max_curr) and check_trafo_current_limits(net, max_curr):

                x_flex_maxv.append(net.res_bus['vm_pu'].loc[net.res_bus['vm_pu'].idxmax()])
                x_flex_minv.append(net.res_bus['vm_pu'].loc[net.res_bus['vm_pu'].idxmin()])
                x_flex_maxload.append(net.res_line['loading_percent'].abs().loc[net.res_line['loading_percent'].abs().idxmax()])

                p_cost = 0
                q_cost = 0
                used_comps = 0
                for iii, x in enumerate(fsp_load):
                    chp = net.load['p_mw'][x]-init_pq_fsp_load[iii][0]
                    chq = net.load['q_mvar'][x]-init_pq_fsp_load[iii][1]
                    if abs(chp) < dp/2:
                        chp = 0
                    if abs(chq) < dq/2:
                        chq = 0
                    p_cost += abs(chp * pricesp[iii])
                    q_cost += abs(chq * pricesq[iii])
                    if chp != 0 or chq != 0:
                        used_comps += 1
                for iii, x in enumerate(fsp_dg):
                    chp = net.sgen['p_mw'][x]-init_pq_fsp_dg[iii][0]
                    chq = net.sgen['q_mvar'][x]-init_pq_fsp_dg[iii][1]
                    if abs(chp) < dp/2:
                        chp = 0
                    if abs(chq) < dq/2:
                        chq = 0
                    p_cost += abs(float(chp*pricesp[len(fsp_load)+iii]))
                    q_cost += abs(float(chq*pricesp[len(fsp_load)+iii]))
                    if chp != 0 or chq != 0:
                        used_comps += 1
                change_per_flex_fsp.append(p_cost+q_cost)
                used_fsps_flex.append(used_comps)

                x_flexible.append(pq_value[0])
                y_flexible.append(pq_value[1])
                prof_flexible.append(profile)
                if 43.8211 < pq_value[0] < 43.8212 and 15.3094 < pq_value[1] < 15.3095 and \
                        37.9179 < p_cost+q_cost < 37.9180:
                    v_plot['a'] = profile

                elif 43.8123 < pq_value[0] < 43.8124 and 14.6274 < pq_value[1] < 14.6275 and \
                        66.4639 < p_cost+q_cost < 66.4641:
                    v_plot['b'] = profile

                elif 43.0565 < pq_value[0] < 43.0566 and 12.7533 < pq_value[1] < 12.7534 and \
                        177.4349 < p_cost+q_cost < 177.4350:
                    v_plot['z'] = profile
            else:
                x_non_flexible.append(pq_value[0])
                y_non_flexible.append(pq_value[1])

                x_nflex_maxv.append(net.res_bus['vm_pu'].loc[net.res_bus['vm_pu'].idxmax()])
                x_nflex_minv.append(net.res_bus['vm_pu'].loc[net.res_bus['vm_pu'].idxmin()])

                x_nflex_maxload.append(net.res_line['loading_percent'].abs().loc[net.res_line['loading_percent'].abs().idxmax()])

                p_cost = 0
                q_cost = 0
                used_comps = 0
                for iii, x in enumerate(fsp_load):
                    chp = net.load['p_mw'][x]-init_pq_fsp_load[iii][0]
                    chq = net.load['q_mvar'][x]-init_pq_fsp_load[iii][1]
                    if abs(chp) < dp/2:
                        chp = 0
                    if abs(chq) < dq/2:
                        chq = 0
                    p_cost += abs(chp*pricesp[iii])
                    q_cost += abs(chq*pricesq[iii])
                    if chp != 0 or chq != 0:
                        used_comps += 1
                for iii, x in enumerate(fsp_dg):
                    chp = net.sgen['p_mw'][x]-init_pq_fsp_dg[iii][0]
                    chq = net.sgen['q_mvar'][x]-init_pq_fsp_dg[iii][1]
                    if abs(chp) < dp/2:
                        chp = 0
                    if abs(chq) < dq/2:
                        chq = 0
                    p_cost += abs(float(chp*pricesp[len(fsp_load)+iii]))
                    q_cost += abs(float(chq*pricesp[len(fsp_load)+iii]))
                    if chp != 0 or chq != 0:
                        used_comps += 1
                change_per_nflex_fsp.append(p_cost+q_cost)
                used_fsps_nflex.append(used_comps)

                prof_non_flexible.append(profile)
        except:
            print(f"Power flow did not converge for profile {profile}")
    t_stop_run_mc_pf = time.time()

    return x_flexible, y_flexible, x_non_flexible, y_non_flexible, t_stop_run_mc_pf - t_start_run_mc_pf, \
           prof_flexible, prof_non_flexible, x_flex_minv, x_flex_maxv, x_flex_maxload, x_nflex_minv, \
           x_nflex_maxv, x_nflex_maxload, change_per_flex_fsp, change_per_nflex_fsp, used_fsps_flex, used_fsps_nflex, \
           v_plot


def run_uc6_volts(settings, net, v_profiles):
    """ Run all power flows for scenarios where loads are included in the FSP.

    :param settings: information of the json file.
    :type settings: object

    :param net: network on which the simulations will be performed.
    :type net: pandapower.network

    :param pq_profiles: P and Q values for each FSP for each iteration in the Monte Carlo simulation.
    :type pq_profiles: list

    :return: feasible P, feasible Q, infeasible P, infeasible Q, duration of simulations [s],
             FSP PQ values for the feasible y, FSP PQ values for the infeasible y.
    :rtype: list, list, list, list, float, list, list
    """
    fsp_dg = settings.fsp_dg
    fsp_load = settings.fsp_load
    dp = settings.dp
    dq = settings.dq
    init_pq_fsp_load = [[net.load['p_mw'][x], net.load['q_mvar'][x]] for x in fsp_load]
    init_pq_fsp_dg = [[net.sgen['p_mw'][x], net.sgen['q_mvar'][x]] for x in fsp_dg]

    pricesq = [40, 50, 60]
    pricesp = [40, 50, 60]

    print(init_pq_fsp_dg, init_pq_fsp_load, net.res_ext_grid['p_mw'].iloc[0], net.res_ext_grid['q_mvar'].iloc[0])
    print(v_profiles)
    if len(net.ext_grid) > 1:
        egid = 1
    else:
        egid = 0
    for key in v_profiles:
        net = update_pqs_wl2(net, profile=v_profiles[key], load_ind=fsp_load, dg_ind=fsp_dg).deepcopy()
        pp.runpp(net, numba=False)
        print(f"Profiles for {key}, resulting to:")
        print(f"    - s^0 = {net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]}")
        print(f"    - min v = {net.res_bus['vm_pu'].loc[net.res_bus['vm_pu'].idxmin()]}")
        p_cost = 0
        q_cost = 0
        used_comps = 0
        for iii, x in enumerate(fsp_load):
            chp = net.load['p_mw'][x]-init_pq_fsp_load[iii][0]
            chq = net.load['q_mvar'][x]-init_pq_fsp_load[iii][1]
            if abs(chp) < dp/2:
                chp = 0
            if abs(chq) < dq/2:
                chq = 0
            p_cost += abs(chp * pricesp[iii])
            q_cost += abs(chq * pricesq[iii])
            if chp != 0 or chq != 0:
                used_comps += 1
        for iii, x in enumerate(fsp_dg):
            chp = net.sgen['p_mw'][x]-init_pq_fsp_dg[iii][0]
            chq = net.sgen['q_mvar'][x]-init_pq_fsp_dg[iii][1]
            if abs(chp) < dp/2:
                chp = 0
            if abs(chq) < dq/2:
                chq = 0
            p_cost += abs(float(chp*pricesp[len(fsp_load)+iii]))
            q_cost += abs(float(chq*pricesp[len(fsp_load)+iii]))
            if chp != 0 or chq != 0:
                used_comps += 1
        print(f"    - Cost = {p_cost+q_cost}")
        fig = pp.plotting.plotly.pf_res_plotly(net, climits_volt=(0.95, 1.03), cpos_load=1.5, bus_size=40,
                                               cmap='rainbow', filename=f'{key}plot.html', aspectratio=(1.7, 1))
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.show()
    return


def get_fsp_ids(fsp, dataframe):
    """ Get ID of fsp number.

    :param fsp: fsp numbers.
    :type fsp: list

    :param dataframe: fsp information.
    :type dataframe: pandas.dataframe

    :return: ids of fsps.
    :rtype: list
    """
    return dataframe.iloc[fsp]['id'].tolist()


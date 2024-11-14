from TensorConvolutionPlus import FA_Estimator as TCP
import pandapower.networks as pn
import pandapower as pp
import numpy as np


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
    #print(net.load)
    return net, rng


def fix_net(net):
    """ Fix the initial network structure

    :param net: network for the case studies
    :type net: pandapower network

    :return: fixed network
    :rtype: pandapower network
    """
    bus_idx = list(net.bus.index.values)
    line_idx = list(net.line.index.values)
    trafo_idx = list(net.trafo.index.values)
    net.bus.index = np.arange(0, len(bus_idx))
    net.line.index = np.arange(0, len(line_idx))
    net.trafo.index = np.arange(0, len(trafo_idx))
    line_from = []
    line_to = []
    for line in net.line.iterrows():
        line_from.append(bus_idx.index(int(net.line.iloc[line[0]]['from_bus'])))
        line_to.append(bus_idx.index(int(net.line.iloc[line[0]]['to_bus'])))
    net.line['from_bus'] = line_from
    net.line['to_bus'] = line_to
    try:
        switch_bus = []
        switch_els = []
        for switch in net.switch.iterrows():
            switch_bus.append(bus_idx.index(int(switch[1]['bus'])))
            switch_els.append(line_idx.index(int(switch[1]['element'])))
        net.switch['element'] = switch_els
        net.switch['bus'] = switch_bus
    except:
        print("Ignoring Switches")
    gen_bus = []

    for gen in net.sgen.iterrows():
        gen_bus.append(bus_idx.index(int(gen[1]['bus'])))
    net.sgen['bus'] = gen_bus
    load_bus = []
    for load in net.load.iterrows():
        load_bus.append(bus_idx.index(int(load[1]['bus'])))
    net.load['bus'] = load_bus
    eg_bus = []
    for eg in net.ext_grid.iterrows():
        eg_bus.append(bus_idx.index(int(eg[1]['bus'])))
    net.ext_grid['bus'] = eg_bus
    traf_hv_bus = []
    traf_lv_bus = []
    for trafo in net.trafo.iterrows():
        traf_hv_bus.append(bus_idx.index(int(trafo[1]['hv_bus'])))
        traf_lv_bus.append(bus_idx.index(int(trafo[1]['lv_bus'])))
    net.trafo['hv_bus'] = traf_hv_bus
    net.trafo['lv_bus'] = traf_lv_bus
    net.sgen.index = np.arange(0, len(net.sgen))
    net.load.index = np.arange(0, len(net.load))
    return net


if __name__ == "__main__":
    #TCP.exhaustive_pf(net_name='MV Oberrhein0', dp=0.11, dq=0.3, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

    #TCP.exhaustive_pf(net_name='MV Oberrhein0', dp=0.15, dq=0.3, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

    #TCP.exhaustive_pf(net_name='MV Oberrhein0', dp=0.01, dq=0.02, fsp_load_indices=[5], fsp_dg_indices=[5])

    #TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=12000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

    #TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=6000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3], distribution='Uniform')

    #TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=6000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3], distribution='Kumaraswamy')

    #TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=6000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

    #TCP.opf(net_name='CIGRE MV', opf_step=0.1, fsp_load_indices=[3, 5, 8], fsp_dg_indices=[8])

    #TCP.opf(net_name='CIGRE MV', opf_step=0.1, fsp_load_indices=[1, 4, 9], fsp_dg_indices=[8])

    #TCP.tc_plus(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2, 3])

    #TCP.tc_plus(net_name='MV Oberrhein0', fsp_load_indices=[1, 2], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2], flex_shape='PQmax')

    #TCP.tc_plus(net_name='CIGRE MV', fsp_load_indices=[3, 4, 5], dp=0.05, dq=0.1, fsp_dg_indices=[8], non_linear_fsps=[8])

    #TCP.tc_plus_merge(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3], dp=0.025, dq=0.05, fsp_dg_indices=[1, 2, 3],
    #                  max_fsps=5)

    #TCP.tc_plus_merge(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3, 4], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2, 3],
    #                  max_fsps=6)

    #TCP.tc_plus_merge(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3, 4], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2, 3],
    #                  max_fsps=6)

    #TCP.tc_plus_save_tensors(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2, 3])

    """
    fsp_load_indices = [1, 2, 3]
    fsp_dg_indices = [1, 2, 3]
    net, net_tmp = pn.mv_oberrhein(separation_by_sub=True)
    net.load['sn_mva'] = list(net.load['p_mw'].pow(2).add(net.load['q_mvar'].pow(2)).pow(0.5))
    net.load['scaling'] = [1 for i in range(len(net.load))]
    net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
    net.switch['closed'] = [True for i in range(len(net.switch))]

    net = fix_net(net)
    rng = np.random.RandomState(212)

    net, rng = rand_resample(net, fsp_load_indices, fsp_dg_indices, rng, 0.05, 0.01, 0.05, 0.01)

    TCP.tc_plus_adapt(net=net, fsp_load_indices=fsp_load_indices, fsp_dg_indices=fsp_dg_indices)

    TCP.tc_plus(net=net, fsp_load_indices=fsp_load_indices, fsp_dg_indices=fsp_dg_indices, dp=0.05, dq=0.1,)
    """
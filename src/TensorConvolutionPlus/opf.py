import time
import pandapower as pp
import numpy as np
import pandas as pd
import logging
logging.getLogger("pandapower").setLevel(logging.ERROR)

__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def opf_fa_pck(net1, fsps_dg=[2, 4, 6], fsps_load=[14, 16], filename='', opf_step: float = 0.1,
               max_curr_per: int = 100, max_volt_pu: float = 1.05, min_volt_pu: float = 0.95):
    """ Estimate flexibility area using OPF method.

    :param net1: network.
    :type net1: pandapower network

    :param fsps_dg: distributed generators offering flexibility.
    :type fsps_dg: list[int]

    :param fsps_load: loads offering flexibility.
    :type fsps_load: list[int]

    :param filename: name to use in the plot.
    :type filename: str

    :param opf_step: step size for optimization. Default=0.1.
    :type opf_step: float

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :return:
    :rtype:
    """

    net1.sgen['scaling'] = 1.
    net1.sgen['sn_mva'] = (net1.sgen['p_mw']**2+net1.sgen['q_mvar']**2)**0.5
    net1.sgen['min_p_mw'] = 0
    net1.sgen['max_p_mw'] = net1.sgen['sn_mva']
    net1.sgen['min_q_mvar'] = -net1.sgen['sn_mva']
    net1.sgen['max_q_mvar'] = net1.sgen['sn_mva']
    net1.sgen['controllable'] = False
    net1.sgen['controllable'].iloc[fsps_dg] = True

    net1.load['scaling'] = 1.
    net1.load['sn_mva'] = (net1.load['p_mw']**2+net1.load['q_mvar']**2)**0.5
    net1.load['min_p_mw'] = 0
    net1.load['max_p_mw'] = 1*(net1.load['p_mw']**2+net1.load['q_mvar']**2)**0.5
    net1.load['min_q_mvar'] = -net1.load['max_p_mw']
    net1.load['max_q_mvar'] = net1.load['max_p_mw']
    net1.load['controllable'] = False
    net1.load['controllable'].iloc[fsps_load] = True

    net1.ext_grid['min_p_mw'] = -net1.ext_grid['s_sc_min_mva']
    net1.ext_grid['max_p_mw'] = net1.ext_grid['s_sc_max_mva']
    net1.ext_grid['min_q_mvar'] = -net1.ext_grid['s_sc_min_mva']
    net1.ext_grid['max_q_mvar'] = net1.ext_grid['s_sc_max_mva']

    net1.bus['min_vm_pu'] = min_volt_pu
    net1.bus['max_vm_pu'] = max_volt_pu
    net1.trafo['max_loading_percent'] = 10000
    net1.line['max_loading_percent'] = max_curr_per
    alphas = np.arange(0, 1+opf_step/2, opf_step)
    conv = 0
    non_conv = 0
    pp.runpp(net1, numba=False)
    init_pq = [float(net1.res_ext_grid['p_mw']), float(net1.res_ext_grid['q_mvar'])]
    res_pqs = []
    pp.create_poly_cost(net1, 0, 'ext_grid', cp1_eur_per_mw=1, cq1_eur_per_mvar=1)
    st_t = time.time()
    for a in alphas:
        b = 1-a
        for sgn in [0, 1, 2, 3]:
            if sgn == 0:
                net1.poly_cost['cp1_eur_per_mw'] = a
                net1.poly_cost['cq1_eur_per_mvar'] = b
            elif sgn == 1:
                net1.poly_cost['cp1_eur_per_mw'] = -a
                net1.poly_cost['cq1_eur_per_mvar'] = b
            elif sgn == 2:
                net1.poly_cost['cp1_eur_per_mw'] = a
                net1.poly_cost['cq1_eur_per_mvar'] = -b
            elif sgn == 3:
                net1.poly_cost['cp1_eur_per_mw'] = -a
                net1.poly_cost['cq1_eur_per_mvar'] = -b
            try:
                pp.runopp(net1, numba=False, calculate_voltage_angles=True, varbose=False, suppress_warnings=True,
                          delta=1e-10)
                conv += 1
                res_pqs.append([float(net1.res_ext_grid['p_mw']), float(net1.res_ext_grid['q_mvar']), sgn])
            except:
                non_conv += 1

    e_t = time.time()
    text = f"Initial pq {init_pq}" \
           f"\nConverged OPFs {conv}" \
           f"\nNon-Converged OPFs {non_conv}" \
           f"\nDuration {e_t-st_t.__round__(5)} [s]"
    print(text)
    res_pqs = np.array(res_pqs)
    df_opf = pd.DataFrame(res_pqs, columns=['P_MW', 'Q_MVAR', 'Objective'])
    return init_pq, df_opf, text


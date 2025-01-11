from json_reader import SettingReader
from scenario_setup import update_settings, apply_cs, get_operating_point
from data_sampler import profile_creation, conv_profile_creation, profile_creation_bf, conv_profile_creation_sq
from conv_simulations import numpy_tensor_conv_simulations_with_delta, torch_tensor_conv_simulations, \
    adaptable_new_op, numpy_tensor_conv_simulations_saving, torch_tensor_conv_large_simulations
from utils import write_conv_result, write_result, assert_limits
from plotting import plot_multi_convolution, get_uncertainty_interpret, plot_mc, plot_opf_res
from monte_carlo import all_pf_simulations
from datetime import datetime
from opf import opf_fa_pck
import networkx as nx
import pandapower as pp
import logging
logging.getLogger("pandapower").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


""" 
:no-index:
FA_Estimator includes the main package functionalities 
"""


def tc_plus(net = None, net_name: str = 'MV Oberrhein0', dp: float = 0.05, dq: float = 0.05,
            max_curr_per: int = 100, max_volt_pu: float = 1.05, min_volt_pu: float = 0.95, l_sens: float = 1,
            v_sens: float = 0.001, non_linear_fsps: list = [], fsp_load_indices: list = [], fsp_dg_indices: list = [],
            scenario_type: dict = {"name": 'CS', "no.": 0}, flex_shape: str = 'Smax'):
    """ Package Function to run the TensorConvolution+ algorithm.

    :param net: Pandapower network (optional). Default=None.
    :type net: pp.networks

    :param net_name: Network name (optional, only used if network is not given). Default='MV Oberrhein0'.
    :type net_name: str

    :param dp: step size in active power (optional). Default=0.05.
    :type dp: float

    :param dq: step size in reactive power (optional). Default=0.05.
    :type dq: float

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :param l_sens: loading sensitivity threshold (optional). Default=1.
    :type l_sens: float

    :param v_sens: voltage sensitivity threshold (optional). Default=0.001.
    :type v_sens: float

    :param non_linear_fsps: indices of non linear_fsps, offering 2 setpoints (optional). Default=[].
    :type non_linear_fsps: list

    :param fsp_load_indices: indices of load FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_load_indices: list

    :param fsp_dg_indices: indices of DG FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_dg_indices: list

    :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 0}.
    :type scenario_type: dict

    :param flex_shape: Shape of flexibility from service providers. To change the shape for different resources,
                       or change the shape for all resources, plase create a new function in place of
                       conv_profile_creation(). Currently supported shapes: "Smax": resource outputs cannot exceed the
                       maximum apparent power (semi-oval shape), "PQmax": resource active and reactive power
                       individually cannot exceed maximum apparent power (rectangle shape). Default="Smax".
    :type flex_shape: dict

    :return: Flexibility area plot and csv of dataframe are stored locally. The function returns nothing.
    :rtype: None
    """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices)+len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.dp = dp
    settings.dq = dq
    settings.v_sens = v_sens
    settings.l_sens = l_sens
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.non_lin_dgs = non_linear_fsps
    settings.scenario_type_dict = scenario_type
    settings.flex_shape = flex_shape
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)
    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                            "Assuming external grid 1 as PCC."
        else:
            egid = 0
        pcc_operating_point = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, pcc_operating_point = apply_cs(net, settings)
        else:
            pcc_operating_point = get_operating_point(settings)

    assert assert_limits(net, settings), "Network operating conditions are out of the set constraints."

    if flex_shape == 'PQmax':
        # Flexibility of P cannot exceed Smax, flexibility of |Q| cannot exceed Smax --> Rectangle shape
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation_sq(settings.dp, settings.dq, net, services=settings.fsps,
                                     flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                     non_linear_dgs=settings.non_lin_dgs)
    else:
        # Flexibility cannot exceed Smax --> Semi-Oval shape
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation(settings.dp, settings.dq, net, services=settings.fsps,
                                  flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                  non_linear_dgs=settings.non_lin_dgs)

    if non_lin_dg_names:
        df, disc_df, cont_df, inf, q_loc, p_loc, dur_str = \
            numpy_tensor_conv_simulations_with_delta(net, pq_profiles, settings.dp, settings.dq,
                                                     pcc_operating_point, small_fsp_prof, non_lin_dg_names,
                                                     comp_fsp_v_sens=settings.v_sens,
                                                     comp_fsp_l_sens=settings.l_sens, max_l=settings.max_curr,
                                                     max_v=settings.max_volt, min_v=settings.min_volt)
        print(dur_str)
    else:
        df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str, extra = \
            torch_tensor_conv_simulations(net, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                          small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                          comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens,
                                          max_l=settings.max_curr)
        print(dur_str)
        # import module

    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'TensorConvolutionPlus'+str(name_ob)
    name = write_conv_result(df, name)
    plot_multi_convolution(name, inf, q_loc, p_loc, dur_str, loc='')
    if disc_df is not None:
        get_uncertainty_interpret(uncert_df=disc_df, safe_df=cont_df, ones_df=ones_df, extra=extra)
    return


def tc_plus_adapt(net = None, net_name: str = 'MV Oberrhein0', max_curr_per: int = 100,
                  max_volt_pu: float = 1.05, min_volt_pu: float = 0.95, non_linear_fsps: list = [],
                  fsp_load_indices: list = [], fsp_dg_indices: list = [],
                  scenario_type: dict = {"name": 'CS', "no.": 0}):
    """ Package Function to run the TensorConvolution+ algorithm and adapt from stored tensors for previous OCs.

    :param net: Pandapower network (optional). Default=None.
    :type net: pp.networks

    :param net_name: Network name (optional, only used if network is not given). Default='MV Oberrhein0'.
    :type net_name: str

    :param dp: step size in active power (optional). Default=0.05.
    :type dp: float

    :param dq: step size in reactive power (optional). Default=0.05.
    :type dq: float

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :param non_linear_fsps: not tested for this modality. Kept as input for future expansion. Default=[].
    :type non_linear_fsps: list

    :param fsp_load_indices: indices of load FSPs (must be the exact same as the scenario it adapts from). Default=[].
    :type fsp_load_indices: list

    :param fsp_dg_indices: indices of DG FSPs (must be the exact same as the scenario it adapts from). Default=[].
    :type fsp_dg_indices: list

    :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 0}.
    :type scenario_type: dict

    :return: Flexibility area plot and csv of dataframe are stored locally. The function returns nothing.
    :rtype: None
    """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices)+len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    if non_linear_fsps:
        assert Warning, "non linear FSPs currently not supported for this function."
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.non_lin_dgs = non_linear_fsps
    settings.scenario_type_dict = scenario_type
    settings.adapt = True
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)
    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                            "Assuming external grid 1 as PCC."
        else:
            egid = 0
        pcc_operating_point = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, pcc_operating_point = apply_cs(net, settings)
        else:
            pcc_operating_point = get_operating_point(settings)
    assert assert_limits(net, settings), "Network operating conditions are out of the set constraints."

    df, disc_df, cont_df, inf, q_loc, p_loc, dur_str = \
        adaptable_new_op(net, pcc_operating_point, minmax_v=[settings.min_volt, settings.max_volt],
                         max_l=settings.max_curr)
    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'TensorConvolutionPlusAdapt' + str(name_ob)
    name = write_conv_result(df, name)
    plot_multi_convolution(name, inf, q_loc, p_loc, dur_str, loc='')
    return


def tc_plus_save_tensors(net = None, net_name: str = 'MV Oberrhein0', dp: float = 0.05, dq: float = 0.05,
                         max_curr_per: int = 100, max_volt_pu: float = 1.05, min_volt_pu: float = 0.95,
                         l_sens: float = 1, v_sens: float = 0.001, non_linear_fsps: list = [],
                         fsp_load_indices: list = [], fsp_dg_indices: list = [],
                         scenario_type: dict = {"name": 'CS', "no.": 0}, flex_shape: str = 'Smax'):
    """ Package Function to run the TensorConvolution+ algorithm and save the tensors during estimation.

    :param net: Pandapower network (optional). Default=None.
    :type net: pp.networks

    :param net_name: Network name (optional, only used if network is not given). Default='MV Oberrhein0'.
    :type net_name: str

    :param dp: step size in active power (optional). Default=0.05.
    :type dp: float

    :param dq: step size in reactive power (optional). Default=0.05.
    :type dq: float

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :param l_sens: loading sensitivity threshold (optional). Default=1.
    :type l_sens: float

    :param v_sens: voltage sensitivity threshold (optional). Default=0.001.
    :type v_sens: float

    :param non_linear_fsps: indices of non linear_fsps, offering 2 setpoints (optional). Default=[].
    :type non_linear_fsps: list

    :param fsp_load_indices: indices of load FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_load_indices: list

    :param fsp_dg_indices: indices of DG FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_dg_indices: list

    :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 0}.
    :type scenario_type: dict

    :param flex_shape: Shape of flexibility from service providers. To change the shape for different resources,
                       or change the shape for all resources, plase create a new function in place of
                       conv_profile_creation(). Currently supported shapes: "Smax": resource outputs cannot exceed the
                       maximum apparent power (semi-oval shape), "PQmax": resource active and reactive power
                       individually cannot exceed maximum apparent power (rectangle shape). Default="Smax".
    :type flex_shape: dict

    :return: Flexibility area plot and csv of dataframe are stored locally. The function returns nothing.
    :rtype: None
    """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices)+len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.dp = dp
    settings.dq = dq
    settings.v_sens = v_sens
    settings.l_sens = l_sens
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.non_lin_dgs = non_linear_fsps
    settings.scenario_type_dict = scenario_type
    settings.save_tensors = True
    settings.flex_shape = flex_shape
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)
    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                           "Assuming external grid 1 as PCC."
        else:
            egid = 0
        pcc_operating_point = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, pcc_operating_point = apply_cs(net, settings)
        else:
            pcc_operating_point = get_operating_point(settings)

    assert assert_limits(net, settings), "Network operating conditions are out of the set constraints."

    if flex_shape == 'PQmax':
        # Flexibility of P cannot exceed Smax, flexibility of |Q| cannot exceed Smax --> Rectangle shape
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation_sq(settings.dp, settings.dq, net, services=settings.fsps,
                                     flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                     non_linear_dgs=settings.non_lin_dgs)
    else:
        # Flexibility cannot exceed Smax --> Semi-Oval shape
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation(settings.dp, settings.dq, net, services=settings.fsps,
                                  flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                  non_linear_dgs=settings.non_lin_dgs)
    df, disc_df, cont_df, inf, q_loc, p_loc, dur_str = \
        numpy_tensor_conv_simulations_saving(net, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                             small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                             comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'TensorConvolutionPlusStore'+str(name_ob)
    name = write_conv_result(df, name)
    plot_multi_convolution(name, inf, q_loc, p_loc, dur_str, loc='')
    return


def monte_carlo_pf(net = None, net_name: str = 'MV Oberrhein0', max_curr_per: int = 100,
                   max_volt_pu: float = 1.05, min_volt_pu: float = 0.95, no_samples: int = 10000,
                   distribution: str = 'Hard', non_linear_fsps: list = [], fsp_load_indices: list = [],
                   fsp_dg_indices: list = [], scenario_type: dict = {"name": 'CS', "no.": 0}):
    """ Package Function to run the Monte Carlo based power flow algorithm.

    :param net: pandapower network (optional). Default=None.
    :type net: pp.networks

    :param net_name: network name (optional, only used if network is not given). Default='MV Oberrhein0'.
    :type net_name: str

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :param no_samples: number of samples for power flows (optional). Default=1000.
    :type no_samples: int

    :param distribution: distribution used for generated samples for power flows (optional). Default='Hard'.
    :type distribution: str

    :param non_linear_fsps: indices of non linear_fsps, offering 2 setpoints (optional). Default=[].
    :type non_linear_fsps: list

    :param fsp_load_indices: indices of load FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_load_indices: list

    :param fsp_dg_indices: indices of DG FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_dg_indices: list

    :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 0}.
    :type scenario_type: dict

    :return: Flexibility area plot and csv of dataframe are stored locally. The function returns nothing.
    :rtype: None
    """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices)+len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    if distribution == 'Hard':
        distribution = "Normal_Limits_Oriented"
    settings.distribution = distribution
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.no_samples = no_samples
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.non_lin_dgs = non_linear_fsps
    settings.scenario_type_dict = scenario_type
    settings.save_tensors = True
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)
    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                            "Assuming external grid 1 as PCC."
        else:
            egid = 0
        pcc_operating_point = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, pcc_operating_point = apply_cs(net, settings)
        else:
            pcc_operating_point = get_operating_point(settings)

    pq_profiles, dur_samples = profile_creation(settings.no_samples, net, settings.distribution,
                                                settings.keep_mp, services=settings.fsps,
                                                flexible_loads=settings.fsp_load,
                                                flexible_dg=settings.fsp_dg,
                                                non_lin_dgs=settings.non_lin_dgs)
    x_flx, y_flx, x_non_flx, y_non_flx, t_pf, prf_flx, prf_non_flx = all_pf_simulations(settings, net, pq_profiles)
    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'MonteCarlo'+str(name_ob)
    plot_mc(x_flx, y_flx, x_non_flx, y_non_flx, pcc_operating_point, settings.no_samples, name, dur_samples, t_pf)
    write_result(x_flx, x_non_flx, y_flx, y_non_flx, name)
    return


def exhaustive_pf(net = None, net_name: str = 'MV Oberrhein0',  dp: float = 0.05, dq: float = 0.05,
                  max_curr_per: int = 100, max_volt_pu: float = 1.05, min_volt_pu: float = 0.95,
                  non_linear_fsps: list = [], fsp_load_indices: list = [],
                  fsp_dg_indices: list = [], scenario_type: dict = {"name": 'CS', "no.": 0}):
    """Package Function to run the exhaustive power flow algorithm.

    :param net: pandapower network (optional). Default=None.
    :type net: pp.networks

    :param net_name: network name (optional, only used if network is not given). Default='MV Oberrhein0'.
    :type net_name: str

    :param dp: step size in active power (optional). Default=0.05.
    :type dp: float

    :param dq: step size in reactive power (optional). Default=0.05.
    :type dq: float

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :param non_linear_fsps: indices of non linear_fsps, offering 2 setpoints (optional). Default=[].
    :type non_linear_fsps: list

    :param fsp_load_indices: indices of load FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_load_indices: list

    :param fsp_dg_indices: indices of DG FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_dg_indices: list

    :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 0}.
    :type scenario_type: dict

    :return: Flexibility area plot and csv of dataframe are stored locally. The function returns nothing.
    :rtype: None
    """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices)+len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.dp = dp
    settings.dq = dq
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.non_lin_dgs = non_linear_fsps
    settings.scenario_type_dict = scenario_type
    settings.save_tensors = True
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)
    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                            "Assuming external grid 1 as PCC."
        else:
            egid = 0
        pcc_operating_point = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, pcc_operating_point = apply_cs(net, settings)
        else:
            pcc_operating_point = get_operating_point(settings)

    pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names, no_samps = \
        profile_creation_bf(settings.dp, settings.dq, net, services=settings.fsps, flexible_loads=settings.fsp_load,
                            flexible_dgs=settings.fsp_dg, non_linear_dgs=settings.non_lin_dgs)
    print(f"Running {no_samps} power flows.")
    x_flx, y_flx, x_non_flx, y_non_flx, t_pf, prf_flx, prf_non_flx = all_pf_simulations(settings, net, pq_profiles)
    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'ExhaustivePowerFlow'+str(name_ob)
    plot_mc(x_flx, y_flx, x_non_flx, y_non_flx, pcc_operating_point, no_samps, name, dur_samples, t_pf, '')
    write_result(x_flx, x_non_flx, y_flx, y_non_flx, name)
    return


def opf(net=None, net_name: str = 'MV Cigre',  opf_step: float = 0.1,
        max_curr_per: int = 100, max_volt_pu: float = 1.05, min_volt_pu: float = 0.95,
        fsp_load_indices: list = [], fsp_dg_indices: list = [], scenario_type: dict = {"name": 'CS', "no.": 101}):
    """Package function to run the optimal power flow algorithm. This function has convergence issues, where only Cigre MV network converges and without transformer loading contraints.

    :param net: pandapower network (optional). Default=None.
    :type net: pp.networks

    :param net_name: network name (optional, only used if network is not given). Default='MV Cigre', currently the only converging option.
    :type net_name: str

    :param opf_step: OPF step size. Default=0.1.
    :type opf_step: float

    :param max_curr_per: network maximum current constraint (optional). Default=100.
    :type max_curr_per: int

    :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
    :type max_volt_pu: float

    :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
    :type min_volt_pu: float

    :param fsp_load_indices: indices of load FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_load_indices: list

    :param fsp_dg_indices: indices of DG FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
    :type fsp_dg_indices: list

    :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 101}.
    :type scenario_type: dict

    :return: Flexibility area plot is stored locally. The function returns nothing.
    :rtype: None
    """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices)+len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.opf_step = opf_step
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.scenario_type_dict = scenario_type
    settings.opf = True
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)
    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                            "Assuming external grid 1 as PCC."
        else:
            egid = 0
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, _ = apply_cs(net, settings)
        else:
            _ = get_operating_point(settings)
    assert assert_limits(net, settings), "Network operating conditions are out of the set constraints."

    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'OptimalPowerFlow'+str(name_ob)
    init_pq, df_opf, text = opf_fa_pck(net, fsps_dg=settings.fsp_dg, fsps_load=settings.fsp_load, filename=name, opf_step=settings.opf_step,
                         max_curr_per=settings.max_curr, max_volt_pu=settings.max_volt, min_volt_pu=min_volt_pu)
    filename = write_conv_result(df_opf, name)
    plot_opf_res(filename, init_pq, text)
    return


def tc_plus_merge(net = None, net_name: str = 'MV Oberrhein0', dp: float = 0.05, dq: float = 0.05,
            max_curr_per: int = 100, max_volt_pu: float = 1.05, min_volt_pu: float = 0.95, l_sens: float = 1,
            v_sens: float = 0.001, fsp_load_indices: list = [], fsp_dg_indices: list = [],
            scenario_type: dict = {"name": 'CS', "no.": 0}, flex_shape: str = 'Smax', max_fsps: int = -1):
    """ Package Function to run the TensorConvolution+ algorithm but merge FSPs for network components that are
    sensitive to more than 'max_fsps' FSPs (to solve possible memory issues but possibly reduce accuracy).

       :param net: Pandapower network (optional). Default=None.
       :type net: pp.networks

       :param net_name: Network name (optional, only used if network is not given). Default='MV Oberrhein0'.
       :type net_name: str

       :param dp: step size in active power (optional). Default=0.05.
       :type dp: float

       :param dq: step size in reactive power (optional). Default=0.05.
       :type dq: float

       :param max_curr_per: network maximum current constraint (optional). Default=100.
       :type max_curr_per: int

       :param max_volt_pu: network maximum voltage constraint (optional). Default=1.05.
       :type max_volt_pu: float

       :param min_volt_pu: network minimum voltage constraint (optional). Default=0.95.
       :type min_volt_pu: float

       :param l_sens: loading sensitivity threshold (optional). Default=1.
       :type l_sens: float

       :param v_sens: voltage sensitivity threshold (optional). Default=0.001.
       :type v_sens: float

       :param fsp_load_indices: indices of load FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
       :type fsp_load_indices: list

       :param fsp_dg_indices: indices of DG FSPs (optional, but at least one of fsp_load_indices or fsp_dg_indices should be non-empty). Default=[].
       :type fsp_dg_indices: list

       :param scenario_type: Scenario for network settings (e.g.CS=closed switches, only used if net is None). Default={"name": 'CS', "no.": 0}.
       :type scenario_type: dict

       :param flex_shape: Shape of flexibility from service providers. To change the shape for different resources,
                          or change the shape for all resources, plase create a new function in place of
                          conv_profile_creation(). Currently supported shapes: "Smax": resource outputs cannot exceed the
                          maximum apparent power (semi-oval shape), "PQmax": resource active and reactive power
                          individually cannot exceed maximum apparent power (rectangle shape). Default="Smax".
       :type flex_shape: dict

       :param max_fsps: Number of FSPs to allow for each component's sensitivity before merging. Default=-1 would take maximum FSPs as the input FSPs -1 (merging at most 2 FSPs into 1).
       :type max_fsps: int

       :return: Flexibility area plot and csv of dataframe are stored locally. The function returns nothing.
       :rtype: None
       """
    settings = SettingReader(scenario_name='Default')
    if len(fsp_dg_indices) + len(fsp_load_indices) == 0:
        assert AssertionError, "Please provide at least one FSP index between the network loads and generators as lists"
    settings.max_curr = max_curr_per
    settings.max_volt = max_volt_pu
    settings.min_volt = min_volt_pu
    settings.dp = dp
    settings.dq = dq
    settings.v_sens = v_sens
    settings.l_sens = l_sens
    settings.fsp_load = fsp_load_indices
    settings.fsp_dg = fsp_dg_indices
    settings.scenario_type_dict = scenario_type
    settings.flex_shape = flex_shape
    if max_fsps == -1:
        max_fsps = len(settings.fsp_load) + len(settings.fsp_dg)
    settings.max_fsps = max_fsps
    settings.tester()

    settings.net_name = net_name
    settings = update_settings(settings)

    if net is not None:
        settings.net = net
        pp.runpp(net)
        if len(net.ext_grid) > 1:
            egid = 1
            assert Warning, "Currently the algorithm only works for a single PCC/External grid. " \
                            "Assuming external grid 1 as PCC."
        else:
            egid = 0
        pcc_operating_point = [net.res_ext_grid['p_mw'].iloc[egid], net.res_ext_grid['q_mvar'].iloc[egid]]
    else:
        net = settings.net
        if settings.scenario_type_dict['name'] == 'CS':
            net, pcc_operating_point = apply_cs(net, settings)
        else:
            pcc_operating_point = get_operating_point(settings)

    assert assert_limits(net, settings), "Network operating conditions are out of the set constraints."


    graph = pp.topology.create_nxgraph(net, calc_branch_impedances=True)
    spl = dict(nx.all_pairs_dijkstra_path_length(graph, weight="z_ohm"))

    dist_dicts = {}
    for fffsp in settings.fsp_dg:
        dist_dicts[net.sgen['name'].iloc[fffsp]] = {}
        min_v = 1000
        min_d = ""
        for ffsp2 in settings.fsp_dg:
            if fffsp != ffsp2:
                dist_dicts[net.sgen['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp2]] = \
                    spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                if spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]] < min_v:
                    min_v = spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                    min_d = net.sgen['name'].iloc[ffsp2]
        for ffsp3 in settings.fsp_load:
            dist_dicts[net.sgen['name'].iloc[fffsp]][net.load['name'].iloc[ffsp3]] = \
                spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
            if spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]] < min_v:
                min_v = spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                min_d = net.load['name'].iloc[ffsp3]
        dist_dicts[net.sgen['name'].iloc[fffsp]]["min"] = [min_v, min_d]
    for fffsp in settings.fsp_load:
        dist_dicts[net.load['name'].iloc[fffsp]] = {}
        min_v = 1000
        min_d = ""
        for ffsp2 in settings.fsp_load:
            if fffsp != ffsp2:
                dist_dicts[net.load['name'].iloc[fffsp]][net.load['name'].iloc[ffsp2]] = \
                    spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                if spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]] < min_v:
                    min_v = spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                    min_d = net.load['name'].iloc[ffsp2]
        for ffsp3 in settings.fsp_dg:
            dist_dicts[net.load['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp3]] = \
                spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
            if spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]] < min_v:
                min_v = spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                min_d = net.sgen['name'].iloc[ffsp3]
        dist_dicts[net.load['name'].iloc[fffsp]]["min"] = [min_v, min_d]



    if flex_shape == 'PQmax':
        # Flexibility of P cannot exceed Smax, flexibility of |Q| cannot exceed Smax --> Rectangle shape
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation_sq(settings.dp, settings.dq, net, services=settings.fsps,
                                     flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                     non_linear_dgs=settings.non_lin_dgs)
    else:
        # Flexibility cannot exceed Smax --> Semi-Oval shape
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation(settings.dp, settings.dq, net, services=settings.fsps,
                                  flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                  non_linear_dgs=settings.non_lin_dgs)


    df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str = \
        torch_tensor_conv_large_simulations(net, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                            small_fsp_prof, dist_dicts, min_max_v=[settings.min_volt, settings.max_volt],
                                            comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens,
                                            no_max=settings.max_fsps - 1)
    print(dur_str)
    # import module

    name_ob = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    name = 'TensorConvolutionPlusMegeFSPs'+str(name_ob)
    name = write_conv_result(df, name)
    plot_multi_convolution(name, inf, q_loc, p_loc, dur_str, loc='')



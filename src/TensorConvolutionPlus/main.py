#!/usr/bin/env python
from monte_carlo import all_pf_simulations, run_uc6
from utils import write_result, write_conv_result, check_limits

from plotting import plot_mc, plot_multi_convolution, get_uncertainty_interpret
from json_reader import SettingReader
from scenario_setup import update_settings, get_operating_point, apply_cs
from data_sampler import profile_creation, conv_profile_creation, profile_creation_bf, conv_profile_creation_sq
from conv_simulations import numpy_tensor_conv_simulations_with_delta, numpy_tensor_conv_simulations_saving, \
    adaptable_new_op, torch_tensor_conv_simulations
import pandapower as pp
import warnings
import sys
from accuracy_estimation import correct_feasible_and_infeasible, accuracy_brute_force, info_options, range_acc_case_study
#from opf import opf_fa


scenario_name = str(sys.argv[1])
#scenario_name = 'UC7/TCP'

__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "1.0.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    """ Main file running the scenario.
    """
    # Read scenario and update it based on the input model
    settings = SettingReader(scenario_name=scenario_name)
    settings = update_settings(settings)
    net = settings.net
    # Run power flow on the initial model
    pp.runpp(net)

    operating_point = get_operating_point(settings)
    pcc_operating_point = operating_point

    # Read the initial unaltered network results
    new_op = []
    # Update the network based on the scenario
    if settings.scenario_type_dict['name'] == 'CS':
        net, new_op = apply_cs(net, settings)
        pcc_operating_point = new_op
    # Create the new PQ values which will be used in each power flow
    check_limits(net, settings)
    if settings.no_samples >= 2 and not settings.conv_sim and \
            not settings.brute_force and not settings.compare_brute and settings.use_case_dict == {}:
        pq_profiles, dur_samples = profile_creation(settings.no_samples, net, settings.distribution,
                                                    settings.keep_mp, services=settings.fsps,
                                                    flexible_loads=settings.fsp_load,
                                                    flexible_dg=settings.fsp_dg,
                                                    non_lin_dgs=settings.non_lin_dgs)

    # If Monte Carlo simulations will be run go here
    if settings.mc_sim and settings.lib == "Pandapower":
        x_flx, y_flx, x_non_flx, y_non_flx, t_pf, prf_flx, prf_non_flx = all_pf_simulations(settings, net, pq_profiles)
        plot_mc(x_flx, y_flx, x_non_flx, y_non_flx, pcc_operating_point, settings.no_samples,
                settings.name, dur_samples, t_pf, loc='')
        write_result(x_flx, x_non_flx, y_flx, y_non_flx, settings.name.replace(" ", "_"))
    elif settings.brute_force:
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names, no_samps = \
            profile_creation_bf(settings.dp, settings.dq, net, services=settings.fsps, flexible_loads=settings.fsp_load,
                                flexible_dgs=settings.fsp_dg,  non_linear_dgs=settings.non_lin_dgs)
        print(f"Running {no_samps} power flows.")
        x_flx, y_flx, x_non_flx, y_non_flx, t_pf, prf_flx, prf_non_flx = all_pf_simulations(settings, net, pq_profiles)
        plot_mc(x_flx, y_flx, x_non_flx, y_non_flx, pcc_operating_point, settings.no_samples,
                settings.name, dur_samples, t_pf, '')
        write_result(x_flx, x_non_flx, y_flx, y_non_flx, settings.name.replace(" ", "_"))
    elif settings.uc6:
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names, no_samps = \
            profile_creation_bf(settings.dp, settings.dq, net, services=settings.fsps, flexible_loads=settings.fsp_load,
                                flexible_dgs=settings.fsp_dg,  non_linear_dgs=settings.non_lin_dgs)
        x_flx, y_flx, x_non_flx, y_non_flx, t_pf, prf_flx, prf_non_flx, \
        x_flex_minv, x_flex_maxv, x_flex_maxload, x_nflex_minv, x_nflex_maxv, x_nflex_maxload, \
        change_per_flex_fsp, change_per_nflex_fsp, used_fsps_flex, used_fsps_nflex, v_plot = run_uc6(settings, net.deepcopy(), pq_profiles)
        info_options(x_flx, x_non_flx, y_flx, y_non_flx, settings.dp, settings.dq, x_flex_maxv,
                     x_flex_minv, x_flex_maxload, change_per_flex_fsp, used_fsps_flex, prf_flx)
    elif settings.uc7:
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation_sq(settings.dp, settings.dq, net, services=settings.fsps,
                                    flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                    non_linear_dgs=settings.non_lin_dgs)
        df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str, extra = \
            torch_tensor_conv_simulations(net, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                          small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                          comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
        print(dur_samples)
        print(dur_str)
        name = write_conv_result(df, settings.name.replace(" ", "_"))
        plot_multi_convolution(name, inf, q_loc, p_loc, loc='UC7')

        #opf_fa(net, fsps_dg=settings.fsp_dg, fsps_load=settings.fsp_load, filename='', step=settings.opf_step)
    elif settings.compare_brute:
        conv_file = settings.compare_settings.get("conv_file")
        brute_force_file = settings.compare_settings.get("brute_force_file")
        accuracy_brute_force(brute_force_file, conv_file, settings.compare_settings.get("decimals"), pcc_operating_point)
    elif settings.conv_sim:
        pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
            conv_profile_creation(settings.dp, settings.dq, net, services=settings.fsps,
                                  flexible_loads=settings.fsp_load, flexible_dgs=settings.fsp_dg,
                                  non_linear_dgs=settings.non_lin_dgs)
        if settings.save_tensors:
            df, disc_df, cont_df, inf, q_loc, p_loc = \
                numpy_tensor_conv_simulations_saving(net, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                                     small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                                     comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
        elif settings.adapt:
            df, disc_df, cont_df, inf, q_loc, p_loc = \
                adaptable_new_op(net, pcc_operating_point,
                                 minmax_v=[settings.min_volt, settings.max_volt])
        elif non_lin_dg_names:
            df, disc_df, cont_df, inf, q_loc, p_loc = \
                numpy_tensor_conv_simulations_with_delta(net, pq_profiles, settings.dp, settings.dq,
                                                         pcc_operating_point, small_fsp_prof, non_lin_dg_names,
                                                         comp_fsp_v_sens=settings.v_sens,
                                                         comp_fsp_l_sens=settings.l_sens,
                                                         max_v=settings.max_volt, min_v=settings.min_volt)
        else:
            df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str, extra = \
                torch_tensor_conv_simulations(net, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                              small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                              comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
        name = write_conv_result(df, settings.name.replace(" ", "_"))
        plot_multi_convolution(name, inf, q_loc, p_loc, loc='')
        if disc_df is not None:
            get_uncertainty_interpret(uncert_df=disc_df, safe_df=cont_df, ones_df=ones_df, extra=extra)
        if settings.accuracy:
            correct_feasible_and_infeasible(settings.ground_truth, name, inf, dur_str)

    elif settings.use_case_dict.get("No.", 0) == 3:
        range_acc_case_study(net, settings, pcc_operating_point)



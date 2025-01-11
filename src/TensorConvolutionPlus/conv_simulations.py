#!/usr/bin/env python
import time
import torch
from utils import tensor_convolve_nd_torch, tensor_convolve_nd_torch_half, fix_missing_point
import pandapower as pp
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import signal
import string
import scipy.ndimage
from sklearn import preprocessing as pre
import tntorch as tn
import json
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


def get_bus_line_and_trafo_names(net):
    """
    Get names of buses, lines, and transformers in a power distribution network.
    This function extracts and returns the names of buses, lines, and transformers from the provided
    `pandapowerNet` object.

    :param net: A pandapowerNet object representing a power distribution network.
    :type net: pandapowerNet

    :return: A list of bus names, line names, and transformer names.
    :rtype: list[str], list[str], list[str]
    """

    # The function retrieves the names of buses, lines, and transformers from the provided pandapowerNet object.

    # Example:
    # >>> bus_names, line_names, trafo_names = get_bus_line_and_trafo_names(power_network)
    # >>> print(bus_names)
    # ['Bus1', 'Bus2', 'Bus3']
    # >>> print(line_names)
    # ['Line1', 'Line2']
    # >>> print(trafo_names)
    # ['Trafo1', 'Trafo2']

    bus_names = list(net.bus['name'])
    line_names = list(net.line['name'])
    trafo_names = list(net.trafo['name'])
    return bus_names, line_names, trafo_names


def create_mat_dict_tensor(result_dict, dp, dq, small_fsps=[]):
    """
    Create matrices and dictionaries from a result dictionary using PyTorch tensors.
    This function takes a dictionary of results and generates matrices and dictionaries
    based on the provided dp and dq values. It performs convolution operations on selected
    matrices and returns the resulting matrices, dictionaries, and axes for power (P) and
    reactive power (Q). Additionally, it calculates the uncertainty factor (unc_fa).

    :param result_dict: A dictionary containing result data.
    :type result_dict: dict

    :param dp: The power resolution (dp) for matrix generation.
    :type dp: float

    :param dq: The reactive power resolution (dq) for matrix generation.
    :type dq: float

    :param small_fsps: A list of keys corresponding to small flexible power systems.
    :type small_fsps: list[str]

    :return: The resulting power matrix (PQ), dictionaries, axes for
             power (P) and reactive power (Q), and the uncertainty factor matrix (unc_fa).
    :rtype: numpy.ndarray, dict, list, list, numpy.ndarray
    """

    # The function processes the result_dict to create matrices, dictionaries, axes, and uncertainty factor.

    # Example:
    # >>> pq_mat, mat_dicts, axs_p, axs_q, unc_fa = create_mat_dict_tensor(result_dict, 0.01, 0.01, ['SmallFSP1'])
    # >>> print(pq_mat)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(mat_dicts)
    # {'Result1': {'PQ': tensor([[[0., 0., 0.],
    #                              [0., 0., 0.],
    #                              [0., 0., 0.]]])},
    #  'Result2': {'PQ': tensor([[[1., 1., 0.],
    #                              [1., 0., 0.],
    #                              [0., 0., 0.]]])}}
    # >>> print(axs_p)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(axs_q)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(unc_fa)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])

    mat_dicts = {}
    axs_p = []
    axs_q = []
    keys = []
    for key in result_dict:
        tmp_dict, ax_p, ax_q = df_to_mat_tensor_torch(result_dict[key], dp, dq)
        mat_dicts[key] = tmp_dict
        axs_p.append(ax_p)
        axs_q.append(ax_q)
        keys.append(key)
    i = 0
    while keys[i] in small_fsps:
        i += 1
    pq_mat = mat_dicts[keys[i]]['PQ'].detach().cpu().numpy()
    for key in keys[i+1:]:
        if key not in small_fsps:
            pq_mat = signal.convolve2d(pq_mat, mat_dicts[key]['PQ'].detach().cpu().numpy())
    unc_fa = pq_mat.copy()
    pq_mat[pq_mat > 0.1] = 1
    return np.flipud(np.fliplr(pq_mat)), mat_dicts, axs_p, axs_q, np.flipud(np.fliplr(unc_fa))


def create_mat_dict_order(result_dict, dp, dq, small_fsps=[]):
    """
    Create matrices and dictionaries from a result dictionary using PyTorch tensors.
    This function takes a dictionary of results and generates matrices and dictionaries
    based on the provided dp and dq values. It performs convolution operations on selected
    matrices and returns the resulting matrices, dictionaries, axes for power (P) and
    reactive power (Q), and the order of keys used in the convolution.

    :param result_dict: A dictionary containing result data.
    :type result_dict: dict

    :param dp: The power resolution (dp) for matrix generation.
    :type dp: float

    :param dq: The reactive power resolution (dq) for matrix generation.
    :type dq: float

    :param small_fsps: A list of keys corresponding to small flexible power systems.
    :type small_fsps: list[str]

    :return: The resulting power matrix (PQ), dictionaries, axes for
             power (P) and reactive power (Q), and the order of keys used in convolution.
    :rtype: numpy.ndarray, dict, list, list, list
    """

    # The function processes the result_dict to create matrices, dictionaries, axes, and key order.

    # Example:
    # >>> pq_mat, mat_dicts, axs_p, axs_q, keys = create_mat_dict_order(result_dict, 0.01, 0.01, ['SmallFSP1'])
    # >>> print(pq_mat)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(mat_dicts)
    # {'Result1': {'PQ': tensor([[[0., 0., 0.],
    #                              [0., 0., 0.],
    #                              [0., 0., 0.]]])},
    #  'Result2': {'PQ': tensor([[[1., 1., 0.],
    #                              [1., 0., 0.],
    #                              [0., 0., 0.]]])}}
    # >>> print(axs_p)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(axs_q)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(keys)
    # ['Result1', 'Result2']

    mat_dicts = {}
    axs_p = []
    axs_q = []
    keys = []
    for key in result_dict:
        tmp_dict, ax_p, ax_q = df_to_mat_tensor_torch(result_dict[key], dp, dq)
        mat_dicts[key] = tmp_dict
        axs_p.append(ax_p)
        axs_q.append(ax_q)
        keys.append(key)
    i = 0
    while keys[i] in small_fsps:
        i += 1
    pq_mat = mat_dicts[keys[i]]['PQ'].detach().cpu().numpy()
    for key in keys[i+1:]:
        if key not in small_fsps:
            pq_mat = signal.convolve2d(pq_mat, mat_dicts[key]['PQ'].detach().cpu().numpy())
    pq_mat[pq_mat > 0.1] = 1
    return np.flipud(np.fliplr(pq_mat)), mat_dicts, axs_p, axs_q, keys


def create_mat_dict_tensorv2(result_dict, dp, dq):
    """
    Create matrices and dictionaries from a result dictionary using PyTorch tensors.
    This function takes a dictionary of results and generates matrices, dictionaries, axes for
    power (P) and reactive power (Q), and initialization IDs based on the provided dp and dq values.

    :param result_dict: A dictionary containing result data.
    :type result_dict: dict

    :param dp: The power resolution (dp) for matrix generation.
    :type dp: float

    :param dq: The reactive power resolution (dq) for matrix generation.
    :type dq: float

    :return: The resulting power matrix (PQ), dictionaries, axes for power (P) and
             reactive power (Q), the initialization IDs, and an uncertainty matrix for feasibility.
    :rtype: numpy.ndarray, dict, list, list, dict
    """

    # The function processes the result_dict to create matrices, dictionaries, axes, initialization IDs,
    # and an uncertainty matrix for feasibility.

    # Example:
    # >>> pq_mat, mat_dicts, axs_p, axs_q, init_ids = create_mat_dict_tensorv2(result_dict, 0.01, 0.01)
    # >>> print(pq_mat)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(mat_dicts)
    # {'Result1': {'PQ': tensor([[[0., 0., 0.],
    #                              [0., 0., 0.],
    #                              [0., 0., 0.]]])},
    #  'Result2': {'PQ': tensor([[[1., 1., 0.],
    #                              [1., 0., 0.],
    #                              [0., 0., 0.]]])}}
    # >>> print(axs_p)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(axs_q)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(init_ids)
    # {'Result1': 12345, 'Result2': 54321}

    mat_dicts = {}
    axs_p = []
    axs_q = []
    keys = []
    init_ids = {}
    for key in result_dict:
        tmp_dict, ax_p, ax_q, init_ids[key] = df_to_mat_tensor_torchv2(result_dict[key], dp, dq)
        mat_dicts[key] = tmp_dict
        axs_p.append(ax_p)
        axs_q.append(ax_q)
        keys.append(key)

    pq_mat = mat_dicts[keys[0]]['PQ'].detach().cpu().numpy()
    for key in keys[1:]:
        pq_mat = signal.convolve2d(pq_mat, mat_dicts[key]['PQ'].detach().cpu().numpy())
    unc_fa = pq_mat.copy()
    pq_mat[pq_mat > 0.1] = 1
    return np.flipud(np.fliplr(pq_mat)), mat_dicts, axs_p, axs_q, np.flipud(np.fliplr(unc_fa)), init_ids


def create_mat_dict_incl_delta(result_dict, dp, dq, non_lin_fsp, init_pq):
    """
    Create matrices, dictionaries, and axes from a result dictionary including delta values.
    This function takes a dictionary of results and generates matrices, dictionaries, axes for
    power (P) and reactive power (Q) with and without delta values. It also includes non-linear
    power systems for the given keys.

    :param result_dict: A dictionary containing result data.
    :type result_dict: dict

    :param dp: The power resolution (dp) for matrix generation.
    :type dp: float

    :param dq: The reactive power resolution (dq) for matrix generation.
    :type dq: float

    :param non_lin_fsp: A list of keys corresponding to non-linear power systems.
    :type non_lin_fsp: list

    :param init_pq: Initial values for power (P) and reactive power (Q).
    :type init_pq: tuple

    :return: The resulting power matrix (PQ), dictionaries, axes for power (P) and
             reactive power (Q) without delta, axes for power (P) and reactive power (Q) with delta,
             and an uncertainty matrix for feasibility.
    :rtype: numpy.ndarray, dict, list, list, list, list, dict
    """

    # The function processes the result_dict to create matrices, dictionaries, axes, and
    # uncertainty matrices with and without delta values.

    # Example:
    # >>> pq_mat, mat_dicts, axs_p, axs_q, axs_p_with_delta, axs_q_with_delta, \
    # unc_fa = create_mat_dict_incl_delta(result_dict, 0.01, 0.01, ['NonLinearFSP'], (0.1, 0.2))
    # >>> print(pq_mat)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(mat_dicts)
    # {'Result1': {'PQ': tensor([[[0., 0., 0.],
    #                              [0., 0., 0.],
    #                              [0., 0., 0.]]])},
    #  'Result2': {'PQ': tensor([[[1., 1., 0.],
    #                              [1., 0., 0.],
    #                              [0., 0., 0.]]])}}
    # >>> print(axs_p)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(axs_q)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(axs_p_with_delta)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(axs_q_with_delta)
    # [array([0., 0.01, 0.02]), array([0., 0.01, 0.02])]
    # >>> print(unc_fa)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])

    mat_dicts = {}
    axs_p = []
    axs_q = []
    axs_p_with_delta = []
    axs_q_with_delta = []
    keys = []
    pq_non_lins = {}
    for key in result_dict:
        if key in non_lin_fsp:
            tmp_list, ax_p, ax_q, pq_non_lin = get_delta(result_dict[key], dp, dq, init_pq)
            mat_dicts[key] = tmp_list
            pq_non_lins[key] = pq_non_lin
        else:
            tmp_dict, ax_p, ax_q = df_to_mat_tensor_scaled_and_init(result_dict[key], dp, dq)
            mat_dicts[key] = tmp_dict
            axs_p.append(ax_p)
            axs_q.append(ax_q)
        axs_p_with_delta.append(ax_p)
        axs_q_with_delta.append(ax_q)
        keys.append(key)
    while keys[0] in non_lin_fsp:
        keys = keys[1:] + [keys[0]]
    pq_mat = mat_dicts[keys[0]]['PQ'].cpu()
    for key in keys[1:]:
        if key not in non_lin_fsp:
            pq_mat = signal.convolve2d(pq_mat, mat_dicts[key]['PQ'].detach().cpu().numpy())
        else:
            pq_mat = signal.convolve2d(pq_mat, pq_non_lins[key].detach().cpu().numpy())
    unc_fa = pq_mat.copy()
    pq_mat[pq_mat > 0.1] = 1
    return np.flipud(np.fliplr(pq_mat)), mat_dicts, axs_p, axs_q, axs_p_with_delta, axs_q_with_delta, \
        np.flipud(np.fliplr(unc_fa))


def create_multi_small_fsp_fas(prof_dict, dp, dq, init_fsp_pq):
    """
    Create a multi-flexibility area (MFA) by combining small feasible space profiles.
    This function combines multiple small feasible space profiles into a single
    multi-flexibility area (MFA) using convolution.

    :param prof_dict: A dictionary containing small feasible space profiles.
    :type prof_dict: dict

    :param dp: The step size for active power (P).
    :type dp: float

    :param dq: The step size for reactive power (Q).
    :type dq: float

    :param init_fsp_pq: Initial values for active power (P) and reactive power (Q) for each profile.
    :type init_fsp_pq: dict

    :return: The multi-flexibility area (MFA) obtained by combining small feasible space profiles.
    :rtype: numpy.ndarray
    """

    # The function combines small flexibility space profiles into a multi-flexibility area (MFA) using convolution.

    # Example:
    # >>> mfa = create_multi_small_fsp_fas(prof_dict, 0.01, 0.01, init_fsp_pq)
    # >>> print(mfa)
    # (Multi-flexibility area as a numpy array)
    pq_mat = None
    for idx, fsp in enumerate(prof_dict):
        if idx != 0:
            pq_mat = signal.convolve2d(pq_mat, profiles_to_mat(prof_dict[fsp], dp, dq, init_fsp_pq[fsp][0]))
        else:
            pq_mat = profiles_to_mat(prof_dict[fsp], dp, dq, init_fsp_pq[fsp][0])
    return pq_mat


def enhance_multi_big_fa(fa, times=5.):
    """
    Enhance the size of a multi-flexibility area (MFA) using bilinear interpolation.
    This function enhances the size of a multi-flexibility area (MFA) by a specified factor using bilinear
    interpolation.

    :param fa: The multi-flexibility area (MFA) to be enhanced.
    :type fa: numpy.ndarray

    :param times: The enhancement factor (default is 5.0).
    :type times: float, optional

    :return: The enhanced multi-flexibility area (MFA).
    :rtype: numpy.ndarray
    """

    # The function enhances the MFA using bilinear interpolation.

    # Example:
    # >>> enhanced_fa = enhance_multi_big_fa(mfa, 5.0)
    # >>> print(enhanced_fa)
    # (Enhanced MFA as a numpy array)

    # order 0 =  nearest interpolation, 1 = bi-linear and 2 = cubic
    # https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
    new_fa = scipy.ndimage.zoom(fa.astype('float64'), times, mode='nearest', order=1)
    return new_fa


def reduce_multi_fa_small(fa, times=0.5):
    """
    Reduce the size of a multi-flexibility area (MFA) using bi-linear interpolation.
    This function reduces the size of a multi-flexibility area (MFA) by a specified factor using bi-linear
    interpolation.

    :param fa: The multi-flexibility area (MFA) to be reduced.
    :type fa: numpy.ndarray

    :param times: The reduction factor (default is 0.5).
    :type times: float, optional

    :return: The reduced multi-flexibility area (MFA).
    :rtype: numpy.ndarray
    """

    # The function reduces the MFA using bi-linear interpolation.

    # Example:
    # >>> reduced_fa = reduce_multi_fa_small(mfa, 0.5)
    # >>> print(reduced_fa)
    # array([[0., 0.],
    #        [1., 0.]])
    # order 0 =  nearest interpolation, 1 = bilinear and 2 = cubic
    # https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
    new_fa = scipy.ndimage.zoom(fa.astype('float64'), times, mode='nearest', order=1)
    return new_fa


def get_multi_uncertain_fa(pq_mat, large_fa, small_prof_dict, dp, dq, init_fsp_pq, p_axes, q_axes):
    """
       Calculate multiple uncertain feasibility areas from given power matrices.
       This function calculates multiple uncertain feasibility areas (UFAs) from the given power matrices.
       It uses both small and large feasibility areas to create UFAs and scales them accordingly.

       :param pq_mat: The power matrix representing the feasibility area.
       :type pq_mat: numpy.ndarray

       :param large_fa: The large feasibility area used for enhancement.
       :type large_fa: numpy.ndarray

       :param small_prof_dict: A dictionary containing small feasibility profiles.
       :type small_prof_dict: dict

       :param dp: The power resolution (dp) for matrix generation.
       :type dp: float

       :param dq: The reactive power resolution (dq) for matrix generation.
       :type dq: float

       :param init_fsp_pq: Initial values for power (P) and reactive power (Q) for small feasibility profiles.
       :type init_fsp_pq: dict

       :param p_axes: The power (P) axes for matrix generation.
       :type p_axes: numpy.ndarray

       :param q_axes: The reactive power (Q) axes for matrix generation.
       :type q_axes: numpy.ndarray

       :return: The scaled uncertain feasibility area (UFA), the scaled safe area, the scaled
                uncertain feasibility area with ones, the new power (P) axes for the large feasibility area, and
                the new reactive power (Q) axes for the large feasibility area.
       :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
       """

    # The function calculates UFAs, scales them, and returns them along with the new axes.

    # Example:
    # >>> scaled_uncertain, safe_area, uncertain_ones_fa, new_p_axes, new_q_axes = \
    # get_multi_uncertain_fa(pq_mat, large_fa, small_prof_dict, 0.01, 0.01, (0.1, 0.2), \
    # np.arange(0, 0.5, 0.01), np.arange(0, 0.5, 0.01))
    # >>> print(scaled_uncertain)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(safe_area)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(uncertain_ones_fa)
    # array([[1., 1., 0.],
    #        [1., 0., 0.],
    #        [0., 0., 0.]])
    # >>> print(new_p_axes)
    # [0.   0.01 0.02]
    # >>> print(new_q_axes)
    # [0.   0.01 0.02]
    fa_small = create_multi_small_fsp_fas(small_prof_dict, dp/10, dq/10, init_fsp_pq)
    fa_large_enhanced = enhance_multi_big_fa(large_fa, 5)
    fa_large_enhanced = np.where((fa_large_enhanced >= 0.5) & (fa_large_enhanced <= 1), 1, fa_large_enhanced)
    fa_large_enhanced[fa_large_enhanced < 0.5] = 0

    ones_enhanced = enhance_multi_big_fa(pq_mat, 5)
    ones_enhanced[ones_enhanced >= 0.5] = 1
    ones_enhanced[ones_enhanced < 0.5] = 0

    fa_small = reduce_multi_fa_small(fa_small, 0.5)

    uncertain_fa = signal.convolve2d(fa_small, fa_large_enhanced, mode='full')
    uncertain_ones_fa = signal.convolve2d(fa_small, ones_enhanced, mode='full')

    large_fa_enh = np.pad(fa_large_enhanced, [(int(len(fa_small)/2), len(fa_small)-int(len(fa_small)/2)-1),
                                              (int(len(fa_small[0])/2), len(fa_small[0])-int(len(fa_small[0])/2)-1)],
                          mode='constant', constant_values=(0,))

    large_fa_ons = fa_large_enhanced.copy()
    large_fa_ons[fa_large_enhanced >= 0.5] = 1
    large_fa_enh_ons = large_fa_enh.copy()
    large_fa_enh_ons[large_fa_enh_ons >= 0.5] = 1

    scaled_uncertain = pre.MinMaxScaler(feature_range=(0, 1)).fit_transform(uncertain_fa)
    safe_area = large_fa_enh.copy()
    safe_area *= 1./safe_area.max()
    safe_area += large_fa_enh_ons

    p_step = abs((p_axes[-1] - p_axes[0])/len(fa_large_enhanced[0]))
    q_step = abs((q_axes[-1] - q_axes[0])/len(fa_large_enhanced))
    new_p_axes = np.arange(p_axes[0]-int(len(fa_small[0])/2)*p_step,
                           p_axes[-1]+(len(fa_small[0])-int(len(fa_small[0])/2)-1)*p_step+0.5*p_step, p_step)
    new_q_axes = np.arange(q_axes[0]-int(len(fa_small)/2)*q_step,
                           q_axes[-1]+(len(fa_small)-int(len(fa_small)/2)-1)*q_step+0.5*q_step, q_step)
    return scaled_uncertain, safe_area, uncertain_ones_fa, new_p_axes[:len(large_fa_enh[0])], \
        new_q_axes[:len(large_fa_enh)]


def get_init_net_state(net):
    """
        Calculate initial network component voltage and loading values.
        This function calculates the initial network component voltage and loading values.

        :return: Two dictionaries containing the initial voltage magnitude for buses,
                 and loading for lines/transformers.
        :rtype: dict, dict
        """

    init_v = {}
    init_load = {}
    for i, row in net.res_bus.iterrows():
        init_v[net.bus['name'].iloc[i]] = float(row['vm_pu'])
    for i, row in net.res_line.iterrows():
        init_load[net.line['name'].iloc[i]] = float(row['loading_percent'])
    for i, row in net.res_trafo.iterrows():
        init_load[net.trafo['name'].iloc[i]] = float(row['loading_percent'])
    return init_v, init_load


def torch_tensor_conv_simulations(net, pq_profiles, dp, dq, init_pcc_pq, small_fsp_prof,
                                  comp_fsp_v_sens=0.001, comp_fsp_l_sens=0.1, min_max_v=[0.95, 1.05], max_l=100):
    """
       Calculate FA using Tensors and Convolutions.
       This function calculates the flexibility area (FA) using the TensorConvolution+ algorithm.

       :param net: The distribution network.
       :type net: pandapower.network

       :param pq_profiles: The sampled p,q setpoints for the DG.
       :type pq_profiles: numpy.ndarray/list

       :param dp: The power resolution (dp) for matrix generation.
       :type dp: float

       :param dq: The reactive power resolution (dq) for matrix generation.
       :type dq: float

       :param init_pcc_pq: Initial values for power (P) and reactive power (Q) of the PCC.
       :type init_pcc_pq: tuple

       :param small_fsp_prof: The sampled p,q setpoints for the small DG.
       :type small_fsp_prof: numpy.ndarray/list

       :param comp_fsp_v_sens: The scenario voltage sensitivity.
       :type comp_fsp_v_sens: float

       :param comp_fsp_l_sens: The scenario loading sensitivity.
       :type comp_fsp_l_sens: float

       :param min_max_v: The scenario network voltage constraints.
       :type min_max_v: list[float, float]

       :param max_l: The scenario network loading constraints.
       :type max_l: float

       :return: The flexibility area (FA), the uncertainty including flexibility area, the reachable area,
                the binary flexibility area, the p index of initial operating point, the q index of initial operating
                point, a string on the simulation duration, extra info depending on scenario.
       :rtype: pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, float, float, str,
               pandas.Dataframe
       """

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    #  -----------------------------------------Preparation------------------------------------------------------------
    t0_s = time.time()
    init_v, init_load = get_init_net_state(net)
    bus_nm, line_nm, trafo_nm = get_bus_line_and_trafo_names(net)
    t0_e = time.time()

    #  -----------------------------------------Power Flows------------------------------------------------------------
    result_dict, std_dict, duration = run_all_tensor_flex_areas(net, pq_profiles, bus_nm, line_nm, trafo_nm,
                                                                init_v, init_load)
    t1_e = time.time()
    #  ------------------Identify FSPs which are smaller than the resolution (to be used for uncertainty)--------------
    small_fsps = []
    small_fsp_init = {}
    for key in pq_profiles:
        if len(pq_profiles[key]) == 1:
            small_fsps.append(key)
            small_fsp_init[key] = pq_profiles[key]
    t2_e = time.time()
    #  -------------------PF results to network component vs FSP sensitivity dictionary--------------------------------
    pq_mat, mat_dicts, axs_p, axs_q, unc_fa = create_mat_dict_tensor(result_dict, dp, dq, small_fsps)
    t3_e = time.time()
    #  --------------Identify effective and non-efective FSPs per component--------------------------------------------
    fsp_effective_per_comp = {}
    fsp_ineffective_per_comp = {}
    comps = []
    for comp in list(std_dict[list(std_dict.keys())[0]]['std'].index):
        fsp_effective_per_comp[comp] = []
        fsp_ineffective_per_comp[comp] = []
        comps.append(comp)
    for no, key in enumerate(std_dict):
        if key not in small_fsps:
            for comp in comps:
                if ('Bus' in comp or 'bus' in comp) and (abs(std_dict[key]['max'][comp]) >= comp_fsp_v_sens or
                                                         abs(std_dict[key]['min'][comp]) >= comp_fsp_v_sens):
                    fsp_effective_per_comp[comp].append(key)
                elif 'Bus' in comp or 'bus' in comp:
                    fsp_ineffective_per_comp[comp].append(key)
                elif abs(std_dict[key]['max'][comp]) >= comp_fsp_l_sens or \
                        abs(std_dict[key]['min'][comp]) > comp_fsp_l_sens:
                    fsp_effective_per_comp[comp].append(key)
                else:
                    fsp_ineffective_per_comp[comp].append(key)
    t4_e = time.time()
    #  ----------------------------Remove components far from constraints----------------------------------------------
    for comp in init_v:
        max_dv = 0
        min_dv = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dv += std_dict[fpc]['max'][comp]
            min_dv += std_dict[fpc]['min'][comp]
        if min_max_v[0]-min_dv <= init_v[comp] <= min_max_v[1]-max_dv:
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
    for comp in init_load:
        max_dl = 0
        min_dl = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dl += std_dict[fpc]['max'][comp]
            min_dl += std_dict[fpc]['min'][comp]
        if max_l >= abs(init_load[comp]+max_dl) and max_l >= abs(init_load[comp]+min_dl):
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
    t5_e = time.time()
    #  ---------------------- Apply convolutions-----------------------------------------------------------------------
    per_comp_flex = []
    # Find a way to remove components with min and max = 0 (0 std)
    print(f"FSP effective per comp {fsp_effective_per_comp}, FSP innefective per comp {fsp_ineffective_per_comp}")
    for key in tqdm(fsp_effective_per_comp, desc="Network Components FAs"):
        if fsp_effective_per_comp[key]:
            conv_key = get_multi_conv_torch(fsp_effective_per_comp[key], mat_dicts, key, bus_nm, init_v, init_load,
                                            max_v=min_max_v[1], min_v=min_max_v[0])

            if fsp_ineffective_per_comp[key]:
                conv_key = get_non_effective_multi_conv(fsp_ineffective_per_comp[key], conv_key, mat_dicts)
                per_comp_flex.append(conv_key)
            else:
                per_comp_flex.append(conv_key)
    if len(per_comp_flex) != 0:
        final_flex = per_comp_flex[0]
        for arr in per_comp_flex[1:]:
            final_flex = np.minimum(final_flex, arr)
    else:
        final_flex = unc_fa
    final_flex = final_flex.astype(float)
    if small_fsp_prof:
        final_flex_tmp = final_flex.copy()
    final_flex *= 1./final_flex.max()
    ff_one = final_flex.copy()
    ff_one[ff_one != 0] = 1
    final_flex += ff_one
    # Apply axes
    t6_e = time.time()

    q_axes, p_axes = get_new_conv_axes2(axs_q, axs_p, len(pq_mat), len(pq_mat.T),
                                        float(init_pcc_pq[1]), float(init_pcc_pq[0]))
    q_axis = q_axes[::-1]
    p_axis = p_axes[::-1]

    extra = []
    conv_total = pd.DataFrame(final_flex, index=q_axis, columns=p_axis)
    inf = pd.DataFrame(pq_mat, index=q_axis, columns=p_axis)

    q_index, _ = find_value_close2list(q_axis, float(init_pcc_pq[1]))
    p_index, _ = find_value_close2list(p_axis, float(init_pcc_pq[0]))
    #conv_total.loc[q_axis[q_index], p_axis[p_index]] = 2
    t7_e = time.time()
    if small_fsp_prof:
        scaled_uncertain, safe_area, uncertain_ones_fa, uncert_p_axs, uncert_q_axs = \
            get_multi_uncertain_fa(pq_mat, final_flex_tmp, small_fsp_prof, dp, dq, small_fsp_init, p_axes, q_axes)
        uncert_q_axs = uncert_q_axs[::-1]
        uncert_p_axs = uncert_p_axs[::-1]
        uncer_conv = pd.DataFrame(np.flipud(np.fliplr(scaled_uncertain)),  index=uncert_q_axs, columns=uncert_p_axs)
        safe_conv = pd.DataFrame(np.flipud(np.fliplr(safe_area)),  index=uncert_q_axs, columns=uncert_p_axs)
        ones_conv = pd.DataFrame(np.flipud(np.fliplr(uncertain_ones_fa)),  index=uncert_q_axs, columns=uncert_p_axs)
        q_index_tmp, _ = find_value_close2list(uncert_q_axs, float(init_pcc_pq[1]))
        p_index_tmp, _ = find_value_close2list(uncert_p_axs, float(init_pcc_pq[0]))
        discrete_conv = uncer_conv
        continuous_conv = safe_conv
        extra = [len(uncert_p_axs)-p_index_tmp, len(uncert_q_axs)-q_index_tmp]
    else:
        discrete_conv = None
        continuous_conv = None
        ones_conv = None
    t8_e = time.time()
    dur_str = f"Duration distribution: \n    -Preparation={t0_e-t0_s}s,\n    -Power Flows={t1_e-t0_e}s," +\
        f"\n    -Net. Component vs FSP dictionary={t3_e-t2_e}s,\n    -Small FSPs={t2_e-t1_e}s," +\
        f"\n    -Effective FSPs per Component={t4_e-t3_e}s," +\
        f"\n    -Removal Safe Components={t5_e-t4_e}s,\n    -(Tensor & 2D) Convolutions={t6_e-t5_e}s,\n" +\
        f"    -Applying Axes and Init Point={t7_e-t6_e}s,\n    -Small FSP Uncertainty calculation={t8_e-t7_e}s,\n"+ \
        f"    -Total={t8_e-t0_s}"
    return conv_total, discrete_conv, continuous_conv, inf, ones_conv, q_index, p_index, dur_str, extra


def torch_tensor_conv_large_simulations(net, pq_profiles, dp, dq, init_pcc_pq, small_fsp_prof, dist_dicts,
                                        comp_fsp_v_sens=0.001, comp_fsp_l_sens=1,
                                        min_max_v=[0.95, 1.05], max_l=100, no_max=5):
    """
       Calculate FA using Tensors and Convolutions, but merge FSPs in components who have no_max or more FSPs
       sensitivities, until maximum sensitive FSPs are no_max-1 per component.
       This function calculates the flexibility area (FA) using the TensorConvolution+ algorithm.

       :param net: The distribution network.
       :type net: pandapower.network

       :param pq_profiles: The sampled p,q setpoints for the DG.
       :type pq_profiles: numpy.ndarray/list

       :param dp: The power resolution (dp) for matrix generation.
       :type dp: float

       :param dq: The reactive power resolution (dq) for matrix generation.
       :type dq: float

       :param init_pcc_pq: Initial values for power (P) and reactive power (Q) of the PCC.
       :type init_pcc_pq: tuple

       :param small_fsp_prof: The sampled p,q setpoints for the small DG.
       :type small_fsp_prof: numpy.ndarray/list

       :param dist_dicts: The dinstance between each FSP pair.
       :type dist_dicts: dict

       :param comp_fsp_v_sens: The scenario voltage sensitivity.
       :type comp_fsp_v_sens: float

       :param comp_fsp_l_sens: The scenario loading sensitivity.
       :type comp_fsp_l_sens: float

       :param min_max_v: The scenario network voltage constraints.
       :type min_max_v: list[float, float]

       :param max_l: The scenario network loading constraints.
       :type max_l: float

       :param no_max: The scenario maximum+1 FSPs per component.
       :type no_max: int

       :return: The flexibility area (FA), the uncertainty including flexibility area, the reachable area,
                the binary flexibility area, the p index of initial operating point, the q index of initial operating
                point, a string on the simulation duration, extra info depending on scenario.
       :rtype: pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, float, float, str,
               pandas.Dataframe
       """
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    #  -----------------------------------------Preparation------------------------------------------------------------
    t0_s = time.time()
    init_v, init_load = get_init_net_state(net)
    bus_nm, line_nm, trafo_nm = get_bus_line_and_trafo_names(net)
    t0_e = time.time()
    #  -----------------------------------------Power Flows------------------------------------------------------------
    result_dict, std_dict, duration = run_all_tensor_flex_areas(net, pq_profiles, bus_nm, line_nm, trafo_nm,
                                                                init_v, init_load)
    t1_e = time.time()
    # --------------------PF results to network component vs FSP sensitivity dictionary--------------------------------
    pq_mat, mat_dicts, axs_p, axs_q, unc_fa, init_ids = create_mat_dict_tensorv2(result_dict, dp, dq)
    t2_e = time.time()
    #  ------------------Identify FSPs which are smaller than the resolution (to be used for uncertainty)--------------
    small_fsps = []
    small_fsp_init = {}
    for key in pq_profiles:
        if len(pq_profiles[key]) == 1:
            small_fsps.append(key)
            small_fsp_init[key] = pq_profiles[key]
    del pq_profiles
    t3_e = time.time()
    #  --------------Identify effective and non-efective FSPs per component--------------------------------------------
    fsp_effective_per_comp = {}
    fsp_ineffective_per_comp = {}
    comps = []
    for comp in list(std_dict[list(std_dict.keys())[0]]['std'].index):
        fsp_effective_per_comp[comp] = []
        fsp_ineffective_per_comp[comp] = []
        comps.append(comp)
    for no, key in enumerate(std_dict):
        if key not in small_fsps:
            for comp in comps:
                if ('Bus' in comp or 'bus' in comp) and (abs(std_dict[key]['max'][comp]) >= comp_fsp_v_sens or
                                                         abs(std_dict[key]['min'][comp]) >= comp_fsp_v_sens):
                    fsp_effective_per_comp[comp].append(key)
                elif 'Bus' in comp or 'bus' in comp:
                    fsp_ineffective_per_comp[comp].append(key)
                elif abs(std_dict[key]['max'][comp]) >= comp_fsp_l_sens or \
                        abs(std_dict[key]['min'][comp]) > comp_fsp_l_sens:
                    fsp_effective_per_comp[comp].append(key)
                else:
                    fsp_ineffective_per_comp[comp].append(key)
    t4_e = time.time()
    #  ----------------------------Remove components far from constraints----------------------------------------------
    for comp in init_v:
        max_dv = 0
        min_dv = 0
        for fpc in fsp_effective_per_comp[comp]:

            max_dv += std_dict[fpc]['max'][comp]
            min_dv += std_dict[fpc]['min'][comp]
        if min_max_v[0]-min_dv <= init_v[comp] <= min_max_v[1]-max_dv:
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
    for comp in init_load:
        max_dl = 0
        min_dl = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dl += std_dict[fpc]['max'][comp]
            min_dl += std_dict[fpc]['min'][comp]
        if max_l >= abs(init_load[comp]+max_dl) and max_l >= abs(init_load[comp]+min_dl):
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
    t5_e = time.time()
    #  ---------------------- Apply convolutions-----------------------------------------------------------------------
    per_comp_flex = []
    for key in tqdm(fsp_effective_per_comp, desc="Network Components FAs"):
        if fsp_effective_per_comp[key]:
            # Apply all tensor convolutions and get 2D binary flexibility area
            if len(fsp_effective_per_comp[key]) >= no_max:
                conv_key = get_multi_conv_torch_split(fsp_effective_per_comp[key], mat_dicts, key, bus_nm, init_v,
                                                      init_load, dist_dicts, init_ids, max_v=min_max_v[1],
                                                      min_v=min_max_v[0], no_max=no_max)
            else:
                conv_key = get_multi_conv_torch(fsp_effective_per_comp[key], mat_dicts, key, bus_nm, init_v,
                                                init_load,
                                                max_v=min_max_v[1], min_v=min_max_v[0])
            if fsp_ineffective_per_comp[key]:
                # Convolute 2D tensor-conv flex area with the innefective FSPs
                conv_key = get_non_effective_multi_conv(fsp_ineffective_per_comp[key], conv_key, mat_dicts)
                per_comp_flex.append(conv_key)
            else:
                per_comp_flex.append(conv_key)
    if len(per_comp_flex) != 0:
        final_flex = per_comp_flex[0]
        for arr in per_comp_flex[1:]:
            final_flex = np.minimum(final_flex, arr)
    else:
        final_flex = unc_fa
    final_flex = final_flex.astype(float)
    if small_fsp_prof:
        final_flex_tmp = final_flex.copy()
    final_flex *= 1./final_flex.max()
    ff_one = final_flex.copy()
    ff_one[ff_one != 0] = 1
    final_flex += ff_one
    # Apply axes
    t6_e = time.time()
    q_axes, p_axes = get_new_conv_axes2(axs_q, axs_p, len(pq_mat), len(pq_mat.T),
                                        float(init_pcc_pq[1]), float(init_pcc_pq[0]))
    q_axis = q_axes[::-1]
    p_axis = p_axes[::-1]

    inf = pd.DataFrame(pq_mat, index=q_axis, columns=p_axis)
    conv_total = pd.DataFrame(final_flex, index=q_axis, columns=p_axis)
    q_index, _ = find_value_close2list(q_axis, float(init_pcc_pq[1]))
    p_index, _ = find_value_close2list(p_axis, float(init_pcc_pq[0]))
    #conv_total.loc[q_axis[q_index], p_axis[p_index]] = 2
    t7_e = time.time()
    if small_fsp_prof:
        scaled_uncertain, safe_area, uncertain_ones_fa, uncert_p_axs, uncert_q_axs = \
            get_multi_uncertain_fa(pq_mat, final_flex_tmp, small_fsp_prof, dp, dq, small_fsp_init, p_axes, q_axes)
        uncert_q_axs = uncert_q_axs[::-1]
        uncert_p_axs = uncert_p_axs[::-1]
        uncer_conv = pd.DataFrame(np.flipud(np.fliplr(scaled_uncertain)),  index=uncert_q_axs, columns=uncert_p_axs)
        safe_conv = pd.DataFrame(np.flipud(np.fliplr(safe_area)),  index=uncert_q_axs, columns=uncert_p_axs)
        ones_conv = pd.DataFrame(np.flipud(np.fliplr(uncertain_ones_fa)),  index=uncert_q_axs, columns=uncert_p_axs)
        q_index_tmp, _ = find_value_close2list(uncert_q_axs, float(init_pcc_pq[1]))
        p_index_tmp, _ = find_value_close2list(uncert_p_axs, float(init_pcc_pq[0]))
        safe_conv.loc[uncert_q_axs[-q_index_tmp], uncert_p_axs[-p_index_tmp]] = 256
        discrete_conv = uncer_conv
        continuous_conv = safe_conv
    else:
        discrete_conv = None
        continuous_conv = None
        ones_conv = None
    t8_e = time.time()
    dur_str = f"Duration distribution: \n    -Preparation={t0_e-t0_s}s,\n    -Power Flows={t1_e-t0_e}s," +\
        f"\n    -Net. Component vs FSP dictionary={t2_e-t1_e}s,\n    -Small FSPs={t3_e-t2_e}s," +\
        f"\n    -Effective FSPs per Component={t4_e-t3_e}s," +\
        f"\n    -Removal Safe Components={t5_e-t4_e}s,\n    -(Tensor & 2D) Convolutions={t6_e-t5_e}s,\n" +\
        f"    -Applying Axes and Init Point={t7_e-t6_e}s,\n    -Small FSP Uncertainty calculation={t8_e-t7_e}s,\n"+\
        f"    -Total={t8_e-t0_s}"

    return conv_total, discrete_conv, continuous_conv, inf, ones_conv, q_index, p_index, dur_str


def split_lin_from_non_lin(comp_dict, non_lin_fsp):
    """
        Split linear from non-linear FSPs.
        This function splits linear from non-linear FSPs.

        :return: Two dictionaries containing continuous capabilities FSPs,
                 and the discrete (only full curtailment) capabilities FSPs.
        :rtype: dict, dict
        """
    non_lin_comp_dict = {}
    for key in comp_dict:
        idx = 0
        to_append = []
        for fsp in comp_dict[key]:
            if fsp in non_lin_fsp:
                to_append.append(comp_dict[key].pop(idx))
            else:
                idx += 1
        non_lin_comp_dict[key] = to_append
    return comp_dict, non_lin_comp_dict


def numpy_tensor_conv_simulations_with_delta(net, pq_profiles, dp, dq, init_pcc_pq, small_fsp_prof, non_lin_fsp=[],
                                             comp_fsp_v_sens=0.001, comp_fsp_l_sens=1,
                                             max_v=1.1, min_v=0.9, max_l=100):
    """
          Calculate FA using Tensors and Convolutions, when at least 1 fsp can only shift in certain setpoints.
          This function calculates the flexibility area (FA) using the TensorConvolution+ algorithm.

          :param net: The distribution network.
          :type net: pandapower.network

          :param pq_profiles: The sampled p,q setpoints for the DG.
          :type pq_profiles: numpy.ndarray/list

          :param dp: The power resolution (dp) for matrix generation.
          :type dp: float

          :param dq: The reactive power resolution (dq) for matrix generation.
          :type dq: float

          :param init_pcc_pq: Initial values for power (P) and reactive power (Q) of the PCC.
          :type init_pcc_pq: tuple

          :param small_fsp_prof: The sampled p,q setpoints for the small DG.
          :type small_fsp_prof: numpy.ndarray/list

          :param non_lin_fsp: The non linear DG.
          :type non_lin_fsp: list

          :param comp_fsp_v_sens: The scenario voltage sensitivity.
          :type comp_fsp_v_sens: float

          :param comp_fsp_l_sens: The scenario loading sensitivity.
          :type comp_fsp_l_sens: float

          :param max_v: The scenario network maximum voltage constraints.
          :type max_v: float

          :param min_v: The scenario network minimum voltage constraints.
          :type min_v: float

          :param max_l: The scenario network loading constraints.
          :type max_l: float

          :return: The flexibility area (FA), the uncertainty including flexibility area, the reachable area,
                   the binary flexibility area, the p index of initial operating point, the q index of initial
                   operating point, a string on the simulation duration, extra info depending on scenario.
          :rtype: pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, float, float, str,
                  pandas.Dataframe
          """
    #  -----------------------------------------Preparation------------------------------------------------------------
    t0_s = time.time()
    init_v, init_load = get_init_net_state(net)
    bus_nm, line_nm, trafo_nm = get_bus_line_and_trafo_names(net)
    t0_e = time.time()
    #  -----------------------------------------Power Flows------------------------------------------------------------
    result_dict, std_dict, duration = run_all_tensor_flex_areas(net, pq_profiles, bus_nm, line_nm, trafo_nm,
                                                                init_v, init_load)
    t1_e = time.time()
    # --------------------PF results to network component vs FSP sensitivity dictionary--------------------------------
    pq_mat, mat_dicts, axs_p, axs_q, axs_p_with_delta, axs_q_with_delta, unc_fa = \
        create_mat_dict_incl_delta(result_dict, dp, dq, non_lin_fsp, init_pcc_pq)
    t2_e = time.time()
    #  ------------------Identify FSPs which are smaller than the resolution (to be used for uncertainty)--------------
    small_fsps = []
    small_fsp_init = {}
    for key in pq_profiles:
        if len(pq_profiles[key]) == 1:
            small_fsps.append(key)
            small_fsp_init[key] = pq_profiles[key]
    t3_e = time.time()
    #  --------------Identify effective and non-efective FSPs per component--------------------------------------------
    fsp_effective_per_comp = {}
    fsp_ineffective_per_comp = {}
    fsps = []
    comps = []
    for comp in list(std_dict[list(std_dict.keys())[0]]['std'].index):
        fsp_effective_per_comp[comp] = []
        fsp_ineffective_per_comp[comp] = []
        comps.append(comp)
    for no, key in enumerate(std_dict):
        if key not in small_fsps:
            if key not in non_lin_fsp:
                fsps.append(key)
            for comp in comps:
                if ('Bus' in comp or 'bus' in comp) and (abs(std_dict[key]['max'][comp]) >= comp_fsp_v_sens or
                                                         abs(std_dict[key]['min'][comp]) >= comp_fsp_v_sens):
                    fsp_effective_per_comp[comp].append(key)
                elif 'Bus' in comp or 'bus' in comp:
                    fsp_ineffective_per_comp[comp].append(key)
                elif abs(std_dict[key]['max'][comp]) >= comp_fsp_l_sens or \
                        abs(std_dict[key]['min'][comp]) > comp_fsp_l_sens:
                    fsp_effective_per_comp[comp].append(key)
                else:
                    fsp_ineffective_per_comp[comp].append(key)
    t4_e = time.time()
    #  ----------------------------Remove components far from constraints----------------------------------------------
    for comp in init_v:
        max_dv = 0
        min_dv = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dv += std_dict[fpc]['max'][comp]
            min_dv += std_dict[fpc]['min'][comp]
        if min_v-min_dv <= init_v[comp] <= max_v-max_dv:
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
    for comp in init_load:
        max_dl = 0
        min_dl = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dl += std_dict[fpc]['max'][comp]
            min_dl += std_dict[fpc]['min'][comp]
        if max_l >= abs(init_load[comp]+max_dl) and max_l >= abs(init_load[comp]+min_dl):
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
    t5_e = time.time()
    #  -------------------------Distinguish Non-Linear Effective FSPs from Comp----------------------------------------
    fsp_effective_per_comp, non_lin_eff_fsp_per_comp = split_lin_from_non_lin(fsp_effective_per_comp, non_lin_fsp)
    fsp_ineffective_per_comp, non_lin_ineff_fsp_per_comp = split_lin_from_non_lin(fsp_ineffective_per_comp, non_lin_fsp)
    #  ---------------------- Apply convolutions-----------------------------------------------------------------------
    per_comp_flex = []
    # Find a way to remove components with min and max = 0 (0 std)
    any_effective = False
    for key in tqdm(fsp_effective_per_comp, desc="Network Components FAs"):
        if fsp_effective_per_comp[key]:
            any_effective = True
            # Apply all tensor convolutions and get 2D binary flexibility area
            conv_key = get_multi_conv_key_with_delta(fsp_effective_per_comp[key], mat_dicts, key, bus_nm,
                                                     init_v, init_load, non_lin_eff_fsp_per_comp[key], [dp, dq],
                                                     max_v, min_v)
            if fsp_ineffective_per_comp[key]:
                # Convolute 2D tensor-conv flex area with the innefective FSPs
                conv_key = get_non_effective_multi_conv_with_delta(fsp_ineffective_per_comp[key], conv_key,
                                                                   mat_dicts, non_lin_ineff_fsp_per_comp[key],
                                                                   [dp, dq])
                per_comp_flex.append(conv_key)
            else:
                per_comp_flex.append(conv_key)
    if not any_effective:
        final_flex = get_multi_result_for_0_effective(fsps, mat_dicts)
        if not (non_lin_eff_fsp_per_comp or non_lin_ineff_fsp_per_comp):
            final_flex = get_result_for_0_effective_with_delta(final_flex, non_lin_eff_fsp_per_comp,
                                                               {list(init_v.keys())[0]: non_lin_fsp}, [dp, dq], bus_nm,
                                                               init_v, init_load, mat_dicts, max_v, min_v)
        else:
            final_flex = get_result_for_0_effective_with_delta(final_flex, non_lin_eff_fsp_per_comp,
                                                               non_lin_ineff_fsp_per_comp, [dp, dq], bus_nm,
                                                               init_v, init_load, mat_dicts, max_v, min_v)

    else:
        final_flex = per_comp_flex[0]
        for arr in per_comp_flex[1:]:
            final_flex = np.minimum(final_flex, arr)
    # Normalize
    if final_flex.max() > 0:
        final_flex *= 1./final_flex.max()
        ff_one = final_flex.copy()
        ff_one[ff_one != 0] = 1
        final_flex += ff_one
    # Apply axes
    t6_e = time.time()
    q_axes, p_axes = get_new_conv_axes2(axs_q_with_delta, axs_p_with_delta, len(pq_mat), len(pq_mat.T),
                                        float(init_pcc_pq[1]), float(init_pcc_pq[0]))
    q_axis = q_axes[::-1]
    p_axis = p_axes[::-1]

    inf = pd.DataFrame(pq_mat, index=q_axis, columns=p_axis)
    conv_total = pd.DataFrame(final_flex, index=q_axis, columns=p_axis)

    q_index, _ = find_value_close2list(q_axis, float(init_pcc_pq[1]))
    p_index, _ = find_value_close2list(p_axis, float(init_pcc_pq[0]))
    #conv_total.loc[q_axis[q_index], p_axis[p_index]] = 2
    t7_e = time.time()
    if small_fsp_prof:
        assert Warning, "Warning: Simultaneous Uncertainty and Discontinuity currently not supported"
    t8_e = time.time()
    dur_str = f"Duration distribution: \n    -Preparation = {t0_e-t0_s}s,\n    -Power Flows = {t1_e-t0_e}s,"\
          f"\n    -Net. Component vs FSP dictionary = {t2_e-t1_e}s,\n    -Small FSPs = {t3_e-t2_e}s,"\
          f"\n    -Effective FSPs per Component = {t4_e-t3_e}s,"\
          f"\n    -Removal Safe Components = {t5_e-t4_e}s,\n    -(Tensor & 2D) Convolutions = {t6_e-t5_e}s,\n"\
          f"    -Applying Axes and Init Point = {t7_e-t6_e}s,\n    -Small FSP Uncertainty calculation = {t8_e-t7_e}s,\n" +\
        f"    -Total={t8_e-t0_s}"
    return conv_total, None, None, inf, q_index, p_index, dur_str


def numpy_tensor_conv_simulations_saving(net, pq_profiles, dp, dq, init_pcc_pq, small_fsp_prof,
                                         comp_fsp_v_sens=0.001, comp_fsp_l_sens=0.1, min_max_v=[0.95, 1.05], max_l=100):
    """
          Calculate FA using Tensors and Convolutions, while saving the tensors (after TTD) to adapt FA
          in other operating conditions.
          This function calculates the flexibility area (FA) using the TensorConvolution+ algorithm while saving the
          tensors (after TTD) to adapt FA in other operating conditions.

          :param net: The distribution network.
          :type net: pandapower.network

          :param pq_profiles: The sampled p,q setpoints for the DG.
          :type pq_profiles: numpy.ndarray/list

          :param dp: The power resolution (dp) for matrix generation.
          :type dp: float

          :param dq: The reactive power resolution (dq) for matrix generation.
          :type dq: float

          :param init_pcc_pq: Initial values for power (P) and reactive power (Q) of the PCC.
          :type init_pcc_pq: tuple

          :param small_fsp_prof: The sampled p,q setpoints for the small DG.
          :type small_fsp_prof: numpy.ndarray/list

          :param comp_fsp_v_sens: The scenario voltage sensitivity.
          :type comp_fsp_v_sens: float

          :param comp_fsp_l_sens: The scenario loading sensitivity.
          :type comp_fsp_l_sens: float

          :param min_max_v: The scenario network voltage constraints.
          :type min_max_v: list[float, float]

          :param max_l: The scenario network loading constraints.
          :type max_l: float

          :return: The flexibility area (FA), the uncertainty including flexibility area, the reachable area,
                   the binary flexibility area, the p index of initial operating point, the q index of initial
                   operating point, a string on the simulation duration, extra info depending on scenario.
          :rtype: pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, float, float, str,
                  pandas.Dataframe
          """
    #  -----------------------------------------Preparation------------------------------------------------------------
    t0_s = time.time()
    init_v, init_load = get_init_net_state(net)
    bus_nm, line_nm, trafo_nm = get_bus_line_and_trafo_names(net)
    t0_e = time.time()
    #  -----------------------------------------Power Flows------------------------------------------------------------
    result_dict, std_dict, duration = run_all_tensor_flex_areas(net, pq_profiles, bus_nm, line_nm, trafo_nm,
                                                                init_v, init_load)
    t1_e = time.time()
    # --------------------PF results to network component vs FSP sensitivity dictionary--------------------------------
    pq_mat, mat_dicts, axs_p, axs_q, fsps_ordered = create_mat_dict_order(result_dict, dp, dq)
    t2_e = time.time()
    #  ------------------Identify FSPs which are smaller than the resolution (to be used for uncertainty)--------------
    small_fsps = []
    small_fsp_init = {}
    for key in pq_profiles:
        if len(pq_profiles[key]) == 1:
            small_fsps.append(key)
            small_fsp_init[key] = pq_profiles[key]
    t3_e = time.time()
    #  --------------Identify effective and non-efective FSPs per component--------------------------------------------
    fsp_effective_per_comp = {}
    fsp_ineffective_per_comp = {}
    comps = []
    for comp in list(std_dict[list(std_dict.keys())[0]]['std'].index):
        fsp_effective_per_comp[comp] = []
        fsp_ineffective_per_comp[comp] = []
        comps.append(comp)
    for no, key in enumerate(std_dict):
        if key not in small_fsps:
            for comp in comps:
                if ('Bus' in comp or 'bus' in comp) and (abs(std_dict[key]['max'][comp]) >= comp_fsp_v_sens or
                                                         abs(std_dict[key]['min'][comp]) >= comp_fsp_v_sens):
                    fsp_effective_per_comp[comp].append(key)
                elif 'Bus' in comp or 'bus' in comp:
                    fsp_ineffective_per_comp[comp].append(key)
                elif abs(std_dict[key]['max'][comp]) >= comp_fsp_l_sens or \
                        abs(std_dict[key]['min'][comp]) > comp_fsp_l_sens:
                    fsp_effective_per_comp[comp].append(key)
                else:
                    fsp_ineffective_per_comp[comp].append(key)
    t4_e = time.time()
    #  ----------------------------Remove components far from constraints----------------------------------------------
    comps_shifts = {}
    for comp in init_v:
        max_dv = 0
        min_dv = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dv += std_dict[fpc]['max'][comp]
            min_dv += std_dict[fpc]['min'][comp]
        if min_max_v[0]-min_dv <= init_v[comp] <= min_max_v[1]-max_dv:
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
        else:
            comps_shifts[comp] = [min_dv, max_dv]
    for comp in init_load:
        max_dl = 0
        min_dl = 0
        for fpc in fsp_effective_per_comp[comp]:
            max_dl += std_dict[fpc]['max'][comp]
            min_dl += std_dict[fpc]['min'][comp]
        if max_l >= abs(init_load[comp]+max_dl) and max_l >= abs(init_load[comp]+min_dl):
            _ = fsp_effective_per_comp.pop(comp, None)
            _ = fsp_ineffective_per_comp.pop(comp, None)
        else:
            comps_shifts[comp] = [min_dl, max_dl]
    t5_e = time.time()
    #  ----------------------Save FSP effective dict-------------------------------------------------------------------
    with open("./FSP_effective_per_Comp.json", "w") as fp:
        json.dump({'Effective': fsp_effective_per_comp, 'Non-Effective': fsp_ineffective_per_comp,
                   'FSPs': fsps_ordered, 'axs_pq': [[ap.tolist() for ap in axs_p], [aq.tolist() for aq in axs_q]],
                   'Shifts': comps_shifts}, fp)
    np.save("./mat_dicts.npy", mat_dicts)
    np.save("./pq_mat.npy", pq_mat)
    #  ---------------------- Apply convolutions-----------------------------------------------------------------------
    per_comp_flex = []
    # Find a way to remove components with min and max = 0 (0 std)
    for key in tqdm(fsp_effective_per_comp, desc="Network Components FAs"):
        if fsp_effective_per_comp[key]:
            # Apply all tensor convolutions and get 2D binary flexibility area
            conv_key = get_multi_conv_key_saving(fsp_effective_per_comp[key], mat_dicts, key, bus_nm, init_v,
                                                 init_load, max_v=min_max_v[1], min_v=min_max_v[0])
            if fsp_ineffective_per_comp[key]:
                # Convolute 2D tensor-conv flex area with the innefective FSPs
                conv_key = get_non_effective_multi_conv(fsp_ineffective_per_comp[key], conv_key, mat_dicts)
                per_comp_flex.append(conv_key)
            else:
                per_comp_flex.append(conv_key)
    final_flex = per_comp_flex[0]
    for arr in per_comp_flex[1:]:
        final_flex = np.minimum(final_flex, arr)
    final_flex *= 1./final_flex.max()
    ff_one = final_flex.copy()
    ff_one[ff_one != 0] = 1
    final_flex += ff_one

    # Apply axes
    t6_e = time.time()
    q_axes, p_axes = get_new_conv_axes2(axs_q, axs_p, len(pq_mat), len(pq_mat.T),
                                        float(init_pcc_pq[1]), float(init_pcc_pq[0]))
    p_shifts = [p_vl - float(init_pcc_pq[0]) for p_vl in p_axes]
    q_shifts = [q_vl - float(init_pcc_pq[1]) for q_vl in q_axes]

    np.save("./p_axis.npy", np.array(p_shifts))
    np.save("./q_axis.npy", np.array(q_shifts))

    q_axis = q_axes[::-1]
    p_axis = p_axes[::-1]


    inf = pd.DataFrame(pq_mat, index=q_axis, columns=p_axis)
    conv_total = pd.DataFrame(final_flex, index=q_axis, columns=p_axis)

    q_index, _ = find_value_close2list(q_axis, float(init_pcc_pq[1]))
    p_index, _ = find_value_close2list(p_axis, float(init_pcc_pq[0]))
    #conv_total.loc[q_axis[q_index], p_axis[p_index]] = 2
    t7_e = time.time()
    if small_fsp_prof:
        assert Warning, "Warning: Not yet implemented"
    t8_e = time.time()
    dur_str = f"Duration distribution: \n    -Preparation = {t0_e-t0_s}s,\n    -Power Flows = {t1_e-t0_e}s,"\
              f"\n    -Net. Component vs FSP dictionary = {t2_e-t1_e}s,\n    -Small FSPs = {t3_e-t2_e}s,"\
              f"\n    -Effective FSPs per Component = {t4_e-t3_e}s,"\
              f"\n    -Removal Safe Components = {t5_e-t4_e}s,\n    -(Tensor & 2D) Convolutions = {t6_e-t5_e}s,\n"\
              f"    -Applying Axes and Init Point = {t7_e-t6_e}s,\n    " \
              f"-Small FSP Uncertainty calculation = {t8_e-t7_e}s"
    return conv_total, None, None, inf, q_index, p_index, dur_str


def adaptable_new_op(net, init_pcc_pq, minmax_v=[0.95, 1.05], max_l=100):
    """
          Calculate FA using previous FA estimation for different operating conditions.
          This function calculates the flexibility area (FA) using previous FA estimation for different
          operating conditions.

          :param net: The distribution network.
          :type net: pandapower.network

          :param init_pcc_pq: Initial values for power (P) and reactive power (Q) of the PCC.
          :type init_pcc_pq: tuple

          :param minmax_v: The scenario network voltage constraints.
          :type minmax_v: list[float, float]

          :param max_l: The scenario network loading constraints.
          :type max_l: float

          :return: The flexibility area (FA), the uncertainty including flexibility area, the reachable area,
                   the binary flexibility area, the p index of initial operating point, the q index of initial
                   operating point, a string on the simulation duration, extra info depending on scenario.
          :rtype: pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, float, float, str,
                  pandas.Dataframe
          """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    #  -----------------------------------------Preparation------------------------------------------------------------
    t0_s = time.time()
    init_v, init_load = get_init_net_state(net)
    bus_nm, line_nm, trafo_nm = get_bus_line_and_trafo_names(net)
    t0_e = time.time()
    # ------------------------Load FSP effective dict------------------------------------------------------------------
    with open("./FSP_effective_per_Comp.json", "r") as fp:
        old_vals = json.load(fp)
    fsp_effective_per_comp = old_vals['Effective']
    fsp_ineffective_per_comp = old_vals['Non-Effective']
    comp_shifts = old_vals['Shifts']
    ignore_comps = []
    for comp in init_v:
        if comp in comp_shifts.keys():
            if minmax_v[0]-comp_shifts[comp][0] <= init_v[comp] <= minmax_v[1]-comp_shifts[comp][1]:
                ignore_comps.append(comp)
    for comp in init_load:
        if comp in comp_shifts.keys():
            if max_l >= abs(init_load[comp]+comp_shifts[comp][1]) and max_l >= abs(init_load[comp]+comp_shifts[comp][0]):
                ignore_comps.append(comp)

    p_shifts = np.load("./p_axis.npy", allow_pickle=True)
    q_shifts = np.load("./q_axis.npy", allow_pickle=True)

    mat_dicts = np.load("./mat_dicts.npy", allow_pickle=True).item()
    pq_mat = torch.Tensor(np.load("./pq_mat.npy", allow_pickle=True))
    t2_e = time.time()
    # ------------------------ Apply convolutions-----------------------------------------------------------------------
    per_comp_flex = []
    # Find a way to remove components with min and max = 0 (0 std)
    for key in tqdm(fsp_effective_per_comp, desc="Network Components FAs"):
        if fsp_effective_per_comp[key] and key not in ignore_comps:
            # Apply all tensor convolutions and get 2D binary flexibility area
            conv_key = get_multi_conv_key_adapting_new_op(key, bus_nm, init_v, init_load,
                                                          min_v=minmax_v[0], max_v=minmax_v[1])
            if fsp_ineffective_per_comp[key]:
                # Convolute 2D tensor-conv flex area with the innefective FSPs
                conv_key = get_non_effective_multi_conv(fsp_ineffective_per_comp[key], conv_key, mat_dicts)
                per_comp_flex.append(conv_key)
            else:
                per_comp_flex.append(conv_key)

    if not per_comp_flex:
        if fsp_effective_per_comp[key]:
            conv_key = mat_dicts[fsp_effective_per_comp[key].pop(0)]['PQ'].cpu()
            for fsp in fsp_effective_per_comp[key]:
                conv_key = signal.convolve2d(conv_key, mat_dicts[fsp]['PQ'].cpu())
            conv_key = np.flipud(np.fliplr(conv_key))
            if fsp_ineffective_per_comp[key]:
                conv_key = get_non_effective_multi_conv(fsp_ineffective_per_comp[key], conv_key, mat_dicts)
                per_comp_flex.append(conv_key)
            else:
                per_comp_flex.append(conv_key)
        else:
            pq_mat = mat_dicts[fsp_ineffective_per_comp[key].pop(0)]['PQ'].cpu()
            for fsp in fsp_ineffective_per_comp[key]:
                pq_mat = signal.convolve2d(pq_mat, mat_dicts[fsp]['PQ'].cpu())
            conv_key = np.flipud(np.fliplr(pq_mat))
            per_comp_flex.append(conv_key)
    final_flex = per_comp_flex[0]
    for arr in per_comp_flex[1:]:
        final_flex = np.minimum(final_flex, arr)
    final_flex[final_flex < 0.5] = 0
    final_flex *= 1. / final_flex.max()
    ff_one = final_flex.copy()
    ff_one[ff_one != 0] = 1
    final_flex += ff_one
    # Apply axes
    t6_e = time.time()
    q_axes = [q_vl + float(init_pcc_pq[1]) for q_vl in q_shifts]
    p_axes = [p_vl + float(init_pcc_pq[0]) for p_vl in p_shifts]
    q_axis = q_axes[::-1]
    p_axis = p_axes[::-1]

    inf = pd.DataFrame(pq_mat, index=q_axis, columns=p_axis)
    conv_total = pd.DataFrame(final_flex, index=q_axis, columns=p_axis)
    q_index, _ = find_value_close2list(q_axis, float(init_pcc_pq[1]))
    p_index, _ = find_value_close2list(p_axis, float(init_pcc_pq[0]))
    #conv_total.loc[q_axis[q_index], p_axis[p_index]] = 2
    t7_e = time.time()
    t8_e = time.time()
    dur_str = f"Duration distribution: \n    -Preparation = {t0_e - t0_s}s,\n    -Load Data = {t2_e - t0_e}s,\n"\
              f"    -(Tensor & 2D) Convolutions = {t6_e - t2_e}s,\n"\
              f"    -Applying Axes and Init Point = {t7_e - t6_e}s,\n"\
              f"    -Small FSP Uncertainty calculation = {t8_e - t7_e}s"
    return conv_total, None, None, inf, q_index, p_index, dur_str


def get_multi_conv_key_adapting_new_op(comp, bus_nm, init_v, init_load, min_v, max_v):
    """
          Calculate flexibility region for component from sensitive FSPs
          using previous FA estimation for different operating conditions.
          This function calculates the flexibility region for component from sensitive FSPs
          using previous FA estimation for different operating conditions.

          :param comp: The distribution network component name (e.g.bus 1).
          :type comp: str

          :param bus_nm: Distribution network bus names.
          :type bus_nm: list[str]

          :param init_v: Initial voltage magnitudes of network buses.
          :type init_v: numpy.ndarray/list

          :param init_load: Initial loading of network lines/transformers.
          :type init_load: numpy.ndarray/list

          :param min_v: The scenario network minimum voltage constraints.
          :type min_v: float

          :param max_v: The scenario network maximum voltage constraints.
          :type max_v: float

          :return: The flexibility region from FSPs in which the component is sensitive.
          :rtype: numpy.array
          """
    tot_conv_key = torch.load(f'./ttd{comp}.pt').torch()
    ones_conv_key = torch.load(f'./ttd_ones{comp}.pt').torch()

    if comp in bus_nm:
        mask = (max_v >= tot_conv_key + init_v[comp]) & (tot_conv_key + init_v[comp] >= min_v)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    else:
        mask = (100 >= tot_conv_key + init_load[comp]) & (tot_conv_key + init_load[comp] >= -100)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    dims = len(tot_conv_key.shape)
    alphabet = string.ascii_lowercase[:dims]
    if len(alphabet) == 2:
        final_mat = ones_mat#np.flipud(np.fliplr(ones_mat))
    else:
        final_mat = np.flipud(np.fliplr(np.einsum(f'{alphabet}->ab', ones_mat)))
    return final_mat


def get_multi_conv_torch(fsps_of_comp, mat_dicts, comp, bus_nm, init_v, init_load, max_v=1.05, min_v=0.95):
    """
          Calculate flexibility region for component from sensitive FSPs.
          This function calculates the flexibility region for component from sensitive FSPs.

          :param fsps_of_comp: The FSPs in which the component is sensitive.
          :type fsps_of_comp: list[str]

          :param mat_dicts: The dictionary of power flow sensitivities from each FSP per component.
          :type mat_dicts: dict

          :param comp: The distribution network component name (e.g.bus 1).
          :type comp: str

          :param bus_nm: Distribution network bus names.
          :type bus_nm: list[str]

          :param init_v: Initial voltage magnitudes of network buses.
          :type init_v: numpy.ndarray/list

          :param init_load: Initial loading of network lines/transformers.
          :type init_load: numpy.ndarray/list

          :param min_v: The scenario network minimum voltage constraints.
          :type min_v: float

          :param max_v: The scenario network maximum voltage constraints.
          :type max_v: float

          :return: The flexibility region from FSPs in which the component is sensitive.
          :rtype: numpy.array
          """
    tot_conv_key = None
    for idx, fsp in enumerate(fsps_of_comp):
        pre_fsp = fsps_of_comp[:idx]
        post_fsp = fsps_of_comp[idx+1:]
        if pre_fsp:
            conv_key = mat_dicts[pre_fsp.pop(0)]['PQ']
            while pre_fsp:
                conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[pre_fsp.pop(0)]['PQ'])
            conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[fsp][comp])
        else:
            conv_key = mat_dicts[fsp][comp]
        while post_fsp:
            conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[post_fsp.pop(0)]['PQ'])
        if tot_conv_key is not None:
            tot_conv_key = torch.add(tot_conv_key, conv_key)
        else:
            tot_conv_key = conv_key
        if idx == 0:
            ones_conv_key = mat_dicts[fsps_of_comp[0]]['PQ']
            for fsp in fsps_of_comp[1:]:
                ones_conv_key = tensor_convolve_nd_torch(ones_conv_key, mat_dicts[fsp]['PQ'])

    if comp in bus_nm:
        mask = (max_v >= tot_conv_key + init_v[comp]) & (tot_conv_key + init_v[comp] >= min_v)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    else:
        mask = (100 >= tot_conv_key + init_load[comp]) & (tot_conv_key + init_load[comp] >= -100)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    dims = len(tot_conv_key.shape)
    alphabet = string.ascii_lowercase[:dims]
    if len(alphabet) == 2:
        final_mat = np.flipud(np.fliplr(ones_mat))
    else:
        final_mat = np.flipud(np.fliplr(np.einsum(f'{alphabet}->ab', ones_mat)))
    return final_mat


def get_multi_conv_torch_split(fsps_of_comp, mat_dicts, comp, bus_nm, init_v, init_load, sim_dict, init_ids,
                               max_v=1.05, min_v=0.95, no_max=5):
    """
          Calculate flexibility region for component from sensitive FSPs, but merge FSPs until maximum fsps are
          no_max-1.
          This function calculates the flexibility region for component from sensitive FSPs, but merge FSPs until
          maximum fsps are no_max-1.

          :param fsps_of_comp: The FSPs in which the component is sensitive.
          :type fsps_of_comp: list[str]

          :param mat_dicts: The dictionary of power flow sensitivities from each FSP per component.
          :type mat_dicts: dict

          :param comp: The distribution network component name (e.g.bus 1).
          :type comp: str

          :param bus_nm: Distribution network bus names.
          :type bus_nm: list[str]

          :param init_v: Initial voltage magnitudes of network buses.
          :type init_v: numpy.ndarray/list

          :param init_load: Initial loading of network lines/transformers.
          :type init_load: numpy.ndarray/list

          :param sim_dict: Estimated electrical dinstance per FSP pair.
          :type sim_dict: dictionary

          :param init_ids: Initial P,Q values per FSP.
          :type init_ids: dictionary

          :param min_v: The scenario network minimum voltage constraints.
          :type min_v: float

          :param max_v: The scenario network maximum voltage constraints.
          :type max_v: float

          :param no_max: The maximum+1 FSPs that can be accounted for in the tensors.
          :type no_max: int

          :return: The flexibility region from FSPs in which the component is sensitive.
          :rtype: numpy.array
          """
    tot_conv_key = None
    new_pairs = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    while len(fsps_of_comp) >= no_max:
        min_pair = []
        min_dist = np.inf
        for idx, fsp in enumerate(fsps_of_comp):
            if fsp in list(sim_dict.keys()):
                if sim_dict[fsp]["min"][0] < min_dist and sim_dict[fsp]["min"][1] in fsps_of_comp:
                    min_dist = sim_dict[fsp]["min"][0]
                    min_pair = [fsp, sim_dict[fsp]["min"][1]]
                elif sim_dict[fsp]["min"][0] < min_dist:
                    for pair in new_pairs:
                        spl_txt = pair.split(":+:")
                        for pre_comb_fsp in spl_txt:
                            if pre_comb_fsp == sim_dict[fsp]["min"][1]:
                                min_pair = [fsp, pair]
                                min_dist = sim_dict[fsp]["min"][0]
        conv_merge0 = tensor_convolve_nd_torch(mat_dicts[min_pair[1]][comp], mat_dicts[min_pair[0]]['PQ'])
        conv_merge1 = tensor_convolve_nd_torch(mat_dicts[min_pair[1]]['PQ'], mat_dicts[min_pair[0]][comp])
        conv_merge = conv_merge0 + conv_merge1

        dims = len(conv_merge.shape)
        alphabet = string.ascii_lowercase[:dims]
        conv_merge = torch.einsum(f'{alphabet}->ab', conv_merge)

        delta_mat1 = torch.zeros_like(mat_dicts[min_pair[1]]['PQ'])
        delta_mat1[init_ids[min_pair[1]][0], init_ids[min_pair[1]][1]] = 1
        tmp1 = torch.clone(mat_dicts[min_pair[0]]['PQ'])
        tmp2 = delta_mat1
        extra_mat_tt1 = torch.Tensor(signal.convolve2d(tmp1.cpu(), tmp2.cpu()))
        delta_mat2 = torch.zeros_like(mat_dicts[min_pair[0]]['PQ'])
        delta_mat2[init_ids[min_pair[0]][0], init_ids[min_pair[0]][1]] = 1
        tmp1 = torch.clone(mat_dicts[min_pair[1]]['PQ'])
        tmp3 = delta_mat2
        extra_mat_tt2 = torch.Tensor(signal.convolve2d(tmp1.cpu(), tmp3.cpu()))
        extra_mat_tt = extra_mat_tt1 + extra_mat_tt2
        extra_mat_tt[extra_mat_tt > 0.5] = 1

        init_ids[str(min_pair[0])+":+:"+str(min_pair[1])] = \
            list(np.transpose(np.nonzero(signal.convolve2d(tmp2.cpu(), tmp3.cpu())))[0])
        tmp1 = torch.clone(mat_dicts[min_pair[0]]['PQ'])
        tmp2 = torch.clone(mat_dicts[min_pair[1]]['PQ'])

        pq_mat_tt = torch.Tensor(signal.convolve2d(tmp1.cpu(), tmp2.cpu())).to(device)

        new_mat = torch.div(conv_merge.to(device), pq_mat_tt.to(device)+extra_mat_tt.to(device))
        new_mat = torch.nan_to_num(new_mat)

        new_fsps_of_comp = []
        for fspp in fsps_of_comp:
            if fspp not in min_pair:
                new_fsps_of_comp.append(fspp)
        new_fsps_of_comp.append(str(min_pair[0])+":+:"+str(min_pair[1]))
        fsps_of_comp = new_fsps_of_comp
        pq_new_matt = pq_mat_tt.clone()
        pq_new_matt[pq_mat_tt > 0.1] = 1
        mat_dicts[str(min_pair[0])+":+:"+str(min_pair[1])] = {'PQ': pq_new_matt}
        mat_dicts[str(min_pair[0])+":+:"+str(min_pair[1])][comp] = new_mat
        new_pairs[str(min_pair[0])+":+:"+str(min_pair[1])] = pq_mat_tt

    for idx, fsp in enumerate(fsps_of_comp):
        pre_fsp = fsps_of_comp[:idx]
        post_fsp = fsps_of_comp[idx+1:]
        if pre_fsp:
            conv_key = mat_dicts[pre_fsp.pop(0)]['PQ']
            while pre_fsp:
                conv_key = tensor_convolve_nd_torch_half(conv_key, mat_dicts[pre_fsp.pop(0)]['PQ'])
            conv_key = tensor_convolve_nd_torch_half(conv_key, mat_dicts[fsp][comp])
        else:
            conv_key = mat_dicts[fsp][comp]
        while post_fsp:
            conv_key = tensor_convolve_nd_torch_half(conv_key, mat_dicts[post_fsp.pop(0)]['PQ'])
        if tot_conv_key is not None:
            tot_conv_key = torch.add(tot_conv_key, conv_key)
        else:
            tot_conv_key = conv_key
        if idx == 0:
            ones_conv_key = mat_dicts[fsps_of_comp[0]]['PQ']
            for fsp in fsps_of_comp[1:]:
                if fsp not in list(new_pairs.keys()):
                    ones_conv_key = tensor_convolve_nd_torch_half(ones_conv_key, mat_dicts[fsp]['PQ'])
                else:
                    ones_conv_key = tensor_convolve_nd_torch_half(ones_conv_key, new_pairs[fsp])

    if comp in bus_nm:
        mask = (max_v >= tot_conv_key + init_v[comp]) & (tot_conv_key + init_v[comp] >= min_v)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    else:
        mask = (100 >= tot_conv_key + init_load[comp]) & (tot_conv_key + init_load[comp] >= -100)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    dims = len(tot_conv_key.shape)
    alphabet = string.ascii_lowercase[:dims]
    if len(alphabet) == 2:
        final_mat = np.flipud(np.fliplr(ones_mat))
    else:
        final_mat = np.flipud(np.fliplr(np.einsum(f'{alphabet}->ab', ones_mat)))
    return final_mat


def get_multi_conv_key_saving(fsps_of_comp, mat_dicts, comp, bus_nm, init_v, init_load, max_v=1.05, min_v=0.95):
    """
          Calculate flexibility region for component from sensitive FSPs, perform TTD on the tensors and save them
          locally.

          This function calculates the flexibility region for component from sensitive FSPs, perform TTD on the
          tensors and save them locally.

          :param fsps_of_comp: The FSPs in which the component is sensitive.
          :type fsps_of_comp: list[str]

          :param mat_dicts: The dictionary of power flow sensitivities from each FSP per component.
          :type mat_dicts: dict

          :param comp: The distribution network component name (e.g.bus 1).
          :type comp: str

          :param bus_nm: Distribution network bus names.
          :type bus_nm: list[str]

          :param init_v: Initial voltage magnitudes of network buses.
          :type init_v: numpy.ndarray/list

          :param init_load: Initial loading of network lines/transformers.
          :type init_load: numpy.ndarray/list

          :param min_v: The scenario network minimum voltage constraints.
          :type min_v: float

          :param max_v: The scenario network maximum voltage constraints.
          :type max_v: float

          :return: The flexibility region from FSPs in which the component is sensitive.
          :rtype: numpy.array
          """
    tot_conv_key = None
    for idx, fsp in enumerate(fsps_of_comp):
        pre_fsp = fsps_of_comp[:idx]
        post_fsp = fsps_of_comp[idx+1:]
        if pre_fsp:
            conv_key = mat_dicts[pre_fsp.pop(0)]['PQ']
            while pre_fsp:
                conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[pre_fsp.pop(0)]['PQ'])
            conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[fsp][comp])
        else:
            conv_key = mat_dicts[fsp][comp]
        while post_fsp:
            conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[post_fsp.pop(0)]['PQ'])
        if tot_conv_key is not None:
            tot_conv_key = torch.add(tot_conv_key, conv_key)
        else:
            tot_conv_key = conv_key
        if idx == 0:
            ones_conv_key = mat_dicts[fsps_of_comp[0]]['PQ']
            for fsp in fsps_of_comp[1:]:
                ones_conv_key = tensor_convolve_nd_torch(ones_conv_key, mat_dicts[fsp]['PQ'])

    tmp = np.float32(tot_conv_key.cpu())
    tensor_save = tn.Tensor(tmp, eps=1e-3)  # A tensor train decomposition
    torch.save(tensor_save, f'./ttd{comp}.pt')
    tmp = np.float32(ones_conv_key.cpu())
    tensor_save = tn.Tensor(tmp, eps=1e-3)  # A tensor train decomposition
    torch.save(tensor_save, f'./ttd_ones{comp}.pt')

    if comp in bus_nm:
        mask = (max_v >= tot_conv_key + init_v[comp]) & (tot_conv_key + init_v[comp] >= min_v)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    else:
        mask = (100 >= tot_conv_key + init_load[comp]) & (tot_conv_key + init_load[comp] >= -100)
        ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
    dims = len(tot_conv_key.shape)
    alphabet = string.ascii_lowercase[:dims]
    if len(alphabet) == 2:
        final_mat = np.flipud(np.fliplr(ones_mat))
    else:
        final_mat = np.flipud(np.fliplr(np.einsum(f'{alphabet}->ab', ones_mat)))
    return final_mat


def get_multi_conv_key_with_delta(fsps_of_comp, mat_dicts, comp, bus_nm, init_v, init_load, comp_non_lin_fsp, pq_steps,
                                  max_v=1.1, min_v=0.9):
    """
          Calculate flexibility region for component from sensitive FSPs, while having discrete FSPs.
          This function calculates the flexibility region for component from sensitive FSPs, while having discrete FSPs.

          :param fsps_of_comp: The FSPs in which the component is sensitive.
          :type fsps_of_comp: list[str]

          :param mat_dicts: The dictionary of power flow sensitivities from each FSP per component.
          :type mat_dicts: dict

          :param comp: The distribution network component name (e.g.bus 1).
          :type comp: str

          :param bus_nm: Distribution network bus names.
          :type bus_nm: list[str]

          :param init_v: Initial voltage magnitudes of network buses.
          :type init_v: numpy.ndarray/list

          :param init_load: Initial loading of network lines/transformers.
          :type init_load: numpy.ndarray/list

          :param comp_non_lin_fsp: Non linear FSPs for which the component is sensitive.
          :type comp_non_lin_fsp: list

          :param pq_steps: Non linear FSPs P,Q setpoints.
          :type pq_steps: numpy.ndarray/list

          :param min_v: The scenario network minimum voltage constraints.
          :type min_v: float

          :param max_v: The scenario network maximum voltage constraints.
          :type max_v: float

          :return: The flexibility region from FSPs in which the component is sensitive.
          :rtype: numpy.array
          """
    tot_conv_key = None
    for idx, fsp in enumerate(fsps_of_comp):
        pre_fsp = fsps_of_comp[:idx]
        post_fsp = fsps_of_comp[idx+1:]
        if pre_fsp:
            conv_key = mat_dicts[pre_fsp.pop(0)]['PQ']
            while pre_fsp:
                conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[pre_fsp.pop(0)]['PQ'])
            conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[fsp][comp])
        else:
            conv_key = mat_dicts[fsp][comp]
        while post_fsp:
            conv_key = tensor_convolve_nd_torch(conv_key, mat_dicts[post_fsp.pop(0)]['PQ'])
        if tot_conv_key is not None:
            tot_conv_key = torch.add(tot_conv_key, conv_key)
        else:
            tot_conv_key = conv_key
        if idx == 0:
            ones_conv_key = mat_dicts[fsps_of_comp[0]]['PQ']
            for fsp in fsps_of_comp[1:]:
                ones_conv_key = tensor_convolve_nd_torch(ones_conv_key, mat_dicts[fsp]['PQ'])

    if len(comp_non_lin_fsp) == 0:
        if comp in bus_nm:
            mask = (max_v >= tot_conv_key + init_v[comp]) & (tot_conv_key + init_v[comp] >= min_v)
            ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
        else:
            mask = (100 >= tot_conv_key + init_load[comp]) & (tot_conv_key + init_load[comp] >= -100)
            ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
        dims = len(tot_conv_key.shape)
        alphabet = string.ascii_lowercase[:dims]
        if len(alphabet) == 2:
            final_mat = np.flipud(np.fliplr(ones_mat))
        else:
            final_mat = np.flipud(np.fliplr(np.einsum(f'{alphabet}->ab', ones_mat)))
        return final_mat
    else:
        shapes = []
        for idx, fsp in enumerate(comp_non_lin_fsp):
            mat_list = mat_dicts[fsp]
            for shift in mat_list:
                if comp in bus_nm:
                    mask = (max_v-shift[comp] >= tot_conv_key + init_v[comp]) \
                           & (tot_conv_key + init_v[comp] >= min_v-shift[comp])
                    ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
                else:
                    mask = (100-shift[comp] >= tot_conv_key + init_load[comp]) \
                           & (tot_conv_key + init_load[comp] >= -100-shift[comp])
                    ones_mat = torch.mul(torch.where(mask, 1, 0), ones_conv_key).cpu().numpy()
                dims = len(tot_conv_key.shape)
                alphabet = string.ascii_lowercase[:dims]
                if len(alphabet) == 2:
                    shapes.append([float(shift['dp']), float(shift['dq']), np.flipud(np.fliplr(ones_mat))])
                else:
                    shapes.append([float(shift['dp']), float(shift['dq']),
                                   np.flipud(np.fliplr(np.einsum(f'{alphabet}->ab', ones_mat)))])
        return combine_shapes(shapes, pq_steps)


def combine_shapes(shapes, pq_steps):
    """
          Combine the feasible regions for different discontinuous area setpoints.
          This function combines the feasible regions for different discontinuous area setpoints.

          :param shapes: The flexibility regions for different setpoints of the non-linear FSP.
          :type shapes: list[str]

          :param pq_steps: Non linear FSPs P,Q setpoints.
          :type pq_steps: numpy.ndarray/list

          :return: The combined flexibility region.
          :rtype: numpy.array
          """
    #                   +  -
    generic_p_shifts = [0, 0]
    generic_q_shifts = [0, 0]
    # Find total padding of pixels (gen_p_sh[0] = right, gen_p_sh[1] = left, gen_q_sh[0] = top, gen_q_sh[1] = bottom)
    for shape in shapes:
        p_pixels = round(shape[0]/pq_steps[0])
        q_pixels = round(shape[1]/pq_steps[1])
        if p_pixels > 0:
            generic_p_shifts[0] += p_pixels
        else:
            generic_p_shifts[1] += p_pixels
        if q_pixels > 0:
            generic_q_shifts[0] += q_pixels
        else:
            generic_q_shifts[1] += q_pixels
    padded_arrays = []
    for shape in shapes:
        specific_p_shifts = [generic_p_shifts[0] - round(shape[0]/pq_steps[0]),
                             -generic_p_shifts[1] + round(shape[0]/pq_steps[0])]
        specific_q_shifts = [generic_q_shifts[0] - round(shape[1]/pq_steps[1]),
                             -generic_q_shifts[1] + round(shape[1]/pq_steps[1])]
        padded_arrays.append(np.pad(shape[2], ((specific_q_shifts[0], specific_q_shifts[1]),
                                               (specific_p_shifts[0], specific_p_shifts[1]))))
    padded_arrays = np.array(padded_arrays)
    return np.einsum('ijk->jk', padded_arrays)


def combine_shapes_const_flex(shapes, pq_steps, flex):
    """
          Combine the feasible regions for different discontinuous area setpoints, in case where no network
          constraint can be reached.
          This function combines the feasible regions for different discontinuous area setpoints, in case where no
          network constraint can be reached.

          :param shapes: The flexibility regions for different setpoints of the non-linear FSP.
          :type shapes: list[str]

          :param pq_steps: Non linear FSPs P,Q setpoints.
          :type pq_steps: numpy.ndarray/list

          :param flex: Flexibility region linear FSPs.
          :type flex: numpy.ndarray/list

          :return: The combined flexibility region.
          :rtype: numpy.array
          """
    #                   +  -
    generic_p_shifts = [0, 0]
    generic_q_shifts = [0, 0]
    # Find total padding of pixels (gen_p_sh[0] = right, gen_p_sh[1] = left, gen_q_sh[0] = top, gen_q_sh[1] = bottom)
    for shape in shapes:
        p_pixels = round(shape[0]/pq_steps[0])
        q_pixels = round(shape[1]/pq_steps[1])
        if p_pixels > 0:
            generic_p_shifts[0] += p_pixels
        else:
            generic_p_shifts[1] += p_pixels
        if q_pixels > 0:
            generic_q_shifts[0] += q_pixels
        else:
            generic_q_shifts[1] += q_pixels
    padded_arrays = []
    for shape in shapes:
        specific_p_shifts = [generic_p_shifts[0] - round(shape[0]/pq_steps[0]),
                             -generic_p_shifts[1] + round(shape[0]/pq_steps[0])]
        specific_q_shifts = [generic_q_shifts[0] - round(shape[1]/pq_steps[1]),
                             -generic_q_shifts[1] + round(shape[1]/pq_steps[1])]
        if shape[2] == 1:
            padded_arrays.append(np.pad(flex, ((specific_q_shifts[1], specific_q_shifts[0]),
                                               (specific_p_shifts[1], specific_p_shifts[0]))))
        else:
            padded_arrays.append(np.pad(np.zeros_like(flex), ((specific_q_shifts[1], specific_q_shifts[0]),
                                                             (specific_p_shifts[1], specific_p_shifts[0]))))
    padded_arrays = np.array(padded_arrays)
    tmp = np.einsum('ijk->jk', padded_arrays)
    return np.flipud(np.fliplr(tmp))


def get_multi_result_for_0_effective(non_effective_fsps, mat_dicts):
    """
          Estimate flexibility area if no component can reach its constraint.
          This function estimated flexibility area, in case where no
          network constraint can be reached.

          :param non_effective_fsps: The FSP names.
          :type non_effective_fsps: list[str]

          :param mat_dicts: The power flow results.
          :type mat_dicts: dict

          :return: The flexibility area.
          :rtype: numpy.array
          """
    pq_mat = mat_dicts[non_effective_fsps.pop(0)]['PQ'].cpu()
    for fsp in non_effective_fsps:
        pq_mat = signal.convolve2d(pq_mat, mat_dicts[fsp]['PQ'].cpu())
    return np.flipud(np.fliplr(pq_mat))


def get_non_effective_multi_conv(non_effective_fsps, effective_conv, mat_dicts):
    """
          Estimate flexibility area by convolving non-effective fsps with the region from the effective fsps.
          This function estimates flexibility area by convolving non-effective fsps with the region
          from the effective fsps.

          :param non_effective_fsps: The non-effective FSP names.
          :type non_effective_fsps: list[str]

          :param effective_conv: The effective FSP flexibility area.
          :type effective_conv: numpy.array

          :param mat_dicts: The power flow results.
          :type mat_dicts: dict

          :return: The flexibility area.
          :rtype: numpy.array
          """
    pq_mat = mat_dicts[non_effective_fsps.pop(0)]['PQ'].cpu()
    for fsp in non_effective_fsps:
        pq_mat = signal.convolve2d(pq_mat, mat_dicts[fsp]['PQ'].cpu())
    return signal.convolve2d(np.flipud(np.fliplr(pq_mat)), effective_conv)


def get_non_effective_multi_conv_with_delta(non_effective_fsps, effective_conv, mat_dicts, non_eff_non_lin_fsps,
                                            pq_steps):
    """
          Estimate flexibility area by convolving non-effective fsps with the region from the effective fsps,
          also accounting for non-effective non-linear FSPs.
          This function estimates flexibility area by convolving non-effective fsps with the region
          from the effective fsps.

          :param non_effective_fsps: The non-effective FSP names.
          :type non_effective_fsps: list[str]

          :param effective_conv: The effective FSP flexibility area.
          :type effective_conv: numpy.array

          :param mat_dicts: The power flow results.
          :type mat_dicts: dict

          :param non_eff_non_lin_fsps: The non-effective non-linear FSP names.
          :type non_eff_non_lin_fsps: list[str]

          :param pq_steps: The P,Q setpoints of the non-effective non-linear FSP names.
          :type pq_steps: dict

          :return: The flexibility area.
          :rtype: numpy.array
          """
    pq_mat = mat_dicts[non_effective_fsps.pop(0)]['PQ']
    for fsp in non_effective_fsps:
        pq_mat = signal.convolve2d(pq_mat, mat_dicts[fsp]['PQ'])
    shapes = []
    for fsp in non_eff_non_lin_fsps:
        mat_list = mat_dicts[fsp]
        for shift in mat_list:
            shapes.append([float(shift['dp']), float(shift['dq']), pq_mat])
    if len(shapes) > 0:
        final_mat = combine_shapes(shapes, pq_steps)
        return signal.convolve2d(np.flipud(np.fliplr(final_mat)), effective_conv)
    else:
        return signal.convolve2d(np.flipud(np.fliplr(pq_mat)), effective_conv)


def get_result_for_0_effective_with_delta(final_flex, non_lin_eff_fsp_per_comp, non_lin_ineff_fsp_per_comp, pq_steps,
                                          bus_nm, init_v, init_l, mat_dicts, max_v, min_v):
    """
          Estimate flexibility area if no component can reach its constraint, and non-linear FSPs exist.
          This function estimated flexibility area, in case where no.
          network constraint can be reached.

          :param final_flex: The linear FSP flexibility area.
          :type final_flex: numpy.array

          :param non_lin_eff_fsp_per_comp: The  non-linear but effective FSP names.
          :type non_lin_eff_fsp_per_comp: dict

          :param non_lin_ineff_fsp_per_comp: The non-linear non-effective FSP names.
          :type non_lin_ineff_fsp_per_comp: dict

          :param pq_steps: The P,Q setpoints of the non-effective non-linear FSP names.
          :type pq_steps: dict

          :param bus_nm: The network bus names.
          :type bus_nm: list[str]

          :param init_v: The network bus initial voltage magnitudes.
          :type init_v: dict

          :param init_l: The network line/transformer initial loading.
          :type init_l: dict

          :param mat_dicts: The power flow results.
          :type mat_dicts: dict

          :param max_v: The network constraint maximum voltage magnitudes.
          :type max_v: dict

          :param min_v: The network constraint minimum voltage magnitudes.
          :type min_v: dict

          :return: The flexibility area.
          :rtype: numpy.array
          """
    flexes = []
    for comp in non_lin_eff_fsp_per_comp:
        shapes = []
        if non_lin_eff_fsp_per_comp[comp]:
            for fsp in non_lin_eff_fsp_per_comp[comp]:
                mat_list = mat_dicts[fsp]
                for shift in mat_list:
                    if comp in bus_nm:
                        if max_v > init_v[comp] + shift[comp] > min_v:
                            shapes.append([float(shift['dp']), float(shift['dq']), 1])
                        else:
                            shapes.append([float(shift['dp']), float(shift['dq']), 0])
                    else:
                        if 100 > init_l[comp] + shift[comp]:
                            shapes.append([float(shift['dp']), float(shift['dq']), 1])
                        else:
                            shapes.append([float(shift['dp']), float(shift['dq']), 0])
            for fsp in non_lin_ineff_fsp_per_comp[comp]:
                mat_list = mat_dicts[fsp]
                for shift in mat_list:
                    shapes.append([float(shift['dp']), float(shift['dq']), 1])
            if len(shapes) > 0:
                flexes.append(combine_shapes_const_flex(shapes, pq_steps, final_flex))
    if len(flexes) == 0:
        shapes = []
        for fsp in non_lin_ineff_fsp_per_comp[list(non_lin_ineff_fsp_per_comp.keys())[0]]:
            mat_list = mat_dicts[fsp]
            for shift in mat_list:
                shapes.append([float(shift['dp']), float(shift['dq']), 1])
        if shapes:
            shapes = np.array(shapes)
            return np.flipud(np.fliplr(combine_shapes_const_flex(shapes, pq_steps, final_flex)))
        else:
            return np.flipud(np.fliplr(final_flex))
    else:
        flexes = np.array(flexes)
        final_flex = flexes[0]
        for arr in flexes[1:]:
            final_flex = np.minimum(final_flex, arr)
        return final_flex


def get_new_conv_axes2(list_of_rows, list_of_columns, conv_q_len, conv_p_len, q, p):
    """
          Combine axes per FSP to get the final flexibility area axes.
          This function combines axes per FSP to get the final flexibility area axes.

          :param list_of_rows: Row axes from all FSPs.
          :type list_of_rows: list

          :param list_of_columns: Column axes from all FSPs.
          :type list_of_columns: list

          :param conv_q_len: Convolution result (reachability) height length in pixels.
          :type conv_q_len: int

          :param conv_p_len: Convolution result (reachability) weight length in pixels.
          :type conv_p_len: int

          :param q: The network PCC initial q.
          :type q: float

          :param p: The network PCC initial p.
          :type p: float

          :return: The flexibility area.
          :rtype: numpy.array
          """
    min_rows = q
    max_rows = q
    for row in list_of_rows:
        min_rows += min(0., float(min(row)) - q)
        max_rows += max(0., float(max(row)) - q)
    min_cols = p
    max_cols = p
    for col in list_of_columns:
        min_cols += min(0., float(min(col)) - p)
        max_cols += max(0., float(max(col)) - p)
    rows = np.linspace(min_rows, max_rows, num=conv_q_len)
    cols = np.linspace(min_cols, max_cols, num=conv_p_len)
    return rows, cols


def run_all_tensor_flex_areas(net, pq_profiles, bus_nm, line_nm, trafo_nm, init_v, init_load):
    """
          Run power flows for all FSP setpoints given.
          This function runs power flows for given FSP setpoints.

          :param net: Distribution network.
          :type net: pandapower.network

          :param pq_profiles: Sapled setpoints per FSP.
          :type pq_profiles: numpy.array/list

          :param bus_nm: Network bus names.
          :type bus_nm: list

          :param line_nm: Network line names.
          :type line_nm: list

          :param trafo_nm: Network transformer names.
          :type trafo_nm: list

          :param init_v: The network initial voltage magnitudes for all buses.
          :type init_v: dict

          :param init_load: The network initial loading for all lines and transformers.
          :type init_load: dict

          :return: The result dictionary, standard deviation information, and duration.
          :rtype: dict, dict, float
          """
    t_start_run_mc_pf = time.time()
    result_dict = {}
    std_dict = {}
    for key in tqdm(pq_profiles, desc="FSP Flexibilities Completed"):
        fsp_type = 'None'
        fsp_idx = -1
        result_dict[key] = pd.DataFrame()
        if 'Load' not in key and 'load' not in key:
            for idx, gen in enumerate(net.sgen.iterrows()):
                if gen[1]['name'] == key:
                    fsp_idx = idx
                    fsp_type = 'Sgen'
        else:
            for idx, load in enumerate(net.load.iterrows()):
                if load[1]['name'] == key:
                    fsp_idx = idx
                    fsp_type = 'Load'
        if fsp_type == 'None':
            assert False, f"Cannot detect FSP name {key}"
        for profile in pq_profiles[key]:
            net = update_conv_pqs(net, fsp_idx=fsp_idx, fsp_type=fsp_type, profile=profile)
            try:
                pp.runpp(net, numba=False)
                tmp_dict = {'p': float(net.res_ext_grid['p_mw'].iloc[0]), 'q': float(net.res_ext_grid['q_mvar'].iloc[0])}
                for i, nm in enumerate(bus_nm):
                    tmp_dict[nm] = float(net.res_bus['vm_pu'].iloc[i]-init_v[nm])
                for i, nm in enumerate(line_nm):
                    tmp_dict[nm] = float(net.res_line['loading_percent'].iloc[i]-init_load[nm])
                for i, nm in enumerate(trafo_nm):
                    tmp_dict[nm] = float(net.res_trafo['loading_percent'].iloc[i]-init_load[nm])
                new_df_row = pd.DataFrame(tmp_dict, index=[0])
                result_dict[key] = pd.concat([new_df_row, result_dict[key].loc[:]]).reset_index(drop=True)
                #result_dict[key] = result_dict[key].append(tmp_dict, ignore_index=True)
            except:
                print(f"Power flow did not converge for profile {profile}")
        net = update_conv_pqs(net, fsp_idx=fsp_idx, fsp_type=fsp_type, profile=pq_profiles[key][0])
        std_dict[key] = {'std': result_dict[key].std().drop(['p', 'q']),
                         'min': result_dict[key].min().drop(['p', 'q']),
                         'max': result_dict[key].max().drop(['p', 'q'])}
    t_stop_run_mc_pf = time.time()

    print(f"Power flows needed {t_stop_run_mc_pf - t_start_run_mc_pf} seconds")
    return result_dict, std_dict, t_stop_run_mc_pf - t_start_run_mc_pf


def df_to_mat_tensor_scaled_and_init(df, dp, dq):
    """
          Take dataframes of power flow results and create matrices of feasible regions, sensitivities
          for tensor convolutions.
          This function takes dataframes of power flow results and creates matrices of feasible regions, sensitivities
          for tensor convolutions.

          :param df: Dataframe with power flow results.
          :type df: pandas.dataframe

          :param dp: Resolution in dp.
          :type dp: float

          :param dq: Resolution in dq.
          :type dq: float

          :return: The result dictionary, p axis and q axis.
          :rtype: dict, np.array, np.array
          """
    max_p = float(df['p'].max())
    min_p = float(df['p'].min())
    max_q = float(df['q'].max())
    min_q = float(df['q'].min())
    ax_p = np.arange(min_p, max_p + 1 * dp, dp)
    ax_q = np.arange(min_q, max_q + 1 * dq, dq)
    mat_dict = {}
    for val in df.columns:
        if val not in ['p', 'q']:
            mat_dict[val] = torch.zeros((len(ax_q), len(ax_p)))
    mat_dict['PQ'] = torch.zeros((len(ax_q), len(ax_p)))
    for index, row in df.iterrows():
        idx, _ = find_value_close2list(ax_p, row['p'])
        jdx, _ = find_value_close2list(ax_q, row['q'])
        mat_dict['PQ'][jdx, idx] = 1
        for key in mat_dict:
            if key not in ['PQ']:
                mat_dict[key][jdx, idx] = row[key]
    return mat_dict, ax_p, ax_q


def df_to_mat_tensor_torch(df, dp, dq):
    """
          Take dataframes of power flow results and create matrices of feasible regions, sensitivities
          for tensor convolutions.
          This function takes dataframes of power flow results and creates matrices of feasible regions, sensitivities
          for tensor convolutions.

          :param df: Dataframe with power flow results.
          :type df: pandas.dataframe

          :param dp: Resolution in dp.
          :type dp: float

          :param dq: Resolution in dq.
          :type dq: float

          :return: The result dictionary, p axis and q axis.
          :rtype: dict, np.array, np.array
          """
    max_p = float(df['p'].max())
    min_p = float(df['p'].min())
    max_q = float(df['q'].max())
    min_q = float(df['q'].min())
    ax_p = np.arange(min_p, max_p + 1 * dp, dp)
    ax_q = np.arange(min_q, max_q + 1 * dq, dq)
    mat_dict = {}
    for val in df.columns:
        if val not in ['p', 'q']:
            mat_dict[val] = torch.zeros((len(ax_q), len(ax_p)))
    mat_dict['PQ'] = torch.zeros((len(ax_q), len(ax_p)))
    for index, row in df.iterrows():
        idx, _ = find_value_close2list(ax_p, row['p'])
        jdx, _ = find_value_close2list(ax_q, row['q'])
        mat_dict['PQ'][jdx, idx] = 1
        for key in mat_dict:
            if key not in ['PQ']:
                mat_dict[key][jdx, idx] = row[key]
    for key in mat_dict:
        mat_dict[key] = fix_missing_point(mat_dict[key])
    return mat_dict, ax_p, ax_q


def df_to_mat_tensor_torchv2(df, dp, dq):
    """
          Take dataframes of power flow results and create matrices of feasible regions, sensitivities
          for tensor convolutions.
          This function takes dataframes of power flow results and creates matrices of feasible regions, sensitivities
          for tensor convolutions.

          :param df: Dataframe with power flow results.
          :type df: pandas.dataframe

          :param dp: Resolution in dp.
          :type dp: float

          :param dq: Resolution in dq.
          :type dq: float

          :return: The result dictionary, p axis and q axis.
          :rtype: dict, np.array, np.array
          """
    max_p = float(df['p'].max())
    min_p = float(df['p'].min())
    max_q = float(df['q'].max())
    min_q = float(df['q'].min())
    ax_p = np.arange(min_p, max_p + 1 * dp, dp)
    ax_q = np.arange(min_q, max_q + 1 * dq, dq)
    mat_dict = {}
    tup = (-1, -1)
    for val in df.columns:
        if val not in ['p', 'q']:
            mat_dict[val] = torch.zeros((len(ax_q), len(ax_p)))
    mat_dict['PQ'] = torch.zeros((len(ax_q), len(ax_p)))
    for index, row in df.iterrows():
        idx, _ = find_value_close2list(ax_p, row['p'])
        jdx, _ = find_value_close2list(ax_q, row['q'])
        mat_dict['PQ'][jdx, idx] = 1
        for key in mat_dict:
            if key not in ['PQ']:
                mat_dict[key][jdx, idx] = row[key]
        if index == 0:
            tup = (jdx, idx)
    for key in mat_dict:
        mat_dict[key] = fix_missing_point(mat_dict[key])
    return mat_dict, ax_p, ax_q, tup


def get_delta(df, dp, dq, init_pq):
    """
          Take dataframes of power flow results and create matrices of feasible regions, sensitivities
          for tensor convolutions, for non-linear FSPS.
          This function takes dataframes of power flow results and creates matrices of feasible regions, sensitivities
          for tensor convolutions, for non-linear FSPS.

          :param df: Dataframe with power flow results.
          :type df: pandas.dataframe

          :param dp: Resolution in dp.
          :type dp: float

          :param dq: Resolution in dq.
          :type dq: float

          :param init_pq: Initial P,Q values.
          :type init_pq: list[float, float]

          :return: The result list, p axis, q axis, and binary matrix.
          :rtype: list, np.array, np.array, np.array
          """
    max_p = float(df['p'].max())
    min_p = float(df['p'].min())
    max_q = float(df['q'].max())
    min_q = float(df['q'].min())
    #ax_p = np.arange(min_p, max_p + 1 * dp, dp)
    #ax_q = np.arange(min_q, max_q + 1 * dq, dq)
    ax_p = np.arange(min_p, max_p + 0.5 * dp, dp)
    ax_q = np.arange(min_q, max_q + 0.5 * dq, dq)
    mat_list = []
    #pq_mat = np.zeros((len(ax_q), len(ax_p)))
    pq_mat = torch.zeros((len(ax_q), len(ax_p)))
    for index, row in df.iterrows():
        tmp_dict = {'dp': np.float16(row['p']-init_pq[0]), 'dq': np.float16(row['q']-init_pq[1])}
        idx, _ = find_value_close2list(ax_p, row['p'])
        jdx, _ = find_value_close2list(ax_q, row['q'])
        pq_mat[jdx, idx] = 1
        for val in df.columns:
            if val not in ['p', 'q']:
                tmp_dict[val] = np.float16(row[val])
        mat_list.append(tmp_dict)
    return mat_list, ax_p, ax_q, pq_mat


def profiles_to_mat(profiles, dp, dq, init_fsp_pq):
    """
          Take dataframes of power flow results and create matrices of feasible regions.
          This function takes dataframes of power flow results and creates matrices of feasible regions.

          :param profiles: Dataframe with power flow results.
          :type profiles: pandas.dataframe

          :param dp: Resolution in dp.
          :type dp: float

          :param dq: Resolution in dq.
          :type dq: float

          :param init_fsp_pq: Initial P,Q values for FSPs.
          :type init_fsp_pq: list[float, float]

          :return: Matrix of shifts sensitivities.
          :rtype: np.array
          """
    prof = np.array(profiles)
    max_p = float(prof[:, 0].max()-init_fsp_pq[0])
    min_p = float(prof[:, 0].min()-init_fsp_pq[0])
    max_q = float(prof[:, 1].max()-init_fsp_pq[1])
    min_q = float(prof[:, 1].min()-init_fsp_pq[1])
    ax_p = np.arange(min_p, max_p + 1 * dp, dp)
    ax_q = np.arange(min_q, max_q + 1 * dq, dq)
    mat = np.zeros((len(ax_q), len(ax_p)))
    for index, row in enumerate(prof):
        idx, _ = find_value_close2list(ax_p, row[0]-init_fsp_pq[0])
        jdx, _ = find_value_close2list(ax_q, row[1]-init_fsp_pq[1])
        mat[jdx, idx] = 1
    return mat


def find_value_close2list(lst, voi):
    """
    Find the index and closest value in a list to a specified value.
    This function searches for the index and value in the input list `lst` that is closest to the specified value `voi`.
    It iterates through the list and calculates the absolute difference between each element and `voi` to find the
    closest match.

    :param lst: The list to search for the closest value.
    :type lst: list[float]

    :param voi: The value to find the closest match for.
    :type voi: float

    :return: The index of the closest value in the list and the closest value itself.
    :rtype: int, float
    """

    # The function finds the index and value in the list `lst` that is closest to the specified value `voi`.
    # It returns a tuple containing the index and the closest value.

    # Example:
    # >>> find_value_close2list([1.0, 2.0, 3.0, 4.0, 5.0], 3.5)
    # (3, 4.0)
    dist = np.inf
    idx = 0
    for i, value in enumerate(lst):
        if abs(value - voi) < dist:
            dist = abs(value - voi)
            idx = i
    return idx, lst[idx]


def update_conv_pqs(net, fsp_idx, fsp_type, profile):
    """
    Update the FSPs to perform power flows.
    This function updates the FSPs to perform power flows.

    :param net: The distribution network.
    :type net: pandapower.network

    :param fsp_idx: The index value of the FSP in the network.
    :type fsp_idx: int

    :param fsp_type: The type of the FSP in the network.
    :type fsp_type: str

    :param profile: The p,q new setpoints for the FSP.
    :type profile: list[float, float]

    :return: Updated network.
    :rtype: pandapower.network
    """

    if fsp_type == 'Sgen':
        net.sgen['p_mw'][fsp_idx] = profile[0]
        net.sgen['q_mvar'][fsp_idx] = profile[1]
    elif fsp_type == 'OLTC':
        net.trafo['tap_pos'][fsp_idx] = profile
    else:
        net.load['p_mw'][fsp_idx] = profile[0]
        net.load['q_mvar'][fsp_idx] = profile[1]
    return net

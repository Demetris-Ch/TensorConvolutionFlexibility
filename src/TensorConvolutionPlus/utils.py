#!/usr/bin/env python
import pandas as pd
import torch
import numpy as np
import numpy.linalg as linalg
from scipy.stats import norm


__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def check_line_current_limits(net, upper_limit=100):
    """ Check if the power flow result on the network caused respects the loading limitations in all lines

    :param net: network model
    :type net: pandapower.network

    :param upper_limit: upper loading percentage limit
    :type upper_limit: int/float

    :return: True/False if the loading of any line exceeds the upper limit
    :rtype: boolean
    """
    return all(upper_limit >= abs(x) for x in net.res_line['loading_percent'])


def check_trafo_current_limits(net, upper_limit=100):
    """ Check if the power flow result on the network caused respects the loading limitations in all transformers

    :param net: network model
    :type net: pandapower.network

    :param upper_limit: upper loading percentage limit
    :type upper_limit: int/float

    :return: True/False if the loading of any line exceeds the upper limit
    :rtype: boolean
    """
    return all(upper_limit >= abs(x) for x in net.res_trafo['loading_percent'])


def check_voltage_limits(voltages, upper_limit, lower_limit):
    """ Check if the power flow result on the network caused respects the voltage limitations in all lines

    :param voltages: list of component voltages to be evaluated
    :type voltages: list of floats

    :param upper_limit: upper voltage percentage limit
    :type upper_limit: float

    :param lower_limit: lower voltage percentage limit [float]
    :type lower_limit: int/float

    :return: True/False if the voltage of any component exceeds the upper limit, or is lower than the lower limit
    :rtype: boolean
    """
    return all(upper_limit >= x >= lower_limit for x in voltages)


def update_pqs(net, flex_wt=None, flex_pv=None, profile=None,  scale_w=1, scale_pv=1):
    """ Update network DG FSP P,Q based on the input values

    :param net: network model
    :type net: pandapower network

    :param flex_wt: Indices of flexible wind turbines,
    :type flex_wt: list of int

    :param flex_pv: Indices of flexible pv,
    :type flex_pv: list of int

    :param profile: P,Q values for each FSP for one iteration on the Monte Carlo algorithm
    :type profile: list of lists of floats

    :param scale_w: If a network with different wt penetration is used, this parameter will scale the wt accordingly,
           (default=1, no scaling)
    :type scale_w: int/float

    :param scale_pv: If a network with different pv penetration is used, this parameter will scale the pv accordingly,
           (default=1, no scaling)
    :type scale_pv: int/float

    :return: updated network model
    :rtype: pandapower network
    """
    if profile:
        for i in range(0, len(profile)):
            if i == len(net.sgen)-1:
                scale = scale_w
            else:
                scale = scale_pv
            # only update sgen which are fsp
            if i in flex_pv or i in flex_wt:
                net.sgen['p_mw'][i] = profile[i][0]*scale
                net.sgen['q_mvar'][i] = profile[i][1]*scale
            else:
                net.sgen['p_mw'][i] = net.sgen['p_mw'][i]*scale
                net.sgen['q_mvar'][i] = net.sgen['p_mw'][i]*scale

    else:
        for i in range(0, len(net.sgen)):
            if i == len(net.sgen)-1:
                scale = scale_w
            else:
                scale = scale_pv
            net.sgen['p_mw'][i] = net.sgen['p_mw'][i]*scale
            net.sgen['q_mvar'][i] = net.sgen['q_mvar'][i]*scale
    return net


def update_pqs2(net, flex_dg=None, profile=None):
    """ Update network DG FSP P,Q based on the input profile

    :param net: network model,
    :type net: pandapower network

    :param flex_dg: Indices of flexible distributed generation,
    :type flex_dg: list of int

    :param profile: P,Q values for each FSP for one iteration on the Monte Carlo algorithm
    :type profile: list of list of floats

    :return: updated network model
    :rtype: pandapower network

    """
    for i in range(0, len(profile)):
        j = flex_dg[i]
        net.sgen['p_mw'][j] = profile[i][0]
        net.sgen['q_mvar'][j] = profile[i][1]
    return net


def get_input_busses_pq(net, input_buses):
    """ Get P and Q of all bus indices in {input_buses} list

    :param net: network model
    :type net: pandapower network

    :param input_buses: bus indices whose P,Q values are needed
    :type input_buses: list of int

    :return: p= list of P of busses of interest, q= list of Q of busses of interest
    :rtype: list, list
    """
    p = []
    q = []
    for bus in net.res_bus.iterrows():
        if bus[0] in input_buses:
            p.append(bus[1]['p_mw'])
            q.append(bus[1]['q_mvar'])
    return p, q


def get_input_busses_v(net, input_buses):
    """Get voltage magnitude v and angle θ of all bus indices in {input_buses} list

    :param net: network model
    :type net: pandapower network

    :param input_buses: bus indices whose v, θ values are needed
    :type input_buses: list of ints

    :return: p= list of v of busses of interest, q= list of θ of busses of interest
    :rtype: list, list
    """
    p = []
    q = []
    for bus in net.res_bus.iterrows():
        if bus[0] in input_buses:
            p.append(bus[1]['vm_pu'])
            q.append(bus[1]['va_degree'])
    return p, q


def get_input_lines_pq(net, input_lines):
    """Get P and Q of all line indices in {input_lines} list

    :param net: network model,
    :type net: pandapower network

    :param input_buses: line indices whose P,Q values are needed,
    :type input_buses: list of int

    :return: p= list of P of lines of interest, q= list of Q of lines of interest
    :rtype: list, list
    """
    p = []
    q = []
    for line in net.res_line.iterrows():
        if line[0] in input_lines:
            p.append(line[1]['p_from_mw'])
            q.append(line[1]['q_from_mvar'])
    return p, q


def update_pqs_wl(net, flex_wt=None, flex_pv=None, profile=None, scale_w=1., scale_pv=1., load_ind=[]):
    """Update network FSP P,Q including loads based on the input values

    :param net: network model
    :type net: pandapower network

    :param flex_wt: indices of flexible wind turbines
    :type flex_wt: list of int

    :param flex_pv: indices of flexible pv,
    :type flex_pv: list of int

    :param profile: P,Q values for each FSP for one iteration on the Monte Carlo algorithm
    :type profile: list of (FSPs) list of (P,Q) floats

    :param scale_w: if a network with different wt penetration is used, this parameter will scale the wt accordingly, (default=1, no scaling),
    :type scale_w: int/float

    :param scale_pv: If a network with different pv penetration is used, this parameter will scale the pv accordingly (default=1, no scaling),
    :type scale_pv: int/float

    :param load_ind: indices of flexible loads
    :type load_ind: list of int

    :return: net=updated network model
    :rtype: pandaPower network
    """
    if profile:
        for i in range(0, len(profile)):
            if i < len(net.sgen):
                if i == len(net.sgen)-1:
                    scale = scale_w
                else:
                    scale = scale_pv
                if i in flex_pv or i in flex_wt:
                    net.sgen['p_mw'][i] = profile[i][0] * scale
                    net.sgen['q_mvar'][i] = profile[i][1] * scale
                else:
                    net.sgen['p_mw'][i] = net.sgen['p_mw'][i] * scale
                    net.sgen['q_mvar'][i] = net.sgen['p_mw'][i] * scale
            else:
                j = load_ind[i-len(net.sgen)]
                net.load['p_mw'][j] = profile[i][0]
                net.load['q_mvar'][j] = profile[i][1]
    else:
        for i in range(0, len(net.sgen)):
            if i == len(net.sgen)-1:
                scale = scale_w
            else:
                scale = scale_pv
            net.sgen['p_mw'][i] = net.sgen['p_mw'][i]*scale
            net.sgen['q_mvar'][i] = net.sgen['q_mvar'][i]*scale
    return net


def update_pqs_wl2(net, profile, load_ind=[], dg_ind=[]):
    """ Update load and distributed generator output values.

    :param net: network model
    :type net: pandapower network

    :param profile: profile with values to update
    :type profile: list

    :param load_ind: indices of loads to update
    :type load_ind: list

    :param dg_ind: indices of distributed generators to update
    :type dg_ind: list

    :return: updated network
    :rtype: pandapower network
    """
    if len(dg_ind) == 0:
        dg_ind = np.arange(0, len(net.sgen))
    for i in range(0, len(profile)):
        if i < len(dg_ind):
            j = dg_ind[i]
            net.sgen['p_mw'][j] = profile[i][0]
            net.sgen['q_mvar'][j] = profile[i][1]
        elif len(load_ind) > 0:
            if len(load_ind) == 0:
                j = i - len(dg_ind)
            elif len(dg_ind) == 0:
                j = load_ind[i]
            else:
                j = load_ind[i-len(dg_ind)]
            net.load['p_mw'][j] = profile[i][0]
            net.load['q_mvar'][j] = profile[i][1]
    return net


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


def update_pqs_wl2_aliander(net, profile, load_ind=[], dg_ind=[]):
    """ Update PQ values using Aliander's PGM

    :param net: network model
    :type net: pandapower model

    :param profile: values for network components
    :type profile: list

    :param load_ind: load indices to update the values
    :type load_ind: list

    :param dg_ind: distributed generation indices to update the values
    :type dg_ind: lost

    :return: updated network model
    :rtype: pandapower network
    """
    for i in range(0, len(profile)):
        if i < len(dg_ind):
            j = dg_ind[i]
            net.sgen['p_mw'][j] = profile[i][0]
            net.sgen['q_mvar'][j] = profile[i][1]
        else:
            j = load_ind[i-len(dg_ind)]
            net.load['p_mw'][j] = profile[i][0]
            net.load['q_mvar'][j] = profile[i][1]
    return net


def write_result(x_flexible, x_non_flexible, y_flexible, y_non_flexible, name):
    """ Sve Monte Carlo simulation result on the folder

    :param x_flexible: feasible P
    :type x_flexible: list of floats

    :param x_non_flexible: infeasible P
    :type x_non_flexible: list of floats

    :param y_flexible: feasible Q
    :type y_flexible: list of floats

    :param y_non_flexible: infeasible Q
    :type y_non_flexible: list of floats

    :param name: name to be used in filename,
    :type name: str
    """
    max_len = max(len(x_flexible), len(x_non_flexible))
    x_flexible_df = np.zeros(max_len)
    x_non_flexible_df = np.zeros(max_len)
    y_flexible_df = np.zeros(max_len)
    y_non_flexible_df = np.zeros(max_len)
    for i in range(0, len(x_flexible)):
        x_flexible_df[i] = float(x_flexible[i])
        y_flexible_df[i] = float(y_flexible[i])
    for i in range(0, len(x_non_flexible)):
        x_non_flexible_df[i] = float(x_non_flexible[i])
        y_non_flexible_df[i] = float(y_non_flexible[i])
    df = pd.DataFrame(list(zip(x_flexible_df, y_flexible_df, x_non_flexible_df, y_non_flexible_df)),
                      columns=['x flex', 'y flex', 'x non-flex', 'y non-flex'])
    df.to_csv(f'./{name}.csv')
    return


def create_result_df(x_flexible, x_non_flexible, y_flexible, y_non_flexible):
    """ Sve Monte Carlo simulation result on the folder
    :param x_flexible: feasible P,
    :type x_flexible: list of floats

    :param x_non_flexible: infeasible P,
    :type x_non_flexible: list of floats

    :param y_flexible: feasible Q,
    :type y_flexible: list of floats

    :param y_non_flexible: infeasible Q,
    :type y_non_flexible: list of floats

    :return:
    :rtype:
    """
    max_len = max(len(x_flexible), len(x_non_flexible))
    x_flexible_df = np.zeros(max_len)
    x_non_flexible_df = np.zeros(max_len)
    y_flexible_df = np.zeros(max_len)
    y_non_flexible_df = np.zeros(max_len)
    for i in range(0, len(x_flexible)):
        x_flexible_df[i] = float(x_flexible[i])
        y_flexible_df[i] = float(y_flexible[i])
    for i in range(0, len(x_non_flexible)):
        x_non_flexible_df[i] = float(x_non_flexible[i])
        y_non_flexible_df[i] = float(y_non_flexible[i])
    return pd.DataFrame(list(zip(x_flexible_df, y_flexible_df, x_non_flexible_df, y_non_flexible_df)),
                        columns=['x flex', 'y flex', 'x non-flex', 'y non-flex'])


def write_conv_result(df, name):
    """ Write results from convolution simulations

    :param df: dataframe with results
    :type df: pandas.dataframe

    :param name: name of scenario
    :type name: str

    :return:
    :rtype:
    """
    print(f"Saving result file in: ./{name}.csv")
    df.to_csv(f'./{name}.csv')
    return name


def check_limits(net, settings):
    """ Check that the network operates within the operation limits and print it

    :param net: network
    :type net: pandapower.network

    :param settings: scenario settings
    :type settings: object

    :return:
    :rtype:
    """
    max_curr = settings.max_curr
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    if check_voltage_limits(net.res_bus['vm_pu'], max_volt, min_volt) and \
       check_line_current_limits(net, max_curr) and check_trafo_current_limits(net, max_curr):
        print("Initial Operating point is feasible")
    else:
        print("Initial Operating Point is Infeasible")
    return


def assert_limits(net, settings):
    """ Check that the network operates within the operation limits and assert an error if not

    :param net: network
    :type net: pandapower.network

    :param settings: scenario settings
    :type settings: object

    :return:
    :rtype:
    """
    max_curr = settings.max_curr
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    if check_voltage_limits(net.res_bus['vm_pu'], max_volt, min_volt) and \
       check_line_current_limits(net, max_curr) and check_trafo_current_limits(net, max_curr):
        return True
    else:
        return False


def check_limits_bool(net, settings):
    """ Check that the network operates within the operation limits and return True(feasible)/False(not feasible)

    :param net: network
    :type net: pandapower.network

    :param settings: scenario settings
    :type settings: object

    :return:
    :rtype:
    """
    max_curr = settings.max_curr
    max_volt = settings.max_volt
    min_volt = settings.min_volt
    if check_voltage_limits(net.res_bus['vm_pu'], max_volt, min_volt) and \
       check_line_current_limits(net, max_curr) and check_trafo_current_limits(net, max_curr):
        return True
    return False


def tensor_convolve_nd_torch(image, kernel):
    """ Apply tensor convolution

    :param image: multi-dim tensor
    :type image: torch tensor

    :param kernel: 2D tensor
    :type kernel: torch tensor

    :return: tensor convolution result
    :rtype: torch tensor
    """
    # Flip the kernel
    kernel = torch.flip(kernel, dims=(0, 1))

    # Add zero padding to the input image
    image_dims = image.dim()
    pad_sizes = [len(kernel)-1, len(kernel)-1, len(kernel[0])-1, len(kernel[0])-1]+[0, 0] * (image_dims - 2)
    pad_sizes = tuple(pad_sizes)[::-1]
    image_padded = torch.nn.functional.pad(image, pad_sizes, mode='constant', value=0)
    # Repeat the kernel n times
    r_kernel = kernel.clone()
    for i in range(2, image_dims):
        r_kernel = r_kernel.unsqueeze(-1).expand(r_kernel.shape + (image.shape[i],))

    # Initialize output tensor
    output_shape = (image.shape[0] + len(kernel) - 1, image.shape[1] + len(kernel[0]) - 1) + r_kernel.shape
    output = torch.zeros(output_shape, dtype=torch.float32)

    # Loop over every pixel of the image
    slice_size = [len(kernel), len(kernel[0])] + [1] * (image_dims - 2)
    for x in range(image.shape[1] + kernel.shape[1] - 1):
        for y in range(image.shape[0] + kernel.shape[0] - 1):
            output[y, x, ] = torch.mul(r_kernel, image_padded[y:y + slice_size[0], x:x + slice_size[1]])

    return output


def tensor_convolve_nd_torch_half(image, kernel):
    """ Apply tensor convolution, but with float32 types (might run into issues exceeding the maximum number)

    :param image: multi-dim tensor
    :type image: torch tensor

    :param kernel: 2D tensor
    :type kernel: torch tensor

    :return: tensor convolution result
    :rtype: torch tensor
    """
    # Flip the kernel
    kernel = torch.flip(kernel, dims=(0, 1))

    # Add zero padding to the input image
    image_dims = image.dim()
    pad_sizes = [len(kernel)-1, len(kernel)-1, len(kernel[0])-1, len(kernel[0])-1]+[0, 0] * (image_dims - 2)
    pad_sizes = tuple(pad_sizes)[::-1]
    image_padded = torch.nn.functional.pad(image, pad_sizes, mode='constant', value=0)
    # Repeat the kernel n times
    r_kernel = kernel.clone()
    for i in range(2, image_dims):
        r_kernel = r_kernel.unsqueeze(-1).expand(r_kernel.shape + (image.shape[i],))

    # Initialize output tensor
    output_shape = (image.shape[0] + len(kernel) - 1, image.shape[1] + len(kernel[0]) - 1) + r_kernel.shape
    output = torch.zeros(output_shape, dtype=torch.float32)

    # Loop over every pixel of the image
    slice_size = [len(kernel), len(kernel[0])] + [1] * (image_dims - 2)
    for x in range(image.shape[1] + kernel.shape[1] - 1):
        for y in range(image.shape[0] + kernel.shape[0] - 1):
            output[y, x, ] = torch.mul(r_kernel, image_padded[y:y + slice_size[0], x:x + slice_size[1]])

    return output


def kumaraswamymontecarlo(a, b, c, LB, UB, num_samples, rng):
    """ Create samples using the Kumaraswamy Monte Carlo distribution

    :param a: alpha
    :type a: float
    :param b: beta
    :type b: float
    :param c: gamma
    :type c: float
    :param LB: lower bound
    :type LB: float
    :param UB: upper bound
    :type UB: float
    :param num_samples: number of samples
    :type num_samples: int
    :param rng: object to sample data from
    :type rng: numpy.random

    :return: samples
    :rtype: numpy array
    """
    num_variables = len(LB)

    MLB = np.repeat(LB[:, np.newaxis], num_samples, 1)
    UBLB = UB - LB
    MUBLB = np.repeat(UBLB[:, np.newaxis], num_samples, 1)

    uncorrelated = rng.standard_normal((num_variables, num_samples))

    cov = c * np.ones(shape=(num_variables, num_variables)) + (1 - c) * np.identity(num_variables)
    L = linalg.cholesky(cov)
    correlated = np.dot(L, uncorrelated)
    cdf_correlated = norm.cdf(correlated)

    karamsy = pow((1 - pow((1 - cdf_correlated), (1 / b))), (1 / a))

    # probabilities = a* *
    MCM = MLB + np.multiply(karamsy, MUBLB)

    return MCM


def fix_missing_point(mat):
    """   Filter in case the sampling of FSP shifts was wrong and left a point behind, it is observed in the power flow
          results if the sensitivity of a pixel is 0, but the upper, lower, left, and right values are non-zero. In that
          case, take their average.

    :param mat: Array with sensitivities or binary values
    :type mat: np.array

    :return: The input array fixed for these issue
    :rtype: np.array
    """
    for i in range(len(mat)-2):
        for j in range(len(mat[0])-2):
            if mat[i, j+1] != 0 and mat[i+1, j] != 0 and mat[i+1, j+2] != 0 and mat[i+2, j+1] != 0 and mat[i+1,j+1] == 0:
                mat[i+1, j+1] = (mat[i, j+1] + mat[i+1, j] + mat[i+1, j+2] + mat[i+2, j+1])/4
    return mat


def fix_missing_pointsv2(mat):
    """    Filter in case the sampling of FSP shifts was wrong and left a point behind, it is observed in the power flow
          results if the sensitivity of a pixel is 0, but the upper, lower, left, and right values are non-zero. In that
          case, take their average.

    :param mat: Array with sensitivities or binary values
    :type mat: np.array

    :return: The input array fixed for these issue
    :rtype: np.array
    """
    for i in range(len(mat)-2):
        for j in range(len(mat[0])-2):
            if mat[i+1, j] != 0 and mat[i+1, j+2] != 0 and mat[i+1, j+1] == 0:
                mat[i+1, j+1] = (mat[i+1, j] + mat[i+1, j+2])/2
    return mat


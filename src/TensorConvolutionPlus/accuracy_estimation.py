import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import os
from utils import fix_missing_pointsv2, create_result_df
from data_sampler import profile_creation, conv_profile_creation, profile_creation_bf
from monte_carlo import all_pf_simulations
from conv_simulations import torch_tensor_conv_simulations, torch_tensor_conv_large_simulations
from tqdm import tqdm
import pandapower as pp
import networkx as nx
from plotting import plot_uc3
import time



def extend_feasible_and_infeasible(ground_truth_file, conv_results_file, inf):
    """
    Extend the feasible and infeasible regions for visualization.
    This function extends the feasible and infeasible regions for visualization purposes.
    It takes a ground truth file, convex results file, and a DataFrame 'inf' representing infeasible points.
    It performs transformations on the data and generates a visualization.

    :param ground_truth_file: The filename of the ground truth data in CSV format.
    :type ground_truth_file: str

    :param conv_results_file: The filename of the convex results data in CSV format.
    :type conv_results_file: str

    :param inf: A DataFrame representing infeasible points.
    :type inf: pandas.DataFrame

    :return:
    :rtype:
    """

    # It generates a visualization but does not return any values.
    # The results are saved as an SVG file.
    inf = inf[inf.columns[:]].iloc[::-1]
    inf = inf[inf.columns[::-1]]
    ground_truth = pd.read_csv(f"../csv_results/{ground_truth_file}")
    x_flexible = ground_truth['x flex']
    y_flexible = ground_truth['y flex']
    x_non_flexible = ground_truth['x non-flex']
    y_non_flexible = ground_truth['y non-flex']

    conv_results = pd.read_csv(f"../csv_results/{conv_results_file}.csv").iloc[::-1]

    conv_results.replace(0, np.nan, inplace=True)
    inf.replace(0, np.nan, inplace=True)

    # Get axis limits
    conv_results = conv_results[conv_results.columns[::-1]]
    x_axis = [float(a) for a in list(conv_results.columns[:-1])]
    y_axis = [float(a) for a in list(conv_results[conv_results.columns[-1]])]

    # Get step size
    dx_b = x_axis[1]-x_axis[0]
    dy_b = y_axis[1]-y_axis[0]
    dx_a = x_axis[-1]-x_axis[-2]
    dy_a = y_axis[-1]-y_axis[-2]

    # Add pixels before and after (to account if ground truth predictions exceed the space of the convolution results)
    x_bef = np.arange(x_axis[0]-5*dx_b, x_axis[0], dx_b)
    x_aft = np.arange(x_axis[-1]+dx_a, x_axis[-1]+6*dx_a, dx_a)
    y_bef = np.arange(y_axis[0]-5*dy_b, y_axis[0], dy_b)
    y_aft = np.arange(y_axis[-1]+dy_a, y_axis[-1]+6*dy_a, dy_a)

    xtend_axis = list(x_bef) + list(x_axis) + list(x_aft)
    ytend_axis = list(y_bef) + list(y_axis) + list(y_aft)
    for i in range(len(x_bef)):
        conv_results.insert(0, x_bef[-i-1], list(np.zeros(len(conv_results))), True)
        inf.insert(0, x_bef[-i-1], list(np.zeros(len(inf))), True)
    for i in range(len(x_aft)):
        conv_results.insert(len(conv_results.columns)-1, x_aft[i], list(np.zeros(len(conv_results))), True)
        inf.insert(len(inf.columns), x_aft[i], list(np.zeros(len(inf))), True)
    inf = inf.reset_index(drop=True)
    if round(y_bef[-1], 5) == round(y_axis[0], 5):
        y_bef = y_bef[:-1]

    for i in range(len(y_bef)):
        new_row = pd.DataFrame([list(np.zeros(len(xtend_axis)))+[y_bef[-i-1]]], columns=conv_results.columns, index=[0])
        new_row_inf = pd.DataFrame([list(np.zeros(len(xtend_axis)))], columns=inf.columns, index=[0])
        conv_results = pd.concat([new_row, conv_results]).reset_index(drop=True)
        inf = pd.concat([new_row_inf, inf]).reset_index(drop=True)
        conv_results.loc[len(conv_results)] = list(np.zeros(len(xtend_axis)))+[y_aft[i]]
        inf.loc[len(inf)] = list(np.zeros(len(xtend_axis)))
    inf.replace(0, np.nan, inplace=True)

    x_fl_new = []
    y_fl_new = []
    x_nf_new = []
    y_nf_new = []
    for pq in zip(x_flexible, y_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            idx, p_val = find_value_close2list(xtend_axis, pq[0])
            jdx, q_val = find_value_close2list(ytend_axis, pq[1])
            x_fl_new.append(idx)
            y_fl_new.append(jdx)
    for pq in zip(x_non_flexible, y_non_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            idx, p_val = find_value_close2list(xtend_axis, pq[0])
            jdx, q_val = find_value_close2list(ytend_axis, pq[1])
            x_nf_new.append(idx)
            y_nf_new.append(jdx)

    plt.locator_params(nbins=10)

    inf = inf[inf.columns[:]].iloc[::-1]
    inf = inf[inf.columns[::-1]]
    sns.set(style="white")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.set(font_scale=1)
    cmap = LinearSegmentedColormap.from_list('', ["white", "darkblue", "red"], 3)
    sns.heatmap(inf, cmap=cmap, center=1, cbar=False, rasterized=True)
    xticks = [str(round(float(a), 3)) for a in list(conv_results.columns[:-1])]
    yticks = [str(round(float(a), 3)) for a in list(conv_results[conv_results.columns[-1]])]
    for dsda in range(len(xticks)):
        if dsda % 10 != 0:
            xticks[dsda] = ''
    for dsda in range(len(yticks)):
        if dsda % 20 != 0:
            yticks[dsda] = ''
    sns.heatmap(conv_results[conv_results.columns[:-1]], linewidths=0.1, linecolor='white', norm=LogNorm(),
                cmap='flare', center=1, xticklabels=xticks, yticklabels=yticks, rasterized=True)

    plt.scatter(x_nf_new, y_nf_new, c="lawngreen", s=10, label='Non Feasible Samples')
    plt.scatter(x_fl_new, y_fl_new, c="cyan", s=10, label='Feasible Samples')

    ax.set_xlabel("P [MW]", fontsize=12)
    ax.set_ylabel("Q [MVAr]", fontsize=12)

    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../plots/UC3/'+f'Overimposed_{ground_truth_file}.svg', bbox_inches='tight', pad_inches=0.5, dpi=500)
    return


def plot_results(ground_truth, conv_results, inf, name, x_axis, y_axis):
    """
    Generate a visualization of feasible and infeasible regions.
    This function generates a visualization of feasible and infeasible regions using input data.
    It takes ground truth data, convex results, infeasible points, a name for the plot, and axis values.
    The regions are plotted, and the resulting plot is saved as a PDF file.

    :param ground_truth: A DataFrame containing ground truth data.
    :type ground_truth: pandas.DataFrame

    :param conv_results: A DataFrame containing convex results.
    :type conv_results: pandas.DataFrame

    :param inf: A DataFrame representing infeasible points.
    :type inf: pandas.DataFrame

    :param name: The name of the plot and output PDF file.
    :type name: str

    :param x_axis: The x-axis values.
    :type x_axis: list

    :param y_axis: The y-axis values.
    :type y_axis: list

    :return:
    :rtype:
    """

    # It generates a plot and saves it as a PDF file.
    # The function does not return any values.
    x_flexible = ground_truth['x flex']
    y_flexible = ground_truth['y flex']
    x_non_flexible = ground_truth['x non-flex']
    y_non_flexible = ground_truth['y non-flex']
    conv_results.replace(0, np.nan, inplace=True)
    inf.replace(0, np.nan, inplace=True)
    dx_b = x_axis[1]-x_axis[0]
    dy_b = y_axis[1]-y_axis[0]
    dx_a = x_axis[-1]-x_axis[-2]
    dy_a = y_axis[-1]-y_axis[-2]
    x_bef = np.arange(x_axis[0]-5*dx_b, x_axis[0], dx_b)
    x_aft = np.arange(x_axis[-1]+dx_a, x_axis[-1]+5.5*dx_a, dx_a)
    y_bef = np.arange(y_axis[0]-5*dy_b, y_axis[0], dy_b)
    y_aft = np.arange(y_axis[-1]+dy_a, y_axis[-1]+5.5*dy_a, dy_a)
    xtend_axis = list(x_bef) + list(x_axis) + list(x_aft)
    ytend_axis = list(y_bef) + list(y_axis) + list(y_aft)
    for i in range(len(x_bef)):
        conv_results.insert(0, x_bef[-i-1], list(np.zeros(len(conv_results))), True)
        inf.insert(0, x_bef[-i-1], list(np.zeros(len(inf))), True)
    for i in range(len(x_aft)):
        conv_results.insert(len(conv_results.columns), x_aft[i], list(np.zeros(len(conv_results))), True)
        inf.insert(len(inf.columns), x_aft[i], list(np.zeros(len(inf))), True)
    inf = inf.reset_index(drop=True)
    if len(y_bef) > len(y_aft):
        y_bef = y_bef[:-1]
    for i in range(len(y_bef)):
        new_row = pd.DataFrame([list(np.zeros(len(xtend_axis)))], columns=conv_results.columns, index=[0])
        new_row_inf = pd.DataFrame([list(np.zeros(len(xtend_axis)))], columns=inf.columns, index=[0])
        conv_results = pd.concat([new_row, conv_results]).reset_index(drop=True)
        inf = pd.concat([new_row_inf, inf]).reset_index(drop=True)
        conv_results.loc[len(conv_results)] = list(np.zeros(len(xtend_axis)))
        inf.loc[len(inf)] = list(np.zeros(len(xtend_axis)))
    inf.replace(0, np.nan, inplace=True)
    x_fl_new = []
    y_fl_new = []
    x_nf_new = []
    y_nf_new = []
    imaged_mc = np.zeros_like(conv_results)
    for pq in zip(x_flexible, y_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            idx, p_val = find_value_close2list(xtend_axis, pq[0])
            jdx, q_val = find_value_close2list(ytend_axis, pq[1])
            x_fl_new.append(idx)
            y_fl_new.append(jdx)
            imaged_mc[jdx, idx] = 1
    for pq in zip(x_non_flexible, y_non_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            idx, p_val = find_value_close2list(xtend_axis, pq[0])
            jdx, q_val = find_value_close2list(ytend_axis, pq[1])
            x_nf_new.append(idx)
            y_nf_new.append(jdx)
    plt.locator_params(nbins=10)
    sns.set(style="white")
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["savefig.format"] = 'svg'
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.set(font_scale=1)
    cmap = LinearSegmentedColormap.from_list('', ["white", "darkblue", "red"], 3)
    sns.heatmap(inf, cmap=cmap, center=1, cbar=False, rasterized=True)
    xticks = [str(round(float(a), 3)) for a in list(xtend_axis)]
    yticks = [str(round(float(a), 3)) for a in list(ytend_axis)]
    for dsda in range(len(xticks)):
        if dsda % 8 != 0:
            xticks[dsda] = ''
    for dsda in range(len(yticks)):
        if dsda % 8 != 0:
            yticks[dsda] = ''
    sns.heatmap(conv_results, linewidths=0.1, linecolor='white', norm=LogNorm(),
                cmap='flare', center=1, xticklabels=xticks, yticklabels=yticks, rasterized=True)
    plt.scatter(x_nf_new, y_nf_new, c="lawngreen", s=10, label='Non Feasible Samples')
    plt.scatter(x_fl_new, y_fl_new, c="cyan", s=10, label='Feasible Samples')
    ax.set_xlabel("P [MW]", fontsize=12)
    ax.set_ylabel("Q [MVAr]", fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    ax.invert_yaxis()
    plt.tight_layout()
    loc = f'../plots/UC2/Loop/'+name+f'.pdf'
    plt.savefig(fname=loc, format='pdf', bbox_inches='tight', pad_inches=0.5, dpi=500)
    return


def correct_feasible_and_infeasible(ground_truth_file, conv_results_file, inf, dur_str):
    """
    Evaluate the feasibility of points.
    This function evaluates the feasibility of points based on ground truth data and results.
    It takes ground truth data, convex results, infeasible points, and a duration string.
    It calculates accuracy metrics and saves them in a log file.

    :param ground_truth_file: The filename of the ground truth data in CSV format.
    :type ground_truth_file: str

    :param conv_results_file: The filename of the convex results data in CSV format.
    :type conv_results_file: str

    :param inf: A DataFrame representing infeasible points.
    :type inf: pandas.DataFrame

    :param dur_str: A duration string.
    :type dur_str: str

    :return:
    :rtype:
    """

    # The function evaluates the feasibility of points based on input data and calculates accuracy metrics.
    # It saves the results in a log file.
    # The function does not return any values.
    inf = inf[inf.columns[:]].iloc[::-1]
    inf = inf[inf.columns[::-1]]
    ground_truth = pd.read_csv(f"../csv_results/{ground_truth_file}")
    x_flexible = ground_truth['x flex']
    y_flexible = ground_truth['y flex']
    x_non_flexible = ground_truth['x non-flex']
    y_non_flexible = ground_truth['y non-flex']

    conv_results = pd.read_csv(f"../csv_results/{conv_results_file}.csv").iloc[::-1]
    conv_results.replace(0, np.nan, inplace=True)
    inf.replace(0, np.nan, inplace=True)
    conv_results = conv_results[conv_results.columns[::-1]]

    x_axis = [float(a) for a in list(conv_results.columns[:-1])]
    y_axis = [float(a) for a in list(conv_results[conv_results.columns[-1]])]
    feas_predicted = 0
    feas_non_predicted = 0
    points_reached = 0
    points_non_reached = 0
    dx = x_axis[1]-x_axis[0]
    dy = y_axis[1]-y_axis[0]
    x_fl_new = []
    y_fl_new = []
    x_nf_new = []
    y_nf_new = []
    for pq in zip(x_flexible, y_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            x_fl_new.append(pq[0])
            y_fl_new.append(pq[1])
            idx, p_val = find_value_close2list(x_axis, pq[0])
            jdx, q_val = find_value_close2list(y_axis, pq[1])
            if pq[0] > max(x_axis)+dx or \
                    pq[1] > max(y_axis)+dy or \
                    pq[0] < min(x_axis)-dx or \
                    pq[1] < min(y_axis)-dy:
                feas_non_predicted += 1
                points_non_reached += 1
            elif not np.isnan(conv_results.iloc[jdx, idx]):
                feas_predicted += 1
                points_reached += 1
            elif not np.isnan(inf.iloc[jdx, idx]):
                points_reached += 1
                feas_non_predicted += 1
            else:
                points_non_reached += 1
                feas_non_predicted += 1
    for pq in zip(x_non_flexible, y_non_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            x_nf_new.append(pq[0])
            y_nf_new.append(pq[1])
            idx, p_val = find_value_close2list(x_axis, pq[0])
            jdx, q_val = find_value_close2list(y_axis, pq[1])
            if inf.iloc[jdx, idx] != np.nan:
                points_reached += 1
            else:
                points_non_reached += 1

    acc_f = feas_predicted/(feas_predicted+feas_non_predicted)
    acc_r = points_reached/(points_reached+points_non_reached)

    extend_feasible_and_infeasible(ground_truth_file, conv_results_file, inf)
    acc = f"Accuracy results:\n    -Feasible accuracy = {acc_f}\n    -Available accuracy = {acc_r}\n" +\
          f"    -Average = {0.5*acc_f+0.5*acc_r}\n"
    print(acc)
    with open(f"../plots/UC3/Logs_{conv_results_file}.txt", 'w') as f:
        f.write(acc+dur_str)
    return


def loop_acc(ground_truth, conv_results, inf):
    """
    Calculate and return accuracy metrics for UC2 scenarios.
    This function calculates accuracy metrics for the UC2 scenarios based on ground truth data, convex results,
    and infeasible points. It takes ground truth data, convex results, infeasible points, and a name for the plot.
    It calculates feasibility accuracy and available accuracy and returns their averages.

    :param ground_truth: A DataFrame containing ground truth data.
    :type ground_truth: pandas.DataFrame

    :param conv_results: A DataFrame containing convex results.
    :type conv_results: pandas.DataFrame

    :param inf: A DataFrame representing infeasible points.
    :type inf: pandas.DataFrame

    :return: feasibility accuracy, available accuracy, average accuracy, a placeholder 0.
    :rtype: float, float, flaot, float
    """

    # The function calculates feasibility accuracy, available accuracy, and their average for loop points.
    # It generates a plot based on input data.
    # The function returns a tuple of accuracy metrics.
    inf = inf[inf.columns[:]].iloc[::-1]
    inf = inf[inf.columns[::-1]]
    x_flexible = ground_truth['x flex']
    y_flexible = ground_truth['y flex']
    x_non_flexible = ground_truth['x non-flex']
    y_non_flexible = ground_truth['y non-flex']
    conv_results = conv_results.iloc[::-1]
    conv_results = conv_results[conv_results.columns[::-1]]
    conv_results.replace(0, np.nan, inplace=True)
    inf.replace(0, np.nan, inplace=True)
    y_axis = list(conv_results.index)
    x_axis = list(conv_results.columns)
    feas_predicted = 0
    feas_non_predicted = 0
    points_reached = 0
    points_non_reached = 0
    dx = x_axis[1]-x_axis[0]
    dy = y_axis[1]-y_axis[0]
    x_fl_new = []
    y_fl_new = []
    x_nf_new = []
    y_nf_new = []
    seen_jdx_idx = []
    for pq in zip(x_flexible, y_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            x_fl_new.append(pq[0])
            y_fl_new.append(pq[1])
            idx, p_val = find_value_close2list(x_axis, pq[0])
            jdx, q_val = find_value_close2list(y_axis, pq[1])
            if [idx, jdx] not in seen_jdx_idx:
                if pq[0] > max(x_axis)+abs(dx) or \
                        pq[1] > max(y_axis)+abs(dy) or \
                        pq[0] < min(x_axis)-abs(dx) or \
                        pq[1] < min(y_axis)-abs(dy):
                    feas_non_predicted += 1
                    points_non_reached += 1
                elif not np.isnan(conv_results.iloc[jdx, idx]) or inf.iloc[jdx, idx] == 2.:
                    feas_predicted += 1
                    points_reached += 1
                elif not np.isnan(inf.iloc[jdx, idx]):
                    points_reached += 1
                    feas_non_predicted += 1
                else:
                    points_non_reached += 1
                    feas_non_predicted += 1
                seen_jdx_idx.append([idx, jdx])
    seen_jdx_idx = []
    for pq in zip(x_non_flexible, y_non_flexible):
        if abs(pq[0])+abs(pq[1]) > 0:
            x_nf_new.append(pq[0])
            y_nf_new.append(pq[1])
            idx, p_val = find_value_close2list(x_axis, pq[0])
            jdx, q_val = find_value_close2list(y_axis, pq[1])
            if [idx, jdx] not in seen_jdx_idx:
                if pq[0] > max(x_axis)+dx or \
                        pq[1] > max(y_axis)+dy or \
                        pq[0] < min(x_axis)-dx or \
                        pq[1] < min(y_axis)-dy:
                    points_non_reached += 1
                elif not np.isnan(inf.iloc[jdx, idx]):
                    points_reached += 1
                else:
                    points_non_reached += 1
                seen_jdx_idx.append([idx, jdx])
    acc_f = feas_predicted/(feas_predicted+feas_non_predicted)
    acc_r = points_reached/(points_reached+points_non_reached)
    return acc_f, acc_r, 0.5*acc_f+0.5*acc_r, 0


def accuracy_brute_force(ground_truth_file, conv_results_file, decimals, operating_point):
    """
    Calculate and print mean squared error (MSE) for feasible and total points.
    This function calculates the mean squared error (MSE) for feasible and total points between ground truth data and
    results. It takes ground truth data, convex results, a number of decimals, and an operating point. It also
    generates a heatmap plot and saves it as an SVG file.

    :param ground_truth_file: The filename of the ground truth data in CSV format.
    :type ground_truth_file: str

    :param conv_results_file: The filename of the convex results data in CSV format.
    :type conv_results_file: str

    :param decimals: The number of decimals for rounding.
    :type decimals: int

    :param operating_point: The operating point as [P, Q] values.
    :type operating_point: list

    :return:
    :rtype:
    """

    # The function calculates and prints the MSE for feasible and total points.
    # It generates a heatmap plot and saves it as an SVG file.
    # The function does not return any values.
    # ----------------------------------------------------------------------------------------------------------------
    # Reading files and obtaining the feasible and infeasible samples from each file
    # ----------------------------------------------------------------------------------------------------------------
    cwd = os.getcwd()+'/csv_results/'  # Get the current working directory (cwd)
    ground_truth = pd.read_csv(cwd + ground_truth_file)
    first_i = len(ground_truth)
    first_j = len(ground_truth)
    for i in range(len(ground_truth)):
        if float(ground_truth['x flex'].iloc[len(ground_truth)-i-1]) != 0 \
                and float(ground_truth['y flex'].iloc[len(ground_truth)-i-1]) != 0:
            first_i = len(ground_truth)-i
            break
    for i in range(len(ground_truth)):
        if float(ground_truth['x non-flex'].iloc[len(ground_truth) - i - 1]) != 0 \
                and float(ground_truth['y non-flex'].iloc[len(ground_truth) - i - 1]) != 0:
            first_j = len(ground_truth) - i
            break

    xy_flex = ground_truth[['x flex', 'y flex']].head(first_i)
    x_flexible = xy_flex['x flex']
    y_flexible = xy_flex['y flex']
    xy_non_flex = ground_truth[['x non-flex', 'y non-flex']].head(first_j)
    x_non_flexible = xy_non_flex['x non-flex']
    y_non_flexible = xy_non_flex['y non-flex']

    conv_results = np.fliplr(pd.read_csv(f"../csv_results/{conv_results_file}.csv").iloc[::-1].to_numpy())
    conv_results = conv_results[:, :-1]
    # ----------------------------------------------------------------------------------------------------------------
    # Estimating multiplicity from PF based simulations with image resolution same as the TensorConv+ results
    # ----------------------------------------------------------------------------------------------------------------
    pixels_p = len(conv_results[0])-1
    pixels_q = len(conv_results)-1

    mix_x1 = round(np.min(x_non_flexible), decimals)
    max_x1 = round(np.max(x_non_flexible), decimals)
    mix_y1 = round(np.min(y_non_flexible), decimals)
    max_y1 = round(np.max(y_non_flexible), decimals)
    mix_x2 = round(np.min(x_flexible), decimals)
    max_x2 = round(np.max(x_flexible), decimals)
    mix_y2 = round(np.min(y_flexible), decimals)
    max_y2 = round(np.max(y_flexible), decimals)
    mix_x = min(mix_x1, mix_x2)
    max_x = max(max_x1, max_x2)
    mix_y = min(mix_y1, mix_y2)
    max_y = max(max_y1, max_y2)
    step_p = (max_x - mix_x)/pixels_p
    step_q = (max_y - mix_y)/pixels_q
    x_axis = np.arange(mix_x, max_x+step_p/2, step_p)
    y_axis = np.arange(mix_y, max_y+step_q/2, step_q)
    inf_heat_mat, x_axis, y_axis, step_p, step_q = get_inf_matrix(x_non_flexible, y_non_flexible, step_p, step_q,
                                                                  x_axis, y_axis)

    heat_mat = get_heatmap_matrix(x_flexible, y_flexible, step_p, step_q, x_axis, y_axis)
    # ----------------------------------------------------------------------------------------------------------------
    # Imputing missing points
    # ----------------------------------------------------------------------------------------------------------------
    heat_mat = fix_missing_pointsv2(heat_mat)
    inf_heat_mat = fix_missing_pointsv2(inf_heat_mat)
    # ----------------------------------------------------------------------------------------------------------------
    # Plotting PowerFlow heatmap
    # ----------------------------------------------------------------------------------------------------------------
    q_index, _ = find_value_close2list(y_axis, float(operating_point[1]))
    p_index, _ = find_value_close2list(x_axis, float(operating_point[0]))

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(7.5, 5))

    cmap = LinearSegmentedColormap.from_list('', ["white", "darkblue", "red"], 3)
    sns.heatmap(inf_heat_mat, cmap=cmap, linewidths=0.0, center=1, cbar=False, rasterized=True)

    xticks = [str(round(float(a), 3)) for a in x_axis]
    yticks = [str(round(float(a), 3)) for a in y_axis]
    for dsda in range(len(xticks)):
        if dsda % 10 != 0:
            xticks[dsda] = ''
    for dsda in range(len(yticks)):
        if dsda % 20 != 0:
            yticks[dsda] = ''
    sns.heatmap(heat_mat, norm=LogNorm(), linewidths=0.1, linecolor='white', cmap='flare', center=1,
                xticklabels=xticks,
                yticklabels=yticks, rasterized=True)
    ax.scatter(p_index, q_index, marker='*', s=300, color='red', rasterized=True)
    ax.set_xlabel("P [MW]", fontsize=12)
    ax.set_ylabel("Q [MVAr]", fontsize=12)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.savefig(f'../plots/UC1/'+f'Compare_{ground_truth_file.replace("UC1/", "").replace(".txt", "")}.svg',
                bbox_inches='tight', pad_inches=0.5, dpi=500)

    # ----------------------------------------------------------------------------------------------------------------
    # Estimating Accuracy on all pixels
    # ----------------------------------------------------------------------------------------------------------------

    # Accuracy on feasible:
    mse_feas = 0
    mse_tot = 0
    samps = 0
    samps_tot = 0

    for i in range(len(heat_mat)):
        for j in range(len(heat_mat[0])):
            if 2 >= conv_results[i, j] > 1. or heat_mat[i, j] > 1.:
                mse_feas += (conv_results[i, j] - heat_mat[i, j]) ** 2
                samps += 1
            mse_tot += (conv_results[i, j]-heat_mat[i, j])**2
            samps_tot += 1
    mse_feas *= 1 / samps
    mse_tot *= 1 / samps_tot
    rmse_tot = mse_tot**0.5
    print(f"Total RMSE: {rmse_tot}")
    return


def info_options(x_flexible, x_non_flexible, y_flexible, y_non_flexible, dp, dq, maxv, minv, maxl,
                 cost, used_fsps, prof_flex):
    """ This function plots and prints information for the case study 'DFC Improving TSOs Flexibility Shift Selection'.

    :param x_flexible: flexible pcc p
    :type x_flexible: list

    :param x_non_flexible: non-flexible pcc p
    :type x_non_flexible: list

    :param y_flexible: flexible pcc q
    :type y_flexible: list

    :param y_non_flexible: non flexible pcc q
    :type y_non_flexible: list

    :param dp: active power resolution [mw]
    :type dp: float

    :param dq: reactive power resolution [mvar]
    :type dq: float

    :param maxv: maximmum voltages [p.u.]
    :type maxv: list

    :param minv: minimum voltages [p.u.]
    :type minv: list

    :param maxl: maximum absolute loading [%]
    :type maxl: list

    :param cost: costs for flexiiblity point [euro]
    :type cost: list

    :param used_fsps: minimum fsps used for each flexibility point [int]
    :type used_fsps: list

    :param prof_flex: flexible profiles
    :type prof_flex: list

    :return:
    :rtype:
    """
    mix_x1 = round(np.min(x_non_flexible), 2)
    max_x1 = round(np.max(x_non_flexible), 2)
    mix_y1 = round(np.min(y_non_flexible), 2)
    max_y1 = round(np.max(y_non_flexible), 2)
    mix_x2 = round(np.min(x_flexible), 2)
    max_x2 = round(np.max(x_flexible), 2)
    mix_y2 = round(np.min(y_flexible), 2)
    max_y2 = round(np.max(y_flexible), 2)
    mix_x = min(mix_x1, mix_x2)
    max_x = max(max_x1, max_x2)
    mix_y = min(mix_y1, mix_y2)
    max_y = max(max_y1, max_y2)
    step_p = dp
    step_q = dq
    x_axis = np.arange(mix_x, max_x+step_p/2, step_p)
    y_axis = np.arange(mix_y, max_y+step_q/2, step_q)
    selected_pts = {'a': [12, 6], 'b': [9, 6], 'z': [1, 3]}
    feas_mat, heat_mat, nflex_mat, min_vxmat, max_vxmat, min_vnmat, max_vnmat, min_lmat, max_lmat, min_cmat, max_cmat, \
    min_umat, max_umat, v_plot = get_info_matrix(x_flexible, y_flexible, x_non_flexible, y_non_flexible, step_p, step_q,
                                                 x_axis, y_axis, maxv, minv, maxl, cost, used_fsps, selected_pts,
                                                 prof_flex)
    sns.set(style="white")


    xticks = [str(round(float(a), 3)) for a in x_axis]
    yticks = [str(round(float(a), 3)) for a in y_axis]
    for dsda in range(len(xticks)):
        if dsda % 1 != 0:
            xticks[dsda] = ''
    for dsda in range(len(yticks)):
        if dsda % 1 != 0:
            yticks[dsda] = ''
    cols = [ 'crest', 'flare', 'Blues', 'flare', 'flare', 'flare', 'flare', 'flare', 'flare', 'YlOrBr',
            'YlOrBr', 'YlOrBr', 'YlOrBr']
    cc = 0
    for mmat, name in zip([feas_mat, heat_mat, nflex_mat, min_vxmat, max_vxmat, min_vnmat, max_vnmat, min_lmat, max_lmat,
                           min_cmat, max_cmat, min_umat, max_umat], ['feas_mat', 'heat_mat', 'nflex_mat', 'min_vxmat',
                                                                     'max_vxmat', 'min_vnmat', 'max_vnmat', 'min_lmat',
                                                                     'max_lmat', 'min_cmat', 'max_cmat', 'min_umat',
                                                                   'max_umat']):
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(mmat, norm=LogNorm(), linewidths=0.1, linecolor='white', cmap=cols[cc], center=1,
                    xticklabels=xticks,
                    yticklabels=yticks, rasterized=True)
        ax.set_xlabel("P [MW]", fontsize=12)
        ax.set_ylabel("Q [MVAr]", fontsize=12)

        ax.invert_yaxis()
        plt.tight_layout()
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.savefig(f'../plots/UC6/{name}.svg', bbox_inches='tight', pad_inches=0.5, dpi=500)
        np.savetxt(f"../csv_results/{name}.csv", mmat, delimiter=",")
        cc += 1
    return


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


def get_inf_matrix(x_flex, y_flex, step_p, step_q, x_axis, y_axis):
    """
    Generate the non-feasible space matrix based on provided data.
    This function generates the non-feasible space matrix where specific coordinates in the matrix are marked as "1"
    based on the provided `x_flex` and `y_flex` coordinates. The matrix dimensions are determined by `x_axis` and `y_axis`, and the
    spacing between points is defined by `step_p` and `step_q`.

    :param x_flex: List of x-coordinates to mark as "1" in the infill matrix.
    :type x_flex: list[float]

    :param y_flex: List of y-coordinates to mark as "1" in the infill matrix.
    :type y_flex: list[float]

    :param step_p: Spacing between x-axis points.
    :type step_p: float

    :param step_q: Spacing between y-axis points.
    :type step_q: float

    :param x_axis: List of x-axis values representing the matrix's x-coordinates.
    :type x_axis: list[float]/numpy array

    :param y_axis: List of y-axis values representing the matrix's y-coordinates.
    :type y_axis: list[float]/numpy array

    :return: An infill matrix with marked coordinates as "1," along with the x and y axis values and spacing.
    :rtype: numpy.ndarray, list[float], list[float], float, float
    """

    # The function generates the inf matrix based on provided data and returns the matrix along with axis values.

    # Example:
    # >>> get_inf_matrix([1.0, 2.0, 3.0], [4.0, 5.0], 1.0, 1.0, [1.0, 2.0, 3.0], [4.0, 5.0])
    # (array([[0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 1., 0.]]), [1.0, 2.0, 3.0], [4.0, 5.0], 1.0, 1.0)

    heat_mat = np.zeros((len(y_axis), len(x_axis)))
    for x, y in zip(x_flex, y_flex):
        x_idx, = np.where(np.isclose(x_axis, x, atol=step_p/2))
        y_idx, = np.where(np.isclose(y_axis, y, atol=step_q/2))
        try:
            heat_mat[y_idx, x_idx] = 1
        except:
            print(y_idx, x_idx, heat_mat.shape, x, y, x_axis, y_axis)
            assert False, "Error"
    return heat_mat, x_axis, y_axis, step_p, step_q


def get_heatmap_matrix(x_flex, y_flex, step_p, step_q, x_axis, y_axis):
    """
    Generate a heatmap matrix based on provided coordinates.
    This function generates a heatmap matrix where specific coordinates in the matrix are incremented based on the
    provided `x_flex` and `y_flex` coordinates. The matrix dimensions are determined by `x_axis` and `y_axis`, and the
    spacing between points is defined by `step_p` and `step_q`.

    :param x_flex: List of x-coordinates to increment in the heatmap matrix.
    :type x_flex: list[float]

    :param y_flex: List of y-coordinates to increment in the heatmap matrix.
    :type y_flex: list[float]

    :param step_p: Spacing between x-axis points.
    :type step_p: float

    :param step_q: Spacing between y-axis points.
    :type step_q: float

    :param x_axis: List of x-axis values representing the matrix's x-coordinates.
    :type x_axis: list[float]

    :param y_axis: List of y-axis values representing the matrix's y-coordinates.
    :type y_axis: list[float]

    :return: A heatmap matrix with incremented coordinates and values normalized to the maximum value.
    :rtype: numpy.ndarray
    """

    # The function generates a heatmap matrix based on provided coordinates and returns the normalized matrix.

    # Example:
    # >>> get_heatmap_matrix([1.0, 2.0, 3.0], [4.0, 5.0], 1.0, 1.0, [1.0, 2.0, 3.0], [4.0, 5.0])
    # array([[0.        , 0.33333333, 0.        ],
    #        [0.        , 0.        , 0.33333333],
    #        [0.        , 0.        , 0.        ]])

    heat_mat = np.zeros((len(y_axis), len(x_axis)))
    for x, y in zip(x_flex, y_flex):
        x_idx, = np.where(np.isclose(x_axis, x, atol=step_p/2))
        y_idx, = np.where(np.isclose(y_axis, y, atol=step_q/2))
        try:
            heat_mat[y_idx, x_idx] += 1
        except:
            if len(y_idx) == 0:
                y_idx = [0]
            elif len(y_idx) == 2:
                y_idx = [y_idx[0]]
            if len(x_idx) == 0:
                x_idx = [0]
            elif len(x_idx) == 2:
                y_idx = [y_idx[0]]
            heat_mat[y_idx, x_idx] += 1
    heat_mat *= 1/np.amax(heat_mat)
    for row in range(len(heat_mat)):
        for col in range(len(heat_mat[0])):
            if heat_mat[row, col] > 0:
                heat_mat[row, col] += 1
    return heat_mat


def get_info_matrix(x_flex, y_flex, x_nflex, y_nflex, step_p, step_q, x_axis, y_axis, maxv, minv, maxl, cost,
                    used_fsps, selected_pts, prof_flex):
    """Get information for the case study "DFC Improving TSOs Flexibility Shift Selection".

    :param x_flex: flexible active power points at pcc.
    :type x_flex: list

    :param y_flex: flexible reacitve power points at pcc.
    :type y_flex: list

    :param x_nflex: not-flexible active power points at pcc.
    :type x_nflex: list

    :param y_nflex: not-flexible reactive power points at pcc.
    :type y_nflex: list

    :param step_p: active power resolution [mw].
    :type step_p: float

    :param step_q: reactive power resolution [mvar].
    :type step_q: float

    :param x_axis: x-axis values.
    :type x_axis: list

    :param y_axis: y-axis values.
    :type y_axis: list

    :param maxv: maximum voltages [p.u.].
    :type maxv: list

    :param minv: minimum voltages [p.u.].
    :type minv: list

    :param maxl: maximum absolute loading [%].
    :type maxl: list

    :param cost: costs for flexibility point [euro].
    :type cost: list

    :param used_fsps: minimum fsps used for each flexibility point [int].
    :type used_fsps: list

    :param selected_pts: selected points to study.
    :type selected_pts: dictionary

    :param prof_flex: flexible profiles.
    :type prof_flex: list

    :return: matrices for number of feasible combinations (feas_mat), DFC (heat_mat), non-flexible combinations
             (nflex_mat), minimum maximum voltages (min_vxmat), maximum maximum voltages (max_vxmat),
             minimum minimum voltages (min_vnmat), maximum minimum voltages (max_vnmat), minimum loading (min_lmat),
             maximum loading (max_lmat), minimum costs (min_cmat), maximum costs (max_cmat), minumum used FSPs (min_umat),
             maximum used FSPs (max_umat), values for scenarios (v_plot).
    :rtype: lists
    """
    heat_mat = np.zeros((len(y_axis), len(x_axis)))
    nflex_mat = np.zeros((len(y_axis), len(x_axis)))
    min_vxmat = 100*np.ones((len(y_axis), len(x_axis)))
    max_vxmat = np.zeros((len(y_axis), len(x_axis)))
    min_vnmat = 100*np.ones((len(y_axis), len(x_axis)))
    max_vnmat = np.zeros((len(y_axis), len(x_axis)))
    min_lmat = 1000*np.ones((len(y_axis), len(x_axis)))
    max_lmat = np.zeros((len(y_axis), len(x_axis)))
    min_cmat = 1000*np.ones((len(y_axis), len(x_axis)))
    max_cmat = np.zeros((len(y_axis), len(x_axis)))
    min_umat = 100*np.ones((len(y_axis), len(x_axis)))
    max_umat = np.zeros((len(y_axis), len(x_axis)))
    v_plot = {'a': [], 'b': [], 'z': []}
    p_i = 0
    for x, y, vx, vn, lx, c, u in zip(x_flex, y_flex, maxv, minv, maxl, cost, used_fsps):
        x_idx, = np.where(np.isclose(x_axis, x, atol=step_p/2))
        y_idx, = np.where(np.isclose(y_axis, y, atol=step_q/2))
        try:
            heat_mat[y_idx, x_idx] += 1
        except:
            if len(y_idx) == 0:
                y_idx = [0]
            elif len(y_idx) == 2:
                y_idx = [y_idx[0]]
            if len(x_idx) == 0:
                x_idx = [0]
            elif len(x_idx) == 2:
                y_idx = [y_idx[0]]
            heat_mat[y_idx, x_idx] += 1
        try:
            if min_vxmat[y_idx, x_idx] > vx:
                min_vxmat[y_idx, x_idx] = vx
        except:
            continue
        if max_vxmat[y_idx, x_idx] < vx:
            max_vxmat[y_idx, x_idx] = vx
        if min_vnmat[y_idx, x_idx] > vn:
            min_vnmat[y_idx, x_idx] = vn
        if max_vnmat[y_idx, x_idx] < vn:
            max_vnmat[y_idx, x_idx] = vn
        if min_lmat[y_idx, x_idx] > lx:
            min_lmat[y_idx, x_idx] = lx
        if max_lmat[y_idx, x_idx] < lx:
            max_lmat[y_idx, x_idx] = lx
        if min_cmat[y_idx, x_idx] > c:
            min_cmat[y_idx, x_idx] = c
        if max_cmat[y_idx, x_idx] < c:
            max_cmat[y_idx, x_idx] = c
        if min_umat[y_idx, x_idx] > u:
            min_umat[y_idx, x_idx] = u
        if max_umat[y_idx, x_idx] < u:
            max_umat[y_idx, x_idx] = u
        if y_idx == selected_pts['a'][0] and x_idx == selected_pts['a'][1] and c <= 37.92:
            print(c, 'a', x, y)
            v_plot['a'] = prof_flex[p_i]
            print(prof_flex[p_i])
        elif y_idx == selected_pts['b'][0] and x_idx == selected_pts['b'][1] and c <= 66.47:
            print(c, 'b', x, y)
            v_plot['b'] = prof_flex[p_i]
            print(prof_flex[p_i])
        elif y_idx == selected_pts['z'][0] and x_idx == selected_pts['z'][1] and c <= 177.44:
            print(c, 'z', x, y)
            v_plot['z'] = prof_flex[p_i]
            print(prof_flex[p_i])
        p_i += 1
    for x, y in zip(x_nflex, y_nflex):
        x_idx, = np.where(np.isclose(x_axis, x, atol=step_p/2))
        y_idx, = np.where(np.isclose(y_axis, y, atol=step_q/2))
        try:
            nflex_mat[y_idx, x_idx] += 1
        except:
            if len(y_idx) == 0:
                y_idx = [0]
            elif len(y_idx) == 2:
                y_idx = [y_idx[0]]
            if len(x_idx) == 0:
                x_idx = [0]
            elif len(x_idx) == 2:
                y_idx = [y_idx[0]]
            nflex_mat[y_idx, x_idx] += 1
    feas_mat = heat_mat.copy()
    heat_mat *= 1/np.amax(heat_mat)
    print("Validate:")
    print(feas_mat[selected_pts['a'][0], selected_pts['a'][1]], feas_mat[selected_pts['b'][0], selected_pts['b'][1]],
          feas_mat[selected_pts['z'][0], selected_pts['z'][1]])
    heat_mat[heat_mat != 0] += 1

    min_vxmat[min_vxmat == 100] = 0
    min_vnmat[min_vnmat == 100] = 0
    min_lmat[min_lmat == 1000] = 0
    min_cmat[min_cmat == 1000] = 0
    min_umat[min_umat == 100] = 0
    return feas_mat, heat_mat, nflex_mat, min_vxmat, max_vxmat, min_vnmat, max_vnmat, min_lmat, max_lmat, min_cmat, \
           max_cmat, min_umat, max_umat, v_plot


def range_acc_case_study(net, settings, pcc_operating_point):
    """ Run some simulations for range accuracy case study.

    :param net: network.
    :type net: pandapower.network

    :param settings: scenario settings.
    :type settings: object

    :param pcc_operating_point: initial PQ values at the PCC.
    :type pcc_operating_point: list[float, float]

    :return:
    :rtype:
    """
    dg_options = net.sgen.index.values.tolist()
    load_options = net.load.index.values.tolist()
    seed = 1994 + settings.use_case_dict.get("Ver.")
    rng = np.random.RandomState(seed)
    no_fsps = settings.use_case_dict.get("FSPs")
    speed_res = []
    acc_res = []
    tg0 = time.time()
    graph = pp.topology.create_nxgraph(net, calc_branch_impedances=True)
    spl = dict(nx.all_pairs_dijkstra_path_length(graph, weight="z_ohm"))
    tg1 = time.time()
    print(f" Initial network minimum and maximum voltages {min(net.res_bus['vm_pu']), max(net.res_bus['vm_pu'])}")
    if settings.use_case_dict.get("plot", False):
        plot_uc3("Oberrhein0", "Oberrhein1")
    elif settings.use_case_dict.get("no_scenarios") > 2 and no_fsps < 8:
        for i in tqdm(range(settings.use_case_dict.get("no_scenarios"))):
            no_of_dgs = rng.randint(0, no_fsps+1)
            no_of_loads = no_fsps-no_of_dgs
            if no_of_loads < 0:
                assert AssertionError, "Error: No of loads should be >=0"
            dgs = []
            loads = []
            if no_of_dgs > 0:
                dgs = rng.choice(dg_options, size=no_of_dgs, replace=False)
            if no_of_loads > 0:
                loads = rng.choice(load_options, size=no_of_loads, replace=False)
            net_h = net.deepcopy()
            net_u = net.deepcopy()
            net_n = net.deepcopy()
            net_tc = net.deepcopy()
            settings.fsp_dg = dgs
            settings.fsp_load = loads

            dist_dicts = {}
            for fffsp in dgs:
                dist_dicts[net.sgen['name'].iloc[fffsp]] = {}
                min_v = 1000
                min_d = ""
                for ffsp2 in dgs:
                    if fffsp != ffsp2:
                        dist_dicts[net.sgen['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp2]] = \
                            spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                        if spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]] < min_v:
                            min_v = spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                            min_d = net.sgen['name'].iloc[ffsp2]
                for ffsp3 in loads:
                    dist_dicts[net.sgen['name'].iloc[fffsp]][net.load['name'].iloc[ffsp3]] = \
                        spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                    if spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]] < min_v:
                        min_v = spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                        min_d = net.load['name'].iloc[ffsp3]
                dist_dicts[net.sgen['name'].iloc[fffsp]]["min"] = [min_v, min_d]
            for fffsp in loads:
                dist_dicts[net.load['name'].iloc[fffsp]] = {}
                min_v = 1000
                min_d = ""
                for ffsp2 in loads:
                    if fffsp != ffsp2:
                        dist_dicts[net.load['name'].iloc[fffsp]][net.load['name'].iloc[ffsp2]] = \
                            spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                        if spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]] < min_v:
                            min_v = spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                            min_d = net.load['name'].iloc[ffsp2]
                for ffsp3 in dgs:
                    dist_dicts[net.load['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp3]] = \
                        spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                    if spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]] < min_v:
                        min_v = spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                        min_d = net.sgen['name'].iloc[ffsp3]
                dist_dicts[net.load['name'].iloc[fffsp]]["min"] = [min_v, min_d]

            tot_p = 0
            tot_q = 0
            min_p = np.inf
            for vv in dgs:
                tot_p += float(net.sgen['p_mw'][vv])
                tot_q += float(net.sgen['q_mvar'][vv])
                if min_p > float(net.sgen['p_mw'][vv]):
                    min_p = float(net.sgen['p_mw'][vv])
            for vv in loads:
                tot_p += float(net.load['p_mw'][vv])
                tot_q += float(net.load['q_mvar'][vv])
                if min_p > float(net.load['p_mw'][vv]):
                    min_p = float(net.load['p_mw'][vv])
            settings.dp = round(min((tot_p+tot_q)/20, min_p), 3)
            settings.dq = round(min((tot_p+tot_q)/10, min_p*2), 3)
            print(min_p, settings.dp)
            print(f"DGs: {dgs}, Loads: {loads}\n"
                  f"DP: {settings.dp}, DQ: {settings.dq}")
            pq_profiles, dur_samples_h = profile_creation(settings.no_samples, net_h, "Normal_Limits_Oriented",
                                                          settings.keep_mp, services="All", flexible_loads=loads,
                                                          flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_h, prf_flx, prf_non_flx = all_pf_simulations(settings, net_h,
                                                                                                  pq_profiles)
            res_hard = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            pq_profiles, dur_samples_u = profile_creation(settings.no_samples, net_u, "Uniform", settings.keep_mp,
                                                         services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_u, prf_flx, prf_non_flx = all_pf_simulations(settings, net_u,
                                                                                                  pq_profiles)
            res_un = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            pq_profiles, dur_samples_k = profile_creation(settings.no_samples, net_n, "Kumaraswamy", settings.keep_mp,
                                                        services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_k, prf_flx, prf_non_flx = all_pf_simulations(settings, net_n,
                                                                                                  pq_profiles)
            res_norm = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)

            # No of PF for alternative:
            pq_profiles, dur_samples_k = profile_creation(settings.no_samples, net_n, "Kumaraswamy", settings.keep_mp,
                                                        services="All", flexible_loads=loads, flexible_dg=dgs)
            _, _, _, _, no_samps = \
                profile_creation_bf(settings.dp, settings.dq, net, services="All",
                                    flexible_loads=loads, flexible_dgs=dgs, non_linear_dgs=settings.non_lin_dgs)
            print(f"Number of PFs needed for exhaustive search would be: {no_samps}")
            del pq_profiles
            del x_flx
            del y_flx
            del x_non_flx
            del y_non_flx
            del t_pf_k
            del prf_flx
            del prf_non_flx
            pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
                conv_profile_creation(settings.dp, settings.dq, net_tc, services="All", flexible_loads=loads,
                                      flexible_dgs=dgs)
            df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str = \
                torch_tensor_conv_simulations(net_tc, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                              small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                              comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
            del pq_profiles
            mes = f"{0}: DGs:{dgs},Loads:{loads},DP:{settings.dp}, Tot: {tot_p+tot_q}, min: {min_p}"
            dur_tmp = dur_str.split("=")[1:]
            new_tmp = [dur_samples]
            for val in dur_tmp:
                new_tmp.append(float(val.split("s,")[0]))
            speed_res.append(new_tmp)
            acc_feas_h, acc_reach_h, acc_avg_h, hull_acc_h = loop_acc(res_hard, df, inf, f"Hard_{settings.name}_{i}")
            acc_feas_u, acc_reach_u, acc_avg_u, hull_acc_u = loop_acc(res_un, df, inf, f"Uniform_{settings.name}_{i}")
            acc_feas_n, acc_reach_n, acc_avg_n, hull_acc_n = loop_acc(res_norm, df, inf,
                                                                      f"Kumaraswamy_{settings.name}_{i}")
            acc_res.append([acc_feas_h, acc_reach_h, acc_avg_h, hull_acc_h,
                           acc_feas_u, acc_reach_u, acc_avg_u, hull_acc_u,
                           acc_feas_n, acc_reach_n, acc_avg_n, hull_acc_n, mes])
            print(acc_feas_h, acc_reach_h, acc_avg_h, hull_acc_h)
        speed_df = pd.DataFrame(speed_res, columns=["Sampling Shifts", "Shape Preparation", "Power Flows",
                                                        "Net. Component vs FSP Dictionary", "Small FSPs Removal",
                                                        "Effective FSPs per Component", "Remove Safe Components",
                                                        "Tensors and Convolutions", "Applying Axes and Initial Point",
                                                        "Small FSP Uncertainty"])
        acc_df = pd.DataFrame(acc_res, columns=["Feasible Hard", "Reachable Hard", "Average Hard", "Hull Shift Hard"
                                                    , "Feasible Uniform", "Reachable Uniform", "Average Uniform",
                                                    "Hull Shift Uniform", "Feasible Kumaraswamy",
                                                    "Reachable Kumaraswamy", "Average Kumaraswamy",
                                                    "Hull Shift Kumaraswamy", "FSPs"])
        speed_df.to_csv(f'../csv_results/UC2/Speed_{settings.name}')
        acc_df.to_csv(f'../csv_results/UC2/Acc_{settings.name}')

    elif no_fsps > 7:
        for i in tqdm(range(settings.use_case_dict.get("no_scenarios"))):
            search = 0
            its = 0
            while search == 0:
                its += 1
                no_of_dgs0 = rng.randint(0, int(no_fsps/2) + 1)
                no_of_dgs1 = rng.randint(0, no_fsps - int(no_fsps/2) + 1)
                no_of_loads0 = int(no_fsps/2) - no_of_dgs0
                no_of_loads1 = no_fsps - int(no_fsps/2) - no_of_dgs1
                dgs0 = []
                loads0 = []
                dgs1 = []
                loads1 = []
                if len(net.bus) < 80:
                    # Ob0
                    load_options0 = [1, 2, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27, 30, 33, 34, 35, 36, 39, 41, 43, 44, 47, 51, 57, 58, 59]
                    load_options1 = [0, 3, 10, 11, 12, 13, 23, 25, 26, 28, 29, 31, 32, 37, 38, 40, 42, 45, 46, 48, 49, 50, 52, 53, 54, 55, 56, 60]
                    dg_options0 = [0, 1, 3, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 25, 27, 29, 36, 37, 38, 39, 43, 44, 45, 46, 47, 52, 53, 54]
                    dg_options1 = [2, 4, 5, 6, 8, 20, 21, 23, 26, 28, 30, 31, 32, 33, 34, 35, 40, 41, 42, 48, 49, 50, 51, 55, 56, 57, 58, 59]
                else:
                    # Ob1
                    load_options0 = [0, 1, 5, 6, 9, 10, 11, 14, 15, 19, 21, 25, 30, 32, 41, 42, 48, 49, 51, 52, 53, 57, 59, 68, 72, 74, 75, 79, 81, 82, 84]
                    load_options1 = [2, 3, 4, 7, 8, 12, 13, 16, 17, 18, 20, 22, 23, 24, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 50, 54, 55, 56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 73, 76, 77, 78, 80, 83, 85]
                    dg_options0 = [7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 29, 30, 38, 46, 47, 48, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 72, 81]
                    dg_options1 = [0, 1, 2, 3, 4, 5, 6, 11, 14, 22, 23, 26, 28, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 49, 50, 51, 52, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]


                if no_of_dgs0 > 0:
                    dgs0 = rng.choice(dg_options0, size=no_of_dgs0, replace=False)
                if no_of_loads0 > 0:
                    loads0 = rng.choice(load_options0, size=no_of_loads0, replace=False)
                if no_of_dgs1 > 0:
                    dgs1 = rng.choice(dg_options1, size=no_of_dgs1, replace=False)
                if no_of_loads1 > 0:
                    loads1 = rng.choice(load_options1, size=no_of_loads1, replace=False)
                dgs = np.concatenate([dgs0, dgs1]).astype(int)
                loads = np.concatenate([loads0, loads1]).astype(int)

                tot_p = 0
                tot_q = 0
                min_p = np.inf
                for vv in dgs:
                    tot_p += float(net.sgen['p_mw'][vv])
                    tot_q += float(net.sgen['q_mvar'][vv])
                    if min_p > float(net.sgen['p_mw'][vv]):
                        min_p = float(net.sgen['p_mw'][vv])
                for vv in loads:
                    tot_p += float(net.load['p_mw'][vv])
                    tot_q += float(net.load['q_mvar'][vv])
                    if min_p > float(net.load['p_mw'][vv]):
                        min_p = float(net.load['p_mw'][vv])
                if round((tot_p + tot_q)/26, 3) < min_p:
                    search = 1
                if its == 100000:
                    assert False, "Cannot find comp"
            settings.dp = round(min((tot_p + tot_q) / 26, min_p), 3)
            settings.dq = round(min((tot_p + tot_q) / 13, min_p * 2), 3)


            net_h = net.deepcopy()
            net_u = net.deepcopy()
            net_n = net.deepcopy()
            net_tc = net.deepcopy()
            settings.fsp_dg = dgs
            settings.fsp_load = loads
            print(dgs, loads)
            dist_dicts = {}
            for fffsp in dgs:
                dist_dicts[net.sgen['name'].iloc[fffsp]] = {}
                min_v = 1000
                min_d = ""
                for ffsp2 in dgs:
                    if fffsp != ffsp2:
                        dist_dicts[net.sgen['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp2]] = \
                            spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                        if spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]] < min_v:
                            min_v = spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                            min_d = net.sgen['name'].iloc[ffsp2]
                for ffsp3 in loads:
                    dist_dicts[net.sgen['name'].iloc[fffsp]][net.load['name'].iloc[ffsp3]] = \
                        spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                    if spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]] < min_v:
                        min_v = spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                        min_d = net.load['name'].iloc[ffsp3]
                dist_dicts[net.sgen['name'].iloc[fffsp]]["min"] = [min_v, min_d]
            for fffsp in loads:
                dist_dicts[net.load['name'].iloc[fffsp]] = {}
                min_v = 1000
                min_d = ""
                for ffsp2 in loads:
                    if fffsp != ffsp2:
                        dist_dicts[net.load['name'].iloc[fffsp]][net.load['name'].iloc[ffsp2]] = \
                            spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                        if spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]] < min_v:
                            min_v = spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                            min_d = net.load['name'].iloc[ffsp2]
                for ffsp3 in dgs:
                    dist_dicts[net.load['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp3]] = \
                        spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                    if spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]] < min_v:
                        min_v = spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                        min_d = net.sgen['name'].iloc[ffsp3]
                dist_dicts[net.load['name'].iloc[fffsp]]["min"] = [min_v, min_d]


            print(min_p, settings.dp)
            print(f"DGs: {dgs}, Loads: {loads}\n"
                  f"DP: {settings.dp}, DQ: {settings.dq}")
            pq_profiles, dur_samples_h = profile_creation(settings.no_samples, net_h, "Normal_Limits_Oriented",
                                                          settings.keep_mp, services="All", flexible_loads=loads,
                                                          flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_h, prf_flx, prf_non_flx = all_pf_simulations(settings, net_h,
                                                                                                  pq_profiles)
            res_hard = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            pq_profiles, dur_samples_u = profile_creation(settings.no_samples, net_u, "Uniform", settings.keep_mp,
                                                          services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_u, prf_flx, prf_non_flx = all_pf_simulations(settings, net_u,
                                                                                                  pq_profiles)
            res_un = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            pq_profiles, dur_samples_k = profile_creation(settings.no_samples, net_n, "Kumaraswamy", settings.keep_mp,
                                                          services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_k, prf_flx, prf_non_flx = all_pf_simulations(settings, net_n,
                                                                                                  pq_profiles)
            res_norm = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            _, _, _, _, no_samps = \
                profile_creation_bf(settings.dp, settings.dq, net, services="All",
                                    flexible_loads=loads, flexible_dgs=dgs, non_linear_dgs=settings.non_lin_dgs)
            print(f"Number of PFs needed for exhaustive search would be: {no_samps}")
            del pq_profiles
            del x_flx
            del y_flx
            del x_non_flx
            del y_non_flx
            del t_pf_k
            del prf_flx
            del prf_non_flx
            pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
                conv_profile_creation(settings.dp, settings.dq, net_tc, services="All", flexible_loads=loads,
                                      flexible_dgs=dgs)
            df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str, _ = \
                torch_tensor_conv_simulations(net_tc, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                              small_fsp_prof, min_max_v=[settings.min_volt, settings.max_volt],
                                              comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
            del pq_profiles
            mes = f"{0}: DGs:{dgs},Loads:{loads},DP:{settings.dp}, Tot: {tot_p + tot_q}, min: {min_p}"
            dur_tmp = dur_str.split("=")[1:]
            new_tmp = [dur_samples]
            for val in dur_tmp:
                new_tmp.append(float(val.split("s,")[0]))
            speed_res.append(new_tmp)
            acc_feas_h, acc_reach_h, acc_avg_h, hull_acc_h = loop_acc(res_hard, df, inf, f"Hard_{settings.name.replace('UC2/', '')}_{i}")
            acc_feas_u, acc_reach_u, acc_avg_u, hull_acc_u = loop_acc(res_un, df, inf, f"Uniform_{settings.name.replace('UC2/', '')}_{i}")
            acc_feas_n, acc_reach_n, acc_avg_n, hull_acc_n = loop_acc(res_norm, df, inf,
                                                                      f"Kumaraswamy_{settings.name.replace('UC2/', '')}_{i}")
            acc_res.append([acc_feas_h, acc_reach_h, acc_avg_h, hull_acc_h,
                            acc_feas_u, acc_reach_u, acc_avg_u, hull_acc_u,
                            acc_feas_n, acc_reach_n, acc_avg_n, hull_acc_n, mes])
            print(acc_feas_h, acc_reach_h, acc_avg_h, hull_acc_h)
        speed_df = pd.DataFrame(speed_res, columns=["Sampling Shifts", "Shape Preparation", "Power Flows",
                                                        "Net. Component vs FSP Dictionary", "Small FSPs Removal",
                                                        "Effective FSPs per Component", "Remove Safe Components",
                                                        "Tensors and Convolutions", "Applying Axes and Initial Point",
                                                        "Small FSP Uncertainty"])
        acc_df = pd.DataFrame(acc_res, columns=["Feasible Hard", "Reachable Hard", "Average Hard", "Hull Shift Hard"
            , "Feasible Uniform", "Reachable Uniform", "Average Uniform",
                                                    "Hull Shift Uniform", "Feasible Kumaraswamy",
                                                    "Reachable Kumaraswamy", "Average Kumaraswamy",
                                                    "Hull Shift Kumaraswamy", "FSPs"])
        speed_df.to_csv(f'../csv_results/UC2/Speed_r_{settings.name}')
        acc_df.to_csv(f'../csv_results/UC2/Acc_r_{settings.name}')
    else:
        for i in tqdm(range(settings.use_case_dict.get("no_scenarios"))):
            search = 0
            its = 0
            while search == 0:
                its += 1
                no_of_dgs = rng.randint(0, no_fsps + 1)
                no_of_loads = no_fsps - no_of_dgs
                if no_of_loads < 0:
                    assert AssertionError, "Error: No of loads should be >=0"
                dgs = []
                loads = []
                if no_of_dgs > 0:
                    dgs = rng.choice(dg_options, size=no_of_dgs, replace=False)
                if no_of_loads > 0:
                    loads = rng.choice(load_options, size=no_of_loads, replace=False)

                tot_p = 0
                tot_q = 0
                min_p = np.inf
                for vv in dgs:
                    tot_p += float(net.sgen['p_mw'][vv])
                    tot_q += float(net.sgen['q_mvar'][vv])
                    if min_p > float(net.sgen['p_mw'][vv]):
                        min_p = float(net.sgen['p_mw'][vv])
                for vv in loads:
                    tot_p += float(net.load['p_mw'][vv])
                    tot_q += float(net.load['q_mvar'][vv])
                    if min_p > float(net.load['p_mw'][vv]):
                        min_p = float(net.load['p_mw'][vv])
                if round((tot_p + tot_q) / 20, 3) < min_p:
                    search = 1
                if its == 100000:
                    assert False, "Cannot find comp"
            settings.dp = round((tot_p + tot_q)/20, 3)
            settings.dq = round((tot_p + tot_q)/10, 3)

            net_h = net.deepcopy()
            net_u = net.deepcopy()
            net_n = net.deepcopy()
            net_tc = net.deepcopy()
            settings.fsp_dg = dgs
            settings.fsp_load = loads
            tg2 = time.time()

            dist_dicts = {}
            for fffsp in dgs:
                dist_dicts[net.sgen['name'].iloc[fffsp]] = {}
                min_v = 1000
                min_d = ""
                for ffsp2 in dgs:
                    if fffsp != ffsp2:
                        dist_dicts[net.sgen['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp2]] = \
                            spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                        if spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]] < min_v:
                            min_v = spl[net.sgen['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp2]]
                            min_d = net.sgen['name'].iloc[ffsp2]
                for ffsp3 in loads:
                    dist_dicts[net.sgen['name'].iloc[fffsp]][net.load['name'].iloc[ffsp3]] = \
                        spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                    if spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]] < min_v:
                        min_v = spl[net.sgen['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp3]]
                        min_d = net.load['name'].iloc[ffsp3]
                dist_dicts[net.sgen['name'].iloc[fffsp]]["min"] = [min_v, min_d]
            for fffsp in loads:
                dist_dicts[net.load['name'].iloc[fffsp]] = {}
                min_v = 1000
                min_d = ""
                for ffsp2 in loads:
                    if fffsp != ffsp2:
                        dist_dicts[net.load['name'].iloc[fffsp]][net.load['name'].iloc[ffsp2]] = \
                            spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                        if spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]] < min_v:
                            min_v = spl[net.load['bus'].iloc[fffsp]][net.load['bus'].iloc[ffsp2]]
                            min_d = net.load['name'].iloc[ffsp2]
                for ffsp3 in dgs:
                    dist_dicts[net.load['name'].iloc[fffsp]][net.sgen['name'].iloc[ffsp3]] = \
                        spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                    if spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]] < min_v:
                        min_v = spl[net.load['bus'].iloc[fffsp]][net.sgen['bus'].iloc[ffsp3]]
                        min_d = net.sgen['name'].iloc[ffsp3]
                dist_dicts[net.load['name'].iloc[fffsp]]["min"] = [min_v, min_d]
            tg3 = time.time()

            print(f"DGs: {dgs}, Loads: {loads}\n"
                  f"DP: {settings.dp}, DQ: {settings.dq}")
            pq_profiles, dur_samples_h = profile_creation(settings.no_samples, net_h, "Normal_Limits_Oriented",
                                                          settings.keep_mp,
                                                          services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_h, prf_flx, prf_non_flx = all_pf_simulations(settings, net_h,
                                                                                                  pq_profiles)
            res_hard = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            pq_profiles, dur_samples_u = profile_creation(settings.no_samples, net_u, "Uniform", settings.keep_mp,
                                                          services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_u, prf_flx, prf_non_flx = all_pf_simulations(settings, net_u,
                                                                                                  pq_profiles)
            res_un = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            pq_profiles, dur_samples_k = profile_creation(settings.no_samples, net_n, "Kumaraswamy", settings.keep_mp,
                                                          services="All", flexible_loads=loads, flexible_dg=dgs)
            x_flx, y_flx, x_non_flx, y_non_flx, t_pf_k, prf_flx, prf_non_flx = all_pf_simulations(settings, net_n,
                                                                                                  pq_profiles)
            res_norm = create_result_df(x_flx, x_non_flx, y_flx, y_non_flx)
            _, _, _, _, no_samps = \
                profile_creation_bf(settings.dp, settings.dq, net, services="All",
                                    flexible_loads=loads, flexible_dgs=dgs, non_linear_dgs=settings.non_lin_dgs)
            print(f"Number of PFs needed for exhaustive search would be: {no_samps}")
            del pq_profiles
            del x_flx
            del y_flx
            del x_non_flx
            del y_non_flx
            del t_pf_k
            del prf_flx
            del prf_non_flx
            if settings.use_case_dict.get("Ver.") not in [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]:
                pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
                    conv_profile_creation(settings.dp, settings.dq, net_tc, services="All", flexible_loads=loads,
                                          flexible_dgs=dgs)
                df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str = \
                    torch_tensor_conv_simulations(net_tc, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                                  small_fsp_prof, settings.ttd, settings.multi,
                                                  min_max_v=[settings.min_volt, settings.max_volt],
                                                  comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens)
                mes = f"0: DGs:{dgs},Loads:{loads},DP:{settings.dp}, Tot: {tot_p+tot_q}, min: {min_p}"
            else:
                pq_profiles, small_fsp_prof, dur_samples, non_lin_dg_names = \
                    conv_profile_creation(settings.dp, settings.dq, net_tc, services="All", flexible_loads=loads,
                                          flexible_dgs=dgs)
                df, disc_df, cont_df, inf, ones_df, q_loc, p_loc, dur_str = \
                    torch_tensor_conv_large_simulations(net_tc, pq_profiles, settings.dp, settings.dq, pcc_operating_point,
                                                        small_fsp_prof, dist_dicts, ttd=settings.ttd, multi=settings.multi,
                                                        min_max_v=[settings.min_volt, settings.max_volt],
                                                        comp_fsp_v_sens=settings.v_sens, comp_fsp_l_sens=settings.l_sens,
                                                        no_max=no_fsps-1)
                mes = f"1: DGs:{dgs},Loads:{loads},DP:{settings.dp}, Tot: {tot_p+tot_q}, min: {min_p}"
            del pq_profiles
            dur_tmp = dur_str.split("=")[1:]
            new_tmp = [dur_samples]
            for val in dur_tmp:
                new_tmp.append(float(val.split("s,")[0]))
            new_tmp.append(tg3-tg2+tg1-tg0)
            speed_res.append(new_tmp)
            acc_feas_h, acc_reach_h, acc_avg_h = loop_acc(res_hard, df, inf, f"Hard_{settings.name}_{i}")
            acc_feas_u, acc_reach_u, acc_avg_u = loop_acc(res_un, df, inf, f"Uniform_{settings.name}_{i}")
            acc_feas_n, acc_reach_n, acc_avg_n = loop_acc(res_norm, df, inf, f"Kumaraswamy_{settings.name}_{i}")
            acc_res.append([acc_feas_h, acc_reach_h, acc_avg_h,
                            acc_feas_u, acc_reach_u, acc_avg_u,
                            acc_feas_n, acc_reach_n, acc_avg_n, mes])
        print(settings.use_case_dict.get("Ver."))
        print("---------------------------------")
        if settings.use_case_dict.get("Ver.") == 0:
            speed_df = pd.DataFrame(speed_res, columns=["Sampling Shifts", "Shape Preparation", "Power Flows",
                                                            "Net. Component vs FSP Dictionary", "Small FSPs Removal",
                                                            "Effective FSPs per Component", "Remove Safe Components",
                                                            "Tensors and Convolutions",
                                                            "Applying Axes and Initial Point",
                                                            "Small FSP Uncertainty", "Electrical Distance"])
            acc_df = pd.DataFrame(acc_res, columns=["Feasible Hard", "Reachable Hard", "Average Hard",
                                                        "Feasible Uniform", "Reachable Uniform", "Average Uniform",
                                                        "Feasible Kumaraswamy", "Reachable Kumaraswamy",
                                                        "Average Kumaraswamy",
                                                        "FSPs"])
            speed_df.to_csv(f'../csv_results/UC2/Speed_{settings.name}.csv')
            acc_df.to_csv(f'../csv_results/UC2/Acc_{settings.name}.csv')
        else:
            acc_csv = pd.read_csv(f'../csv_results/UC3/Acc_{settings.name}.csv')
            speed_csv = pd.read_csv(f'../csv_results/UC3/Speed_{settings.name}.csv')
            column = ["Sampling Shifts", "Shape Preparation", "Power Flows",
                       "Net. Component vs FSP Dictionary", "Small FSPs Removal",
                       "Effective FSPs per Component", "Remove Safe Components",
                       "Tensors and Convolutions", "Applying Axes and Initial Point",
                       "Small FSP Uncertainty"]
            speed_dct = {column[nn]: speed_res[0][nn] for nn in range(len(column))}

            speed_csv = speed_csv.iloc[:, 1:].append(speed_dct, ignore_index=True)
            column = ["Feasible Hard", "Reachable Hard", "Average Hard",
                       "Feasible Uniform", "Reachable Uniform", "Average Uniform",
                       "Feasible Kumaraswamy", "Reachable Kumaraswamy",
                       "Average Kumaraswamy",
                       "FSPs"]
            print(acc_csv)
            acc_dct = {column[nn]: acc_res[0][nn] for nn in range(len(column))}
            print(acc_dct)
            acc_csv = acc_csv.iloc[:, 1:].append(acc_dct, ignore_index=True)
            print(acc_csv)
            speed_csv.to_csv(f'../csv_results/UC3/Speed_{settings.name}.csv')
            acc_csv.to_csv(f'../csv_results/UC3/Acc_{settings.name}.csv')
    return




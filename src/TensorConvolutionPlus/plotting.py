#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import os


sns.set(style="white")
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['lines.dash_capstyle'] = 'butt'


__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def plot_opf_res(filename, init_pq, text):
    """ Plot the OPF results.

    :param filename: name of the OPF results to get for the plot.
    :type filename: str
    :param init_pq: initial active and reactive power of the PCC.
    :type init_pq: list[floats]
    :param text: information accompanying plot.
    :type text: str
    :return:
    :rtype:
    """
    res_pqs = pd.read_csv(f"./{filename}.csv").to_numpy()[:, 1:]
    sgns_lsts = [[], [], [], []]
    for idx, i in enumerate(list(res_pqs[:, 2])):
        sgns_lsts[int(i)].append(idx)
    res_pqs = res_pqs[:, :2]
    if len(res_pqs) > 0:
        try:
            hull = ConvexHull(res_pqs)
            plt.plot(res_pqs[sgns_lsts[0], 0], res_pqs[sgns_lsts[0], 1], 'o', c='blue')
            plt.plot(res_pqs[sgns_lsts[1], 0], res_pqs[sgns_lsts[1], 1], 'o', c='green')
            plt.plot(res_pqs[sgns_lsts[2], 0], res_pqs[sgns_lsts[2], 1], 'o', c='pink')
            plt.plot(res_pqs[sgns_lsts[3], 0], res_pqs[sgns_lsts[3], 1], 'o', c='purple')
            #plt.plot(res_pqs[:, 0], res_pqs[:, 1], 'o')
            for simplex in hull.simplices:
                plt.plot(res_pqs[simplex, 0], res_pqs[simplex, 1], 'k-')
            plt.plot(init_pq[0], init_pq[1], '*', c='red')
        except:
            plt.plot(res_pqs[:, 0], res_pqs[:, 1], 'o')
            plt.plot(init_pq[0], init_pq[1], '*', c='red')
    xmin, xmax, ymin, ymax = plt.axis()
    print(f"Saving figure in: {filename}.pdf")
    with open(f'./{filename}.txt', 'w') as f:
        f.write(text+f'\nFigure axis limits: x =[{xmin}, {xmax}], y=[{ymin}, {ymax}]')
    plt.savefig(f'./{filename}.pdf', bbox_inches='tight', pad_inches=0.5, dpi=500)
    return


def get_rel_dicts(net_name):
    """ Get dictionaries from results of UC2.

    :param net_name: network name.
    :type net_name: str

    :return: dictionaries with info, names of accuracy columns, names of time columns.
    :rtype: dict, list, list
    """
    file_list = os.listdir('../csv_results/UC2')
    relevant_files = []
    for i, file in enumerate(file_list):
        if net_name in str(file):
            relevant_files.append(file)
    relevant_dicts = {2: ['', ''], 3: ['', ''], 4: ['', ''], 5: ['', ''], 6: ['', ''], 7: ['', ''],
                      10: ['', ''], 11: ['', ''], 12: ['', ''], 13: ['', ''], 14: ['', ''], 15: ['', '']}
    for file in relevant_files:
        if 'v0911' in str(file):
            no = int(str(str(file).split("FSPs")[0][-2])+str(str(file).split("FSPs")[0][-1]))
        else:
            no = int(str(file).split("FSPs")[0][-1])
        if str(file)[:3] == 'Acc':
            relevant_dicts[no][0] = pd.read_csv(f'../csv_results/UC2/{file}')
        elif str(file)[:5] == 'Speed':
            relevant_dicts[no][1] = pd.read_csv(f'../csv_results/UC2/{file}')
        else:
            assert False, f"Error: Unknown file {file}"
    acc_cols = relevant_dicts[2][0].columns
    time_cols = relevant_dicts[2][1].columns
    return relevant_dicts, acc_cols, time_cols


def get_estimated_times():
    """ Get dictionaries from results of UC2.

    :param net_name: network name.
    :type net_name: str

    :return: dictionaries with info, names of accuracy columns, names of time columns.
    :rtype: dict, list, list
    """
    file = pd.read_csv(f'../csv_results/UC2/EstimatedTimes.csv', sep=';')
    fileOb0 = file.loc[file['Net'] == 'Ob0']
    fileOb1 = file.loc[file['Net'] == 'Ob1']
    filenOb0 = fileOb0.loc[fileOb0['No FSPs'] < 8]
    filewOb0 = fileOb0.loc[fileOb0['No FSPs'] > 8]
    return filenOb0, fileOb1, filewOb0


def plot_uc3(net1, net2):
    """ Plot use case 3 figures.

    :param net1: network 1.
    :type net1: str

    :param net2: network 2.
    :type net2: str

    :return:
    :rtype:
    """
    net1_dict, acc_cols, time_cols = get_rel_dicts(net1)
    net2_dict, acc_cols, time_cols = get_rel_dicts(net2)
    ob0_df, ob1_df, ob0_wdf = get_estimated_times()
    # plot 1: Box Feas Acc Per Net vs No of FSPs, plot 2: Box Reach Acc Per Net vs No of FSPs,
    # plot 3: Box Avg time per net vs no of fsps, plot 4: Box avg distribution of time per net vs part of algo
    pl1_dict = {net1: {2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 10: {}, 11: {}, 12: {}, 13: {}, 14: {}, 15: {}},
                net2: {2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}}}
    pl2_dict = {net1: {2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 10: {}, 11: {}, 12: {}, 13: {}, 14: {}, 15: {}},
                net2: {2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}}}
    pl3_dict = {net1: {}, net2: {}, net1+'W': {}}
    for key in net1_dict:
        if key >= 10:
            for acol in acc_cols:
                if "Feasible" in acol:
                    pl1_dict[net1][key][acol.replace("Feasible ", "")] = list(net1_dict[key][0][acol])
                elif "Reachable" in acol:
                    pl2_dict[net1][key][acol.replace("Reachable ", "")] = list(net1_dict[key][0][acol])
            tmp_df = net1_dict[key][1]
            tmp_df = tmp_df[tmp_df.columns[1:]]
            pl3_dict[net1+'W'][key] = list(tmp_df.sum(axis=1))
        else:
            for acol in acc_cols:
                if "Feasible" in acol:
                    pl1_dict[net1][key][acol.replace("Feasible ", "")] = list(net1_dict[key][0][acol])
                    pl1_dict[net2][key][acol.replace("Feasible ", "")] = list(net2_dict[key][0][acol])
                elif "Reachable" in acol:
                    pl2_dict[net1][key][acol.replace("Reachable ", "")] = list(net1_dict[key][0][acol])
                    pl2_dict[net2][key][acol.replace("Reachable ", "")] = list(net2_dict[key][0][acol])
            tmp_df = net1_dict[key][1]
            tmp_df = tmp_df[tmp_df.columns[1:]]
            tmp_df2 = net2_dict[key][1]
            tmp_df2 = tmp_df2[tmp_df2.columns[1:]]
            pl3_dict[net1][key] = list(tmp_df.sum(axis=1))
            pl3_dict[net2][key] = list(tmp_df2.sum(axis=1))
    ress1 = get_means12(pl1_dict)
    ress2 = get_means12(pl2_dict)
    print(f"Average Feasible Accuracy per FSP amount:")
    for key in ress1:
        print(f"Network {key}")
        for nf in ress1[key]:
            print(nf, ":", ress1[key][nf])
    print(f"Average Reachable Accuracy per FSP amount:")
    for key in ress2:
        print(f"Network {key}")
        for nf in ress2[key]:
            print(nf, ":", ress2[key][nf])
    box_plt3(pl3_dict, {'Oberrhein0': ob0_df, 'Oberrhein1' : ob1_df, 'Oberrhein0W': ob0_wdf})
    return


def get_means12(plt_dict):
    """ Get mean values from results.

    :param plt_dict: dictionary with results.
    :type plt_dict: dict

    :return: dictionary with mean values.
    :rtype: dict
    """
    mean_dict = {}
    for net in plt_dict:
        mean_dict[net] = {}
        for fsp_no in plt_dict[net]:
            mean_dict[net][fsp_no] = {}
            for dist in plt_dict[net][fsp_no]:
                mean_dict[net][fsp_no][dist] = np.mean(plt_dict[net][fsp_no][dist])
    return mean_dict


def box_plt3(plt1_dict, est_dict):
    """ Generate speed plots.

    :param plt1_dict: dictionary with speed information.
    :type plt1_dict: dict

    :param est_dict: estimation dictionary.
    :type est_dict: dict

    :return:
    :rtype:
    """
    plt.rcParams['svg.fonttype'] = 'none'

    for net in plt1_dict:
        fig = plt.figure(figsize=(4, 3))
        data = []
        labels = []
        min_y = 1
        for fsp_no in plt1_dict[net]:
            labels.append(fsp_no)
            data.append(plt1_dict[net][fsp_no])
        ax = fig.add_axes([0, 0, 1, 1])
        bp = sns.boxplot(data, color='grey')
        plt.xticks(np.arange(len(labels)), labels)
        plt.xlabel("No. FSPs")
        plt.ylabel("Time [s]")
        fig.savefig('../plots/UC2/' + net + '_speed_tot_uc3.svg', bbox_inches='tight', pad_inches=0.5,
                    dpi=500)
    for net in plt1_dict:
        fig = plt.figure(figsize=(4, 3))
        data = []
        est_data = []
        labels = []
        df_data = []
        min_y = 1
        for fsp_no in plt1_dict[net]:
            tmp_est = est_dict[net].loc[est_dict[net]['No FSPs'] == fsp_no]
            labels.append(fsp_no)
            data.append(np.mean(plt1_dict[net][fsp_no]))
            est_data.append(float(tmp_est.loc[:, 'Time [s]'].mean()))
        df_data = pd.DataFrame(np.stack((labels+labels, data + est_data, list(np.zeros_like(labels))+list(np.ones_like(labels))), axis=-1), columns=['|\Omega^{FSP}|', 'Time[s]', 'Type'])
        ax = fig.add_axes([0, 0, 1, 1])
        #sns.scatterplot(df_data, x='|\Omega^{FSP}|', y='Time[s]', hue='Type', markers=['o', 'x'], color=['black', 'black'])
        ax.scatter(labels, est_data, s=60, alpha=0.7, c="black", marker='x')
        ax.scatter(labels, data, s=60, alpha=0.7, c="black")
        ax.set_yscale("log")
        #plt.xticks(np.arange(len(labels)), labels)
        plt.xlabel("No. FSPs")
        plt.ylabel("Time [s]")
        ax.figure.savefig('../plots/UC2/' + net + '_speed_log.svg', bbox_inches='tight', pad_inches=0.5,
                    dpi=500)
    return


def box_plt1(plt1_dict, acc_type):
    """ Plot accuracy information.

    :param plt1_dict: dictionary with accuracy info.
    :type plt1_dict: dict

    :param acc_type: type of accuracy.
    :type acc_type: str

    :return:
    :rtype:
    """
    for net in plt1_dict:
        fig = plt.figure(figsize=(7, 3))
        data = []
        labels = []
        min_y = 1
        for fsp_no in plt1_dict[net]:
            for dist in plt1_dict[net][fsp_no]:
                labels.append(dist[0])
                data.append(plt1_dict[net][fsp_no][dist])
                if min(plt1_dict[net][fsp_no][dist]) < min_y:
                    min_y = min(plt1_dict[net][fsp_no][dist])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.text(8., 1.02, "No. FSPs")
        ax.text(1., 1.01, "2")
        ax.text(4., 1.01, "3")
        ax.text(7., 1.01, "4")
        ax.text(10., 1.01, "5")
        ax.text(13., 1.01, "6")
        ax.text(16., 1.01, "7")
        plt.plot([2.5, 2.5], [min_y, 1], color='grey', linestyle='dashed')
        plt.plot([5.5, 5.5], [min_y, 1], color='grey', linestyle='dashed')
        plt.plot([8.5, 8.5], [min_y, 1], color='grey', linestyle='dashed')
        plt.plot([11.5, 11.5], [min_y, 1], color='grey', linestyle='dashed')
        plt.plot([14.5, 14.5], [min_y, 1], color='grey', linestyle='dashed')

        bp = sns.boxplot(data)
        plt.xticks(np.arange(len(labels)), labels)
        plt.xlabel("Distribution")
        plt.ylabel("Accuracy")
        fig.savefig('../plots/UC2/'+ acc_type + net + '_acc_uc3.svg', bbox_inches='tight', pad_inches=0.5,
                    dpi=500)
    return


def plot_mc(x_flexible, y_flexible, x_non_flexible, y_non_flexible, operating_point, no_samples, scenario_name,
            dur_samples, dur_pf, loc=''):
    """Plot Monte Carlo simulation results.

    :param x_flexible: feasible P.
    :type x_flexible: list

    :param y_flexible: feasible Y.
    :type y_flexible: list

    :param x_non_flexible: infeasible P.
    :type x_non_flexible: list

    :param y_non_flexible: infeasible Q.
    :type y_non_flexible: list

    :param operating_point: initial PCC p,q.
    :type operating_point: list[float, float]

    :param no_samples: number of samples run.
    :type no_samples: int

    :param scenario_name: name of used scenario.
    :type scenario_name: str

    :param dur_samples: duration for sample creations [s].
    :type dur_samples: float

    :param dur_pf: duration for power flows [s].
    :type dur_pf: float

    :param loc: location on disc.
    :type loc: str

    :return:
    :rtype:
    """
    text = f'Total duration: {dur_samples + dur_pf} [s]\n{no_samples} Sample creation duration: {dur_samples} [s]' \
           f'\nTotal Power Flow duration: {dur_pf} [s]'
    plot_only_feasible(x_flexible, y_flexible, operating_point, scenario_name, loc)
    plot_feasible_and_infeasible(x_flexible, y_flexible, x_non_flexible, y_non_flexible, operating_point,
                                 scenario_name, text, loc)
    return


def plot_feasible_and_infeasible(x_flexible, y_flexible, x_non_flexible, y_non_flexible, operating_point,
                                 scenario_name, text, loc=''):
    """ Plot Monte Carlo simulation results including infeasible samples.

    :param x_flexible: feasible P.
    :type x_flexible: list

    :param y_flexible: feasible Y.
    :type y_flexible: list

    :param x_non_flexible: infeasible P.
    :type x_non_flexible: list

    :param y_non_flexible: infeasible Q.
    :type y_non_flexible: list

    :param operating_point: initial PCC p,q.
    :type operating_point: list[float, float]

    :param scenario_name: name of used scenario.
    :type scenario_name: str

    :param text: information accompanying plot.
    :type text: str

    :param loc: location on disc.
    :type loc: str

    :return:
    :rtype:
    """
    sns.set(style="white")
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(x_non_flexible, y_non_flexible, c="darkblue", s=10, label='Non Feasible Samples', rasterized=True)
    plt.scatter(x_flexible, y_flexible, c="darkorange", s=10, label='Feasible Samples', rasterized=True)
    plt.scatter(operating_point[0], operating_point[1], marker='*', c="red", s=50, label='Operating Point',
                rasterized=True)
    plt.grid()
    plt.xlabel("P [MW]")
    plt.ylabel("Q [MVAR]")
    plt.grid()
    xmin, xmax, ymin, ymax = plt.axis()
    print(f"Saving figure at: " + loc + scenario_name + '_incl_infeasible.pdf')
    with open(loc+scenario_name+'.txt', 'w') as f:
        f.write(text+f'\nFigure axis limits: x =[{xmin}, {xmax}], y=[{ymin}, {ymax}]')
    fig.savefig(loc + scenario_name + '_incl_infeasible.pdf', bbox_inches='tight', pad_inches=0.5, dpi=500,
                format='pdf')
    return


def plot_only_feasible(x_flexible, y_flexible, operating_point, scenario_name, loc=''):
    """ Plot Monte Carlo simulation results only for feasible samples.

    :param x_flexible: feasible P.
    :type x_flexible: list

    :param y_flexible: feasible Y.
    :type y_flexible: list

    :param operating_point: initial PCC p,q.
    :type operating_point: list[float, float]

    :param scenario_name: name of used scenario.
    :type scenario_name: str

    :param loc: location on disc.
    :type loc: str

    :return:
    :rtype:
    """
    sns.set(style="white")
    plt.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(x_flexible, y_flexible, c="darkorange", s=10, label='Feasible Samples', rasterized=True)
    plt.scatter(operating_point[0], operating_point[1], marker='*',  c="red", s=50, label='Operating Point',
                rasterized=True)
    plt.grid()
    plt.xlabel("P [MW]")
    plt.ylabel("Q [MVAR]")
    plt.grid()
    print(f"Saving figure at: "+loc+scenario_name+'.pdf')
    fig.savefig(loc+scenario_name+'.pdf', bbox_inches='tight', pad_inches=0.5, dpi=500, format='pdf')
    return


def plot_multi_convolution(filename: str, inf, q_loc, p_loc, text, loc=''):
    """ Plot convolution algorithm result.

    :param filename: name of file with results.
    :type filename: str

    :param inf: dataframe with infeasible samples.
    :type inf: pandas.dataframe

    :param q_loc: height pixel of initial Q of PCC.
    :type q_loc: int

    :param p_loc: width pixel of initial P of PCC.
    :type p_loc: int

    :param text: information accompanying plot.
    :type text: str

    :param loc: location on disc.
    :type loc: str

    :return:
    :rtype:
    """
    plt.locator_params(nbins=10)

    inf = inf[inf.columns[:]].iloc[::-1]
    inf = inf[inf.columns[::-1]]
    sns.set(style="white")

    df = pd.read_csv(f"./{filename}.csv").iloc[::-1]
    df = df[df.columns[::-1]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.set(font_scale=1)

    if df[df.columns[:-1]].to_numpy().max() == 0 and False:
        cmap = LinearSegmentedColormap.from_list('', ["lightgrey", "darkblue", "red"], 3)
        sns.heatmap(inf, cmap=cmap, center=1, cbar=False,
                    xticklabels=[round(float(a), 2) for a in list(df.columns[:-1])],
                    yticklabels=[round(float(a), 2) for a in list(df[df.columns[-1]])])
    else:
        cmap = LinearSegmentedColormap.from_list('', ["white", "darkblue", "red"], 3)
        sns.heatmap(inf, cmap=cmap, center=1, cbar=False, rasterized=True)
        p_loc = len(df.columns)-1-int(p_loc)
        q_loc = len(df[df.columns[-1]])-int(q_loc)
        xticks = [str(round(float(a), 3)) for a in list(df.columns[:-1])]
        yticks = [str(round(float(a), 3)) for a in list(df[df.columns[-1]])]
        for dsda in range(len(xticks)):
            if dsda % 5 != 0:
                xticks[dsda] = ''
        for dsda in range(len(yticks)):
            if dsda % 5 != 0:
                yticks[dsda] = ''
        sns.heatmap(df[df.columns[:-1]], linewidths=0.1, linecolor='white', norm=LogNorm(), cmap='flare', center=1,
                    xticklabels=xticks,
                    yticklabels=yticks, rasterized=True)

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width * fig.dpi, bbox.height * fig.dpi  # Width and height in pixels
        cell_size = min(width / len(xticks), height / len(yticks))  # Approximate cell size in pixels
        star_size = cell_size*40  # Adjust factor as needed
        # Plot Init OP
        cmap = LinearSegmentedColormap.from_list('', ["red"], 1)
        ax.scatter(p_loc-0.5, q_loc-0.5, marker='*', s=star_size, color='red', rasterized=True)

    ax.grid()
    ax.set_xlabel("P [MW]", fontsize=12)
    ax.set_ylabel("Q [MVAR]", fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    ax.invert_yaxis()
    plt.tight_layout()
    print(f"Saving figure in: "+f"./{filename}.pdf")
    with open(f"./{filename}.txt", 'w') as f:
        f.write(text+f'\nFigure axis limits: x =[{list(df.columns[:-1])[0]}, {list(df.columns[:-1])[-1]}], '
                     f'y=[{list(df[df.columns[-1]])[0]}, {list(df[df.columns[-1]])[-1]}]')

    plt.savefig(f'./{filename}.pdf', bbox_inches='tight', pad_inches=0.5, dpi=500)
    return


def get_uncertainty(df, name='', plot_type='jpg'):
    """ Plot results including uncertainty.

    :param df: dataframe with results.
    :type df: pandas.dataframe

    :param name: name for saving.
    :type name: str

    :param plot_type: type of figure to be saved.
    :type plot_type: str

    :return:
    :rtype:
    """
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df, norm=LogNorm(), cmap='flare',
                xticklabels=[round(float(a), 2) for a in list(df.columns)],
                yticklabels=[round(float(a), 2) for a in list(df.index)]
                )
    ax.grid()
    ax.set_xlabel("P [MW]", fontsize=12)
    ax.set_ylabel("Q [MVAR]", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    print(f'Plotting small FSPs at: Uncertainty'+name+'.'+plot_type)
    fig.savefig('Uncertainty'+name+'.'+plot_type, bbox_inches='tight', pad_inches=0.5, dpi=500)
    return


def get_uncertainty_interpret(uncert_df, safe_df, ones_df, name='', plot_type='pdf', extra=[]):
    """ Plot results including uncertainty.

    :param uncert_df: uncertainty dataframe.
    :type uncert_df: pandas.dataframe

    :param safe_df: certain dataframe.
    :type safe_df: pandas.dataframe

    :param ones_df: reachable dataframe.
    :type ones_df: pandas.dataframe

    :param name: name to save results.
    :type name: str

    :param plot_type: type of plot to save in.
    :type plot_type: str

    :param extra: initial P,Q values pixel location.
    :type extra: list

    :return:
    :rtype:
    """
    uncert_df.replace(0, np.nan, inplace=True)
    safe_df.replace(0, np.nan, inplace=True)
    ones_df.replace(0, np.nan, inplace=True)

    sns.set(style="white")
    sns.light_palette("seagreen", as_cmap=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = LinearSegmentedColormap.from_list('', ['darkblue', "darkblue"], 2)
    sns.heatmap(ones_df, norm=LogNorm(), cmap=cmap, ax=ax, cbar=False, rasterized=True)
    sns.heatmap(uncert_df, norm=LogNorm(), cmap="Greens", ax=ax, rasterized=True, vmin=0., vmax=1.2)
    itx = list(safe_df.columns)
    itx.reverse()
    ity = list(safe_df.index)
    ity.reverse()
    x_tick_labs = [round(float(a), 3) for a in itx]
    for i in range(1, len(x_tick_labs)):
        if i % 20 != 0:
            x_tick_labs[i] = None
    y_tick_labs = [round(float(a), 3) for a in ity]
    for i in range(1, len(y_tick_labs)):
        if i % 20 != 0:
            y_tick_labs[i] = None
    sns.heatmap(safe_df, linewidths=0.1, linecolor='white', norm=LogNorm(), cmap='flare', center=1,
                xticklabels=x_tick_labs,  vmin=0, vmax=2,
                yticklabels=y_tick_labs, rasterized=True)
    ax.scatter(extra[0], extra[1],  label='Initial operating point', s=300,   marker='*', color='red', rasterized=True)

    ax.grid()
    ax.set_xlabel("P [MW]", fontsize=12)
    ax.set_ylabel("Q [MVAR]", fontsize=12)
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print(f'Plotting small FSPs at: Uncertainty_Interpreted'+name+'.'+plot_type)
    fig.savefig('Uncertainty_Interpreted'+name+'.'+plot_type, bbox_inches='tight', pad_inches=0.5, dpi=500)
    return




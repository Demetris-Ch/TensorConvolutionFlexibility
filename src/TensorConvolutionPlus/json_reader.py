#!/usr/bin/env python
import os
import json


__author__ = "Demetris Chrysostomou"
__credits__ = ["Demetris Chrysostomou", "Jose Luis Rueda Torres", "Jochen Lorenz Cremer"]
__version__ = "0.1.0"
__maintainer__ = "Demetris Chrysostomou"
__email__ = "D.Chrysostomou@tudelft.nl"
__status__ = "Production"


def default_dict():
    return {"name": "Default",
     "scenario_settings": {
         "network": "MV Oberrhein0",
         "dp": 0.05,
         "dq": 0.05,
         "resolution": 100,
         "distribution": "Thorough",
         "TTD": False,
         "keep_mp": False,
         "max_curr_per": 100,
         "max_volt_pu": 1.05,
         "min_volt_pu": 0.95,
         "l_sens": 0.1,
         "v_sens": 0.0001,
         "Convolution_simulation": True,
         "accuracy": False,
         "ground_truth_file": "Flexibility_area_Unaltered_Model_Scenario",
         "multiplicity": True,
         "uncertainty": False,
         "save_tensors": False,
         "adapt": False,
         "FSPs": "All",
         "FSP_load_indices": [57],
         "FSP_DG_indices": [29],
         "Non-Linear DGs": [],
         "observable_lines_indices": [-1],
         "observable_buses_indices": [-1],
         "scenario_type": {"name": "CS",
                           "no.": 0},
         "plot_settings": {"convex_hull": False,
                           "plot_combination": False,
                           "multiplicity": False,
                           "convolution": True,
                           "filenames": [],
                           "output type": "png"
                           }
     }
     }


class SettingReader:
    """ Class parsing through the json scenario input files, checking their validity and saving their information.
    """
    def __init__(self, scenario_name='unaltered_network'):
        """ Initialize the class, read the input scenario file, test if the information is as expected, and save
        it for further usage.

        :param scenario_name: name of the scenario file within the scenarios folder (excluding the folder and the .json).
        :type scenario_name: str
        """

        if 'Default' not in scenario_name:
            cwd = os.getcwd()  # Get the current working directory (cwd)
            files = os.listdir(cwd)
            f = open(f'./{scenario_name}.json')
            self.data = json.load(f)
        else:
            self.data = default_dict()
        self.name = self.data.get('name', 'Unnamed').replace(' ', '_')
        self.scenario_settings = self.data.get('scenario_settings', {})
        self.net_name = self.scenario_settings.get("network", "CIGRE MV")
        self.lib = self.scenario_settings.get("library", "Pandapower")
        self.ttd = self.scenario_settings.get("TTD", False)
        self.no_samples = self.scenario_settings.get("no_samples", 100)
        self.resolution = self.scenario_settings.get("resolution", 100)
        self.distribution = self.scenario_settings.get("distribution", "Normal_Limits_Oriented")
        self.keep_mp = self.scenario_settings.get("keep_mp", False)
        self.max_curr = self.scenario_settings.get("max_curr_per", 100)
        self.max_volt = self.scenario_settings.get("max_volt_pu", 1.05)
        self.min_volt = self.scenario_settings.get("min_volt_pu", 0.95)
        self.dp = self.scenario_settings.get("dp", 0.01)
        self.dq = self.scenario_settings.get("dq", 0.01)
        self.v_sens = self.scenario_settings.get("v_sens", 0.001)
        self.l_sens = self.scenario_settings.get("l_sens", 1)
        self.mc_sim = self.scenario_settings.get("Monte_Carlo_simulation", False)
        self.conv_sim = self.scenario_settings.get("Convolution_simulation", False)
        self.brute_force = self.scenario_settings.get("brute_force", False)
        self.uc6 = self.scenario_settings.get("uc6", False)
        self.uc7 = self.scenario_settings.get("uc7", False)
        self.compare_brute = self.scenario_settings.get("compare_brute", False)
        self.multi = self.scenario_settings.get("multiplicity", False)
        self.uncertainty = self.scenario_settings.get("uncertainty", False)
        self.save_tensors = self.scenario_settings.get("save_tensors", False)
        self.adapt = self.scenario_settings.get("adapt", False)
        self.opf = self.scenario_settings.get("OPF", False)
        self.max_fsps = self.scenario_settings.get("max_fsps", -1)
        self.opf_step = self.scenario_settings.get("opf_step", 0.1)
        self.accuracy = self.scenario_settings.get("accuracy", False)
        self.ground_truth = self.scenario_settings.get("ground_truth_file", None)
        self.fsps = self.scenario_settings.get("FSPs", "All")
        self.fsp_wt = self.scenario_settings.get("FSP_WT_indices", [-1])
        self.fsp_pv = self.scenario_settings.get("FSP_PV_indices", [-1])
        self.fsp_load = self.scenario_settings.get("FSP_load_indices", [-1])
        self.fsp_dg = self.scenario_settings.get("FSP_DG_indices", [-1])
        self.non_lin_dgs = self.scenario_settings.get("Non-Linear DGs", [])
        self.observ_lines = self.scenario_settings.get("observable_lines_indices", [0, 1, 10, 11])
        self.observ_buses = self.scenario_settings.get("observable_buses_indices", [0, 1, 2, 3, 12, 13, 14])
        self.scenario_type_dict = self.scenario_settings.get("scenario_type", {})
        self.plot_settings_dict = self.scenario_settings.get("plot_settings", {})
        self.compare_settings = self.scenario_settings.get("compare_settings", {})
        self.scale_pv = self.scenario_settings.get("scale_pv", 1)
        self.scale_wt = self.scenario_settings.get("scale_wt", 1)
        self.flex_shape = self.scenario_settings.get("flex_shape", "Smax")
        self.use_case_dict = self.scenario_settings.get("UseCase", {})
        self.tester()
        return

    def tester(self):
        """ Test that the imported information from the file is as expected and should not cause issues later.

        :return:
        :rtype:
        """
        if self.scenario_settings == {}:
            assert False, "Please give scenario settings within the input json file"
        if self.net_name not in ["CIGRE MV", "Four bus", "MV Open Ring", "MV Closed Ring", "MV Oberrhein",
                                 "MV Oberrhein0", "MV Oberrhein1"]:
            assert False, f"Currently only 'CIGRE MV', 'Four bus', 'MV Open Ring', 'MV Closed Ring', 'MV Oberrhein', " \
                          f"'MV Oberrhein0', 'MV Oberrhein1' networks are supported, not {self.net_name}"
        if self.lib not in ["Pandapower", "Aliander"]:
            assert False, f"Currently only 'Pandapower' and 'Aliander' are supported, not {self.lib}"
        if type(self.no_samples) != int:
            assert False, f"Only integer number of samples supported, not {type(self.no_samples)}"
        if type(self.max_fsps) != int:
            assert False, f"Only integer maximum number of FSPs supported, not {type(self.max_fsps)}"
        if self.distribution not in ["Normal_Limits_Oriented", "Uniform", 'Kumaraswamy', "Thorough"]:
            assert False, f"Only 'Normal_Limits_Oriented', 'Uniform', 'Kumaraswamy', and 'Thorough' " \
                          f"distributions supported, not {self.distribution}"
        if self.keep_mp not in [True, False]:
            assert False, f"keep_mp should be given as a boolean 'true/false' not {self.keep_mp}"
        if self.multi not in [True, False]:
            assert False, f"multiplicity should be given as a boolean 'true/false' not {self.multi}"
        if self.uncertainty not in [True, False]:
            assert False, f"uncertainty should be given as a boolean 'true/false' not {self.uncertainty}"
        if self.save_tensors not in [True, False]:
            assert False, f"save_tensors should be given as a boolean 'true/false' not {self.save_tensors}"
        if self.adapt not in [True, False]:
            assert False, f"adapt should be given as a boolean 'true/false' not {self.adapt}"
        if self.ttd not in [True, False]:
            assert False, f"TTD should be given as a boolean 'true/false' not {self.ttd}"
        if type(self.max_curr) != int and type(self.max_curr) != float:
            assert False, f"Only integer or float maximum component loading supported, not {type(self.max_curr)}"
        if type(self.max_volt) != float:
            assert False, f"Only float maximum voltage supported, not {type(self.max_volt)}"
        if type(self.min_volt) != float:
            assert False, f"Only float minimum voltage supported, not {type(self.min_volt)}"
        if type(self.opf_step) != float:
            assert False, f"Only float OPF step size is supported, not {type(self.opf_step)}"
        elif self.opf_step > 1:
            assert False, f"OPF step size must be between 0-1 p.u., not {self.opf_step}"
        if self.mc_sim not in [True, False]:
            assert False, f"Monte_Carlo_simulation should be given as a boolean 'true/false' not {self.mc_sim}"
        if self.conv_sim not in [True, False]:
            assert False, f"Convolution_simulation should be given as a boolean 'true/false' not {self.conv_sim}"
        if self.opf not in [True, False]:
            assert False, f"OPF should be given as a boolean 'true/false' not {self.conv_sim}"
        if self.brute_force not in [True, False]:
            assert False, f"brute_force should be given as a boolean 'true/false' not {self.brute_force}"
        if self.compare_brute not in [True, False]:
            assert False, f"compare_brute should be given as a boolean 'true/false' not {self.compare_brute}"
        if self.fsps not in ['DG only', 'Load only', 'All']:
            assert False, f"FSPs currently supported are 'All' 'DG only', or 'Load only', not {self.fsps}"
        if type(self.fsp_wt) != list:
            assert False, f"Please set WT indices which are FSP as a list, not {type(self.fsp_wt)}"
        if type(self.fsp_wt) != list:
            assert False, f"Please set WT indices which are FSP as a list, not {type(self.fsp_wt)}"
        if type(self.fsp_pv) != list:
            assert False, f"Please set PV indices which are FSP as a list, not {type(self.fsp_pv)}"
        if type(self.fsp_load) != list:
            assert False, f"Please set load indices which are FSP as a list, not {type(self.fsp_load)}"
        if type(self.fsp_dg) != list:
            assert False, f"Please set DG indices which are FSP as a list, not {type(self.fsp_dg)}"
        if type(self.compare_settings) != dict:
            assert False, f"Please set compare_settings as a dict, not {type(self.compare_settings)}"
        if type(self.observ_buses) != list:
            assert False, f"Please set observable bus indices as a list, not {type(self.observ_buses)}"
        if type(self.resolution) != int:
            assert False, f"Please set resolution as an integer, not {type(self.resolution)}"
        if type(self.dp) != float:
            assert False, f"Please set DP as a float, not {type(self.dp)}"
        if type(self.dq) != float:
            assert False, f"Please set DQ as a float, not {type(self.dq)}"
        if type(self.flex_shape) != str:
            assert False, f"Please set {self.flex_shape} as string, not {type(self.flex_shape)}"
        elif self.flex_shape not in ['Smax', 'PQmax']:
            assert False, f"Flexibility shape {self.flex_shape} not available. Select one of ['Smax', 'PQmax']. " \
                          f"Alternatively, you can create a new function instead of conv_profile_creation() " \
                          f"to accommodate other shapes."
        if self.accuracy not in [True, False]:
            assert False, f"accuracy should be given as a boolean 'true/false' not {self.accuracy}"
        elif self.accuracy and type(self.ground_truth) != str:
            assert False, f"ground_truth_file should be a string not {type(self.ground_truth)}"
        return

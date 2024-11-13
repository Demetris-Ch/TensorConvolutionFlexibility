======
Usage
======

---------------
I. Reference
---------------
The package has been submitted in SoftwareX and is currently available in ArXiv under:

[![DOI](https://doi.org/10.1109/TSG.2024.3453667)

Available at: https://ieeexplore.ieee.org/document/10663439

The TensorConvolution+ algorithm was published in IEEE Transactions on Smart Grid and can be read and referenced using:

[![DOI](https://doi.org/10.1109/TSG.2024.3453667)

Available at: https://ieeexplore.ieee.org/document/10663439

---------------------
II. Acknowledgements
---------------------
The authors are part of the IEPG group and AI Energy lab of TU Delft.
This research is part of the research program 'MegaMind - Measuring, Gathering, Mining and Integrating Data for Self-management in the Edge of the Electricity System', (partly) financed by the Dutch Research Council (NWO) through the Perspectief program under number P19-25.

.. image:: //../plots/IEPG_logo.jpg
  :width: 150
.. image:: //../plots/DAI_Energy_logo.png
  :width: 150
.. image:: //../plots/Tu-delft.png
  :width: 150
.. image:: //../plots/MegaMindLogo.png
  :width: 120


---------------------
III. Package
---------------------
The proposed package includes the functionalities to perform flexibility area estimation.

To install the package through a file, place the **tensorconvolutionplus-0.1.0.tar.gz** file in an accessible path and run:

.. code-block:: console

   (.venv) $ pip install path/to/package/tensorconvolutionplus-0.1.0.tar.gz

To install the package online (currently unavailable), run:

.. code-block:: console

   (.venv) $ pip install tensorconvolutionplus

Main functionalities include:

#. TensorConvolution+, estimating FAs using the TensorConvolution+ algorithm. In the current version, the estimations can vary in:
    - Pandapower network (names) for MV Oberrhein0, MV Oberrhein1, Cigre MV. If another network is in similar format as these pandapower networks, it can also be an input instead.
    - Resolutions for the discretization of the flexibility resources.
    - Network voltage and loading constraints.
    - Voltage and loading sensitivity thresholds.
    - Including/Excluding FSPs only offering full output reductions, or limited setpoints ( including these FSPs currently uses the numpy library and not pytorch).
    - Flexibility service providers.
    - Network structure and initial operating conditions.
    - Shape of flexibility resources. Currently FSP limits are:
        - the output cannot exceed the maximum apparent power of the FSP (resulting in a semi-oval shape),
        - the output P cannot exceed the maximum apparent power of the FSP, the output abs(Q) cannot exceed the maximum apparent power of the FSP (resulting in rectangular shape).
        - additional shapes can be adopted by modifying the sample generation function (not impacting the TensorConvolution+ aggregation).

#. TensorConvolution+, while storing estimated tensors and other useful information to adapt flexibility areas for different operating conditions. In the current version, the estimations can vary in:
    - Pandapower network (names) for MV Oberrhein0, MV Oberrhein1, Cigre MV. If another network is in similar format as these pandapower networks, it can also be an input instead.
    - Resolutions for the discretization of the flexibility resources.
    - Network voltage and loading constraints.
    - Voltage and loading sensitivity thresholds.
    - Flexibility service providers.
    - Network structure and initial operating conditions.
    - Shape of flexibility resources. Currently FSP limits are:
        - the output cannot exceed the maximum apparent power of the FSP (resulting in a semi-oval shape),
        - the output P cannot exceed the maximum apparent power of the FSP, the output abs(Q) cannot exceed the maximum apparent power of the FSP (resulting in rectangular shape).
        - additional shapes can be adopted by modifying the sample generation function (not impacting the TensorConvolution+ aggregation).

#. TensorConvolution+, while loading previously estimated tensors and other useful information to adapt flexibility areas from prior different operating conditions. In the current version, the estimations can vary in:
    - Pandapower network (names) for MV Oberrhein0, MV Oberrhein1, Cigre MV. If another network is in similar format as these pandapower networks, it can also be an input instead. The network must be the same as the stored one.
    - Resolutions for the discretization of the flexibility resources. Must be the same as the stored simulation.
    - Network voltage and loading constraints.
    - Flexibility service providers.  Must be the same as the stored simulation.
    - Network structure and initial operating conditions.
    - Shape of flexibility resources.  Must be the same as the stored simulation.

#. Monte Carlo power flow based flexibility area estimation. In the current version, the estimations can vary in:
    - Pandapower network (names) for MV Oberrhein0, MV Oberrhein1, Cigre MV. If another network is in similar format as these pandapower networks, it can also be an input instead.
    - Network voltage and loading constraints.
    - Number of samples.
    - Distribution used for samples, including:
        - 'Hard': Exploring the limit from each resource flexibility.
        - 'Uniform': Applying uniform distribution.
        - 'Kumaraswamy': Applying the Kumaraswamy distribution.
    - Flexibility service providers.
    - Including/Excluding FSPs only offering full output reductions, or limited setpoints (including these FSPs currently uses the numpy library and not pytorch).
    - Network structure and initial operating conditions.

#. Exhaustive power flow based flexibility area estimation. In the current version, the estimations can vary in:
    - Pandapower network (names) for MV Oberrhein0, MV Oberrhein1, Cigre MV. If another network is in similar format as these pandapower networks, it can also be an input instead.
    - Network voltage and loading constraints.
    - Resolutions for the discretization of the flexibility resources.
    - Flexibility service providers.
    - Including/Excluding FSPs only offering full output reductions, or limited setpoints (including these FSPs currently uses the numpy library and not pytorch).
    - Network structure and initial operating conditions.

#. Optimal power flow based flexibility area estimation. In the current version, the estimations can vary in:
    - Pandapower network. Cigre MV in radial structure converges whereas alternative networks might fail to converge.
    - Network voltage and loading constraints. Transformer loading is excluded due to convergence issues.
    - Flexibility service providers.
    - Network structure and initial operating conditions.


------------------
IV. Examples
------------------
All main functionalities require first importing the FA_Estimator script from the package. Therefore, all the following examples start with the Python line:

.. code-block:: console

    from TensorConvolutionPlus import FA_Estimator as TCP

IV.A) Monte Carlo Power Flow
---------------------------------------
This section includes examples using the Monte Carlo PF estimation functionality. These examples used the Python script code:

.. code-block:: console

    TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=6000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3], distribution='Uniform')

    TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=6000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3], distribution='Kumaraswamy')

    TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=6000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

    TCP.monte_carlo_pf(net_name='MV Oberrhein0', no_samples=12000, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

The examples vary in sampling distribution and number of samples.
The figures bellow illustrate the resulting FA for each line respectively. The lines without *distribution* input automatically obtain the 'Hard' distribution.

.. image:: //../plots/MonteCarlo2024-11-08_14-30-32_incl_infeasible.jpg
  :width: 400
.. image:: //../plots/MonteCarlo2024-11-08_14-25-40_incl_infeasible.jpg
  :width: 400
.. image:: //../plots/MonteCarlo2024-11-05_17-06-58_incl_infeasible.jpg
  :width: 400
.. image:: //../plots/MonteCarlo2024-11-08_13-15-01_incl_infeasible.jpg
  :width: 400

IV.B) Exhaustive Power Flow
---------------------------------------

This section includes examples using the exhaustive power flow-based functionality. The script for the examples is:

.. code-block:: console

    TCP.exhaustive_pf(net_name='MV Oberrhein0', dp=0.15, dq=0.3, fsp_load_indices=[1, 2, 3], fsp_dg_indices=[1, 2, 3])

    TCP.exhaustive_pf(net_name='MV Oberrhein0', dp=0.01, dq=0.02, fsp_load_indices=[5], fsp_dg_indices=[5])


The examples vary in resolution and number of FSPs.
The figures bellow illustrate the resulting FA for each line respectively.

.. image:: //../plots/ExhaustivePowerFlow2024-11-05_17-00-00_incl_infeasible.png
  :width: 380
.. image:: //../plots/ExhaustivePowerFlow2024-11-08_14-10-05_incl_infeasible.jpg
  :width: 400

IV.C) Optimal Power Flow
---------------------------------------
This section illustrates examples using the OPF estimation functionality. These examples used the Python script code:

.. code-block:: console

    TCP.opf(net_name='CIGRE MV', opf_step=0.1, fsp_load_indices=[3, 5, 8], fsp_dg_indices=[8])

    TCP.opf(net_name='CIGRE MV', opf_step=0.1, fsp_load_indices=[1, 4, 9], fsp_dg_indices=[8])


The examples vary in FSPs.
The figures bellow illustrate the resulting FA for each line respectively.

.. image:: //../plots/OptimalPowerFlow2024-11-12_17-40-42.jpg
  :width: 400
.. image:: //../plots/OptimalPowerFlow2024-11-12_17-47-00.jpg
  :width: 400

IV.D) TensorConvolution+
---------------------------------------
This section illustrates examples using the TensorConvolution+ FA estimation functionality. The first examples, showcasing the different shapes of flexibility from FSPs use the lines:

.. code-block:: console

    TCP.tc_plus(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2, 3])

    TCP.tc_plus(net_name='MV Oberrhein0', fsp_load_indices=[1, 2], dp=0.05, dq=0.1, fsp_dg_indices=[1, 2], flex_shape='PQmax')


The examples vary in number of FSPs and shapes of flexibility offers. The example without the *flex_shape* input automatically obtains the value 'Smax'.
The figures bellow illustrate the resulting FA for each line respectively.

.. image:: //../plots/TensorConvolutionPlus2024-11-05_18-18-32.jpg
  :width: 400
.. image:: //../plots/TensorConvolutionPlus2024-11-08_14-38-25.jpg
  :width: 400


TensorConvolution+ can also simulate FAs with FSPs offering discrete setpoints of flexibility. For such scenarios, the input *non_linear_fsps* specifies which of the FSPs are non linear. The example line is:

.. code-block:: console

    TCP.tc_plus(net_name='CIGRE MV', fsp_load_indices=[3, 4, 5], dp=0.05, dq=0.1, fsp_dg_indices=[8], non_linear_fsps=[8])

The resulting figure is:

.. image:: //../plots/TensorConvolutionPlus2024-11-08_17-08-10.jpg
  :width: 400

IV.E) TensorConvolution+ Merge
---------------------------------------
This section showcases the function merging FSPs using the TensorConvolution+ algorithm.
For this functionality, the *max_fsps* input determines the maximum FSPs for which a network component can be sensitive before merging their flexibility.
The example line is:

.. code-block:: console

    TCP.tc_plus_merge(net_name='MV Oberrhein0', fsp_load_indices=[1, 2, 3], dp=0.025, dq=0.05, fsp_dg_indices=[1, 2, 3], max_fsps=5)

The resulting figure is:

.. image:: //../plots/TensorConvolutionPlusMegeFSPs2024-11-08_17-13-06.jpg
  :width: 400

IV.F) TensorConvolution+ Adapt
---------------------------------------
This section showcases the function storing information using the TensorConvolution+ algorithm, and then uses the stored information to adapt flexibility area for altered operating conditions.

.. code-block:: console

    # Define the consistent FSPs for the storing and adapting functions
    fsp_load_indices = [1, 2, 3]
    fsp_dg_indices = [1, 2, 3]

    # Estimate the FA and store the relevant information for adaptation
    TCP.tc_plus_save_tensors(net_name='MV Oberrhein0', fsp_load_indices=fsp_load_indices, dp=0.05, dq=0.1, fsp_dg_indices=fsp_dg_indices)

    # Modify the network operating conditions
    net, net_tmp = pn.mv_oberrhein(separation_by_sub=True)
    net.load['sn_mva'] = list(net.load['p_mw'].pow(2).add(net.load['q_mvar'].pow(2)).pow(0.5))
    net.load['scaling'] = [1 for i in range(len(net.load))]
    net.sgen['scaling'] = [1 for i in range(len(net.sgen))]
    net.switch['closed'] = [True for i in range(len(net.switch))]

    net = fix_net(net) # This function is included in the appendix

    rng = np.random.RandomState(212)

    net, rng = rand_resample(net, fsp_load_indices, fsp_dg_indices, rng, 0.05, 0.01, 0.05, 0.01) # This function is also included in the appendix

    # Adapt the FA using the locally stored information
    TCP.tc_plus_adapt(net=net, fsp_load_indices=fsp_load_indices, fsp_dg_indices=fsp_dg_indices)

    # Estimate the FA without adapting to compare with the above adapted result
    TCP.tc_plus(net=net, fsp_load_indices=fsp_load_indices, fsp_dg_indices=fsp_dg_indices, dp=0.05, dq=0.1)


The resulting figures for the stored, adapted and validated flexibility areas are:


.. image:: //../plots/TensorConvolutionPlusStore2024-11-08_17-44-24.jpg
  :width: 400
.. image:: //../plots/TensorConvolutionPlusAdapt2024-11-08_17-59-04.jpg
  :width: 400
.. image:: //../plots/TensorConvolutionPlus2024-11-08_17-59-28.jpg
  :width: 400

-------------------------------------------------------------------
V. Files for IEEE Transactions on Smart Grid Publication Scenarios
-------------------------------------------------------------------
To run use case scenarios, you can use the json files under the ``scenarios`` folder. The scripts used are under the **src/SmartGridScripts/** folder

V.A) Accuracy in Population Estimation
---------------------------------------

The result files are in the **csv_results/UC1** folder.
Example figures generated for these scenarios are in the folder **plots/UC1**

.. image:: //../plots/UC1/Compare_Flexibility_area_BruteOb0_2.svg
  :width: 400
.. image:: //../plots/UC1/Conv_multi_Conv_Conv_Brute_Ob0_2.svg
  :width: 400

V.B) Speed and Range Accuracy
---------------------------------

The CSV results are under **csv_results/UC2**
The figures are saved under **plots/UC2**

.. image:: //../plots/UC2/Loop/Kumaraswamy_MC_Oberrhein0_4FSPs_20_8.svg
  :width: 400
.. image:: //../plots/UC2/Loop/Uniform_MC_Oberrhein0_4FSPs_20_8.svg
  :width: 400
.. image:: //../plots/UC2/Loop/Hard_MC_Oberrhein0_4FSPs_20_8.svg
  :width: 400
.. image:: //../plots/UC2/Oberrhein0_speed_log.svg
  :width: 400
.. image:: //../plots/UC2/Oberrhein0W_speed_log.svg
  :width: 400
.. image:: //../plots/UC2/Oberrhein1_speed_log.svg
  :width: 400
.. image:: //../plots/UC2/Conv_multi_Conv_LargeRadial.svg
  :width: 400

V.C) Disjoint Flexibility Areas
---------------------------------

The result files are in the **csv_results/UC3** folder.
Example figures generated for these scenarios are in the folder **plots/UC3**

.. image:: //../plots/UC3/Disc_Scenario_121416012388_incl_infeasible.svg
  :width: 400
.. image:: //../plots/UC3/Conv_multi_Conv_Conv_Discontinuous.svg
  :width: 400

V.D) Uncertainty Estimation for Small FSPs
-------------------------------------------
The results are in the **csv_results/UC4** folder.
Example figure generated for the scenario is in the folder **plots/UC4**

.. image:: //../plots/UC4/Uncertainty_Interpreted.png
  :width: 400


V.E) Adaptability
---------------------------------

Example results from the case study can be found in the **csv_results/UC5** folder.
Example plots are in the **plots/UC5** folder:

.. image:: //../plots/UC5/TCP_Conv_SaveTsOb0.svg
  :width: 400
.. image:: //../plots/UC5/TCP_Conv_NoLoadFlexOb0.svg
  :width: 400
.. image:: //../plots/UC5/TCP_Conv_LoadFlexOb0.svg
  :width: 400
.. image:: //../plots/UC5/TCP_Conv_SaveFaTs.svg
  :width: 400
.. image:: //../plots/UC5/TCP_Conv_NoLoadTs_ax.svg
  :width: 400
.. image:: //../plots/UC5/TCP_Conv_LoadTs_ax.svg
  :width: 400


V.F) Case study for DFC
-------------------------------------------
The results are in the **csv_results/UC6** folder.
Example figures generated for the scenario are in the folder **plots/UC6**

.. image:: //../plots/UC6/feas_mat.svg
  :width: 250
.. image:: //../plots/UC6/heat_mat.svg
  :width: 250
.. image:: //../plots/UC6/min_cmat.svg
  :width: 250
.. image:: //../plots/UC6/nflex_mat.svg
  :width: 250


V.G) Case study for OPFs
-------------------------------------------

Example figure generated for this scenario is in the folder **plots/UC7**

.. image:: //../plots/UC7/OPF_.svg
  :width: 500


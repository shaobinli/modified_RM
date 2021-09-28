# modified_RM

This document provides the descrption on constructing modifed response matrix (RM).
Steps for implementing the traditional RM method are provided as following:
	Step 1, generate a set of response matrices, whose entries indicate the landscape yields for each subwatershed in the case of 100% adoption for a single conservation practice, gathered from SWAT simulation outputs. 
	Step 2, create a decision vector representing the fraction of subwatershed area allocated to each potential conservation practice; the decision vector represents users’ decision on adopting different BMPs at a subwatershed level.
	Step 3, calculate land area for each subwatershed; land area can be agricultural area only or total land are for the subwatershed, dependent on modelers’ decision on maximum land allowed for agricultural production in their case studies. 
	Step 4, construct a connectivity matrix describing the upstream-downstream relationships of all subwatersheds, with off-diagonal elements w_(i,j|i≠j ) equal to one if subwatershed j is upstream of subwatershed i and zero otherwise (w_(i,i )=1 ∀i). 
	Step 5, apply Equation 1 to estimate landscape yield during month t across all subwatersheds. 
	Step 6, Estimate in-stream loads at the outlet of each subwatershed by summing its own yield and all upstream yields with following modifications:
		Step 6.1 Account for reservoir's trapping effects
		Step 6.2 Add point sources of nitrate and phosphorus
		Step 6.3 Incorporate effects of in-stream phosphorus load
		Step 6.4 Use streamflow to estimate in-stream sediment load

Strucutre of codes and data is provided below:
modifed_RM - root directory
	|
	|-> modified_RM_main.py - script used for constructing the modified RM method with modifications 
	|-> swat_vs_RM.py - script used for comparing performance of modified RM, traditional RM, and SWAT original results
	|-> data.py - script used for calling data
	|-> point_source.py - script used for formulating point source into appropriate format
	|-> metrics.py - script used for calculating perfomrmance metrics: p-bias and NSE
	|-> streamflow_sediment_coefs.py - script used for estimating coefficients of linear and polynomial regressions for in-stream sediment.
	|-> instream_P_coefs.py - script used for estimating coeffiients of mult-linear regressioin between in-stream P and (landscape P and inverse streamflow)
	|->	support_data - folder contains support data required for constructing RM, including connectivity matrix, land use area, point source data.
	|-> response_matrix_csv - folder contains all response matrices data generated in Step 1.
	|-> 100Randomizations - folder contains SWAT simulation results of 100 randomized BMP combination allocations; used for validation.
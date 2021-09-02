# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)
"""


def pbias(obs, sim):
    '''
    obs and sim should be array
    The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. 
    Positive values indicate overestimation bias, whereas negative values indicate model underestimation bias
    '''
    obs_flat = obs.flatten()
    sim_flat = sim.flatten()
    bias = 100*sum(sim_flat-obs_flat)/sum(obs_flat)
    return bias
    
def nse(obs, sim):
    '''
    obs and sim should be array
    An efficiency of 1 (NSE = 1) corresponds to a perfect match of modeled discharge to the observed data.
    An efficiency of 0 (NSE = 0) indicates that the model predictions are as accurate as the mean of the observed data, 
    whereas an efficiency less than zero (NSE < 0) occurs when the observed mean is a better predictor than the model
    '''
    obs_flat = obs.flatten()
    obs_ave = obs.mean()
    sim_flat = sim.flatten()
    nse0 = 1 - sum((obs_flat - sim_flat)**2)/sum((obs_flat-obs_ave)**2) 
    return nse0

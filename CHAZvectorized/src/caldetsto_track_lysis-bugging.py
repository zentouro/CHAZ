#!/usr/bin/env python
###
# adding lysis
# adding hard-threshold for non-WPC storms to have higher initial storm intensity. 
####
import numpy as np
import pickle
import dask.array as da
import sys
import gc
import copy
import pandas as pd
import tools.module_stochastic as module_sto
from tools.readbst import read_ibtracs
import time
from tools import util
import random
from datetime import datetime
from netCDF4 import Dataset,date2num
from module_riskModel import fst2bt
from tools.util import int2str
from dask.diagnostics import ProgressBar
import Namelist as gv
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tools.readbst import read_ibtracs
import numpy.ma as ma
import netCDF4 as nc
from netCDF4 import Dataset
import tools.regression4 as reg4 

jetcmap = plt.cm.get_cmap("jet",31)
jet_vals = jetcmap(np.arange(31))
cmap1 = mcolors.LinearSegmentedColormap.from_list('newjet', jet_vals)
colors = jet_vals
colorcat = np.arange(0,155,5) 
xbin,ybin,basinMap = util.getbasinMap()

seed = 42
rng = np.random.default_rng(seed=seed)

def get_determin_vectorized():
    '''
    This function finds the deterministic component of the autoregressive TC intensity model.
    '''
    nStorms = bt.StormLon.shape[1]
    nTimes = bt.StormLon.shape[0]
    
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    lT = 2
    ih = 12
    lT_pred = int(ih/6)
    
    ## initialize arrays for all storms
    v0_array = np.full(nStorms, np.nan)
    dvdt_array = np.zeros(nStorms)
    active_storms = np.zeros(nStorms, dtype=bool)
    it1_array = np.zeros(nStorms, dtype=int)
    it2_array = np.zeros(nStorms, dtype=int)
    finished_storms = np.zeros(nStorms, dtype=bool)
    
    ## idk if this speeds things up but it is easier for me to read
    StormLon_all = bt.StormLon
    StormLat_all = bt.StormLat
    landmaskMean_all = bt.landmaskMean
    
    ## find valid storms (non-NaN and abs(lat) >= 5)
    dummy_mask = ~np.isnan(StormLon_all)  # (nTimes, nStorms)
    has_data = np.any(dummy_mask, axis=0)  # (nStorms,)
    abs_lat_check = np.abs(StormLat_all[0, :]) >= 5.0  # (nStorms,)
    active_storms = has_data & abs_lat_check
    
    # find last valid index for each storm
    last_valid_idx = np.full(nStorms, -1, dtype=int)
    for iS in range(nStorms):
        if active_storms[iS]:
            ## LOCAL RANDOM
            init_seed = seed + iS * 1000000 + 1
            init_rng = np.random.default_rng(seed = init_seed)

            valid_indices = np.where(dummy_mask[:, iS])[0]
            if len(valid_indices) > 0:
                last_valid_idx[iS] = min(valid_indices[-1], nTimes - 5)
    
    it1_array[active_storms] = 0
    it2_array[active_storms] = last_valid_idx[active_storms]
    

    region_mask = (StormLat_all[0, :] >= 0) & (StormLon_all[0, :] >= 120) & (StormLon_all[0, :] <= 180)
    
    ## TODO: FREEZING RANDOM
    # bt.determin[0, active_storms & region_mask] = np.maximum(20, intV[intV == intV][0])
    # bt.determin[0, active_storms & ~region_mask] = np.maximum(25, intV[intV == intV][1])

    ## with LOCAL RANDOM
    bt.determin[0, active_storms & region_mask] = np.maximum(20, init_rng.choice(intV[intV == intV]))
    bt.determin[0, active_storms & ~region_mask] = np.maximum(25, init_rng.choice(intV[intV == intV]))

    ## from random seed only
    #bt.determin[0, active_storms & region_mask] = np.maximum(20, rng.choice(intV[intV == intV]))
    #bt.determin[0, active_storms & ~region_mask] = np.maximum(25, rng.choice(intV[intV == intV]))

    
    v0_array[active_storms] = bt.determin[0, active_storms]
    dvdt_array[active_storms] = 0.0
    

    max_it2 = np.max(it2_array[active_storms]) if np.any(active_storms) else 0
    
    for it in range(0, int(max_it2) + 2, 2):
        ## find which storms are active at this timestep
        ## create mask for active storms at this timestep
        in_time_range = (it >= it1_array) & (it <= it2_array)
        not_finished = ~finished_storms
        has_future = (it + lT_pred) < nTimes
        
        ## check for NaN values
        lon_valid = ~np.isnan(StormLon_all[it, :])
        landmask_current_valid = ~np.isnan(landmaskMean_all[it, :])
        
        landmask_future_valid = np.zeros(nStorms, dtype=bool)
        if it + lT_pred < nTimes:
            landmask_future_valid = ~np.isnan(landmaskMean_all[it + lT_pred, :])
        
        v0_valid = ~np.isnan(v0_array)
        
        ## combine
        storms_mask = (active_storms & in_time_range & not_finished & has_future & 
                      lon_valid & landmask_current_valid & landmask_future_valid & v0_valid)
        
        storms_to_process = np.where(storms_mask)[0]
        
        if len(storms_to_process) == 0:
            continue
        
        ## group storms by landmask
        if it + lT_pred < nTimes:
            landmask_it = landmaskMean_all[it, storms_to_process]
            landmask_itlT = landmaskMean_all[it + lT_pred, storms_to_process]
            
            water_mask = (landmask_itlT <= -0.5) & (landmask_it <= -0.5)
            water_storms = storms_to_process[water_mask]
            land_storms = storms_to_process[~water_mask]
        else:
            water_storms = np.array([])
            land_storms = np.array([])
        
        ## this could probably be a seperate function (same in stochastic)
        ## to avoid duplication
        ## water
        if len(water_storms) > 0:
            v0_water = v0_array[water_storms]
            dvdt_water = dvdt_array[water_storms]
            
            predictors = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean',
                         'dPIwspd2','dPIwspd3','dVdt2']
            
            v1_water = reg4.getPrediction_v0input_result_vectorized(
                bt, meanX_w, meanY_w, stdX_w, stdY_w, it,
                water_storms, [ih], result_w, predictors,
                TimeDepends, v0_water, dvdt_water)
            
            bt.determin[it + lT_pred, water_storms] = v1_water
            
            ## mark storms with v1 < 10 as finished
            too_weak = v1_water < 10
            finished_storms[water_storms[too_weak]] = True
            bt.determin[it + lT_pred, water_storms[too_weak]] = np.nan
            
            ## update v0 and dvdt for continuing storms
            continuing = ~too_weak
            dvdt_array[water_storms[continuing]] = v1_water[continuing] - v0_array[water_storms[continuing]]
            v0_array[water_storms[continuing]] = v1_water[continuing]
            
            ## check conditions
            for idx, iS in enumerate(water_storms):
                if not too_weak[idx]:
                    if ((bt.determin[it1_array[iS]:it+lT_pred:lT_pred, iS].max() > 35) and 
                        (bt.determin[it, iS] <= 35) and 
                        (it >= lT_pred) and (bt.determin[it - lT_pred, iS] <= 35)):
                        finished_storms[iS] = True
        
        ## land
        if len(land_storms) > 0:
            v0_land = v0_array[land_storms]
            dvdt_land = dvdt_array[land_storms]
            
            predictors = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean',
                         'dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
            
            v1_land = reg4.getPrediction_v0input_result_vectorized(
                bt, meanX_l, meanY_l, stdX_l, stdY_l, it,
                land_storms, [ih], result_l, predictors,
                TimeDepends, v0_land, dvdt_land)
            
            bt.determin[it + lT_pred, land_storms] = v1_land
            
            ## mark storms with v1 < 10 as finished
            too_weak = v1_land < 10
            finished_storms[land_storms[too_weak]] = True
            bt.determin[it + lT_pred, land_storms[too_weak]] = np.nan
            
            ## update v0 and dvdt for continuing storms
            continuing = ~too_weak
            dvdt_array[land_storms[continuing]] = v1_land[continuing] - v0_array[land_storms[continuing]]
            v0_array[land_storms[continuing]] = v1_land[continuing]
            
            ## check conditions
            for idx, iS in enumerate(land_storms):
                if not too_weak[idx]:
                    if ((bt.determin[it1_array[iS]:it+lT_pred:lT_pred, iS].max() > 35) and 
                        (bt.determin[it, iS] <= 35) and 
                        (it >= lT_pred) and (bt.determin[it - lT_pred, iS] <= 35)):
                        finished_storms[iS] = True
    
    ## intermediate timesteps
    for iS in range(nStorms):
        if not active_storms[iS]:
            continue
        
        it1 = it1_array[iS]
        it2 = min(it2_array[iS] + 1, nTimes - 2)
        
        iit_indices = np.arange(it1 + 1, it2, 2)
        if len(iit_indices) == 0:
            continue
        
        prev_vals = bt.determin[iit_indices - 1, iS]
        next_vals = bt.determin[iit_indices + 1, iS]
        
        mask = ~np.isnan(prev_vals * next_vals)
        bt.determin[iit_indices[mask], iS] = 0.5 * (prev_vals[mask] + next_vals[mask])


def get_stochastic_vectorized(iNN):
    '''
    This function finds the stochastic component of the autoregressive TC intensity model.
    '''
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    
    nStorms = bt.StormLon.shape[1]
    nTimes = bt.StormLon.shape[0]
    
    ih = 12
    lT = int(ih/6)
    
    v0_array = np.full(nStorms, np.nan)
    dvdt_array = np.zeros(nStorms)
    active_storms = np.zeros(nStorms, dtype=bool)
    it1_array = np.zeros(nStorms, dtype=int)
    it2_array = np.zeros(nStorms, dtype=int)
    finished_storms = np.zeros(nStorms, dtype=bool)
    
    StormLon_all = bt.StormLon
    StormLat_all = bt.StormLat
    landmaskMean_all = bt.landmaskMean
    
    dummy_mask = ~np.isnan(StormLon_all)
    has_data = np.any(dummy_mask, axis=0)
    abs_lat_check = np.abs(StormLat_all[0, :]) >= 5.0
    active_storms = has_data & abs_lat_check
    
    ## find last valid index for each storm
    last_valid_idx = np.full(nStorms, -1, dtype=int)

    init_values = np.full(nStorms, np.nan)
    for iS in range(nStorms):
        if active_storms[iS]:
            init_seed = seed + iNN * 1000000 + iS * 10000 + 1 # +1 for init phase
            init_rng = np.random.default_rng(seed=init_seed)

            ## Store the random choice for this storm
            # if region_mask[iS]:
            #     init_values_region[iS] = init_rng.choice(intV[intV == intV])
            # else:
            #     init_values_other[iS] = init_rng.choice(intV[intV == intV])
            init_values[iS] = init_rng.choice(intV[intV == intV])

            valid_indices = np.where(dummy_mask[:, iS])[0]
            if len(valid_indices) > 0:
                last_valid_idx[iS] = min(valid_indices[-1], nTimes - 5)
    
    it1_array[active_storms] = 0
    it2_array[active_storms] = last_valid_idx[active_storms]
    
    ## initialize v0 based on location
    region_mask = (StormLat_all[0, :] >= 0) & (StormLon_all[0, :] >= 120) & (StormLon_all[0, :] <= 180)
    
    ## TODO: FREEZING RANDOM
    #bt.stochastic[0, active_storms & region_mask, iNN] = np.maximum(20, intV[intV == intV][0])
    #bt.stochastic[0, active_storms & ~region_mask, iNN] = np.maximum(25, intV[intV == intV][1])

    ## with random, but not comparable 
    #bt.stochastic[0, active_storms & region_mask, iNN] = np.maximum(20, rng.choice(intV[intV == intV]))
    #bt.stochastic[0, active_storms & ~region_mask, iNN] = np.maximum(25, rng.choice(intV[intV == intV]))

    ## LOCAL RANDOM
    #bt.stochastic[0, active_storms & region_mask, iNN] = np.maximum(20, init_rng.choice(intV[intV == intV]))
    #bt.stochastic[0, active_storms & ~region_mask, iNN] = np.maximum(25, init_rng.choice(intV[intV == intV]))

    #bt.stochastic[0, active_storms & region_mask, iNN] = np.maximum(20, init_values_region[active_storms & region_mask,])
    #bt.stochastic[0, active_storms & ~region_mask, iNN] = np.maximum(25, init_values_other[active_storms & ~region_mask])

    bt.stochastic[0, active_storms & region_mask, iNN] = np.maximum(20, init_values[active_storms & region_mask])
    bt.stochastic[0, active_storms & ~region_mask, iNN] = np.maximum(25, init_values[active_storms & ~region_mask])
    
    v0_array[active_storms] = bt.stochastic[0, active_storms, iNN]
    dvdt_array[active_storms] = 0.0
    
    ## process all storms timestep by timestep
    max_it2 = np.max(it2_array[active_storms]) if np.any(active_storms) else 0
    
    for it in range(0, int(max_it2) + 2, 2):
        ## find which storms are active at this timestep
        in_time_range = (it >= it1_array) & (it <= it2_array)
        not_finished = ~finished_storms
        has_future = (it + lT) < nTimes
        
        lon_valid = ~np.isnan(StormLon_all[it, :])
        landmask_current_valid = ~np.isnan(landmaskMean_all[it, :])
        
        landmask_future_valid = np.zeros(nStorms, dtype=bool)
        if it + lT < nTimes:
            landmask_future_valid = ~np.isnan(landmaskMean_all[it + lT, :])
        
        v0_valid = ~np.isnan(v0_array)
        
        storms_mask = (active_storms & in_time_range & not_finished & has_future & 
                      lon_valid & landmask_current_valid & landmask_future_valid & v0_valid)
        
        storms_to_process = np.where(storms_mask)[0]
        
        if len(storms_to_process) == 0:
            continue
        
        ## group storms by landmask 
        if it + lT < nTimes:
            landmask_it = landmaskMean_all[it, storms_to_process]
            landmask_itlT = landmaskMean_all[it + lT, storms_to_process]
            
            water_mask = (landmask_itlT <= -0.5) & (landmask_it <= -0.5)
            water_storms = storms_to_process[water_mask]
            land_storms = storms_to_process[~water_mask]
        else:
            water_storms = np.array([])
            land_storms = np.array([])
        
        ## water
        if len(water_storms) > 0:
            v0_water = v0_array[water_storms]
            dvdt_water = dvdt_array[water_storms]
            
            predictors = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean',
                         'dPIwspd2','dPIwspd3','dVdt2']
            
            v1_water = reg4.getPrediction_v0input_result_vectorized(
                bt, meanX_w, meanY_w, stdX_w, stdY_w, it,
                water_storms, [ih], result_w, predictors,
                TimeDepends, v0_water, dvdt_water)
            
            ## calculate errors
            #error_water = findError_vectorized(E0, v0E, v0_water, cat1)
            ## LOCAL ERROR 
            error_water = module_sto.findError_vectorized(E0, v0E, v0_water, cat1, water_storms, it, iNN)

            bt.error[it, water_storms, iNN] = error_water
            
            v1_water = v1_water - error_water
            
            bt.stochastic[it + lT, water_storms, iNN] = v1_water
            
            ## mark storms with v1 < 10 as finished
            too_weak = v1_water < 10
            finished_storms[water_storms[too_weak]] = True
            
            ## update v0 and dvdt for continuing storms
            continuing = ~too_weak
            dvdt_array[water_storms[continuing]] = v1_water[continuing] - v0_array[water_storms[continuing]]
            v0_array[water_storms[continuing]] = v1_water[continuing]
            
            ## check conditions
            for idx, iS in enumerate(water_storms):
                if not too_weak[idx]:
                    if ((bt.stochastic[it1_array[iS]:it+lT:lT, iS, iNN].max() > 35) and 
                        (bt.stochastic[it, iS, iNN] <= 35) and 
                        (it >= lT) and (bt.stochastic[it - lT, iS, iNN] <= 35)):
                        finished_storms[iS] = True
        
        ## land
        if len(land_storms) > 0:
            v0_land = v0_array[land_storms]
            dvdt_land = dvdt_array[land_storms]
            
            predictors = ['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean',
                         'dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
            
            v1_land = reg4.getPrediction_v0input_result_vectorized(
                bt, meanX_l, meanY_l, stdX_l, stdY_l, it,
                land_storms, [ih], result_l, predictors,
                TimeDepends, v0_land, dvdt_land)
            
            ## calculate errors
            #error_land = findError_vectorized(E0, v0E, v0_land, cat1)

            ## LOCAL ERROR
            error_land = module_sto.findError_vectorized(E0, v0E, v0_land, cat1, land_storms, it, iNN)

            bt.error[it, land_storms, iNN] = error_land
            
            v1_land = v1_land - error_land
            
            bt.stochastic[it + lT, land_storms, iNN] = v1_land
            
            ## mark storms with v1 < 10 as finished
            too_weak = v1_land < 10
            finished_storms[land_storms[too_weak]] = True
            
            ## update v0 and dvdt for continuing storms
            continuing = ~too_weak
            dvdt_array[land_storms[continuing]] = v1_land[continuing] - v0_array[land_storms[continuing]]
            v0_array[land_storms[continuing]] = v1_land[continuing]
            
            ## check conditions
            for idx, iS in enumerate(land_storms):
                if not too_weak[idx]:
                    if ((bt.stochastic[it1_array[iS]:it+lT:lT, iS, iNN].max() > 35) and 
                        (bt.stochastic[it, iS, iNN] <= 35) and 
                        (it >= lT) and (bt.stochastic[it - lT, iS, iNN] <= 35)):
                        finished_storms[iS] = True
    
    ## intermediate timesteps
    for iS in range(nStorms):
        if not active_storms[iS]:
            continue
        
        it1 = it1_array[iS]
        it2 = min(it2_array[iS] + 1, nTimes - 2)
        
        iit_indices = np.arange(it1 + 1, it2, 2)
        if len(iit_indices) == 0:
            continue
        
        prev_vals = bt.stochastic[iit_indices - 1, iS, iNN]
        next_vals = bt.stochastic[iit_indices + 1, iS, iNN]
        
        mask = ~np.isnan(prev_vals * next_vals)
        bt.stochastic[iit_indices[mask], iS, iNN] = 0.5 * (prev_vals[mask] + next_vals[mask])


def get_determin(iiS,block_id=None):
    '''
    This function finds the deterministic component of the autoregressive TC intensity model.
    iiS: bt data for intensity model
    '''
    iS = np.int_(iiS.mean(keepdims=True))
    #iS  = iiS
    lT = 2
    dummy = bt.StormLon[:,iS][bt.StormLon[:,iS]==bt.StormLon[:,iS]]
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    if ((dummy.any()) and (np.abs(bt.StormLat[0,iS])>=5.)):
       it1 = 0
       it2 = np.min([np.argwhere(bt.StormLon[:,iS]==dummy[-1])[-1,0],bt.StormLon.shape[0]-5])
       if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
          bt.determin[0,iS] = np.max([20,random.choice(intV[intV==intV])]) # kts 
       else:
          bt.determin[0,iS] = np.max([25,random.choice(intV[intV==intV])]) # kts 
       #bt.determin[0,iS] = intV[iS] # kts 
       dvdt = 0.0 
       ih = 12
       lT = np.int_(ih/6) ### track model formate is every 12 hours
       v0 = bt.determin[0,iS]
       for it in range(it1,it2+2,2):
          if ((it+lT<bt.StormLon.shape[0]) and (bt.StormLon[it,iS]==bt.StormLon[it,iS])\
                and (bt.landmaskMean[it,iS]==bt.landmaskMean[it,iS]) \
                and (bt.landmaskMean[it+lT,iS]==bt.landmaskMean[it+lT,iS]) and not np.isnan(v0)):
             if(((bt.landmaskMean[it+lT,iS]<= -0.5)&(bt.landmaskMean[it,iS]<= -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
                 result,meanX,meanY,stdX,stdY = np.copy(result_w),np.copy(meanX_w),\
                                                np.copy(meanY_w),np.copy(stdX_w),np.copy(stdY_w)
             elif(((bt.landmaskMean[it+lT,iS]> -0.5)|(bt.landmaskMean[it,iS]> -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
                 result,meanX,meanY,stdX,stdY = np.copy(result_l),np.copy(meanX_l),\
                                                np.copy(meanY_l),np.copy(stdX_l),np.copy(stdY_l)
             h1,v1 = reg4.getPrediction_v0input_result\
                     (bt,meanX,meanY,stdX,stdY,it,\
                     iS,[ih],result,predictors,\
                     TimeDepends,v0,dvdt)
             v1 = v1[1]
             if v1 < 10:
                v1 = np.float64('Nan') 
                break;
             bt.determin[it+lT,iS] = v1
             if ((bt.determin[it1:it+lT:lT,iS].max()>35) and (bt.determin[it,iS]<=35) and (bt.determin[it-lT,iS]<=35)):
                break;
             dvdt = v1-v0
             v0 = v1
       for iit in range(it1+1,np.min([it2+1,bt.StormLon.shape[0]-2]),2):
           a = bt.determin[iit-1,iS]*bt.determin[iit+1,iS]
           if a==a:
              bt.determin[iit,iS] = \
              0.5*(bt.determin[iit-1,iS]+bt.determin[iit+1,iS])
    return iS

def get_stochastic(iiS,block_id=None):
    '''
    This function finds the stochastic component of the autoregressive TC intensity model.
    iiS: bt data for intensity model
    '''
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    iS = np.int_(iiS.mean(keepdims=True))
    dummy = bt.StormLon[:,iS][bt.StormLon[:,iS]==bt.StormLon[:,iS]]
    if ((dummy.any()) and (np.abs(bt.StormLat[0,iS])>=5.)):
       it1 = 0
       it2 = np.min([np.argwhere(bt.StormLon[:,iS]==dummy[-1])[-1,0],bt.StormLon.shape[0]-5])
       if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
          bt.stochastic[0,iS,iNN] = np.max([20,random.choice(intV[intV==intV])]) # kts 
       else:
          bt.stochastic[0,iS,iNN] = np.max([25,random.choice(intV[intV==intV])]) # kts 
       ih = 12
       lT = np.int_(ih/6)
       v0 = bt.stochastic[0,iS,iNN]
       dvdt = 0.
       for it in range(it1,it2+2,2):
          if ((it+lT<bt.StormLon.shape[0]) and (bt.StormLon[it,iS]==bt.StormLon[it,iS])\
                and (bt.landmaskMean[it,iS]==bt.landmaskMean[it,iS]) \
                and (bt.landmaskMean[it+lT,iS]==bt.landmaskMean[it+lT,iS]) and not np.isnan(v0)):
             if(((bt.landmaskMean[it+lT,iS]<= -0.5)&(bt.landmaskMean[it,iS]<= -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2']
                 result,meanX,meanY,stdX,stdY = result_w,meanX_w,\
                                                meanY_w,stdX_w,stdY_w
             elif(((bt.landmaskMean[it+lT,iS]> -0.5)|(bt.landmaskMean[it,iS]> -0.5))):
                 predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
                 result,meanX,meanY,stdX,stdY = result_l,meanX_l,\
                                                meanY_l,stdX_l,stdY_l
             h1,v1 = reg4.getPrediction_v0input_result\
                     (bt,meanX,meanY,stdX,stdY,it,\
                     iS,[ih],result,predictors,\
                     TimeDepends,v0,dvdt)
             error = module_sto.findError(E0,v0E,v0,cat1)
             bt.error[it,iS,iNN] = error 
             v1 = v1[1]
             v1 = v1-error
             if v1 < 10:
                break;
             #if ((it >= it1+4*lT) and (bt.stochastic[it,iS,iNN]<=35) and (bt.stochastic[it-lT,iS,iNN]<=35)):
             bt.stochastic[it+lT,iS,iNN] = v1
             if ((bt.stochastic[it1:it+lT:lT,iS,iNN].max()>35) and (bt.stochastic[it,iS,iNN]<=35) and (bt.stochastic[it-lT,iS,iNN]<=35)):
                break;
             dvdt = v1-v0
             v0 = v1
       for iit in range(it1+1,np.min([it2+1,bt.StormLon.shape[0]-2]),2):
           a = bt.stochastic[iit-1,iS,iNN]*bt.stochastic[iit+1,iS,iNN]
           if a==a:
              bt.stochastic[iit,iS,iNN] = \
              0.5*(bt.stochastic[iit-1,iS,iNN]+bt.stochastic[iit+1,iS,iNN])


    return iS

def basic_checking_plot(ds):
    import matplotlib.pyplot as plt
    import cartopy as cart
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    MyFigsize,MyFontsize = (6.5,4),8
    londis, latdis = 30, 15
    lon1, lon2 = 10,350
    lat1, lat2 = -60, 60
    my_proj = ccrs.PlateCarree(central_longitude=180)
    data_ccrs = ccrs.PlateCarree()

    fig = plt.figure(figsize=MyFigsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(zorder=100)

    lon = ds['longitude'].values
    lat = ds['latitude'].values
    mwspd = ds['Mwspd'].values
    plt.plot(lon,lat,'k-',transform=ccrs.PlateCarree(),linewidth=0.2)
    plt.scatter(lon.ravel(),lat.ravel(),s=10,c=mwspd[1,:,:].ravel(),transform=ccrs.PlateCarree(), cmap='jet')
    #plt.colorbar()


## tidied this up so it isn't a giant list of basinMap[a==0000]
## TODO double check that all the basinMap code works and matches original 
def getbasinMap(ifplot=False):
    '''
    This function returns x bins, y bins, and a plot of the basin map.
    '''
    xbin = np.arange(0, 365, 5)
    ybin = np.arange(-90, 95, 5)
    xcenter = 0.5 * (xbin[:-1] + xbin[1:])
    ycenter = 0.5 * (ybin[:-1] + ybin[1:])
    
    basinMap = np.zeros((xcenter.size, ycenter.size))
    lonc, latc = np.meshgrid(xcenter, ycenter, indexing='ij')
    
    # define basins using a dictionary 
    basins = {
        1: (lonc >= 235) & (latc >= 0) & (latc <= 45),                 # n_atl
        2: (lonc >= 180) & (lonc < 235) & (latc >= 0) & (latc <= 45),  # n_enp
        3: (lonc >= 100) & (lonc < 180) & (latc >= 0) & (latc <= 45),  # n_wnp
        4: (lonc < 100) & (latc >= 0) & (latc <= 45),                  # n_ni
        5: (lonc < 90) & (latc < 0) & (latc >= -45),                   # n_sin
        6: (lonc >= 90) & (lonc < 160) & (latc < 0) & (latc >= -45),   # n_aus
        7: (lonc >= 160) & (lonc < 240) & (latc < 0) & (latc >= -45)   # n_spc
    }
    
    for basin_id, mask in basins.items():
        basinMap[mask] = basin_id

    # fix specific indices (flattened)
    ## pretty sure i got all of them
    fix_indices = [
        1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719,  
        1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 
        1782, 1783, 1784, 1785, 1786, 1787, 1818, 1819, 1820, 1821, 
        1822, 1823, 1854, 1855, 1856, 1857, 1890, 1891, 1892, 1893, 
        1926, 1927, 1928, 1962, 1963, 1964, 1998, 1999
    ]
    basinMap.ravel()[np.isin(np.arange(basinMap.size), fix_indices)] = 2
    
    return xbin, ybin, basinMap

## TODO: Move to UTILS
def defineBasin(lon0_obs,lat0_obs,basinlon,basinlat,basinMap):
    '''
    Defines basin based on observations and knowledge about the basin.
    lon0_obs: initial longitude observed
    lat0_obs: initial latitude observed
    basinlon: basin longitude
    basinlat: basin latitude 
    basinMap: map of basin
    '''
    ### avoid NaN
    basin = np.zeros(lon0_obs.shape)
    ###
    lat0_obs[lat0_obs>=90] = 89.9
    lat0_obs[lat0_obs<=-90] = -89.9
    notNaN_arg = np.where(~np.isnan(lon0_obs*lat0_obs))
    x = np.floor(lon0_obs[notNaN_arg]/np.diff(basinlon)[0])
    y = np.floor((lat0_obs[notNaN_arg]-basinlat[0])/np.diff(basinlat)[0])
    x[x==basinMap.shape[0]] = basinMap.shape[0]-1
    y[y==basinMap.shape[1]] = basinMap.shape[1]-1
    basin[notNaN_arg] = basinMap[np.int_(x),np.int_(y)]
    return(basin)

## TODO: Move to UTILS
def clean_up_save(tc_downscaled, iy):
    xbin,ybin,basinMap = getbasinMap()

    bt1 = copy.copy(tc_downscaled)

    lon0,lat0 = bt1.StormLon[0,:],bt1.StormLat[0,:]
    basin = defineBasin(lon0,lat0,xbin,ybin,basinMap)

    #### get rid of storms  that has never developed
    bt1.stochastic[bt1.stochastic==0] = np.float64('nan')
    max5 = np.nanmax(bt1.stochastic[0:21,:,:],axis=0)
    iS1 = np.argwhere(max5<35)[:,0]
    iN1 = np.argwhere(max5<35)[:,1]
    bt1.stochastic[:,iS1,iN1] = np.float64('nan')
    maxall = np.nanmax(bt1.stochastic,axis=0)
    v0 = bt1.stochastic[0,:,:]
    #### get rid of storms that becoming unstable
    iS1 = np.argwhere(maxall>300)[:,0]
    iN1 = np.argwhere(maxall>300)[:,1]
    bt1.stochastic[:,iS1,iN1] = np.float64('nan')
    #### get rid of storms that initially formed with 2 degree lon,lat
    iS1 = np.argwhere(np.abs(bt1.StormLat[0,:])<2)[:,0]
    bt1.stochastic[:,iS1,:] = np.float64('nan')
    bt1.StormLon[0,iS1] = np.float64('nan')
    iS1 = np.argwhere(basin==0)
    bt1.StormLon[0,iS1] = np.float64('nan')

    #### get rid of storms that have no intensity record
    maxall = np.nanmax(bt1.stochastic,axis=0)
    #changed
    iS1 = np.argwhere(maxall<=0)[:,0]
    iN1 = np.argwhere(maxall<=0)[:,1]
    bt1.stochastic[:,iS1,iN1] = np.float64('nan')
    #bt1.stochastic[:,basin!=1,:] = np.float('nan')
    arg = np.argwhere(bt1.StormLon[0,:] == bt1.StormLon[0,:])[:][:,0]

    newlon = bt1.StormLon[:,arg]
    newlat = bt1.StormLat[:,arg]
    newwspd = bt1.stochastic[:,arg]
    newdatenum = bt1.Time[:,arg]
    newYear = bt1.StormYear[arg]
    newMonth = bt1.StormInitMonth[arg]

    #dummy dataframe
    ensembleNum = np.arange(newwspd.shape[2])
    stormID = np.arange(arg.shape[0])
    lifelength = np.arange(newdatenum.shape[0])

    ds = xr.Dataset({'longitude': xr.DataArray(data = newlon,
                                               dims = ['lifelength','stormID'],
                                               coords = {'lifelength':lifelength, 
                                                         'stormID':stormID},
                                               attrs = {'_FillValue': np.float64('nan'),
                                                        'units': 'degrees east'}
                                                ),
                    'latitude': xr.DataArray(data = newlat,
                                             dims = ['lifelength','stormID'],
                                             coords = {'lifelength':lifelength, 
                                                       'stormID':stormID},
                                             attrs = {'_FillValue': np.float64('nan'),
                                                      'units'     : 'degrees north'}
                                            ),
                    'Mwspd': xr.DataArray(data =np.rollaxis(newwspd,2,0),
                                          dims = ['ensembleNum','lifelength','stormID'],
                                          coords = {'ensembleNum':ensembleNum,'lifelength':lifelength, 'stormID':stormID},
                                          attrs = {'_FillValue': np.float64('nan'),
                                                   'units'     : 'kt'}
                                            ),
                    'year': xr.DataArray(data = newYear,
                                         dims = ['stormID'],
                                         coords = {'stormID':stormID},
                                         attrs = {'units'     : 'year'}
                                        ),
                    'time': xr.DataArray(data = date2num(newdatenum, 
                                                         units='days since 1950-01-01 00:00', 
                                                         calendar='standard'),
                                         dims = ['lifelength','stormID'],
                                         coords = {'lifelength':lifelength, 'stormID':stormID},
                                         attrs = {'units'     : 'days since 1950-01-01 00:00'}
                                         )
                    })

    ## TODO: fix/align with original
    iens = 0
    file_name = gv.output_path+gv.Model+'_'+int2str(iy,4)+'_ens'+int2str(iens,3)+'.nc'       
    print(file_name)
    ds.to_netcdf(file_name)


def calIntensity_vectorized(iy, ichaz):
    '''
    This function calculates the intesity using an autoregressive model (Lee et al. (2015, 2016a)).
    EXP: location of historical simulations defined in Namelist.py
    iy: year of current iteration in CHAZ.py
    ichaz: current ensemble iteration in CHAZ.py`
    
    '''
    
    ## load datasets 
    bt2 = xr.open_dataset(gv.opath)
    observed_data_ds = xr.open_dataset(gv.ipath + 'result_w.nc')
    coefficient_meanstd_ds = xr.open_dataset(gv.pre_path + 'coefficient_meanstd.nc')
    result_w_ds = xr.open_dataset(gv.ipath + 'result_w.nc')
    result_l_ds = xr.open_dataset(gv.ipath + 'result_l.nc')
    
    global result_w, result_l, meanX_w, meanY_w, stdX_w
    global stdY_w, meanX_l, meanY_l, stdX_l, stdY_l
    global E0, v0E, cat1, intV

    ## convert all variables into numpy arrays 
    result_w = result_w_ds['params'].values
    result_l = result_l_ds['params'].values
    meanX_w = coefficient_meanstd_ds.meanX_w.values
    meanY_w = coefficient_meanstd_ds.meanY_w.values
    stdX_w  = coefficient_meanstd_ds.stdX_w.values
    stdY_w  = coefficient_meanstd_ds.stdY_w.values
    meanX_l = coefficient_meanstd_ds.meanX_l.values
    meanY_l = coefficient_meanstd_ds.meanY_l.values
    stdX_l  = coefficient_meanstd_ds.stdX_l.values
    stdY_l  = coefficient_meanstd_ds.stdY_l.values

    NS = np.where((bt2['StormYear']>=1981)&(bt2['StormYear']<=2012))[0]
    intV = bt2['StormMwspd'][:][0,NS].values
    E0, v0E, cat1 = module_sto.get_E1_vectorized(bt2)

    #read the synthetic tracks
    global bt
    with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','rb')as f:  ### changed from 'r' to 'rb'
        bt = pickle.load(f)
        f.close()

    ## initialize bt
    nsto = gv.CHAZ_Int_ENS
    bt.__dict__['determin'] = np.zeros(bt.StormLon.shape)*np.float64('nan')
    bt.__dict__['stochastic'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float64('nan')
    bt.__dict__['error'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float64('nan')

    ## process deterministic 
    get_determin_vectorized()

    ## process stochastic 
    for iNN in range(nsto):
        get_stochastic_vectorized(iNN)

    gc.collect()

    ds = clean_up_save(bt, iy)
    #basic_checking_plot(ds)


def calIntensity(EXP,iy,ichaz):
    '''
    This function calculates the intesity using an autoregressive model (Lee et al. (2015, 2016a)).
    EXP: location of historical simulations defined in Namelist.py
    iy: year of current iteration in CHAZ.py
    ichaz: current ensemble iteration in CHAZ.py`
    '''
    ipath2 = gv.ipath+gv.Model+'/HIST/'
    global result_w, meanY_w, stdX_w, meanX_w, stdY_w
    global result_l, meanY_l, stdX_l, meanX_l, stdY_l

    ### read coefficient from reanalysis data trainning,
    ### as well as recording the mean and STD
    # with open (gv.opath,'rb') as f:
    # bt = pickle.load(f)
    # result_w = pickle.load(f) 
    # meanX_w_obs = pickle.load(f)
    # stdX_w_obs = pickle.load(f)
    # meanY_w_obs = pickle.load(f)
    # stdY_w_obs = pickle.load(f)
    # result_l = pickle.load(f) 
    # meanX_l_obs = pickle.load(f)
    # stdX_l_obs = pickle.load(f)
    # meanY_l_obs = pickle.load(f)
    # stdY_l_obs = pickle.load(f)
    # f.close()
    bt2 = nc.Dataset(gv.opath,'r')
    result_wi = nc.Dataset(gv.ipath + 'result_w.nc', 'r')
    result_w = ma.getdata(result_wi['params'][:])
    result_li = nc.Dataset(gv.ipath + 'result_l.nc','r')
    result_l = ma.getdata(result_li['params'][:])
    observed_data = nc.Dataset(gv.ipath + 'observed_data.nc', 'r')
    meanX_w_obs = ma.getdata(observed_data['meanX_w_obs'][:])
    stdX_w_obs = ma.getdata(observed_data['stdX_w_obs'][:])
    meanY_w_obs = ma.getdata(observed_data['meanY_w_obs'][:])
    stdY_w_obs = ma.getdata(observed_data['stdY_w_obs'][:])
    meanX_l_obs = ma.getdata(observed_data['meanX_l_obs'][:])
    stdX_l_obs = ma.getdata(observed_data['stdX_l_obs'][:])
    meanY_l_obs = ma.getdata(observed_data['meanY_l_obs'][:])
    stdY_l_obs = ma.getdata(observed_data['stdY_l_obs'][:])

    NS = np.argwhere((bt2['StormYear'][:]>=1981)&(bt2['StormYear'][:]<=2012))[:,0]
    global E0, v0E, cat1, intV
    intV = bt2['StormMwspd'][:][0,NS]
    E0, v0E,cat1, = module_sto.get_E1(bt2)
    del bt2

    #read the synthetic tracks
    global bt
    with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','rb')as f:  ### changed from 'r' to 'rb'
        bt = pickle.load(f)
        f.close()

        
    ### read mean and std from the HIST data
    #with open (gv.ipath+'/coefficient_meanstd.pik','r') as f:
    # dummy = pickle.load(f)
    # meanX_w = pickle.load(f)
    # stdX_w = pickle.load(f)
    # meanY_w = pickle.load(f)
    # stdY_w = pickle.load(f)
    # dummy = pickle.load(f)
    # meanX_l=pickle.load(f)
    # stdX_l = pickle.load(f)
    # meanY_l = pickle.load(f)
    # stdY_l = pickle.load(f)
    # f.close()
    coeff_meanstd = nc.Dataset(gv.pre_path+'coefficient_meanstd.nc','r')
    meanX_w = ma.getdata(coeff_meanstd['meanX_w'][:])
    stdX_w = ma.getdata(coeff_meanstd['stdX_w'][:])
    meanY_w = ma.getdata(coeff_meanstd['meanY_w'][:])
    stdY_w = ma.getdata(coeff_meanstd['stdY_w'][:])
    meanX_l = ma.getdata(coeff_meanstd['meanX_l'][:])
    stdX_l = ma.getdata(coeff_meanstd['stdX_l'][:])
    meanY_l = ma.getdata(coeff_meanstd['meanY_l'][:])
    stdY_l = ma.getdata(coeff_meanstd['stdY_l'][:])

    nsto = gv.CHAZ_Int_ENS 
    bt.__dict__['determin'] = np.zeros(bt.StormLon.shape)*np.float64('nan')
    bt.__dict__['stochastic'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float64('nan')
    bt.__dict__['error'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float64('nan')

    NS = np.arange(0,bt.StormYear.shape[0],1)
    #NNS1,NNS2 = np.meshgrid(NS,np.arange(0,nsto,1))
    #nnS1 = da.from_array(NNS1.ravel().astype(dtype=np.int32),chunks=(1,))
    #nnS2 = da.from_array(NNS2.ravel().astype(dtype=np.int32),chunks=(1,))
    nS = da.from_array(NS.astype(dtype=np.int32),chunks=(1,))  

    new = da.map_blocks(get_determin,nS,chunks=(1,), dtype=nS.dtype)
    with ProgressBar():
         n = new.compute(scheduler='synchronous',num_workers=5)
    del new
    gc.collect()
    #for iS in NS:
    #    print iS
    #    b = get_determin(np.array([iS, iS]))
    #print 'down calculate deterministic'

    new =da.map_blocks(get_stochastic,nS,chunks=(1,), dtype=nS.dtype)
    global iNN
    for iNN in range(nsto):
        time1 = time.time()
        with ProgressBar():
            n = new.compute(scheduler='synchronous',num_workers=5)
        gc.collect()
	    #print iNN,time.time()-time1
    del new

   # with open (gv.output_path+'bt_stochastic_det'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','w') as f:
    #     pickle.dump(bt,f)
    #f.close()

    for iens in range(0,1):
        count1 = 0
        if count1 ==0:
            bt1 = bt
            maxCol = bt.PIslp.shape[0]
            count1 = 1
        else:
            for iv in dir(bt):
                if (('__' not in iv) and ('Time' not in iv) and (getattr(bt,iv).ndim==2)):
                    b = np.zeros([maxCol,getattr(bt,iv).shape[1]])*np.float64('nan')
                    b[0:getattr(bt,iv).shape[0],:] = getattr(bt,iv)
                    setattr(bt,iv,b)
                elif (('__' not in iv) and ('Time' not in iv) and ('timeY' not in iv) and (getattr(bt,iv).ndim==3)):
                    b = np.zeros([maxCol,getattr(bt,iv).shape[1],getattr(bt,iv).shape[2]])*np.float64('nan')
                    b[0:getattr(bt,iv).shape[0],:,:] = getattr(bt,iv)
                    setattr(bt,iv,b)
                elif 'Time' in iv:
                    b = np.empty([maxCol,getattr(bt,iv).shape[1]],dtype=object)
                    b[:] = datetime(1800, 1, 1, 0)
                    b[0:getattr(bt,iv).shape[0],:] = getattr(bt,iv)
                    setattr(bt,iv,b)
            for iv in dir(bt1):
                if (('__' not in iv) and ('predictors' not in iv) and ('timeY' not in iv)):
                    b = np.hstack([getattr(bt1,iv), getattr(bt,iv)])
                    setattr(bt1,iv,b)
            del bt
        #### get basin-information
        lon0,lat0 = bt1.StormLon[0,:],bt1.StormLat[0,:]
        basin = util.defineBasin(lon0,lat0,xbin,ybin,basinMap)

        ####
        dummy = bt1.StormLon*bt1.StormLat
        id1 = np.argwhere(dummy!=dummy)[:,0]
        iS1 = np.argwhere(dummy!=dummy)[:,1]
        
        bt1.stochastic[id1,iS1,:] = np.float64('nan')
        mask_arr_stochastic = ma.masked_invalid(bt1.stochastic)

        #### get rid of storms that move fron N.H. to S.H from SH. to NH, from ATL to ENP
        iS1 = np.argmax(mask_arr_stochastic,axis=0)
        basin1 = np.array([util.defineBasin(bt1.StormLon[iS1[nn],nn],\
                  bt1.StormLat[iS1[nn],nn],xbin,ybin,basinMap) \
                  for nn in range(iS1.shape[0])])
        basin0 = np.tile(basin,(40,1)).T
        arg_last = np.array([[pd.Series(bt1.stochastic[:,nn,iN]).last_valid_index()\
                    for nn in range(bt1.StormLon.shape[1])] for iN in range(40)])
        arg_last[arg_last == np.array(None)] = 0
        arg_last = arg_last.astype(None)
        arg_last = np.int_(arg_last).T
        basin2 = np.array([util.defineBasin(bt1.StormLon[arg_last[nn],nn],\
                   bt1.StormLat[arg_last[nn],nn],xbin,ybin,basinMap) \
                   for nn in range(arg_last.shape[0])])
        iS1, iN1 = np.where((((basin0<5)&(basin2>=5))|((basin0>=5)&(basin2<5))|\
                    ((basin0==0)&(basin2==1))))
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')
        iS1,iN1 = np.where((((basin0<5)&(basin1>=5))|((basin0>=5)&(basin1<5))|\
                    ((basin0==0)&(basin1==1))))
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')            
        
        lon0, lat0 = bt1.StormLon[0,:],bt1.StormLat[0,:]
        basin = util.defineBasin(lon0,lat0,xbin,ybin,basinMap)

        #### get rid of storms  that has never developed
        bt1.stochastic[bt1.stochastic==0] = np.float64('nan')
        max5 = np.nanmax(bt1.stochastic[0:21,:,:],axis=0)
        iS1 = np.argwhere(max5<35)[:,0]
        iN1 = np.argwhere(max5<35)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')
        maxall = np.nanmax(bt1.stochastic,axis=0)
        v0 = bt1.stochastic[0,:,:]
        #### get rid of storms that becoming unstable
        iS1 = np.argwhere(maxall>300)[:,0]
        iN1 = np.argwhere(maxall>300)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')
        #### get rid of storms that initially formed with 2 degree lon,lat
        iS1 = np.argwhere(np.abs(bt1.StormLat[0,:])<2)[:,0]
        bt1.stochastic[:,iS1,:] = np.float64('nan')
        bt1.StormLon[0,iS1] = np.float64('nan')
        
        #### get rid of storms that are not from within the range
        iS1 = np.argwhere(basin==0)
        bt1.StormLon[0,iS1] = np.float64('nan')
        
        #### get rid of storms that have no intensity record
        maxall = np.nanmax(bt1.stochastic,axis=0)
        #changed
        iS1 = np.argwhere(maxall<=0)[:,0]
        iN1 = np.argwhere(maxall<=0)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')
        #bt1.stochastic[:,basin!=1,:] = np.float64('nan')
        arg = np.argwhere(bt1.StormLon[0,:] == bt1.StormLon[0,:])[:][:,0]
        
        newlon = bt1.StormLon[:,arg]
        newlat = bt1.StormLat[:,arg]
        newwspd = bt1.stochastic[:,arg]
        newdatenum = bt1.Time[:,arg]
        newYear = bt1.StormYear[arg]
        newMonth = bt1.StormInitMonth[arg]
        
        #for iN in range(20,21):
         #   tracks={'xlong':[],'xlat':[],'v':[],'color':[],'color_n':[],'year':[],'month':[],'day':[],'hour':[]}
          #  maxloc={'xlong':[],'xlat':[],'vmax':[]}
            #arg = np.argwhere((basin_vmax[:,iN]==1)&(basin==1)).ravel()
           # tracks,maxloc = \
            #  get_tracks_maxloc(bt,bt.stochastic[:,:,iN],
             # tracks,maxloc,colorcat,colors)
            #tracks = module_vmaxMap.get_colorCode(tracks,colorcat,colors)

            #plot the spaghetti plots for Obs
            #var = tracks['v'].tolist()
            #titleName = 'ATL '+int2str(iy,4)+'_ens'+int2str(iens,3) 
            #figName='MLR'
            #tracks1 = tracks

            #lon1, lon2, londis = 240,360,30
            #lat1, lat2, latdis = 0,50,15
            #MyFigsize, MyFontsize = (3.5,2),10
            #plcbar = True
            #fig = module_vmaxMap.plt_spaghetti(tracks1,titleName,lon1,lon2,lat1,lat2,MyFigsize,MyFontsize,londis,latdis,jetcmap,colorcat,colors,plcbar)
            #plt.savefig(gv.Model+'_ATL_'+int2str(iy,4)+'_ens'+int2str(iens,3)+'_int_'+int2str(iN,2)+'.png',dpi=300)

        #print('newdatenum:', newdatenum.shape)
        #print('newwspd:', newwspd.shape)
        #print('arg:',arg.shape)
        ### netcdf via xarray
        #times = pd.date_range(start='1950-01-01 00:00', freq:'H',periods=arg.shape[0])

        #dummy dataframe
        ensembleNum = np.arange(newwspd.shape[2])
        stormID = np.arange(arg.shape[0])
        lifelength = np.arange(newdatenum.shape[0])

        ds = xr.Dataset({
          'longitude': xr.DataArray(
                         data = newlon,
                         dims = ['lifelength','stormID'],
                         coords = {'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             '_FillValue': np.float64('nan'),
                             'units': 'degrees east'
                             }
                         ),
           

          'latitude': xr.DataArray(
                         data = newlat,
                         dims = ['lifelength','stormID'],
                         coords = {'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             '_FillValue': np.float64('nan'),
                             'units'     : 'degrees north'
                             }
                         ),
         
          'Mwspd': xr.DataArray(
                         data =np.rollaxis(newwspd,2,0),
                         dims = ['ensembleNum','lifelength','stormID'],
                         coords = {'ensembleNum':ensembleNum,'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             '_FillValue': np.float64('nan'),
                             'units'     : 'kt'
                             }
                         ),
          'year': xr.DataArray(
                         data = newYear,
                         dims = ['stormID'],
                         coords = {'stormID':stormID},
                         attrs = {
                             'units'     : 'year'
                             }
                         ), 
          'time': xr.DataArray(
                         data = date2num(newdatenum, units='days since 1950-01-01 00:00', calendar='standard'),
                         dims = ['lifelength','stormID'],
                         coords = {'lifelength':lifelength, 'stormID':stormID},
                         attrs = {
                             'units'     : 'days since 1950-01-01 00:00'
                             }
                         )
                }
            )

        #print(iens)
        file_name = gv.output_path+gv.Model+'_'+int2str(iy,4)+'_ens'+int2str(iens,3)+'.nc'       
        print(file_name)
        ds.to_netcdf(file_name)

    
    return()	





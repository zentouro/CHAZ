### FILE FROM: `/home/clee/CMIP6-multipleMember/src_uv850t600` (on taroko)

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
import tools.regression4 as reg4
import tools.module_stochastic as module_sto
import time
import random
from datetime import datetime
#from netCDF4 import Dataset
from netCDF4 import Dataset,date2num

import netCDF4 as nc
import numpy.ma as ma
from tools.util import int2str
from tools import util
from dask.diagnostics import ProgressBar
import Namelist as gv

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import xarray as xr 


## adding random number generator with set seed to allow comparisons and reproducibility
## set seed = None to use computer chaos
seed = gv.random_seed
rng = np.random.default_rng(seed=seed)

local_random_state = gv.local_random

## LOCAL SEED HELPER FUNCTION
# import hashlib
# def make_seed(base_seed, storm_id, phase=0, ensemble_id=0):
#     h = hashlib.md5(f"{base_seed}_{storm_id}_{phase}_{ensemble_id}".encode()).digest()
#     return int.from_bytes(h[:4], 'little')

jetcmap = plt.cm.get_cmap("jet",31)
jet_vals = jetcmap(np.arange(31))
cmap1 = mcolors.LinearSegmentedColormap.from_list('newjet', jet_vals)
colors = jet_vals
colorcat = np.arange(0,155,5) 
xbin,ybin,basinMap = util.getbasinMap()

def get_determin(iiS,block_id=None):
    iS = np.int_(iiS.mean(keepdims=True)[0])

    ## LOCAL RANDOM 
    ## using storm-based seed for initialization
    if local_random_state == True:
        init_seed = seed + iS * 1000000 + 1  # +1 for init phase
        init_rng = np.random.default_rng(seed=init_seed)

    #iS  = iiS
    lT = 2
    dummy = bt.StormLon[:,iS][bt.StormLon[:,iS]==bt.StormLon[:,iS]]
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    if ((dummy.any()) and (np.abs(bt.StormLat[0,iS])>=5.)):
        it1 = 0
        it2 = np.int_(np.min([np.argwhere(bt.StormLon[:,iS]==dummy[-1])[-1,0],bt.StormLon.shape[0]-5]))

        ## added if statements for random state
        ## *** LOCAL RANDOM ***
        if local_random_state == True:
            
            if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
                bt.determin[0,iS] = np.max([20,init_rng.choice(intV[intV==intV])]) # kts
        
            else:
                bt.determin[0,iS] = np.max([25,init_rng.choice(intV[intV==intV])]) # kts

        ## *** NON LOCAL RANDOM ***
        if local_random_state == False:
       
            if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
                ## update to use set rng
                ##bt.determin[0,iS] = np.max([20,random.choice(intV[intV==intV])]) # kts
                bt.determin[0,iS] = np.max([20,rng.choice(intV[intV==intV])]) # kts
            
            else:
                ##bt.determin[0,iS] = np.max([25,random.choice(intV[intV==intV])]) # kts
                bt.determin[0,iS] = np.max([25,rng.choice(intV[intV==intV])]) # kts

        
        ## debugging
        #print("determin initial values (first 10):", bt.determin[0, :10])
        #print("unique determin[0] values:", np.unique(bt.determin[0, active_storms]))
        #print("num unique:", len(np.unique(bt.determin[0, active_storms])))


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
                    result,meanX,meanY,stdX,stdY = copy.copy(result_w),copy.copy(meanX_w),\
                                                    copy.copy(meanY_w),copy.copy(stdX_w),copy.copy(stdY_w)
                elif(((bt.landmaskMean[it+lT,iS]> -0.5)|(bt.landmaskMean[it,iS]> -0.5))):
                    predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
                    result,meanX,meanY,stdX,stdY = copy.copy(result_l),copy.copy(meanX_l),\
                                                    copy.copy(meanY_l),copy.copy(stdX_l),copy.copy(stdY_l)
                
               
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
    TimeDepends = ['dThetaEsMean','T200Mean','rhMean','rh500_300Mean','div200Mean']
    iS = np.int_(iiS.mean(keepdims=True)[0])
    dummy = bt.StormLon[:,iS][bt.StormLon[:,iS]==bt.StormLon[:,iS]]
    if ((dummy.any()) and (np.abs(bt.StormLat[0,iS])>=5.)):
        it1 = 0
        it2 = np.min([np.argwhere(bt.StormLon[:,iS]==dummy[-1])[-1,0],bt.StormLon.shape[0]-5])

        ## LOCAL RANDOM
        if local_random_state == True:
            init_seed = seed + iNN * 1000000 + iS * 10000 + 1  # +1 for init phase
            init_rng = np.random.default_rng(seed=init_seed)

        ## added if statements for random state
        ## *** LOCAL RANDOM ***
        if local_random_state == True:

            if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
                bt.stochastic[0, iS, iNN] = np.max([20, init_rng.choice(intV[intV==intV])]) #kts

            else:
                bt.stochastic[0, iS, iNN] = np.max([25, init_rng.choice(intV[intV==intV])])

        ## *** NON LOCAL RANDOM ***
        if local_random_state == False:
            if ((bt.StormLat[0,iS] >=0) and (bt.StormLon[0,iS]>=120) and (bt.StormLon[0,iS]<=180)):
                ## update to use set rng
                ##bt.stochastic[0,iS,iNN] = np.max([20,random.choice(intV[intV==intV])]) # kts
                bt.stochastic[0, iS, iNN] = np.max([20, rng.choice(intV[intV==intV])]) #kts

            else:
                ##bt.stochastic[0,iS,iNN] = np.max([25,random.choice(intV[intV==intV])]) # kts
                bt.stochastic[0, iS, iNN] = np.max([25, rng.choice(intV[intV==intV])]) #kts

        ## debugging
        #print("stochastic initial values (first 10):", bt.stochastic[0, :10, iNN])
        #print("unique stochastic[0] values:", np.unique(bt.stochastic[0, active_storms, iNN]))
        #("num unique:", len(np.unique(bt.stochastic[0, active_storms, iNN])))  

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
                    result,meanX,meanY,stdX,stdY = copy.copy(result_w),copy.copy(meanX_w),\
                                                    copy.copy(meanY_w),copy.copy(stdX_w),copy.copy(stdY_w)
                elif(((bt.landmaskMean[it+lT,iS]> -0.5)|(bt.landmaskMean[it,iS]> -0.5))):
                    predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
                    result,meanX,meanY,stdX,stdY = copy.copy(result_l),copy.copy(meanX_l),\
                                                    copy.copy(meanY_l),copy.copy(stdX_l),copy.copy(stdY_l)
                
                
                h1,v1 = reg4.getPrediction_v0input_result\
                        (bt,meanX,meanY,stdX,stdY,it,\
                        iS,[ih],result,predictors,\
                        TimeDepends,v0,dvdt)
                
                ## DEBUGGING 
                # if iS == 0 and it == 0:
                #     print("=== scalar version iS=0 it=0 ===")
                #     print("v0:", v0)
                #     print("dvdt:", dvdt)
                #     print("v1:", v1)
                
                #error = module_sto.findError(E0,v0E,v0,cat1)
                # ## LOCAL ERROR
                error = module_sto.findError(E0, v0E, v0, cat1, iS, it, iNN)

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

# def calIntensity(iy,ichaz):
#         ipath2 = './'
#         global result_w, meanY_w, stdX_w, meanX_w, stdY_w
#         global result_l, meanY_l, stdX_l, meanX_l, stdY_l

#         bt2 = nc.Dataset(gv.opath,'r')
#         result_wi = nc.Dataset(gv.ipath + 'result_w.nc', 'r')
#         result_w = ma.getdata(result_wi['params'][:])
#         result_li = nc.Dataset(gv.ipath + 'result_l.nc','r')
#         result_l = ma.getdata(result_li['params'][:])
#         observed_data = nc.Dataset(gv.ipath + 'observed_data.nc', 'r')
#         meanX_w_obs = ma.getdata(observed_data['meanX_w_obs'][:])
#         stdX_w_obs = ma.getdata(observed_data['stdX_w_obs'][:])
#         meanY_w_obs = ma.getdata(observed_data['meanY_w_obs'][:])
#         stdY_w_obs = ma.getdata(observed_data['stdY_w_obs'][:])
#         meanX_l_obs = ma.getdata(observed_data['meanX_l_obs'][:])
#         stdX_l_obs = ma.getdata(observed_data['stdX_l_obs'][:])
#         meanY_l_obs = ma.getdata(observed_data['meanY_l_obs'][:])
#         stdY_l_obs = ma.getdata(observed_data['stdY_l_obs'][:])

#         NS = np.argwhere((bt2['StormYear'][:]>=1981)&(bt2['StormYear'][:]<=2012))[:,0]
#         global E0, v0E, cat1, intV
#         intV = bt2['StormMwspd'][:][0,NS]
#         E0, v0E,cat1, = module_sto.get_E1(bt2)
#         del bt2

#         #read the synthetic tracks
#         global bt
#         ## updating 
#         #with open('trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','rb')as f:                 ## github
#         with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','rb')as f:  ### changed from 'r' to 'rb' ~~ 'update' taroko

#                 bt = pickle.load(f)
#                 f.close()
#         ### read mean and std from the HIST data
#         coeff_meanstd = nc.Dataset(gv.pre_path+'coefficient_meanstd.nc','r')     ##github
#         #coeff_meanstd = nc.Dataset('coefficient_meanstd.nc','r')                ##'updated'
#         meanX_w = ma.getdata(coeff_meanstd['meanX_w'][:])
#         stdX_w = ma.getdata(coeff_meanstd['stdX_w'][:])
#         meanY_w = ma.getdata(coeff_meanstd['meanY_w'][:])
#         stdY_w = ma.getdata(coeff_meanstd['stdY_w'][:])
#         meanX_l = ma.getdata(coeff_meanstd['meanX_l'][:])
#         stdX_l = ma.getdata(coeff_meanstd['stdX_l'][:])
#         meanY_l = ma.getdata(coeff_meanstd['meanY_l'][:])
#         stdY_l = ma.getdata(coeff_meanstd['stdY_l'][:])


#         nsto = gv.CHAZ_Int_ENS
#         bt.__dict__['determin'] = np.zeros(bt.StormLon.shape)*np.float64('nan')
#         bt.__dict__['stochastic'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float64('nan')
#         bt.__dict__['error'] = np.zeros([bt.StormLon.shape[0],bt.StormLon.shape[1],nsto])*np.float64('nan')

#         NS = np.arange(0,bt.StormYear.shape[0],1)
#         #NNS1,NNS2 = np.meshgrid(NS,np.arange(0,nsto,1))
#         #nnS1 = da.from_array(NNS1.ravel().astype(dtype=np.int32),chunks=(1,))
#         #nnS2 = da.from_array(NNS2.ravel().astype(dtype=np.int32),chunks=(1,))
#         nS = da.from_array(NS.astype(dtype=np.int32),chunks=(1,))

#         new = da.map_blocks(get_determin,nS,chunks=(1,), dtype=nS.dtype)
#         with ProgressBar():
#                 n = new.compute(scheduler='synchronous',num_workers=5)
#         del new
#         gc.collect()
#         #for iS in NS:
#         #    print iS
#         #    b = get_determin(np.array([iS, iS]))
#         #print 'down calculate deterministic'

#         new =da.map_blocks(get_stochastic,nS,chunks=(1,), dtype=nS.dtype)
#         global iNN
#         for iNN in range(nsto):
#                 time1 = time.time()
#                 with ProgressBar():
#                         n = new.compute(scheduler='synchronous',num_workers=5)
#                 gc.collect()
#                 #print iNN,time.time()-time1
#         del new

#         with open ('bt_stochastic_det'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','wb') as f:
#                 pickle.dump(bt,f)
#         f.close()
#         del bt
#         return()

## version from GitHub
def calIntensity(iy,ichaz):
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

    ## DEBUGGING
    ## Compare intensity distributions between versions
    print("Mean initial intensity:", np.nanmean(bt.determin[0,:]))
    print("Storms with any valid intensity:", np.sum(np.any(~np.isnan(bt.determin), axis=0)))

    ## Check what intV looks like
    print("intV min/max/mean:", np.nanmin(intV), np.nanmax(intV), np.nanmean(intV))
    print("intV below 25:", np.sum(intV < 25))

   # with open (gv.output_path+'bt_stochastic_det'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','w') as f:
    #     pickle.dump(bt,f)
    #f.close()

    print('***CLEANING UP and SAVING .NC***')

    print("After determin computed:", np.sum(np.any(~np.isnan(bt.determin), axis=0)))
    print("After stochastic computed:", np.sum(np.any(~np.isnan(bt.stochastic), axis=(0,2))))


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
        print("After never-developed filter:", np.sum(np.any(~np.isnan(bt1.stochastic), axis=(0,2))))  # ADD


        maxall = np.nanmax(bt1.stochastic,axis=0)
        v0 = bt1.stochastic[0,:,:]
        #### get rid of storms that becoming unstable
        iS1 = np.argwhere(maxall>300)[:,0]
        iN1 = np.argwhere(maxall>300)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')
        print("After unstable filter:", np.sum(np.any(~np.isnan(bt1.stochastic), axis=(0,2))))  # ADD


        #### get rid of storms that initially formed with 2 degree lon,lat
        iS1 = np.argwhere(np.abs(bt1.StormLat[0,:])<2)[:,0]
        bt1.stochastic[:,iS1,:] = np.float64('nan')
        bt1.StormLon[0,iS1] = np.float64('nan')
        print("After lat<2 filter:", np.sum(np.any(~np.isnan(bt1.stochastic), axis=(0,2))))  # ADD

        
        #### get rid of storms that are not from within the range
        iS1 = np.argwhere(basin==0)
        bt1.StormLon[0,iS1] = np.float64('nan')
        print("After basin==0 filter:", np.sum(np.any(~np.isnan(bt1.stochastic), axis=(0,2))))  # ADD

        
        #### get rid of storms that have no intensity record
        maxall = np.nanmax(bt1.stochastic,axis=0)
        #changed
        iS1 = np.argwhere(maxall<=0)[:,0]
        iN1 = np.argwhere(maxall<=0)[:,1]
        bt1.stochastic[:,iS1,iN1] = np.float64('nan')
        #bt1.stochastic[:,basin!=1,:] = np.float64('nan')
        print("After maxall<=0 filter:", np.sum(np.any(~np.isnan(bt1.stochastic), axis=(0,2))))  # ADD


        arg = np.argwhere(bt1.StormLon[0,:] == bt1.StormLon[0,:])[:][:,0]
        print("Final storms written to file:", arg.shape[0])  # ADD

        
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



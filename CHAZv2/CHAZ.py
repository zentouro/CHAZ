#!/usr/bin/env python
import numpy as np

#import subprocess
#import pandas as pd
#import dask.array as da
#import src.module_riskModel as mrisk

import os

import src.module_GenBamPred as GBP
import src.module_GenBamPred_v2 as GBP_v2
import Namelist as gv
import src.caldetsto_track_lysis as CHAZ_detsto
import src.caldetsto_track_lysis_v2 as CHAZ_detsto_v2

import src.calWindCov as calWindCov
import src.preprocess as preprocess
import src.calA as calA

import src.calWindCov_v2 as calWindCov_v2
import src.preprocess_v2 as preprocess_v2
import src.calA_v2 as calA_v2

import src.bt_gen as bt_gen
import src.calCoefficients as calCoefficients

from tools.util import int2str

import pickle
import time

## for exporting namelist variables
import datetime, sys, xarray, types, yaml


start = time.time()


global fpath
fpath = gv.pre_path

#######################
### Training      #####
#######################
### Still in dev  #####
if gv.runTraining:
   if gv.btgen:
      ## creates bt_*.nc in input/
      bt_gen.run_btgen()
   if gv.calCoefficient:
      ## creates coefficient_meanstd.nc
      ## and bt_global_predictors.nc in /input
      ## TODO: implement overwrite protection
      calCoefficients.run_calCoefficients()

#######################
### Pre-Processes #####
#######################
if gv.runPreprocess: 

   ## ORIGINAL
   if not gv.v2: 
      t0 = time.time()
      if not gv.quiet: print('***STOCK PREPROCESSING***')
      if gv.calWind:
         if not gv.quiet: print('calWind')
         calWindCov.run_windCov()
      t1 = time.time()
      if gv.calpreProcess:
         if not gv.quiet: print('calpreProcess')
         ## highest speedup priority
         preprocess.run_preProcesses()
      t2 = time.time()
      if gv.calA:
         if not gv.quiet: print('calA')
         calA.run_calA()
      t3 = time.time()

   ## UPDATED
   if gv.v2: 
      t0 = time.time()
      if not gv.quiet: print('***PREPROCESSING using CHAZ v2 ***')
      if gv.calWind:
         if not gv.quiet: print('calWind')
         calWindCov_v2.run_windCov()

      t1 = time.time()
      if gv.calpreProcess:
         if not gv.quiet: print('calpreProcess')
         preprocess_v2.run_preProcesses()

      t2 = time.time()
      if gv.calA:
         if not gv.quiet: print('calA')
         calA_v2.run_calA()
      t3 = time.time()

   if not gv.quiet: print(f'Preprocess run times: calWind={t1-t0:.2f}  calpreProcess={t2-t1:.2f}  calA={t3-t2:.2f}')


#######################
### Run CHAZ       ####
#######################
if gv.runCHAZ:

   ## ORIGINAL
   if not gv.v2:
      if not gv.quiet: print('*** Running stock CHAZ ***')
      ### get seeding ratio in this experiment
      if gv.TCGIinput != 'random':
         if not gv.quiet: print ('get Seeding ratio for', gv.Model, gv.ENS)
         ipath = gv.pre_path
         ratio = GBP.get_seeding_ratio(ipath, 1981, 2005)
         if not gv.quiet: print ('The seeding ratio is',ratio)

      ### running genesis,track,predictors,& intensity
      ### output yearly 
      for ichaz in range(gv.CHAZ_ENS_0, gv.CHAZ_ENS):
         for iy in range(gv.Year1, gv.Year2+1):

            ### genesis 
            t0 = time.time()
            if gv.calGen:
               if gv.TCGIinput == 'random':
                  print (iy,'random seeding has not yet tested')
                  climInitDate, climInitLon, climInitLat = GBP.randomSeeding(iy,gv.seedN)
               else:
                  print (iy, 'TCGI')
                  climInitDate, climInitLon, climInitLat = GBP.getSeeding(fpath, iy,ratio)

               
               if gv.debugging: 
                  print('saving ClimInit data')
                  np.save(f'{gv.output_path}climInitDate_{iy}', climInitDate)
                  np.save(f'{gv.output_path}climInitLon_{iy}', climInitLon)
                  np.save(f'{gv.output_path}climInitLat_{iy}', climInitLat)
                  print('saved')

            t1 = time.time()
            if gv.calBam:
               if not gv.quiet: print (iy, 'Bam')
               fst = GBP.getBam(climInitDate, climInitLon, climInitLat, iy, ichaz)

               try:
                  fst_file = open(f'{gv.output_path}fst_{iy}', 'wb')
                  pickle.dump(fst, fst_file)
                  fst_file.close()
                  if not gv.quiet: print('fst saved to dictionary')

               except:
                  print('unable to save fst as dictionary')
               
               if not gv.quiet: print (iy, 'calculate predictors')
               GBP.calPredictors(fst,iy,ichaz)
               del fst
            
            t2 = time.time()
            if gv.calInt:
               if not gv.quiet: print (iy, 'calculate Intensities')
               CHAZ_detsto.calIntensity(iy, ichaz)

            t3 = time.time()

      dir_name = gv.ipath
      t = os.listdir(dir_name)

      for item in t:
         if item.endswith(".pik"):
               os.remove(os.path.join(dir_name, item))              

   ## UPDATED
   if gv.v2:
      print('*** Running CHAZ v2 ***')
      ### get seeding ratio in this experiment
      if gv.TCGIinput != 'random':
         if not gv.quiet: print ('get Seeding ratio for', gv.Model, gv.ENS)
         ipath = gv.pre_path

         ## TODO: Add rational for year selection
         ratio = GBP_v2.get_seeding_ratio(ipath, 1981, 2005)
         if not gv.quiet: print ('The seeding ratio is',ratio)

      
      ### running genesis,track,predictors, & intensity
      ### output yearly 
      for ichaz in range(gv.CHAZ_ENS_0, gv.CHAZ_ENS):
         for iy in range(gv.Year1, gv.Year2+1):

            output_fname = f"{gv.output_path}{gv.Model}_{iy}_ens{int2str(ichaz,3)}.nc"
            print(output_fname)
            ### if the file does not exist and we do not want to ovewrite
            ### skip running this section
            if not gv.overwrite and os.path.exists(output_fname):
               print(f'{output_fname} already exists, skipping')
               pass

            ### if overwrite is true, regardless of file existence
            ### or if overwrite is false but the file exists
            ### run this section
            else:
               try:
                  ### genesis 
                  t0 = time.time()
                  if gv.calGen:
                    
                     if gv.TCGIinput == 'random':
                        print (iy,'random seeding has not yet tested')
                        climInitDate, climInitLon, climInitLat = GBP_v2.randomSeeding(iy,gv.seedN)
                     else:
                        if not gv.quiet: print (iy, 'TCGI')
                        climInitDate, climInitLon, climInitLat = GBP_v2.getSeeding(fpath, iy, ratio)

                     ## debugging
                     if gv.debugging: 
                        print('saving calGen initiation data')
                        np.save(f'{gv.output_path}climInitDate_{iy}', climInitDate)
                        np.save(f'{gv.output_path}climInitLon_{iy}', climInitLon)
                        np.save(f'{gv.output_path}climInitLat_{iy}', climInitLat)
                        print('saved')
                  
                  t1 = time.time()
                  if gv.calBam:
                     if not gv.quiet: print (iy, 'Bam')
                     fst = GBP_v2.getBam(climInitDate, climInitLon, climInitLat, iy, ichaz)
                     
                     try:
                        fst_file = open(f'{gv.output_path}fst_{iy}', 'wb')
                        pickle.dump(fst, fst_file)
                        fst_file.close()
                        if not gv.quiet: print('fst saved to dictionary')

                     except:
                        print('unable to save fst as dictionary')
                     
                     if not gv.quiet: print (iy, 'calculate predictors')
                     GBP_v2.calPredictors(fst,iy,ichaz)
                     del fst
                  
                  t2 = time.time()
                  if gv.calInt:
                     
                     if not gv.quiet: print (iy, 'calculate Intensities')
                     CHAZ_detsto_v2.calIntensity(iy, ichaz)

                  t3 = time.time()
               except Exception as e:
                  print(f'Failed for {iy} and realization {ichaz}')
                  print(e)

      dir_name = gv.ipath
      t = os.listdir(dir_name)

      for item in t:
         if item.endswith(".pik"):
               os.remove(os.path.join(dir_name, item))  

   ## output timing for most recent year
   if not gv.quiet: print(f'CHAZ run times (for {iy}): calGen={t1-t0:.2f}  calBam={t2-t1:.2f}  calInt={t3-t2:.2f}')


end = time.time()

## save out Namelist information
print(f'Logging Namelist variables at {gv.output_path}/Namelist_config.yaml')
cfg = {
    k: v for k, v in vars(gv).items()
    if not k.startswith("_")        ## don't need to save all the python files
    and not k.startswith('ll')      ## don't need to save the generated landmask arrays
    and not k.startswith('ld')      ## don't need to save the generated landmask arrays
    and not isinstance(v, types.ModuleType)
    and not callable(v)
}

meta = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "python": sys.version.split()[0],
    "xarray": xarray.__version__,
    "argv": sys.argv,
}

with open(gv.output_path+'Namelist_config.yaml', 'w') as f:
    yaml.safe_dump({"_meta": meta, **cfg}, f, sort_keys = False)
          

## output total runtime
print(f"Total runtime of CHAZ is {np.round(end - start,2)} seconds")

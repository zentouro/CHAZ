#!/usr/bin/env python

### suppresses the following, uncomment to see if new ones have appeared 
# RuntimeWarning: Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or 
# arrays referring to its data still exist. All data arrays obtained from such files refer 
# directly to data on disk, and must be copied before the file can be cleanly closed. 
# (See netcdf_file docstring for more information on mmap.)
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

from scipy.io import loadmat, netcdf_file

def get_landmask(filename):
    """
    This function reads landmask.nc. 
    read 0.25degree landmask.nc  -- updating to 0.125
    output:
        lon: longitude, 1D
        lat: latitude, 1D
        landmask:2D
  
    """
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask

### Experiment settings
Model = 'ERA5'
ENS = 'r1i1p1f1'
TCGIinput = 'TCGI_CRH_PI'          # or "TCGI_SD_RI" or 'TCGI_CRH'
CHAZ_ENS_0 = 0           
CHAZ_ENS = 1                      # Number of ensemble realizations.  
CHAZ_Int_ENS = 40                  # Number of intensity realizations

### CHAZ parameters
### TODO: describe how to create these 
monthlycsv = '/home/miriamn/CHAZ/CHAZvectorized/list.txt'
dailycsv = '/home/miriamn/CHAZ/CHAZvectorized/list.txt'

uBeta = -1.5
vBeta = 2.0
survivalrate = 0.78
#seedN = 1                              # annual seeding rate for random seeding, not yet ready to use
seedN = 1000                                 
landmaskfile = 'input/landmask.nc'  
ipath = 'input/original/'    
#ipath = 'input/updating/updated-error/'                        ## ipath contains input data from observations (best tracks for all the basins from IBTrACS)
opath = 'input/original/bt_global_predictors.nc'
#opath = 'input/updating/updated-error/bt_global_predictors.nc' ## and a global best track with intial predictors 

## Preprocessing output path 
## AND
## CHAZ import path
pre_path = '/data0/miriamn/CHAZvectorized/pre/'

## CHAZ output path
output_path = '/data0/miriamn/CHAZvectorized/output/101826-landmask-check/'

## Years to run preproecessing AND CHAZ
## Training (when implemented) will default to 1980 to Year2
Year1 = 1950
Year2 = 2025

### Defining landmask
llon, llat, lldmask = get_landmask(landmaskfile)
lldmask = lldmask[::-1,:]               ## flips latitudes so they are in the right order 
ldmask = lldmask[::-24,  ::24]          ## 2º
ldldmask = lldmask[::-9, ::9]           ## .75º [used in module_GenBamPred.bam and .get_predictors]
ldlon = llon[::9]
ldlat = llat[::9]
#lldmask = lldmask[::-1,:]              ## flips latitudes so they are in the right order
#lldmask = ldldmask                      ## uses updated landmask throughotu                          

####################################################
#### run CHAZ using v1 (beta)                   ####
#### or v2 (updated and vectorized)             ####
####                                            ####
#####################################################
#[use the updated v2 version of CHAZ, rather than the original]
v2 = True       ## applied to both preprocess AND CHAZ 

## for CHAZ
## random number generator with set seed to allow comparisons and reproducibility
## set seed = None to use computer chaos
random_seed = 42
local_random = False     ## for debugging, prescribes identical randomness to both CHAZ v1 and v2 


## TODO UPDATE TO LOGGING
## output progress and timing logs 
# log_path = '/logs/'
## log_level = TK 

## debugging/verbose
## these only work for v2 implementation
debugging = True     ## select True to include debug print statements and file saves    
quiet = False        ## select True to suppress print statements while running           

overwrite = True      ## select True to overwrite existing output                         

#####################################################
## Preprocesses                                   ###
## ignore variables when runPreprocess = False    ###
#####################################################
runPreprocess = False 
calWind = True 
calpreProcess = True 
calA = True

#####################################################
## Training                                       ###
## ignore variables when runTraining = False      ###
## not yet implemented                            ###
#####################################################
### NOTE: TRAINING REQUIRES PREPROCESSING         ###

### STILL IN DEVELOPMENT
runTraining = False
btgen = True
calCoefficient = True


#####################################################
## CHAZ                                           ###
## ignore variables when runCHAZ = False          ###
#####################################################
runCHAZ = True
### genesis 
calGen = True   
### track
calBam =  True
### intensiy
calInt = True 


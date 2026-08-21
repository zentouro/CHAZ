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
TCGIinput = 'TCGI_CRH_PI'       # or "TCGI_SD_RI" or 'TCGI_CRH'
CHAZ_ENS_0 = 0           
CHAZ_ENS = 15                    # Number of track realizations.  
CHAZ_Int_ENS = 40                # Number of intensity realizations
#PImodelname = 'ERA5'            ### No longer needed

### CHAZ parameters
#monthlycsv = '/home/miriamn/CHAZ/CHAZvectorized/era5data/list1.txt' #'era5data/list1.txt'                    
#dailycsv = '/home/miriamn/CHAZ/CHAZvectorized/era5data/list1.txt' #'era5data/list1.txt'

## ERA5 data from /xpt/taroko.local/data0/clee/ERA5/2D
monthlycsv = '/home/miriamn/CHAZ/CHAZvectorized/era5data/list2.txt'
dailycsv = '/home/miriamn/CHAZ/CHAZvectorized/era5data/list2.txt'

uBeta = -1.5
vBeta = 2.0
survivalrate = 0.78
#seedN = 1                              # annual seeding rate for random seeding, not yet ready to use
seedN = 1000                            # what is in the tutorial, not sure if this will break things     
landmaskfile = 'input/landmask.nc'      
ipath = 'input/'                        
opath = 'input/bt_global_predictors.nc'

## output for preprocessing data and/or to import from for CHAZ (when data loaded locally)
pre_path = '/home/miriamn/CHAZ/CHAZvectorized/pre/'   
## when running chaz in beta/use known pre-processing data                        
#pre_path = '/xpt/taroko.local/data0/clee/ERA5/wdir-Landmask075_20250514/'  
#  
output_path = 'output/'     ## output of CHAZ run

### local diretory preprocessing limited to year 2000-2009
### can expand when using known preprocessing 
### TODO: update list/preprocessing pointers to run on full timeseries 
Year1 = 2000
Year2 = 2009


### UPDATING LANDMASK
llon, llat,lldmask = get_landmask(landmaskfile)
ldmask = lldmask[::-24,  ::24]          ## 2º
ldldmask = lldmask[::-9, ::9]           ## .75º [used in module_GenBamPred.bam and .get_predictors]
ldlon = llon[::9]
ldlat = llat[::9]
lldmask = lldmask[::-1,:]               ## flips latitudes so they are in the right order


####################################################
#### use Beta version of CHAZ with optimized    ####
#### and vectorized code                        ####
####                                            ####
#####################################################

vectorize = True    ## applied to both preprocess AND CHAZ

## for CHAZ
random_seed = 42
local_random = False 

# debugging/verbose
debugging = True      ## select True to include debug print statements and file saves     --  # partially implemented
quiet = False         ## select True to suppress print statements while running           --  # partially implemented
overwrite = False     ## select True to overwrite existing output                         -- # partially implemented (for CHAZ)

#####################################################
# Preprocesses                                   ####
# ignore variables when run Preprocess is False ####
#####################################################
runPreprocess = False 
calWind = True 
calpreProcess = True 
calA = True
###################################################
# CHAZ                                         ####
# ignore variables when run CHAZ is False     ####
###################################################
runCHAZ = True
### genesis 
calGen = True   
### track
calBam =  True
### intensiy
calInt = True 


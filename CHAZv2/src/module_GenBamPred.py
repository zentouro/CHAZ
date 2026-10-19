## FROM: /home/clee/CMIP6-multipleMember/src_uv850t600/module_GenBamPred.py
## REPLACED np.float_ with np.float64 for NumPy 2.0
## REPLACED np.NaN with np.nan
## ADDED: random seeding for repeatability 


#!/usr/bin/env python
import numpy as np
import calendar
import random
import time
import sys
import dask.array as da
import os
import gc
import pickle
import copy
import pandas as pd
import Namelist as gv
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
from scipy import stats
from tools.util import argminDatetime
from tools.util import int2str,date_interpolation
from scipy.io import loadmat, netcdf_file
from datetime import datetime,timedelta
from dask.diagnostics import ProgressBar

import src.calWindCov as calWindCov

## adding random number generator with set seed to allow comparisons and reproducibility
## set seed = None to use computer chaos
seed = gv.random_seed
rng = np.random.default_rng(seed=seed)

## TODO: If using, replace with rng
def randomSeeding(iy):
    '''
    random seeding
    '''
    climInitLon = []
    climInitLat = []
    climInitDate = []
    a= np.arange(gv.lldmask.size).reshape(gv.lldmask.shape[0],gv.lldmask.shape[1])
    llon1,llat1 = np.meshgrid(gv.llon,gv.llat)
    dummy = np.random.choice(a[(gv.lldmask==0)&(np.abs(llat1)<=65)&(np.abs(llat1)>=3)],gv.seedN)
    y = dummy/a.shape[1]
    x = np.mod(dummy,a.shape[1])
    yeardays = np.array([calendar.monthrange(iy,im)[1] for im in range(1,13)]).sum()
    a = np.arange(0,yeardays,1)
    days = np.random.choice(a,gv.seedN)
    date = pd.date_range(str(iy)+'-01-01', periods=365+calendar.isleap(iy)*1)[days]
    for idd in range(date.size):
        x1 = np.arange(gv.llon[x[idd]]-1,gv.llon[x[idd]]+1.01,0.01)
        y1 = np.arange(gv.llat[y[idd]]-1,gv.llat[y[idd]]+1.01,0.01)
        xx = np.random.choice(x1,1)
        yy = np.random.choice(y1,1)
        climInitDate.append(datetime.combine(date[idd].date(), datetime.min.time()))
        climInitLon.append(xx)
        climInitLat.append(yy)
    climInitDate = np.array(climInitDate).ravel()
    climInitLon = np.array(climInitLon).ravel()
    climInitLat = np.array(climInitLat).ravel()

    return climInitDate, climInitLon,climInitLat

def TCgiSeeding (gxlon,gxlat,gi,climInitLon,climInitLat,climInitDate,ratio,iy):
        '''
        This is a subroutine for calculate genesis location based on gi.
        gi (Month X nlat X nlon): 
                gi is tropical cyclone genesis index. We use nansum(gi) to for 
                totoal number of seeds, and then randomly select location using gi 
                as genesis probability.
        gxlon (nlat X nlon): mesh-grid longitude
        gxlat (nlat X nlon): mesh-grid latitude 
        climInitLon,climInitLat,climInitDate: list contains previous seeds information
        ratio: how much more seeds we should gives, based on the survival rate
        '''
        for im in range(0,12,1):
                xk,pk = np.arange(gi[im,:,:].size),gi[im,:,:].ravel()
                
                dummy = xk*pk
                xk,pk = xk[dummy==dummy],pk[dummy==dummy]
                
                ## update random
                custm = stats.rv_discrete(name='custm', values=(xk, pk/pk.sum()), seed = rng)
                #custm = stats.rv_discrete(name='custm', values=(xk, pk/pk.sum()))
                r = custm.rvs(size=np.int_(np.rint(pk.sum()*ratio)))
                
                iix = gxlon.ravel()[r.ravel()]
                iiy = gxlat.ravel()[r.ravel()]
                
                ## update random
                iday = rng.choice(np.arange(calendar.monthrange(iy,im+1)[1]),size=np.int_(np.rint(pk.sum()*ratio)),replace=True)
                #iday = np.random.choice(np.arange(calendar.monthrange(iy,im+1)[1]),size=np.int_(np.rint(pk.sum()*ratio)),replace=True)
                
                if iday.size>0:
                    for id in range(iday.size):

                        x1 = np.arange(iix[id]-1,iix[id]+1.01,0.01)
                        y1 = np.arange(iiy[id]-1,iiy[id]+1.01,0.01)

                        ## update random
                        xx = rng.choice(x1, 1)
                        yy = rng.choice(y1,1)
                        #xx = np.random.choice(x1,1)
                        #yy = np.random.choice(y1,1)

                        climInitDate.append(datetime(iy,im+1,iday[id]+1,0,0))
                        climInitLon.append(xx)
                        climInitLat.append(yy)
        return climInitLon,climInitLat,climInitDate


def calF(nday):
     #### furior function
     dt = 1.0*60*60 #1 hr 
     #T = np.float64(nday) 
     T = np.float64(15)
     N = 15
     nt = np.arange(0,nday*60*60*24,dt)
     F = np.zeros([nt.shape[0],4])
     F1 = np.zeros([nt.shape[0],4])

     #F = np.zeros([24,4])
     #F1 = np.zeros([24,4])
     for iff in range(0,4,1):
         X = np.zeros([15])
         ## update random
         X = [rng.uniform(0,1) for iN in range(N)]
         #X = [random.uniform(0,1) for iN in range(N)]
         for itt in range(nt.shape[0]):
         #for itt in range(24):
             F[itt,iff] = (np.sqrt(2.0/np.sum([iN**(-3.0) for iN in range(1,N+1,1)]))*\
                       np.sum([(iN**(-3.0/2.0))*np.sin(2.0*np.pi*(iN*itt/(24.*T)+X[iN-1]))
                       for iN in range(1,N+1,1)]))

     return F


def getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate):

       #### modify Beta
       earth_rate = 7.2921150e-5 #mean earth rotation rate in radius per second
       r0 = 6371000 # mean earth radius in m
       lat0 = np.arange(-90,100,10)
       phi0 = lat0/180.*np.pi #original latitude in radian (ex. at 15 degree N)
       beta0 = 2.0*earth_rate*np.cos(phi0)/r0 # per second per m
       beta0 = beta0/beta0[10]
       ratio = np.interp(fstLat,np.arange(-90,100,10),beta0)
       uBeta = gv.uBeta*ratio
       vBeta = gv.vBeta*ratio
       ################

       alpha = 0.8
       uTrack = alpha*u850+(1.-alpha)*u250+uBeta
       vTrack = alpha*v850+(1.-alpha)*v250+vBeta*np.sign(np.sin(fstLat*np.pi/180.))
       dx = uTrack*dt
       dy = vTrack*dt
       lon2,lat2 = getLonLatfromDistance(fstLon,fstLat,dx,dy)
       fstLon,fstLat = lon2,lat2
       fstDate += timedelta(seconds=dt)
       #print uBeta, vBeta,fstLon,fstLat
       return fstLon,fstLat,fstDate

def getLonLatfromDistance(lonInit,latInit,dx,dy):
    er = 6371000 #km
    londis = 2*np.pi*er*np.cos(latInit/180*np.pi)/360.
    lon2 = lonInit+dx/londis
    latdis = 2*np.pi*er/360.
    lat2 = latInit+dy/latdis
    return lon2,lat2


## from file source
# def bam(iS,block_id=None):
#         #print iS
#         neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
#         dt = 1.0*60*60
#         T = np.float64(15)
#         N = 15
#         F = calF(15)
#         nt = np.arange(0,T*60*60*24,dt)
#         b = np.int_(iS.mean(keepdims=True)[0])
#         #b = iS
#         fstDate = climInitDate[b]
#         fstLon = climInitLon[b]
#         fstLat = climInitLat[b]
#         fstlon[0,b] = fstLon
#         fstlat[0,b] = fstLat
#         fstldmask[0,b] = 0
        
#         fstu250[0,b] = 0
#         fstv250[0,b] = 0
#         fstu850[0,b] = 0
#         fstv850[0,b] = 0
        
#         endhours = fstlon.shape[0]-1
#         endDate = climInitDate[b] + timedelta(hours = endhours)
#         count,year0,month0,day0 = 1,0,0,0
#         fpath = './'
#         while fstDate < endDate:
#                 if (fstDate.year != year0) :
#                         fileName = fpath+int2str(fstDate.year,4)+'_'+gv.ENS+'.nc'
#                         #print fstDate.year,year0
#                         if os.path.isfile(fileName):
#                                 nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
#                                 u250m = nc.variables['ua2502'][:]
#                                 u850m = nc.variables['ua8502'][:]
#                                 v250m = nc.variables['va2502'][:]
#                                 v850m = nc.variables['va8502'][:]
#                                 u850m = calWindCov.fillinNaN(u850m,neighbors)
#                                 v850m = calWindCov.fillinNaN(v850m,neighbors)
#                                 xlong = nc.variables['Longitude'][:]
#                                 xlat = nc.variables['Latitude'][:]
#                                 xxlong,xxlat = np.meshgrid(xlong,xlat)
#                                 nc.close()
#                                 year0 = fstDate.year
#                         else:
#                                 print('no'+fileName)
#                                 break
#                 if fstDate.day != day0:
#                         #print fstDate.day,day0 
#                         u250m2d = date_interpolation(fstDate,u250m)
#                         v250m2d = date_interpolation(fstDate,v250m)
#                         u850m2d = date_interpolation(fstDate,u850m)
#                         v850m2d = date_interpolation(fstDate,v850m)
#                         day0=fstDate.day

#                 FileAName =\
#                 fpath+'A_'+int2str(fstDate.year,4)+int2str(fstDate.month,2)+'.nc'
#                 ncA = Dataset(FileAName,'r',format='NETCDF3_CLASSIC')
#                 A = ncA.variables['A'][:,fstDate.day-1,:,:]
#                 ncA.close()
#                 day0 = fstDate.day

#                 distance = np.sqrt((fstLon-xxlong)**2+(fstLat-xxlat)**2)
#                 iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
#                 iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
#                 iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])

#                 iit = np.mod(count,nt.shape[0])
#                 u250 = u250m2d[iy1:iy2+1,ix1:ix2+1]+A[0,iy1:iy2+1,ix1:ix2+1]*F[iit,0]
#                 v250 = v250m2d[iy1:iy2+1,ix1:ix2+1]+A[1,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+A[2,iy1:iy2+1,ix1:ix2+1]*F[iit,1]
#                 u850 = u850m2d[iy1:iy2+1,ix1:ix2+1]+A[3,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
#                                              A[4,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[5,iy1:iy2+1,ix1:ix2+1]*F[iit,2]
#                 v850 = v850m2d[iy1:iy2+1,ix1:ix2+1]+A[6,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
#                                              A[7,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[8,iy1:iy2+1,ix1:ix2+1]*F[iit,2]+\
#                                              A[9,iy1:iy2+1,ix1:ix2+1]*F[iit,3]

#                 #try: 
#                 #   mytest = np.min(u850[~u850.mask])
#                 #except ValueError: 
#                 #   print 'mountainous'
#                 #   break;

#                 u250 = np.nanmean(u250)
#                 u850 = np.nanmean(u850)
#                 v250 = np.nanmean(v250)
#                 v850 = np.nanmean(v850)

#                 fstLon, fstLat, fstDate = getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate)
#                 #print fstDate,year0,day0
#                 if ((fstLon<0.0) or (fstLon>360) or (fstLat<-60) or (fstLat>60)):
#                         #print b, 'break for going to the space'
#                         break
#                 fstlon[count,b] = fstLon
#                 fstlat[count,b] = fstLat
#                 fstldmask[count,b] = np.rint(np.nanmean(gv.ldmask[iy1:iy2+1,ix1:ix2+1]))
#                 fstu250[count,b] = u250
#                 fstv250[count,b] = v250
#                 fstu850[count,b] = u850
#                 fstv850[count,b] = v850
#                 del u250,u850,v250,v850
#                 count += 1
#         return b

## FROM GITHUB
def bam(iS,block_id=None):
    '''
    This function uses a beta advection model to find track. 
    '''
    #print iS
    neighbors=((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2),(2,0),(-2,0))
    
    dt = 1.0*60*60
    #T = 15
    N = 15
    #F = calF(15)

    T = np.float64(15)      # 15-day period of the fourier series (i think)
    F = calF(15)            # computer fourier series 
    
    nt = np.arange(0,T*60*60*24,dt)
    #b = np.int_(iS.mean(keepdims=True))
    b = np.int_(iS.mean(keepdims=True)[0])
    ## this slows things down
    #print(b, 'b')
    #print(climInitDate[0].year, 'climInitDate[0].year')
    #b = iS
    fstDate = climInitDate[b]
    fstLon = climInitLon[b]
    fstLat = climInitLat[b]
    fstlon[0,b] = fstLon
    fstlat[0,b] = fstLat
    fstldmask[0,b] = 0
    endhours = fstlon.shape[0]-1
    endDate = climInitDate[b] + timedelta(hours = endhours)
    count,year0,month0,day0 = 1,0,0,0
    fpath = gv.pre_path

    ### using 0.75 ldmask ###
    ldxxlong,ldxxlat = np.meshgrid(gv.ldlon,gv.ldlat)
    ######

    while fstDate < endDate:
        if (fstDate.year != year0) :
            #print (fstDate.year,year0) 
            fileName = fpath+int2str(fstDate.year,4)+'_'+gv.ENS+'.nc'
            
            if os.path.isfile(fileName):
                nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
                u250m = nc.variables['ua2502'][:]
                u850m = nc.variables['ua8502'][:]
                v250m = nc.variables['va2502'][:]
                v850m = nc.variables['va8502'][:]
                u850m = calWindCov.fillinNaN(u850m,neighbors)
                v850m = calWindCov.fillinNaN(v850m,neighbors)
                xlong = nc.variables['Longitude'][:]
                xlat = nc.variables['Latitude'][:]
                xxlong,xxlat = np.meshgrid(xlong,xlat)
                nc.close()	
                year0 = fstDate.year

            else:
                print ('no'+fileName)
                break

        if fstDate.day != day0:
            #print fstDate.day,day0	
            u250m2d = date_interpolation(fstDate,u250m)
            v250m2d = date_interpolation(fstDate,v250m)
            u850m2d = date_interpolation(fstDate,u850m)
            v850m2d = date_interpolation(fstDate,v850m)
            day0=fstDate.day

        FileAName =\
        fpath+'A_'+int2str(fstDate.year,4)+int2str(fstDate.month,2)+'.nc'
        ncA = Dataset(FileAName,'r',format='NETCDF3_CLASSIC')
        A = ncA.variables['A'][:,fstDate.day-1,:,:]
        ncA.close()
        day0 = fstDate.day

        distance = np.sqrt((fstLon-xxlong)**2+(fstLat-xxlat)**2)
        iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
        iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
        iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])

        iit = np.mod(count,nt.shape[0])
        u250 = u250m2d[iy1:iy2+1,ix1:ix2+1]+A[0,iy1:iy2+1,ix1:ix2+1]*F[iit,0]
        v250 = v250m2d[iy1:iy2+1,ix1:ix2+1]+A[1,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+A[2,iy1:iy2+1,ix1:ix2+1]*F[iit,1]
        u850 = u850m2d[iy1:iy2+1,ix1:ix2+1]+A[3,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
                            A[4,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[5,iy1:iy2+1,ix1:ix2+1]*F[iit,2]
        v850 = v850m2d[iy1:iy2+1,ix1:ix2+1]+A[6,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
                            A[7,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[8,iy1:iy2+1,ix1:ix2+1]*F[iit,2]+\
                            A[9,iy1:iy2+1,ix1:ix2+1]*F[iit,3]

        #try: 
        #   mytest = np.min(u850[~u850.mask])
        #except ValueError: 
        #   print 'mountainous'
        #   break;
            
        u250 = np.nanmean(u250)
        u850 = np.nanmean(u850)
        v250 = np.nanmean(v250)
        v850 = np.nanmean(v850)

        fstLon, fstLat, fstDate = getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate)
        #print fstDate,year0,day0
        if ((fstLon<0.0) or (fstLon>360) or (fstLat<-60) or (fstLat>60)):
                #print b, 'break for going to the space'
            ## not sure what this break is? 
            break
        fstlon[count,b] = fstLon
        fstlat[count,b] = fstLat


        #### recalculating the ix,iy landmask ####  
        distance = np.sqrt((fstLon-ldxxlong)**2+(fstLat-ldxxlat)**2)  
        iy,ix = np.unravel_index(np.argmin(distance),distance.shape)  
        iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])  
        iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])  
        fstldmask[count,b] = np.rint(np.nanmean(gv.ldldmask[iy1:iy2+1,ix1:ix2+1]))  
        ########################################### 

        ##### NEEDS UPDATING (see recalculating `/data0/clee/ERA5/wdir-Landmask075_20250514`)
        #fstldmask[count,b] = np.rint(np.nanmean(gv.ldmask[iy1:iy2+1,ix1:ix2+1])) 
        del u250,u850,v250,v850
        count += 1

    return b

            
def get_landmask(filename):
    """
    read 0.25degree landmask.nc 
    output:
    lon: 1D
    lat: 1D
    landmask:2D
  
    """
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask

def removeland(iS):
    #print iS
    b = np.int_(iS.mean(keepdims=True)[0])
    iT3 = -1
    if b<fstldmask.shape[1]:
        a = np.argwhere(fstldmask[:,b]==3).ravel()
        if a.size:
           if a.size>3:
              iT3 = a[0]+2
              fstlon[iT3:,b]=np.nan
              fstlat[iT3:,b]=np.nan
    return iT3


class fst2bt(object):
    """
      convert data format from fst to be format
    """
    def __init__(self,data):
        self.StormId = np.arange(0,data['lon'].shape[1],1)
        self.StormYear = []
        self.StormInitMonth = []
        for iS in self.StormId:
            if data['Time'][0,iS] is not None:
               self.StormYear.append(data['Time'][0,iS].year)
               self.StormInitMonth.append(data['Time'][0,iS].month)
            else:
               self.StormYear.append(1800)
               self.StormInitMonth.append(1)
        self.StormYear = np.array(self.StormYear)
        self.StormInitMonth = np.array(self.StormInitMonth)
        for iS in range(data['Time'].shape[1]):
            data['Time'][:,iS] = np.array([datetime(1800,1,1,0) if v is None else v for v in data['Time'][:,iS]])
        data['Time'][data['lon']!=data['lon']] = datetime(1800,1,1,0)
        data['lon'][data['Time']==datetime(1800,1,1,0)] = np.float64('Nan')
        self.StormLon = data['lon']
        self.StormLat = data['lat']
        self.Time = data['Time']
        self.PIwspd = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.PIslp = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.PIwspdMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dPIwspd = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.PIslpMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.UShearMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.UShear = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.VShearMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.VShear = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.div200Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.div200 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T200Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T200 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rh500_300Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rh500_300 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rhMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.rh = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T100Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.T100 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaEMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaE = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaEs = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.dThetaEsMean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.u250Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.u250 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.v250Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.v250 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.u850Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.u850 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.v850Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.v850 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        #self.ta600Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        #self.ta600 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        #self.hur600Mean = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        #self.hur600 = np.empty(data['lon'].shape,dtype=float)*np.float64('Nan')
        self.landmask = data['ldmask']
        self.landmask = data['ldmask']
        self.landmaskMean = data['ldmask']
        self.trSpeed = np.zeros(data['lon'].shape)*np.float64('Nan')
        self.trDir = np.zeros(data['lon'].shape)*np.float64('Nan')
        self.dVdt = np.zeros(data['lon'].shape)

def func_first(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x.first_valid_index()
def func_last(x):
    if x.last_valid_index() is None:
        return None
    else:
        return x.last_valid_index()

def get_predictors(iiS,block_id=None):
        iS = np.int_(iiS.mean(keepdims=True)[0])
        predictors=['StormMwspd','dVdt','trSpeed','dPIwspd','SHRD','rhMean','dPIwspd2','dPIwspd3','dVdt2','landmaskMean']
        #if bt.StormYear[iS]>=1980:
        #fpath = './'
        #fileName = fpath+int2str(bt.StormYear[iS],4)+'_'+gv.ENS+'.nc'
        fpath = gv.pre_path         ## update
        fileName = fpath+int2str(bt.StormYear[iS],4)+'_'+gv.ENS+'.nc'
        
        nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
        xlong = nc.variables['Longitude'][:]
        xlat = nc.variables['Latitude'][:]
        PIVmax = nc.variables['PI2'][:,:,:]*1.94384449 #m/s - kt
        u250 = nc.variables['ua2502'][:]
        u850 = nc.variables['ua8502'][:]
        v250 = nc.variables['va2502'][:]
        v850 = nc.variables['va8502'][:]
        #ta600 = nc.variables['ta6002'][:]
        #hur600 = nc.variables['hur6002'][:]
        u = u250-u850
        v = v250-v850
        meanrh = nc.variables['hur2'][:]
        xxlong,xxlat = np.meshgrid(xlong,xlat)
        del xlong, xlat
        nc.close()
        for it in range(0,bt.StormLon[:,iS].shape[0],1):
                if bt.Time[it,iS] != datetime(1800,1,1,0):
                        distance = np.empty(xxlong.shape,dtype=float)
                        er = 6371.0 #km
                        londis = 2*np.pi*er*np.cos(xxlat/180*np.pi)/360
                        dx = londis*(xxlong - bt.StormLon[it,iS])
                        dy = 110 * (xxlat - bt.StormLat[it,iS])
                        distance = np.sqrt(dx*dx+dy*dy)
                        (j0,i0) = np.unravel_index(np.argmin(distance),distance.shape)

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],PIVmax),0,500
                        bt.PIwspdMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.PIwspd[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],u),200,800
                        bt.UShearMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.UShear[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],v),200,800
                        bt.VShearMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.VShear[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],meanrh),200,800
                        bt.rhMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.rh[it,iS] = var[j0,i0]

                        ## TODO: RECALCULATE LANDMASK 
                        var,radius1,radius2 = copy.copy(gv.ldmask),0,300
                        var[var==0] = -1.
                        var[var==3] = 0.0
                        bt.landmaskMean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.landmask[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],u250),200,800
                        bt.u250Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.u250[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],v250),200,800
                        bt.v250Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.v250[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],u850),200,800
                        bt.u850Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.u850[it,iS] = var[j0,i0]

                        var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],v850),200,800
                        bt.v850Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        bt.v850[it,iS] = var[j0,i0]

                        #var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],ta600),200,800
                        #bt.ta600Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        #bt.ta600[it,iS] = var[j0,i0]

                        #var,radius1,radius2 =  date_interpolation(bt.Time[it,iS],hur600),200,800
                        #bt.hur600Mean[it,iS] = np.mean(var[(distance<=radius2) & (distance>=radius1) & (var==var)])
                        #bt.hur600[it,iS] = var[j0,i0]

        return iS

def getStormTranslation(lon,lat,time):

        er = 6371.0 #km
        timeInt=[]
        lonInt=[]
        latInt=[]

        for iN in range(time.shape[0]-1):
                timeInt.append(time[iN])
                lonInt.append(lon[iN])
                latInt.append(lat[iN])
                #delt = (time[iN+1]-time[iN]).seconds/60/60
                #if delt ==0:
                #   print time
                delt = 6
                inv = 1./np.float64(delt)
                #if time[iN+1]-time[iN] == timedelta(hours=delt):
                for iM in range(1,delt,1):
                        timeInt.append(time[iN]+timedelta(hours=iM))
                        lonInt.append((1.-iM*inv)*lon[iN]+iM*inv*lon[iN+1])
                        latInt.append((1.-iM*inv)*lat[iN]+iM*inv*lat[iN+1])


        speed = np.zeros(lon.shape[0],dtype=float)+float('nan')
        sdir = np.zeros(lon.shape[0],dtype=float)+float('nan')
        count = 0
        for it in time:
                nup = argminDatetime(it+timedelta(hours=3),timeInt)
                ndn = argminDatetime(it-timedelta(hours=3),timeInt)
                londis = 2*np.pi*er*np.cos(latInt[nup]/180*np.pi)/360
                dlon = lonInt[nup]-lonInt[ndn]
                dlat = latInt[nup]-latInt[ndn]
                dx = londis*(lonInt[nup]-lonInt[ndn])
                dy = 110*(latInt[nup]-latInt[ndn])
                distance = np.sqrt(dx*dx+dy*dy) #km
                sdir[count]=np.arctan2(dlat,dlon)
                speed[count]=distance*1000./(nup-ndn+1)/60/60 #m/s
                count+= 1

        return sdir,speed


def getSpeedDir(iiS,block_id=None):
    iS = np.int_(iiS.mean(keepdims=True)[0])
    if (bt.StormLat[:,iS]==bt.StormLat[:,iS]).any():
        it1 = np.argwhere(bt.StormLat[:,iS]==bt.StormLat[:,iS])[0,0]
        it2 = np.argwhere(bt.StormLat[:,iS]==bt.StormLat[:,iS])[-1,0]
        if it2 - it1 >=2:
           bt.trDir[it1:it2,iS],bt.trSpeed[it1:it2,iS]=\
             getStormTranslation(bt.StormLon[it1:it2,iS],\
             bt.StormLat[it1:it2,iS],bt.Time[it1:it2,iS])
    return iS

def get_seeding_ratio(fpath,iy1,iy2):
    '''
    This function finds the seeding ratio.
    fpath:(str) location of historical simulations/predictions 
    iy1: (int) first year
    iy2: (int) second year
    ratio: (int) seeding ratio
    '''
    f = gv.opath
    bt_obs = nc.Dataset(f,'r')
    NS = np.argwhere((bt_obs['StormYear'][:]>=iy1)&(bt_obs['StormYear'][:]<=iy2))[:,0].shape[0]
    totalNS = np.float64(NS)/gv.survivalrate
    ratio = 1.0
    for iratio in range(0,1): ## I mad mistake to make it 2 for most of CMIP5 runs, and need to fix that
            ## seeding storms ####
            time1 = time.time()
            climInitLon = []
            climInitLat = []
            climInitDate = []
            for iy in range(iy1,iy2+1,1):
                        tcgiFile = fpath+'/'+gv.TCGIinput+'_'+int2str(iy,4)+'.mat'
                        gi = np.rollaxis(loadmat(tcgiFile)['TCGI'],2,0)
                        #print iy, np.nansum(gi)
                        if iy == iy1:
                                    xlon =loadmat(tcgiFile)['lon']
                                    xlat =loadmat(tcgiFile)['lat']
                                    gxlon,gxlat = np.meshgrid(xlon,xlat)
                        climInitLon,climInitLat,climInitDate = \
                            TCgiSeeding(gxlon,gxlat,gi,climInitLon,climInitLat,climInitDate,ratio,iy)
    NS = len(climInitLon)
    ratio = np.float64(totalNS)/np.float64(NS)
    return ratio


def getSeeding(fpath, iy,ratio):
    climInitLon = []
    climInitLat = []
    climInitDate = []
    tcgiFile = fpath+gv.TCGIinput+'_'+int2str(iy,4)+'.mat'
    #tcgiFile = './'+gv.TCGIinput+'_'+int2str(iy,4)+'.mat'

    gi = np.rollaxis(loadmat(tcgiFile)['TCGI'],2,0)
    xlon =loadmat(tcgiFile)['lon']
    xlat =loadmat(tcgiFile)['lat']
    gxlon,gxlat = np.meshgrid(xlon,xlat)
    climInitLon,climInitLat,climInitDate = \
            TCgiSeeding (gxlon,gxlat,gi,climInitLon,climInitLat,climInitDate,ratio,iy)
    climInitDate = np.array(climInitDate).ravel()
    climInitLon = np.array(climInitLon).ravel()
    climInitLat = np.array(climInitLat).ravel()
    return climInitDate,climInitLon,climInitLat


def getBam(cDate,cLon,cLat,iy,ichaz):
        global climInitDate,climInitLon,climInitLat
        climInitDate,climInitLon,climInitLat = cDate,cLon,cLat
        nnt = np.int_(31) # longest track time
        nS = climInitLon.shape[0]
        global fstlon, fstlat, fstldmask, fstu250, fstv250, fstu850, fstv850
        fstlon = np.zeros([nnt*24+1,nS])*np.nan
        fstlat = np.zeros([nnt*24+1,nS])*np.nan
        fstldmask = np.zeros(fstlat.shape)
        fstu250 = np.zeros([nnt*24+1,nS])*np.nan
        fstv250 = np.zeros([nnt*24+1,nS])*np.nan
        fstu850 = np.zeros([nnt*24+1,nS])*np.nan
        fstv850 = np.zeros([nnt*24+1,nS])*np.nan
        diS = da.from_array(np.arange(0,nS,1).astype(dtype=np.int32),chunks=(1,))
        niS = np.arange(0,nS,1).astype(dtype=np.int32)
        new = da.map_blocks(bam,diS,chunks=(1,), dtype=diS.dtype)
        with ProgressBar():
                b = new.compute(scheduler='synchronous',num_workers=5)
                #     for iS in niS:
                #           b = bam(np.array([iS,iS])) 
        print('removeland')
        fstlon = fstlon[::6,:]
        fstlat = fstlat[::6,:]
        fstldmask = fstldmask[::6,:]
        fstu250 = fstu250[::6,:]
        fstv250 = fstv250[::6,:]
        fstu850 = fstu850[::6,:]
        fstv850 = fstv850[::6,:]
        for iS in niS:
                c = removeland(np.array([iS,iS]))
        #new = da.map_blocks(removeland,diS,chunks=(1,), dtype=diS.dtype)
        #with ProgressBar():
        #     c = new.compute(scheduler='synchronous',num_workers=5)
        print('give times')
        ### give times
        fsttime = np.empty(fstlon.shape,dtype=object)
        fsttime[0,:] = climInitDate
        dummy = pd.DataFrame(fstlon)
        #import pdb;pdb.set_trace()
        iT1 = np.int16(dummy.apply(func_first,axis=0))+1
        iT2 = np.int16(dummy.apply(func_last,axis=0))
        for iS in niS:
                fsttime[iT1[iS]:iT2[iS]+1,iS] = \
                [climInitDate[iS]+timedelta(hours=6*iit) for iit in range(iT1[iS],iT2[iS]+1,1)]
                fst = {'lon':fstlon,'lat':fstlat,'Time':fsttime,'ldmask':fstldmask,'u250md':fstu250,'v250md':fstv250,'u850md':fstu850,'v850md':fstv850}
        with open('track_'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','wb+')as f:
                pickle.dump(fst,f)
        f.close()
        return fst

def calPredictors(fst,iy,ichaz):
        global bt
        time1 = time.time()
        bt = fst2bt(fst)
        nS = fst['lon'].shape[1]
        #diS = da.from_array(np.int32(np.arange(0,nS,1)),chunks=(1,))
        #niS = np.int32(np.arange(0,nS,1))
        diS = da.from_array(np.arange(0,nS,1).astype(dtype=np.int32),chunks=(1,))
        niS = np.arange(0,nS,1).astype(dtype=np.int32)
        new = da.map_blocks(getSpeedDir,diS,chunks=(1,), dtype=diS.dtype)
        with ProgressBar():
                new.compute(scheduler='synchronous',num_workers=5)
        print('done translation speed', time.time()-time1)
        del new
        gc.collect()

        time1 = time.time()
        new = da.map_blocks(get_predictors,diS,chunks=(1,), dtype=diS.dtype)
        with ProgressBar():
                a = new.compute(scheduler='synchronous',num_workers=5)
        print(' done calPredictors', time.time()-time1)
        gc.collect()
        with open(gv.output_path+'trackPredictorsbt'+int2str(iy,4)+'_ens'+int2str(ichaz,3)+'.pik','wb+')as f:
                pickle.dump(bt,f)
        f.close()
        return



                                                                     
import numpy as np
import datetime as dt
from swarmx.toolboxes.secs import SecsInputs
import swarmx.toolboxes.secs.aux_tools as auto
from swarmx.toolboxes.secs.aux_tools import sub_Swarm_grids,sph2sph,sub_FindLongestNonZero
from sub_fit_1D_DivFree import SwarmMag2J_test_fit_1D_DivFree
import xarray as xr




def get_data_slices(t1=dt.datetime(2016, 3, 18, 11, 3, 0),t2=dt.datetime(2016, 3, 18, 11, 40, 0),model='IGRF'):
    """_summary_

    Parameters
    ----------
    t1 : _type_, optional
        _description_, by default dt.datetime(2016, 3, 18, 11, 3, 0)
    t2 : _type_, optional
        _description_, by default dt.datetime(2016, 3, 18, 11, 40, 0)
    model : str, optional
        _description_, by default 'IGRF'

    Returns
    -------
    _type_
        _description_
    """

    inputs = SecsInputs(
    start_time=t1,
    end_time=t2,
    model="IGRF")

    SwA = auto.get_eq(inputs.s1.xarray)

    SwC = auto.get_eq(inputs.s2.xarray)

    return SwA, SwC

def analyze(dataA,dataC):


    return





def sub_load_real_data(result):

    #Radius of the Earth and radius of the ionospheric current layer
    result["Re"] = 6371                     #default 6371 km
    result["Ri"] = result["Re"] + 130       #default Re+130
    ###Not needed because we already have apex coordinates
    #FPalt = result["Ri"] - result["Re"]

    #Select the latitude range abs(lat) <= limit where the satellite data is taken.
    limitSwarmLat = 60

    #Get Swarm A and C data
    inputs = SecsInputs(
    start_time=dt.datetime(2016, 3, 18, 11, 3, 0),
    end_time=dt.datetime(2016, 3, 18, 11, 40, 0),
    model="IGRF",)

    SwA = inputs.s1.xarray
    SwA = SwA.where(abs(SwA["Latitude"]) <= limitSwarmLat, drop=True)

    SwC = inputs.s2.xarray
    SwC = SwC.where(abs(SwA["Latitude"]) <= limitSwarmLat, drop=True)

    #Define the output grid.
    DlatOut = 0.5       #resolution in latitude
    LonRatioOut = 2     #ratio (satellite separation)/(grid resolution) in longitude
    ExtLatOut = 0       #Number of points to extend outside satellite data area in latitude
    ExtLonOut = 3       #Number of points to extend outside satellite data area in longitude
    
    #Make grid around [-X,X] latitudes
    limitOutputLat = 40
    ind = np.nonzero(abs(SwA["Latitude"].data) <= limitOutputLat)
    lat1 = SwA["Latitude"].data[ind]
    lon1 = SwA["Longitude"].data[ind]
    ind = np.nonzero(abs(SwC["Latitude"].data) <= limitOutputLat)
    lat2 = SwC["Latitude"].data[ind]
    lon2 = SwC["Longitude"].data[ind]

    ###result should also be xarray
    ###check with Heikki, sub_Swarm_grids from sub_Swarm_grids_2D.m returns more than 2 things
    result["ggLat"],result["ggLon"],_,_,_ = sub_Swarm_grids(lat1,lon1,lat2,lon2,DlatOut,LonRatioOut,ExtLatOut,ExtLonOut)
    #Transpose into [Nlon,Nlat] matrices
    ### need to add this to dataarray somehow
    result["ggLat"] = np.transpose(result["ggLat"])
    result["ggLon"] = np.transpose(result["ggLon"])

    return result, SwA, SwC


def sub_FindPole(SwA,result):

    #define search grid
    dlat = 0.5      #latitude step [degree]
    minlat=60       #latitude range
    maxlat = 90
    dlon = 5        #longitude step [degree]
    minlon = 0      #longitude range
    maxlon = 360 - 0.1 * dlon
    #create [Nlon,Nlat] matrices
    latP,lonP = np.mgrid[minlat:maxlat + dlat:dlat, minlon:maxlon + dlon:dlon]
    latP = latP.flatten()
    lonP = lonP.flatten()

    #format error matrix
    errMat = np.full_like(latP, np.nan)

    #Could Limit the latitude range of the Swarm-A measurement points used in the optimization
    indA = np.nonzero(abs(SwA["Latitude"].data) < 100)
    
    #Loop over possible pole locations
    for n in range(len(latP)):
    #Rotate the main field unit vector at Swarm-A measurement points to the system
    #whose pole is at (latP(n), lonP(n)).
        lat,_,Bt,Bp = sph2sph(latP[n],lonP[n],SwA["Latitude"].data[indA],SwA["Longitude"].data[indA],SwA["ggUvT"].data[indA],SwA["ggUvP"].data[indA])
        Br = SwA["UvR"]

        #Remove points that are very close to this pole location (otherwise 1/sin is problematic)
        ind = np.nonzero(abs(lat) < 89.5)
        lat = lat[ind]
        Bt = Bt[ind]    #theta=south
        Bp = Bp[ind]    #east
        Br = Br[ind]    #radial

        #Calculate unit vector in dipole system centered at (latP(n), lonP(n)).
        tmp = (1 + 3 * np.sin(np.radians(lat))**2)
        BxD = np.cos(np.radians(lat)) / tmp     #north
        ByD = np.zeros_like(lat)                #east
        BzD = 2 * np.sin(np.radians(lat)) / tmp #down

        #Difference between the measured unit vectors and dipole unit vectors,
        #averaged over all points 
        errMat[n] = np.nanmean((Bt + BxD)**2 + (Bp - ByD)**2 + (Br + BzD)**2)

    #Find pole location with minimum error
    ind = np.argmin(errMat)
    result["PoleLat"] = latP[ind]
    result["PoleLon"] = lonP[ind]
    
    ###skipped plotting routine here (see sub_FindPole.m), also flattened latP,lonP and errMat

    return result


def sub_rotate(result,model,SwA,SwC,suunta):

    latP = result["PoleLat"]
    lonP = result["PoleLon"]

    if suunta == 'geo2mag':
        #Rotate input coordinates and horizontal part of input vectors to the local dipole system.
        result["magLat"],result["magLon"],_,_ = sph2sph(latP,lonP,result["ggLat"],result["ggLon"],[],[])  #output grid
        result["magLon"] = result["magLon"] % 360
        _,_,SwA["magUvT"],SwA["magUvP"] = sph2sph(latP,lonP,SwA["Latitude"].data,SwA["Longitude"].data,SwA["ggUvT"].data,SwA["ggUvP"].data)  #unit vector along SwA magnetic field
        _,_,SwC["magUvT"],SwC["magUvP"] = sph2sph(latP,lonP,SwC["Latitude"].data,SwC["Longitude"].data,SwC["ggUvT"].data,SwC["ggUvP"].data)  #unit vector along SwC magnetic field
        SwA["magLat"],SwA["magLon"],SwA["magBt"],SwA["magBp"] = sph2sph(latP,lonP,SwA["Latitude"].data,SwA["Longitude"].data,SwA["B_NEC"].sel(NEC='N').data,SwA["B_NEC"].sel(NEC='E').data)  #SwA locations & data
        SwC["magLat"],SwC["magLon"],SwC["magBt"],SwC["magBp"] = sph2sph(latP,lonP,SwC["Latitude"].data,SwC["Longitude"].data,SwC["B_NEC"].sel(NEC='N').data,SwC["B_NEC"].sel(NEC='E').data)  #SwC locations & data
        SwA["magLon"] = SwA["magLon"] % 360
        SwC["magLon"] = SwC["magLon"] % 360

        #Our analysis requires that satellites move mostly in the north/south direction (problems in grid generation etc),
        #but that may not be the case in the magnetic coordinate system. Solve this by selecting latitude range where satellite's
        #eastward velocity is sufficiently small.
        #First Swarm-A
        Vx = np.gradient(SwA["magLat"])     #northward velocity [deg/step]
        Vy = np.gradient(SwA["magLon"]) * np.cos(np.radians(SwA["magLat"]))  #eastward velocity [deg/step]
        _,ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))
        #SwA["apexLat"] = SwA["apexLat"][ind]
        #SwA.r=SwA.r(ind);      SwA.ggLat=SwA.ggLat(ind);  SwA.ggLon=SwA.ggLon(ind);  SwA.dn=SwA.dn(ind);
        #SwA.Br=SwA.Br(ind);    SwA.ggBt=SwA.ggBt(ind);    SwA.ggBp=SwA.ggBp(ind);    SwA.Bpara=SwA.Bpara(ind);
        #SwA.UvR=SwA.UvR(ind);  SwA.ggUvT=SwA.ggUvT(ind);  SwA.ggUvP=SwA.ggUvP(ind);
        #SwA.magLat=SwA.magLat(ind);      SwA.magLon=SwA.magLon(ind);
        #SwA.magBt=SwA.magBt(ind);        SwA.magBp=SwA.magBp(ind);
        #SwA.magUvT=SwA.magUvT(ind);      SwA.magUvP=SwA.magUvP(ind);
        ### replace with indexing whole dataset
        SwA = SwA.sel(Timestamp=SwA["Timestamp"][ind]) 

        #Then Swarm-C
        Vx = np.gradient(SwC["magLat"])
        Vy = np.gradient(SwC["magLon"]) * np.cos(np.radians(SwC["magLat"]))
        _,ind = sub_FindLongestNonZero(abs(Vy) < abs(Vx))
        #SwC.apexLat=SwC.apexLat(ind);
        #SwC.r=SwC.r(ind);      SwC.ggLat=SwC.ggLat(ind);  SwC.ggLon=SwC.ggLon(ind);  SwC.dn=SwC.dn(ind);
        #SwC.Br=SwC.Br(ind);    SwC.ggBt=SwC.ggBt(ind);    SwC.ggBp=SwC.ggBp(ind);    SwC.Bpara=SwC.Bpara(ind);
        #SwC.UvR=SwC.UvR(ind);  SwC.ggUvT=SwC.ggUvT(ind);  SwC.ggUvP=SwC.ggUvP(ind);
        #SwC.magLat=SwC.magLat(ind);      SwC.magLon=SwC.magLon(ind);
        #SwC.magBt=SwC.magBt(ind);        SwC.magBp=SwC.magBp(ind);
        #SwC.magUvT=SwC.magUvT(ind);      SwC.magUvP=SwC.magUvP(ind);
        ### replace with indexing whole dataset
        SwC = SwC.sel(Timestamp=SwC["Timestamp"][ind]) 

    elif suunta == 'mag2geo':
        #Rotate analysis results back to geographic.
        lat = result["magLat"].data
        lon = result["magLon"].data
        _,_,result["df1dGgJt"],result["df1dGgJp"] = sph2sph(latP,0,lat,lon,result["df1dMagJt"].data,result["df1dMagJp"].data) #J from 1D DF SECS
        _,_,result["df2dGgJt"],result["df2dGgJp"] = sph2sph(latP,0,lat,lon,result["df2dMagJt"].data,result["df2dMagJp"].data) #J from 2D DF SECS
        _,_,result["cf1dDipGgJt"],result["cf1dDipGgJp"] = sph2sph(latP,0,lat,lon,result["cf1dDipMagJt"].data,result["cf1dDipMagJp"].data) #J from dipolar 1D CF SECS
        _,_,result["cf2dDipGgJt"],result["cf2dDipGgJp"] = sph2sph(latP,0,lat,lon,result["cf2dDipMagJt"].data,result["cf2dDipMagJp"].data) #J from dipolar 2D CF SECS
        _,_,result["remoteCf2dDipGgJt"],result["remoteCf2dDipGgJp"] = sph2sph(latP,0,lat,lon,result["remoteCf2dDipMagJt"].data,result["remoteCf2dDipMagJp"].data) #J from remote dipolar 2D CF SECS
        
        lat = SwA["magLat"].data
        lon = SwA["magLon"].data
        _,_,SwA["df1dGgBt"],SwA["df1dGgBp"] = sph2sph(latP,0,lat,lon,SwA["df1dMagBt"].data,SwA["df1dMagBp"].data) #B at SwA from 1D DF SECS
        _,_,SwA["df2dGgBt"],SwA["df2dGgBp"] = sph2sph(latP,0,lat,lon,SwA["df2dMagBt"].data,SwA["df2dMagBp"].data) #B at SwA from 2D DF SECS
        _,_,SwA["cf1dDipGgBt"],SwA["cf1dDipGgBp"] = sph2sph(latP,0,lat,lon,SwA["cf1dDipMagBt"].data,SwA["cf1dDipMagBp"].data) #B at SwA from dipolar 1D CF SECS
        _,_,SwA["cf2dDipGgBt"],SwA["cf2dDipGgBp"] = sph2sph(latP,0,lat,lon,SwA["cf2dDipMagBt"].data,SwA["cf2dDipMagBp"].data) #B at SwA from dipolar 2D CF SECS
        _,_,SwA["remoteCf2dDipGgBt"],SwA["remoteCf2dDipGgBp"] = sph2sph(latP,0,lat,lon,SwA["remoteCf2dDipMagBt"].data,SwA["remoteCf2dDipMagBp"].data) #B at SwA from remote dipolar 2D CF SECS
        
        lat = SwC["magLat"].data
        lon = SwC["magLon"].data
        _,_,SwC["df1dGgBt"],SwC["df1dGgBp"] = sph2sph(latP,0,lat,lon,SwC["df1dMagBt"].data,SwC["df1dMagBp"].data)
        _,_,SwC["df2dGgBt"],SwC["df2dGgBp"] = sph2sph(latP,0,lat,lon,SwC["df2dMagBt"].data,SwC["df2dMagBp"].data)
        _,_,SwC["cf1dDipGgBt"],SwC["cf1dDipGgBp"] = sph2sph(latP,0,lat,lon,SwC["cf1dDipMagBt"].data,SwC["cf1dDipMagBp"].data)
        _,_,SwC["cf2dDipGgBt"],SwC["cf2dDipGgBp"] = sph2sph(latP,0,lat,lon,SwC["cf2dDipMagBt"].data,SwC["cf2dDipMagBp"].data)
        _,_,SwC["remoteCf2dDipGgBt"],SwC["remoteCf2dDipGgBp"] = sph2sph(latP,0,lat,lon,SwC["remoteCf2dDipMagBt"].data,SwC["remoteCf2dDipMagBp"].data)
    
    else:
        raise ValueError('Direction must be either geo2mag or mag2geo.')
    
    return result, model, SwA, SwC


def Swarm_B2J_real_analyze():

    #result = xr.DataArray()
    result = {}
    #result["Event"] = eventti

    ###get input from SecsInputs
    result, SwA, SwC = sub_load_real_data(result)

    #Geographical components of unit vector in the main field direction
    ### from sub_ReadMagDataLR, check again model and signs
    tmp = np.sqrt(SwA["B_NEC_Model"].sel(NEC='N')**2 + SwA["B_NEC_Model"].sel(NEC='E')**2 + SwA["B_NEC_Model"].sel(NEC='C')**2)
    SwA["ggUvT"] = -SwA["B_NEC_Model"].sel(NEC='N') / tmp  #south
    SwA["ggUvP"] = SwA["B_NEC_Model"].sel(NEC='E') / tmp   #east
    SwA["UvR"] = -SwA["B_NEC_Model"].sel(NEC='C') / tmp    #radial

    tmp = np.sqrt(SwC["B_NEC_Model"].sel(NEC='N')**2 + SwC["B_NEC_Model"].sel(NEC='E')**2 + SwC["B_NEC_Model"].sel(NEC='C')**2)
    SwC["ggUvT"] = -SwC["B_NEC_Model"].sel(NEC='N') / tmp  #south
    SwC["ggUvP"] = SwC["B_NEC_Model"].sel(NEC='E') / tmp   #east
    SwC["UvR"] = -SwC["B_NEC_Model"].sel(NEC='C') / tmp    #radial

    #Calculate field-aligned magnetic field disturbances by taking the dot product between the measured
    #magnetic disturnabce and the unit vector along the main field. It is needed in fitting the DF SECS.
    ### check that NEC are right here
    SwA["Bpara"] = SwA["UvR"] * SwA["B_NEC"].sel(NEC='C') + SwA["ggUvT"] * SwA["B_NEC"].sel(NEC='N') + SwA["ggUvP"] * SwA["B_NEC"].sel(NEC='E')
    SwC["Bpara"] = SwC["UvR"] * SwC["B_NEC"].sel(NEC='C') + SwC["ggUvT"] * SwC["B_NEC"].sel(NEC='N') + SwC["ggUvP"] * SwC["B_NEC"].sel(NEC='E')
    #SwA["Bpara"] = SwA["UvR"] * SwA["Br"] + SwA["ggUvT"] * SwA["ggBt"] + SwA["ggUvP"] * SwA["ggBp"]
    #SwC["Bpara"] = SwC["UvR"] * SwC["Br"] + SwC["ggUvT"] * SwC["ggBt"] + SwC["ggUvP"] * SwC["ggBp"]

    #Analysis is done in a local dipole coordinate system, where the magnetic field
    #most closely resembles dipole field.
    #Find optimum location for the local dipole's north pole.
    result = sub_FindPole(SwA,result)
    #Rotate input variables (coordinates & vectors) to the local dipole system
    result,_,SwA,SwC = sub_rotate(result,[],SwA,SwC,'geo2mag')

    #Fit 1D DF SECS
    SwA,SwC,result = SwarmMag2J_test_fit_1D_DivFree(SwA,SwC,result)

#Swarm_B2J_real_analyze()
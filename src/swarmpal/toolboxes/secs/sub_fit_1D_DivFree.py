import numpy as np
from swarmpal.toolboxes.secs.dsecs_algorithms import SECS_1D_DivFree_magnetic, SECS_1D_DivFree_vector
from swarmpal.toolboxes.secs.aux_tools import sub_Swarm_grids_1D, sub_inversion

def SwarmMag2J_test_fit_1D_DivFree(SwA,SwC,result):

    #extract radii of the Earth and ionospheric current layer
    #Re = result["Re"] ###not used?
    Ri = result["Ri"]

    ###check dimensions, row or column vector??
    #Combine data from the two satellites.
    latB = np.hstack((SwA["magLat"].data.flatten(), SwC["magLat"].data.flatten()))
    rB = np.hstack((SwA["Radius"].data.flatten(), SwC["Radius"].data.flatten()))

    #Fit is done to the parallel magnetic disturbance.
    uvR = np.hstack((SwA["UvR"].data.flatten(), SwC["UvR"].data.flatten()))[:,np.newaxis]
    uvT = np.hstack((SwA["magUvT"].data.flatten(), SwC["magUvT"].data.flatten()))[:,np.newaxis]
    Bpara = np.hstack((SwA["Bpara"].data.flatten(), SwC["Bpara"].data.flatten()))

    #Calculate 1D SECS grid in a separate sub-function
    Dlat1D = 0.5    #latitude spacing [degree]
    ExtLat1D = 5    #Number of points to extend outside the data range
    lat1D, mat1Dsecond = sub_Swarm_grids_1D(SwA["magLat"].data,SwC["magLat"].data,Dlat1D,ExtLat1D)
    lat1D = lat1D.flatten()
    N1d = len(lat1D)

    #Calculate B-matrices that give magnetic field from SECS amplitudes and form the field-aligned matrix.
    matBr, matBt = SECS_1D_DivFree_magnetic(latB,lat1D,rB,Ri,500)
    matBpara = np.tile(uvR,(1,N1d)) * matBr + np.tile(uvT,(1,N1d)) * matBt

    #Fit the 1D DF SECS to the field-aligned magnetic disturbance.
    #Use zero constraint on the 2nd latitudinal derivative.
    alpha = 1e-5
    epsSVD = 4e-4
    Idf = sub_inversion(matBpara, mat1Dsecond, epsSVD, alpha, Bpara)  #scaling factors of the 1D DF SECS

    #Calculate the eastward current produced by the 1D DF SECS.
    latJ = result["magLat"].flatten()
    matJp = SECS_1D_DivFree_vector(latJ,lat1D,Ri)
    ###add to result xarray
    result["df1dMagJp"]= np.reshape(matJp @ Idf, np.shape(result["magLat"]),order='F')
    result["df1dMagJt"] = np.zeros(np.shape(result["magLat"]))

    #Calculate the magnetic field produced by the 1D DF SECS.
    Na = np.size(SwA["magLat"])
    sizeA = np.shape(SwA["magLat"])
    sizeC = np.shape(SwC["magLat"])
    apu = matBr @ Idf
    ###add to SwA and SwC xarray?
    SwA["df1dBr"] = np.reshape(apu[:Na], sizeA, order='F')
    SwC["df1dBr"] = np.reshape(apu[Na:], sizeC, order='F')
    apu = matBt @ Idf
    SwA["df1dMagBt"] = np.reshape(apu[:Na], sizeA, order='F')
    SwC["df1dMagBt"] = np.reshape(apu[Na:], sizeC, order='F')
    SwA["df1dMagBp"] = np.zeros(sizeA)
    SwC["df1dMagBp"] = np.zeros(sizeC)
    apu = matBpara @ Idf
    SwA["df1dBpara"] = np.reshape(apu[:Na], sizeA, order='F')
    SwC["df1dBpara"] = np.reshape(apu[Na:], sizeC, order='F')

    return SwA,SwC,result
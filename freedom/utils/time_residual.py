import numpy as np
from scipy import constants, stats

def TimeResidual(hits, params, Index=1.33):
    hitpos = hits[:,:3]
    T_exp = CherenkovTime(params, hitpos, Index)
    T_meas = hits[:,3] - params[:,3]
    return T_meas - T_exp

def CherenkovTime(params, position, Index=1.33):
    changle = np.arccos(1/Index)
    length = np.clip(params[:, 7], 0.1, 1e4) * 5 #m
    Length = np.stack([length, length, length], axis=1)
    
    # closest point on (inf) track, dist and dist along track
    appos, apdist, s, Dir = ClosestApproachCalc(params, position)
    a = s - apdist/np.tan(changle)
    
    return np.where(a <= 0.,
                np.linalg.norm(position-params[:,:3], axis=1) * Index/constants.c * 1e9,
                np.where(a <= length,
                         (a + apdist/np.sin(changle)*Index) / constants.c * 1e9,
                         (length + np.linalg.norm(position-(params[:,:3] + Length*Dir), axis=1)*Index) / constants.c * 1e9
                        )
               )
    
def ClosestApproachCalc(params, position): 
    theta  = params[:,5]
    phi    = params[:,4]
    pos0_x = params[:,0]
    pos0_y = params[:,1]
    pos0_z = params[:,2]

    e_x = -np.sin(theta)*np.cos(phi)
    e_y = -np.sin(theta)*np.sin(phi)
    e_z = -np.cos(theta)
    
    h_x = position[:,0] - pos0_x
    h_y = position[:,1] - pos0_y
    h_z = position[:,2] - pos0_z
    
    s = e_x*h_x + e_y*h_y + e_z*h_z
    
    pos2_x = pos0_x + s*e_x
    pos2_y = pos0_y + s*e_y
    pos2_z = pos0_z + s*e_z
    
    appos = np.stack([pos2_x, pos2_y, pos2_z], axis=1)
    apdist = np.linalg.norm(position-appos, axis=1)
    Dir = np.stack([e_x, e_y, e_z], axis=1)
    
    return appos, apdist, s, Dir

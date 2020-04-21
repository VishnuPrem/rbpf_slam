# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:29:24 2020

@author: ravit
"""


import numpy as np
import utils 

def Scan_matcher(prev_scan, prev_pose, curr_scan, curr_best_pose,thresh = 0.5):
    
    """
    prev_scan: Scan at time "t-1". Shape: (n,2)
    prev_pose: Pose of the current particle at previous time step: (3,)
    prev_scan: Scan at time "t". Shape: (n,2)
    prev_pose: Current estimate of the Pose of the particle at time t: (3,)
    thresh: error threshold
    
    ---------
     Returns:
         Flag: Indicates if the scan matching is successful or not
         pos: Best pose of the particle after scan matching
    
    """
    
    Flag = False
    pos = np.zeros(3)
    d_pose = curr_best_pose - prev_pose 
    iters = 0
    
    trans_scan = utils.transformation_scans(prev_scan,d_pose)
    
    Correspondance = get_correspondance(curr_scan,trans_scan)
    
    curr_error = cal_error(curr_scan,trans_scan,Correspondance)
    
    prev_error = 1e8
    
    while (curr_error < prev_error or iters < 100):
        
        prev_error = curr_error
        
        d_pose += get_estimate(curr_scan,trans_scan,Correspondance)
        
        trans_scan = utils.transform(trans_scan,d_pose)
        
        Correspondance = get_correspondance(curr_scan,trans_scan)
        
        curr_error = cal_error(curr_scan,trans_scan,Correspondance)
        
        iters += 1
        
    if curr_error < thresh:
        Flag = True
        
    return Flag,pos


def get_correspondance(curr_scan, trans_scan):
    
    """
    curr_scan:  Lidar scan at current time step (n1,2)
    
    trans_scan: Transformed previous scan based on best known pose (n2,2)

    Returns
    -------
    correspondance: Indices of the closest lidar point in curr_scan to trans_scan (n2,)

    """
    x_curr = curr_scan[:,0]
    y_curr = curr_scan[:,1]
    x_trans = trans_scan[:,0][:,None]
    y_trans = trans_scan[:,1][:,None]
    
    dist = np.square(x_curr - x_trans) + np.square(y_curr - y_trans)
    
    correspondance = np.argmin(dist,axis = 1)
    
    assert correspondance.shape == x_trans.shape[0]
        
    return correspondance


def cal_error(curr_scan,trans_scan,Correspondance):
    
    """
    curr_scan:  Lidar scan at current time step (n,2)
    
    trans_scan: Transformed previous scan based on best known pose (n,2)

    correspondance: Indices of the closest lidar point in curr_scan to trans_scan (n,)
    

    Returns
    -------
    Error: L2 dist between corresponding points

    """
    
    error = np.linalg.norm(curr_scan[Correspondance,:] - trans_scan, axis = 1)
    
    error = np.mean(error)
    
    
    
    return error
 
def get_estimate(curr_scan,trans_scan,correspondance):
    
    """
    curr_scan:  Lidar scan at current time step (n,2)
    
    trans_scan: Transformed previous scan based on best known pose (n,2)
    

    Returns
    -------
    d_pose: difference in pose between two scans

    """
    d_pose = np.zeros(3) 
    
    curr_mean = np.mean(curr_scan,axis = 0)
    trans_mean = np.mean(trans_scan,axis = 0)
    
    curr_scan -= curr_mean
    trans_scan -= trans_mean
    
    corres_scan = curr_scan[correspondance,:]
    
    assert corres_scan.shape == trans_scan.shape
    
    
    W = np.dot(curr_scan.T,trans_scan.T)
    u,s,vt = np.linalg.svd(W)
    
    R = np.dot(u,vt)
    d_pose[:2] = curr_mean - np.dot(R,trans_mean)
    d_pose[2] = np.atan2(R[1,0],R[0,0])

      
    return d_pose
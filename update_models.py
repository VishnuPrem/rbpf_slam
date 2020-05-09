# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:53:36 2020

@author: ravit
"""


import numpy as np
import utils as utils
import scan_matching as match
from math import sqrt
from scipy.stats import norm


def measurement_model(scan, pose, lidar_angles, occupied_indices):
    """
    

    Parameters
    ----------
    scan : (1080,) distance array
    pose : (3,) numpy array
    lidar_angles : (1080,) numpy array
    occupied_indices : (n,2) numpy array

    Returns
    -------
    prob : probability

    """
    p_hit =  0.99 ###define
    p_random = 1 - p_hit
    z_max = 10 
    sigma = 0.5
    
    pose = pose.reshape((3,1))
    
    #pose_angles = lidar_angles + pose[2,1]
    
    xy = utils.dist_to_xy(scan,scan_angles)
    xy = utils.transformation_scans(xy,pose)
    
    _, min_dist = match.get_correspondance(occupied_indices,xy)
    
    exp = (p_hit/(sqrt(2*np.pi)*sigma))*np.exp(-d/(2*sigma**2))
    
    prob = exp + p_random*(1/z_max)
    
    prob = np.sum(np.log(prob))
    
    
    return np.exp(prob)  ###confirm over usage of exp


def odometry_model(prev,curr,curr_odom,prev_odom):
    """
    

    Parameters
    ----------
    prev : (3,) numpy array
        prev time steps position
    curr :  (3,) np array
        curr time steps position
    odom_curr : (3,)
        DESCRIPTION.
    odom_prev : (3,)
        DESCRIPTION.

    Returns
    -------
    prob : TYPE
        DESCRIPTION.

    """
    
    
    alpha = [ , , , ]
    
    delta_rot1 = np.arctan2(curr_odom[1] - prev_odom[1], curr_odom[0] - curr_odom[1]) - odom_prev[2]
    delta_trans = np.linal.norm(odom_curr[:2]-odom_prev[:2])
    delta_rot2 = curr_odom[2] - prev_odom[2] - delta_rot1
    
    delta_hat_rot1 = np.arctan2(curr[1] - prev[1], curr[0] - curr[1]) - prev[2]  
    delta_hat_trans = np.linal.norm(curr[:2]-prev[:2])
    delta_hat_rot2 = curr[2] - prev[2] - delta_hat_rot1
    
    p1 = stats.pdf(delta_rot1 - delta_hat_rot1, mean = alpha[0]*delta_hat_rot1 + alpha[1]*delta_hat_trans)
    p2 = stats.pdf(delta_trans - delta_hat_trans, mean = alpha[2]*delta_hat_trans + alpha[3]*(delta_hat_rot1 + delta_hat_rot2))
    p3 = stats.pdf(delta_rot2 - delta_hat_rot2, mean = alpha[0]*delta_hat_rot2 + alpha[1]*delta_hat_trans)
        
    
    
    return p1*p2*p3
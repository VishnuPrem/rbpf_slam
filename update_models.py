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
    obstacle = scan<10 
    xy = utils.dist_to_xy(scan[obstacle], lidar_angles[obstacle])
    xy = utils.transformation_scans(xy,pose)
    
    _, min_dist = match.get_correspondance(occupied_indices,xy)
    print('Min dist: ', max(min_dist))
    exp = (p_hit/(sqrt(2*np.pi)*sigma))*np.exp(-min_dist/(2*sigma**2))
    
    prob = exp + p_random*(1/z_max)
    
    prob = np.sum(np.log(prob))
    
    print('Measurement: ', prob)
    return prob  ###confirm over usage of exp


def odometry_model(prev,curr, odom_curr, odom_prev):
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
    
    
    alpha = [0.25, 0.25, 0.25, 0.25]
    
    delta_rot1 = np.arctan2(odom_curr[1] - odom_prev[1], odom_curr[0] - odom_curr[1]) - odom_prev[2]
    delta_trans = np.linalg.norm(odom_curr[:2]-odom_prev[:2])
    delta_rot2 = odom_curr[2] - odom_prev[2] - delta_rot1
    
    delta_hat_rot1 = np.arctan2(curr[1] - prev[1], curr[0] - curr[1]) - prev[2]  
    delta_hat_trans = np.linalg.norm(curr[:2]-prev[:2])
    delta_hat_rot2 = curr[2] - prev[2] - delta_hat_rot1
    
    p1 = norm.pdf(delta_rot1 - delta_hat_rot1, loc = alpha[0]*delta_hat_rot1 + alpha[1]*delta_hat_trans)
    p2 = norm.pdf(delta_trans - delta_hat_trans, loc = alpha[2]*delta_hat_trans + alpha[3]*(delta_hat_rot1 + delta_hat_rot2))
    p3 = norm.pdf(delta_rot2 - delta_hat_rot2, loc = alpha[0]*delta_hat_rot2 + alpha[1]*delta_hat_trans)
    
    print('Odom model: ', p1*p2*p3)
    return p1*p2*p3
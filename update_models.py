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
    sigma = 0.38
    #sigma = 1/sqrt(2*np.pi)
    
    pose = pose.reshape((3,1))
    
    #pose_angles = lidar_angles + pose[2,1]
    obstacle = scan<10 
    max_dist = np.arange(10,0,-1)
        
    #print('here:',np.sum(scan <= 10))
    xy = utils.dist_to_xy(scan[obstacle], lidar_angles[obstacle])
    xy = utils.transformation_scans(xy,pose)
    #normalizer = np.zeros((xy.shape[0],1))
    '''for i in range(xy.shape[0]):
        for j in max_dist:
            if scan[i] < j:
                _,min_dist = match.get_correspondance(occupied_indices,xy[i][:,None].T)
                normalizer[i] += (1/(sqrt(2*np.pi)*sigma))*np.exp(-min_dist/(2*sigma**2))'''
    
    _, min_dist = match.get_correspondance(occupied_indices,xy)
    #print('Min dist: ', np.sum(min_dist))
    #print(normalizer)
    #print(np.where(normalizer == 0.0))
    #print('alse:',np.max(min_dist))
    normalizer = np.where(normalizer != 0, normalizer ,1)
    #exp = (p_hit/(sqrt(2*np.pi)*sigma))*np.exp(-min_dist/(2*sigma**2))
    exp = (p_hit)*np.exp(-min_dist/(2*sigma**2))
    #print(np.where(normalizer == 0))
    #print(exp[np.where(normalizer == 0.0)[0]])
    
    #exp = exp/normalizer[:,0]
    #print('updated exp:',exp)
    #print()
    ##print(np.sum(np.isnan(exp)))
    #print(fdasfag)
    #print(normalizer.shape,exp.shape)
    
    #print(exp)
    #prob = exp + p_random*(1/z_max)
   
    
    #print(prob)
    #print(prob.shape,np.sum(prob),0.991**prob.shape[0])
    
    #prob = np.sum(np.log(prob))
    
    #print('Measurement: ', prob, np.exp(prob))
    return np.mean(exp)  ###confirm over usage of exp prob,np.exp(prob),


def odometry_model(prev,curr, odom_prev, odom_curr):
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
    
    
    alpha = [0.13, 0.13, 0.13, 0.065]
    
    delta_rot1 = np.arctan2(odom_curr[1] - odom_prev[1], odom_curr[0] - odom_curr[1]) - odom_prev[2]
    delta_trans = np.linalg.norm(odom_curr[:2]-odom_prev[:2])
    delta_rot2 = odom_curr[2] - odom_prev[2] - delta_rot1
    
    delta_hat_rot1 = np.arctan2(curr[1] - prev[1], curr[0] - curr[1]) - prev[2]  
    delta_hat_trans = np.linalg.norm(curr[:2]-prev[:2])
    delta_hat_rot2 = curr[2] - prev[2] - delta_hat_rot1
    
    p1 = norm.pdf(delta_rot1 - delta_hat_rot1, loc = 0, scale = alpha[0]*np.abs(delta_hat_rot1) + alpha[1]*delta_hat_trans)
    p2 = norm.pdf(delta_trans - delta_hat_trans, loc = 0, scale = alpha[2]*delta_hat_trans + alpha[3]*(np.abs(delta_hat_rot1) + np.abs(delta_hat_rot2)))
    p3 = norm.pdf(delta_rot2 - delta_hat_rot2, loc = 0, scale = alpha[0]*np.abs(delta_hat_rot2) + alpha[1]*delta_hat_trans)
    
    #print('Odom model: ',p1,p2,p3, p1*p2*p3)
    return p1*p2*p3
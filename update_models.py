import numpy as np
import utils as utils
import scan_matching as match
from math import sqrt
from scipy.stats import norm
import rbpf_dataloader as data_


def measurement_model(scan, pose, lidar_angles, occupied_indices, MAP,data_):
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
    p_hit =  1 ###define
    p_random = 1 - p_hit
    z_max = 10 
    sigma = 0.38

    
    pose = pose.reshape((3,1))
    

    obstacle = scan<10 

    xy = utils.dist_to_xy(scan[obstacle], lidar_angles[obstacle])
    xy = utils.transformation_scans(xy,pose)
    map_cordinates = np.zeros_like(xy)
    
    map_cordinates[:,0],map_cordinates[:,1] = data_._world_to_map(xy[:,0], xy[:,1] , MAP)

    
    _, min_dist = match.get_correspondance(occupied_indices,map_cordinates)

    exp = (p_hit)*np.exp(-min_dist/(2*sigma**2))
    
    #print('measurement model:',np.mean(exp))

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
    
    
    #alpha = [0.13, 0.13, 0.13, 0.065]
    alpha = [0.18,  0.4,  0.18,  0.0025]
    
    delta_rot1 = np.arctan2(odom_curr[1] - odom_prev[1], odom_curr[0] - odom_prev[0]) - odom_prev[2]
    delta_trans = np.linalg.norm(odom_curr[:2]-odom_prev[:2])
    delta_rot2 = odom_curr[2] - odom_prev[2] - delta_rot1
    
    delta_hat_rot1 = np.arctan2(curr[1] - prev[1], curr[0] - prev[0]) - prev[2]  
    delta_hat_trans = np.linalg.norm(curr[:2]-prev[:2])
    delta_hat_rot2 = curr[2] - prev[2] - delta_hat_rot1
    
    p1 = norm.pdf(angle_diff(delta_rot1 - delta_hat_rot1), loc = 0, scale = alpha[0]*np.abs(delta_hat_rot1) + alpha[1]*delta_hat_trans)
    p2 = norm.pdf(angle_diff(delta_trans - delta_hat_trans), loc = 0, scale = alpha[2]*delta_hat_trans + alpha[3]*(np.abs(delta_hat_rot1) + np.abs(delta_hat_rot2)))
    p3 = norm.pdf(angle_diff(delta_rot2 - delta_hat_rot2), loc = 0, scale = alpha[0]*np.abs(delta_hat_rot2) + alpha[1]*delta_hat_trans)
    
    print('Odom model: ', p1,p2,p3,p1*p2*p3)
    return 1#p1*p2*p3


def angle_diff(angle):
    
    angle = angle%(2*np.pi)
    
    if angle > np.pi:
        angle = angle - 2*np.pi
      
    return angle
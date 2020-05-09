# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:29:24 2020

@author: ravit
"""


import numpy as np
import matplotlib.pyplot as plt
import utils 

def Scan_matcher(prev_scan, prev_pose, curr_scan, curr_best_pose,thresh = 0.45):
    
    """
    Send it in the reverse order.It overlays current_time_steps_map onto prev_time steps scan and then computes inverse
    trasformation to find updated curr_pose. Keep in mind here prev_scan refers to curr_scan and vice versa.
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
    d_pos_total = np.zeros(3)
    d_pose = curr_best_pose - prev_pose #determine sign
    d_pos_total = d_pose
    iters = 0
    
    
    
    curr_scan = utils.transformation_scans(curr_scan,curr_best_pose)
    prev_pose_trial = prev_pose.copy()
    prev_scan = utils.transformation_scans(prev_scan,prev_pose)

    

    trans_scan = utils.transformation_scans(prev_scan,d_pose)

    prev_pose_trial = utils.transformation_scans(prev_pose_trial[:2][None,:],d_pose)
    #plot_graph(curr_scan,trans_scan)

    Correspondance,_ = get_correspondance(curr_scan,trans_scan)
    
    #print('correspondance:',Correspondance)
    
    curr_error = cal_error(curr_scan,trans_scan,Correspondance)
    
    prev_error = 1e8

    while (curr_error < prev_error and iters < 100):

        d_pos_total_prev = d_pos_total
        prev_iter_pose_trial = prev_pose_trial
        
        prev_error = curr_error
        
        d_pose = get_estimate(curr_scan.copy(),trans_scan.copy(),Correspondance)

        
        trans_scan = utils.transformation_scans(trans_scan,d_pose)

        prev_pose_trial = utils.transformation_scans(prev_pose_trial,d_pose)

        d_pos_total = d_pos_total - d_pose

        #plot_graph(curr_scan,trans_scan,title = str(iters))
        Correspondance,_ = get_correspondance(curr_scan,trans_scan)

        
        curr_error = cal_error(curr_scan,trans_scan,Correspondance)
        
        iters += 1
        

        
    if curr_error < thresh:
        Flag = True
    
    
    pose = np.zeros((3,))
    pose[:2] = prev_iter_pose_trial[0,:]
    pose[2] = (prev_pose + d_pos_total_prev)[2] 
    #added for reversal
    d_pose = pose - prev_pose
    d_pose[2] = -d_pose[2]
    R = utils.twoDRotation(d_pose[2])
    d_pose[:2] = -R@d_pose[:2]
    pose_x = curr_best_pose + d_pose
    #print(pose_x)
    return Flag,pose_x


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
    
    min_dist = np.amin(dist,axis = 1)
    
    #print(correspondance.shape)
    #print(x_trans.shape[0])
    
    assert correspondance.shape[0] == x_trans.shape[0]
        
    return correspondance,min_dist


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
    
    #print(error)
    
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
    
    
    W = np.dot(corres_scan.T,trans_scan)
    u,s,vt = np.linalg.svd(W)
    
    R = np.dot(u,vt)
    d_pose[:2] = curr_mean - np.dot(R,trans_mean)
    d_pose[2] = np.arctan2(R[1,0],R[0,0])

      
    return d_pose

def plot_graph(curr_scan,trans_scan,title = 'initial'):
    
    fig,axs = plt.subplots()
    axs.scatter(curr_scan[:,0],curr_scan[:,1],label = 'curr-scan',s = 0.5)
    axs.scatter(trans_scan[:,0],trans_scan[:,1],label = 'trans-scan',s = 0.5)
    axs.legend()
    axs.set_title(title)



def unit_test():
    
    prev_scan = np.array([[1,1],[-1,1]],dtype = np.float64)
    curr_scan = np.array([[1,0],[-1,0]],dtype = np.float64)
    prev_pose = np.array([0,0,0],dtype = np.float64)
    curr_best_pose = np.array([0,2,0],dtype = np.float64)
        
    Flag, pose = Scan_matcher(prev_scan, prev_pose, curr_scan, curr_best_pose,thresh = 0.45)
    print(Flag,pose)
    
if __name__ == "__main__":
    print('in main')
    unit_test()
############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
############################################

import numpy as np

import utils as utils
import scan_matching as match
from math import cos as cos
from math import sin as sin
import update_models as models
import matplotlib.pyplot as plt
import transformations as tf
import scan_matching as matching

class Particle():
    
    def __init__(self, map_dimension, map_resolution, num_p, delta = 0.05, sample_size = 15):

        
        self._init_map(map_dimension, map_resolution)              
        # self.weight_ = 1/num_p #inital value???
        self.weight_factor_ = None
        self.delta = delta
        self.sample_size = sample_size
        self.trajectory_ = np.zeros((3,1),dtype=np.float64) 
        self.traj_indices_ = np.zeros((2,1)).astype(int)
        
        self.log_p_true_ = np.log(9)
        self.log_p_false_ = np.log(1.0/9.0)
        self.p_thresh_ = 0.6
        self.logodd_thresh_ = np.log(self.p_thresh_/(1-self.p_thresh_))
    
    def _init_map(self, map_dimension=20, map_resolution=0.05):
        '''
        map_dimension: map dimention from origin to border
        map_resolution: distance between two grid cells (meters)
        '''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -map_dimension  #meters
        MAP['ymin']  = -map_dimension
        MAP['xmax']  =  map_dimension
        MAP['ymax']  =  map_dimension
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)
       
    def _build_first_map(self, data_, t):
        '''
        Builds initial map using lidar scan 'z' at initial pose
        '''
        scan = data_.lidar_['scan'][t]
        obstacle = scan < data_.lidar_max_
        world_x, world_y = data_._polar_to_cartesian(scan, None)
        map_x, map_y = data_._world_to_map(world_x, world_y, self.MAP_)
        r_map_x, r_map_y = data_._world_to_map(0, 0, self.MAP_)
        
        for ray_num in range(len(scan)):
            cells_x, cells_y = data_._bresenham2D(r_map_x, r_map_y, map_x[ray_num], map_y[ray_num], self.MAP_)
            self.log_odds_[cells_x[:-1], cells_y[:-1]] += self.log_p_false_
            if obstacle[ray_num]:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_true_
            else:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_false_
        self.occu_ = 1 - (1/ (1 + np.exp(self.log_odds_)))
        self.MAP_['map'] = self.occu_ > self.p_thresh_
        
        self.traj_indices_[0], self.traj_indices_[1] = r_map_x, r_map_y
        
        occupied_map = np.where(self.log_odds_ > self.logodd_thresh_)
        self.occupied_pts_ = data_._map_to_world(occupied_map[0], occupied_map[1], self.MAP_)
        
        # plt.imshow(self.MAP_['map'])
        # plt.show()
        # plt.imshow(self.occu_)
        # plt.show()   
        
    def _predict(self, data_, t, mov_cov, scan_odom, scan_flag):
        '''
        Applies motion model on last pose in 'trajectory'
        Returns predicted pose
        '''
        old_pose = self.trajectory_[:,-1]      
        
        if scan_flag[t-1] and scan_flag[t]:
            old_odom = scan_odom[:,t-1]
            new_odom = scan_odom[:,t]
        else:
            old_odom = data_._odom_at_lidar_idx(t-1)
            new_odom = data_._odom_at_lidar_idx(t)    
            
        odom_diff = tf.twoDSmartMinus(new_odom, old_odom)     
        noise = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).flatten()
        
        pred_pose = tf.twoDSmartPlus(old_pose, odom_diff)
        pred_with_noise = tf.twoDSmartPlus(pred_pose, noise)
        
        return pred_pose, pred_with_noise
        
    
    def _scan_matching(self, data_, t , pred_odom):
        '''
        Performs scan matching and returns (true,scan matched pose) or (false,None)
        '''
        
        curr_scan = data_.lidar_['scan'][t]
        prev_scan = data_.lidar_['scan'][t-1]
        
        curr_coordinates = utils.dist_to_xy(curr_scan, data_.lidar_angles_)
        curr_odom = pred_odom
        prev_coordinates = utils.dist_to_xy(prev_scan, data_.lidar_angles_)
        prev_odom = self.trajectory_[:,-1]
        
        flag, updated_pose = matching.Scan_matcher(curr_coordinates.copy(), curr_odom.copy(), prev_coordinates.copy(), prev_odom.copy())      
        return flag, updated_pose
    
    
    def _sample_poses_in_interval(self, scan_match_pose):  ## RAVI
        '''
        scan matched pose: (3,1)
        
        Returns list of samples (3,sample_size)
        '''
        scan_match_pose = scan_match_pose.reshape((3,1))
        
        samples = np.random.random_sample((3,self.sample_size))    #### can allocate different delta's for x,y,theta
        samples = samples*self.delta
        samples = samples + scan_match_pose
    
        return samples
        
    
    def _compute_new_pose(self, data_, t, pose_samples):           ##RAVI
        '''
        Computes mean,cov,weight factor from pose_samples
        Samples new_pose from gaussian and appends to trajectory
        Updates weight
        '''
        mean = np.zeros((3,))
        variance = np.zeros((3,3))
        eta = np.zeros(pose_samples.shape[1])
        
        odom = data_._odom_at_lidar_idx(t)
        odom_prev = data_._odom_at_lidar_idx(t-1)
        scan = data_.lidar_['scan'][t]
        
        pose_prev = self.trajectory_[:,-1]
        
        for i in range(pose_samples.shape[1]):
            prob_measurement = models.measurement_model(scan, pose_samples[:,i], data_.lidar_angles_, self.occupied_pts_.T)
            odom_measurement = models.odometry_model(pose_prev, pose_samples[:,i], odom_prev, odom)
            eta[i] = (prob_measurement)*(odom_measurement)
            mean += pose_samples[:,i]*eta[i]
            
        print('Eta: ',np.sum(eta))
        
        mean = mean/np.sum(eta)
        mean = np.reshape(mean,(3,1))
        
        for i in range(pose_samples.shape[1]):
            variance += (pose_samples[:,i] - mean)@((pose_samples[:,i] - mean).T)*eta[i]
        
        variance = variance/np.sum(eta)   
        new_pose = np.random.multivariate_normal(mean.flatten(), variance)       
        self.weight_ = self.weight_ * np.sum(eta)
        
        return new_pose
        
    def _get_lidar_map_correspondence(self, data_, t, pose):
        '''
            Find number of lidar pts that hits obstacle cell in prev map
        '''
        scan = data_.lidar_['scan'][t]
        obstacle = scan < data_.lidar_max_
        world_x, world_y = data_._polar_to_cartesian(scan, pose)
        map_x, map_y = data_._world_to_map(world_x, world_y, self.MAP_)
        
        obstacle_x, obstacle_y = map_x[obstacle], map_y[obstacle]
        MAP_obstacle = self.log_odds_>self.logodd_thresh_
        num_hit = np.sum(MAP_obstacle[obstacle_x, obstacle_y])
        
        return num_hit
    
    def _update_map(self, data_, t, pose):
        '''
            Updates map with lidar scan z for last pose in trajectory

        '''
        scan = data_.lidar_['scan'][t]
        obstacle = scan < data_.lidar_max_
        world_x, world_y = data_._polar_to_cartesian(scan, pose)
        map_x, map_y = data_._world_to_map(world_x, world_y, self.MAP_)
        r_map_x, r_map_y = data_._world_to_map(pose[0], pose[1], self.MAP_)
        
        for ray_num in range(len(scan)):
            cells_x, cells_y = data_._bresenham2D(r_map_x, r_map_y, map_x[ray_num], map_y[ray_num], self.MAP_)
            self.log_odds_[cells_x[:-1], cells_y[:-1]] += self.log_p_false_
            if obstacle[ray_num]:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_true_
            else:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_false_
        self.occu_ = 1 - (1/ (1 + np.exp(self.log_odds_)))
        self.MAP_['map'] = self.occu_ > self.p_thresh_
        
        self.traj_indices_ = np.append(self.traj_indices_, np.array([[r_map_x],[r_map_y]]), 1)
        self.trajectory_ = np.append(self.trajectory_, np.reshape(pose,(3,1)), 1)
        
        occupied_map = np.where(self.log_odds_ > self.logodd_thresh_)
        self.occupied_pts_ = data_._map_to_world(occupied_map[0], occupied_map[1], self.MAP_)
        
    
    
    
    
    
    
    
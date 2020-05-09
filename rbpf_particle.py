############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
############################################

import numpy as np
import matplotlib.pyplot as plt
import transformations as tf

class Particle():
    
    def __init__(self, map_dimension, map_resolution, num_p):
        
        self._init_map(map_dimension, map_resolution)              
        self.weight_ = 1/num_p #inital value???
        self.weight_factor_ = None
        self.trajectory_ = np.zeros((3,1),dtype=np.float64) 
        self.traj_indices_ = np.zeros((2,1)).astype(int)
        
        self.log_p_true = np.log(9)
        self.log_p_false = np.log(1.0/9.0)
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
        world_x, world_y = data_._polar_to_cartesian(scan)
        map_x, map_y = data_._world_to_map(world_x, world_y, self.MAP_)
        r_map_x, r_map_y = data_._world_to_map(0, 0, self.MAP_)
        
        for ray_num in range(len(scan)):
            cells_x, cells_y = data_._bresenham2D(r_map_x, r_map_y, map_x[ray_num], map_y[ray_num], self.MAP_)
            self.log_odds_[cells_x[:-1], cells_y[:-1]] += self.log_p_false
            if obstacle[ray_num]:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_true
            else:
                self.log_odds_[cells_x[-1], cells_y[-1]] += self.log_p_false
        self.occu_ = 1 - (1/ (1 + np.exp(self.log_odds_)))
        self.MAP_['map'] = self.occu_ > self.p_thresh_
        
        self.traj_indices_[0], self.traj_indices_[1] = r_map_x, r_map_y
        # plt.imshow(self.MAP_['map'])
        # plt.show()
        # plt.imshow(self.occu_)
        # plt.show()   
        
    def _predict(self, data_, t, mov_cov):
        '''
        Applies motion model on last pose in 'trajectory'
        Returns predicted pose
        '''
        old_pose = self.trajectory_[:,-1]      
        
        old_odom = data_._odom_at_lidar_idx(t-1)
        new_odom = data_._odom_at_lidar_idx(t)     
        odom_diff = tf.twoDSmartMinus(new_odom, old_odom)     
        noise = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).flatten()
        
        smart_plus = tf.twoDSmartPlus(old_pose, odom_diff)
        pred_pose = tf.twoDSmartPlus(smart_plus, noise)
        
        return pred_pose
        
    
    def _scan_matching(self, predicted_pose, search_interval, z):
        '''
        Performs scan matching and returns (true,scan matched pose) or (false,None)
        '''
        pass
    
    def _sample_poses_in_interval(self, scan_match_pose, interval):  ## RAVI
        '''
        Samples around scan matched pose
        Returns list of samples
        '''
        pass
    
    def _compute_new_pose(self, pose_samples):           ##RAVI
        '''
        Computes mean,cov,weight factor from pose_samples
        Samples new_pose from gaussian and appends to trajectory
        Updates weight
        '''
    
    def _update_map(self, data_, t):
        '''
        updates map with lidar scan z for last pose in trajectory

        '''
        pass
    
    
    
    
    
    
    
    
    
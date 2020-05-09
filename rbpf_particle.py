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

class Particle():
    
    '''
    Things to consider:
        -how to represent poses/trajectory, poses are list, trajectory as list of poses? any advantage to making it np array?
        -initial weight of particles
        
    '''
    def __init__(self, map_dimension, map_resolution, num_p, delta = 0.05, sample_size = 15):
        
        self._init_map(map_dimension, map_resolution)              
        self.weight_ = 1/num_p #inital value???
        self.weight_factor_ = None
        self.delta = delta
        self.sample_size = sample_size
        self.trajectory_ = np.zeros((3,1),dtype=np.float64) 
        
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
       
    def _build_first_map(self, z):
        '''
        Updates initial map using lidar scan 'z' at initial pose
        '''
        pass
        
    
    def _predict(self, control):
        '''
        Applies motion model on last pose in 'trajectory'
        Returns predicted pose

        '''
        pass
    
    def _scan_matching(self, predicted_pose, search_interval, z):
        '''
        Performs scan matching and returns (true,scan matched pose) or (false,None)
        '''
        pass
    
    def _sample_poses_in_interval(self, scan_match_pose):  ## RAVI
        '''
        scan matched pose: (3,1)
        
        Returns list of samples (3,sample_size)
        '''
        scan_match_pose = scan_match_pose.reshape((3,1))
        
        sample = np.random.random_sample((3,self.sample_size))    #### can allocate different delta's for x,y,theta
        sample = sample*self.delta
        sample = sample + scan_match_pose

        
        return sample
        
    
    def _compute_new_pose(self, t, pose_samples, particle_index):           ##RAVI
        '''
        Computes mean,cov,weight factor from pose_samples
        Samples new_pose from gaussian and appends to trajectory
        Updates weight
        '''
        mean = np.zeros((3,))
        variance = np.zeros((3,))
        eta = np.zeros(pose_samples.shape[1])
        lidar_angles = np.arange(data.lidar_specs_['angle_min'],data.lidar_specs_['angle_max'],data.lidar_specs_['angle_increment'])
        odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][t]))
        odom_index_prev = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][t-1]))
        for i in range(pose_samples.shape[1]):
            prob_measurement = models.measurement_model(scan, pose_samples[:,i],lidar_angles,occupied_indices)
            odom_measurement = models.odom_model(best_particle_prev,pose_samples[:,i],odom[odom_index_prev],odom[odom_index])
            mean += pose_samples[:,i]*prob_measurement*odom_measurement
            eta[i] = (prob_measurement)*(odom_measurement)
        
        mean = mean/np.sum(eta)
        #sigma = 0
        
        for i in range(pose_samples.shape[1]):
            variance += (pose_samples - mean)@((pose_samples - mean).T)*eta[i]
        
        variance = variance/np.sum(eta)
        
        new_pose = np.random.multivariate_normal(mean, variance)
        
        self.weigth[particle_index] = self.weight[particle_index]*np.sum(eta)
        
        return new_pose
        
        
    
    def _update_map(self, z):
        '''
        updates map with lidar scan z for last pose in trajectory

        '''
        pass
    
    
    
    
    
    
    
    
    
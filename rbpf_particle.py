############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
############################################

import numpy as np

class Particle():
    
    '''
    Things to consider:
        -how to represent poses/trajectory, poses are list, trajectory as list of poses? any advantage to making it np array?
        -initial weight of particles
        
    '''
    def __init__(self, map_dimension, map_resolution):
        
        self._init_map(map_dimension, map_resolution)              
        self.weight_ = None #inital value???
        self.weight_factor_ = None
        self.trajectory_ = np.zeros((3,1),dtype=np.float64) #should it just be a list of poses???
        
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
        Performs scan matching and returns true,scan matched pose or false,None
        '''
        pass
    
    def _sample_poses_in_interval(self, scan_match_pose, interval):
        '''
        Samples around scan matched pose
        Returns list of samples
        '''
        pass
    
    def _compute_new_pose(self, pose_samples):
        '''
        Computes mean,cov,weight factor from pose_samples
        Samples new_pose from gaussian and appends to trajectory
        Updates weight
        '''
    
    def _update_map(self, z):
        '''
        updates map with lidar scan z for last pose in trajectory

        '''
        pass
    
    
    
    
    
    
    
    
    
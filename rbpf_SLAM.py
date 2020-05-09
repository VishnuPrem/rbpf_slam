# -*- coding: utf-8 -*-

############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
#             RBPF SLAM Class
############################################


import numpy as np
from rbpf_particle import Particle
from rbpf_dataloader import DataLoader
import matplotlib.pyplot as plt
import cv2
import transformations as tf

class SLAM():
    
    def __init__(self, data_path, mov_cov, num_p = 20, map_resolution = 0.05, map_dimension = 20, Neff_thresh = 0.6):
        
        self.num_p_ = num_p
        self.Neff_ = 0
        self.Neff_thresh_ = Neff_thresh
        self.weights_ = np.ones(num_p)/num_p
        self.mov_cov_ = mov_cov
        
        self.particles_ = []
        for i in range(self.num_p_):
            self.particles_.append(Particle(map_dimension, map_resolution, num_p))
        
        self.data_ = DataLoader(data_path[0], data_path[1], data_path[2])
        
        
    def _resample(self):
        pass
    
    def _slam_update(self):       
        for p in self.particles_:
            pass     
        if self.Neff_ > self.Neff_thresh_:
            self._resample()
        
    def _run_slam(self, t0, t_end = None):
        '''
            Performs SLAM
        '''
        t_end = self.data_.lidar_['num_data'] if t_end is None else t_end + 1
                 
        for t in range(t0, t_end):                     
            if t == t0:
                print("----Building first map----")
                for p in self.particles_:
                    p._build_first_map(self.data_, t)
                    self._gen_map(p)
                continue
            
            for p in self.particles_:
                
                # predict with motion model
                pred_pose = p._predict(self.data_, t, self.mov_cov_)
                
                #### REMEAINING STEPS
                est_pose = pred_pose
                ####
                
                p._update_map(self.data_, t, est_pose)
                
        self._gen_map(p)
    
    
    def _mapping_with_known_poses(self, t0, t_end = None, interval = 1):
        '''
            Uses noiseless odom data to generate entire map
        '''
        t_end = self.data_.lidar_['num_data'] if t_end is None else t_end + 1
        p = self.particles_[0]
        for t in range(t0, t_end, interval):                             
            odom = self.data_._odom_at_lidar_idx(t)  
            p._update_map(self.data_, t, odom)
            if t%50==0:
                self._gen_map(p)
            print(t)                
        self._gen_map(p)
                
            
    def _gen_map(self, particle):
        '''
            Generates map for visualization
        '''
        log_odds      = particle.log_odds_
        logodd_thresh = particle.logodd_thresh_
        MAP = particle.MAP_
        traj = particle.traj_indices_
        
        MAP_2_display = 255*np.ones((MAP['sizex'],MAP['sizey'],3),dtype=np.uint8)
        wall_indices = np.where(log_odds > logodd_thresh)
        MAP_2_display[list(wall_indices[0]),list(wall_indices[1]),:] = [0,0,0]
        unexplored_indices = np.where(abs(log_odds) < 1e-1)
        MAP_2_display[list(unexplored_indices[0]),list(unexplored_indices[1]),:] = [150,150,150]
        MAP_2_display[traj[0],traj[1]] = [70,70,228]
        plt.imshow(MAP_2_display)
        plt.show()
        cv2.imwrite('logs/map.png', MAP_2_display)
        return MAP_2_display
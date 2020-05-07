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

class SLAM():
    
    def __init__(self,  mov_cov, num_p = 20, map_resolution = 0.05, map_dimension = 20, Neff_thresh = 0.6):
        
        self.num_p = num_p
        self.Neff = 0
        self.Neff_thresh = Neff_thresh
        self._init_particles(map_dimension, map_resolution)
        
        self.particles = []
        for i in range(self.num_p):
            self.particles.append(Particle(map_dimension, map_resolution))
        
        lidar_scan_path = "data/processed_lidar.pkl"
        odom_path = "data/processed_odom.pkl"
        lidar_specs_path = "data/lidar_specs.pkl"
        self.data = DataLoader(lidar_scan_path, odom_path, lidar_specs_path)
        
    def resample(self):
        pass
    
    def slam_update(self):
        
        for p in self.particles:
            pass
        
        if self.Neff > self.Neff_thresh:
            self.resample()
        
        
    # def genMap(particle, end_t=None):
        
    #     log_odds      = particle.log_odds_
    #     logodd_thresh = particle.logodd_thresh_

    #     t0             = slam_inc.t0
    #     if end_t is None:
    #         end_t          = slam_inc.num_data_ - 1
    
    #     MAP_2_display = 255*np.ones((MAP['map'].shape[0],MAP['map'].shape[1],3),dtype=np.uint8)
    
    #     wall_indices = np.where(log_odds > logodd_thresh)
    #     MAP_2_display[list(wall_indices[0]),list(wall_indices[1]),:] = [0,0,0]
    #     unexplored_indices = np.where(abs(log_odds) < 1e-1)
    #     MAP_2_display[list(unexplored_indices[0]),list(unexplored_indices[1]),:] = [150,150,150]
    #     MAP_2_display[best_p_indices[0,t0:end_t], best_p_indices[1,t0:end_t],:] = [70,70,228]#
        
    #     return MAP_2_display
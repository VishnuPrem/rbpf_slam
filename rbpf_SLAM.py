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

class SLAM():
    
    def __init__(self):
        
        self.num_p = 20
        map_resolution = 0.05
        map_dimension = 20
        self._init_particles(map_dimension, map_resolution)
        self.Neff = 0
        self.Neff_thresh = 0.6
        
    def _init_particles(self, map_dimension, map_resolution):   
        
        self.particles = []
        for i in range(self.num_p):
            self.particles.append(Particle(map_dimension, map_resolution))
            
    def resample(self):
        pass
    
    def slam_update(self):
        
        for p in self.particles:
            pass
        
        if self.Neff > self.Neff_thresh:
            self.resample()
            
    
            
    
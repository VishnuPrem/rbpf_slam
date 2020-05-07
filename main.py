# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:23:19 2020

@author: Vishnu Prem
"""
import numpy as np
from rbpf_slam import SLAM

mov_cov = np.array([[1e-4, 0, 0],
                    [0, 1e-4, 0],
                    [0, 0 , 1e-4]])
map_resolution = 0.05
map_dimension = 20
num_p = 60
p_thresh = 0.5
Neff_thresh = 0.6

slam = SLAM(num_p, map_resolution, map_dimension, Neff_thresh)
num_data = slam.data_.lidar_['num_data']



# -*- coding: utf-8 -*-

############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
#          ---main script---
############################################

import numpy as np
from rbpf_SLAM import SLAM



lidar_scan_path = "data/processed_lidar.pkl"
odom_path = "data/processed_odom.pkl"
lidar_specs_path = "data/lidar_specs.pkl"
data_path = [lidar_scan_path, odom_path, lidar_specs_path]

mov_cov = np.array([[1e-4, 0, 0],
                    [0, 1e-4, 0],
                    [0, 0 , 1e-4]])

map_resolution = 0.05
map_dimension = 25
num_p = 1
p_thresh = 0.5
Neff_thresh = 0.6

t0 = 0
t_end = 1 # timestep to end SLAM

slam = SLAM(data_path, mov_cov, num_p, map_resolution, map_dimension, Neff_thresh)
slam._run_slam(t0, t_end)

# slam._mapping_with_known_poses(t0,t_end)

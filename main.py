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
import matplotlib.pyplot as plt
import pickle

scan_match_path = "data/scan_match.npy"
flag_path = "data/flags.npy"

scan_match_odom = np.load(scan_match_path)
scan_match_flag = np.load(flag_path)
scan_match_odom[0] = scan_match_odom[0] - scan_match_odom[0,0]  
scan_match_odom[1] = scan_match_odom[1] - scan_match_odom[1,0]

lidar_scan_path = "data/processed_lidar.pkl"
odom_path = "data/processed_odom.pkl"
lidar_specs_path = "data/lidar_specs.pkl"
data_path = [lidar_scan_path, odom_path, lidar_specs_path]

mov_cov = np.array([[1e-5, 0, 0],
                    [0, 1e-5, 0],
                    [0, 0 , 1e-5]])

map_resolution = 0.05
map_dimension = 25
num_p = 20
p_thresh = 0.5
Neff_thresh = 0.6

t0 = 50
t_end = None # timestep to end SLAM

slam = SLAM(data_path, mov_cov, num_p, map_resolution, map_dimension, Neff_thresh)
slam._run_slam_simple(scan_match_odom, scan_match_flag, t0, t_end)

# best_p_idx = np.argmax(slam.weights_)
# print('Best particle: ', best_p_idx)

# slam._mapping_with_known_poses(scan_match_odom, scan_match_flag, t0,t_end)

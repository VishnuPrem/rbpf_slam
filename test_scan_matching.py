# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:44:27 2020

@author: ravit
"""


import rbpf_dataloader as data_loader
import scan_matching as matching
import numpy as np
import utils as utils
import matplotlib.pyplot as plt
import tqdm


lidar_scan_path = "data\processed_lidar.pkl"
odom_path = "data\processed_odom.pkl"
lidar_specs_path = "data\lidar_specs.pkl"
data = data_loader.DataLoader(lidar_scan_path,odom_path,lidar_specs_path)

"""
print(data.lidar_['num_data'])
print(data.lidar_['scan'].shape)
print(data.odom_['x'].shape)
print(data.odom_['num_data'])"""

theta_min = data.lidar_specs_['angle_min']
lidar_angles = np.arange(data.lidar_specs_['angle_min'],data.lidar_specs_['angle_max'],data.lidar_specs_['angle_increment'])
updated_pos = np.zeros((3,data.lidar_['num_data']))
Flags = np.zeros((data.lidar_['num_data']))
Flags[0] = True

odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][0]))
prev_odom = np.zeros((3,))
prev_odom[0] = data.odom_['x'][odom_index]
prev_odom[1] = data.odom_['y'][odom_index]
prev_odom[2] = data.odom_['theta'][odom_index]
prev_scan = data.lidar_['scan'][0]
prev_coordinates = utils.dist_to_xy(prev_scan,lidar_angles)

for i in tqdm.tqdm(range(1,data.lidar_['num_data'])):
    odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][i]))
    curr_odom = np.zeros((3,))
    curr_odom[0] = data.odom_['x'][odom_index]
    curr_odom[1] = data.odom_['y'][odom_index]
    curr_odom[2] = data.odom_['theta'][odom_index]
    curr_scan = data.lidar_['scan'][i]
    
    curr_coordinates = utils.dist_to_xy(curr_scan,lidar_angles)
    Flags[i], updated_pos[:,i] = matching.Scan_matcher(prev_coordinates, prev_odom, curr_coordinates, curr_odom) 
    
    prev_coordinates = curr_coordinates
    prev_odom = curr_odom
    
true_pos = updated_pos[:,Flags.astype('bool')]
false_pos = updated_pos[:,np.logical_not(Flags.astype("bool"))]    
plt.plot(data.odom_['x'], data.odom_['y'])
plt.plot(updated_pos[0,:],updated_pos[1,:])
plt.scatter(true_pos[0,:], true_pos[1,:],color = 'green',s = 0.5)
#plt.scatter(false_pos[0,:], false_pos[1,:],color = 'red',s = 0.5)
plt.show()
print(np.sum(Flags))
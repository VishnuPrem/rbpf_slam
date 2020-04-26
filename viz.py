# -*- coding: utf-8 -*-
'''
Created on Sat Apr 25 21:48:04 2020

@author: Vishnu Prem


 Script to vizualise the lidar scan and odometry
 
'''

from rbpf_dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# def odom_viz():

lidar_scan_path = "data/processed_lidar.pkl"
odom_path = "data/processed_odom.pkl"
lidar_specs_path = "data/lidar_specs.pkl"

data = DataLoader(lidar_scan_path, odom_path, lidar_specs_path)
    
angle_min = data.lidar_specs_['angle_min']
angle_max = data.lidar_specs_['angle_max']
angle_increment = data.lidar_specs_['angle_increment']

for i in range(0,data.lidar_['num_data'],5):
    
    lidar_angles = np.arange(angle_min, angle_max, angle_increment)
    lidar_scan = data.lidar_['scan'][i]
    lidar_scan[lidar_scan > 20] = 0
    lidar_ptx = lidar_scan * np.cos(lidar_angles)
    lidar_pty = lidar_scan * np.sin(lidar_angles)
            
    plt.scatter(-lidar_pty, lidar_ptx,0.2)
    plt.axis('scaled')
    map_size = 20
    plt.xticks(np.arange(-map_size, map_size+1, 5))
    plt.yticks(np.arange(-map_size, map_size+1, 5))
       
    plt.show()

plt.plot(data.odom_['x'], data.odom_['y'])
plt.show()

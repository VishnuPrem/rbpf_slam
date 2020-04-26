# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:44:27 2020

@author: ravit
"""


import rbpf_dataloader as data_loader
import scan_matching as matching
import numpy as np
import matplotlib.pyplot as plt


lidar_scan_path = "data\processed_lidar.pkl"
odom_path = "data\processed_odom.pkl"
lidar_specs_path = "data\lidar_specs.pkl"
data = data_loader.DataLoader(lidar_scan_path,odom_path,lidar_specs_path)
print(data.lidar_['num_data'])
print(data.lidar_.shape)
print(data.odom_.shape)
print(data.odom_['num_data'])
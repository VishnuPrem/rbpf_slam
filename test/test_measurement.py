import update_models as models
import rbpf_dataloader as data_loader
import numpy as np
import utils as utils
import matplotlib.pyplot as plt


lidar_scan_path = "data\processed_lidar.pkl"
odom_path = "data\processed_odom.pkl"
lidar_specs_path = "data\lidar_specs.pkl"
data = data_loader.DataLoader(lidar_scan_path,odom_path,lidar_specs_path)
lidar_angles = data.lidar_angles_

odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][0]))
prev_odom = np.zeros((3,))
prev_odom[0] = data.odom_['x'][odom_index]
prev_odom[1] = data.odom_['y'][odom_index]
prev_odom[2] = data.odom_['theta'][odom_index]
prev_scan = data.lidar_['scan'][0]
prev_coordinates = utils.dist_to_xy(prev_scan,lidar_angles)
prev_coordinates = utils.transformation_scans(prev_coordinates,prev_odom)

curr_scan = data.lidar_['scan'][1]
odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][1]))
curr_odom = np.zeros((3,))
curr_odom[0] = data.odom_['x'][odom_index]
curr_odom[1] = data.odom_['y'][odom_index]
curr_odom[2] = data.odom_['theta'][odom_index]


#prob_measurement,prob,prob1 = models.measurement_model(curr_scan, curr_odom, lidar_angles, prev_coordinates)
#print(prob_measurement,prob,prob1)

curr = curr_odom.copy()
curr[0] += 0.2





odom_measurement = models.odometry_model(prev_odom, curr_odom, prev_odom, curr_odom)

print(odom_measurement)


import rbpf_dataloader as data_loader
import scan_matching as matching
import numpy as np
import utils as utils



lidar_scan_path = "data\processed_lidar.pkl"
odom_path = "data\processed_odom.pkl"
lidar_specs_path = "data\lidar_specs.pkl"
data = data_loader.DataLoader(lidar_scan_path,odom_path,lidar_specs_path)

"""
print(data.lidar_['num_data'])
print(data.lidar_['scan'].shape)
print(data.odom_['x'].shape)
print(data.odom_['num_data'])"""

lidar_angles = data.lidar_angles_
#lidar_angles = np.arange(data.lidar_specs_['angle_min'],data.lidar_specs_['angle_max'],data.lidar_specs_['angle_increment'])
#updated_pos = np.zeros((3,data.lidar_['num_data']))
Flags = np.zeros((data.lidar_['num_data']))
Flags[0] = True
index1 = 250
odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][index1]))
prev_odom = np.zeros((3,))
prev_odom[0] = data.odom_['x'][odom_index]
prev_odom[1] = data.odom_['y'][odom_index]
prev_odom[2] = data.odom_['theta'][odom_index]
prev_scan = data.lidar_['scan'][index1]
prev_coordinates = utils.dist_to_xy(prev_scan,lidar_angles)
index2 = 251
odom_index = np.argmin(abs(data.odom_['time'] - data.lidar_['time'][index2]))
curr_odom = np.zeros((3,))
curr_odom[0] = data.odom_['x'][odom_index]
curr_odom[1] = data.odom_['y'][odom_index]
curr_odom[2] = data.odom_['theta'][odom_index]
curr_scan = data.lidar_['scan'][index2]

curr_coordinates = utils.dist_to_xy(curr_scan,lidar_angles)
Flags, updated_pos = matching.Scan_matcher(curr_coordinates.copy(), curr_odom.copy(), prev_coordinates.copy(), prev_odom.copy()) 

print("updated position",updated_pos)
print("prev position",prev_odom)
print("curr position",curr_odom)
print('unaltered position:',data.odom_['x'][odom_index],data.odom_['y'][odom_index])

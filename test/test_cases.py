import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import scan_matching as match

x = np.arange(-5,5,0.1)
x = x[:,None]
y = x**2
prev_scan = np.hstack((x,y))
prev_pose = np.array([0,0,0])
pose = np.array([6,12,np.pi/6])
curr_scan = utils.transformation_scans(prev_scan,-pose)
curr_pose = np.array([5,11,np.pi/4])
plt.scatter(x,y,s = 0.5)
plt.scatter(curr_scan[:,0],curr_scan[:,1],s = 0.5)
#d = np.random()
Flags, updated_pos = match.Scan_matcher(prev_scan, prev_pose, curr_scan, curr_pose)
print(updated_pos)
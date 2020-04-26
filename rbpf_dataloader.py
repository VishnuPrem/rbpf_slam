############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
#           Data Loader Class
############################################

import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

class DataLoader():
    '''
    self.lidar_['time'] (time in milliseconds)
    self.lidar_['scan'] (num_data x 1080)
    self.lidar_['num_data'] 
    
    self.odom_['time'] (time in milliseconds)
    self.odom_['x'] (meters??)
    self.odom_['y']
    self.odom_['theta'] (radians)
    self.odom_['num_data']
    '''
    def __init__(self, lidar_path, odom_path, specs_path):
        
        lidar = pickle.load(open(lidar_path, "rb"))
        odom = pickle.load(open(odom_path, "rb"))
        lidar_specs = pickle.load(open(specs_path, "rb"))
        
        self.lidar_ = lidar
        self.odom_ = odom
        self.lidar_specs_ =  lidar_specs  
        
        self.lidar_['num_data'] = len(self.lidar_['time'])
        self.odom_['num_data'] = len(self.odom_['time'])
        
        
# create Dataloader instance like this    
def test_data_loader():
    
    lidar_scan_path = "data/processed_lidar.pkl"
    odom_path = "data/processed_odom.pkl"
    lidar_specs_path = "data/lidar_specs.pkl"
    
    data = DataLoader(lidar_scan_path, odom_path, lidar_specs_path)
    
    
    
#### ONLY FOR CLEANING PICKLED BAG DATA ####
def clean_data():
    # converts pickled bag into clean format for loading
    lidar_scan_path = "data/lidar_scan.pkl"
    odom_path = "data/odom.pkl"   
    
    lidar = pickle.load(open(lidar_scan_path, "rb"))
    odom = pickle.load(open(odom_path, "rb"))
    
    lidar_t, lidar_scan = clean_lidar(lidar)
    odom_t, odom_x, odom_y, odom_theta = clean_odom(odom)
    
    lidar_data = {'time':lidar_t, 'scan':lidar_scan}
    odom_data = {'time':odom_t, 'x':odom_x, 'y':odom_y, 'theta':odom_theta}
    
    pickle.dump( lidar_data, open( "processed_lidar.pkl", "wb" ) )
    pickle.dump( odom_data, open( "processed_odom.pkl", "wb" ) )
    print('done')
    
def clean_odom(odom):
    
    sec = np.array(odom['sec'])
    nsec = np.array(odom['nsec'])   
    sec = sec - sec[0]
    msec = nsec/1000000   
    tot_msec = sec*1000 + msec
    tot_msec = tot_msec.astype(int)
    
    x = np.array(odom['posx'])
    y = np.array(odom['posy'])
    x = x - x[0]
    y = y - y[0]
    
    quatz = np.array(odom['quatz'])
    quatw = np.array(odom['quatw'])   
    quatx = np.zeros(quatz.shape)
    quaty = np.zeros(quatz.shape)
    
    quat = np.vstack((quatx,quaty,quatz,quatw))
    quat = quat.transpose()   
    r = R.from_quat(quat)
    euler = r.as_euler('zyx', degrees=False)
    
    theta = euler[:,0]
    theta = theta - theta[0]
    
    angle_range = np.pi
    theta[theta>angle_range] = theta[theta>angle_range] - angle_range*2 
    theta[theta<-angle_range] = theta[theta<-angle_range] + angle_range*2 
    
    tot_msec, x, y, theta = tot_msec[::50], x[::50], y[::50], theta[::50]
    return tot_msec, x, y, theta
    
def clean_lidar(lidar):
    # returns time in millisecs
    
    sec = np.array(lidar['sec'])
    nsec = np.array(lidar['nsec'])
    scan = np.array(lidar['scan'])
    
    sec = sec - sec[0]
    msec = nsec/1000000
    
    tot_msec = sec*1000 + msec
    tot_msec = tot_msec.astype(int)
    
    tot_msec, scan = tot_msec[::50], scan[::50]
    return tot_msec, scan
    
def data_correspondence():
    pass
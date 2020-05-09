
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
    
    self.lidar_specs_ :
    {
        'range_max': 100.0,
        'angle_min': -3.1415927410125732,
        'range_min': 0.0,
        'angle_increment': 0.005823155865073204,
        'angle_max': 3.1415927410125732
    }
    
    self.odom_['time'] (time in milliseconds)
    self.odom_['x'] (meters??)
    self.odom_['y']
    self.odom_['theta'] (radians)
    self.odom_['num_data']
    
    self.lidar_angles_: angles of each ray in self.lidar_['scan'] in radian
    '''
    def __init__(self, lidar_path, odom_path, specs_path):
        
        lidar = pickle.load(open(lidar_path, "rb"))
        odom = pickle.load(open(odom_path, "rb"))
        lidar_specs = pickle.load(open(specs_path, "rb"))
        
        self.lidar_ = lidar
        self.lidar_['num_data'] = len(self.lidar_['time'])
        
        self.odom_ = odom
        self.odom_['num_data'] = len(self.odom_['time'])
        
        angle_min = lidar_specs['angle_min']
        angle_max = lidar_specs['angle_max']
        angle_increment = lidar_specs['angle_increment']
        
        self.lidar_angles_ = np.arange(angle_min, angle_max, angle_increment)
        self.lidar_max_ = 10
        
    def _odom_at_lidar_idx(self, idx):
        '''
            Return odom data corresponding to time t in lidar data
        '''
        lidar_t = self.lidar_['time'][idx]
        odom_idx = np.argmin(np.abs(self.odom_['time'] - lidar_t))
        return np.array([self.odom_['x'][odom_idx], self.odom_['y'][odom_idx], self.odom_['theta'][odom_idx]])
        
    def _polar_to_cartesian(self, scan):
        '''
            Converts polar scan to cartisian x,y coordinates
        '''
        scan[scan > self.lidar_max_ ] = self.lidar_max_ 
        lidar_ptx = scan * np.cos(self.lidar_angles_)
        lidar_pty = scan * np.sin(self.lidar_angles_)
        return lidar_ptx, lidar_pty
    
    def _world_to_map(self, world_x, world_y, MAP):
        '''
            Converts x,y from meters to map coods
        '''
        map_x = np.ceil((world_x - MAP['xmin']) / MAP['res']).astype(np.int16)-1
        map_y = np.ceil((world_y - MAP['ymin']) / MAP['res']).astype(np.int16)-1
        return map_x, map_y
    
    
    def _bresenham2D(self,sx,sy,ex,ey, MAP):
        
        sx = int(np.round(sx))
        sy = int(np.round(sy))
        ex = int(np.round(ex))
        ey = int(np.round(ey))
        dx = abs(ex-sx)
        dy = abs(ey-sy)
        steep = abs(dy)>abs(dx)
        if steep:
          dx,dy = dy,dx # swap 
        
        if dy == 0:
          q = np.zeros((dx+1,1))
        else:      
          arange = np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1, -dy)  
          mod = np.mod(arange,dx)
          diff = np.diff(mod)  
          great =  np.greater_equal(diff,0) 
          q = np.append(0, great)
        
        if steep:
          if sy <= ey:
            y = np.arange(sy,ey+1)
          else:
            y = np.arange(sy,ey-1,-1)
          if sx <= ex:
            x = sx + np.cumsum(q)
          else:
            x = sx - np.cumsum(q)
        else:
          if sx <= ex:
            x = np.arange(sx,ex+1)
          else:
            x = np.arange(sx,ex-1,-1)
          if sy <= ey:
            y = sy + np.cumsum(q)
          else:
            y = sy - np.cumsum(q)
            
        x_valid = np.logical_and(x>=0, x<MAP['sizex'])
        y_valid = np.logical_and(y>=0, y<MAP['sizey'])
        cell_valid = np.logical_and(x_valid, y_valid)
        x = x[cell_valid]
        y = y[cell_valid]
        
        return np.vstack((x,y)).astype(int)

       
# create Dataloader instance like this    
def test_data_loader():
    
    lidar_scan_path = "data/processed_lidar.pkl"
    odom_path = "data/processed_odom.pkl"
    lidar_specs_path = "data/lidar_specs.pkl"
    
    return DataLoader(lidar_scan_path, odom_path, lidar_specs_path)
    
    
if __name__ == '__main__':
    
    d = test_data_loader()
    res = d.bresenham2D(3,3,10,10)     

    
    
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
    

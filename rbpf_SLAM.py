# -*- coding: utf-8 -*-

############################################
#       University of Pennsylvania
#            ESE 650 Project
#     Authors: Vishnu Prem & Ravi Teja
#   Rao-Blackwellized Paricle Filter SLAM
#             RBPF SLAM Class
############################################


import numpy as np
from rbpf_particle import Particle
from rbpf_dataloader import DataLoader
import matplotlib.pyplot as plt
import cv2
import transformations as tf
import tqdm
import update_models as models
import copy
import utils

class SLAM():
    
    def __init__(self, data_path, mov_cov, num_p = 20, map_resolution = 0.05, map_dimension = 20, Neff_thresh = 0.6):
        
        self.num_p_ = num_p
        self.Neff_ = 0
        self.Neff_thresh_ = Neff_thresh
        self.weights_ = np.ones(num_p)/num_p
        self.mov_cov_ = mov_cov
        
        self.particles_ = []
        for i in range(self.num_p_):
            self.particles_.append(Particle(map_dimension, map_resolution, num_p))
        
        self.data_ = DataLoader(data_path[0], data_path[1], data_path[2])
        
        
    def _resample(self):
        
        c = self.weights_[0]
        j = 0
        u = np.random.uniform(0,1.0/self.num_p_)
        new_particles = []
        for k in range(self.num_p_):
            beta = u + float(k)/self.num_p_
            while beta > c:
                j += 1
                c += self.weights_[j]
            new_particles.append(copy.deepcopy(self.particles_[j]))
        
        self.weights_ = np.ones(self.num_p_)/self.num_p_     
        self.particles_ = new_particles
     
    def _run_slam_simple(self, t0, t_end = None):
        '''
            Performs SLAM
        '''
        t_end = self.data_.lidar_['num_data'] if t_end is None else t_end + 1
        resample_num = 0
        Neff_curve = []
        
        for t in range(t0, t_end):                     
        # for t in tqdm.tqdm(range(t0, t_end)):                     
            if t == t0:
                print("----Building first map----")
                for p in self.particles_:
                    p._build_first_map(self.data_, t)
                    # self._gen_map(p)
                continue
            
            correlation = np.zeros(self.num_p_)
            
            for i,p in enumerate(self.particles_):
                
                # predict with motion model
                pred_pose, pred_with_noise = p._predict(self.data_, t, self.mov_cov_)
                est_pose = pred_with_noise
                # sucess, scan_match_pose =  p._scan_matching(self.data_, t , pred_with_noise)
                
                # if not sucess:
                #     # use motion model for pose estimate
                #     est_pose = pred_with_noise
                # else:
                #     est_pose = scan_match_pose
                    
                correlation[i] = p._get_lidar_map_correspondence(self.data_, t, est_pose)               
                p._update_map(self.data_, t, est_pose)
            
            self.weights_ = utils.update_weights(correlation, self.weights_)             
            self.Neff_ = 1.0/np.sum(np.dot(self.weights_,self.weights_))
            
            print('t: ',t, ' Neff: ',self.Neff_, ' condition: ',self.Neff_thresh_*self.num_p_, 'resamples: ', resample_num)
            # with np.printoptions(precision=2):
            #     print(' weight: ', self.weights_)
            
            Neff_curve.append(self.Neff_/self.num_p_)
            
            if self.Neff_ < self.Neff_thresh_*self.num_p_:
                resample_num += 1
                self._resample() 
            
            # if t%20 ==0:
            #     self._gen_map(self.particles_[np.argmax(self.weights_)], t)
            if t%20 == 0:
                for i in range(self.num_p_):
                    self._save_map(self.particles_[i], t, i)
                plt.plot(Neff_curve)
                plt.show()
                
        
        plt.plot(Neff_curve)
        plt.show()
        
        
    def _run_slam(self, t0, t_end = None):
        '''
            Performs SLAM
        '''
        t_end = self.data_.lidar_['num_data'] if t_end is None else t_end + 1
        
        for t in range(t0, t_end):                     
        # for t in tqdm.tqdm(range(t0, t_end)):                     
            if t == t0:
                print("----Building first map----")
                for p in self.particles_:
                    p._build_first_map(self.data_, t)
                    # self._gen_map(p)
                continue
            
            scan = self.data_.lidar_['scan'][t]
                
            for i,p in enumerate(self.particles_):
                
                # predict with motion model
                pred_pose, pred_with_noise = p._predict(self.data_, t, self.mov_cov_)
                
                sucess, scan_match_pose =  p._scan_matching(self.data_, t , pred_pose)
                
                if not sucess:
                    # use motion model for pose estimate
                    est_pose = pred_with_noise
                    p.weight_ = p.weight_ * models.measurement_model(scan, pred_with_noise, self.data_.lidar_angles_, p.occupied_pts_.T)
                    self.weights_[i] = p.weight_
                
                else:
                    # sample around scan match pose
                    sample_poses = p._sample_poses_in_interval(scan_match_pose)
                    est_pose = p._compute_new_pose(self.data_, t, sample_poses)
                    self.weights_[i] = p.weight_
                
                p._update_map(self.data_, t, est_pose)
            
            self.Neff = 1/np.linalg.norm(self.weights_)
            # if self.Neff < self.Neff_thresh_:
            #     self._resample() 

            self._gen_map(self.particles_[np.argmax(self.weights_)])
    
    
    def _mapping_with_known_poses(self, t0, t_end = None, interval = 1):
        '''
            Uses noiseless odom data to generate entire map
        '''
        t_end = self.data_.lidar_['num_data'] if t_end is None else t_end + 1
        p = self.particles_[0]
        for t in range(t0, t_end, interval):                             
            odom = self.data_._odom_at_lidar_idx(t)  
            p._update_map(self.data_, t, odom)
            if t%50==0:
                self._gen_map(p)
            print(t)                
        self._gen_map(p)
                
            
    def _gen_map(self, particle, t):
        '''
            Generates map for visualization
        '''
        log_odds      = particle.log_odds_
        logodd_thresh = particle.logodd_thresh_
        MAP = particle.MAP_
        traj = particle.traj_indices_
        
        MAP_2_display = 255*np.ones((MAP['sizex'],MAP['sizey'],3),dtype=np.uint8)
        wall_indices = np.where(log_odds > logodd_thresh)
        MAP_2_display[list(wall_indices[0]),list(wall_indices[1]),:] = [0,0,0]
        unexplored_indices = np.where(abs(log_odds) < 1e-1)
        MAP_2_display[list(unexplored_indices[0]),list(unexplored_indices[1]),:] = [150,150,150]
        MAP_2_display[traj[0],traj[1]] = [70,70,228]
        # plt.imshow(MAP_2_display)
        # plt.title(str(t))
        # plt.show()
        # cv2.imwrite('logs/map.png', MAP_2_display)
        return MAP_2_display
    
    def _save_map(self, particle, t, p_num):
        MAP = self._gen_map(particle, t)
        file_name = 'logs/t_'+ str(t)+'_p_'+str(p_num)+'.png'
        cv2.imwrite(file_name, MAP)
        
    def _disp_map(self, particle, t, p_num):
        MAP = self._gen_map(particle, t)
        plt.imshow(MAP)
        plt.title(str(t)+"_p: "+str(p_num))
        plt.show()
        
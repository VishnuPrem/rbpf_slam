import math
import numpy
import numpy as np
from math import cos, sin
from copy import deepcopy

def twoDSmartPlus(x1,x2,type='pose'):
    """Return smart plus of two poses in order (x1 + x2)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = x1[2]
    R_theta1 = twoDRotation(theta1)
    # print '------ Rotation theta1:', R_theta1
    theta2 = x2[2]
    sum_theta = theta2 + theta1
    p1 = x1[0:2]
    p2 = x2[0:2]
    # print 'p2:', p2
    trans_of_u = p1 + np.dot(R_theta1, p2)
    # print '------ transition of u:', trans_of_u-p1
    if type=='pose':
        return np.array([trans_of_u[0], trans_of_u[1],sum_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(sum_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])

def twoDSmartMinus(x2,x1,type='pose'):
    """Return smart minus of two poses in order (x2 - x1)as defined in particle filter
    :param
    x1,x2: two poses in form of (x,y,theta)
    type:  which type of return you choose. 'pose' to return (x,y,theta) form
                                            ' rot' to return transformation matrix (3x3)
    """
    theta1 = x1[2]
    R_theta1 = twoDRotation(theta1)
    theta2 = x2[2]
    delta_theta = theta2 - theta1
    p1 = x1[0:2]
    p2 = x2[0:2]
    trans_of_u = np.dot(R_theta1.T, (p2-p1))
    if type=='pose':
        return np.array([trans_of_u[0], trans_of_u[1],delta_theta])
    # if type == 'rot'
    rot_of_u = twoDRotation(delta_theta)
    return np.array([[rot_of_u[0,0],rot_of_u[0,1],trans_of_u[0]],\
                     [rot_of_u[1,0],rot_of_u[1,1],trans_of_u[1]],\
                     [0            ,   0         ,   1]])


def twoDRotation(theta):
    """Return rotation matrix of rotation in 2D by theta"""
    return np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

def twoDTransformation(x,y,theta):
    """Return transformation matrix of rotation in 2D by theta combining with a translation
    (x, y)"""
    return np.array([[cos(theta), -sin(theta), x],[sin(theta), cos(theta), y],[0,0,1]])
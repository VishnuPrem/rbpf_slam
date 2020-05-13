#!pip install -U opencv-contrib-python==3.4.0.12
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# %matplotlib inline


def compute_homography(X, Y):	

  A = np.zeros((8,9))
  for i in range(4):
      A[2*i][0:2] = - X[i][0:2]
      A[2*i][2] = -1
      A[2*i][6] = X[i][0]*Y[i][0]
      A[2*i][7] = X[i][1]*Y[i][0]
      A[2*i][8] = Y[i][0]
      A[2*i+1][3:5] = - X[i][0:2]
      A[2*i+1][5] = -1
      A[2*i+1][6] = X[i][0]*Y[i][1]
      A[2*i+1][7] = X[i][1]*Y[i][1]
      A[2*i+1][8] = Y[i][1]
  u,s,vh = np.linalg.svd(A)
  h = vh[len(vh)-1,:]
  H = np.reshape(h,(3,3),'C')
  H = H/H[2][2]
  #print(np.linalg.norm(H))

  return H

def match_features(f1,f2):

  
  dist = cdist(f2, f1, metric = 'euclidean')
  dist1_sort = np.argsort(dist, axis = 1)  
  match = np.empty((0, 2), dtype = int)
  match_fwd = np.empty((0, 2), dtype = int)  
  fmap1 = np.take_along_axis(dist, dist1_sort, axis = 1)
  dist_ratio = fmap1[:,0]/fmap1[:,1]

  for i in range(0, f1.shape[0]):
    if dist_ratio[j] <= 0.7:
      match_fwd = np.vstack([match_fwd, np.array([i, dist1_sort[i, 0]])])


  dist2_sort = np.argsort(dist, axis = 1)
  fmap2 = np.take_along_axis(dist, dist2_sort, axis = 1)
  match_bkwd = np.empty((0, 2), dtype = int)
  dist_ratio = fmap2[:,0]/fmap2[:,1]
  for j in range(0, f2.shape[0]):
    if dist_ratio[j] <= 0.7:
      match_bkwd = np.vstack([match_bkwd, np.array([dist2_sort[j, 0], j])])

  for m in range(match_fwd.shape[0]):
    for n in range(match_bkwd.shape[0]):
      if np.array_equal(match_fwd[m], match_bkwd[n]):
        match = np.vstack([match, np.array([match_fwd[m, 0], match_fwd[m, 1]])])

  return match, match_fwd, match_bkwd

def ransac_homography(p1, p2):
  
  iteration = 8 #tunable parameter

  H = np.empty(iteration, dtype = object)
  best_H = np.eye(3)
  inlier = 0
  max_inlier = 0

  for k in range(iteration):
    sample_indexes = np.random.choice(p1.shape[0], 4, replace = False)
    p1_indices = p1[sample_indices,:]
    p2_indices = p2[sample_indices,:]
    p2_homo = np.c_[p2, np.ones((p1.shape[0], 1))].T 
    H = compute_homography(p1_indices, p2_indices)
    p2_transposed = np.matmul(H, p2_homo)
    p2_transposed = p2_transposed[:2]/p2_transposed[2]
    dist = np.linalg.norm(p2_transposed - p1)
    inliers = np.sum(dist < 0.7)
    if inlier > max_inlier:
      best_H = H

  return best_H

def plot_corr(I1, I2, p1, p2):

  I = np.hstack((I1, I2))
  sy,sx = I1.shape[0:2]

  plt.figure()
  plt.imshow(I)
  plt.plot(p1[:, 0], p1[:, 1],'bo')
  plt.plot(sx + p2[:, 0], p2[:, 1],'rx')
  plt.plot(np.c_[p1[:, 0], sx + p2[:, 0]].T, np.c_[p1[:, 1],p2[:, 1]].T, 'g-')
  plt.show()

def stitch(I1, I2, H):

    eps = 1e-7
    sy1, sx1, sz1 = I1.shape
    sy2, sx2, sz2 = I2.shape
    x2, y2 = np.meshgrid(np.arange(sx2), np.arange(sy2))
    # map I2 to I1
    p1_hat = H @ np.r_[x2.ravel(), y2.ravel(), np.ones_like(x2.ravel())].reshape(3, -1)
    p1_hat = p1_hat[0:2] / (p1_hat[2] + eps)

    # create new dimensions to accomodate points from I2
    p1_hat_xmax = np.max(p1_hat[0])
    p1_hat_xmin = np.min(p1_hat[0])
    p1_hat_ymax = np.max(p1_hat[1])
    p1_hat_ymin = np.min(p1_hat[1])

    xmin = np.rint(np.floor(np.minimum(p1_hat_xmin, 0))).astype(np.int32)
    xmax = np.rint(np.ceil(np.maximum(p1_hat_xmax, sx2))).astype(np.int32)
    ymin = np.rint(np.floor(np.minimum(p1_hat_ymin, 0))).astype(np.int32)
    ymax = np.rint(np.ceil(np.maximum(p1_hat_ymax, sy2))).astype(np.int32)

    # create images for mapping
    I1_ = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
    I2_ = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
    I_ = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)

    # I1 is just translated in I_
    I1_[-ymin:sy1 - ymin, -xmin:sx1 - xmin, :] = I1[:min(sy1, ymax), :min(sx1, xmax), :]

    # map I_ to I2 (translation then homography)
    sy2_, sx2_, sz2_ = I2_.shape
    x2_, y2_ = np.meshgrid(np.arange(sx2_), np.arange(sy2_))
    p2_hat = np.linalg.inv(H) @ np.r_[x2_.ravel() + xmin, y2_.ravel() + ymin, np.ones(x2_.size)].reshape(3, -1)
    p2_hat = np.rint(p2_hat[0:2] / (p2_hat[2] + eps)).astype(np.int32)

    # keep only the valid coordinates of I2
    good_x = np.logical_and(p2_hat[0, :] >= 0, p2_hat[0, :] < sx2)
    good_y = np.logical_and(p2_hat[1, :] >= 0, p2_hat[1, :] < sy2)
    good = np.logical_and(good_x, good_y)


    # I2 transformed by homography in I_
    I2_[y2_[good.reshape(x2_.shape)], x2_[good.reshape(x2_.shape)]] = I2[p2_hat[1, good], p2_hat[0, good]]

    # nonoverlapping regions do not require blending
    I2_sum = np.sum(I2_, axis=2)
    I1_sum = np.sum(I1_, axis=2)

    # in no blend area, one of I1_ and I2_ is all 0 
    no_blend_area = np.logical_or(I2_sum == 0, I1_sum == 0)
    I_[no_blend_area] = I2_[no_blend_area] + I1_[no_blend_area]

    # in blend area, take the average of I1_ and I2_
    blend_area = np.logical_and(I2_sum > 0, I1_sum > 0)
    I_[blend_area] = (I2_[blend_area] * .5 + I1_[blend_area] * .5).astype(np.uint8)
    return I_

# load images in OpenCV BGR format
I1 = cv2.imread('image3_1.jpeg')
I2 = cv2.imread('image3_2.jpeg')

# create grayscale images
I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY);

# convert images to RGB format for display
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

# compute SIFT features
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(I1_gray, None)
kp2, des2 = sift.detectAndCompute(I2_gray, None)

# match features
match, match_fwd, match_bkwd = match_features(des1, des2)

# get corresponding points p1, p2 
p1 = np.array([kp.pt for kp in kp1])[match[:, 0]]
p2 = np.array([kp.pt for kp in kp2])[match[:, 1]]

# plot first 20 matching points 
plot_corr(I1, I2, p1[::40], p2[::40])

# estimate homography transform with RANSAC
H = ransac_homography(p1, p2)

# stitch two images together and show the results
I = stitch(I1,I2,H)
plt.figure()
plt.imshow(I)
plt.show()
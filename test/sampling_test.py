import numpy as np
import copy

particles_ = [1,    2,  3,  4]
weights_ =   np.array([1,2,3,4])
weights_ = weights_/np.sum(weights_)

num_p_ = len(particles_) 
new_paricles_ = []
    
c = weights_[0]
j = 0
u = np.random.uniform(0,1.0/num_p_)
new_particles = []
for k in range(num_p_):
    beta = u + float(k)/num_p_
    while beta > c:
        j += 1
        c += weights_[j]
    new_particles.append(copy.deepcopy(particles_[j]))

weights_ = np.ones(num_p_)/num_p_
# print('Weights: ', self.weights_) 
print(new_particles)
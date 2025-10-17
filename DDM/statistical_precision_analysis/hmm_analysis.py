#######################
### Import packages ###
#######################

import matplotlib.pyplot as plt
import sys
import time
from IPython.core import ultratb
import dill
import numpy as np
from tqdm import tqdm
import scipy.optimize as opt
import noisyopt 
from synthetic_data_generation_functions import *
from synthetic_data_analysis_functions import *
from matplotlib import animation
import time
sys.excepthook = ultratb.FormattedTB(call_pdb=False)

plt.style.use('paper.mplstyle')

# Time counter
start_time = time.time()

##################
### Parameters ###
##################

n_simulations_list = [20, 60, 100, 200, 600]
n_simulations = 100

################
### Analysis ###
################

with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}_fulltraining.pkl', 'rb') as file:
    model = dill.load(file)

with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{5000}_fulltraining.pkl', 'rb') as file:
    model = dill.load(file)

# fig, ax = plt.subplots()


transmat = model.transmat_
emission_vect = model.emissionprob_[:,1]
mat = transmat
sorted_indexes = np.argsort(emission_vect)
vector = np.ones([len(transmat),1])/len(transmat)

# for i in range(5):

#     mat = np.matmul(transmat,mat)

#     # vector = np.matmul(transmat,vector)

#     # ax.clear()
#     # ax.imshow(vector, vmin=0, vmax=1)
#     ax.imshow(mat, vmin=0, vmax=1)
#     plt.show()

#     # time.sleep(1)

###################

# for i in range(500):

#     mat = np.matmul(transmat,mat)
    # print(mat)

# new_mat = np.array([mat[:,sorted_indexes[i]] for i in len(sorted_indexes)])

##

temp_transmat = []

for i in sorted_indexes:
    temp_transmat.append(transmat[i,:])

temp_transmat = np.array(temp_transmat)

new_transmat = []

for i in sorted_indexes:
    new_transmat.append(temp_transmat[:,i])

new_transmat = np.transpose(new_transmat)

##

new_mat = new_transmat

for i in range(500):

    new_mat = np.matmul(new_mat,new_transmat)


##

new_emissionmat = []

for i in sorted_indexes:
    new_emissionmat.append(model.emissionprob_[i,:])

new_emissionmat = np.array(new_emissionmat)

##

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,2)
ax1 = plt.subplot(row[0,0])
ax2 = plt.subplot(row[0,1])

states_number = len(model.transmat_)

ax1.imshow(new_transmat, vmin=0, vmax=1)
ax1.set_xticks(np.arange(states_number))
ax1.set_yticks(np.arange(states_number))

ax1.set_title(f'Transition matrix, {n_simulations} simulations', fontsize=7)
ax1.set_xlabel('To state')
ax1.set_ylabel('From state')

ax2.imshow(new_mat, vmin=0, vmax=1)
ax2.set_xticks(np.arange(states_number))
ax2.set_yticks(np.arange(states_number))

ax2.set_title(f'Final matrix, {n_simulations} simulations', fontsize=7)
ax2.set_xlabel('To state')
ax2.set_ylabel('From state')

#

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,1)
ax = plt.subplot(row[:])

# proba_dist = new_mat[0] #np.sort(new_mat[0])
# proba_dist = np.sort(new_mat[0])
# proba_dist = new_mat[0]*new_emissionmat[:,1]
proba_dist = new_emissionmat[:,1]

print(np.std(proba_dist))

ax.plot(proba_dist)
ax.set_xlabel('State')
ax.set_ylabel('Proba to chose 1')

#

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,1)
ax = plt.subplot(row[:])

ax.imshow(new_emissionmat)
ax.set_xticks([0,1], labels=[0,1], rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks(np.arange(states_number))
ax.set_title(f'Emission matrix, {n_simulations} simulations', fontsize=7)
ax.set_xlabel('Choice')
ax.set_ylabel('State')

#

fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
gs = fig.add_gridspec(1, 1)
row = gs[0].subgridspec(1,1)
ax = plt.subplot(row[:])

print('res=',np.matmul(new_mat,new_emissionmat[:,1]))

new_mat_i = new_transmat
res_list = []

x = np.arange(500)

for i in x:

    new_mat_i = np.matmul(new_mat_i,new_transmat)
    res = np.matmul(new_mat_i,new_emissionmat[:,1])
    res2 = np.matmul(new_mat_i,np.ones(len(transmat))/len(transmat))
    
    # print(res)
    
    res_list.append(res)
    # ax.scatter(i,res[2], c='blue', marker='+')

res_array = np.array(res_list)

for j in range(len(new_emissionmat)):

    ax.plot(x, res_array[:,j])

# ax.scatter(x,0.88 - (0.22)/np.sqrt(x), c='red', marker='+', alpha=0.5)
    # ax.scatter(i,res2[0], c='red')

steps_number = 500
noise_amplitude = 0.1
# delta = 0.05
drift = 0.0
p_a = 0.5
p_a_reward = 1

steps = np.arange(steps_number)

delta_range = [0.03,0.04,0.05,0.06,0.07] #np.linspace(0.01,0.1,10)

for delta in tqdm(delta_range):

   mean_trajectory = compute_simulations_average(p_a, p_a_reward, steps_number, noise_amplitude, delta, drift, n_simulations=5000)

   ax.plot(steps, mean_trajectory, alpha=0.5, linestyle='--')
   ax.text(steps[-1],mean_trajectory[-1], f'drift = {np.round(delta,3)}', fontsize=5)

# ax.imshow(np.matmul(new_mat,new_emissionmat[:,1]))
# ax.set_xticks([0,1], labels=[0,1], rotation=30, ha="right", rotation_mode="anchor")
# ax.set_yticks(np.arange(states_number))
# ax.set_title(f'Final matrix * Emission vector, {n_simulations} simulations', fontsize=7)
# ax.set_xlabel('Choice')
# ax.set_ylabel('State')

plt.show()

###################

# im = plt.imshow(vector, vmin=0, vmax=1)

# def init():
#     im.set_data(np.zeros((ncols, nrows)))

# def animate(i):
#     xi = i // nrows
#     yi = i % nrows
#     vector[xi, yi] = np.matmul(transmat,vector)

#     im.set_data(vector)
#     return im

# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nrows * ncols,
#                                interval=50)


# def update(frame):

#     image.clear()
#     image = ax.imshow(vector[frame])

# for i in range(10):



quit()

############
### Plot ###
############

### Action and transition matrixes ###
for n_simulations in n_simulations_list:

    with open(f'DDM/statistical_precision_analysis/simulations_batches/best_model_score_{n_simulations}.pkl', 'rb') as file:
        model = dill.load(file)

    fig=plt.figure(figsize=(3.5, 3), dpi=300, constrained_layout=False, facecolor='w')
    gs = fig.add_gridspec(1, 1)
    row = gs[0].subgridspec(1,2)
    ax1 = plt.subplot(row[0,0])
    ax2 = plt.subplot(row[0,1])

    states_number = len(model.transmat_)

    ax1.imshow(model.transmat_)
    ax1.set_xticks(np.arange(states_number))
    ax1.set_yticks(np.arange(states_number))

    ax1.set_title(f'Transition matrix, {n_simulations} simulations', fontsize=7)
    ax1.set_xlabel('To state')
    ax1.set_ylabel('From state')


    ax2.imshow(model.emissionprob_)
    ax2.set_xticks([0,1], labels=[0,1], rotation=30, ha="right", rotation_mode="anchor")
    ax2.set_yticks(np.arange(states_number))
    ax2.set_title(f'Emission matrix, {n_simulations} simulations', fontsize=7)
    ax2.set_xlabel('Choice')
    ax2.set_ylabel('State')


plt.show()







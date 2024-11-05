"""
Perform simulation with pure epsilon-greedy agent
Not parallelized, runs only on CPU
"""

# Include path of RL_Playground to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[1])
sys.path.append(root_path)

import time

import numpy as np
import numpy.random as np_rand
import matplotlib.pyplot as plt

from rl_env import MAB
 

# Config
# =================================================================================================
'''
n_arm : number of actions
n_run : number of simulation runs
stationary : is environment stationary mab
n_phase : number of phase
          true action value(mean) will be changed (n_phase - 1) times in simulation
          n_phase = 1 for stationary environment,
          n_phase = larger than 1 for non-stationary environment
n_step_per_phase : number of steps in a phase.
                   n_step = n_phase * n_step_per_phase
mean_range : range of true action value
             true action value will be chosen in this range randomly
std_range : range of standard deviation of reward
            std will be chosen in this range randomly
np_rand_seed : numpy random seed
'''


n_arm = 10

n_run = 10000

stationary = True
n_phase = 1
n_step_per_phase = 1600

mean_range = (0, 1)
std_range = (0.1, 0.5)

np_rand_seed = 0


# Run Simulation
# =================================================================================================
def get_values_from_range(rng):
    res = []
    for _ in range(n_phase):
        res.append(np_rand.uniform(low=rng[0], high=rng[1], size=n_arm))
        
    return np.array(res, dtype=np.float32)


np_rand.seed(np_rand_seed)    

mean = get_values_from_range(mean_range)
std = get_values_from_range(std_range)

n_step = n_phase * n_step_per_phase

agent = MAB.CustomAgent(n_arm=n_arm, init_q=0.0,
                        act_sel=MAB.ActionSelection.EPSILON_GREEDY,
                        update_method=MAB.UpdateMethod.AVERAGE,
                        eps=0.1)

st_tm = time.time()
res = MAB.run_sim(n_run=n_run, n_arm=n_arm, n_step=n_step,
                  mean=mean, std=std, agent=agent,
                  stationary=stationary, n_step_per_phase=n_step_per_phase)
en_tm = time.time()

run_tm = en_tm - st_tm
print('Run Time : {:d}m {:02d}s'.format(int(run_tm // 60), int(run_tm % 60)))

q_hist = res[0]
r_hist = res[1]
is_optim_hist = res[2]

# Plotting Graph
# =================================================================================================
width_px = 3000
height_px = 1000

fig_dpi = 100
px = 1 / fig_dpi

steps = np.arange(1, n_step + 1)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(width_px * px, height_px * px), dpi=fig_dpi)
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.12, wspace=0.2)

ax[0].plot(steps, q_hist)
ax[0].set_ylabel('Action Value', fontsize=20)

ax[1].plot(steps, r_hist, color='#000000')
ax[1].set_ylabel('Reward', fontsize=20)

ax[2].plot(steps, is_optim_hist, color='#000000')
ax[2].set_ylabel('% Optimal Action', fontsize=20)

for idx in range(3):
    ax[idx].grid(True, linestyle='--')
    ax[idx].set_xlabel('Step', fontsize=20)
    ax[idx].margins(0.05)
    ax[idx].set_ylim(bottom=-0.05)
    ax[idx].tick_params(axis='both', labelsize=15)

plt.savefig('run_example_cpu.png')

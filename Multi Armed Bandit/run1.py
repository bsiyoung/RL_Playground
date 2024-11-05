# Include path of RL_Playground to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[1])
sys.path.append(root_path)

import copy
import time

import numpy as np
import numpy.random as np_rand
import matplotlib.pyplot as plt

from rl_env.MAB import cuda_mab


# Constants (will not change)
# =================================================================================================
n_arm = 10
n_run = 5000

mean_range = (0, 1)
std_range = (0.1, 0.5)

np_rand_seed = 100
cuda_rand_seed = 100

n_block = 128
n_th_in_blk = 64

# Run Simulation
# =================================================================================================
np_rand.seed(np_rand_seed)


def run():
    n_test = 100  # How many epsilon/conf_lvl values will be tested
    n_step = 20  # How many steps will be plotted on figure (= number of lines in fig)
    
    mean = get_values_from_range(mean_range)
    std = get_values_from_range(std_range)
    
    # Steps which will be plotted on figure
    step_list = np.logspace(10, 13, num=n_step, base=2, dtype=np.int32)
    print('Steps :', step_list, '\n')
    
    # Dictionary contains optimum action percentage data
    optim_act_res = {}
    optim_act_res_buf = np.zeros((len(step_list), n_test), dtype=np.float32)
    
    # 1. Pure Epsilon Greedy
    # ===============================================================================================
    print('Pure Epsilon Greedy')
    
    eps_list = np.logspace(-5, 0, num=n_test, base=10)
    st_tm = time.time()
    for eps_idx, eps in enumerate(eps_list):
        agent = cuda_mab.CustomAgent(act_sel=cuda_mab.ActionSelection.EPSILON_GREEDY,
                                     update_method=cuda_mab.UpdateMethod.AVERAGE,
                                     init_q=0.0, eps=eps)
        
        res = run_sim_simple(agent, step_list[-1], mean, std)
        optim_act_res_buf[:, eps_idx] = res[2][step_list-1]
    
        print_progress(st_tm, n_test, eps_idx + 1)
    print('\n')
    optim_act_res['pure_eps_greedy'] = copy.deepcopy(optim_act_res_buf)
    
    # 2. Pure Upper Confidence Boundary (UCB)
    # ===============================================================================================
    print('Pure UCB')
    c_lvl_list = np.logspace(-2, 2, num=n_test, base=10)
    st_tm = time.time()
    for idx, conf_lvl in enumerate(c_lvl_list):
        agent = cuda_mab.CustomAgent(act_sel=cuda_mab.ActionSelection.UCB,
                                     update_method=cuda_mab.UpdateMethod.AVERAGE,
                                     init_q=0.0, conf_lvl=conf_lvl)
        
        res = run_sim_simple(agent, step_list[-1], mean, std)
        optim_act_res_buf[:, idx] = res[2][step_list-1]
    
        print_progress(st_tm, n_test, idx + 1)
    print('\n')
    optim_act_res['pure_ucb'] = copy.deepcopy(optim_act_res_buf)
    
    # Plot figures
    # ===============================================================================================
    plot_fig(optim_act_res, 'pure_eps_greedy', eps_list, 'Epsilon')
    plot_fig(optim_act_res, 'pure_ucb', c_lvl_list, 'Confidence Level')
    plot_figs(optim_act_res,
              ['pure_eps_greedy', 'pure_ucb'],
              [eps_list, c_lvl_list],
              np.array([(31, 119, 180), (255, 127, 14)]) / 255.0,
              ['Pure ε-Greedy', 'Pure UCB'],
              'ε, Confidence Level',
              'pure_eps_and_ucb.png')


def get_values_from_range(rng, n_phase=1):
    res = []
    for _ in range(n_phase):
        res.append(np_rand.uniform(low=rng[0], high=rng[1], size=n_arm))
        
    return np.array(res, dtype=np.float32)


def run_sim_simple(agent, n_step_per_phase, mean, std, stationary=True, n_phase=1, verbose=False):
    '''
    Simple version of cuda_mab.run_sim
    '''
    n_step = n_phase * n_step_per_phase
    
    res = cuda_mab.run_sim(n_block, n_th_in_blk, n_run, n_arm, n_step, 
                           mean, std, agent, stationary, n_step_per_phase,
                           cuda_rand_seed, verbose=verbose)
    
    return res


def print_progress(st_tm, n_test, curr_test):
    elapsed_tm = time.time() - st_tm
    total_tm = n_test / curr_test * elapsed_tm
    progress = curr_test / n_test * 100
    
    print('\r{}/{} ({:.3f}%)  {}m {:02d}s / {}m {:02d}s'
          .format(curr_test, n_test, progress,
                  int(elapsed_tm // 60), int(elapsed_tm % 60),
                  int(total_tm // 60), int(total_tm % 60)),
          end='')


def plot_fig(optim_act_res, dict_key, x_data, x_label):
    plot_figs(optim_act_res, [dict_key], [x_data], [(0, 0, 0)], [''], x_label, dict_key + '.png')
    
    
def plot_figs(optim_act_res, dict_keys, x_datas, colors, labels, x_label, file_name):
    n_step = len(optim_act_res[dict_keys[0]])
    
    dpi = 150
    px = 1 / dpi
    plt.figure(figsize=(1200 * px, 900 * px), dpi=dpi)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.12, top=0.95, right=0.95)
        
    plt.grid(True, axis='y', which='major', linestyle='-')
    plt.grid(True, axis='x', which='major', linestyle='-')
    plt.grid(True, axis='x', which='minor', linestyle='-')
    
    plt.xlabel(x_label)
    plt.xscale('log')
    
    plt.ylabel('% Optimal Action')
    plt.yticks(np.arange(0.0, 1.05, 0.1))
    plt.ylim(bottom=-0.05, top=1.05)
    
    for key_idx, key in enumerate(dict_keys):
        for res_idx, res in enumerate(optim_act_res[key]):
            plt.plot(x_datas[key_idx], res, color=(*colors[key_idx], (res_idx+1)/n_step),
                     label=labels[key_idx] if (res_idx + 1 == n_step) else '')
    
    if len(dict_keys) > 1:
        plt.legend()
    
    plt.savefig('run_plot/' + file_name)
    plt.clf()


if __name__ == '__main__':
    run()


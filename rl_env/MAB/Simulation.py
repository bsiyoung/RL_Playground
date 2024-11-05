from numpy.typing import NDArray

import time
    
import numpy as np
import numpy.random as np_rand

from .Agent import BaseAgent


def _run_sim(n_arm: int, n_step: int,
             mean: NDArray[np.float32], std: NDArray[np.float32],
             agent: BaseAgent,
             stationary: bool=True, n_step_per_phase: int=1):
    q_hist = np.zeros((n_step, n_arm), dtype=np.float32)
    r_hist = np.zeros(n_step, dtype=np.float32)
    is_optim_hist = np.full(n_step, False, dtype=np.bool_)
    
    q_hist[0] = agent.q
    curr_mean, curr_std = mean[0], std[0]
    curr_optim = int(np.argmax(curr_mean))
    for curr_step in range(1, n_step + 1):
        agent.on_step_begin()
        
        action = agent.select_action()
        reward = np_rand.normal(curr_mean[action], curr_std[action])
        agent.update_q(reward, action)
        
        q_hist[curr_step-1] = agent.q
        r_hist[curr_step-1] = reward
        is_optim_hist[curr_step-1] = (action == curr_optim)
        
        agent.on_step_end()
        
        if stationary is True:
            continue
        
        if curr_step % n_step_per_phase != 0 or curr_step == n_step:
            continue
        
        phase = curr_step // n_step_per_phase + 1
        if len(mean) < phase or len(std) < phase:
            continue
        
        curr_mean, curr_std = mean[phase-1], std[phase-1]
        curr_optim = int(np.argmax(curr_mean))
        
    return q_hist, r_hist, is_optim_hist


def run_sim(n_run: int, n_arm: int, n_step: int,
            mean: NDArray[np.float32], std: NDArray[np.float32],
            agent: BaseAgent,
            stationary: bool=True, n_step_per_phase: int=1,
            average: bool=True,
            verbose: bool=True, verbose_freq: int=50):
    q_hist = np.zeros((n_run, n_step, n_arm), dtype=np.float32)
    r_hist = np.zeros((n_run, n_step), dtype=np.float32)
    is_optim_hist = np.full((n_run, n_step), False, dtype=np.bool_)
    
    st_tm = time.time()
    
    for curr_run in range(1, n_run + 1):
        agent.reset()
        
        res = _run_sim(n_arm, n_step, mean, std, agent, stationary, n_step_per_phase)
        
        q_hist[curr_run-1] = res[0]
        r_hist[curr_run-1] = res[1]
        is_optim_hist[curr_run-1] = res[2]
        
        if verbose is True and curr_run % verbose_freq == 0:
            progress = curr_run / n_run * 100
            
            curr_tm = time.time()
            elapsed_tm = curr_tm - st_tm
            est_tm = n_run / curr_run * elapsed_tm
            
            print('\r{}/{} ({:.3f}%%)  {:d}m {:02d}s / {:d}m {:02d}s'
                  .format(curr_run, n_run, progress,
                          int(elapsed_tm // 60), int(elapsed_tm % 60),
                          int(est_tm // 60), int(est_tm % 60)),
                  end='')
        
    if verbose is True:
        print()
        
    if average is True:
        q_hist = np.mean(q_hist, axis=0)
        r_hist = np.mean(r_hist, axis=0)
        is_optim_hist = np.mean(is_optim_hist, axis=0)
        
    return q_hist, r_hist, is_optim_hist
        
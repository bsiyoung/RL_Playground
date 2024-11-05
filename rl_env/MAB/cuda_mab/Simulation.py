# Include cuda_mab folder to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[0])
sys.path.append(root_path)

from typing import List
from numpy.typing import NDArray

import time
    
import numba.cuda as cuda
import numba.cuda.random as cu_rand

import numpy as np

import Agent
import cuda_util


@cuda.jit
def _run_sim(q_hist, r_hist, is_optim_hist, q_buf, idx_buf, cnt_action_buf,
             mean, std, agent, conf, stationary, n_step_per_phase, rng_state, progress_cnt):
    th_id = cuda.grid(1)  # type: ignore (vscode) # = cuda.threadIdx + cuda.blockDim * cuda.blockIdx
    stride = cuda.gridsize(1)  # type: ignore (vscode) # = cuda.blockDim * cuda.gridDim
    
    n_runs = len(q_hist)
    n_step = len(q_hist[0])
    
    sel_act, update_method = agent[0], agent[1]
    init_q = conf[0]
    
    _q_buf = q_buf[th_id]
    _idx_buf = idx_buf[th_id]
    _cnt_action_buf = cnt_action_buf[th_id]
    
    run_cnt = 0
    for run_no in range(th_id, n_runs, stride):
        Agent.reset(_q_buf, _cnt_action_buf, init_q)
        
        curr_mean, curr_std = mean[0], std[0]
        curr_optim = cuda_util.argmax(curr_mean)
        
        for curr_step in range(1, n_step + 1):
            Agent.on_step_begin()
            
            action = Agent.select_action(sel_act, _q_buf, _idx_buf, conf, _cnt_action_buf,
                                        curr_step, th_id, rng_state)
            
            # De-normalization
            reward = cuda_util.rand_normal(th_id, rng_state) * curr_std[action] + curr_mean[action]
            
            Agent.update_q(update_method, _q_buf, action, reward, conf, _cnt_action_buf)
            
            cuda_util.copy_1d_arr(q_hist[run_no-1][curr_step-1], _q_buf)
            r_hist[run_no-1][curr_step-1] = reward
            is_optim_hist[run_no-1][curr_step-1] = (action == curr_optim)
            
            Agent.on_step_end()
            
            if stationary is True:
                continue
            
            if curr_step % n_step_per_phase != 0 or curr_step == n_step:
                continue
            
            phase = curr_step // n_step_per_phase + 1
            if len(mean) < phase or len(std) < phase:
                continue
            
            curr_mean, curr_std = mean[phase-1], std[phase-1]
            curr_optim = cuda_util.argmax(curr_mean)
            
        run_cnt += 1
        
        if run_cnt == 1000 or run_no + stride >= n_runs:
            cuda.atomic.add(progress_cnt, 0, run_cnt)  # type: ignore
            run_cnt = 0
        


def _prepare_gpu_sim(n_run, n_step, n_arm, n_block, n_th_in_blk, mean, std, agent, cuda_seed):
    # Assign memory, copy data from host to device
    d_q_hist = cuda.device_array((n_run, n_step, n_arm), dtype=np.float32)
    d_r_hist = cuda.device_array((n_run, n_step), dtype=np.float32)
    d_is_optim_hist = cuda.device_array((n_run, n_step), dtype=np.bool_)  # type: ignore (vs code)
    
    n_th = n_block * n_th_in_blk
    d_q_buf = cuda.device_array((n_th, n_arm), dtype=np.float32)
    d_cnt_action_buf = cuda.device_array((n_th, n_arm), dtype=np.int32)  # type: ignore (vs code)
    d_idx_buf = cuda.device_array((n_th, n_arm), dtype=np.int32)  # type: ignore (vs code)
    
    d_mean = cuda.to_device(mean)
    d_std = cuda.to_device(std)
    
    # Agent and configs
    agent_arr = np.array([agent.act_sel.value, agent.update_method.value], dtype=np.int8)
    conf_arr = np.array([
        agent.args['init_q'] if 'init_q' in agent.args else 0.0,
        agent.args['eps'] if 'eps' in agent.args else 0.0,
        agent.args['alpha'] if 'alpha' in agent.args else 0.0,
        agent.args['conf_lvl'] if 'conf_lvl' in agent.args else 0.0
    ], dtype=np.float32)
    
    d_agent = cuda.to_device(agent_arr)
    d_conf = cuda.to_device(conf_arr)
    
    # Create RNG states
    rng_state = cu_rand.create_xoroshiro128p_states(n_th, seed=cuda_seed)
    
    # A counter integer to calculate progress
    m_progress_cnt = cuda.mapped_array(1, dtype=np.int64)  # type: ignore (vs code)
    
    cuda.synchronize()
    
    return d_q_hist, d_r_hist, d_is_optim_hist, d_q_buf, d_idx_buf, d_cnt_action_buf, \
        d_mean, d_std, d_agent, d_conf, rng_state, m_progress_cnt


def run_sim(n_block: int, n_th_in_blk: int,
            n_run: int, n_arm: int, n_step: int,
            mean: NDArray[np.float32], std: NDArray[np.float32],
            agent: Agent.CustomAgent,
            stationary: bool=True, n_step_per_phase: int=1, cuda_seed: int=0,
            average: bool=True, verbose: bool=True):
    d_q_hist, d_r_hist, d_is_optim_hist, d_q_buf, d_cnt_action_buf, d_idx_buf, \
        d_mean, d_std, d_agent, d_conf, rng_state, m_progress_cnt = \
            _prepare_gpu_sim(n_run, n_step, n_arm, n_block, n_th_in_blk, mean, std, agent, cuda_seed)
    
    _run_sim[n_block, n_th_in_blk](d_q_hist, d_r_hist, d_is_optim_hist,  # type: ignore (vs code)
                                   d_q_buf, d_idx_buf, d_cnt_action_buf,
                                   d_mean, d_std, d_agent, d_conf,
                                   stationary, n_step_per_phase, rng_state, m_progress_cnt)

    # Print progress
    if verbose is True:
        while m_progress_cnt[0] <= n_run:
            time.sleep(0.1)
            progress = m_progress_cnt[0] / n_run * 100
            print('\r{}/{}  ({:.3f}%)'.format(m_progress_cnt[0], n_run, progress), end='')
            if m_progress_cnt == n_run:
                break
        print()
    
    cuda.synchronize()
    
    # Copy datas from device to host
    q_hist = d_q_hist.copy_to_host()
    r_hist = d_r_hist.copy_to_host()
    is_optim_hist = d_is_optim_hist.copy_to_host()
        
    if average is True:
        q_hist = np.mean(q_hist, axis=0)
        r_hist = np.mean(r_hist, axis=0)
        is_optim_hist = np.mean(is_optim_hist, axis=0)
        
    return q_hist, r_hist, is_optim_hist
        
from enum import Enum
import math

import numba.cuda as cuda

import cuda_util


class ActionSelection(Enum):
    GREEDY = 1
    EPSILON_GREEDY = 2
    UCB = 3
    
    
class UpdateMethod(Enum):
    AVERAGE = 1
    WEIGHTED_AVERAGE = 2


@cuda.jit(device=True)
def action_selection_greedy(q, idx_buf, th_id, rng_state):
    '''
    # same with
    max_q = np.max(q)
    all_max_idx = np.where(max_q == q)[0]
    action = np_rand.choice(all_max_idx)
    
    return action
    '''
    
    cnt = 1
    max_q = q[0]
    idx_buf[0] = 0
    
    for i in range(1, len(q)):
        if max_q > q[i]:
            continue
        elif max_q == q[i]:
            idx_buf[cnt] = i
            cnt += 1
        else:
            idx_buf[0] = i
            max_q = q[i]
            cnt = 1
            
    action = idx_buf[cuda_util.rand_int(cnt, th_id, rng_state)]
    return action


@cuda.jit(device=True)
def action_selection_epsilon_greedy(q, eps, idx_buf, th_id, rng_state):
    n_arm = len(q)
    
    break_tie = eps > cuda_util.rand_uniform(th_id, rng_state)
    if break_tie is True:
        return cuda_util.rand_int(n_arm, th_id, rng_state)
    
    return action_selection_greedy(q, idx_buf, th_id, rng_state)


@cuda.jit(device=True)
def action_selection_ucb(q, idx_buf, cnt_action, conf_lvl, curr_step, th_id, rng_state):
    '''
    # same with
    new_q = q + conf_lvl * np.sqrt(np.log(curr_step) / cnt_action)
    
    max_q = np.max(new_q)
    all_max_idx = np.where(max_q == new_q)[0]
    action = np_rand.choice(all_max_idx)
    
    return action
    '''
    
    def new_q(idx):
        return q[idx] + conf_lvl * math.sqrt(math.log(curr_step) / cnt_action[idx])
    
    cnt = 1
    max_q = new_q(0)
    idx_buf[0] = 0
    
    for i in range(1, len(q)):
        nq = new_q(i)
        
        if max_q > nq:
            continue
        elif max_q == nq:
            idx_buf[cnt] = i
            cnt += 1
        else:
            idx_buf[0] = i
            max_q = nq
            cnt = 1
            
    action = idx_buf[cuda_util.rand_int(cnt, th_id, rng_state)]
    return action


@cuda.jit(device=True)
def update_method_average(cnt_action, action):
    cnt = cnt_action[action]
    return 1 / (cnt if cnt != 0 else 1)


@cuda.jit(device=True)
def update_method_weighted_average(alpha):
    return alpha


@cuda.jit(device=True)
def reset(q, cnt_action, init_q):
    q[:] = init_q
    cnt_action[:] = 0
    

@cuda.jit(device=True)
def select_action(sel_act, q, idx_buf, conf, cnt_action, curr_step, th_id, rng_state):
    if sel_act == ActionSelection.GREEDY.value:
        action = action_selection_greedy(q, idx_buf, th_id, rng_state)
    elif sel_act == ActionSelection.EPSILON_GREEDY.value:
        eps = conf[1]
        action = action_selection_epsilon_greedy(q, eps, idx_buf, th_id, rng_state)
    elif sel_act == ActionSelection.UCB.value:
        conf_lvl = conf[3]
        action = action_selection_ucb(q, idx_buf, cnt_action, conf_lvl,
                                      curr_step, th_id, rng_state)
    else:
        raise ValueError('Not correct ActionSelection value')
    
    cnt_action[action] += 1
    
    return action


@cuda.jit(device=True)
def update_q(update_method, q, action, reward, conf, cnt_action):
    if update_method == UpdateMethod.AVERAGE.value:
        sz_step = update_method_average(cnt_action, action)
    elif update_method == UpdateMethod.WEIGHTED_AVERAGE.value:
        alpha = conf[2]
        sz_step = update_method_weighted_average(alpha)
    else:
        raise ValueError('Not correct UpdateMethod value')
    
    old_q = q[action]
    q[action] = old_q + sz_step * (reward - old_q)


@cuda.jit(device=True)
def on_step_begin():
    pass


@cuda.jit(device=True)
def on_step_end():
    pass


class CustomAgent:
    def __init__(self, act_sel: ActionSelection, update_method: UpdateMethod, **args) -> None:
        self.act_sel = act_sel
        self.update_method = update_method
        self.args = args

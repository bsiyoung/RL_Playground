from enum import Enum
from numpy.typing import NDArray

import numpy as np
import numpy.random as np_rand
    
    
class ActionSelection(Enum):
    GREEDY = 1
    EPSILON_GREEDY = 2
    UCB = 3
    
    
class UpdateMethod(Enum):
    AVERAGE = 1
    WEIGHTED_AVERAGE = 2


def action_selection_greedy(q: NDArray[np.float32]) -> int:
    max_q = np.max(q)
    all_max_idx = np.where(max_q == q)[0]
    action = np_rand.choice(all_max_idx)
        
    # action = int(np.argmax(q))
    return action


def action_selection_epsilon_greedy(n_arm: int, q: NDArray[np.float32], eps: float) -> int:
    break_tie = eps > np_rand.uniform(low=0, high=1)
    if break_tie is True:
        return np_rand.randint(0, n_arm)
    
    return action_selection_greedy(q)


def action_selection_ucb(curr_step: int, q: NDArray[np.float32],
                         cnt_action: NDArray[np.int32], conf_lvl: float) -> int:
    new_q = q + conf_lvl * np.sqrt(np.log(curr_step) / cnt_action)
    
    return int(np.argmax(new_q))


def update_method_average(cnt_action: NDArray[np.int32], action: int) -> float:
    cnt = cnt_action[action]
    return 1 / (cnt_action[action] if cnt != 0 else 1)


def update_method_weighted_average(alpha: float) -> float:
    return alpha


class BaseAgent:
    def __init__(self, n_arm: int, init_q: float, **args) -> None:
        self.n_arm = n_arm
        self.init_q = init_q
        self.args = args
        
        self.q = np.full(self.n_arm, self.init_q, dtype=np.float32)
        self.cnt_action = np.zeros(self.n_arm, dtype=np.int32)
        self.curr_step = 0
        
        self.reset()
        
        
    def reset(self):
        self.q[:] = self.init_q
        self.cnt_action[:] = 0
        self.curr_step = 0
        
    
    def select_action(self) -> int:
        max_q = np.max(self.q)
        all_max_idx = np.where(max_q == self.q)[0]
        action = np_rand.choice(all_max_idx)
        
        # action = int(np.argmax(self.q))
        
        self.cnt_action[action] += 1
        
        return action
    
    
    def update_q(self, reward: float, action: int) -> None:
        sz_step = 1 / self.cnt_action[action]
        old_q = self.q[action]
        self.q[action] = old_q + sz_step * (reward - old_q)
        
        
    def on_step_begin(self) -> None:
        self.curr_step += 1
        
        
    def on_step_end(self) -> None:
        pass


class CustomAgent(BaseAgent):
    def __init__(self,
                 n_arm: int=5, init_q: float=0.0,
                 act_sel: ActionSelection=ActionSelection.GREEDY,
                 update_method: UpdateMethod=UpdateMethod.AVERAGE,
                 **args) -> None:
        super().__init__(n_arm, init_q, **args)
        self.act_sel = act_sel
        self.update_method = update_method
    
        
    def select_action(self) -> int:
        if self.act_sel == ActionSelection.GREEDY:
            action = action_selection_greedy(self.q)
        elif self.act_sel == ActionSelection.EPSILON_GREEDY:
            action = action_selection_epsilon_greedy(self.n_arm, self.q, self.args['eps'])
        elif self.act_sel == ActionSelection.UCB:
            action = action_selection_ucb(self.curr_step, self.q,
                                          self.cnt_action, self.args['conf_lvl'])
        else:
            raise ValueError('Not correct ActionSelection value')
        
        self.cnt_action[action] += 1
        
        return action
    
    
    def update_q(self, reward: float, action: int) -> None:
        if self.update_method == UpdateMethod.AVERAGE:
            sz_step = update_method_average(self.cnt_action, action)
        elif self.update_method == UpdateMethod.WEIGHTED_AVERAGE:
            sz_step = update_method_weighted_average(self.args['alpha'])
        else:
            raise ValueError('Not correct UpdateMethod value')
        
        old_q = self.q[action]
        self.q[action] = old_q + sz_step * (reward - old_q)
    
    
    def on_step_begin(self) -> None:
        return super().on_step_begin()
    
    
    def on_step_end(self) -> None:
        return super().on_step_end()

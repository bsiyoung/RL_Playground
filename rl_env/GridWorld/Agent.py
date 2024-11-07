# Include GridWorld/DP folder to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[0])
sys.path.append(root_path)

import random
import numpy as np

from World import World
from common import *


class Agent:
    def __init__(self, n_rows: int, n_cols: int, init_v: int=-10, gamma=0.5) -> None:
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._init_v = init_v
        
        self._v = np.full((n_rows, n_cols), init_v, dtype=np.float32)
        self._policy = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
        
        self._gamma = gamma
        
        
    def init(self, env: World):
        for row in range(self._n_rows):
            for col in range(self._n_cols):
                # Init policy of (row, col) by one of possible action
                actions = env.get_actions((row, col))
                if len(actions) == 0:
                    continue
                
                a = actions[random.randint(0, len(actions) - 1)]
                self._policy[row, col, a.value] = 1.0
                
        for t_row, t_col in env._terminals:
            self._v[t_row, t_col] = 0.0
        
        
    def policy_eval(self, env: World, threshold: float=1e-4):
        '''
        Run policy evaluation loop
        until change of every state value is sufficiently small (<threshold)
        '''
        
        while True:
            delta = 0
            
            for row in range(self._n_rows):
                for col in range(self._n_cols):
                    if env.is_non_movable(row, col):
                        continue
                    
                    s = (row, col)
                    actions = env.get_actions(s)
                    
                    new_v = 0
                    for a in actions:
                        action_prob = self._policy[row, col, a.value]
                        
                        pairs = env.get_available_pairs(s, a)
                        buf = 0
                        for s_, r in pairs:
                            p = env.prob(s_, r, s, a)
                            buf += p * (r + self._gamma * self._v[s_[0], s_[1]])
                    
                        new_v += action_prob * buf
                        
                    orig_v = self._v[row, col]
                    self._v[row, col] = new_v
                    
                    delta = max(delta, abs(orig_v - new_v))
                    
            if delta < threshold:
                break
            
            
    def value_iter(self, env: World, threshold: float=1e-4):
        '''
        Run value iteration loop
        until change of every state value is sufficiently small (<threshold)
        '''
        
        while True:
            delta = 0
            
            for row in range(self._n_rows):
                for col in range(self._n_cols):
                    if env.is_non_movable(row, col):
                        continue
                    
                    s = (row, col)
                    orig_v = self._v[row, col]
                    
                    actions = env.get_actions(s)
                    
                    max_val = None
                    for a in actions:
                        pairs = env.get_available_pairs(s, a)
                        
                        val = 0
                        for s_, r in pairs:
                            val += env.prob(s_, r, s, a) * (r + self._gamma * self._v[s_[0], s_[1]])

                        if max_val == None:
                            max_val = val
                            continue
                        
                        if max_val < val:
                            max_val = val
                            
                    self._v[row, col] = max_val
                    delta = max(delta, abs(max_val - orig_v))
                    
            
            if delta < threshold:
                break
        
    
    def policy_improvement(self, env: World):
        '''
        Do policy improvement
        '''
        
        for row in range(self._n_rows):
            for col in range(self._n_cols):
                if env.is_non_movable(row, col):
                    continue
                
                s = (row, col)
                actions = env.get_actions(s)
                
                max_exp_v = None
                max_a = actions[0]
                for a in actions:
                    pairs = env.get_available_pairs(s, a)
                    
                    exp_v = 0
                    for s_, r in pairs:
                        prob = env.prob(s_, r, s, a)
                        exp_v += prob * (r + self._gamma * self._v[s_[0], s_[1]])
                    
                    if max_exp_v is None:
                        max_exp_v = exp_v
                        max_a = a
                        continue
                        
                    if max_exp_v < exp_v:
                        max_exp_v = exp_v
                        max_a = a
                        
                self._policy[row, col, :] = 0
                self._policy[row, col, max_a.value] = 1.0
                        
                        
    def print_policy(self, env: World):
        for row in range(self._n_rows):
            for col in range(self._n_cols):
                policy = Action(np.argmax(self._policy[row, col]))
                ch_direction = '  '
                if env.is_wall(row, col):
                    print('  ', end='')
                    continue
                
                if env.is_terminal(row, col):
                    print('O ', end='')
                    continue
                
                if policy == Action.UP:
                    ch_direction = '↑ '
                elif policy == Action.DOWN:
                    ch_direction = '↓ '
                elif policy == Action.LEFT:
                    ch_direction = '← '
                elif policy == Action.RIGHT:
                    ch_direction = '→ '
                    
                print(ch_direction, end='')
            print()
        
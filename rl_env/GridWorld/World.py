# Include GridWorld/DP folder to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[0])
sys.path.append(root_path)

from typing import Union
from typing import Tuple, List

from common import *


class World:
    def __init__(self, n_rows: int, n_cols: int, trans_reward: float=-1.0) -> None:
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._trans_reward = trans_reward
        
        self._walls: List[Tuple[int, int]] = []
        self._terminals: List[Tuple[int, int]] = []
    
    
    def add_wall(self, row: int, col: int) -> None:
        if self.is_out_of_bound(row, col):
            raise ValueError
        
        if self.is_terminal(row, col):
            raise ValueError
        
        if self.is_wall(row, col):
            return
        
        self._walls.append((row, col))
        
        
    def rm_wall(self, row: int, col: int) -> None:
        try:
            self._walls.remove((row, col))
        except ValueError:
            pass
        
    
    def is_wall(self, row: int, col: int) -> bool:
        return (row, col) in self._walls
    
    
    def add_terminal(self, row: int, col: int) -> None:
        if self.is_out_of_bound(row, col):
            raise ValueError
        
        if self.is_wall(row, col):
            raise ValueError
        
        if self.is_terminal(row, col):
            return
        
        self._terminals.append((row, col))
        
        
    def rm_terminal(self, row: int, col: int) -> None:
        try:
            self._terminals.remove((row, col))
        except ValueError:
            pass
    
    
    def is_terminal(self, row: int, col: int) -> bool:
        return (row, col) in self._terminals
    
    
    def is_out_of_bound(self, row: int, col: int) -> bool:
        if row < 0 or row >= self._n_rows:
            return True
        
        if col < 0 or col >= self._n_cols:
            return True
        
        return False
    
    
    def is_non_movable(self, row: int, col: int) -> bool:
        if self.is_out_of_bound(row, col):
            return True
        
        return self.is_wall(row, col) or self.is_terminal(row, col)
    
    
    def get_next_state(self, s: Tuple[int, int], a: Action) \
        -> Union[Tuple[int, int], Tuple[None, None]]:
        if self.is_out_of_bound(*s) or self.is_wall(*s):
            return (None, None)
        
        s_ = [s[0], s[1]]
        
        if a == Action.UP:
            s_[0] -= 1
        elif a == Action.DOWN:
            s_[0] += 1
        elif a == Action.LEFT:
            s_[1] -= 1
        elif a == Action.RIGHT:
            s_[1] += 1
            
        if self.is_out_of_bound(*s_) or self.is_wall(*s_):
            return (None, None)
            
        return (s_[0], s_[1])
    
    
    def get_actions(self, s: Tuple[int, int]) -> List[Action]:
        res = []
        
        if self.is_non_movable(*s):
            return res
        
        for a in Action:
            if self.get_next_state(s, a) != (None, None):
                res.append(a)
                
        return res
    
    
    def get_available_pairs(self, s: Tuple[int, int], a: Action) \
        -> List[Tuple[Tuple[int, int], float]]:
        res = []
        
        next_state = self.get_next_state(s, a)
        if next_state != (None, None):
            res.append((next_state, self._trans_reward))
        
        return res
            
    
    def prob(self, s_: Tuple[int, int], r: float, 
             s: Tuple[int, int], a: Action) -> float:
        if r != self._trans_reward:
            return 0.0
        
        if self.get_next_state(s, a) == s_:
            return 1.0
        return 0.0
    
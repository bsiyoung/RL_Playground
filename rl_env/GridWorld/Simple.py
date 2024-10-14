from typing import TYPE_CHECKING

from typing import Literal
from typing import Sequence
from typing import List, Tuple

if TYPE_CHECKING is True:
    pass

import copy
import numpy as np


class Simple:
    """
    Class provides simple grid world environment
    There are only terminal state and transition reward in simple grid world

    About location
        location is described as tuple of two integers : (x, y)
            e.g) (0, 2), (3, 5)
        (0, 0) : bottom left, (width - 1, height - 1) : top right
    """
    GridAction = Literal['left', 'right', 'up', 'down']

    def __init__(self, width: int, height: int, trans_reward: float = -1.0):
        """
        Initialize class

        Args:
            width : width of grid world
            height : height of grid world
            trans_reward : transition reward (usually negative float)
        """
        self.__width: int = width
        self.__height: int = height
        self.__terminal: List[Tuple[int, int]] = []  # locations of terminal state

        self.trans_reward = trans_reward


    def set_terminals(self, locations: Sequence[Tuple[int, int]]):
        for loc in locations:
            if loc[0] < 0 or loc[0] > self.__width:
                raise ValueError('x location out of index')
            if loc[1] < 0 or loc[1] > self.__height:
                raise ValueError('y location out of index')
            
            if loc in self.__terminal:
                continue
            
            self.__terminal.append(loc)


    def get_terminals(self) -> List[Tuple[int, int]]:
        return copy.deepcopy(self.__terminal)


    def get_actions(self, state: Tuple[int, int]) -> Tuple:
        """
        Get available actions on current state
        """
        actions: List[Simple.GridAction] = []

        if state in self.__terminal:
            # There is no action at terminal location
            return ()
        
        if state[0] != 0:
            actions.append('left')
        
        if state[0] != self.__width - 1:
            actions.append('right')

        if state[1] != 0:
            actions.append('down')

        if state[1] != self.__height - 1:
            actions.append('up')

        return tuple(actions)


    def get_next_state_reward_pairs(self, state: Tuple[int, int], action: GridAction) -> Tuple[dict, ...]:
        """
        Get every pairs of (next_state, reward) when perform 'action' on 'state'
        Because the simple grid world is deterministic, there is only one pair as result
        """
        next_state = list(state)
        if action == 'left':
            next_state[0] = max(0, next_state[0] - 1)
        elif action == 'right':
            next_state[0] = min(self.__width - 1, next_state[0] + 1)
        elif action == 'up':
            next_state[1] = min(self.__height - 1, next_state[1] + 1)
        elif action == 'down':
            next_state[1] = max(0, next_state[1] - 1)
        else:
            raise ValueError('action must be one of (\'left\', \'right\', \'up\', \'down\')')
        
        return tuple([
            {'next_state': tuple(next_state), 'reward': self.trans_reward},
        ])


    def get_probability(self,
                        next_state: Tuple[int, int], reward: float,
                        state: Tuple[int, int], action: GridAction) -> float:
        """
        Get probability p(s',r|s,a)
        Because the simple grid world is deterministic, there are only 1.0 and 0.0 as result
        """

        _next_state, _reward = self.get_next_state_reward_pairs(state, action)

        if next_state == _next_state and reward == _reward:
            return 1.0

        return 0.0


    def do_action(self, state: Tuple[int, int], action: GridAction) -> dict:
        """
        Get next state and reward when actor do 'action' on 'state'
        """

        return self.get_next_state_reward_pairs(state, action)[0]

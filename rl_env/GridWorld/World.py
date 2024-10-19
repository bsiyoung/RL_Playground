from typing import Sequence, Tuple, List
from typing import Dict
from typing import Union

from enum import Enum
import copy

import numpy as np


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class World:
    """
    Class provides grid world environment

    About location
        location is described as tuple of two integers : (x, y)
            e.g) (0, 2), (3, 5)
        (0, 0) : bottom left, (width - 1, height - 1) : top right
    """

    def __init__(self,
                 width: int, height: int,
                 trans_reward: float = -1,
                 init_pos: Tuple[int, int] = (0, 0)):
        """
        Initialize class

        Args:
            width : width of grid world
            height : height of grid world
            trans_reward : transition reward (usually negative float)
        """
        self.__width = width
        self.__height = height
        self.__pos: List[int] = list(init_pos)

        self.__walls: List[Tuple[int, int]] = []
        self.__terminals: List[Tuple[int, int]] = []

        self.__trans_reward = trans_reward


    def set_terminals(self, locations: Sequence[Tuple[int, int]]) -> None:
        for loc in locations:
            if loc in self.__terminals:
                continue

            self.__terminals.append(loc)


    def get_terminals(self) -> List[Tuple[int, int]]:
        return copy.deepcopy(self.__terminals)
    

    def rm_terminals(self, locations: Sequence[Tuple[int, int]]) -> None:
        for loc in locations:
            if loc not in self.__terminals:
                continue

            self.__terminals.remove(loc)


    def clear_terminals(self) -> None:
        self.__terminals.clear()


    def is_on_terminal(self) -> bool:
        """
        Check is current position is terminal state
        """
        return (self.__pos[0], self.__pos[1]) in self.__terminals


    def set_walls(self, locations: Sequence[Tuple[int, int]]) -> None:
        for loc in locations:
            if loc in self.__walls:
                continue

            self.__walls.append(loc)


    def get_walls(self) -> List[Tuple[int, int]]:
        return copy.deepcopy(self.__walls)


    def rm_walls(self, locations: Sequence[Tuple[int, int]]) -> None:
        for loc in locations:
            if loc not in self.__walls:
                continue

            self.__walls.remove(loc)

    
    def clear_walls(self) -> None:
        self.__walls.clear()

    
    def set_position(self, x: int, y: int) -> None:
        self.__pos = [x, y]


    def get_position(self) -> Tuple:
        return tuple(self.__pos)


    def get_actions(self, pos: Union[Tuple[int, int], None] = None) -> Sequence[Action]:
        """
        Get every action which can perform on current position(state)
        """

        if pos is None:
            pos = (self.__pos[0], self.__pos[1])

        if pos in self.__walls or pos in self.__terminals:  # On a wall or terminal state
            return []
        
        res = []
        if pos[0] != 0 and (pos[0] - 1, pos[1]) not in self.__walls:
            res.append(Action.LEFT)

        if pos[0] != self.__width - 1 and (pos[0] + 1, pos[1]) not in self.__walls:
            res.append(Action.RIGHT)

        if pos[1] != self.__height - 1 and (pos[0], pos[1] + 1) not in self.__walls:
            res.append(Action.UP)

        if pos[1] != 0 and (pos[0], pos[1] - 1) not in self.__walls:
            res.append(Action.DOWN)

        return res


    def do_acition(self, action: Action) -> float:
        """
        Do an action and receive a reward
        """

        if action not in self.get_actions():
            raise ValueError('Cannot move')
        
        if action == Action.LEFT:
            self.__pos[0] -= 1
        elif action == Action.RIGHT:
            self.__pos[0] += 1
        elif action == Action.UP:
            self.__pos[1] += 1
        elif action == Action.DOWN:
            self.__pos[1] -= 1

        return self.__trans_reward


    def get_next_state_reward(self, action: Action, pos: Union[Tuple[int, int], None] = None) -> List[Dict]:
        """
        Get every combination of (next_state, reward, probability) when perform 'action' on 'state'
        Because the simple grid world is deterministic, there is only one as result with probaiblity 1.0
        """

        if pos is None:
            pos = (self.__pos[0], self.__pos[1])

        if action not in self.get_actions(pos):
            raise ValueError('Action not available on that position | action={}, pos={}'.format(action, pos))
        
        next_state = list(pos)
        if action == Action.LEFT:
            next_state[0] -= 1
        elif action == Action.RIGHT:
            next_state[0] += 1
        elif action == Action.UP:
            next_state[1] += 1
        elif action == Action.DOWN:
            next_state[1] -= 1

        return [{'state': tuple(next_state), 'reward': self.__trans_reward, 'prob': 1.0}]


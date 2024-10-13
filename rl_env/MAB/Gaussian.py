from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING is True:
    pass

import copy
import numpy as np

class Gaussian:
    """
    Class provides stationary Gaussian MAB(Multi Armed Bandit) function
    """
    def __init__(
            self,
            n_arm: int = 1,
            mean: NDArray[np.float64] = np.array([0.0]),
            std: NDArray[np.float64] = np.array([1.0])
            ) -> None:
        """
        Initialize class

        Args:
            n_arm: num of arms (= num of actions)
            mean: mean reward of each actions
            std: standard deviation of reward of each actions
        """

        if n_arm < 1:
            raise ValueError('n_arm must be same or greater than 1')
        self.__n_arm: int = n_arm
        self.__curr_step: int = 0

        self.__mean: NDArray[np.float64]
        self.__std: NDArray[np.float64]
        self.set_mean(mean)
        self.set_std(std)

    def set_mean(self, mean: NDArray[np.float64]) -> None:
        """
        Set mean values of each actions
        Save a copy of argument to prevent unintentional value change

        Raises:
            ValueError
        """
        if len(mean) != self.__n_arm:
            raise ValueError('len(mean) must be equal with n_arm')
        
        self.__mean = copy.deepcopy(mean)

    def get_mean(self) -> NDArray[np.float64]:
        """
        Return setting value of mean of returns
        Return a copy to prevent unintentional value change
        """
        return copy.deepcopy(self.__mean)

    def set_std(self, std: NDArray[np.float64]) -> None:
        """
        Set standard deviations of rewards of each actions
        Save a copy of argument to prevent unintentional value change

        Raises:
            ValueError
        """
        if len(std) != self.__n_arm:
            raise ValueError('len(std) must be equal with n_arm')
        
        self.__std = copy.deepcopy(std)

    def get_std(self) -> NDArray[np.float64]:
        """
        Return setting value of standard deviation of returns
        Return a copy to prevent unintentional value change
        """
        return copy.deepcopy(self.__std)

    def do_action(self, action_no: int) -> np.float64:
        """
        Return a reward depend on action and setting values(mean and std)

        Raises:
            ValueError
        """
        if action_no < 0 or action_no > self.__n_arm - 1:
            raise ValueError('action_no must be a non-negative integer less than n_arm={} | action_no={}'.format(self.__n_arm, action_no))
        
        reward = np.random.normal(self.__mean[action_no], self.__std[action_no], 1)[0]
        self.__curr_step += 1

        return reward

    def get_curr_step(self) -> int:
        return self.__curr_step
    
    def set_curr_step(self, step: int) -> None:
        self.__curr_step = step


class Normal(Gaussian):
    """
    Class provides stationary Gaussian MAB(Multi Armed Bandit) function
    """
    pass

import gym
import os
from copy import deepcopy

from main.experimental.util import module_from_path

# Obtain the mountain_car module so we can subclass MountainCarEnv
mount_car_file = deepcopy(*gym.envs.__path__) + '/classic_control/mountain_car.py'
mountain_car_path = os.path.abspath(mount_car_file)
MountainCarEnv = module_from_path('mountain_car', mountain_car_path).MountainCarEnv

acrobot_file = deepcopy(*gym.envs.__path__) + '/classic_control/acrobot.py'
acrobot_path = os.path.abspath(acrobot_file)
AcrobotEnv = module_from_path('acrobot', acrobot_path).AcrobotEnv


class MountainCarCostAwareEnv(MountainCarEnv):
    """
    Extension of the OpenAI MountainCarEnv to include a cost as well as
    a reward. Used to test RL algorithms that maximize the ratio of long-run
    average reward over long-run average cost.

    User can pass in a cost function that takes the current state as input
    and outputs a corresponding cost. Default cost function is 1.0, reducing
    to the reward-only case.
    """

    def __init__(self, goal_velocity=0, cost_fn=lambda x: 1.0):
        MountainCarEnv.__init__(self, goal_velocity)
        self.cost_fn = cost_fn

    def step(self, action):
        state, reward, done, d = MountainCarEnv.step(self, action)
        return state, (reward, self.cost_fn(state)), done, d


class AcrobotCostAwareEnv(AcrobotEnv):
    """
    This is a costaware version of the OpenAI Gym ACRobot Environment
    """
    def __init__(self, cost_fn=lambda x: 1.0):
        AcrobotEnv.__init__(self)
        self.cost_fn = cost_fn

    def step(self, action):
        state, reward, done, d = AcrobotEnv.step(self, action)
        return state, (reward, self.cost_fn(state)), done, d
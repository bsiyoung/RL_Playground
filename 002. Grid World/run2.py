'''
Perform value iteration DP algorithm on Grid World
'''

# Include path of RL_Playground to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[1])
sys.path.append(root_path)

from rl_env.GridWorld import *


# Create agent and environment
agent = Agent(10, 10, gamma=0.9)
env = World(10, 10)

# Add terminal and wall
env.add_terminal(9, 9)
env.add_terminal(9, 0)

env.add_wall(3, 1)
env.add_wall(3, 2)
env.add_wall(3, 3)

env.add_wall(5, 3)
env.add_wall(5, 4)
env.add_wall(5, 5)

# Init agent
agent.init(env)

# Run value iteration
agent.value_iter(env)
agent.policy_improvement(env)
print('\n')

# Print Results
print('Policy :')
agent.print_policy(env)

print()
print('State Values :')
np.set_printoptions(precision=2, suppress=True)
print(agent._v)

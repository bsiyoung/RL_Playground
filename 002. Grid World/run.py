'''
Perform conventional DP algorithm on Grid World
'''

# Include path of RL_Playground to system path
import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[1])
sys.path.append(root_path)

from rl_env.GridWorld import *


n_step = 100

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

# Run GPI(Generalized Policy Improvement) loop
for curr_step in range(1, n_step + 1):
    agent.policy_eval(env)
    agent.policy_improvement(env)
    
    if curr_step % 10 == 0 or curr_step == n_step:
        print('\r{}/{}'.format(curr_step, n_step), end='')
print('\n')

# Print Results
print('Policy :')
agent.print_policy(env)

print()
print('State Values :')
np.set_printoptions(precision=2, suppress=True)
print(agent._v)

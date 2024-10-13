import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[2])  # root path of RL_Playground
sys.path.append(root_path)

import copy

from rl_env import MAB

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(777)


def run() -> None:
    n_step = 18000
    n_avg_reward_step = 70  # Number of steps to calculate average reward

    n_arm: int = 5  # Number of arms (actions)
    colors = ('#8DD3C7', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854')  # Colors of each action for plotting graph

    eps = 0.01  # Probability of breaking ties (Not choosing greedy action)
    alpha = 1.0  # A step size of action value update

    # Create MAB environment
    mean = np.random.uniform(low=2, high=10, size=(n_arm))
    std = np.random.uniform(low=0.5, high=2, size=(n_arm))
    mab = MAB.Gaussian(n_arm=n_arm, mean=mean, std=std)
    print_actions_info(mean, std)

    q = np.zeros(n_arm)  # Action values of current step
    q_hist = []  # History of action values

    cnt_action = np.zeros(n_arm, dtype=np.int32)  # Count of how many each action performed

    avg_reward_hist = []  # History of average reward of last n_avg_step steps
    recent_rewards = list(np.zeros(n_avg_reward_step))  # Rewards of last n_avg_step steps

    greedy_true_q = []  # True action value of greedy action

    # Loop
    for curr_step in range(1, n_step + 1):
        # Select an action
        break_ties = np.random.random() < eps
        greedy_action = int(np.argmax(q))
        if break_ties:
            action = int(np.random.randint(0, n_arm))
        else:
            action = greedy_action

        # Apply an action and receive a reward
        reward = mab.do_action(action)
        cnt_action[action] += 1

        # Calculate average reward of last n_avg_step steps
        del recent_rewards[0]
        recent_rewards.append(reward)
        avg_reward_hist.append(np.average(recent_rewards))

        # Update action value
        alpha = 1 / cnt_action[action]

        q[action] = q[action] + alpha * (reward - q[action])
        q_hist.append(copy.deepcopy(q))  # Save history

        # Save true action value of current step's greedy action
        greedy_true_q.append(mean[greedy_action])

    do_plot(n_arm, n_step, n_avg_reward_step,
            mean, avg_reward_hist, greedy_true_q, q_hist,
            colors)


def print_actions_info(mean, std):
    n_action = len(mean)

    print(' Num Actions :', n_action)
    print('───────────┬─────────────────────')
    for i in range(n_action):
        print(' Action {:2d} │ Mean = {:.8f}'.format(i, mean[i]))
        print('           │  Std = {:.8f}'.format(std[i]))
        print('───────────{}─────────────────────'.format('┴' if i == n_action - 1 else '┼'))


def do_plot(n_arm, n_step, n_avg_step,
            true_q, avg_reward_hist, greedy_true_q, q_hist,
            colors):
    # plot1 (Average Of Received Rewards)
    # ======================================================================
    plt.figure(figsize=(16, 8), dpi=100)
    plt.title('Average Of Received Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.xlim(1, n_step)
    for action_no in range(n_arm):
        plt.hlines(true_q[action_no], xmin=1, xmax=n_step, colors=[colors[action_no]], linestyles='dashed', label='Action {} true action value'.format(action_no))
    plt.plot(np.arange(n_avg_step, n_step + 1), avg_reward_hist[n_avg_step-1:], color='#000000', label='average reward (recent {} steps)'.format(n_avg_step))
    plt.plot(greedy_true_q, color='#FF0000', label='greedy action true action value')
    plt.legend()
    plt.savefig('plot1.png', bbox_inches='tight', pad_inches=0.3)

    # plot2 (Action values)
    # ======================================================================
    if type(q_hist) != np.ndarray:
        q_hist = np.array(q_hist)

    plt.cla()
    plt.clf()
    plt.figure(figsize=(16, 8), dpi=100)
    plt.title('Action Values')
    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.xlim(1, n_step)
    for action_no in range(n_arm):
        plt.hlines(true_q[action_no], xmin=1, xmax=n_step, colors=[colors[action_no]], linestyles='dashed', label='Action {} true action value'.format(action_no))
        plt.plot(q_hist[:, action_no], color=colors[action_no], label='Action {} approximated action value'.format(action_no))
    plt.legend()
    plt.savefig('plot2.png', bbox_inches='tight', pad_inches=0.3)


if __name__ == '__main__':
    run()
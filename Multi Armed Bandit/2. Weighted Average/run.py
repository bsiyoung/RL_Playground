import sys, os, pathlib
root_path = str(pathlib.Path(os.path.abspath(__file__)).parents[2])  # root path of RL_Playground
sys.path.append(root_path)

import copy

from rl_env import MAB

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


def run():
    n_step = 15000
    n_phase_step = 3000  # Number of steps per phase (new MAB environment for new phase)
    n_avg_reward_step = 70  # Number of steps to calculate average reward

    n_arm: int = 5  # Number of arms (actions)
    colors = ('#8DD3C7', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854')  # Colors of each action for plotting graph

    eps = 0.07  # Probability of breaking ties (Not choosing greedy action)
    alpha = 0.1  # A step size of action value update (0 <= alpha <= 1)

    #  True : calculte weighted average
    # False : calculate action's average reward (same with 1. Simple Bandit Algorithm)
    use_constant_alpha = True

    # Create MAB environment
    print(' Phase 1')
    mab_info = create_new_mab_env(n_arm)
    mab, mean, std = mab_info['MAB'], mab_info['mean'], mab_info['std']

    q = np.zeros(n_arm)  # Action values of current step
    q_hist = []  # History of action values
    true_q_hist = [mean]  # History of true action values

    action_cnt = np.zeros(n_arm, dtype=np.int32)  # Count of how many each action performed

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
        action_cnt[action] += 1

        # Calculate average reward of last n_avg_step steps
        del recent_rewards[0]
        recent_rewards.append(reward)
        avg_reward_hist.append(np.average(recent_rewards))

        # Update action value
        if use_constant_alpha is False:
            alpha = 1 / action_cnt[action]

        q[action] = q[action] + alpha * (reward - q[action])
        q_hist.append(copy.deepcopy(q))  # Save history

        # Save true action value of current step's greedy action
        greedy_true_q.append(mean[greedy_action])

        if curr_step % n_phase_step == 0 and curr_step != n_step:  # New pahse! Create new MAB environment
            print('\n\n Phase', curr_step // n_phase_step + 1)
            mab_info = create_new_mab_env(n_arm)
            mab, mean, std = mab_info['MAB'], mab_info['mean'], mab_info['std']
            true_q_hist.append(mean)

    do_plot(n_arm, n_step, n_phase_step, n_avg_reward_step,
            true_q_hist, avg_reward_hist, greedy_true_q, q_hist,
            colors)
    

def create_new_mab_env(n_arm, verbose=True):
    mean = np.random.uniform(low=2, high=10, size=(n_arm))
    std = np.random.uniform(low=0.5, high=2, size=(n_arm))
    mab = MAB.Gaussian(n_arm=n_arm, mean=mean, std=std)

    if verbose is True:
        print_actions_info(mean, std)

    return {
        'MAB': mab,
        'mean': mean,
        'std': std
    }


def print_actions_info(mean, std):
    n_action = len(mean)

    print(' Num Actions :', n_action)
    print('───────────┬─────────────────────')
    for i in range(n_action):
        print(' Action {:2d} │ Mean = {:.8f}'.format(i, mean[i]))
        print('           │  Std = {:.8f}'.format(std[i]))
        print('───────────{}─────────────────────'.format('┴' if i == n_action - 1 else '┼'))


def do_plot(n_arm, n_step, n_phase_step, n_avg_step,
            true_q_hist, avg_reward_hist, greedy_true_q, q_hist,
            colors):
    n_phase = len(true_q_hist)

    # plot1 (Average Of Received Rewards)
    # ======================================================================
    plt.figure(figsize=(16, 8), dpi=100)
    plt.title('Average Of Received Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.xlim(1, n_step)
    plot_true_action_value(n_arm, n_step, n_phase_step, true_q_hist, colors)
    plt.plot(np.arange(n_avg_step, n_step + 1),
             avg_reward_hist[n_avg_step-1:],
             color='#000000',
             label='average reward (recent {} steps)'.format(n_avg_step))
    plt.plot(greedy_true_q, color='#FF0000', label='greedy action true action value')
    plt.legend(bbox_to_anchor=(1.02, 0), loc='lower left')
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
    plot_true_action_value(n_arm, n_step, n_phase_step, true_q_hist, colors)
    for action_no in range(n_arm):
        plt.plot(q_hist[:, action_no],
                 color=colors[action_no],
                 label='Action {} approximated action value'.format(action_no))
        
    handles, labels = plt.gca().get_legend_handles_labels()
    order = []
    for i in range(n_arm):
        order += [i, i + n_arm]
    plt.legend([handles[idx] for idx in order],
               [labels[idx] for idx in order],
               bbox_to_anchor=(1.02, 0), loc='lower left')
    plt.savefig('plot2.png', bbox_inches='tight', pad_inches=0.3)


def plot_true_action_value(n_arm, n_step, n_phase_step, true_q_hist, colors):
    n_phase = len(true_q_hist)

    for phase in range(1, len(true_q_hist) + 1):
        for action_no in range(n_arm):
            xmin = n_phase_step * (phase - 1) + 1
            xmax = n_phase_step * phase if phase != n_phase else n_step
            plt.hlines(true_q_hist[phase-1][action_no],
                       xmin=xmin, xmax=xmax,
                       colors=[colors[action_no]], linestyles='dashed',
                       label='Action {} true action value'.format(action_no) if phase == 1 else '')


if __name__ == '__main__':
    run()
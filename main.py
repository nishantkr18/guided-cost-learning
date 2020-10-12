import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from experts.PG import PG
from cost import CostNN
from gcl_training import guided_cost_learning

from torch.optim.lr_scheduler import StepLR

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ENV SETUP
env_name = 'CartPole-v0'
env = gym.make(env_name).unwrapped
if seed is not None:
    env.seed(seed)
n_actions = env.action_space.n
state_shape = env.observation_space.shape
state = env.reset()

# LOADING EXPERT SAMPLES
expert_samples = np.load('expert_samples/pg_cartpole.npy', allow_pickle=True)

# INITILIZING POLICY AND REWARD FUNCTION
policy = PG(state_shape, n_actions)
cost = CostNN(state_shape[0] + 1)
model_optimizer = torch.optim.Adam(policy.parameters(), 1e-3)
cost_optimizer = torch.optim.Adam(cost.parameters(), 1e-3)

mean_rewards = []
mean_costs = []
size = 100
samples = [policy.generate_session(env) for _ in range(int(size/2))]

for i in range(100):
    traj = [policy.generate_session(env) for _ in range(int(size))]
    samples = samples + traj

    # SAMPLING TRAJECTORIES BATCHES
    expert_trajs_ids = np.random.choice(range(len(expert_samples)), size)
    expert_trajs = expert_samples[expert_trajs_ids]
    sampled_trajs_ids = np.random.choice(range(len(samples)), size)
    sampled_trajs = np.array(samples)[sampled_trajs_ids]

    rewards, costs = [],  []
    for (expert_traj, sampled_traj) in zip(expert_trajs, sampled_trajs):
        rew, cost_item = guided_cost_learning(policy, env, cost, expert_traj,
                                              sampled_traj, model_optimizer, cost_optimizer)
        rewards.append(rew)
        costs.append(cost_item)

    mean_rewards.append(np.mean(rewards))
    mean_costs.append(np.mean(costs))

    # PLOTTING PERFORMANCE
    if i % 1 == 0:
        # clear_output(True)
        print("mean reward:%.3f" % (np.mean(rewards)))

        plt.figure(figsize=[16, 6])
        plt.subplot(1, 2, 1)

        plt.title(f"Mean reward per {size} games")
        plt.plot(mean_rewards)
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title(f"Mean cost per {size} games")
        plt.plot(mean_costs)
        plt.grid()

        # plt.show()
        plt.savefig('plots/GCL_learning_curve.png')
        plt.close()

    if np.mean(rewards) > 500:
        break

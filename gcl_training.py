import gym
import random
import numpy as np
import torch
import torch.nn as nn


def get_cumulative_rewards(rewards, gamma=0.99):
    G = np.zeros_like(rewards, dtype=float)
    G[-1] = rewards[-1]
    for idx in range(-2, -len(rewards)-1, -1):
        G[idx] = rewards[idx] + gamma * G[idx+1]
    return G


def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def guided_cost_learning(
    model,
    env,
    cost_f,
    expert_traj,
    sampled_traj,
    model_optimizer,
    cost_optimizer,
    gamma=0.99,
    entropy_coef=1e-2
):

    states, actions, rewards = sampled_traj
    states_expert, actions_expert, _ = expert_traj

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    states_expert = torch.tensor(states_expert, dtype=torch.float32)
    actions_expert = torch.tensor(actions_expert, dtype=torch.int32)

    costs = cost_f(states)
    costs_expert = cost_f(states_expert)

    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    logits_expert = model(states_expert)
    probs_expert = nn.functional.softmax(logits_expert, -1)

    # LOSS CALCULATION FOR IOC (COST FUNCTION)
    loss_IOC = torch.mean(costs_expert) + \
               torch.log(torch.mean(torch.exp(-costs)) +
                         torch.mean(torch.exp(-costs_expert))
                         )
    # UPDATING THE COST FUNCTION
    cost_optimizer.zero_grad()
    loss_IOC.backward()
    cost_optimizer.step()

    # LOSS CALCULATION FOR POLICY (PG)
    costs = cost_f(states).detach().numpy()
    cumulative_returns = np.array(get_cumulative_rewards(-costs, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
    # print(cumulative_returns)

    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = torch.sum(
        log_probs * to_one_hot(actions, env.action_space.n), dim=1)

    entropy = -torch.mean(torch.sum(probs*log_probs), dim=-1)
    loss = -torch.mean(log_probs_for_actions *
                       cumulative_returns - entropy*entropy_coef)

    # UPDATING THE POLICY NETWORK
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()

    return sum(rewards), np.sum(costs)

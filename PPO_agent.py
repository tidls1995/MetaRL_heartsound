#!/usr/bin/env python
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HSSNet(nn.Module):

    def __init__(self):
        super(HSSNet, self).__init__()

        same_padding = (5 - 1) // 2

        # self.conv1 = nn.Conv2d(1, 10, 5, padding=same_padding)
        # self.conv2 = nn.Conv2d(10, 10, 5, padding=same_padding)

        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # self.conv3 = nn.Conv2d(128, 256, 5, padding=same_padding)
        # self.conv4 = nn.Conv2d(256, 128, 5, padding=same_padding)
        self.lin1 = nn.Linear(128 * 8 * 8, 50)

        self.out_dir = nn.Linear(50, 8)  # 방향
        self.out_digit = nn.Linear(50, 2)  # normal , abnormal
        self.out_critic = nn.Linear(50, 1)


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)


        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 8 * 8)
        x = self.lin1(x)
        pi1 = self.out_digit(x)
        pi1 = F.softmax(pi1, dim=-1)
        pi2 = self.out_dir(x)
        pi2 = F.softmax(pi2, dim=-1)


        y1 = torch.bmm(pi1.unsqueeze(2), pi2.unsqueeze(1))
        y1 = y1.view(-1, 16)

        y2 = self.out_critic(x)
        #   y1 - action distribution
        #   y2 - critic's estimate of value

        return y1, y2


def torch_to_numpy(tensor):
    return tensor.data.cpu().numpy()


def numpy_to_torch(array):
    # return torch.tensor(array).float() 기존
    return torch.tensor(np.array(array)).float()


class PPO:


    def __init__(self, new_network, params=None, obs_to_input=lambda x: x,
                 lr=0.0003, df=0.95, alpha=0.5):
        # df : discount value
        # model and parameters
        # # if params is not None:
        # #     self.model = new_network(params)
        # else:
        self.n_epoch = 3
        self.model = new_network
        self.model = self.model.to(device)
        self.clip_ratio = 0.2



        if isinstance(self.model, torch.nn.Module):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.df = df  # discount factor
        self.alpha = alpha  # multiply critic updates by this factor
        self.act_count = 0
        self.ent_coef = 0.01


        # initialize replay history
        self.replay = []

        # function that converts observation into input of dimension D_in
        self.obs_to_input = obs_to_input

        # if trainable is changed to false, the model won't be updated
        self.trainable = True

    def act(self, o, env=None, display=False):
        # feed observation as input to the network to get the distribution as output
        x = self.obs_to_input(o)
        x = numpy_to_torch([x])
        x = x.to(device)
        y1, y2 = self.model(x)
        # y1 is the action distribution, y2 is the value estimate of the state
        # y1 = probability distribution of possible actions (shape: [, 8])
        # y2 = value estimation of the current state, representing the total expected future rewards from this state
        pi = torch_to_numpy(y1).flatten()  # Convert y1 to a numpy array for further calculations
        v = torch_to_numpy(y2).squeeze()  # Convert y2 to a numpy array for further calculations
        # sample action from distribution
        a = np.random.choice(np.arange(16),
                             p=pi)  # Randomly select an action based on the given probability distribution

        if display:  # If display is True, print the selected action and its details
            direction, digit = a % 8, a // 8  # Represents the direction and type of action (predicted value)
            pi1 = pi.reshape((2, 8)).sum(
                axis=0)  # Reshape the probability distribution for the 8 actions and sum along columns to get total probabilities per direction
            pi2 = pi.reshape((2, 8)).sum(axis=1)  # Sum along rows to get total probabilities per digit
            print("")
            print("Sampled action:", (direction, digit))
            print("Value estimate:", v)
            print("Distributions:", pi1, pi2, sep='\n')

        # update the current episode in replay with observation and chosen action
        if self.trainable:
            self.replay[-1]['observations'].append(o)  # Store past situations, including observations and actions
            self.replay[-1]['actions'].append(a)

        return np.array(
            a)  # Return the randomly selected action based on the probability distribution for the next step

    # Used to start a new episode and store new state and reward information
    def new_episode(self):
        # start a new episode in replay
        self.replay.append({'observations': [], 'actions': [], 'rewards': []})

    def store_reward(self, r):
        # insert 0s for actions that received no reward; end with reward r
        episode = self.replay[-1]  # Get the last episode
        T_no_reward = len(episode['actions']) - len(episode['rewards']) - 1
        # Calculate the number of actions that have not received rewards
        episode['rewards'] += [0.0] * T_no_reward + [r]  # Store rewards

    def _calculate_discounted_rewards(self):
        # calculate and store discounted rewards per episode
        for episode in self.replay:
            R = episode['rewards']  # Get the reward list of the current episode
            R_disc = []  # Initialize an empty list for discounted rewards
            R_sum = 0  # Initialize the sum of discounted rewards
            # Iterate through the reward list in reverse order
            for r in R[::-1]:
                R_sum = r + self.df * R_sum  # Calculate the cumulative discounted reward
                R_disc.insert(0, R_sum)  # Insert the new cumulative discounted reward at the beginning of the list
            episode['rewards_disc'] = R_disc  # Store the discounted reward list in the current episode data

    def calculate_weights(self):
        # Calculate weights for episodes
        rewards = [np.sum(episode['rewards']) for episode in self.replay]
        min_reward = min(rewards)
        weights = [min_reward / (r + 1e-10) for r in rewards]  # Inverse of rewards as weights
        return weights, rewards

    def weighted_sample(self, weights, Temperature=2.0):
        # Perform weighted sampling of episodes based on rewards
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        probabilities = F.softmax(weights_tensor / Temperature, dim=0).numpy()
        sampled_index = np.random.choice(len(self.replay), p=probabilities)
        sampled_episode = self.replay[sampled_index]
        return sampled_episode

    def update(self):
        assert self.trainable

        episode_losses = torch.tensor(0.0).to(device)  # Cumulative loss tensor per episode
        N = len(self.replay)  # Number of episodes in replay
        self._calculate_discounted_rewards()  # Calculate discounted rewards for each episode

        eps = 1e-8
        if N == 0:
            return

        total_loss = []
        weights, rewards = self.calculate_weights()

        # Calculate weights and sample episodes based on them
        sampled_episodes = self.weighted_sample(weights, Temperature=2.0)

        O = sampled_episodes['observations']
        A = sampled_episodes['actions']
        R = sampled_episodes['rewards']
        R_disc = numpy_to_torch(sampled_episodes['rewards_disc'])
        T = len(R_disc)

        # Forward pass, Y1 is pi(a | s), Y2 is V(s)
        X = numpy_to_torch([self.obs_to_input(o) for o in O])  # Convert observations for network input
        X = X.to(device)
        Y1, Y2 = self.model(X)  # Network outputs action policy and state value

        action_probs = F.softmax(Y1, dim=-1).to(device)
        A = torch.tensor(A, dtype=torch.int64).to(device)

        old_log_probs = torch.log(action_probs.gather(1, A.unsqueeze(-1)))
        old_log_prob = old_log_probs.mean().detach()

        epoch_count = 0

        for e in range(self.n_epoch):
            epoch_count += 1

            replay_count = 0
            for episode in self.replay:
                replay_count += 1

                O_ = episode['observations']
                A_ = episode['actions']
                R_ = numpy_to_torch(episode['rewards'])
                R_disc_ = numpy_to_torch(episode['rewards_disc'])
                T_ = len(R_disc_)

                # Forward pass, Y1 is pi(a | s), Y2 is V(s)
                X_ = numpy_to_torch([self.obs_to_input(o) for o in O_])  # Convert observations for network input
                X_ = X_.to(device)
                Y1_, Y2_ = self.model(X_)  # Network outputs action policy and state value

                Vs_curr_ = Y2_.view(-1)
                action_probs = F.softmax(Y1_, dim=-1).to(device)
                A_ = torch.tensor(A_, dtype=torch.int64).to(device)

                log_probs = torch.log(action_probs.gather(1, A_.unsqueeze(-1)))
                log_prob = log_probs.mean()

                std = torch.exp(torch.randn(T_, 16))

                entropy = 0.5 + 0.5 * torch.log(2 * torch.pi * std.pow(2))
                entropy_mean = entropy.mean()
                entropy_bonus_manual = -entropy_mean

                Vs_next = torch.cat((Vs_curr_[1:], torch.tensor([0.]).to(device)))  # Estimate value of the next state
                adv = R_.to(device) + self.df * Vs_next - Vs_curr_.to(device)  # Advantage calculation

                # Ignore gradients so the critic isn't affected by actor loss
                adv = adv.detach()

                ratio = (log_prob - old_log_prob).exp()

                surr1 = adv * ratio
                surr2 = adv * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                actor_loss = -torch.min(surr1, surr2).mean()  # Newly calculated policy

                value_loss = (R_disc_ - Vs_curr_).pow(2).mean()

                # Determine the loss for each episode
                loss = actor_loss.to(device) + value_loss.to(device) * self.alpha + entropy_bonus_manual * self.ent_coef

                total_loss.append(loss)

        self.optimizer.zero_grad()
        total_loss_mean = sum(total_loss) / len(total_loss)
        total_loss_mean.backward()

        self.optimizer.step()

        self.replay = []

    def copy(self):  # For Meta learning
        model_copy = copy.deepcopy(self.model).to(device)

        # Create a copy of this agent with frozen weights
        agent = PPO(model_copy, 0, self.obs_to_input)
        agent.model = copy.deepcopy(self.model).to(device)
        agent.trainable = False
        for param in agent.model.parameters():
            param.requires_grad = False

        return agent

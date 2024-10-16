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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HSSNet(nn.Module):
    '''
    A CNN with ReLU activations and a three-headed output, two for the 
    actor and one for the critic
    
    y1 - action distribution
    y2 - critic's estimate of value
    
    Input shape:    (batch_size, D_in)
    Output shape:   (batch_size, 40), (batch_size, 1)
    '''
    
    def __init__(self):
        
        super(HSSNet, self).__init__()
        
        same_padding = (5 - 1) // 2

        self.conv1 = nn.Conv2d(1, 64, 5, padding=same_padding)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=same_padding)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=same_padding)
        self.conv4 = nn.Conv2d(256, 128, 5, padding=same_padding)
        self.lin1 = nn.Linear(128 * 8 * 8, 50)

        
        self.out_dir = nn.Linear(50, 8) #방향
        self.out_digit = nn.Linear(50, 2) #normal , abnormal
        self.out_critic = nn.Linear(50, 1)
    
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        #print(x.shape)

        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        #print(x.shape)

        x = self.conv3(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        #print(x.shape)

        x = self.conv4(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        #print(x.shape)

        x = x.view(-1, 128 * 8 * 8)
        x = self.lin1(x)
        
        pi1 = self.out_digit(x)
        pi1 = F.softmax(pi1, dim=-1)

        pi2 = self.out_dir(x)
        pi2 = F.softmax(pi2, dim=-1)

        # https://discuss.pytorch.org/t/batch-outer-product/4025
        y1 = torch.bmm(pi1.unsqueeze(2), pi2.unsqueeze(1))
        y1 = y1.view(-1, 16)

        #rint(y1)
        y2 = self.out_critic(x)
        #   y1 - action distribution
        #   y2 - critic's estimate of value
        
        return y1, y2

def torch_to_numpy(tensor):
    return tensor.data.cpu().numpy()

def numpy_to_torch(array):
    #return torch.tensor(array).float() 기존
    return torch.tensor(np.array(array)).float()


class ActorCriticNNAgent:
    '''
    Neural-net agent that trains using the actor-critic algorithm. The critic
    is a value function that returns expected discounted reward given the
    state as input. We use advantage defined as

        A = r + g * V(s') - V(s)

    Notation:
        A - advantage
        V - value function
        r - current reward
        g - discount factor
        s - current state
        s' - next state
    '''

    def __init__(self, new_network, params=None, obs_to_input=lambda x: x,
                 lr=0.0003, df=0.5, alpha=0.5):
        # df: discount factor, alpha: factor multiplied to critic updates
        # model and parameters
        # if params is not None:
        #     self.model = new_network(params)
        # else:
        self.model = new_network
        self.model = self.model.to(device)
        if isinstance(self.model, torch.nn.Module):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.df = df  # discount factor
        self.alpha = alpha  # multiply critic updates by this factor

        # initialize replay history
        self.replay = []

        # function that converts observation into input of dimension D_in
        self.obs_to_input = obs_to_input

        # if trainable is changed to false, the model won't be updated
        self.trainable = True

    def act(self, o, env=None, display=False):

        # feed observation as input to the network to get distribution as output
        x = self.obs_to_input(o)
        x = numpy_to_torch([x])
        x = x.to(device)
        y1, y2 = self.model(x)
        # y1: action distribution, y2: value estimate of the state
        # y1 = probability distribution of possible actions (shape: [, 8])
        # y2 = value estimate of the current state, representing the total expected future rewards

        pi = torch_to_numpy(y1).flatten()  # Convert y1 to numpy array for further calculations
        v = torch_to_numpy(y2).squeeze()  # Convert y2 to numpy array for further calculations
        # sample action from distribution
        a = np.random.choice(np.arange(16),
                             p=pi)  # Select an action randomly based on the given probability distribution
        # Randomly select one of 16 actions based on the probability distribution in pi

        if display:  # Display selected action and other details
            direction, digit = a % 8, a // 8  # Represents the direction and type of action (predicted value)
            pi1 = pi.reshape((2, 8)).sum(
                axis=0)  # Reshape pi to a 2x8 matrix and sum along columns to provide total probabilities for each direction
            pi2 = pi.reshape((2, 8)).sum(axis=1)  # Sum along rows to provide total probabilities for each digit
            print("")
            print("Sampled action:", (direction, digit))
            print("Value estimate:", v)
            print("Distributions:", pi1, pi2, sep='\n')

        # Update the current episode in replay with observation and chosen action
        if self.trainable:
            self.replay[-1]['observations'].append(o)  # Store the observation and chosen action in the replay history
            self.replay[-1]['actions'].append(a)

        return np.array(
            a)  # Return the randomly selected action based on the probability distribution, used for the next step

    def new_episode(self):
        # Start a new episode in replay
        self.replay.append({'observations': [], 'actions': [], 'rewards': []})

    def store_reward(self, r):  # Process the reward 'r' passed as an argument
        # Insert 0s for actions that received no reward; end with reward r
        episode = self.replay[-1]  # Get the last episode in replay
        T_no_reward = len(episode['actions']) - len(episode['rewards']) - 1
        # Calculate the number of actions that have not received a reward yet
        episode['rewards'] += [0.0] * T_no_reward + [r]  # Store the rewards in the list

    def _calculate_discounted_rewards(self):  # Convert future rewards to present value
        # Calculate and store discounted rewards per episode

        for episode in self.replay:
            R = episode['rewards']  # Get the reward list of the current episode
            R_disc = []  # Initialize an empty list to store discounted rewards
            R_sum = 0  # Initialize the sum of discounted rewards
            # Traverse the reward list in reverse order
            for r in R[::-1]:
                R_sum = r + self.df * R_sum  # Apply the discount factor to the cumulative reward
                # Calculate the new cumulative reward
                R_disc.insert(0, R_sum)  # Insert the newly calculated cumulative reward at the beginning of the list
            episode['rewards_disc'] = R_disc  # Store the discounted rewards in the current episode data

    def update(self):
        assert self.trainable

        episode_losses = torch.tensor(0.0).to(device)  # Cumulative loss tensor per episode
        N = len(self.replay)  # Total number of episodes in the replay history
        self._calculate_discounted_rewards()  # Calculate discounted rewards for each episode

        for episode in self.replay:
            O = episode['observations']
            A = episode['actions']
            R = numpy_to_torch(episode['rewards'])
            R_disc = numpy_to_torch(episode['rewards_disc'])
            R = R.to(device)
            R_disc = R_disc.to(device)
            T = len(R_disc)

            # Forward pass, Y1 is pi(a | s), Y2 is V(s)
            X = numpy_to_torch([self.obs_to_input(o) for o in O])  # Convert observations for network input
            X = X.to(device)
            Y1, Y2 = self.model(X)  # The model outputs action policy and state value
            pi = Y1
            Vs_curr = Y2.view(-1)
            # Log probabilities of selected actions
            log_prob = torch.log(pi[np.arange(T), A])  # Calculate log probabilities for selected actions
            # Advantage of selected actions over expected reward given state
            Vs_next = torch.cat((Vs_curr[1:], torch.tensor([0.0], device=device)))  # Estimate value of the next state

            adv = R + self.df * Vs_next - Vs_curr  # Calculate the advantage of selected actions: current reward and future value minus current value
            # Ignore gradients so the critic isn't affected by actor loss
            adv = adv.detach()  # Detach the advantage to avoid gradients affecting the critic

            # Actor loss is -1 * advantage-weighted sum of log likelihood
            # Critic loss is the SE between values and discounted rewards
            actor_loss = -torch.dot(log_prob, adv)  # Weighted sum of log probabilities with negative advantage
            critic_loss = torch.sum(
                (R_disc - Vs_curr) ** 2)  # Squared error between discounted rewards and state values
            episode_losses += actor_loss + critic_loss * self.alpha  # Calculate loss for each episode

        # Backward pass
        self.optimizer.zero_grad()  # Reset gradients
        loss = episode_losses / N  # Perform backward pass for loss, update model parameters via optimization
        loss.backward()
        self.optimizer.step()

        # Reset the replay history
        self.replay = []  # Clear the replay history after training

    def copy(self):  # For Meta learning
        model_copy = copy.deepcopy(self.model).to(device)

        # Create a copy of this agent with frozen weights
        agent = ActorCriticNNAgent(model_copy, 0, self.obs_to_input)  # Create a new agent object with frozen weights
        agent.model = copy.deepcopy(self.model).to(device)
        agent.trainable = False
        for param in agent.model.parameters():
            param.requires_grad = False

        return agent

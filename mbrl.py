import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from buffer import Buffer

class Model(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control: bool):
        super(Model, self).__init__()
        
        # since we are implementing one_hot actions
        self.fc1 = nn.Linear(dim_states + dim_actions, 64)

        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_input -> 64 -> 64 -> dim_states
        # F(S_t, A_t) --> S_t+1        

        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_states)

    def forward(self, state, action):
        
        input = torch.concat([state, action], dim = 1)
        
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        
        return self.fc3(input)


class RSPlanner:

    def __init__(self, dim_states, dim_actions, continuous_control, model, planning_horizon, nb_trajectories, reward_function):
        self._dim_states = dim_states
        self._dim_actions = dim_actions
        self._continuous_control = continuous_control

        self._model = model

        self._planning_horizon = planning_horizon
        self._nb_trajectories = nb_trajectories
        self._reward_function = reward_function

        
    def generate_plan(self, observation):
        # Generate a sequence of random actions
        if self._continuous_control:
            random_actions = None
        else:
            random_actions = None
        
        # Construct initial observation 
        o_t = None

        rewards = torch.zeros((self._nb_trajectories, ))
        for i in range(self._planning_horizon):
            # Get a_t
            if self._continuous_control:
                a_t = None
            else:
                a_t = None

            # Predict next observation using the model

            # Compute reward (use reward_function)
            
            o_t = o_t1

        # Return the best sequence of actions
        return None


class MBRLAgent:

    def __init__(self, dim_states, dim_actions, continuous_control, model_lr, buffer_size, batch_size, 
                       planning_horizon, nb_trajectories, reward_function, action_space, device = 'cpu'):

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control
        
        self._action_space = action_space

        self._model_lr = model_lr

        self._model = Model(self._dim_states, self._dim_actions, self._continuous_control).to(device)

        # Adam optimizer
        self._model_optimizer = Adam(params = self._model.parameters(), lr = self._model_lr)

        self._buffer = Buffer(self._dim_states, self._dim_actions, buffer_size, batch_size, continuous_control)
        
        self._planner = RSPlanner(self._dim_states, self._dim_actions, self._continuous_control, 
                                  self._model, planning_horizon, nb_trajectories, reward_function)

        self.device = device
        
        self._loss_list = []


    def select_action(self, observation, random=False):

        if random:
            # Return random action
            return self._action_space.sample()

        # Generate plan
        plan = None

        # Return the first action of the plan
        if self._continuous_control:
            return None
        
        return None


    def store_transition(self, s_t, a_t, s_t1):
        
        self._buffer.store_transition(s_t, a_t, s_t1)


    def update_model(self):
        batches = self._buffer.get_batches() # get all batches from buffer
        n_batches = len(batches) # number of batches
        
        total_loss = 0 # initialize total loss
        for ob_t, a_t, ob_t1 in batches: # for all batches
    
            # transform to tensor
            ob_t = torch.tensor(ob_t, device = self.device).float()
            a_t = torch.tensor(a_t, device = self.device).float()
            ob_t1 = torch.tensor(ob_t1, device = self.device).float()
            
            # reset gradient
            self._model_optimizer.zero_grad()
            
            # generate prediction
            y_pred = self._model(ob_t, a_t)
            
            # calculate MSE loss
            loss = F.mse_loss(y_pred, ob_t1)
            
            # backward
            loss.backward()
            
            # change parameters
            self._model_optimizer.step()
            
            # sum loss to total_loss
            total_loss += loss.item()
            
        # store mean loss
        self._loss_list += [total_loss / n_batches]
        
    def plot_loss(self, exp_name):
        plt.figure(figsize = (8, 5))
        plt.plot(range(len(self._loss_list)), self._loss_list, marker = '.', color = 'C0')
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.tight_layout()
        plt.savefig(f'figures/{exp_name}.pdf')
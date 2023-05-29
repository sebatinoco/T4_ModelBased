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

    def __init__(self, dim_states, dim_actions, continuous_control, model, planning_horizon, nb_trajectories, reward_function, device = 'cpu', correction = True):
        self._dim_states = dim_states
        self._dim_actions = dim_actions
        self._continuous_control = continuous_control

        self._model = model

        self._planning_horizon = planning_horizon
        self._nb_trajectories = nb_trajectories
        self._reward_function = reward_function
        
        self._device = device
        
        self._correction = correction # bool for applying correction to rewards of pendulum

        
    def generate_plan(self, observation):
        # Generate a sequence of random actions
        if self._continuous_control:
            # sample from pendulum action space
            random_actions = np.random.uniform(-2, 2, size = (self._nb_trajectories, self._planning_horizon))
        else:
            # sample from cartpole action space
            random_actions = np.random.choice([0, 1], size = (self._nb_trajectories, self._planning_horizon))
        
        # Construct initial observation 
        observation = np.tile(observation, (self._nb_trajectories, 1)) # repeat observation _nb_trajectories times
        o_t = torch.tensor(observation, device = self._device)

        rewards = torch.zeros((self._nb_trajectories, ))
        for i in range(self._planning_horizon):
            # Get a_t
            a_t = random_actions[:, i]
            
            if self._continuous_control:
                a_t = np.expand_dims(a_t, 1)
            else:
                a_t = np.eye(self._dim_actions)[a_t] # one hot actions if discrete
            
            a_t = torch.tensor(a_t, device = self._device).float()

            # Predict next observation using the model
            with torch.no_grad():
                o_t1 = self._model(o_t, a_t)

            # Compute reward (use reward_function)
            if self._continuous_control:
                rewards += self._reward_function(o_t, a_t, self._correction)
            else:
                rewards += self._reward_function(o_t, a_t)
            
            o_t = o_t1

        # Return the best sequence of actions
        return random_actions[np.argmax(rewards), :]


class MBRLAgent:

    def __init__(self, dim_states, dim_actions, continuous_control, model_lr, buffer_size, batch_size, 
                       planning_horizon, nb_trajectories, reward_function, device = 'cpu', correction = True):

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control

        self._model_lr = model_lr

        self._model = Model(self._dim_states, self._dim_actions, self._continuous_control).to(device)

        # Adam optimizer
        self._model_optimizer = Adam(params = self._model.parameters(), lr = self._model_lr)

        self._buffer = Buffer(self._dim_states, self._dim_actions, buffer_size, batch_size, continuous_control)
        
        self._planner = RSPlanner(self._dim_states, self._dim_actions, self._continuous_control, 
                                  self._model, planning_horizon, nb_trajectories, reward_function, 
                                  device = device, correction = correction)

        self.device = device
        
        self._loss_list = []

    def select_action(self, observation, random=False):

        if random:
            # Return random action
            
            if self._continuous_control:
                return np.random.uniform(-2, 2, size = (1,))
            else:
                return np.random.choice([0, 1], size = (1,))

        # Generate plan
        plan = self._planner.generate_plan(observation)

        # Return the first action of the plan        
        if self._continuous_control:
            return np.array(plan[0], ndmin = 1)
        
        return plan[0]

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
        #self._loss_list += [total_loss / n_batches]
        
    def plot_loss(self, exp_name):
        plt.figure(figsize = (8, 5))
        plt.plot(range(len(self._loss_list)), self._loss_list, marker = '.', color = 'C0')
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.tight_layout()
        plt.savefig(f'figures/{exp_name}.pdf')
        plt.close()
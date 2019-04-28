# AI for a self driving car

import numpy as np
import random
import os  # for saving the brain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Architecture of the nn


class NN(nn.Module):

    def __init__(self, input_neurones, actions_output):
        super(NN, self).__init__()
        self.input_neurones = input_neurones
        self.actions_output = actions_output
        self.fc1 = nn.Linear(input_neurones, 30)
        self.fc2 = nn.Linear(30, actions_output)

    def forward(self, state):
        # Activation using rectifier function
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Memory for past states


class replayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # event -->[last state, new state, last action, last reward]
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # Makes a reshape that puts together the same atribs for each event taken
        samples = zip(*random.sample(self.memory, batch_size))
        # Concatena los elementos respecto al elemento de la primera dimension de samples [0] así estan alineados
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Neural network


class Dqn():

    def __init__(self, input_neurones, actions_output, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = NN(input_neurones, actions_output)
        self.memory = replayMemory(100000)
        # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # unsqueeze for nest the list into another list so pythorch can iterate over it
        self.last_state = torch.Tensor(input_neurones).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        # Softmax function will give the highest prob to the highest q value
        # state is a torch.Tensor so model will accept it
        # Multiplying the q values makes the one with the highest prob even more probable to happen and the lowest less probable
        with torch.no_grad():
            probs = F.softmax(self.model(Variable(state))*1000, dim=1)
        # print('probs = ', probs)
        # print('self.model = ', self.model(Variable(state))*7)
        # print('state = ', state)
        action = probs.multinomial(1)
        # print('action = ', action)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Action index = 1
        # Gather selects the best action
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)
        # Maximun of the q values(index 1) of the next state
        # Detach is used to detach all the outputs of the several states of batch
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        # Backpropagation
        td_loss.backward(retain_graph=True)
        # Updating weights
        self.optimizer.step()

    def update(self, reward, new_signal):
        # new_signal is an state (list made of sensors and orientation) so we have to convert it into a tensor[]
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # Converting an integer and a float to tensors
        self.memory.push((self.last_state, new_state, torch.LongTensor(
            [int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
                100)
            self.learn(batch_state, batch_next_state,
                       batch_reward, batch_action.long()) ###################
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        # + 1 because we dont want 0 on the denominator
        return sum(self.reward_window)/(len(self.reward_window)+1)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('Loading network...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Success')
        else:
            print('No network saved')

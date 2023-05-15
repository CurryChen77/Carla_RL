# -*-coding: utf-8 -*-
# written by chenkeyu

import torch
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torchvision.models as models
from Carla_Env import CarEnv


class DQN(nn.Module):
    def __init__(self, input_channel, action_dim):
        super(DQN, self).__init__()
        self.input_channel = input_channel
        self.action_dim = action_dim
        # the action dimension is one
        self.mobilnet = models.mobilenet_v3_small(num_classes=self.action_dim)

    def forward(self, x):
        x = self.mobilnet(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # state, next_state: numpy [H, W, 3]
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # state, next_state -> tuple:(np[H, W, 3],np[H, W, 3],np[H, W, 3]...)
        # action, reward, done -> tuple:(int,int,int)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self,
                 im_width,
                 im_height,
                 throttle_list,
                 steer_list,
                 batch_size=8,
                 show_cam=True,
                 pretrain_path=None,
                 input_channel=3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.9999,
                 epsilon_min=0.01,
                 lr=1e-3,
                 buffer_capacity=10000,
                 update_interval=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channel = input_channel
        self.im_width = im_width
        self.im_height = im_height
        self.show_cam = show_cam
        self.pretrain_path = pretrain_path
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.memory = ReplayBuffer(buffer_capacity)
        self.epsilon = epsilon
        # create possible action list
        self.throttle_list = throttle_list
        self.steer_list = steer_list
        self.throttle_dim = len(self.throttle_list)
        self.steer_dim = len(self.steer_list)
        self.action_dim = self.throttle_dim + self.steer_dim  # 根据throttle和steer计算动作空间大小

        # init the network
        self.q_network = DQN(self.input_channel, self.action_dim).to(self.device)
        self.target_network = DQN(self.input_channel, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_function = nn.SmoothL1Loss()

        # load the pre-trained model parameters
        if self.pretrain_path:
            self.load_model(self.pretrain_path)

        # create the carla environment
        self.env = CarEnv(im_width=self.im_width, im_height=self.im_height,
                          throttle_list=self.throttle_list, steer_list=self.steer_list, show_cam=self.show_cam)

    def act(self, state):
        # state: tensor [1, 3, H, W]
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        if random.random() > self.epsilon:
            with torch.no_grad():
                # state: tensor[1, 3, H, W]
                state = state.to(self.device)
                q_values = self.q_network(state)
                # the front self.throttle_dim of q_values belong to throttle
                throttle_q_values = q_values[:self.throttle_dim]
                # find the max index of throttle
                throttle = torch.argmax(throttle_q_values).item()
                # the rest q_values belong to the throttle
                steer_q_values = q_values[self.throttle_dim:]
                # find the max index of steer
                steer = torch.argmax(steer_q_values).item()
        else:
            # random choose the index of throttle and steer
            throttle = random.randrange(self.throttle_dim)
            steer = random.randrange(self.steer_dim)

        # clever way to save the throttle index and steer index using one index
        # multiply self.steer_dim because steer will never bigger than self.steer_dim
        return throttle * self.steer_dim + steer

    def update_network(self):
        # state tuple([H, W, 3], [H, W, 3]........)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(np.array(state)).permute(0, 3, 1, 2).to(self.device)
        # state [batch_size, 3, H, W]
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        # action [batch_size, 1]
        reward = torch.FloatTensor(reward).to(self.device)
        # reward [batch_size]
        next_state = torch.FloatTensor(np.array(next_state)).permute(0, 3, 1, 2).to(self.device)
        # next_state [batch_size, 3, H, W]
        done = torch.FloatTensor(done).to(self.device)
        # done [batch_size]

        # transform the unify index of throttle and steer to their individual one
        # throttle_action [batch_size, 1]
        throttle_action = torch.div(action, self.steer_dim, rounding_mode="floor")  # The action index of throttle
        # steer_action [batch_size, 1]
        steer_action = action % self.steer_dim  # the remainder

        # seperate the unify q_values to their individual one using their corresponding index
        # q_values [batch_size, self.throttle_dim + self.steer_dim]
        q_values = self.q_network(state)
        # throttle_q_value [batch_size, self.throttle_dim]
        throttle_q_values = q_values[:, :self.throttle_dim]
        # steer_q_value [batch_size, self.steer_dim]
        steer_q_values = q_values[:, self.throttle_dim:]

        # normalize the throttle_q_values
        # throttle_q_value [batch_size, self.throttle_dim]
        throttle_q_values = F.softmax(throttle_q_values, dim=-1)
        # throttle_action [batch_size, 1]
        throttle_q_values = throttle_q_values.gather(1, throttle_action).squeeze(1)  # throttle_q_value [batch_size]
        steer_q_values = steer_q_values.gather(1, steer_action).squeeze(1)  # steer_q_value [batch_size]

        # current_q_values [batch_size]
        current_q_values = throttle_q_values + steer_q_values

        # compute the target q values
        next_q_values = self.target_network(next_state)  # next_q_values [batch_size,self.throttle_dim + self.steer_dim]
        next_throttle_q_values = next_q_values[:, :self.throttle_dim]  # next_throttle_q_value [batch_size, self.throttle_dim]
        next_steer_q_values = next_q_values[:, self.throttle_dim:]  # next_steer_q_value [batch_size, self.steer_dim]
        # return the max value index in self.throttle_dim
        next_throttle_actions = next_throttle_q_values.max(-1)[1]  # [batch_size]
        # return the max value index in self.steer_dim
        next_steer_actions = next_steer_q_values.max(-1)[1]  # [batch_size]
        next_q_values = next_throttle_q_values.gather(1, next_throttle_actions.unsqueeze(1)).squeeze(1) + \
                        next_steer_q_values.gather(1, next_steer_actions.unsqueeze(1)).squeeze(1)

        # compute the target_q_values
        target_q_values = (reward + self.gamma * next_q_values * (1 - done))  # target_q_values [batch_size]

        # compute the loss of the q_network
        loss = F.mse_loss(current_q_values, target_q_values.detach())
        # only update the parameter of the q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def load_model(self, pretrain_path):
        # only load the parameter, need to have a initial model structrue
        pretrain_model = torch.load(pretrain_path)
        print("successfully load the pretrained model")
        self.q_network.load_state_dict(pretrain_model["model_dict"])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer.load_state_dict(pretrain_model['optimizer_dict'])
        self.pretrain_episode = pretrain_model["episode"]
        self.epsilon = pretrain_model['epsilon']
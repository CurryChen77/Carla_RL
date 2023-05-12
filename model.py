# -*-coding: utf-8 -*-
# written by chenkeyu

import torch
import math
import random
import time
import numpy as np
import carla
from collections import deque
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


class CarEnv:
    def __init__(self):
        # Create a Carla client and connect to the Carla server
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        # Get the Carla world and blueprint library
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # get the map and weather conditions
        self.world_map = self.world.get_map()
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        # Get the vehicle blueprint and spawn the vehicle
        self.model3 = self.blueprint_library.find("vehicle.tesla.model3")

    def reset(self):
        self.actor_list = []
        self.sensor_list = []
        self.collision_hist = []
        # Reset the vehicle to a random position and velocity
        car_transform = random.choice(self.world_map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model3, car_transform)
        self.vehicle.set_simulate_physics(True)
        self.actor_list.append(self.vehicle)

        # create a list to detect collision
        colsensor = self.blueprint_library.find("sensor.other.collision")
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.colsensor = self.world.spawn_actor(colsensor, sensor_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # Get the vehicle's state information
        state = self.get_state()
        return state

    def collision_data(self, event):
        self.collision_hist.append(event)

    def get_state(self):
        # Get the vehicle's location, velocity, and orientation
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        orientation = self.vehicle.get_transform().rotation.yaw
        # Return a numpy array of the state information
        state = np.array([location.x, location.y, velocity.x])
        return state

    def step(self, action):
        control = carla.VehicleControl(throttle=0.8, steer=float(action-1)/2)
        self.vehicle.apply_control(control)
        time.sleep(0.05)
        location = self.vehicle.get_location()
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        return (location.x, location.y, kmh), reward, done


# class DQN(nn.Module):
#     def __init__(self):
#         super(DQN, self).__init__()
#         # load the model
#         model = models.resnet50(weights=False)
#         # get the input fc features
#         fc_features = model.fc.in_features
#         # modify the output class of the prediction head to 3(right, left, straight)
#         model.fc = nn.Linear(fc_features, 3)
#         # add the softmax
#         model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1))
#         self.model = model
#
#     def forward(self, x):
#         x = self.model(x)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01, lr=1e-3, buffer_capacity=10000, batch_size=64, update_interval=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.memory = ReplayBuffer(buffer_capacity)
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()
        # create the carla environment
        self.env = CarEnv()

    def act(self, state):
        # state: (location.x, location.y, kmh)
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action = q_values.argmax().item()
        else:
            action = random.randrange(self.action_dim)
        # action: 0, 1, 2
        return action

    def update_network(self, i):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.q_network(state).gather(1, action)
        target_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)
        target_q_values = reward + self.gamma * target_q_values * (1 - done)

        loss = F.mse_loss(q_values, target_q_values.detach())
        # only update the parameter of the q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
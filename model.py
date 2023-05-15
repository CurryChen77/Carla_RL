# -*-coding: utf-8 -*-
# written by chenkeyu

import torch
import math
import random
import time
import numpy as np
import carla
import cv2
from collections import deque
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


class CarEnv:
    def __init__(self,
                 im_width,
                 im_height,
                 throttle_list,
                 steer_list,
                 desired_speed=40,
                 show_cam=True):
        # Create a Carla client and connect to the Carla server
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)

        # Get the Carla world and blueprint library
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # get the map and weather conditions
        self.world_map = self.world.get_map()
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        # Get the vehicle blueprint and spawn the vehicle
        self.model3 = self.blueprint_library.find("vehicle.tesla.model3")

        # desired speed
        self.desired_speed = desired_speed

        # image width height
        self.im_width = im_width
        self.im_height = im_height
        self.show_cam = show_cam
        self.front_camera = None

        self.throttle_list = throttle_list
        self.steer_list = steer_list

    def reset(self):
        self.actor_list = []
        self.sensor_list = []
        self.collision_hist = []
        self.cross_lane_hist = []
        # Reset the vehicle to a random position and velocity
        car_transform = random.choice(self.world_map.get_spawn_points())
        try:
            self.vehicle = self.world.spawn_actor(self.model3, car_transform)
        except RuntimeError:
            print("first spawn failed, create a new spawn points")
            car_transform_new = random.choice(self.world_map.get_spawn_points())
            self.vehicle = self.world.spawn_actor(self.model3, car_transform_new)
        self.vehicle.set_simulate_physics(True)
        self.actor_list.append(self.vehicle)

        # when init a car, first pause 2s
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        # create a sensor to get the rgb image
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(self.rgb_cam, cam_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))

        # create a sensor to detect collision
        colsensor = self.blueprint_library.find("sensor.other.collision")
        colsensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.colsensor = self.world.spawn_actor(colsensor, colsensor_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # create a sensor to detect crossing lane
        cross_lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        cro_sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.5))
        self.cross_lane_sensor = self.world.spawn_actor(cross_lane_sensor, cro_sensor_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.cross_lane_sensor)
        self.cross_lane_sensor.listen(lambda event: self.cross_lane_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
        # self.front_camera [H ,W, 3]
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def cross_lane_data(self, event):
        self.cross_lane_hist.append(event)

    def process_img(self, image):
        raw_image = np.array(image.raw_data)
        image_4channel = raw_image.reshape((self.im_height, self.im_width, 4))
        rgb_image = image_4channel[:, :, :3]
        if self.show_cam:
            cv2.imshow("front_image", rgb_image)
            cv2.waitKey(1)
        # rgb_image numpy form [H, W, 3]
        self.front_camera = rgb_image

    def step(self, action, steer_dim):
        throttle_action = torch.div(action, steer_dim, rounding_mode="floor")  # The number of times you can go into it
        steer_action = action % steer_dim  # the remainder
        control = carla.VehicleControl(throttle=self.throttle_list[throttle_action], steer=self.steer_list[steer_action])
        self.vehicle.apply_control(control)
        # time.sleep(0.05)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # collision flag
        if len(self.collision_hist) != 0:
            done = True
            col_flag = 1
        else:
            col_flag = 0
            done = False

        # over speed flag
        if kmh > self.desired_speed:
            overspeed_flag = 1
        else:
            overspeed_flag = 0

        # out of the lane
        out_of_lane_flag = len(self.cross_lane_hist)

        reward = -300 * col_flag + 0.1 * kmh + -1 * overspeed_flag + -1 * out_of_lane_flag
        # self.front_camera: numpy [H, W, 3]
        return self.front_camera, reward, done


class DQN(nn.Module):
    def __init__(self, input_channel, action_dim):
        super(DQN, self).__init__()
        self.input_channel = input_channel
        self.action_dim = action_dim
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
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self,
                 im_width,
                 im_height,
                 throttle_list,
                 steer_list,
                 input_channel=3,
                 show_cam=True,
                 pretrain_path=None,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.9999,
                 epsilon_min=0.01,
                 lr=1e-3,
                 buffer_capacity=10000,
                 batch_size=8,
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
                # state: tensor [1, 3, H, W]
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
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).permute(0, 3, 1, 2).to(self.device)
        # next_state [batch_size, 3, H, W]
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # transform the unify index of throttle and steer to their individual one
        throttle_action = torch.div(action, self.steer_dim, rounding_mode="floor")  # The action index of throttle
        steer_action = action % self.steer_dim  # the remainder

        # seperate the unify q_values to their individual one using their corresponding index
        q_values = self.q_network(state)
        throttle_q_values = q_values[:, :self.throttle_dim]
        steer_q_values = q_values[:, self.throttle_dim:]

        # normalize the throttle_q_values
        throttle_q_values = F.softmax(throttle_q_values, dim=-1)
        throttle_q_values = throttle_q_values.gather(1, throttle_action).squeeze(1)
        steer_q_values = steer_q_values.gather(1, steer_action).squeeze(1)

        # current_q_values [batch_size]
        current_q_values = throttle_q_values + steer_q_values

        # compute the target q values
        next_q_values = self.target_network(next_state)
        next_throttle_q_values = next_q_values[:, :self.throttle_dim]
        next_steer_q_values = next_q_values[:, self.throttle_dim:]
        next_throttle_actions = next_throttle_q_values.max(-1)[1]
        next_steer_actions = next_steer_q_values.max(-1)[1]
        next_q_values = next_throttle_q_values.gather(1, next_throttle_actions.unsqueeze(1)).squeeze(0) + \
                        next_steer_q_values.gather(1, next_steer_actions.unsqueeze(1)).squeeze(0)

        # target_q_values [batch_size]
        target_q_values = (reward + self.gamma * next_q_values * (1 - done)).squeeze(1)

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
# -*-coding: utf-8 -*-
# written by chenkeyu

import torch
import math
import random
import time
import numpy as np
import carla
import cv2


class CarEnv:
    def __init__(self,
                 im_width,
                 im_height,
                 throttle_list,
                 steer_list,
                 desired_speed=40,
                 second_per_episode=15,
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

        # the possible action list in throttle and steer
        self.throttle_list = throttle_list
        self.steer_list = steer_list

        # the longest duration time of one episode
        self.second_per_episode = second_per_episode

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

        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

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

        # record the starting time of this episode
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
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

        # prevent the car is always circling
        if self.episode_start + self.second_per_episode < time.time():
            done = True

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
# -*-coding: utf-8 -*-
# written by chenkeyu

import math
import os
import random
import time
import numpy as np
import carla
import torch
from torch.utils.tensorboard import SummaryWriter
from model import DQNAgent


class AgentTrainer:
    def __init__(self,
                 agent,
                 episodes,
                 save_episodes=100,
                 target_update_episodes=10,
                 model_name="MobileNet_v3",
                 save_path="pretrained_models/",
                 auto_save=True):
        self.agent = agent
        self.episodes = episodes
        self.auto_save = auto_save
        self.save_path = save_path
        self.save_episodes = save_episodes
        self.target_update_episodes = target_update_episodes
        self.model_name = model_name

    def train(self):
        # check whether need to reopen a new summary writer of just use the old one
        if self.agent.pretrain_path:
            # new to reuse the pretrained logs
            start_episode = self.agent.pretrain_episode
            # self.agent.pretrain_path (pretrained_models/05-13_MLP_50.pth)
            pretrain_time = os.path.basename(self.agent.pretrain_path).split("_")[0]
            writer = SummaryWriter(f"./logs/{pretrain_time}")
            training_range = range(start_episode, self.episodes)
        else:
            # local time 5-13_12-45
            local_time = time.strftime("%Y-%m-%d", time.localtime())
            start_episode = 0
            # open a new one
            writer = SummaryWriter(f"./logs/{local_time}")
            training_range = range(start_episode, self.episodes)
        try:
            for i in training_range:
                state = self.agent.env.reset()
                # state: numpy [H, W, 3]
                done = False
                total_reward = 0
                while not done:
                    tensor_state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2)
                    # tensor_state [1, H, W, 3]
                    action = self.agent.act(tensor_state)
                    next_state, reward, done = self.agent.env.step(action, self.agent.steer_dim)
                    total_reward += reward
                    # state, next_state: numpy [H, W, 3]
                    self.agent.memory.push(state, action, reward, next_state, done)
                    state = next_state

                    if len(self.agent.memory) > self.agent.batch_size:
                        loss = self.agent.update_network()
                        # write the log file
                        writer.add_scalar("train_loss", loss.item(), i)

                # each 10 episode, target network will load the parameter from the q_network
                if i % self.target_update_episodes == 0:
                    self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())

                # save the model for every 100 episodes
                if i % self.save_episodes == 0 and self.auto_save and i != start_episode:
                    save_time = time.strftime("%Y-%m-%d", time.localtime())
                    self.save_model(os.path.join(self.save_path, f"{save_time}_{self.model_name}_{i}.pth"), i)
                    print(f"save model in episodes {i}")

                # print the training log
                v = self.agent.env.vehicle.get_velocity()
                kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
                print("Episode: {}, Total reward: {:.2f}, epsilon: {:.2f}".format(
                    i, total_reward, self.agent.epsilon)
                )

                # write the summary_writer file
                writer.add_scalar("Total reward", total_reward, i)
                writer.add_scalar("epsilon", self.agent.epsilon, i)
                # destory every vehicle in the actor_list when finishing each episodes
                self.agent.env.client.apply_batch([carla.command.DestroyActor(x) for x in self.agent.env.actor_list])
                for sensor in self.agent.env.sensor_list:
                    sensor.destroy()

            writer.close()
        except KeyboardInterrupt:
            print("--------------")
            print("Exit")
        finally:
            if len(self.agent.env.actor_list) != 0:
                # destory every vehicle in the actor_list when finishing each episodes
                self.agent.env.client.apply_batch([carla.command.DestroyActor(x) for x in self.agent.env.actor_list])
            if len(self.agent.env.sensor_list) != 0:
                for sensor in self.agent.env.sensor_list:
                    sensor.destroy()
            print("destory all agent and sensor")

    def save_model(self, path, episode):
        # only save the parameter of the model, not the whole network structure
        torch.save({'episode': episode,
                    'epsilon': self.agent.epsilon,
                    'optimizer_dict': self.agent.optimizer.state_dict(),
                    'model_dict': self.agent.q_network.state_dict()},
                   path)


def main():
    # predefined parameters
    episodes = 1000
    save_episodes = 50
    target_update_episodes = 10
    show_cam = True
    model_name = "MobileNet_v3"
    batch_size = 8
    im_width = 320
    im_height = 240
    throttle_list = [0.2, 0.4, 0.6, 0.8, 1]
    steer_list = [0, 1, 2]
    pretrain_path = None
    # pretrain_path = 'pretrained_models/2023-05-14_MobileNet_v3_50.pth'

    # Create a DQNAgent and start training
    agent = DQNAgent(im_width, im_height, throttle_list=throttle_list, steer_list=steer_list, show_cam=show_cam,
                     batch_size=batch_size, pretrain_path=pretrain_path)
    trainer = AgentTrainer(
        agent, episodes, save_episodes, target_update_episodes, model_name=model_name
    )
    print("start training !!")
    trainer.train()


if __name__ == '__main__':
    main()
# -*-coding: utf-8 -*-
# written by chenkeyu

import math
import random
import time
import numpy as np
import carla
import torch
from torch.utils.tensorboard import SummaryWriter
from Carla_RL.model import DQNAgent


class AgentTrainer:
    def __init__(self, agent, episode):
        self.agent = agent
        self.episode = episode

    def train(self, episodes=1000):
        # each time of training will start the log
        writer = SummaryWriter("./logs")
        for i in range(episodes):
            state = self.agent.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.agent.env.step(action)
                total_reward += reward
                self.agent.memory.push(state, action, reward, next_state, done)
                state = next_state

                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.update_network(i)
                    # write the log file
                    writer.add_scalar("train_loss", loss.item(), i)

            # each 10 episode, target network will load the parameter from the q_network
            if i % 10 == 0:
                self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())

            # save the model for every 100 episodes
            if i % 100 == 0:
                self.agent.save_model(f"models/model_{i}.pt")

            v = self.agent.env.vehicle.get_velocity()
            kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
            print("Episode: {}, Total reward: {}, speed: {:.2f}".format(
                i, total_reward, kmh)
            )
            # write the log file
            writer.add_scalar("Total reward", total_reward, i)
            writer.add_scalar("epsilon", self.agent.epsilon, i)

            # destory every vehicle in the actor_list when finishing each episodes
            # print('destroying actors')
            self.agent.env.client.apply_batch([carla.command.DestroyActor(x) for x in self.agent.env.actor_list])
            for sensor in self.agent.env.sensor_list:
                sensor.destroy()
            # print('done')
        writer.close()

    def save_model(self, path):
        torch.save(self.agent.q_network.state_dict(), path)

    def load_model(self, path):
        self.agent.q_network.load_state_dict(torch.load(path))
        self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())


def main():
    # Create a DQNAgent and start training
    state_dim = 3
    action_dim = 3
    episodes = 1000
    agent = DQNAgent(state_dim, action_dim)
    trainer = AgentTrainer(agent, episodes)
    trainer.train()


if __name__ == '__main__':
    main()
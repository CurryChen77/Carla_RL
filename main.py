# -*-coding: utf-8 -*-
# written by chenkeyu

import argparse
from DQN_Agent import DQNAgent
from Agent_Trainer import AgentTrainer

def get_parser():
    # argparse.ArgumentParser生成argparse对象 description为描述信息，当在命令行输入需要显示帮助信息时，会显示
    parser = argparse.ArgumentParser(description="Training DQN in Carla")
    # training episode
    parser.add_argument("--episodes", default=1000, type=int)
    # saving per episodes
    parser.add_argument("--save_episodes", default=50, type=int)
    # target network undate per episodes
    parser.add_argument("--target_update_episodes", default=10, type=int)
    # whether to show cam, if need to show the cam, the add the "--show_cam" in command
    parser.add_argument("--show_cam", action='store_true')
    # DQN network name
    parser.add_argument("--model_name", default="MobileNet_v3", type=str)
    # batch_size
    parser.add_argument("--batch_size", default=8, type=int)
    # image_width
    parser.add_argument("--im_width", default=320, type=int)
    # image_height
    parser.add_argument("--im_height", default=240, type=int)
    # throttle_list
    parser.add_argument("--throttle_list", default=[0.2, 0.4, 0.6, 0.8, 1], type=list)
    # steer_list
    parser.add_argument("--steer_list", default=[0, 1, 2], type=list)
    # pretrain_path
    parser.add_argument("--pretrain_path", default=None, type=str)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Create a DQNAgent and start training
    agent = DQNAgent(im_width=args.im_width, im_height=args.im_height,
                     throttle_list=args.throttle_list, steer_list=args.steer_list,
                     batch_size=args.batch_size, show_cam=args.show_cam, pretrain_path=args.pretrain_path)
    # initial the Agent Trainer
    trainer = AgentTrainer(
        agent, episodes=args.episodes, save_episodes=args.save_episodes,
        target_update_episodes=args.target_update_episodes, model_name=args.model_name
    )
    print("start training !!")
    trainer.train()


if __name__ == '__main__':
    main()
### root@user:/workspace/isaaclab# python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 5 --checkpoint logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew_343.82913.pth
### root@user:/workspace/isaaclab# python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 5
### -------------------------------- these comands for running the official agent trained from isaac lab


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ppo on factory env")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### -----------------------------------------------------------
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from isaaclab_tasks.utils import parse_env_cfg
import wandb
import imageio

from utils_1 import TestingAgent, env_wrapper
from agents_defn import Agent
from utils_1 import target_dof_torque_log, log_values
    
def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.factory.factory_env_diff_ik:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.factory.factory_env_cfg_diff_ik:FactoryTaskPegInsertCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    env = env_wrapper(env, video_folder, output_type=output_type, enable_normalization_rewards=False)
    
    return env


def main():
    exp_name = "diff_ik_2"
    file_path_csv = os.path.join("custom_scripts", "logs", "ppo_factory", "csv_files", f"{exp_name}.csv")
    video_folder = os.path.join("custom_scripts", "logs", "ppo_factory", exp_name)
    checkpoint_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_folder, f"{exp_name}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(video_folder, output_type="torch")
    # env = make_env(output_type="torch")
    env.eval()
    agent = Agent(env, eval=True)
    agent.to(device)




    avg_reward = TestingAgent(env, device, args_cli.num_envs, agent, checkpoint_path = checkpoint_path, num_episodes=2, recording_enabled=env.enable_recording, sim_step_func=log_values, file_path = file_path_csv)

    print(f"Average reward over 4 episodes: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":

    main()



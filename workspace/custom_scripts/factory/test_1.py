import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="factory Gym environment.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to spawn.")

# parser.add_argument("--video", action="store_true", help="Enable video recording during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=400, help="Interval between video recordings (in steps).")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import torch
import os


import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
import imageio
import time
from utils_1 import env_wrapper


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

def generate_trajectory(time):
    x = 0.05
    y = 0.05
    z = 0.05
    return x,y,z

def error_generation(time):
    dx = 0.0
    dy = 0.0
    dz = 0.0
    dx_angle = 0.0
    dy_angle = 0.0
    dz_angle = 0.0
    del_pose = np.array([[dx, dy, dz, dx_angle, dy_angle, dz_angle]])
    return del_pose



def main():

    env = make_env(video_folder=None, output_type="numpy")

    obs, info = env.reset()
    episode = 0
    step = 0
    while episode<5:
        actions = error_generation(0)
        actions = np.repeat(actions, args_cli.num_envs, axis=0)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        dones = np.logical_or(terminations, truncations)
        # print(dones)
        
        print("step ", step)
        print("raw action \n", env.unwrapped.logging_values['raw_action'])
        print("smoothed action \n", env.unwrapped.logging_values['smoothed_action'])
        print("processed action \n", env.unwrapped.logging_values['processed_action'])
        
        
        if np.any(dones):
            obs, info = env.reset()
            episode += 1
            step = 0
        step += 1




    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

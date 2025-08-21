# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# import isaaclab_tasks  # noqa: F401
# import custom_scripts.factory

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.io import load_yaml
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
import imageio
import time


# PLACEHOLDER: Extension template (do not remove this comment)


class env_wrapper_rl_games_video(gym.Wrapper):
    '''
    this function is to make a wrapper around gym env which is sent to rl games config
    its used availing the camera to record the video
    '''
    def __init__(self, env, video_folder:str | None = None):
        super().__init__(env)
        self.env = env

        if video_folder is not None:
            self.enable_recording = True
            os.makedirs(video_folder, exist_ok=True)
            self.vid_writers = []
            self.step_cntr = 0

            self.camera_flag = 0
            self.recording_step = 0
            self.video_length = 350
            self.record_freq = 5000
            
            for i in range(len(self.unwrapped.cameras)):
                writer = imageio.get_writer(os.path.join(video_folder, f"cam{i}_video.mp4"), fps=20)
                self.vid_writers.append(writer)

    def step(self, actions, *args, **kwargs):
        obs, rewards, terminations, truncations, info = self.env.step(actions, *args, **kwargs)
        ## you can also use super().step(actions)
        
        if self.enable_recording:
            for camera in self.unwrapped.cameras:
                camera.update(self.unwrapped.step_dt)
            self.step_cntr+=1
    
        return obs, rewards, terminations, truncations, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        if self.enable_recording:
            self.step_cntr = 0
            self.camera_flag = 0
            self.recording_step = 0
            
            ## setting the pose for cameras
            ##--- cam 1
            self.set_camera_pose_fixed_asset(0, 0)

            ##--- cam2
            env_id = np.random.randint(1, self.unwrapped.num_envs)
            print(f"[INFO]: Randomly selected env_id for cam2: {env_id}")
            self.set_camera_pose_fixed_asset(1, env_id)
            

        return obs, info
    
    def set_camera_pose_fixed_asset(self, camera_id, env_id):

        fixed_asset_default_state =  self.unwrapped._fixed_asset.data.default_root_state[env_id].clone()
        camera_target = fixed_asset_default_state[:3] + torch.tensor([0.0, 0.0, 0.005], device=self.unwrapped.device) + self.scene.env_origins[env_id]
        eye_camera = camera_target + torch.tensor([0.5, -0.9, 0.3], device= self.unwrapped.device)

        self.unwrapped.cameras[camera_id].set_world_poses_from_view(eye_camera.unsqueeze(0), camera_target.unsqueeze(0))
        
    def record_cameras(self):
        for i, camera in enumerate(self.unwrapped.cameras):

            cam_image = camera.data.output["rgb"]
            if isinstance(cam_image, torch.Tensor):
                cam_image = cam_image.detach().cpu().numpy()
            
            ## removing batch dimension
            if cam_image.shape[0] == 1:
                cam_image = cam_image[0]
            
            cam_image = cam_image.astype(np.uint8)
            self.vid_writers[i].append_data(cam_image)

def make_env(video_folder=None):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.factory_env_rl_games.factory_env:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.factory_env_rl_games.factory_env_cfg:FactoryTaskPegInsertCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array") 

    if video_folder is not None:
        env = env_wrapper_rl_games_video(env, video_folder)

    return env, env_cfg


# @hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
rl_games_config = "/workspace/custom_scripts/factory_env_rl_games/rl_games_params.yaml"
def main(agent_cfg_path:str=rl_games_config):  # (env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = load_yaml(agent_cfg_path)
    video_folder = os.path.join("custom_scripts", "logs", "trial_videos", "tr1_vid")

    env, env_cfg = make_env(video_folder)
    # env_cfg = env.unwrapped.env_cfg
    
    # env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    # env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    # env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

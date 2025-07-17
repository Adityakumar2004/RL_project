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

class LSTMwithDones(nn.Module):
    """ lstm that handles done flags for rl games"""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.device = next(self.parameters()).device 
            
    def forward(self, inputs, lstm_state, done):
        """
        lstm_state : 2--> cell state and hidden state (num_layers, batch_size, hidden_size)
        done : (seq_len, batch_size)
        input: (seq_len, batch_size, input_size)
        """
        # done = done.to(dtype=inputs.dtype, device= self.device)
        new_hidden = []

        for x, d in zip(inputs, done):
            # print("x shape:", x.shape, "d shape:", d.shape)
            # print("d dtype:", d.dtype, "d device:", d.device, "d min/max:", d.min().item(), d.max().item())
            # assert torch.all((d == 0) | (d == 1)), f"d contains values other than 0 or 1: {d}"
            h, lstm_state = self.lstm(
                x.unsqueeze(0), ## shape: (1, B, input_size)
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h] ## h shape: (1, B, 1024=> hidden_size)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1) ## shape: (T × B, hidden_size)
        return new_hidden, lstm_state

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, envs, hidden_size = 1024, num_layers = 2):
        super().__init__()
        self.envs = envs
        self.lstm = LSTMwithDones(envs.total_obs_space["policy"].shape[-1], hidden_size, num_layers)

        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.mlp_after_lstm = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ELU(),  # From RL-Games config: activation: elu
            layer_init(nn.Linear(512, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
        )

        self.actor_mean = layer_init(nn.Linear(64, envs.action_space.shape[-1]), std=0.01)
        self.actor_logstd = layer_init(nn.Linear(64, envs.action_space.shape[-1]), std=0.01)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)    

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2    

    def get_states(self, x, lstm_state, done):
        ## x shape: (seq_len × B, total_obs_space["policy"].shape[-1])
        ## lstm_state shape: 2 (num_layers, B, hidden_size)
        ## done shape: (seq_len × B,)

        batch_size = lstm_state[0].shape[1]  ## batch_size = B = num_envs
        
        x = x.reshape((-1, batch_size, self.envs.total_obs_space["policy"].shape[-1])) ## shape: (seq_len, B, input_size=total_obs_space["policy"].shape[-1])

        done = done.reshape((-1, batch_size)) ## shape: (seq_len, B)
        
        new_hidden, new_lstm_state = self.lstm(x, lstm_state, done)

        ## new_hidden shape: (seq_len × B, hidden_size)
        ## new_lstm_state shape: 2 (num_layers, B, hidden_size)

        return new_hidden, new_lstm_state 
        
    def get_action(self, x, lstm_state, done, action=None):
        """
        x : (seq_len * B, total_obs_space["policy"].shape[-1])
        lstm_state : 2 (num_layers, B, hidden_size)
        done : (seq_len * B,)
        action : (seq_len * B, ) or None if we want to sample an action

        """

        hidden, new_lstm_state = self.get_states(x, lstm_state, done) 
        hidden = self.layer_norm(hidden)
        
        ## hidden shape: (seq_len × B, hidden_size)

        mlp_output = self.mlp_after_lstm(hidden)
        ## mlp_output shape: (seq_len × B, 64)

        action_mean = self.actor_mean(mlp_output)
        ## action_mean shape: (seq_len × B, action_space.shape[-1])

        action_logstd = self.actor_logstd(mlp_output)
        action_logstd = torch.clamp(action_logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = action_logstd.exp()
        ## std shape: (seq_len × B, action_space.shape[-1])

        dist = Normal(action_mean, std)
        if action is None:
            action = dist.sample()
        
        return action,dist.log_prob(action).sum(-1), dist.entropy().sum(-1), new_lstm_state


class critic(nn.Module):
    def __init__(self, envs, hidden_size = 1024, num_layers = 2):
        super().__init__()
        self.envs = envs
        self.lstm = LSTMwithDones(envs.total_obs_space["critic"].shape[-1], hidden_size, num_layers)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.mlp_after_lstm = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
        )
        
        self.critic_value = layer_init(nn.Linear(64, 1), std=1.0)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
       
    def get_states(self, x, lstm_state, done):
        ## x shape: (seq_len × B, total_obs_space["critic"].shape[-1])
        ## lstm_state shape: 2 (num_layers, B, hidden_size)
        ## done shape: (seq_len × B,)

        batch_size = lstm_state[0].shape[1]  ## batch_size = B = num_envs
        
        x = x.reshape((-1, batch_size, self.envs.total_obs_space["critic"].shape[-1])) 
        ## x shape: (seq_len, B, input_size=total_obs_space["critic"].shape[-1])

        done = done.reshape((-1, batch_size)) ## shape: (seq_len, B)
        
        new_hidden, new_lstm_state = self.lstm(x, lstm_state, done)

        ## new_hidden shape: (seq_len × B, hidden_size)
        ## new_lstm_state shape: 2 (num_layers, B, hidden_size)

        return new_hidden, new_lstm_state 

    def get_value(self, x, lstm_state, done):
        """
        x : (seq_len * B, total_obs_space["critic"].shape[-1])
        lstm_state : 2 (num_layers, B, hidden_size)
        done : (seq_len * B,)
        
        """
        hidden, new_lstm_state = self.get_states(x, lstm_state, done)
        hidden = self.layer_norm(hidden)
        ## hidden shape: (seq_len × B, hidden_size)

        mlp_output = self.mlp_after_lstm(hidden)
        ## mlp_output shape: (seq_len × B, 64)

        value = self.critic_value(mlp_output)
        ## value shape: (seq_len × B, 1)

        return value, new_lstm_state
        

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.hidden_size = 1024
        self.num_layers = 2
        

        self.actor = Actor(envs, self.hidden_size, self.num_layers)
        self.critic = critic(envs, self.hidden_size, self.num_layers)

    def get_action(self, x, lstm_state, done, action=None):

        x = x["policy"]
        action, log_prob, entropy, new_lstm_state = self.actor.get_action(x, lstm_state, done, action)

        return action, log_prob, entropy, new_lstm_state
    
    def get_value(self, x, lstm_state, done):

        x = x["critic"]
        value, new_lstm_state = self.critic.get_value(x, lstm_state, done)
        return value, new_lstm_state
    

class RunningNormalizer:
    def __init__(self, size, epsilon=1e-4, clip_range=5.0):
        self.size = size
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = epsilon
        self.clip_range = clip_range

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray):
        std = np.sqrt(self.var) + 1e-8
        x_norm = (x - self.mean) / std
        return np.clip(x_norm, -self.clip_range, self.clip_range)


class env_wrapper(gym.Wrapper):
    '''
        this gives obs, rewards, terminations, truncations in numpy format removes info from the env.step output

    '''
    def __init__(self, env, video_folder:str | None =None, output_type: str = "numpy", enable_normalization_obs = True, enable_normalization_rewards = False):
        super().__init__(env)
        self.env = env
        self.total_obs_space = {"policy": env.observation_space, "critic": env.state_space}
        self.output_type = output_type

        self.enable_normalization_rewards = enable_normalization_rewards
        self.enable_normalization_obs = enable_normalization_obs
        self.training = True

        if enable_normalization_obs:
            self.normalizers = {
                "policy": RunningNormalizer(self.total_obs_space["policy"].shape[-1], clip_range=8.0),
                "critic": RunningNormalizer(self.total_obs_space["critic"].shape[-1], clip_range=8.0),
            }
                
        if enable_normalization_rewards:
            self.normalizers["rewards"] = RunningNormalizer(1, clip_range=5.0)


        self.enable_recording = False
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
     
    def step(self, actions):
        
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

        ## clipping the actions 
        actions = torch.clamp(actions, -1, 1)

        obs, rewards, terminations, truncations, info = self.env.step(actions)

        info_custom = {}
        info_custom["org_reward"] = rewards.cpu().numpy()

        if self.enable_normalization_obs:
            ## updating the normalizers
            
            if self.training:
                self.normalizers["policy"].update(obs["policy"].cpu().numpy())
                self.normalizers["critic"].update(obs["critic"].cpu().numpy())

            with torch.no_grad():
                obs["policy"] = torch.tensor(self.normalizers["policy"].normalize(obs["policy"].cpu().numpy()), device=self.env.device)
                obs["critic"] = torch.tensor(self.normalizers["critic"].normalize(obs["critic"].cpu().numpy()), device=self.env.device)
        
        if self.enable_normalization_rewards:
            if self.training:
                self.normalizers["rewards"].update(rewards.cpu().numpy())
                rewards = torch.tensor(self.normalizers["rewards"].normalize(rewards.cpu().numpy()), device=self.env.device)


        if self.output_type == "numpy":
            obs = {k: v.cpu().numpy() for k, v in obs.items()}
            rewards = rewards.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()

        if self.enable_recording:
            for camera in self.unwrapped.cameras:
                camera.update(self.unwrapped.step_dt)
            
            # if self.step_cntr % self.record_freq == 0 or self.camera_flag == 1:
            #     if self.camera_flag == 0:
            #         print("[INFO]: Recording video...")
            #         env_id = np.random.randint(1, self.unwrapped.num_envs)
            #         print(f"[INFO]: Randomly selected env_id for cam2: {env_id}")

            #         self.set_camera_pose_fixed_asset(1, env_id)

            #         self.camera_flag = 1
            #         self.recording_step = 0
            #     if self.recording_step < self.video_length:
            #         self.record_cameras()
            #         self.recording_step += 1
            #     else:
            #         self.camera_flag = 0

            self.step_cntr+=1


        return obs, rewards, terminations, truncations, info_custom
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)


        if self.enable_normalization_obs:
            ## updating the normalizers
            if self.training:
                self.normalizers["policy"].update(obs["policy"].cpu().numpy())
                self.normalizers["critic"].update(obs["critic"].cpu().numpy())
                
        
            ## applying normalization
            with torch.no_grad():
                obs["policy"] = torch.tensor(self.normalizers["policy"].normalize(obs["policy"].cpu().numpy()), device=self.env.device)
                obs["critic"] = torch.tensor(self.normalizers["critic"].normalize(obs["critic"].cpu().numpy()), device=self.env.device)


        if self.output_type == "numpy":
            obs = {k: v.cpu().numpy() for k, v in obs.items()}
        

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

    def train(self):
        print("[INFO] Environment set to TRAINING mode.")
        self.training = True
    
    def eval(self):
        print("[INFO] Environment set to EVALUATION mode.")
        self.training = False


def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.factory.factory_env:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.factory.factory_env_cfg:FactoryTaskPegInsertCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    env = env_wrapper(env, video_folder, output_type=output_type, enable_normalization_rewards=False)
    
    return env


def TestingAgent(env, device, agent: Agent, num_episodes = 2, recording_enabled=True):
    with torch.no_grad():
        env.eval()  # set the env to evaluation mode
        # obs, _ = env.reset()
        total_reward = 0.0
        step_cntr = 0
        for _ in range(num_episodes):
            env.eval()
            obs, _ = env.reset()
            done_flag = False

            next_lstm_state_actor = (
            torch.zeros(agent.num_layers, args_cli.num_envs, agent.hidden_size).to(device),
            torch.zeros(agent.num_layers, args_cli.num_envs, agent.hidden_size).to(device),
            )
            dones = torch.zeros((env.num_envs), device = device)

            while not done_flag:
                actions, _, _, next_lstm_state_actor= agent.get_action(obs, next_lstm_state_actor, dones)

                next_obs, rewards, terminations, truncations, _ = env.step(actions)
                
                dones = (terminations | truncations).float()

                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.cpu().numpy()
                if isinstance(terminations, torch.Tensor):
                    terminations = terminations.cpu().numpy()
                if isinstance(truncations, torch.Tensor):
                    truncations = truncations.cpu().numpy()

                obs = next_obs
                total_reward += rewards.mean()
                done_flag = bool(np.any(terminations) or np.any(truncations))
                step_cntr += 1

                ## recording the video
                if recording_enabled:
                    env.record_cameras()
            
    
                
    print(f"avg reward over {step_cntr / num_episodes} steps: {total_reward / num_episodes:.2f}")
    return total_reward / num_episodes

def main():

    video_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "videos_lstm_2_test")
    checkpoint_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_folder, "cp_lstm_2_rnd.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(video_folder, output_type="torch")
    env.eval()
    agent = Agent(env)
    agent.to(device)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if hasattr(env, 'normalizers'):
        for k, state in checkpoint['normalizer_state'].items():
            if k in env.normalizers:
                env.normalizers[k].mean = state['mean']
                env.normalizers[k].var = state['var']
                env.normalizers[k].count = state['count']
                env.normalizers[k].clip_range = state['clip_range']

    
    agent.load_state_dict(checkpoint["agent"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    avg_reward = TestingAgent(env, device, agent, num_episodes=10, recording_enabled=True)

    print(f"Average reward over 4 episodes: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":

    main()
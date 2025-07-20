
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ppo on factory env")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
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
    


### -----------------------------------------------------------------

@dataclass
class Args:
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    num_envs: int = args_cli.num_envs
    """the number of parallel game environments"""
    ## this is the user input 

    total_timesteps: int = 10_00_000 #10_00_000 ------------- 
    ## this is changed in the code runtime: total_timesteps = num_updates * batch_size
    """total timesteps of the experiments"""
    
    num_updates: int = 200 # ----------------------------
    """the number of updates for the entire loop on top of the env roll out (num_steps), update_epochs"""
    ## this is the user input 

    learning_rate: float = 1.0e-4 ##---------
    """the learning rate of the optimizer"""
    num_steps: int = 128 #256 #16 
    """the number of steps to run in each environment per policy rollout"""
    ## this is the user input 

    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gae: bool = True
    """Use GAE for advantage computation"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32 #64 #4 #16 -------------
    """the number of mini-batches"""
    ## this is the user input 

    update_epochs: int = 4 #15 ---------------mini_epochs
    """the K epochs to update the policy"""
    ## this is the user input

    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 #0.1#0.2 -------------
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0001#0.0
    """coefficient of the entropy"""
    vf_coef: float = 2 #0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float =0.008 # None-------------
    """the target KL divergence threshold"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    ## batch_size  =  num_steps * num_envs

    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    norm_value: bool = True  # Add this flag to control value normalization

    

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

        rewards = self.calculate_rewards(output_type=self.output_type)


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

    def calculate_rewards(self, output_type):

        asset_info = self.unwrapped.get_asset_information()
        held_asset_coords = asset_info['held_asset_bottom_coords']

        hole_center_coords = asset_info["hole_center_coords"]
        radius = asset_info["fixed_asset_diameter"]/2

        reward = reward_function(hole_center_coords, held_asset_coords, xy_threshold = (1.5*radius)**2, alpha = 100.0, beta = 100)

        if output_type == "numpy":
            reward = reward.cpu().numpy()
        elif output_type == "torch":
            reward = torch.tensor(reward, device=self.env.device)

        return reward


def reward_function(x_desired, x_current, xy_threshold, alpha = 15.0, beta = 50):
    """
    Computes the reward as exp(-alpha * ||x_current - x_desired||^2)
    using a consistent NumPy-style computation.
    
    Inputs can be either np.ndarray or torch.Tensor (any shape ending in dimension D).
    Returns: reward of shape [...], same as batch dimensions of input
    """
    if isinstance(x_current, torch.Tensor):
        x = (x_current - x_desired).cpu().numpy()
    elif isinstance(x_current, np.ndarray):
        x = x_current - x_desired
    else:
        raise AssertionError("x_current and x_desired must be torch.Tensor or np.ndarray")
    
    squared_x = x*x

    norm_squared_xy = np.sum(squared_x[:,:2], axis=-1)  
    reward = np.exp(-alpha * norm_squared_xy)/4

    mask = norm_squared_xy < xy_threshold
    z_term = np.exp(-beta * squared_x[:,2])/4
    # reward += np.where(mask, 0.25, z_term)
    reward += z_term

    larger_mask = np.sum(squared_x, axis=-1) < 0.00004
    reward += np.where(larger_mask, 0.25, 0.0)
    

    return reward



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


def TestingAgent(env, agent: Agent, num_episodes = 2, recording_enabled=False):
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
            torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
            torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
            )
            dones = torch.zeros((env.num_envs), device = device)

            while not done_flag:
                actions, _, _, next_lstm_state_actor = agent.get_action(obs, next_lstm_state_actor, dones)

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



if __name__ == "__main__":

    ## start the env 
    # video_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "videos_lstm_1")
    checkpoint_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "checkpoints")
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "cp_lstm_task1.pt")
    
    args = Args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    args.batch_size = int(args.num_envs * args.num_steps) # 64 * 256
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 64*256 //32 = 512
    # args.num_updates = args.total_timesteps // args.batch_size # 10_00_000 // 64*256 = 625
    args.total_timesteps = args.num_updates * args.batch_size


    # envs = make_env(video_folder, output_type="torch")
    envs = make_env(output_type="torch")
    envs.train()  # set the env to training mode

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    tracking_enabled = True
    if tracking_enabled:
        wandb.init(
            project = "Space_RL",
            name = f"ppo_lstm_task1{int(time.time())}"
            
        )  

    obs = {"policy": torch.zeros((args.num_steps, args_cli.num_envs, envs.total_obs_space["policy"].shape[-1]), device=device),
           "critic": torch.zeros((args.num_steps, args_cli.num_envs, envs.total_obs_space["critic"].shape[-1]), device=device)}
    
    actions = torch.zeros((args.num_steps, args_cli.num_envs, envs.action_space.shape[-1]), device=device)
    log_probs = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    values = torch.zeros((args.num_steps, args_cli.num_envs), device=device)    

    global_step = 0

    start_time = time.time()


    # --- Resume logic ---
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    raw_episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    all_returns = []
    all_raw_returns = []
    all_lengths = []
    start_update = 1
    if args_cli.resume:
        import os
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.load_state_dict(checkpoint["agent"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            # restoring the normalizers
            if hasattr(envs, 'normalizers'):
                for k, state in checkpoint['normalizer_state'].items():
                    if k in envs.normalizers:
                        envs.normalizers[k].mean = state['mean']
                        envs.normalizers[k].var = state['var']
                        envs.normalizers[k].count = state['count']
                        envs.normalizers[k].clip_range = state['clip_range']

            # Restore learning rate
            optimizer.param_groups[0]["lr"] = checkpoint.get("learning_rate", args.learning_rate)
            global_step = checkpoint.get("global_step", 0)
            start_update = checkpoint.get("update", 1) + 1
            all_returns = checkpoint.get("all_returns", [])
            all_lengths = checkpoint.get("all_lengths", [])
            print(f"[INFO] Resumed training from checkpoint at update {start_update-1}, global_step {global_step}.")
        else:
            print(f"[WARNING] --resume flag set but checkpoint not found at {checkpoint_path}. Starting from scratch.")

    num_updates = args.total_timesteps // args.batch_size
    # initial_lstm_state_actor = (
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # )

    # initial_lstm_state_critic = (
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    # )

    next_lstm_state_actor = (
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    )

    next_lstm_state_critic = (
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    torch.zeros(agent.num_layers, args.num_envs, agent.hidden_size).to(device),
    )

    envs.train()
    next_obs, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    

    for update in range(start_update, num_updates + 1):

        # envs.train()
        # next_obs, _ = envs.reset()
        # next_lstm_state_actor = (initial_lstm_state_actor[0].clone(), initial_lstm_state_actor[1].clone())
        # next_lstm_state_critic = (initial_lstm_state_critic[0].clone(), initial_lstm_state_critic[1].clone())

        # next_done = torch.zeros(args.num_envs).to(device)
        
        initial_lstm_state_actor = (next_lstm_state_actor[0].clone(), next_lstm_state_actor[1].clone())
        initial_lstm_state_critic = (next_lstm_state_critic[0].clone(), next_lstm_state_critic[1].clone())


        
        ## doubt about initial_lstm_state and next_lstm_state

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args_cli.num_envs
           
            obs["policy"][step] = next_obs["policy"]
            obs["critic"][step] = next_obs["critic"]

            dones[step] = next_done

            
            with torch.no_grad():
                action, log_prob, _, next_lstm_state_actor = agent.get_action(next_obs, next_lstm_state_actor, next_done)
                value, next_lstm_state_critic = agent.get_value(next_obs, next_lstm_state_critic, next_done)

                values[step] = value.flatten()        

            actions[step] = action
            log_probs[step] = log_prob

            next_obs, reward, terminated, truncated, info_custom = envs.step(action)
            next_done = (terminated | truncated).float()
            rewards[step] = reward

            ### for calculating the episodic returns --------------------------------
            raw_reward = info_custom.get('org_reward', None)
            reward_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
            done_np = (terminated | truncated).cpu().numpy() if isinstance(terminated, torch.Tensor) else (terminated | truncated)

            episode_returns += reward_np
            raw_episode_returns += raw_reward
            episode_lengths += 1

            # if step % 10 == 0 and hasattr(envs, 'normalizers'):
            #     norm = envs.normalizers["rewards"]
            #     print(f"Normalizer mean: {norm.mean}, var: {norm.var}, count: {norm.count}")
            #     print(f"Raw reward: {info_custom.get('org_reward', None)}")
            #     print(f"Normalized reward: {reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward}")
           
            # print(done_np)

            if np.any(done_np):
                all_returns.extend(episode_returns[done_np == 1])
                all_raw_returns.extend(raw_episode_returns[done_np == 1])
                all_lengths.extend(episode_lengths[done_np == 1])
                episode_returns[done_np == 1] = 0
                raw_episode_returns[done_np == 1] = 0
                episode_lengths[done_np == 1] = 0
                # print("here 1")


        # bootstrap value if not done
        with torch.no_grad():
            ## doubt need to verify this segment
            next_value, _ = agent.get_value(
                next_obs,
                next_lstm_state_critic,
                next_done,
            )
            next_value = next_value.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values


        b_obs = {
            "policy": obs["policy"].reshape(-1, envs.total_obs_space["policy"].shape[-1]),
            "critic": obs["critic"].reshape(-1, envs.total_obs_space["critic"].shape[-1]),
        }
        
        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1, envs.action_space.shape[-1])
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        ## doubt about the sequence order of the data .ravel, .reshape,
        ## optmizing the policy and value nn
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches # 64 // 32 = 2
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []


        # print("b dones shape:", b_dones.shape, "b_dones dtype :", b_dones.dtype, "b_dones device:", b_dones.device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                mb_inds = torch.as_tensor(mb_inds, dtype=torch.long, device=device)

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                # --- Value normalization (per minibatch) ---
                if args.norm_value:
                    mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
                # print("b_actions.shape:", b_actions.shape)
                # print("mb_inds.shape:", mb_inds.shape)
                # print("mb_inds min/max:", mb_inds.min().item(), mb_inds.max().item())
                # print("b_actions length:", len(b_actions))
                assert mb_inds.max() < len(b_actions), f"mb_inds contains out-of-bounds indices! max: {mb_inds.max().item()}, len(b_actions): {len(b_actions)}"
                _, new_logprob, entropy, _ = agent.get_action(
                 mb_obs,
                 (initial_lstm_state_actor[0][:, mbenvinds], initial_lstm_state_actor[1][:, mbenvinds]),
                 b_dones[mb_inds],
                 b_actions[mb_inds]
                 )

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                newvalue, _ = agent.get_value(
                    mb_obs,
                    (initial_lstm_state_critic[0][:, mbenvinds],initial_lstm_state_critic[1][:, mbenvinds]),
                    b_dones[mb_inds]
                )


                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ## Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                ## value loss
                



                newvalue = newvalue.view(-1)
                # Use mb_returns in value loss below
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
        if tracking_enabled:
            wandb.log({

                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "explained_variance": explained_var,
                "clipfrac": np.mean(clipfracs)
                }, step=global_step)


        ## inference 
        # avg_reward = TestingAgent(envs, agent, num_episodes=2, recording_enabled=envs.enable_recording)
        # checkpoint = {
        #     "agent": agent.state_dict(),
        #     "optimizer": optimizer.state_dict()
        # }

        # if tracking_enabled:
        #     wandb.log({"avg_reward": avg_reward}, step=global_step)
        # print(f"Iteration/update {update + 1}/{args.num_updates}, Global Step: {global_step}, Avg Reward: {avg_reward:.2f}, Time: {time.time() - start_time:.2f}s")
        
        ## logging the episodic returns

            

        # print("this is the length of all_returns ", len(all_returns))
        # print("this is all returns \n " ,all_returns)
        # print("--"*20)

        if len(all_returns) > 0:
            avg_return = np.mean(all_returns[-100:])
            avg_raw_return = np.mean(all_raw_returns[-100:])
            avg_length = np.mean(all_lengths[-100:])
            if tracking_enabled:
                wandb.log({
                    "avg_return": avg_return,
                    "avg_length": avg_length,
                    "avg_raw_return": avg_raw_return,
                }, step=global_step)
            print(f"Iteration/update {update}/{num_updates}, Global Step: {global_step}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.1f}, Time: {time.time() - start_time:.2f}s")

        # Save checkpoint with all necessary information for resuming training
        checkpoint = {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "update": update,
            "all_returns": all_returns,
            "all_lengths": all_lengths,
            # Add any other stateful variables you want to resume
        }

        normalizer_state = {}
        if hasattr(envs, 'normalizers'):
            for k, norm in envs.normalizers.items():
                normalizer_state[k] = {
                    'mean': norm.mean,
                    'var': norm.var,
                    'count': norm.count,
                    'clip_range': norm.clip_range,
                }
        checkpoint['normalizer_state'] = normalizer_state

        torch.save(checkpoint, checkpoint_path)



    envs.close()
    wandb.finish()





        








        

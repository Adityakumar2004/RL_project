import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="factory environment.")
parser.add_argument("--num_envs", type=int, default=30, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
# import tyro
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from isaaclab_tasks.utils import parse_env_cfg
import wandb
import imageio

class env_wrapper(gym.Wrapper):
    '''
        this gives obs, rewards, terminations, truncations in numpy format removes info from the env.step output

    '''
    def __init__(self, env, video_folder:str | None =None):
        super().__init__(env)
        self.env = env
        self.total_obs_space = {"policy": env.observation_space, "critic": env.state_space}

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

        obs, rewards, terminations, truncations, info = self.env.step(actions)

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


        return obs, rewards, terminations, truncations
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
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
            
def make_env(video_folder:str | None =None):

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
     
    env = env_wrapper(env, video_folder)
    
    return env

class Replay_Buffer():
    def __init__(self, buffer_size, batch_size, env, device):
        self.env = env
        action_space = env.action_space
        total_obs_space_dict = env.total_obs_space
        self.device = device

        self.max_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.obs = {}
        self.next_obs = {}
        for k in total_obs_space_dict.keys():
            size = total_obs_space_dict[k].shape[-1]
            self.obs[k] = np.zeros((self.max_size, size), dtype=np.float32)
            self.next_obs[k] = np.zeros((self.max_size, size), dtype=np.float32)
        
        self.actions = np.zeros((self.max_size, action_space.shape[-1]), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, actions, rewards, next_obs, dones):
        
        assert isinstance(obs, dict), f"Observations must be a dictionary but got {type(obs), obs}"
        assert isinstance(next_obs, dict), f"Next observations must be a dictionary but got {type(next_obs), next_obs}"
        assert all(isinstance(v, np.ndarray) for v in obs.values()), "Observation values must be numpy arrays"
        assert all(isinstance(v, np.ndarray) for v in next_obs.values()), "Next observation values must be numpy arrays"
        assert isinstance(actions, np.ndarray), "Actions must be a numpy array"
        assert isinstance(rewards, np.ndarray), "Rewards must be a numpy array"
        assert isinstance(dones, np.ndarray), "Dones must be a numpy array"
        
        
        num_envs = rewards.shape[0]
        idxs = np.arange(self.ptr, self.ptr + num_envs) % self.max_size
        # print("these are the idxs of the replay buffer", idxs)


        for k in self.obs.keys():
            self.obs[k][idxs] = obs[k]
            self.next_obs[k][idxs] = next_obs[k]
        
        if rewards.shape == (num_envs,):
            rewards = np.expand_dims(rewards, axis=-1)
        if dones.shape == (num_envs,):
            dones = np.expand_dims(dones, axis=-1)

        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.ptr = (self.ptr + num_envs) % self.max_size
        self.size = min(self.size + num_envs, self.max_size)
    
    def sample(self, normalize_fn_dict=None):
        '''returns a batch of samples from the replay buffer 
            dtype: being torch.tensors 
            obs_batch: dict of observations dict{'key':(batch_size, obs_dim)}
            rewards: dim (batch_size, 1)
        '''
        assert self.batch_size <= self.size, "Batch size cannot be larger than the current size of the buffer"
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        
        
        obs_batch = {}
        next_obs_batch = {}
        for k in self.obs.keys():
            obs_data = self.obs[k][idxs]
            next_obs_data = self.next_obs[k][idxs]

            # normalize if a normalizer is provided
            if normalize_fn_dict and k in normalize_fn_dict:
                obs_data = normalize_fn_dict[k].normalize(obs_data)
                next_obs_data = normalize_fn_dict[k].normalize(next_obs_data)

            obs_batch[k] = torch.tensor(obs_data, dtype=torch.float32, device=self.device)
            next_obs_batch[k] = torch.tensor(next_obs_data, dtype=torch.float32, device=self.device)


        actions_batch = torch.tensor(self.actions[idxs], dtype=torch.float32, device = self.device)
        rewards_batch = torch.tensor(self.rewards[idxs], dtype=torch.float32, device = self.device)
        dones_batch = torch.tensor(self.dones[idxs], dtype=torch.float32, device = self.device)


        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch
    
    def __len__(self):
        return self.size

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None 
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1.8e5) #int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 200
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 4 #2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.3
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""

## model definitions

class SoftQNetwork(nn.Module):
    def __init__(self, env, keyword:str="critic"):
        super().__init__()
        self.fc1 = nn.Linear(
            env.total_obs_space[keyword].shape[-1] + env.action_space.shape[-1],
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env, keyword:str="policy"):
        super().__init__()
        self.fc1 = nn.Linear(env.total_obs_space[keyword].shape[-1],  256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        upper_bound = 1
        lower_bound = -1
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (upper_bound - lower_bound) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (upper_bound + lower_bound) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std
    
    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


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



class Agent():
    def __init__(self, args: Args, envs: gym.Env, device: torch.device, checkpoint_path):
        
        obs_actor_key = "policy"
        obs_critic_key = "critic"
        self.obs_actor_key = obs_actor_key
        self.obs_critic_key = obs_critic_key
        
        self.actor = Actor(envs, obs_actor_key).to(device)
        self.qf1 = SoftQNetwork(envs, obs_critic_key).to(device)
        self.qf2 = SoftQNetwork(envs, obs_critic_key).to(device)
        self.qf1_target = SoftQNetwork(envs, obs_critic_key).to(device)
        self.qf2_target = SoftQNetwork(envs, obs_critic_key).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.normalizers = {
            "policy": RunningNormalizer(envs.total_obs_space["policy"].shape[-1]),
            "critic": RunningNormalizer(envs.total_obs_space["critic"].shape[-1]),
        }


        if args.autotune:
    
            self.target_entropy = torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)

        else: 
            self.alpha = args.alpha

        self.rb = Replay_Buffer(
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            env=envs,
            device=device
            )
        
        self.args = args
        self.device = device

        self.checkpoint_path = checkpoint_path
    
    def store_buffer(self, obs, actions, rewards, next_obs, terminations, truncations):
        
        # updating the normalizer parameters Before storing to replay buffer:
        for k in obs.keys():
            self.normalizers[k].update(obs[k])
            self.normalizers[k].update(next_obs[k])


        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        dones = np.logical_or(terminations, truncations)
        self.rb.add(obs, actions, rewards, next_obs, dones)
        
    def get_action(self, obs, grad=True, normalize = False):

        if normalize:
            for k in obs.keys():
                obs[k] = self.normalizers[k].normalize(obs[k])

        obs = {k:(torch.tensor(v, device=self.device, dtype=torch.float32) if isinstance(v,np.ndarray) else v) for k,v in obs.items()}
        
        if not grad:
            with torch.no_grad():
                actions, log_probs, means = self.actor.get_action(obs[self.obs_actor_key])
                return actions.cpu().numpy(), log_probs.cpu().numpy(), means.cpu().numpy()
    
        else:
            actions, log_probs, means = self.actor.get_action(obs[self.obs_actor_key])
            return actions, log_probs, means

    def train_critic(self):
        
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.rb.sample(self.normalizers)

        
        ## dim of obs_batch is dict{'key':(batch_size, obs_dim)}
        ## dim of actions_batch is (batch_size, action_dim)
        ## dim of rewards_batch is (batch_size, 1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(next_obs_batch[self.obs_actor_key])
            target_q1 = self.qf1_target(next_obs_batch[self.obs_critic_key], next_actions)
            target_q2 = self.qf2_target(next_obs_batch[self.obs_critic_key], next_actions) ## dim (batch_size, 1)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs ## dim (batch_size, 1)
            target_q = rewards_batch + (1 - dones_batch) * self.args.gamma * target_q ## dim (batch_size, 1)

            # clamping q_target to avoid exploding gradients
            target_q = torch.clamp(target_q, -100.0, 100.0)
        
        current_q1 = self.qf1(obs_batch[self.obs_critic_key], actions_batch)  # (batch_size, 1)
        current_q2 = self.qf2(obs_batch[self.obs_critic_key], actions_batch)  # (batch_size, 1)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        ## optimizing the critic model
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return q1_loss.item(), q2_loss.item()
        pass

    def Update_QTargetNetworks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def train_actor(self):
        for _ in range(self.args.policy_frequency):
            obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.rb.sample(self.normalizers)
            actions, log_probs, _ = self.actor.get_action(obs_batch[self.obs_actor_key])
            q1 = self.qf1(obs_batch[self.obs_critic_key], actions)
            q2 = self.qf2(obs_batch[self.obs_critic_key], actions)
            min_q = torch.min(q1, q2)

            actor_loss = (self.alpha * log_probs - min_q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            ## autotuning the alpha
            if self.args.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(obs_batch[self.obs_actor_key])
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                

        if self.args.autotune:
            return actor_loss.item(), alpha_loss.item()

        return actor_loss.item()

    def save_checkpoint(self, global_step):
        path = self.checkpoint_path
        checkpoint = {
            "global_step": global_step,
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "normalizers": {
                key : {
                    "mean": normalizer.mean,
                    "var": normalizer.var,
                    "count": normalizer.count
                }
                for key, normalizer in self.normalizers.items()
            }
        }
        if args.autotune:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["a_optimizer"] = self.a_optimizer.state_dict()
        torch.save(checkpoint, path) 

    def load_from_checkpoint(self, path): ## returns global step 

        # path = self.checkpoint_path
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        self.qf1_target.load_state_dict(checkpoint["qf1_target"])
        self.qf2_target.load_state_dict(checkpoint["qf2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])

        for key, norm_stats in checkpoint["normalizers"].items():
            self.normalizers[key].mean = norm_stats["mean"]
            self.normalizers[key].var = norm_stats["var"]
            self.normalizers[key].count = norm_stats["count"]


        if self.args.autotune:
            
            self.log_alpha = checkpoint["log_alpha"]
            self.a_optimizer.load_state_dict(checkpoint["a_optimizer"])
            self.alpha = self.log_alpha.exp().item()
        return checkpoint["global_step"]       


def TestingAgent(env, agent: Agent, num_episodes = 3, recording_enabled=True):
    obs, _ = env.reset()
    total_reward = 0.0
    step_cntr = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            actions, _, _ = agent.get_action(obs, grad=False, normalize=True)
            next_obs, rewards, terminations, truncations = env.step(actions)
            obs = next_obs
            total_reward += rewards.mean()
            done = np.any(terminations) or np.any(truncations)
            step_cntr += 1

            ## recording the video
            if recording_enabled:
                env.record_cameras()

                
    print(f"avg reward over {step_cntr / num_episodes} steps: {total_reward / num_episodes:.2f}")
    return total_reward / num_episodes
    

if __name__ == "__main__":
    
    ## start the env 
    video_folder = os.path.join("custom_scripts", "logs", "sac_factory", "videos")
    checkpoint_folder = os.path.join("custom_scripts", "logs", "sac_factory", "checkpoints")
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint_autotune_4.pt")
    
    # env = make_env(video_folder)
    env = make_env()

    tracking_enabled = True
    if tracking_enabled:
        wandb.init(
            project = "Space_RL",
            name = f"SAC_Factory_{int(time.time())}"
            
        )
    ## resetting the env

    global_step = 0
    args = Args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    agent = Agent(args, env, device, checkpoint_path)

    
    ## loading the trained weights
    pretrained_continuation = False
    if pretrained_continuation:
        path_pretrained = os.path.join(checkpoint_folder, "checkpoint_1.pt")
        agent.load_from_checkpoint(path_pretrained)


    

    obs, _ = env.reset()
    while simulation_app.is_running() and global_step < args.total_timesteps:

        if global_step < args.learning_starts:

            if pretrained_continuation:
                actions, _, _ = agent.get_action(obs, grad=False, normalize=True)
            
            else:
                actions = np.random.uniform(-1,1,env.action_space.shape)
            
        
        else:
            # print("random sampling done")
            actions, _, _ = agent.get_action(obs, grad=False)
        

        ## stepping through the all the parallel envs
        next_obs, rewards, terminations, truncations = env.step(actions)
        
        ## storing the collected samples in the replay buffer
        agent.store_buffer(obs, actions, rewards, next_obs, terminations, truncations)

        ## updating the obs
        obs = next_obs

        ## training logic 
        if global_step > args.learning_starts:
            q1_loss, q2_loss = agent.train_critic()

            if global_step % args.policy_frequency == 0:
                if args.autotune:
                    actor_loss, alpha_loss = agent.train_actor()
                else:
                    actor_loss = agent.train_actor()
            
            if global_step % args.target_network_frequency == 0:
                agent.Update_QTargetNetworks()


            if global_step % 300 == 0:
                print("global step ", global_step)
                if tracking_enabled:
                    wandb.log({
                        "global_step": global_step,
                        # "rewards": rewards.mean(),
                        "q1_loss": q1_loss,
                        "q2_loss": q2_loss,
                        "actor_loss": actor_loss,
                        
                    }, step = global_step)
                
                    if args.autotune:
                        wandb.log({
                            "alpha_loss": alpha_loss,
                            "alpha" : agent.alpha
                        }, step = global_step)
                    

            if global_step % 4000 == 0:
                agent.save_checkpoint(global_step)

        ## inference
        if global_step % 3000 == 0:
            print(f"Testing agent at global step {global_step}")
            avg_rewards = TestingAgent(env, agent, num_episodes=2, recording_enabled=env.enable_recording)
            if tracking_enabled:
                wandb.log({"test_reward": avg_rewards.mean()}, step=global_step)

        
                
        global_step+=1


    simulation_app.close()
    if tracking_enabled:
        wandb.finish()
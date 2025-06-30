import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

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

class env_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
     
    def step(self, actions):
        
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

        obs, rewards, terminations, truncations, info = self.env.step(actions)

        obs = {k: v.cpu().numpy() for k, v in obs.items()}
        rewards = rewards.cpu().numpy()
        terminations = terminations.cpu().numpy()
        truncations = truncations.cpu().numpy()
    
        return obs, rewards, terminations, truncations
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = {k: v.cpu().numpy() for k, v in obs.items()}
        return obs, info
    
class Replay_Buffer():
    def __init__(self, buffer_size, batch_size, env, device):
        self.env = env
        action_space = env.action_space
        observation_space_dict = env.observation_space
        self.device = device

        self.max_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.obs = {}
        self.next_obs = {}
        for k in observation_space_dict.keys():
            size = observation_space_dict[k].shape[-1]
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
    
    def sample(self):
        '''returns a batch of samples from the replay buffer 
            dtype: being torch.tensors 
            obs_batch: dict of observations dict{'key':(batch_size, obs_dim)}
            rewards: dim (batch_size, 1)
        '''
        assert self.batch_size <= self.size, "Batch size cannot be larger than the current size of the buffer"
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        
        obs_batch = {k: torch.tensor(v[idxs], dtype=torch.float32, device = self.device) for k, v in self.obs.items()}
        next_obs_batch = {k: torch.tensor(v[idxs], dtype=torch.float32, device = self.device) for k, v in self.next_obs.items()}
        
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
    buffer_size: int = int(1.5e5) #int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 4000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 4 #2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            env.observation_space["policy"].shape[-1] + env.action_space.shape[-1],
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
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space["policy"].shape[-1],  256)
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

class Agent():
    def __init__(self, args: Args, envs: gym.Env, device: torch.device):
            self.actor = Actor(envs).to(device)
            self.qf1 = SoftQNetwork(envs).to(device)
            self.qf2 = SoftQNetwork(envs).to(device)
            self.qf1_target = SoftQNetwork(envs).to(device)
            self.qf2_target = SoftQNetwork(envs).to(device)
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
            self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)
    
            self.alpha = args.alpha
    
            self.rb = Replay_Buffer(
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                env=envs,
                device=device
                )
            
            self.args = args
            self.device = device
    
    def store_buffer(self, obs, actions, rewards, next_obs, terminations, truncations):
        
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        dones = np.logical_or(terminations, truncations)
        self.rb.add(obs, actions, rewards, next_obs, dones)
        
    def get_action(self, obs, grad=True):
        obs = {k:(torch.tensor(v, device=self.device, dtype=torch.float32) if isinstance(v,np.ndarray) else v) for k,v in obs.items()}
        
        if not grad:
            with torch.no_grad():
                actions, log_probs, means = self.actor.get_action(obs["policy"])
                return actions.cpu().numpy(), log_probs.cpu().numpy(), means.cpu().numpy()
    
        else:
            actions, log_probs, means = self.actor.get_action(obs["policy"])
            return actions, log_probs, means

    def train_critic(self):
        
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.rb.sample()
        ## dim of obs_batch is dict{'key':(batch_size, obs_dim)}
        ## dim of actions_batch is (batch_size, action_dim)
        ## dim of rewards_batch is (batch_size, 1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(next_obs_batch["policy"])
            target_q1 = self.qf1_target(next_obs_batch["policy"], next_actions)
            target_q2 = self.qf2_target(next_obs_batch["policy"], next_actions) ## dim (batch_size, 1)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs ## dim (batch_size, 1)
            target_q = rewards_batch + (1 - dones_batch) * self.args.gamma * target_q ## dim (batch_size, 1)
        
        current_q1 = self.qf1(obs_batch["policy"], actions_batch)  # (batch_size, 1)
        current_q2 = self.qf2(obs_batch["policy"], actions_batch)  # (batch_size, 1)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        ## optimizing the critic model
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()


        pass

    def Update_QTargetNetworks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def train_actor(self):
        for _ in range(self.args.policy_frequency):
            obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.rb.sample()
            actions, log_probs, _ = self.actor.get_action(obs_batch["policy"])
            q1 = self.qf1(obs_batch["policy"], actions)
            q2 = self.qf2(obs_batch["policy"], actions)
            min_q = torch.min(q1, q2)

            actor_loss = (self.alpha * log_probs - min_q).mean()
        
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            ## autotuning the alpha
            

def make_env():
    id_name = "CartPole-v1-uw"
    gym.register(
        id=id_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"workspace.custom_scripts.tut_scripts.cartpole_scene_uw:CartpoleRLEnvCfg",
        },
    )

    env_cfg = parse_env_cfg(
    id_name,
    num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg=env_cfg)
    env = env_wrapper(env)
    return env
    



if __name__ == "__main__":
    
    ## start the env 
    env = make_env()

    ## resetting the env

    global_step = 0
    args = Args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    agent = Agent(args, env, device)
    
    obs, _ = env.reset()
    while simulation_app.is_running() and global_step < args.total_timesteps:

        if global_step < args.learning_starts:
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
            agent.train_critic()

            if global_step % args.policy_frequency == 0:
                agent.train_actor()
            
            if global_step % args.target_network_frequency == 0:
                agent.Update_QTargetNetworks()

        global_step+=1

        if global_step % 500 == 0:
            print("global step ", global_step)


    simulation_app.close()
        


        


    


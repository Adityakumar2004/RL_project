import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ppo on factory env")
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to spawn.")

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
from torch.distributions.normal import Normal

# import tyro
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from isaaclab_tasks.utils import parse_env_cfg
import wandb
import imageio


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
    def __init__(self, env, video_folder:str | None =None, output_type: str = "numpy", enable_normalization = False):
        super().__init__(env)
        self.env = env
        self.total_obs_space = {"policy": env.observation_space, "critic": env.state_space}
        self.output_type = output_type

        self.enable_normalization = enable_normalization
        self.training = True

        if enable_normalization:
            self.normalizers = {
                "policy": RunningNormalizer(self.total_obs_space["policy"].shape[-1]),
                "critic": RunningNormalizer(self.total_obs_space["critic"].shape[-1]),
            }


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

        if self.enable_normalization:
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


        if self.enable_normalization:
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
     
    env = env_wrapper(env, video_folder, output_type=output_type)
    
    return env


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
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 10_00_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = args_cli.num_envs
    """the number of parallel game environments"""
    num_steps: int = 400
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(env.total_obs_space["critic"].shape[-1], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(env.total_obs_space["policy"].shape[-1], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.shape[-1]), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.action_space.shape[-1]))

    def get_value(self, x):
        return self.critic(x)


    def get_action_and_value(self, obs, action=None):
        """
        note the action that can be passed to this function should be the raw action 
        """
        action_mean = self.actor_mean(obs["policy"])
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        
        tanh_action = torch.tanh(action)

        ## log prob correction 
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs["critic"])

        return action, tanh_action, log_prob, entropy, value
        # return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(obs["critic"])

def TestingAgent(env, agent: Agent, num_episodes = 2, recording_enabled=True):
    with torch.no_grad():
        env.eval()  # set the env to evaluation mode
        # obs, _ = env.reset()
        total_reward = 0.0
        step_cntr = 0
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, tanh_action, _, _, _ = agent.get_action_and_value(obs)
                next_obs, rewards, terminations, truncations = env.step(tanh_action)
                
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.cpu().numpy()
                if isinstance(terminations, torch.Tensor):
                    terminations = terminations.cpu().numpy()
                if isinstance(truncations, torch.Tensor):
                    truncations = truncations.cpu().numpy()

                obs = next_obs
                total_reward += rewards.mean()
                done = np.any(terminations) or np.any(truncations)
                # done = terminations | truncations
                step_cntr += 1

                ## recording the video
                if recording_enabled:
                    env.record_cameras()

                
    print(f"avg reward over {step_cntr / num_episodes} steps: {total_reward / num_episodes:.2f}")
    return total_reward / num_episodes


"""

                                num_iterations
                                       
        collects the data for a total of num_steps in parallellely in all the envs, 

        calculates the advantages and returns afte the end of num_steps

        updates the policy and value networks for update_epochs number of times

                                            |
                                            |
                                            v
                num_steps <-----------------+------------------> update_epochs
                                                                   
 in each step the data is collected         divides the total batch_size of data into minibatches of size minibatch_size
                                            in each epoch this data is randomly shuffled among the minibatches 
                                            and all the minibatches are used for training
                                                                    |
                                                                    |
                                                                    v
                                                    looping over the minibatches
                 
"""


if __name__ == "__main__":
    
    ## start the env 
    video_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "videos_tanh")
    checkpoint_folder = os.path.join("custom_scripts", "logs", "ppo_factory", "checkpoints")
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "cp_tanh.pt")
    
    args = Args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    envs = make_env(video_folder, output_type="torch")
    envs.train()  # set the env to training mode

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    tracking_enabled = True
    if tracking_enabled:
        wandb.init(
            project = "Space_RL",
            name = f"ppo_Factory_tanh_{int(time.time())}"
            
        )


    
    args.batch_size = int(args_cli.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size



    obs = {"policy": torch.zeros((args.num_steps, args_cli.num_envs, envs.total_obs_space["policy"].shape[-1]), device=device),
           "critic": torch.zeros((args.num_steps, args_cli.num_envs, envs.total_obs_space["critic"].shape[-1]), device=device)}
    
    actions = torch.zeros((args.num_steps, args_cli.num_envs, envs.action_space.shape[-1]), device=device)
    log_probs = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args_cli.num_envs), device=device)
    values = torch.zeros((args.num_steps, args_cli.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_done = torch.zeros((args_cli.num_envs), device=device)



    for iteration in range(args.num_iterations):
        envs.train()
        if args.anneal_lr:
            frac = 1.0 - (iteration -1) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            ## doubt about this step 


        for step in range(args.num_steps):
            global_step += args_cli.num_envs
           
            obs["policy"][step] = next_obs["policy"]
            obs["critic"][step] = next_obs["critic"]

            dones[step] = next_done
            
            with torch.no_grad():
                action, tanh_action, log_prob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                ## doubt about the value.flatten            

            actions[step] = action
            log_probs[step] = log_prob

            next_obs, reward, terminated, truncated = envs.step(tanh_action)
            next_done = terminated | truncated
            rewards[step] = reward
            
        with torch.no_grad():
            next_value = agent.get_value(next_obs["critic"]).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values


        b_obs = {
            "policy": obs["policy"].reshape(-1, envs.total_obs_space["policy"].shape[-1]),
            "critic": obs["critic"].reshape(-1, envs.total_obs_space["critic"].shape[-1]),
        }

        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1, envs.action_space.shape[-1])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        ## optmizing the policy and value nn

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                
                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = -(logratio).mean()
                    approx_kl = (ratio - 1 - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                ## policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()


                ## value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
        avg_reward = TestingAgent(envs, agent, num_episodes=2, recording_enabled=True)
        checkpoint = {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        wandb.log({"avg_reward": avg_reward}, step=global_step)
        print(f"Iteration {iteration + 1}/{args.num_iterations}, Global Step: {global_step}, Avg Reward: {avg_reward:.2f}, Time: {time.time() - start_time:.2f}s")



    envs.close()
    wandb.finish()




## scaling rewards, clipping them in the range -10 to 10 
## tanh seqeezed implementation details also entropy in tanh squeezed








            



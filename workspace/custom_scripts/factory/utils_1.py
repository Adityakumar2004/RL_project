import numpy as np
import imageio
import os
import random
import time
import gymnasium as gym
import torch
import torch.nn as nn


from agents_defn import Agent

import csv

class RunningNormalizer:
    def __init__(self, size, epsilon=1e-5, clip_range=5.0):
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
        
        ## custom rewards
        rewards = self.calculate_rewards(rewards_task2=rewards, output_type=self.output_type)


        
        # Capture success rates from environment extras
        if info is not None:
            if "successes" in info:
                info_custom["success_rate"] = info["successes"]
            if "success_times" in info:
                info_custom["success_times"] = info["success_times"]

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
            rewards = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()

        if self.enable_recording:
            for camera in self.unwrapped.cameras:
                camera.update(self.unwrapped.step_dt)
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
            if self.unwrapped.num_envs > 1:
                env_id = np.random.randint(1, self.unwrapped.num_envs)
            else:
                env_id = 0
            print(f"[INFO]: Randomly selected env_id for cam2: {env_id}")
            self.set_camera_pose_fixed_asset(1, env_id)
            

        return obs, info
    
    def set_camera_pose_fixed_asset(self, camera_id, env_id):

        fixed_asset_default_state =  self.unwrapped._fixed_asset.data.default_root_state[env_id].clone()
        camera_target = fixed_asset_default_state[:3] + torch.tensor([0.0, 0.0, 0.005], device=self.unwrapped.device) + self.scene.env_origins[env_id]
        eye_camera = camera_target + torch.tensor([0.5, -0.9, 0.3], device= self.unwrapped.device)

        self.unwrapped.cameras[camera_id].set_world_poses_from_view(eye_camera.unsqueeze(0), camera_target.unsqueeze(0))
        
    def record_cameras(self):
        frame_dict = {}
        for i, camera in enumerate(self.unwrapped.cameras):

            cam_image = camera.data.output["rgb"]
            if isinstance(cam_image, torch.Tensor):
                cam_image = cam_image.detach().cpu().numpy()
            
            ## removing batch dimension
            if cam_image.shape[0] == 1:
                cam_image = cam_image[0]

            frame_dict[f"camera_{i}"] = cam_image

            cam_image = cam_image.astype(np.uint8)
            self.vid_writers[i].append_data(cam_image)

        return frame_dict

    def train(self):
        print("[INFO] Environment set to TRAINING mode.")
        self.training = True
    
    def eval(self):
        print("[INFO] Environment set to EVALUATION mode.")
        self.training = False

    def calculate_rewards(self, output_type, rewards_task2):

        asset_info = self.unwrapped.get_asset_information()
        held_asset_coords = asset_info['held_asset_bottom_coords']

        hole_center_coords = asset_info["hole_center_coords"]
        radius = asset_info["fixed_asset_diameter"]/2

        reward = reward_function(rewards_task2, hole_center_coords, held_asset_coords, xy_threshold = (1.5*radius)**2, z_epsilon = 0.006, alpha = 100.0, beta = 100)

        if output_type == "numpy":
            if isinstance(reward, torch.Tensor):
                print("i am here in the torch type reward")
                reward = reward.cpu().numpy()
        elif output_type == "torch":
            if isinstance(reward, np.ndarray):
                reward = torch.tensor(reward, device=self.env.device)

        return reward


def reward_function(rewards_task2, x_desired, x_current, xy_threshold, z_epsilon = 0.006, alpha = 15.0, beta = 50):
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
    
    if isinstance(rewards_task2, torch.Tensor):
        rewards_task2 = rewards_task2.cpu().numpy()
    elif isinstance(rewards_task2, np.ndarray):
        pass
    else:
        raise AssertionError("rewards_task2 must be torch.Tensor or np.ndarray")
    
    squared_x = x*x

    norm_squared_xy = np.sum(squared_x[:,:2], axis=-1)  
    reward = np.exp(-alpha * norm_squared_xy)/4

    mask_xy = norm_squared_xy < xy_threshold
    z_term = np.exp(-beta * squared_x[:,2])/4
    # reward += np.where(mask_xy, 0.25, z_term)


    ## this reward worked very good for task1 only havent checked in commbination with task2 

    # reward += z_term
    # larger_mask = np.sum(squared_x, axis=-1) < 0.00004    
    # reward += np.where(larger_mask, 0.25, 0.0)
    
    ## reward for task 1 and task 2 combined
    z_mask =  x_current[:, 2] < x_desired[:, 2] + z_epsilon

    total_mask = mask_xy & z_mask

    reward += np.where(total_mask, 0.5, z_term) + np.where(total_mask, rewards_task2, 0.0)

    

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


def TestingAgent(env, device, num_envs, agent, checkpoint_path, num_episodes = 2, recording_enabled=True, critic_normalization = True, sim_step_func=None, **kwargs):
    
    # file_path = os.path.join("custom_scripts", "logs", "ppo_factory", "csv_files", "log_values_ik.csv")

    file_path = kwargs["file_path"]

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

    # if critic_normalization:
    #     agent.critic_normalizer.load_state_dict(checkpoint["critic_normalizer"])

    print(f"Loaded checkpoint from {checkpoint_path}")

    

    with torch.no_grad():
        env.eval()  # set the env to evaluation mode
        # obs, _ = env.reset()
        total_reward = 0.0
        total_raw_reward = 0.0
        step_cntr = 0
        # if critic_normalization:
        #     agent.critic_normalizer.eval() ## no updates in mean

        rewards_list = []
        raw_reward_list = []
        all_success_rates = []



        for i in range(num_episodes):
            env.eval()
            obs, _ = env.reset()
            done_flag = False

            next_lstm_state_actor = (
            torch.zeros(agent.num_layers, num_envs, agent.hidden_size).to(device),
            torch.zeros(agent.num_layers, num_envs, agent.hidden_size).to(device),
            )
            dones = torch.zeros((env.unwrapped.num_envs), device = device)
            
            # step_cntr = 0
            if sim_step_func is not None:
                sim_step_func(env=env, step = step_cntr, file_path = file_path)

            while not done_flag:
                step_cntr += 1
                actions, _, _, next_lstm_state_actor, a_mu, a_std= agent.get_action(obs, next_lstm_state_actor, dones)

                if sim_step_func is not None:
                    sim_step_func(env=env, step = step_cntr, file_path = file_path)

                next_obs, rewards, terminations, truncations, info_custom = env.step(actions)
                
                raw_rewards = info_custom["org_reward"]

                dones = (terminations | truncations).float()

                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.cpu().numpy()
                if isinstance(raw_rewards, torch.Tensor):
                    raw_rewards = raw_rewards.cpu().numpy()
                if isinstance(terminations, torch.Tensor):
                    terminations = terminations.cpu().numpy()
                if isinstance(truncations, torch.Tensor):
                    truncations = truncations.cpu().numpy()


                obs = next_obs
                total_reward += rewards.mean()
                total_raw_reward += raw_rewards.mean()
                done_flag = bool(np.any(terminations) or np.any(truncations))
                
                done_np = (terminations | truncations).cpu().numpy() if isinstance(terminations, torch.Tensor) else (terminations | truncations)

                if np.any(done_np):
                    if 'success_rate' in info_custom:
                        success_rate = info_custom['success_rate']
                        if isinstance(success_rate, torch.Tensor):
                            success_rate = success_rate.cpu().numpy()
                        # Add the success rate for each completed episode
                        for _ in range(done_np.sum()):
                            all_success_rates.append(success_rate)

                    print(f"ep {i} average_total_reward: {total_reward}, avg_raw_reward: {total_raw_reward} success_rate: {success_rate}")
                    rewards_list.append(total_reward)
                    raw_reward_list.append(total_raw_reward)
                    total_reward = 0
                    total_raw_reward = 0

                # step_cntr += 1

                ## recording the video
                if recording_enabled:
                    env.record_cameras()
                
                

    
    avg_reward = np.mean(rewards_list[:])
    avg_raw_reward = np.mean(raw_reward_list[:])
    # print(f"avg reward over {step_cntr / num_episodes} steps: {total_reward / num_episodes:.2f}")
    print(f"avg reward over {i} episodes : {avg_reward}, raw_reward: {avg_raw_reward} success_rate {np.mean(all_success_rates[:])} ")
    return avg_reward

class AdaptiveScheduler():
    def __init__(self, kl_threshold = 0.008):
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu)**2)/(2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    # print("kl1 shape", kl.shape)
    kl = kl.sum(dim=-1) # returning mean between all steps of sum between all actions
    # print("kl2 shape", kl.shape)
    if reduce:
        return kl.mean()
    else:
        return kl

def bound_loss(mu):

    soft_bound = 1.1
    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
    b_loss = b_loss.mean()

    return b_loss


## utils for analysis 
def target_dof_torque_log(env, step, file_path:str):

    joint_torques = env.unwrapped.joint_torque[0,:7].cpu().numpy().tolist()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            header = ['step'] + [f"joint_{i}" for i in range(len(joint_torques))]
            print("writing the header")
            writer.writerow(header)
        
        writer.writerow([step] + joint_torques)

def log_values(env, step, file_path:str):
    '''this function logs all the values of the logging_values dict for the 0th env id'''

    logging_values = env.unwrapped.logging_values
    joint_torques = logging_values.get("dof_torques", np.zeros((1, 7)))[0].tolist()
    raw_actions = logging_values.get("raw_action", np.zeros((1, 3)))[0, :3].tolist()
    processed_actions = logging_values.get("processed_action", np.zeros((1, 3)))[0, :3].tolist()
    target_fingertip_pos = logging_values.get("target_fingertip_pos", np.zeros((1, 3)))[0].tolist()
    fixed_pos_obs_frame = logging_values.get("fixed_pos_obs_frame", np.zeros((1, 3)))[0].tolist()
    fingertip_midpoint_pos = logging_values.get("fingertip_midpoint_pos", np.zeros((1, 3)))[0].tolist()
    current_joint_pos = logging_values.get("current_joint_pos", np.zeros((1, 7)))[0].tolist()
    applied_dof_torque = logging_values.get("applied_dof_torque", np.zeros((1, 7)))[0].tolist()
    
    if "target_joint_pos" in logging_values:
        target_joint_pos = logging_values.get("target_joint_pos", np.zeros((1, 7)))[0].tolist()

    if "applied_wrench" in logging_values:
        applied_wrench = logging_values.get("applied_wrench", np.zeros((1, 6)))[0].tolist()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            header = ["step"]
            header = header + [f"dof_torque_{i}" for i in range(len(joint_torques))]
            header = header + [f"raw_action_{i}" for i in range(len(raw_actions))]
            header = header + [f"processed_action_{i}" for i in range(len(processed_actions))]
            header = header + [f"target_fingertip_pos_{i}" for i in range(len(target_fingertip_pos))]
            header = header + [f"fixed_pos_obs_frame_{i}" for i in range(len(fixed_pos_obs_frame))]
            header = header + [f"fingertip_midpoint_pos_{i}" for i in range(len(fingertip_midpoint_pos))]
            header = header + [f"current_joint_pos_{i}" for i in range(len(current_joint_pos))]
            header = header + [f"applied_dof_torque_{i}" for i in range(len(applied_dof_torque))]


            if "target_joint_pos" in logging_values:
                header = header + [f"target_joint_pos_{i}" for i in range(len(target_joint_pos))]

            if "applied_wrench" in logging_values:
                header = header + [f"applied_wrench_{i}" for i in range(len(applied_wrench))]

            writer.writerow(header)

        writer.writerow([step] 
                        + joint_torques
                        + raw_actions
                        + processed_actions
                        + target_fingertip_pos
                        + fixed_pos_obs_frame
                        + fingertip_midpoint_pos
                        + current_joint_pos
                        + applied_dof_torque
                        + (
                            target_joint_pos
                            if "target_joint_pos" in logging_values
                            else []
                          )
                        + (applied_wrench if "applied_wrench" in logging_values else [])
                        )

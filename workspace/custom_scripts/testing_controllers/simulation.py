## CUDA_VISIBLE_DEVICES=1 python custom_scripts/testing_controllers/simulation.py --headless

import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="factory Gym environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the simulation on.")

# parser.add_argument("--video", action="store_true", help="Enable video recording during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=400, help="Interval between video recordings (in steps).")
parser.add_argument("--replay", type=bool, default=False, help="Replay from log file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# from utils import create_marker_spheres, visualize_spheres 
from typing import Union
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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

# import time
# from utils_1 import env_wrapper

## utils
def create_marker_spheres(env, count, color=(1.0, 0.0, 0.0), radius = 0.001):

    sphere_markers = {}
    for i in range(count):
        sphere_markers[f"sphere_{i}"] = sim_utils.SphereCfg(
            radius = radius,#0.001,
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = color),
        )
    
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ShpereMarkers",
        markers=sphere_markers

    )

    env_marker_visualizer = VisualizationMarkers(marker_cfg)
    return(env_marker_visualizer)

def visualize_markers(env, env_marker_visualizer:VisualizationMarkers, pose: Union[np.ndarray, torch.Tensor], quat = None):
    '''
    Args:
    pose: tensor or numpy array of positions with dim --> (num_envs, num_spheres, 3) or (num_envs, 3) 
    
    '''
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    elif isinstance(pose, np.ndarray):
        pass
    else :
        assert False, "pose must be a torch.Tensor or np.ndarray"
    
    if quat is None:
        identity_quat = np.array([1, 0, 0, 0])  # identity quat for sphere

    if len(pose.shape) == 3:
        (num_envs, num_spheres, _) = pose.shape
    elif len(pose.shape) == 2:
        (num_envs, _) = pose.shape
        num_spheres = 1
        pose = pose[:, None, :]  # Add extra dimension to make it (num_envs, 1, _)
    else:
        raise ValueError(f"pose must have 2 or 3 dimensions, got shape {pose.shape}")

    if quat is not None:    
        if len(quat.shape) == 3:
            (num_envs, num_spheres, _) = quat.shape
        elif len(quat.shape) == 2:
            (num_envs, _) = quat.shape
            num_spheres = 1
            quat = quat[:, None, :]  # Add extra dimension to make it (num_envs, 1, _)
        else:
            raise ValueError(f"pose must have 2 or 3 dimensions, got shape {quat.shape}")

    translations = np.empty((num_envs * num_spheres, 3), dtype=np.float32)
    orientations = np.empty((num_envs * num_spheres, 4), dtype=np.float32)
    marker_indices = np.empty((num_envs * num_spheres,), dtype=np.int32)

    for env_id in range(num_envs):
        for count in range(num_spheres):
            translations[(num_spheres*env_id + count)] = pose[env_id, count, :3]
            if quat is None:
                orientations[(num_spheres*env_id + count)] = identity_quat
            else:
                orientations[(num_spheres*env_id + count)] = quat[env_id, count, :]
            marker_indices[(num_spheres*env_id + count)] = count

    env_marker_visualizer.visualize(
        translations=translations,
        orientations=orientations,
        marker_indices=marker_indices
    )

def create_marker_frames(env, count=1, scale=(0.01,0.01,0.01)):
    frame_markers = {}
    for i in range(count):
        frame_markers[f"frame_{i}"] = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=scale,
            )
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FrameMarkers",
        markers = frame_markers
    )

    marker_visualizer = VisualizationMarkers(marker_cfg)

    return marker_visualizer

def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.testing_controllers.factory_env_task_space:RobotEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.testing_controllers.factory_env_task_space:RobotEnvCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    # env = env_wrapper(env, video_folder, output_type=output_type, enable_normalization_rewards=False)
    
    return env

# def generate_trajectory(time):
#     x = 0.05
#     y = 0.05
#     z = 0.05
#     return x,y,z

# def error_generation(time):
#     dx = 0.0
#     dy = 0.0
#     dz = 0.0
#     dx_angle = 0.0
#     dy_angle = 0.0
#     dz_angle = 0.0
#     del_pose = np.array([[dx, dy, dz, dx_angle, dy_angle, dz_angle, 1.0]])
#     return del_pose

class handling_log:
    def __init__(self):
        self.trajectory = {}
    
    def log_value(self, key, value):
        
        if isinstance(value, torch.Tensor):
            value = value.clone().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            assert False, "value must be a torch.Tensor or np.ndarray"

        self.trajectory[key] = value

    def log_trajectory(self, key, value):
        if key not in self.trajectory.keys():
            self.trajectory[key] = []

        if isinstance(value, torch.Tensor):
            value = value.clone().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            assert False, "value must be a torch.Tensor or np.ndarray"

        self.trajectory[key].append(value)
    
    def save_log(self, file_path):
        # file_path = "logs/trajectory_run1.npz"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # create parent folder
        np.savez(file_path, **self.trajectory)
        print(f"Trajectory saved to {file_path}")

    def load_log(self, file_path):
        loaded = np.load(file_path, allow_pickle=True)
        self.data = {k: loaded[k] for k in loaded.keys()}
        # self.data = np.load(file_path)
        print(f"Trajectory loaded from {file_path}")
    
    def get_log_value(self, key, output_type = "numpy"):
        assert key in self.data.keys(), f"{key} not found in trajectory"
        if output_type == "numpy":
            return self.data[key]
        elif output_type == "torch":
            return torch.tensor(self.data[key], dtype=torch.float32)
        else:
            assert False, "output_type must be 'numpy' or 'torch'"
    
    def get_log_trajectory(self, key, index, output_type = "numpy"):
        assert key in self.data.keys(), f"{key} not found in data"
        assert index < len(self.data[key]), f"index {index} out of range for data {key} with length {len(self.data[key])}"
        if output_type == "numpy":
            return self.data[key][index]
        elif output_type == "torch":
            return torch.tensor(self.data[key][index], dtype=torch.float32)
        else:
            assert False, "output_type must be 'numpy' or 'torch'"


def main():

    env = make_env(video_folder=None, output_type="numpy")
    exp_name = "test_run"
    log_path = os.path.join("custom_scripts", "testing_controllers", "logs", f"{exp_name}_trajectory.npz")
    # replay = True
    replay = args_cli.replay
    obs, info = env.reset()

    log_handler = handling_log()
    log_handler.log_value("initial_joint_pose", env.unwrapped._robot.data.default_joint_pos)


    fingertip_viz = create_marker_spheres(env, 1, radius=0.008)
    fingertip_frame_viz = create_marker_frames(env, 1)

    episode = 0
    step = 0
    flag = True
    
    actions = torch.zeros((args_cli.num_envs, 7), dtype=torch.float32)
    start_time = time.time()
    if replay:
        log_handler.load_log(log_path)
    while episode<5:
        # actions = torch.zeros((args_cli.num_envs, 7), dtype=torch.float32)
        # actions[:,-1] = torch.tensor(1.0) if flag else torch.tensor(0.0)
        if replay == True:
            
            actions = log_handler.get_log_trajectory("raw_actions", step, output_type="torch")
            # print("relayed actions ")
            # print(actions)
        else:
            actions[:, 6] = torch.tensor(1.0) if flag else torch.tensor(0.0)
            # print(actions)
        if step % 8 == 0:
            flag = not flag

        
        # actions = np.repeat(actions, args_cli.num_envs, axis=0)
        # actions = torch.tensor(actions, dtype=torch.float32)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        if not replay:
            log_handler.log_trajectory("raw_actions", env.unwrapped.log_values["raw_action"])

        terminations = terminations.cpu().clone().numpy()
        truncations = truncations.cpu().clone().numpy()

        fingertip_pos = env.unwrapped.fingertip_midpoint_pos.clone().cpu()
        fingertip_quat = env.unwrapped.fingertip_midpoint_quat.clone().cpu()
        # print("this is the quat ",fingertip_quat[0,:])
        visualize_markers(env, fingertip_viz, fingertip_pos)
        visualize_markers(env, fingertip_frame_viz, fingertip_pos, fingertip_quat)
        # print(truncations, '\n' ,terminations)
        dones = np.logical_or(terminations, truncations)
        # print(dones)
        
        if step%30 == 0:
            print(f"time at {step} is for replay {replay}", time.time() - start_time)
            # if not replay:
            #     log_handler.save_log(log_path)

        if np.any(dones):
            obs, info = env.reset()
            episode += 1
            step = 0
        step += 1



    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


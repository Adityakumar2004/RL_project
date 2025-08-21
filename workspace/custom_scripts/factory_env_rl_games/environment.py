import gymnasium as gym


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


from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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



class visualize_sphere_env_markers:

    def __init__(env, count):
        self.env = env
        self.num_envs = self.env.unwrapped.num_envs

    def create_marker_spheres(self, count, color=(1.0, 0.0, 0.0), radius = 0.001):

        sphere_markers = {}
        for i in range(count):
            sphere_markers[f"sphere_{i}"] = sim_utils.SphereCfg(
                radius = radius,#0.001,
                visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = color),
            )
        
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/envMarkers",
            markers=sphere_markers

        )

        env_marker_visualizer = VisualizationMarkers(marker_cfg)
        self.env_marker_visualizer = env_marker_visualizer
        return(env_marker_visualizer)

    def visualize_spheres(self, pose):
        
        num_envs = self.num_envs
        env_marker_visualizer = self.env_marker_visualizer
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        elif isinstance(pose, np.ndarray):
            pass
        else :
            assert False, "pose must be a torch.Tensor or np.ndarray"
        
        identity_quat = np.array([1, 0, 0, 0])  # identity quat for sphere

        if len(pose.shape) == 3:
            (num_envs, num_spheres, _) = pose.shape
        elif len(pose.shape) == 2:
            (num_envs, _) = pose.shape
            num_spheres = 1
            pose = pose[:, None, :]  # Add extra dimension to make it (num_envs, 1, _)
        else:
            raise ValueError(f"pose must have 2 or 3 dimensions, got shape {pose.shape}")

        translations = np.empty((num_envs * num_spheres, 3), dtype=np.float32)
        orientations = np.empty((num_envs * num_spheres, 4), dtype=np.float32)
        marker_indices = np.empty((num_envs * num_spheres,), dtype=np.int32)

        for env_id in range(num_envs):
            for count in range(num_spheres):
                translations[(num_spheres*env_id + count)] = pose[env_id, count, :3]
                orientations[(num_spheres*env_id + count)] = identity_quat
                marker_indices[(num_spheres*env_id + count)] = count

        env_marker_visualizer.visualize(
            translations=translations,
            orientations=orientations,
            marker_indices=marker_indices
        )



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

    return env


def main():

    video_folder = os.path.join("custom_scripts", "logs", "trial_videos", "tr1_vid")
    env = make_env(video_folder)

    # print(f"[INFO]: Gym observation space: {env.observation_space}")
    # print(f"[INFO]: Gym single Observation space shape: {env.single_observation_space.shape}")
    # print(f"[INFO]: Gym action space: {env.action_space}")
    # print(f"[INFO]: Gym single action space shape: {env.unwrapped.single_action_space.shape}")
    # print(f"**"*10)
    # print(f"[INFO]: Single action space high: {env.single_action_space.high}")
    # print(f"[INFO]: Single action space low: {env.unwrapped.single_action_space.low}")
    # print(f"[INFO]: gym state space : {env.state_space}")

    obs, info = env.reset()
    step = 0
    while simulation_app.is_running():

        actions = torch.zeros(env.action_space.shape, device=env.device)
        print(step)
        env.step(actions)
        step+=1

        if video_folder is not None:
            env.record_cameras()


if __name__ == "__main__":
    # main()
    main()
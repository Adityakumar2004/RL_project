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

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import torch
import os


import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
import imageio

class env_wrapper(gym.Wrapper):
    def __init__(self, env, video_folder:str | None =None):
        super().__init__(env)
        self.env = env

        self.enable_recording = False
        if video_folder is not None:
            self.enable_recording = True
            os.makedirs(video_folder, exist_ok=True)
            self.vid_writers = []
            self.step_cntr = 0

            self.camera_flag = 0
            self.recording_step = 0
            self.video_length = 100
            self.record_freq = 300   
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
            
            if self.step_cntr % self.record_freq == 0 or self.camera_flag == 1:
                if self.camera_flag == 0:
                    print("[INFO]: Recording video...")
                    env_id = np.random.randint(1, self.unwrapped.num_envs)
                    print(f"[INFO]: Randomly selected env_id for cam2: {env_id}")

                    self.set_camera_pose_fixed_asset(1, env_id)

                    self.camera_flag = 1
                    self.recording_step = 0
                if self.recording_step < self.video_length:
                    self.record_cameras()
                    self.recording_step += 1
                else:
                    self.camera_flag = 0

            self.step_cntr+=1
        return obs, rewards, terminations, truncations
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = {k: v.cpu().numpy() for k, v in obs.items()}
        
        ## for marker visualization
        # self.unwrapped.create_markers()

        if self.enable_recording:
            self.step_cntr = 0
            self.camera_flag = 0
            self.recording_step = 0
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
            
class normalizer()
def make_env(video_folder:str | None =None):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.factory.factory_env_kinova:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.factory.factory_env_cfg_kinovo:FactoryTaskPegInsertCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    env = env_wrapper(env, video_folder)
    
    
    return env


def main():
    
    video_folder = os.path.join("custom_scripts", "logs", "sac_factory", "videos2")
    
    env = make_env(video_folder)
    # env = make_env()

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym single Observation space shape: {env.single_observation_space.shape}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Gym single action space shape: {env.unwrapped.single_action_space.shape}")
    print(f"**"*10)
    print(f"[INFO]: Single action space high: {env.single_action_space.high}")
    print(f"[INFO]: Single action space low: {env.unwrapped.single_action_space.low}")
    print(f"[INFO]: gym state space : {env.state_space}")


    # ## setting the camera pose
    # fixed_asset_default_state =  env.unwrapped._fixed_asset.data.default_root_state[0].clone()
    # camera_target = fixed_asset_default_state[:3] + torch.tensor([0.0, 0.0, 0.005], device=env.unwrapped.device) + env.scene.env_origins[0]
    # eye_camera = camera_target + torch.tensor([0.5, -0.9, 0.3], device= env.unwrapped.device)
    

    # env.unwrapped.camera1.set_world_poses_from_view(eye_camera.unsqueeze(0), camera_target.unsqueeze(0))

    env.set_camera_pose_fixed_asset(0, env_id=0)
    # reset environment
    obs, info = env.reset()
    # simulate environment
    print("max length of episode ", env.unwrapped.max_episode_length)
    print("action space shape ", env.action_space.shape)
    print(env.unwrapped.scene["robot"].joint_names)


    # print(env.unwrapped.actions.action_term_groups["joint_efforts"])
    # If you want to inspect available action terms, try accessing them via the scene or print available attributes:
    # print(dir(env.unwrapped))
    # Or, for example, if the scene has action terms:
    # print(dir(env.unwrapped.action_manager))
    # print("checking the variables: decimation, episode_length_s, sim.dt in env ", env.unwrapped.episode_length_s/(env.unwrapped.sim.dt*env.unwrapped.decimation))

    
    camera_flag = 0
    recording_step = 0
    video_length = 100
    record_freq = 1   


    flag = 0
    cnt = 0


    joint_efforts = env.unwrapped.scene["robot"].data.applied_torque
    num_joints = joint_efforts.shape[-1]
    print(f"Number of joints: {num_joints}")
    print(f"Joint values: {joint_efforts.cpu().numpy()[0:]}")



    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # actions = np.random.uniform(-1,1,env.action_space.shape)
            
            if cnt % 20 == 0:
                if flag == 0:
                    actions = 1*torch.ones(env.action_space.shape, device=env.device)
                    flag = 1
                else:
                    actions = -1*torch.ones(env.action_space.shape, device=env.device)
                    flag = 0
                # print("[INFO]: Current observations: \n", obs)
                # print("[INFO]: terminations and truncations \n",terminations, "\n",truncations)
                # print("actions \n",actions)
                # print("[INFO]: infos: \n", infos)
                print("count :", cnt)
            
            
            
            # apply actions
            next_obs, rewards, terminations, truncations =  env.step(actions)

            # env.unwrapped.scene["robot"].
            
            # env_ids = np.arange(env.unwrapped.num_envs)
            # env.unwrapped.visualize_env_markers()

            # if terminations.any() or truncations.any():
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            #     # reset the environment
            #     # env.reset()
            #     print("[INFO]: Current observations: \n", next_obs)
            #     print("[INFO]: terminations and truncations \n",terminations, "\n",truncations)
            #     # print("[INFO]: infos: \n", infos)
            #     print("count :", cnt)
            #     cnt = 0


        cnt+= 1

        # Print the number of joints and their current values

            # print(env.unwrapped.episode_length_buf)
            
        

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


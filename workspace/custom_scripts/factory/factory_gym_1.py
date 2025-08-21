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

    def calculate_rewards(self):

        asset_info = self.unwrapped.get_asset_information()
        held_asset_coords = asset_info['held_asset_bottom_coords']

        hole_center_coords = asset_info["hole_center_coords"]
        radius = asset_info["fixed_asset_diameter"]/2

        reward = reward_function(hole_center_coords, held_asset_coords, xy_threshold = (1.5*radius)**2, alpha = 15.0, beta = 50)
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
    reward += np.where(mask, 0.25, z_term)
    

    return reward



def create_marker_spheres(env, count, color=(1.0, 0.0, 0.0), radius = 0.001):

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
    return(env_marker_visualizer)

def visualize_spheres(env, env_marker_visualizer, pose):
    
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


def make_env(video_folder:str | None =None):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        # entry_point="custom_scripts.factory.factory_env_kinova:FactoryEnv",
        entry_point="custom_scripts.factory.factory_env_markers:FactoryEnv",
        disable_env_checker=True,
        kwargs={
            # "env_cfg_entry_point":"custom_scripts.factory.factory_env_cfg_kinovo:FactoryTaskPegInsertCfg",
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
 

def main():
    
    video_folder = os.path.join("custom_scripts", "logs", "sac_factory", "videos2")
    
    # env = make_env(video_folder)
    env = make_env()

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
    # env.unwrapped.visualize_env_markers()
    # env.unwrapped.create_marker_spheres(4)
    # held_asset_viz = create_marker_spheres(env, 4)
    # fixed_asset_viz = create_marker_spheres(env, 4, (0.0, 1.0, 0.0))
    held_asset_coord_viz = create_marker_spheres(env, 1, (0.0, 0.0, 1.0), radius = 0.005) ## blue
    custom_noised_peg_tip_viz = create_marker_spheres(env, 1, (1.0, 0.0, 0.0), radius = 0.005) ## red
    unnoised_peg_tip_viz = create_marker_spheres(env, 1, (0.0, 1.0, 0.0), radius = 0.05) ## green

    fixed_asset_coord_viz = create_marker_spheres(env, 3, (0.0, 1.0, 0.0))
    

    
    camera_flag = 0
    recording_step = 0
    video_length = 100
    record_freq = 1   


    flag = 0
    cnt = 0


    joint_efforts = env.unwrapped.scene["robot"].data.applied_torque
    num_joints = joint_efforts.shape[-1]
    # print(f"Number of joints: {num_joints}")
    # print(f"Joint values: {joint_efforts.cpu().numpy()[0:]}")



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
            
            actions = torch.zeros(env.action_space.shape, device=env.device)
            # print("keypoints_held: \n", env.unwrapped.keypoints_held)

            # print(env.unwrapped.keypoints_held.shape)
            # print(len(env.unwrapped.keypoints_fixed.shape))
            # print("keypoints_fixed: \n", env.unwrapped.keypoints_fixed)
            
            # print("----"*10)
            # print(env.unwrapped.scene.env_origins.shape)

            asset_info = env.unwrapped.get_asset_information()
            held_asset_coords = asset_info['held_asset_bottom_coords']

            fixed_asset_coords = asset_info["hole_center_coords"]

            radius = asset_info["fixed_asset_diameter"]/2

            expr1 = "5 * radius"
            expr2 = "3*radius + radius/2"
            r1 = round(float(eval(expr1)), 5)
            r2 = round(float(eval(expr2)), 5)

            fixed_asset_radii_point = fixed_asset_coords + np.array([0, r1, 0])
            fixed_asset_radii_point_half = fixed_asset_coords + np.array([0, r2, 0])

            fingertip_midpoint_pos = env.unwrapped.fingertip_midpoint_pos
            fingertip_midpoint_pos = fingertip_midpoint_pos + env.unwrapped.scene.env_origins
            fingertip_midpoint_pos = fingertip_midpoint_pos.cpu().numpy()

            stacked_hole_pts = np.stack([fixed_asset_coords, fixed_asset_radii_point, fixed_asset_radii_point_half], axis=1)

            # print(f"radius {radius}, {expr1} {r1}, {expr2} {r2}")
            
            # print("fixed_asset_height ", asset_info["fixed_asset_height"])
            # print("held_asset_height ", asset_info["held_asset_height"] )
            # print("fingertip_midpoint_pos ", fingertip_midpoint_pos)
            
            custom_noised_fingertip_midpoint_positions = env.unwrapped.custom_noised_fingertip_midpoint_positions.cpu().numpy()
            custom_noised_peg_tip_positions = env.unwrapped.custom_noised_peg_tip_positions.cpu().numpy()
            # unnoised_peg_tip_positions = env.unwrapped.unnoised_peg_tip_pos.cpu().numpy()
            # unnoised_fingertip_midpoint_positions = env.unwrapped.unnoised_fingertip_midpoint_positions.cpu().numpy()
            # print("custom_positons ", custom_positions)


            origins = env.unwrapped.scene.env_origins.cpu().numpy()
            position = np.zeros((env.unwrapped.num_envs, 3))
            position[:, 2] = 2
            position = position + origins

            fixed_asset_pos = env.unwrapped._fixed_asset.data.default_root_state.clone().cpu().numpy()
            fixed_asset_pos = fixed_asset_pos[:,:3]
            
            # print("noised_peg_tip_pos \n", custom_noised_peg_tip_positions)
            # print("noised_fingertip_midpoint_pos \n", custom_noised_fingertip_midpoint_positions)

            fixed_state = env.unwrapped.fixed_state[:,:3].cpu().numpy()

            # print("fixed_asset_pos \n ", )





            # print("---"*10)
            


            # print(held_asset_coords)

            
            # keypoints_held = env.unwrapped.scene.env_origins[:,None,:] + env.unwrapped.keypoints_held
            # keypoints_fixed = env.unwrapped.scene.env_origins[:,None,:] + env.unwrapped.keypoints_fixed
            # env.unwrapped.visualize_spheres(keypoints_held)

            ## -----------------   visualization stuff -----------------------------
            visualize_spheres(env, held_asset_coord_viz, custom_noised_fingertip_midpoint_positions) #fingertip_midpoint_pos) ## blue
            visualize_spheres(env, custom_noised_peg_tip_viz, custom_noised_peg_tip_positions) #fixed_state) ## red
            # visualize_spheres(env, unnoised_peg_tip_viz, unnoised_peg_tip_positions) ## green
            # visualize_spheres(env, fixed_asset_coord_viz, stacked_hole_pts)
            # visualize_spheres(env, fixed_pos_coords_viz, fixed_pos_coords)
            # visualize_spheres(env, held_asset_viz, keypoints_held)
            # visualize_spheres(env, fixed_asset_viz, keypoints_fixed)
            
            # apply actions
            next_obs, rewards, terminations, truncations =  env.step(actions)
            # env.unwrapped.step_sim_no_action()
            # time.sleep(0.1)
            # print("rewards \n", rewards)



        cnt+= 1


            
        

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


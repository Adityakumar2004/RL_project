import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=" keyboard teleop")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# parser.add_argument("--video", action="store_true", help="Enable video recording during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=400, help="Interval between video recordings (in steps).")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# add argparse arguments
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
### ------------
import omni.ui as ui

### ------------


"""Rest everything follows."""
from typing import Union
import torch
import time
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
# from utils_1 import env_wrapper, log_values
import os
from custom_keyboard_2 import keyboard_custom
dt = 0.1


def make_env(video_folder:str | None =None, output_type: str = "numpy"):

    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.testing_controllers.factory_env_task_space:RobotEnvTaskSpace",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":"custom_scripts.testing_controllers.factory_env_task_space:RobotEnvCfgTaskSpace",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg = env_cfg, render_mode="rgb_array")
     
    # env = env_wrapper(env, video_folder, output_type=output_type, enable_normalization_rewards=False)
    
    return env


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


class ControllerWindow:
    """A UI window for controlling robot parameters with sliders."""
    
    def __init__(self, env:gym.Env=None):
        """
        Initialize the controller window.
        
        Args:
            initial_kp (float): Initial Kp value
            initial_kd (float): Initial Kd value
        """
        # Store parameters
        # self.params = {"kp": initial_kp, "kd": initial_kd}
        
        # Create value models for the sliders
        # self.kp_model = ui.SimpleFloatModel(initial_kp)
        # self.kd_model = ui.SimpleFloatModel(initial_kd)
        
        # self.models = {"kp": self.kp_model, "kd": self.kd_model}
        # # Create the UI window
        # self.window = None
        # self.kp_slider = None
        # self.kd_slider = None
        
        ## name:- {model, value, min, max, callback_function, args}
        self.slider_params = {}
        # self.env = env


        # self._create_window()
        # self._setup_callbacks()
    
    def _create_window(self):
        """Create the UI window and its contents."""
        self.window = ui.Window("Controller Panel", width=300, height=200, 
                               flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        
        with self.window.frame:
            with ui.VStack(spacing=10):
                # ui.Label("Kp Parameter (Proportional Gain)")
                # self.kp_slider = ui.FloatSlider(model=self.kp_model, min=0.0, max=500.0)
                
                # ui.Spacer(height=10)
                
                # ui.Label("Kd Parameter (Derivative Gain)")
                # self.kd_slider = ui.FloatSlider(model=self.kd_model, min=0.0, max=50.0)
                # for param_name, model in self.models.items():
                #     ui.Label(f"{param_name}")
                #     ui.FloatSlider(model=model, min=0.0, max=500.0)
                #     ui.Spacer(height=10)

                for param_name in self.slider_params.keys():
                    ui.Label(f"{param_name}")
                    ui.FloatSlider(
                        model=self.slider_params[param_name]['model'], 
                        min=self.slider_params[param_name]['min'], 
                        max=self.slider_params[param_name]['max']
                    )
                    ui.Spacer(height=10)
        
        # Ensure window is visible
        self.window.visible = True
        print("UI Controller Panel created and should be visible")
        # print(f"Initial values: kp={self.kp_model.as_float}, kd={self.kd_model.as_float}")
    
    def _setup_callbacks(self):
        """Setup callbacks for model changes."""
        # self.models['kp'].add_value_changed_fn(self._on_kp_changed)
        # self.models['kd'].add_value_changed_fn(self._on_kd_changed)
        pass;

    def get_params(self, param_name):
        """Get current parameters as dictionary."""
        return self.slider_params[param_name]["value"]
    
    def set_visible(self, visible):
        """Show or hide the window."""
        if self.window:
            self.window.visible = visible
    
    def destroy(self):
        """Clean up the window."""
        if self.window:
            self.window.destroy()
            self.window = None

    def create_new_slider_widget(self, param_name, value, min_val=0, max_val=1000, callback_fn=None, **kwargs):

        if param_name in self.slider_params.keys():
            print(f"Model {param_name} already exists.")
            return
        self.slider_params[param_name] = {
            'model': ui.SimpleFloatModel(value),
            'value': value,
            'min': min_val,
            'max': max_val,
        }
        callback_fn_model = lambda m: callback_fn(self.slider_params[param_name], m, **kwargs)
        self.slider_params[param_name]['model'].add_value_changed_fn(callback_fn_model)


## callback_fn 
def callback_kp(params, model, env:gym.Env, group_name:str):
    '''
    both approaches will work isaac lab made a wrapper for setting stiffness to the physx sim 
    the other approach is to directly set the stiffness to the physx sim
    '''

    new_kp_value = model.as_float
    joint_names = env.unwrapped._robot.actuators[group_name].joint_names
    joint_indices = env.unwrapped._robot.actuators[group_name].joint_indices
    # env.unwrapped._robot.actuators[group_name].stiffness[:,:] = new_kp_value
    env.unwrapped._robot.data.joint_stiffness[:,joint_indices] = new_kp_value
    stiffness_tensor = env.unwrapped._robot.data.joint_stiffness.clone()

    ## approach 1: using the wrapper function
    # env.unwrapped._robot.write_joint_stiffness_to_sim(stiffness_tensor, joint_indices)
    
    ## approach 2: directly setting to physx sim
    env_ids = torch.arange(env.unwrapped.scene.num_envs, device="cpu")
    env.unwrapped._robot.root_physx_view.set_dof_stiffnesses(stiffness_tensor.cpu(), env_ids)
    params["value"] = model.as_float

def callback_kd(params, model, env:gym.Env, group_name):
    new_kd_value = model.as_float
    joint_indices = env.unwrapped._robot.actuators[group_name].joint_indices
    # env.unwrapped._robot.actuators[group_name].damping[:,:] = new_kd_value
    env.unwrapped._robot.data.joint_damping[:,joint_indices] = new_kd_value
    damping_tensor = env.unwrapped._robot.data.joint_damping.clone()
    # env.unwrapped._robot.write_joint_stiffness_to_sim(damping_tensor, joint_indices)

    env_ids = torch.arange(env.unwrapped.scene.num_envs, device="cpu")
    env.unwrapped._robot.root_physx_view.set_dof_dampings(damping_tensor.cpu(), env_ids)
    params["value"] = model.as_float



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



def callback_kp_lin_task_space(params, model, env:gym.Env):
    new_kp_value = model.as_float
    kp_value = env.task_prop_gains
    kp_value[:, :3] = new_kp_value
    env.unwrapped.set_task_gains(kp_value, rot_deriv_scale=1.0)
    params["value"] = model.as_float

def callback_kp_rot_task_space(params, model, env:gym.Env):
    new_kp_value = model.as_float
    kp_value = env.task_prop_gains
    kp_value[:, 3:6] = new_kp_value
    env.unwrapped.set_task_gains(kp_value, rot_deriv_scale=1.0)
    params["value"] = model.as_float

def callback_kd_lin_task_space(params, model, env:gym.Env):
    new_kd_value = model.as_float
    kd_value = env.task_deriv_gains
    kd_value[:, :3] = new_kd_value
    env.unwrapped.set_task_gains(env.task_prop_gains, deriv_gains=kd_value, rot_deriv_scale=1.0)
    params["value"] = model.as_float

def callback_kd_rot_task_space(params, model, env:gym.Env):
    new_kd_value = model.as_float
    kd_value = env.task_deriv_gains
    kd_value[:, 3:6] = new_kd_value
    env.unwrapped.set_task_gains(env.task_prop_gains, deriv_gains=kd_value, rot_deriv_scale=1.0)
    params["value"] = model.as_float
    

def main():
    """Main function."""


    env = make_env(video_folder=None, output_type="numpy")

    env.reset()
    sphere_viz = create_marker_spheres(env,1, radius=0.008)
    fingertip_midpoint_viz = create_marker_spheres(env,1, color=(0.0, 1.0, 0.0), radius=0.005)
    # obs = env._get_observations()

    # Create the controller window
    controller = ControllerWindow()
    
    ## ---------------- joint space gains

    # controller.create_new_slider_widget(
    #     param_name="kp_arm1", 
    #     value=800.0, 
    #     min_val=0.0, 
    #     max_val=2000.0, 
    #     callback_fn=callback_kp, 
    #     env=env,
    #     group_name = "panda_arm1"
    # )
    
    # controller.create_new_slider_widget(
    #     param_name="kd_arm1", 
    #     value=160.0, 
    #     min_val=0.0, 
    #     max_val=2000.0, 
    #     callback_fn=callback_kd, 
    #     env=env,
    #     group_name = "panda_arm1"
    # )

    # controller.create_new_slider_widget(
    #     param_name="kp_arm2", 
    #     value=800.0, 
    #     min_val=0.0, 
    #     max_val=2000.0,
    #     callback_fn=callback_kp,
    #     env=env,
    #     group_name="panda_arm2"
    # )

    # controller.create_new_slider_widget(
    #     param_name="kd_arm2",
    #     value=160,
    #     min_val=0.0,
    #     max_val=2000,
    #     callback_fn=callback_kd,
    #     env=env,
    #     group_name="panda_arm2"
    # )

    ## --------------- task space gains

    controller.create_new_slider_widget(
        param_name="kp_lin_task_space",
        value=100.0,
        min_val=0.0,
        max_val=2000.0,
        callback_fn=callback_kp_lin_task_space,
        env=env
    )

    controller.create_new_slider_widget(
        param_name="kp_rot_task_space",
        value=30.0,
        min_val=0.0,
        max_val=2000.0,
        callback_fn=callback_kp_rot_task_space,
        env=env
    )

    controller.create_new_slider_widget(
        param_name="kd_lin_task_space",
        value=20.0,
        min_val=0.0,
        max_val=2000.0,
        callback_fn=callback_kd_lin_task_space,
        env=env
    )

    controller.create_new_slider_widget(
        param_name="kd_rot_task_space",
        value=11.0,
        min_val=0.0,
        max_val=2000.0,
        callback_fn=callback_kd_rot_task_space,
        env=env
    )

    controller._create_window()

    keyboard = keyboard_custom(pos_sensitivity=1.0*args_cli.sensitivity, rot_sensitivity=1.0*args_cli.sensitivity)
    keyboard.reset()
    print(f"\n\n{keyboard}\n\n")


    exp_name = "test_run"
    log_path = os.path.join("custom_scripts", "testing_controllers", "logs", f"{exp_name}_trajectory.npz")
    log_handler = handling_log()
    log_handler.log_value("initial_joint_pose", env.unwrapped._robot.data.default_joint_pos)

    # simulate physics
    count = 0  
    recording_state = 0
    while simulation_app.is_running():
        # ui.update()
        with torch.inference_mode():

            # Get keyboard input
            keyboard_output = keyboard.advance()
            pose_action = keyboard_output["pose_command"]
            close_gripper = keyboard_output["gripper_command"]
            recording_state = keyboard_output["recording_state"]

            if keyboard_output["reset_state"]:
                print("\n i am in the reset state \n ", "---"*10, "\n")
                env.reset()

            pose_action[:3] = pose_action[:3] * 0.03 
            pose_action[3:6] = pose_action[3:6] * 0.06
            # pose_action = pose_action * np.array([0.03, 0.03, 0.03, 0.06, ])
            if close_gripper:
                action = np.concatenate((pose_action, np.array([-1.0])), axis=0)
            else:
                action = np.concatenate((pose_action, np.array([1.0])), axis=0)

            # Convert to float32 tensor and replicate for all environments
            actions = torch.from_numpy(action).float().repeat(env.unwrapped.scene.num_envs, 1)

            # print(type(actions))
            # step the environment
            print(actions)
            print("count is ", count)
            obs, rew, terminated, truncated, info = env.step(actions)

            if recording_state == 1:
                log_handler.log_trajectory("raw_actions", env.unwrapped.log_values["raw_action"])
                print("Recording state is 1 and logging")
            
            if recording_state == 2:
                log_handler.save_log(log_path)
                print("Recording state is 2 and saving")

            # kp_value = controller.get_params('kp_arm2')
            # kd_value = controller.get_params('kd_arm2')
            # print("kp and kd values are ", kp_value, kd_value)
            # print("kp_arm2 and kd_arm2 of robot \n", env.unwrapped._robot.actuators["panda_arm2"].stiffness, env.unwrapped._robot.actuators["panda_arm2"].damping)
            # print("robot data joint stiffness \n", env.unwrapped._robot.data.joint_stiffness)
            # joint_names_exp = ["panda_joint[1-4]"]
            # joint_names = env.unwrapped._robot.actuators["panda_arm1"].joint_names
            # print("joint_indices ,", env.unwrapped._robot.actuators["panda_arm1"].joint_indices)
            # print()
            # print(env.unwrapped._robot.find_joints(joint_names)[0])
            # print(list(range(env.unwrapped.scene.num_envs)))
            
            print("held asset pos ")
            print((env.unwrapped._held_asset.data.root_pos_w-env.unwrapped.scene.env_origins).cpu().numpy())
            visualize_markers(env, fingertip_midpoint_viz, env.unwrapped.fingertip_midpoint_pos.clone())
            visualize_markers(env, sphere_viz, env.unwrapped._held_asset.data.root_pos_w.clone())
            count+=1
            
    # close the environment 
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()


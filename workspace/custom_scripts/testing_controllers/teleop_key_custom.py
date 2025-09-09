import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Custom keyboard teleop with extended functionality")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
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

"""Rest everything follows."""
import torch
import time
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import os

# Import your custom keyboard controller
from custom_keyboard import CustomKeyboard

dt = 0.1


def make_env(video_folder: str | None = None, output_type: str = "numpy"):
    id_name = "peg_insert-v0-uw"
    gym.register(
        id=id_name,
        entry_point="custom_scripts.testing_controllers.factory_env_task_space:RobotEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "custom_scripts.testing_controllers.factory_env_task_space:RobotEnvCfg",
        },
    )

    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg=env_cfg, render_mode="rgb_array")
    
    return env


def apply_friction_to_environment(env, friction_value):
    """Apply friction value to the environment dynamically."""
    # This is a placeholder - you'll need to implement based on your specific environment
    # For example, if your environment has a method to set friction:
    # env.unwrapped.scene.set_friction(friction_value)
    print(f"Applying friction: {friction_value}")


def start_recording(env, step_count):
    """Start video recording."""
    print(f"Started recording at step {step_count}")
    # Implement your recording logic here
    # For example: env.start_recording()


def stop_recording(env, step_count):
    """Stop video recording."""
    print(f"Stopped recording at step {step_count}")
    # Implement your recording logic here
    # For example: env.stop_recording()


def main():
    """Main function."""

    env = make_env(video_folder=None, output_type="numpy")
    env.reset()

    # Initialize custom keyboard controller
    keyboard = CustomKeyboard(
        pos_sensitivity=1.0 * args_cli.sensitivity, 
        rot_sensitivity=1.0 * args_cli.sensitivity,
        friction_step=0.05  # Friction adjustment step size
    )
    keyboard.reset()
    
    # Add custom callbacks for specific functionality
    def custom_callback_example():
        print("Custom callback triggered!")
    
    keyboard.add_callback("M", custom_callback_example)  # Bind 'M' key to custom function
    
    print(f"\n\n{keyboard}\n\n")
    
    # State tracking variables
    count = 0
    last_friction = keyboard.get_friction()
    last_recording_state = keyboard.get_recording_state()
    
    print("Starting simulation loop...")
    print("Press keys to control the robot. Press SPACE for emergency stop.")
    
    # simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():

            # Get keyboard input using the new custom controller
            keyboard_output = keyboard.advance()
            
            # Extract SE(3) command (compatible with original code)
            pose_action = keyboard_output["pose_command"]
            close_gripper = keyboard_output["gripper_command"]
            
            # Handle emergency stop
            if keyboard_output["emergency_stop"]:
                print("EMERGENCY STOP ACTIVATED - Sending zero actions")
                actions = torch.zeros(env.unwrapped.scene.num_envs, 7)
            else:
                # Scale pose action
                pose_action[:3] = pose_action[:3] * 0.03  # position scaling
                pose_action[3:6] = pose_action[3:6] * 0.06  # rotation scaling
                
                # Create action array
                if close_gripper:
                    action = np.concatenate((pose_action, np.array([-1.0])), axis=0)
                else:
                    action = np.concatenate((pose_action, np.array([1.0])), axis=0)

                # Convert to float32 tensor and replicate for all environments
                actions = torch.from_numpy(action).float().repeat(env.unwrapped.scene.num_envs, 1)

            # Check for friction changes
            current_friction = keyboard_output["current_friction"]
            if current_friction != last_friction:
                apply_friction_to_environment(env, current_friction)
                last_friction = current_friction

            # Check for recording state changes
            current_recording_state = keyboard_output["recording_state"]
            if current_recording_state != last_recording_state:
                if current_recording_state == "recording" and last_recording_state == "stopped":
                    start_recording(env, count)
                elif current_recording_state == "stopped" and last_recording_state in ["recording", "paused"]:
                    stop_recording(env, count)
                elif current_recording_state == "paused":
                    print(f"Recording paused at step {count}")
                elif current_recording_state == "recording" and last_recording_state == "paused":
                    print(f"Recording resumed at step {count}")
                
                last_recording_state = current_recording_state

            # Print debug information if debug mode is active
            if keyboard_output["debug_mode"] and count % 60 == 0:  # Print every 60 steps in debug mode
                print(f"Step {count}: Pose: {pose_action[:3]}, Rotation: {pose_action[3:6]}, "
                      f"Gripper: {close_gripper}, Friction: {current_friction}")

            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            count += 1
            
    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

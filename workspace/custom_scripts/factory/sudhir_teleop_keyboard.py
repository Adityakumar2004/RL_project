import sys
sys.path.append("IsaacLab/source")

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the Franka Cabinet Direct RL environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
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

from sim_env.envs.dishwasher_env import DishwasherEnv, DishwasherEnvCfg
from isaaclab.devices import Se3Keyboard
dt = 0.1

def main():
    """Main function."""
    # create environment configuration
    env_cfg = DishwasherEnvCfg()
    # env_cfg = FrankaEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # setup Direct RL environment
    env = DishwasherEnv(env_cfg)
    # env = FrankaEnv(env_cfg)

    env.reset()
    obs = env._get_observations()

    keyboard = Se3Keyboard(pos_sensitivity=0.1*args_cli.sensitivity, rot_sensitivity=1.0*args_cli.sensitivity)
    keyboard.reset()
    print(f"\n\n{keyboard}\n\n")
    
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():

            # Get keyboard input
            pose_action, close_gripper = keyboard.advance()
            if close_gripper:
                action = np.concatenate((pose_action, np.array([-1.0])), axis=0)
            else:
                action = np.concatenate((pose_action, np.array([1.0])), axis=0)
            
            actions = torch.from_numpy(action).repeat(env_cfg.scene.num_envs, 1)
            
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # update counter
            count += 1
            # time.sleep(dt)
    
    # close the environment
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

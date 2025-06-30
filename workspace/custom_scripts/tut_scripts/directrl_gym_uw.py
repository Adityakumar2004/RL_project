import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="CartPole Gym environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

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

id_name = "CartPole-v1-uw"
gym.register(
    id=id_name,
    entry_point="custom_scripts.tut_scripts.DirectRLCartPole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":"custom_scripts.tut_scripts.DirectRLCartPole:CartpoleEnvCfg",
    },
)


import gymnasium as gym
import numpy as np
import torch


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
    

def main():
    ## creating env config
    # print("hello i am here ")
    env_cfg = parse_env_cfg(
        id_name,
        num_envs=args_cli.num_envs
    )

    env = gym.make(id_name, cfg=env_cfg)
    env = env_wrapper(env)

    # # print info (this is vectorized environment)
    # print(f"[INFO]: Gym observation space: {env.unwrapped.observation_space}")
    # print(f"[INFO]: Gym single Observation space shape: {env.unwrapped.single_observation_space.shape}")
    # print(f"[INFO]: Gym action space: {env.unwrapped.action_space}")
    # print(f"[INFO]: Gym single action space shape: {env.unwrapped.single_action_space.shape}")
    # print(f"[INFO]: Single action space high: {env.single_action_space.high}")
    # print(f"[INFO]: Single action space low: {env.single_action_space.low}")


    print(f"[INFO]: Gym observation space: {env.observation_space}, {type(env.observation_space)}")
    print(f"[INFO]: Gym Observation space shape:", env.observation_space["policy"])

    print(f"[INFO]: Gym single Observation space shape: {env.single_observation_space.shape}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Gym single action space shape: {env.single_action_space.shape}")
    print(f"[INFO]: Single action space high: {env.single_action_space.high}")
    print(f"[INFO]: Single action space low: {env.unwrapped.single_action_space.low}")


    # reset environment
    env.reset()
    # simulate environment
    print("max length of episode ", env.unwrapped.max_episode_length)
    print("action space shape ", env.action_space.shape)
    print(env.unwrapped.scene["cartpole"].joint_names)


    # print(env.unwrapped.actions.action_term_groups["joint_efforts"])
    # If you want to inspect available action terms, try accessing them via the scene or print available attributes:
    # print(dir(env.unwrapped))
    # Or, for example, if the scene has action terms:
    # print(dir(env.unwrapped.action_manager))
    # print("checking the variables: decimation, episode_length_s, sim.dt in env ", env.unwrapped.episode_length_s/(env.unwrapped.sim.dt*env.unwrapped.decimation))
    
    
    cnt = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # actions = 1*torch.ones(env.action_space.shape, device=env.device)
            actions = np.random.uniform(-1,1,env.action_space.shape)
            
            
            # apply actions
            
            next_obs, rewards, terminations, truncations =  env.step(actions)
            
            if terminations.any() or truncations.any():
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # reset the environment
                # env.reset()
                print("[INFO]: Current observations: \n", next_obs)
                print("[INFO]: terminations and truncations \n",terminations, "\n",truncations)
                # print("[INFO]: infos: \n", infos)
                print("count :", cnt)
                cnt = 0
            
            if cnt % 20 == 0:
                print("[INFO]: Current observations: \n", next_obs)
                print("[INFO]: terminations and truncations \n",terminations, "\n",truncations)
                print("actions \n",actions)
                # print("[INFO]: infos: \n", infos)
                print("count :", cnt)

        cnt+= 1

            # print(env.unwrapped.episode_length_buf)
            

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()




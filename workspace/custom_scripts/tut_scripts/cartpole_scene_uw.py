# import argparse
# from isaaclab.app import AppLauncher
# # add argparse arguments
# parser = argparse.ArgumentParser(description="cartpole scene.")
# parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()
# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets import CARTPOLE_CFG  # isort:skip

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg  as Rewterm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import math

## helping functions 
## reward function 
def rewards_clubed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_names = asset.joint_names  # list of joint names in order
    joint_pos = asset.data.joint_pos  # shape: (num_envs, num_joints)
    joint_vel = asset.data.joint_vel

    # assume joint 0 is slider_to_cart and joint 1 is cart_to_pole
    slider_id = joint_names.index("slider_to_cart")
    pole_id = joint_names.index("cart_to_pole")

    slider_vel = joint_vel[:, slider_id]
    pole_pos = joint_pos[:, pole_id]
    pole_vel = joint_vel[:, pole_id]
    is_alive = (~(env.termination_manager.terminated)).float()
    # termination_penality = env.terminaion_manager.terminated.float() 
    slider_vel_reward = -0.01*torch.abs(slider_vel)  # reward for slider velocity
    pole_vel_reward = -0.005*torch.abs(pole_vel)  # reward for pole velocity
    pole_pos_reward = -torch.square(pole_pos)  # reward for pole position

    alive_bonus = 1.0  # alive bonus
    dead_penality = -5.0  # termination penalty
    ## wt1 for alive bonus and wt2 is for termination penality
    reward = is_alive * (pole_pos_reward + pole_vel_reward + slider_vel_reward + alive_bonus) \
                + (1-is_alive) * dead_penality # alive bonus
    
    return reward

## termination function 
# def termination_clubed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):

@configclass
class cartpole_scene_cfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    ## articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class Actioncfg:

    joint_efforts = mdp.JointEffortActionCfg(
        asset_name="cartpole", 
        joint_names=["slider_to_cart"],
        scale=10.0,
        offset=0.0,
        clip={"slider_to_cart": (-10.0, 10.0)}  # Optional, to enforce bounds after scaling
    )

@configclass
class ObservationCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("cartpole")})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("cartpole")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
        
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cartpole", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )
    
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cartpole", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):

    ## scene settings 
    scene = cartpole_scene_cfg(num_envs = 3, env_spacing=2.5)
    observations = ObservationCfg()
    actions = Actioncfg()
    events = EventCfg()

    def __post_init__(self):
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

@configclass
class RewardsCfg:
    alive = Rewterm(
        func = rewards_clubed,
        weight = 1.0,
        params = {
            "asset_cfg": SceneEntityCfg("cartpole")
        }
    )

@configclass 
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("cartpole", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class CartpoleRLEnvCfg(ManagerBasedRLEnvCfg):
    # scene 
    scene: cartpole_scene_cfg = cartpole_scene_cfg(num_envs= 6, env_spacing=4.0)
    
    ## environment settings 
    observations: ObservationCfg = ObservationCfg()
    actions: Actioncfg = Actioncfg()
    events: EventCfg = EventCfg()

    ## RL specific settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


# def main():

#     # env_cfg = CartpoleEnvCfg()
#     env_cfg = CartpoleRLEnvCfg()
#     env_cfg.scene.num_envs = args_cli.num_envs
#     env_cfg.sim.device = args_cli.device

#     # env = ManagerBasedEnv(cfg=env_cfg)
#     env = ManagerBasedRLEnv(cfg=env_cfg)

#     # simulate physics
#     count = 0
#     print(f"[INFO]: Gym observation space: {env.observation_space}")
#     # print(f"[info] single observation space {env.single_observation_space.shape}")
#     print(f"[INFO]: Gym single Observation space shape: {env.unwrapped.single_observation_space.shape}")
#     print(f"[INFO]: Gym action space: {env.action_space}")
#     print(f"[INFO]: Gym single action space shape: {env.single_action_space.shape}")
#     print(f"[INFO]: Single action space high: {env.single_action_space.high}")
#     print(f"[INFO]: Single action space low: {env.single_action_space.low}")
#     while simulation_app.is_running():
#         with torch.inference_mode():
#             # reset
#             if count % 300 == 0:
#                 count = 0
#                 env.reset()
#                 print("-" * 80)
#                 print("[INFO]: Resetting environment...")
#             # sample random actions
#             joint_efforts = torch.randn_like(env.action_manager.action)
#             # step the environment
#             # obs, _ = env.step(joint_efforts)
#             obs, rew, terminated, truncated, info = env.step(joint_efforts)
#             # print current orientation of pole
#             # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
#         # update counter
#         count += 1
#     # close the environment
#     env.close()


# if __name__ == "__main__":  
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()  # type: ignore

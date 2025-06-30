import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="cartpole scene.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import carb
# import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.utils.math import axis_angle_from_quat

# from custom_scripts.factory import factory_control as fc
# from custom_scripts.factory.factory_env_cfg_custom import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg
# from custom_scripts.factory.factory_env_cfg_custom import FactoryTaskPegInsertCfg
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
## -------------------------------------------------------------------------------

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
# from custom_scripts.factory.factory_control_custom import compute_dof_torque

###------------------------------------------- file tasks_cfg -----------------------------------------

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05

@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05

@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75

@configclass
class FactoryTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = -90.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_thread task only.
    action_penalty_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]  # General movement towards fixed object.
    keypoint_coef_coarse: list = [50, 2]  # Movement to align the assets.
    keypoint_coef_fine: list = [100, 0]  # Smaller distances for threading or last-inch insertion.
    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019

@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0

@configclass
class PegInsert(FactoryTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # Held Asset (applies to all tasks)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        # fixed_base=False,
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            # fixed_base = False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # fixed_base=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


### --------------------------------------------------###--------------------------------------



# from custom_scripts.factory.factory_tasks_cfg_custom import ASSET_DIR, FactoryTask, PegInsert
# ASSET_DIR, FactoryTask, PegInsert = import_fun()


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):

    decimation = 8
    action_space = 6
    observation_space = 13
    state_space = 3
    # obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    # state_order: list = [
    #     "fingertip_pos",
    #     "fingertip_quat",
    #     "ee_linvel",
    #     "ee_angvel",
    #     "joint_pos",
    #     "held_pos",
    #     "held_pos_rel_fixed",
    #     "held_quat",
    #     "fixed_pos",
    #     "fixed_quat",   
    # ]
    
    task: FactoryTask = FactoryTask()
    
    episode_length_s = 5
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn = sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions = False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos = {
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos = (0.0, 0.0, 0.0),
            rot = (1.0, 0.0, 0.0, 0.0),  # quaternion
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    # task_name = "peg_insert"  
    task: PegInsert = PegInsert()

## -------------------------------- defining the factory control ------------------------------
import math
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat

def compute_dof_torque(
    cfg, 
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,

):
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pose_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat, 
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pose_error, axis_angle_error), dim=1)

    lin_vel_error, ang_vel_error = vel_error(
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        ctrl_target_fingertip_midpoint_linvel= torch.zeros_like(fingertip_midpoint_linvel, device=device),
        ctrl_target_fingertip_midpoint_angvel=torch.zeros_like(fingertip_midpoint_angvel, device=device),
    )
    delta_fingertip_vel = torch.cat((lin_vel_error, ang_vel_error), dim=1)
    
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        delta_fingertip_vel=delta_fingertip_vel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )

    task_wrench += task_wrench_motion

    # useful tensors
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )  # ETH eq. 3.86; geometric Jacobian is assumed
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    # nullspace computation
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # TODO: Verify it's okay to no longer do gripper control here.
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def _apply_task_space_gains(
    delta_fingertip_pose,
    delta_fingertip_vel,
    task_prop_gains,
    task_deriv_gains,
):

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    ## Applying proportional and derivative gains to the lin components
    task_wrench[:, :3] = task_prop_gains[:, :3] * delta_fingertip_pose[:, :3] + \
                        task_deriv_gains[:, :3] * delta_fingertip_vel[:, :3] 
    
    ## Applying proportional and derivative gains to the ang components
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * delta_fingertip_pose[:, 3:6] + \
                        task_deriv_gains[:, 3:6] * delta_fingertip_vel[:, 3:6]

    return task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type="geometric",
    rot_error_type="axis_angle",
):

    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim = 1)
    # ctrl_target_fingertip_midpoint_quat[quat_dot < 0] *= -1.0
    ## gradients if any it will take care of that as there isnt any operation going on 
    ctrl_target_fingertip_midpoint_quat = torch.where(
        quat_dot.expand(-1,4) >=0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
    )

    fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # scalar component
    fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
    quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
    axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error
    else:
        # Always return a tuple to avoid NoneType errors
        return pos_error, torch.zeros_like(pos_error)


def vel_error(
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    ctrl_target_fingertip_midpoint_linvel,
    ctrl_target_fingertip_midpoint_angvel,
):
    """Compute task-space velocity error between target Franka fingertip velocity and current velocity."""
    linvel_error = ctrl_target_fingertip_midpoint_linvel - fingertip_midpoint_linvel
    angvel_error = ctrl_target_fingertip_midpoint_angvel - fingertip_midpoint_angvel
    return linvel_error, angvel_error

## --------------------------------------------------------------------------------

class FactoryEnvDirectRL(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_task = cfg.task
        super().__init__(cfg, render_mode, **kwargs)
            
    def _setup_scene(self):
        
        spawn_ground_plane(prim_path = "/World/ground", 
                           cfg=GroundPlaneCfg(),
                           translation = (0.0, 0.0, -1.05))
        ## adding lights 
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        self._robot = Articulation(self.cfg.robot)

        self.scene.clone_environments(copy_from_source=False)
        
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        self.scene.articulations["robot"] = self._robot

    def _init_tensors(self):

        self.last_update_timestamp = 0.0
        # self.prev_fingertip_pos = 

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)
        

    def _get_observations(self) -> dict:

        obs_dict = {
            "fingertip_midpoint_pos": self.fingertip_midpoint_pos,
            "fingertip_midpoint_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
        }

        # obs = torch.zeros(self.num_envs, self.observation_space.shape[1])
        # obs_dict = {"policy": obs}
        return obs_dict

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        return rewards
    
    def _get_dones(self):
        time_out = self.episode_length_buf>=self.max_episode_length - 1
        return time_out, time_out
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self.step_sim_no_action()
    
    def _set_assets_to_default_pose(self, env_ids):

        offset = torch.tensor([[0.0, 0.0, 0.15]], device=self.device, dtype=torch.float32)
        # Replicate the offset for each environment in env_ids
        offset_repeated = offset.repeat(len(env_ids), 1)
        
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += offset_repeated + self.scene.env_origins[env_ids]
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids)
        self._fixed_asset.reset()

        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += offset_repeated + self.scene.env_origins[env_ids]
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids)
        self._held_asset.reset()

        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))

    # def __set_franka_to_default_pose(self, env_ids):
    def step_sim_no_action(self): 
        '''
        it steps through all the envs without any action or previous action being continued 
        just the physics sim is being stepped and is not rendered 
        so naturally its used for resets inorder to first apply the physics
        on the first step and make it stable based on the constraints provided 
        '''
        self.scene.write_data_to_sim() ## Pushes new data into the simulation (e.g. poses, joint states).
        self.sim.step(render = False) ## Advances physics simulation by dt.
        self.scene.update(dt = self.physics_dt) ## Pulls updated data from simulation into Isaac Lab buffers (e.g. observation states).
        
        ## calculating and pulling updated values which are of interest to us 
        ## like the values going to observations, to apply actions, etc  
        self._compute_intermediate_values(dt = self.physics_dt)

    def _compute_intermediate_values(self, dt):

        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

    def generate_ctrl_signals(self):
        self.joint_torque, self.task_wrench = compute_dof_torque(
            cfg = self.cfg,
            dof_pos = self.joint_pos,
            dof_vel = self.joint_vel,
            fingertip_midpoint_pos = self.fingertip_midpoint_pos,
            fingertip_midpoint_quat = self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel = self.ee_linvel_fd
            fingertip_midpoint_angvel = self.ee_angvel_fd,
            jacobian = self.fingertip_midpoint_jacobian,
            arm_mass_matrix = self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat = self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains = self.task_prop_gains,
            task_deriv_gains = self.task_deriv_gains,
            device = self.device
        )

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    

def main():

    print("i am here \n","--"*20)
    factory_cfg = FactoryTaskPegInsertCfg()
    factory_cfg.scene.num_envs = args_cli.num_envs

    env = FactoryEnvDirectRL(factory_cfg)

    print("[INFO]: Factory environment created successfully ----- yo ----.")
    # simulate physics
    count = 0
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[info] single observation space {env.single_observation_space.shape}")
    # print(f"[INFO]: Gym single Observation space shape: {env.unwrapped.single_observation_space.shape}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Gym single action space shape: {env.single_action_space.shape}")
    print(f"[INFO]: Single action space high: {env.single_action_space.high}")
    print(f"[INFO]: Single action space low: {env.single_action_space.low}")
    count = 0
    # env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            
            if count % 100 == 0:
                print(f"[INFO]: Simulation step: {count}")
                count = 0
                env.reset()
            
            ## sampling random actions
            actions = torch.randn((env.num_envs, env.single_action_space.shape[0]), device=env.device)
            env.step(actions)
            count += 1
        
    env.close()
           

if __name__ == "__main__":

    main()

    simulation_app.close()
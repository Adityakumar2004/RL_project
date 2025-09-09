from __future__ import annotations
import os

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat, quat_inv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.envs.common import ViewerCfg

import torch

# -------------------------------------------------------------------------------
# Configuration Class for the Kinova Environment
# -------------------------------------------------------------------------------
@configclass
class KinovaEnvCfg(DirectRLEnvCfg):
    # Basic environment parameters
    episode_length_s = 20  # (20 seconds = 100*0.01*20 = steps*dt*decimation)
    decimation = 20
    action_space = 7
    observation_space = 8  # ee_pos (3) + ee_quat (4) + gripper_pos (1)
    state_space = 0
    debug_visualization = False  # Flag to control workspace cuboid visualization

    # Viewer configuration for viewport
    viewer = ViewerCfg(
        eye=(-1.0, 1.0, 0.5),
        lookat=(0.2, 0.0, 0.2),
        origin_type="env"
    )

    print("ISAAC_NUCLEUS_DIR:", ISAAC_NUCLEUS_DIR)

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        render=sim_utils.RenderCfg(
            enable_shadows=True,
            enable_reflections=True,
            enable_direct_lighting=True,
            samples_per_pixel=4,
            enable_ambient_occlusion=True,
            antialiasing_mode="DLAA"
        )
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=3.0,
        replicate_physics=True
    )

    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "usd")

    # Robot (Kinova arm) configuration.
    # Note: The asset only provides 7 joints named "joint_1" through "joint_7",
    # but the actual body names in the articulation are different.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "Robots/Kinova/gen3n7.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "gen3_joint_1": -0.0390,
                "gen3_joint_2": 0.8417,
                "gen3_joint_3": -0.0531,
                "gen3_joint_4": 2.2894,
                "gen3_joint_5": -0.0744,
                "gen3_joint_6": -1.5667,
                "gen3_joint_7": -1.5310,
                "finger_joint": 0.0, # 0, 0.8
                "left_inner_knuckle_joint": 0.0, # 0, 0.8757
                "right_inner_knuckle_joint": 0.0, # 0, 0.8757
                "right_outer_knuckle_joint": 0.0, # 0, 0.81
                "left_inner_finger_joint": 0.0, #-0.8757, 0
                "right_inner_finger_joint": 0.0, #-0.8757, 0
            },
        ),
        actuators={
            "kinova_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[1-4]"],
                effort_limit=80.0,
                velocity_limit=2.0,
                stiffness=800.0,
                damping=160.0,
            ),
            "kinova_forearm": ImplicitActuatorCfg(
                joint_names_expr=["gen3_joint_[5-7]"],
                effort_limit=10.0,
                velocity_limit=2.5,
                stiffness=800.0,
                damping=160.0,
            ),
            "kinova_gripper": ImplicitActuatorCfg(
                joint_names_expr=['finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'],
                effort_limit=200.0,
                velocity_limit=5.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    # Ground plane configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    # Additional parameters
    action_scale = 1.0


# -------------------------------------------------------------------------------
# Environment Class Definition
# -------------------------------------------------------------------------------
class KinovaEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: KinovaEnvCfg

    def __init__(self, cfg: KinovaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in environment-local coordinates."""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real
            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Get robot joint limits and set speed scales.
        self.robot_joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_joint_speed_scales = torch.ones_like(self.robot_joint_lower_limits)

        self.robot_joint_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Create diff_ik controller
        self.diff_ik_controller = DifferentialIKController(cfg.diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # Print Joint and Body Names
        print("Links:", self._robot.body_names)
        print("Joints:", self._robot.joint_names)

        # Links: ['world', 'gen3_shoulder_link', 'gen3_half_arm_1_link', 'gen3_half_arm_2_link', 'gen3_forearm_link', 'gen3_spherical_wrist_1_link', 'gen3_spherical_wrist_2_link', 'gen3_bracelet_link', 'left_outer_knuckle', 'left_inner_knuckle', 'right_inner_knuckle', 'right_outer_knuckle', 'left_inner_finger', 'right_inner_finger']
        # Joints: ['gen3_joint_1', 'gen3_joint_2', 'gen3_joint_3', 'gen3_joint_4', 'gen3_joint_5', 'gen3_joint_6', 'gen3_joint_7', 'finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint']

        ee_frame_name = "gripper_end_effector_link"
        arm_joint_names = ["gen3_joint_.*"]
        self.ee_frame_idx = self._robot.find_bodies(ee_frame_name)[0][0]
        self.arm_joint_ids = self._robot.find_joints(arm_joint_names)[0]
        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.ee_frame_idx
        else:
            self.ee_jacobi_idx = self.ee_frame_idx

        # Gripper related
        self.gripper_joint_names = ['finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 
                                  'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint']
        self.gripper_joint_ids = self._robot.find_joints(self.gripper_joint_names)[0]
        self.gripper_open_val = torch.tensor([0.1], device=self.device)
        self.gripper_close_val = torch.tensor([0.8], device=self.device)
        self.gripper_multiplier = torch.tensor([1, 1.0, 1.0, 1, -0.8, -0.8], device=self.device)

        # Define end-effector position limits
        self.ee_pos_min = torch.tensor([0.0, -0.5, 0.0], device=self.device)  # Minimum limits in x, y, z
        self.ee_pos_max = torch.tensor([1.5, 0.5, 1.5], device=self.device)    # Maximum limits in x, y, z

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Visualization
        if self.cfg.debug_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
            self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # reset robot joint positions
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_joint_lower_limits, self.robot_joint_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_joint_targets[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset ik controller
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        delta_ee_pose = torch.zeros((self.num_envs, 6), device=self.device)
        self.diff_ik_controller.reset(env_ids)
        self.diff_ik_controller.set_command(command=delta_ee_pose, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

        self._compute_intermediate_values(env_ids)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        delta_ee_pose = self.actions[:, :6] * self.cfg.action_scale
        self.diff_ik_controller.set_command(delta_ee_pose, ee_pos_curr, ee_quat_curr)

        # Update gripper joint targets
        gripper_joint_pos = self._robot.data.joint_pos[:, self.gripper_joint_ids]
        gripper_action = self.actions[:, -1:]
        # Convert gripper action (-1 to 1) to joint positions (0.1 to 0.8)
        gripper_pos = torch.where(gripper_action < 0, self.gripper_close_val, self.gripper_open_val)
        gripper_joint_targets = gripper_pos * self.gripper_multiplier
        self._robot.set_joint_position_target(gripper_joint_targets, self.gripper_joint_ids)
        
        # Visualization
        if self.cfg.debug_visualization:
            # update marker positions
            ee_pos_w = self._robot.data.body_pos_w[:, self.ee_frame_idx]
            ee_quat_w = self._robot.data.body_quat_w[:, self.ee_frame_idx]
            self.ee_marker.visualize(ee_pos_w, ee_quat_w)

    def _apply_action(self):
        # Read current end-effector pose
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]

        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self.diff_ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        # set the joint position command
        self._robot.set_joint_position_target(joint_pos_des, self.arm_joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = False
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        return self._compute_rewards()

    def _get_observations(self) -> dict:
        # Get end-effector pose
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        
        # Calculate gripper position (-1 to 1)
        gripper_joint_pos = self._robot.data.joint_pos[:, self.gripper_joint_ids]
        finger_angle = gripper_joint_pos[:, 0]  # Use finger_joint as reference
        # Normalize gripper state: open (0.1) -> 1.0, closed (0.8) -> -1.0
        gripper_pos = -2.0 * (finger_angle - self.gripper_open_val[0]) / (self.gripper_close_val[0] - self.gripper_open_val[0]) + 1.0
        gripper_pos = torch.clamp(gripper_pos, -1.0, 1.0)
        
        # Combine base observations
        obs = torch.cat([
            ee_pos_b,  # Current position (3)
            ee_quat_b,  # Current orientation (4)
            gripper_pos.unsqueeze(-1),  # Current gripper state (1)
        ], dim=-1)
        
        # Create observation dictionary
        observations = {"policy": obs}
        
        return observations

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

    def _compute_rewards(self):
        # Always return 0.0 as the reward
        return torch.zeros(self.num_envs, device=self.device)
    
    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_frame_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self.ee_frame_idx]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_joint_ids]
        base_rot = self._robot.data.root_quat_w
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        return jacobian 
    

### doubts
## jacobian transformation
## implicit actuator model for applying the gains

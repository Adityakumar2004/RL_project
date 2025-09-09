
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.lights import spawn_light

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

import torch

from isaaclab.utils.math import matrix_from_quat, quat_inv
from isaaclab.utils.math import subtract_frame_transforms


from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05

@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class RobotEnvCfg(DirectRLEnvCfg):

    episode_length_s = 20 ## 20sec = (rl_steps)*decimation*step_dt ## naming shouldnt be changed
    decimation = 20  ## decimation 20 means after every 20 sim steps one rl steps
    action_space = 7 ## 6- pose 1- gripper
    observation_space = 3 ## randomly set
    state_space = 3 ## randomly set


    '''
    physical_material
    This acts as the fallback material for any rigid body in the scene that doesnt explicitly override its material.

    Its also convenient when you want all objects to share the same baseline friction (and then later apply randomization on top).
    '''
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
        render=sim_utils.RenderCfg(
            enable_shadows=True,
            enable_reflections=True,
            enable_direct_lighting=True,
            samples_per_pixel=4,
            enable_ambient_occlusion=True,
            antialiasing_mode="DLAA"
        )
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
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
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00,#0.00871,
                "panda_joint2": 0.611, #-0.10368,
                "panda_joint3":  0.010,#-0.00794,
                "panda_joint4": -1.880,#-1.49139,
                "panda_joint5": -0.009,#-0.00083,
                "panda_joint6": 2.489,#1.38774,
                "panda_joint7": 0.740,#0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=800.0,
                damping=160.0,
                # friction=0.3,
                # armature=0.0,
                # effort_limit=87,
                effort_limit= 100, #80,
                # velocity_limit=124.6,
                velocity_limit=2.0,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=800.0,
                damping=160.0,
                # friction=0.3,
                # armature=0.0,
                # effort_limit=12,
                effort_limit= 100, #10,
                # velocity_limit=149.5,
                velocity_limit=2.5,
                
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                # effort_limit=40.0,
                effort_limit=200.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )
    
    held_asset_cfg: HeldAssetCfg = Peg8mm()
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


    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    # Additional parameters
    action_scale = 1.0



'''
pre_physics_step()
    only actions as far as i have seen 

for i in range(decimation):
    self._apply_action() 
    --> low level controller it updates the info like current pose, target pose based on the action at physcis step 
        and gives it to the controller 
    --> breifly put it applys the same action given at pre physics step but uses the latest env information

    physics_step()


    
'''
class RobotEnv(DirectRLEnv):

    def __init__(self, cfg:RobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.rl_dt = self.cfg.sim.dt * self.cfg.decimation


        # print(self._robot.joint_names)
        # print(self._robot.body_names)
        '''
        ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'force_sensor', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'panda_fingertip_centered']
        
        '''
        panda_fingers = ['panda_finger_joint1', 'panda_finger_joint2']
        self.gripper_joint_ids = self._robot.find_joints(panda_fingers)[0]
        self.fingertip_midpoint_idx = self._robot.find_bodies("panda_fingertip_centered")[0][0]
        # print("fingertip midpoint idx \n ",fingertip_midpoint_idx)
        
        self.ee_jacobi_idx = self.fingertip_midpoint_idx -1

        
        print(self.gripper_joint_ids)
        ## gripper actions 
        self.close_gripper = torch.tensor([0.0], device=self.device)
        self.open_gripper = torch.tensor([0.04], device=self.device)
        self.gripper_multiplier = torch.tensor([[1.0, 1.0]], device=self.device)


        self.diff_ik_controller = DifferentialIKController(cfg.diff_ik_cfg, self.num_envs, self.device)
        
        self._init_tensors() ## dont we need it in reset_idx 
        self.log_values = {}
        ##
        ## friction
        pass

    def _init_tensors(self):
        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def gripper_action(self):
        # print(self.actions)
        action = self.actions[:, -1:].clone()  ## (num_envs, )

        # print("action shape ",action.shape)
        gripper_command = torch.where(action[:, :] > 0.2, self.open_gripper, self.close_gripper)
        # print(gripper_command)
        # print("gripper command  shape1", gripper_command.shape)

        gripper_command = gripper_command * self.gripper_multiplier ## (num_envs, 1) * (1, 2) --> (num_envs, 2)
        # print("gripper command  shape2", gripper_command.shape)

        self.target_joint_pose[:, self.gripper_joint_ids] = gripper_command

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        spawn_ground_plane(prim_path="/World/ground_plane", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, 0.0))
        

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)
        spawn_light(prim_path="/World/Light", cfg=light_cfg)

        self._held_asset = Articulation(self.cfg.held_asset)
        self.scene.articulations["held_asset"] = self._held_asset


        self.scene.clone_environments(copy_from_source=False) ## if set to true we can have independent usd prims => independet robot cfgs, other assets

    def _pre_physics_step(self, actions):
        self.actions = actions
        self.log_values["raw_action"] = actions.clone().cpu()

        self.target_joint_pose = self._robot.data.joint_pos.clone()
        self.gripper_action()
        self._robot.set_joint_position_target(self.target_joint_pose[:, 7:], self.gripper_joint_ids)
        
        delta_pos  = actions[:, :6]
        fingertip_midpoint_pos_b, fingertip_midpoint_quat_b  = self._compute_frame_pose()
        self.diff_ik_controller.set_command(delta_pos, fingertip_midpoint_pos_b, fingertip_midpoint_quat_b)
        
        self._compute_intermediate_values()
        # print('curr joint pos \n', self._robot.data.joint_pos[0,:7])
        # print('target joint pos \n', self._robot.data.joint_pos_target.clone().cpu().numpy()[0,:])

    def _apply_action(self):
        # print(self._robot.data.joint_pos.clone().cpu().numpy()[0,7:])
        # self.target_joint_pose = self._robot.data.joint_pos.clone()
        # self._robot.set_joint_position_target(self.target_joint_pose[:, 7:], self.gripper_joint_ids)
        # self.gripper_action()
        self._compute_intermediate_values()
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:,0:7]
        jacobian_b = self._compute_frame_jacobian()
        self.target_joint_pose[:, 0:7] = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        self._robot.set_joint_position_target(self.target_joint_pose[:, 0:7], list(range(7)))
        pass

    def _compute_intermediate_values(self):

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_midpoint_idx]
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_midpoint_idx]


        pass

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)

        return rewards

    def _get_observations(self):
        pass

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        # print('default joint pos \n', joint_pos.cpu().numpy())
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.step_sim_no_action()
        # self._compute_intermediate_values()
        self._hold_asset_in_gripper(env_ids)
        gripper_command = self.close_gripper * self.gripper_multiplier ## (num_envs, 1) * (1, 2) --> (num_envs, 2)
        # print("gripper command  shape2", gripper_command.shape)

        joint_pos[:, self.gripper_joint_ids] = gripper_command
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        
        # print('joint pos after writing \n', self._robot.data.joint_pos.clone().cpu().numpy()) 
        pass
    
    def _get_dones(self):
        
        truncated = self.episode_length_buf >= self.max_episode_length-1
        return truncated, truncated
    
    def _get_observations(self):
        observations = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        return observations
    

        ## setup_scene
        ## pre_physics_step() 
        ## action
        ## get_dones, rewards, observations, infos, reset_idx, 
        ## default_pose

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_pos_w[:, self.fingertip_midpoint_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self.fingertip_midpoint_idx]
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
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, 0:7]
        base_rot = self._robot.data.root_quat_w
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        return jacobian 

    def _hold_asset_in_gripper(self, env_ids: torch.Tensor ):
        held_asset_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_asset_state[:, :3] = self.fingertip_midpoint_pos[env_ids,:] + torch.tensor([0.0, 0.0, -0.03], device=self.device) #+ self.scene.env_origins[env_ids]
        held_asset_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_asset_state[:, :7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_asset_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()
        # self._held_asset.data.root_pos_w
        pass




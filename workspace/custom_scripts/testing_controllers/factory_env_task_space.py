
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


from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

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
                effort_limit=80,
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
                effort_limit=10,
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

        self.diff_ik_controller = DifferentialIKController(cfg.diff_ik_cfg, self.num_envs, self.device)

        print(self._robot.joint_names)
        print(self._robot.body_names)
        '''
        ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'force_sensor', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'panda_fingertip_centered']
        
        '''
        panda_fingers = ['panda_finger_joint1', 'panda_finger_joint2']
        self.gripper_joint_ids = self._robot.find_joints(panda_fingers)[0]
        
        # print(gripper_joint_ids)
        ## gripper actions 
        self.close_gripper = torch.tensor([0.0], device=self.device)
        self.open_gripper = torch.tensor([0.04], device=self.device)
        self.gripper_multiplier = torch.tensor([[1.0, 1.0]], device=self.device)

        ##
        ## friction
        ## gripper indexing
        
        pass

    def gripper_action(self):
        # print(self.actions)
        action = self.actions[:, -1:].clone()  ## (num_envs, )

        # print("action shape ",action.shape)
        gripper_command = torch.where(action[:, :] < 0.1, self.open_gripper, self.close_gripper)
        print(gripper_command)
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


        self.scene.clone_environments(copy_from_source=False) ## if set to true we can have independent usd prims => independet robot cfgs, other assets

    def _pre_physics_step(self, actions):
        self.actions = actions

        self.target_joint_pose = self._robot.data.joint_pos.clone()
        self.gripper_action()
        self._robot.set_joint_position_target(self.target_joint_pose[:, 7:], self.gripper_joint_ids)
        # print('curr joint pos \n', self._robot.data.joint_pos.clone().cpu().numpy()[0,:7])
        # print('target joint pos \n', self._robot.data.joint_pos_target.clone().cpu().numpy()[0,:])

    def _apply_action(self):
        # print(self._robot.data.joint_pos.clone().cpu().numpy()[0,7:])
        # self.target_joint_pose = self._robot.data.joint_pos.clone()
        # self._robot.set_joint_position_target(self.target_joint_pose[:, 7:], self.gripper_joint_ids)
        # self.gripper_action()
        pass

    def _compute_intermediate_values(self):

        # self.fingertip_midpoint_pos = 
        pass

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)

        return rewards

    def _get_observations(self):
        pass

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        print('default joint pos \n', joint_pos.cpu().numpy())
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        print('joint pos after writing \n', self._robot.data.joint_pos.clone().cpu().numpy()) 
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
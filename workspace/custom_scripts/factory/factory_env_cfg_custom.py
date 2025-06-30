import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from custom_scripts.factory.factory_tasks_cfg_custom import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert

@configclass
class FactoryEnvCfg(DirectRLEnvCfg):

    decimation = 8
    action_space = 6
    observation_space = 3  
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


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"  
    task: PegInsert = PegInsert()
    

    

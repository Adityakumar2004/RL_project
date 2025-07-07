# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Minimal script to visualize Kinova Gen3 arm in an InteractiveScene, following best practices from FrankaCabinetEnv.
"""
import argparse

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_assets import KINOVA_GEN3_N7_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

from custom_scripts.factory.factory_tasks_cfg import ASSET_DIR

dir_robot = os.path.join(os.path.dirname(__file__), "usd")
robot_cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(dir_robot, "Robots/Kinova/gen3n7.usd"),
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
            "gen3_joint_1": 0.00871,
            "gen3_joint_2": -0.10368,
            "gen3_joint_3": -0.00794,
            "gen3_joint_4": -1.49139,
            "gen3_joint_5": -0.00083,
            "gen3_joint_6": 1.38774,
            "gen3_joint_7": 0.0,
            "finger_joint1": 0.04,
            "left_inner_knuckle_joint": 0.0, # 0, 0.8757
            "right_inner_knuckle_joint": 0.0, # 0, 0.8757
            "right_outer_knuckle_joint": 0.0, # 0, 0.81
            "left_inner_finger_joint": 0.0, #-0.8757, 0
            "right_inner_finger_joint": 0.0, #-0.8757, 0
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "kinova_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["gen3_joint_[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=87,
            velocity_limit=124.6,
        ),
        "kinova_forearm": ImplicitActuatorCfg(
            joint_names_expr=["gen3_joint_[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=12,
            velocity_limit=149.5,
        ),
        "kinova_gripper": ImplicitActuatorCfg(
            joint_names_expr=['finger_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'],
            effort_limit=40.0,
            velocity_limit=0.04,
            stiffness=7500.0,
            damping=173.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)


robot_panda = ArticulationCfg(
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
            "panda_joint1": 0.00871,
            "panda_joint2": -0.10368,
            "panda_joint3": -0.00794,
            "panda_joint4": -1.49139,
            "panda_joint5": -0.00083,
            "panda_joint6": 1.38774,
            "panda_joint7": 0.0,
            "panda_finger_joint2": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
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
class KinovaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot = robot_panda
    robot = robot_cfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    robot = scene["robot"]
    
    sim_dt = sim.get_physics_dt()
    count = 0

    # left_finger_body_idx = robot.body_names.index("panda_leftfinger")
    # right_finger_body_idx = robot.body_names.index("panda_rightfinger")
    # fingertip_body_idx = robot.body_names.index("panda_fingertip_centered")

    # Simulation loop
    print(robot.data.joint_names)
    print("*"*5," -- body names -- ", "*"*5)
    print(robot.body_names)
    # print("-- left finger")
    # print(robot.data.body_pos_w[:,left_finger_body_idx ])
    # print(robot.data.body_quat_w[:,left_finger_body_idx ])

    # print("-- RIGHT finger")
    # print(robot.data.body_pos_w[:,right_finger_body_idx ])
    # print(robot.data.body_quat_w[:,right_finger_body_idx ])


    # print("-- fingertip centered")
    # print(robot.data.body_pos_w[:,fingertip_body_idx ])
    # print(robot.data.body_quat_w[:,fingertip_body_idx ])


    while simulation_app.is_running():


        # Perform step
        sim.step()
        # Increment counter
        # count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    scene_cfg = KinovaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
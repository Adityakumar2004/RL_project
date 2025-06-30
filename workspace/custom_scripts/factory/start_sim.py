import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Start Isaac Sim and run a Kinova scene.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to spawn")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum simulation steps")
    return parser.parse_args()

def main():
    args = parse_args()

    # Launch Isaac Sim app FIRST
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    print("Isaac Sim app launched.")

    # Import Kinova scene config after app launch
    from custom_scripts.factory.kinovo_scene_uw import KinovaSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.reset()

    # Create scene with Kinova Gen3, allow num_envs override
    scene_cfg = KinovaSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Access the robot articulation (optional)
    robot = scene["robot"]

    # Run the simulator for a fixed number of steps or until closed
    step = 0
    while simulation_app.is_running() and step < args.max_steps:
        sim.step()
        step += 1

    print(f"Simulation finished after {step} steps.")
    simulation_app.close()

if __name__ == "__main__":
    main()

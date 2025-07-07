# Expanded Overview: RL Agent Training for Factory Environment using rl_games

This document provides a comprehensive, detailed overview of the file structure and all files involved in the training process of a reinforcement learning (RL) agent for the factory environment using the `rl_games` framework and the `train.py` script. All references are specific to the `isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/factory` folder and the RL pipeline.

---

## 1. Main Training Script

### `isaaclab/scripts/reinforcement_learning/rl_games/train.py`
- **Purpose:** Entry point for training RL agents using the `rl_games` library and Isaac Lab environments.
- **Key Responsibilities:**
  - Parses command-line arguments for training configuration (e.g., video recording, number of environments, task name, seed, distributed training, checkpoint, sigma, max iterations).
  - Launches the Isaac Sim simulator using `AppLauncher`.
  - Imports and registers the factory environment from `isaaclab_tasks.direct.factory`.
  - Loads environment and agent configurations (Hydra-based and YAML-based).
  - Sets up logging directories and saves configuration snapshots.
  - Instantiates the environment using `gym.make()` with the selected task and configuration.
  - Handles multi-agent to single-agent conversion if needed.
  - Wraps the environment for video recording and for compatibility with `rl_games`.
  - Registers the environment with the `rl_games` registry.
  - Loads the RL agent configuration and checkpoint if provided.
  - Runs the training loop using `rl_games`'s `Runner` and closes the simulator on completion.
- **Key Imports:**
  - `isaaclab_tasks.direct.factory` (for environment and config)
  - `isaaclab.envs`, `isaaclab.utils`, `isaaclab_rl.rl_games`, `rl_games` modules
  - `isaaclab_tasks.utils.hydra` for Hydra config

---

## 2. Factory Environment Package Structure

### Folder: `isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/`

#### Main Files:
- **`__init__.py`**
  - Registers the factory environments with Gymnasium (e.g., `Isaac-Factory-PegInsert-Direct-v0`).
  - Imports and exposes the main environment and configuration classes.
  - Associates each environment with its config and agent YAML file (e.g., `rl_games_ppo_cfg.yaml`).
- **`factory_env.py`**
  - Implements the `FactoryEnv` class, which inherits from `DirectRLEnv`.
  - Handles simulation setup, asset spawning, tensor initialization, observation/state construction, reward and success computation, and control signal generation.
  - Integrates with the control module and configuration files.
- **`factory_env_cfg.py`**
  - Defines configuration classes for the environment, including observation/state dimensions, control parameters, and task-specific settings.
  - Imports task definitions from `factory_tasks_cfg.py`.
- **`factory_tasks_cfg.py`**
  - Contains all task, asset, and robot configuration classes (e.g., `PegInsert`, `GearMesh`, `NutThread`).
  - Specifies asset paths, physical properties, reward shaping, and task logic.
- **`factory_control.py`**
  - Implements low-level control logic and utility functions for the robot (e.g., torque computation, pose error calculation).
  - Used by `factory_env.py` for generating control signals.

#### Subfolder: `agents/`
- **`__init__.py`**
  - (Empty, but required for Python package structure.)
- **`rl_games_ppo_cfg.yaml`**
  - YAML configuration for the RL agent (PPO/A2C) used by `rl_games`.
  - Specifies network architecture, training hyperparameters, device, normalization, reward shaping, and more.

#### Subfolder: `__pycache__/`
- Contains Python bytecode cache files for faster imports (not directly relevant to training logic).

---

## 3. File Interactions and Data Flow

- **Environment Registration:**
  - `__init__.py` registers each factory task as a Gym environment, linking to the correct config and agent YAML.
- **Environment Instantiation:**
  - `train.py` uses `gym.make()` with the task name (e.g., `Isaac-Factory-PegInsert-Direct-v0`), which loads `FactoryEnv` and the associated config from `factory_env_cfg.py`.
- **Configuration Loading:**
  - Environment config is loaded from Python classes in `factory_env_cfg.py` and task/asset details from `factory_tasks_cfg.py`.
  - Agent config is loaded from `agents/rl_games_ppo_cfg.yaml`.
- **Simulation and Control:**
  - `FactoryEnv` sets up the simulation, assets, and tensors, and uses `factory_control.py` for robot control.
- **Training Loop:**
  - `train.py` wraps the environment for `rl_games`, registers it, and runs the training loop using the loaded agent config.
- **Logging and Checkpoints:**
  - Training logs, videos, and checkpoints are saved to the `logs/rl_games/` directory, with configuration snapshots for reproducibility.

---

## 4. Detailed File Reference

### `train.py` (RL Training Script)
- **Arguments:** Video recording, number of envs, task, seed, distributed, checkpoint, sigma, max_iterations
- **Simulator:** Uses `AppLauncher` to start Isaac Sim
- **Environment:** Loads from Gym registry (registered in `__init__.py`)
- **Config:** Loads Hydra config for environment, YAML for agent
- **Agent:** Uses `rl_games` PPO/A2C agent, config in `rl_games_ppo_cfg.yaml`
- **Logging:** Saves all configs and logs to `logs/rl_games/<experiment>`
- **Key Functions:**
  - `main(env_cfg, agent_cfg)`: Orchestrates the full training process
  - `gym.make()`: Instantiates the environment
  - `Runner`: Handles the RL training loop

### `factory_env.py`
- **Class:** `FactoryEnv(DirectRLEnv)`
- **Responsibilities:**
  - Simulation setup, asset spawning, tensor initialization
  - Observation/state construction, reward and success computation
  - Control signal generation (calls `factory_control.py`)
  - Handles task-specific logic and resets

### `factory_env_cfg.py`
- **Classes:** `FactoryEnvCfg`, `FactoryTaskPegInsertCfg`, `FactoryTaskGearMeshCfg`, `FactoryTaskNutThreadCfg`, etc.
- **Responsibilities:**
  - Defines observation/state/action spaces, control parameters
  - Imports and links task/asset configs from `factory_tasks_cfg.py`

### `factory_tasks_cfg.py`
- **Classes:** `FactoryTask`, `PegInsert`, `GearMesh`, `NutThread`, `FixedAssetCfg`, `HeldAssetCfg`, `RobotCfg`, etc.
- **Responsibilities:**
  - Defines all assets, robots, and tasks used in the factory environment
  - Specifies physical properties, reward shaping, and task logic

### `factory_control.py`
- **Functions:** `compute_dof_torque`, etc.
- **Responsibilities:**
  - Implements robot control logic (e.g., torque computation, pose error)
  - Used by `FactoryEnv` for low-level control

### `agents/rl_games_ppo_cfg.yaml`
- **YAML Config:**
  - RL agent hyperparameters, network architecture, training settings
  - Used by `train.py` and `FactoryEnv` via Gym registration

---

## 5. Summary Table: File Structure

| File/Folder                                      | Purpose/Role                                                                 |
|--------------------------------------------------|------------------------------------------------------------------------------|
| `train.py`                                       | Main RL training script                                                      |
| `__init__.py`                                    | Registers factory environments with Gym                                      |
| `factory_env.py`                                 | Implements the main environment logic                                        |
| `factory_env_cfg.py`                             | Environment and task configuration classes                                   |
| `factory_tasks_cfg.py`                           | Task, asset, and robot configuration                                         |
| `factory_control.py`                             | Robot control logic and utilities                                            |
| `agents/rl_games_ppo_cfg.yaml`                   | RL agent configuration for rl_games                                          |
| `agents/__init__.py`                             | (Empty, for package structure)                                               |
| `__pycache__/`                                   | Python bytecode cache (not directly used)                                    |
| `Runner` (from rl_games)                          | Orchestrates the RL training loop: loads agent config, manages environment, collects rollouts, updates policy/value networks, handles logging, checkpointing, and evaluation. Used in `train.py` as the main training engine. |

---

### About `Runner` (from rl_games)

The `Runner` class from the `rl_games` library is the main engine that drives RL training in this pipeline. It is responsible for:
- Loading the agent and environment configuration (`runner.load(agent_cfg)`).
- Initializing the environment and agent.
- Managing the main training loop (`runner.run(...)`), which includes:
  - Collecting rollouts (interacting with the environment).
  - Updating the policy and value networks.
  - Logging statistics and saving checkpoints.
  - Optionally running evaluation episodes.
- Supporting distributed/multi-GPU training if configured.

**Usage in `train.py`:**
- `runner = Runner(IsaacAlgoObserver())` creates the runner with a custom observer for logging/hooks.
- `runner.load(agent_cfg)` loads the agent and environment configuration.
- `runner.reset()` resets the agent and environment state.
- `runner.run({...})` starts the training loop with the provided options (e.g., `train=true`, `play=false`).

**Summary:**
The `Runner` is the central orchestrator for RL training, handling all aspects of the training lifecycle as configured by your script and agent/environment configs.

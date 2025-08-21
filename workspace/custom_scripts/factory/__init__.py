# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# from . import agents
# from .factory_env_kinova import FactoryEnv
# from .factory_env_cfg_kinovo import FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Factory-PegInsert-Direct-kinova-v0",
    entry_point="custom_scripts.factory.factory_env_kinova:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "custom_scripts.factory.factory_env_cfg_kinovo:FactoryTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"custom_scripts.factory.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-PegInsert-Direct-Custom-v0",
    entry_point="custom_scripts.factory.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "custom_scripts.factory.factory_env_cfg:FactoryTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"custom_scripts.factory.agents:rl_games_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Factory-GearMesh-Direct-v0",
#     entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskGearMeshCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Factory-NutThread-Direct-v0",
#     entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskNutThreadCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Minimal script to visualize Kinova Gen3 arm in an InteractiveScene, following best practices from FrankaCabinetEnv.
"""

from isaaclab_assets import KINOVA_GEN3_N7_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class KinovaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = KINOVA_GEN3_N7_CFG.replace(prim_path="/World/Robot")
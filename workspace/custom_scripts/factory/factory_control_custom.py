
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

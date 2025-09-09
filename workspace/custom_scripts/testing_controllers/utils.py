
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import torch
import numpy as np
from typing import Union

def create_marker_spheres(env, count, color=(1.0, 0.0, 0.0), radius = 0.001):

    sphere_markers = {}
    for i in range(count):
        sphere_markers[f"sphere_{i}"] = sim_utils.SphereCfg(
            radius = radius,#0.001,
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = color),
        )
    
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/envMarkers",
        markers=sphere_markers

    )

    env_marker_visualizer = VisualizationMarkers(marker_cfg)
    return(env_marker_visualizer)


def visualize_spheres(env, env_marker_visualizer:VisualizationMarkers, pose: Union[np.ndarray, torch.Tensor], quat = None):
    '''
    Args:
    pose: tensor or numpy array of positions with dim --> (num_envs, num_spheres, 3) or (num_envs, 3) 
    
    '''
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    elif isinstance(pose, np.ndarray):
        pass
    else :
        assert False, "pose must be a torch.Tensor or np.ndarray"
    
    if quat is None:
        identity_quat = np.array([1, 0, 0, 0])  # identity quat for sphere

    if len(pose.shape) == 3:
        (num_envs, num_spheres, _) = pose.shape
    elif len(pose.shape) == 2:
        (num_envs, _) = pose.shape
        num_spheres = 1
        pose = pose[:, None, :]  # Add extra dimension to make it (num_envs, 1, _)
    else:
        raise ValueError(f"pose must have 2 or 3 dimensions, got shape {pose.shape}")

    translations = np.empty((num_envs * num_spheres, 3), dtype=np.float32)
    orientations = np.empty((num_envs * num_spheres, 4), dtype=np.float32)
    marker_indices = np.empty((num_envs * num_spheres,), dtype=np.int32)

    for env_id in range(num_envs):
        for count in range(num_spheres):
            translations[(num_spheres*env_id + count)] = pose[env_id, count, :3]
            if quat is None:
                orientations[(num_spheres*env_id + count)] = identity_quat
            else:
                orientations[(num_spheres*env_id + count)] = quat[env_id, count, :]
            marker_indices[(num_spheres*env_id + count)] = count

    env_marker_visualizer.visualize(
        translations=translations,
        orientations=orientations,
        marker_indices=marker_indices
    )


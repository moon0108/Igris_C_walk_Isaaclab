import torch
from isaaclab.envs import ManagerBasedRLEnv


def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    asset_cfg,
    min_height: float,
) -> torch.Tensor:
    """Terminate if the base/root height goes below a threshold.

    Args:
        env: IsaacLab manager-based env.
        asset_cfg: SceneEntityCfg("robot", body_names="base_link") or just "robot" root.
        min_height: threshold in meters.
    Returns:
        Bool tensor (num_envs,) indicating termination.
    """
    asset = env.scene[asset_cfg.name]
    # root position in world frame: (num_envs, 3)
    z = asset.data.root_pos_w[:, 2]
    return z < min_height

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine
from .base.vec_task import VecTask


class Vertifarm(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        num_obs = 21
        num_actions = 8 

    def create_sim(self):

        # setup simulation 
        #setup call python script brdige
        #initial pos etc
        raise NotImplementedError()


    def compute_reward(self, actions):
        # must define in a way that if the step is in the actual trajectory
        #then reward, else penalty. 
        raise NotImplementedError()

    def compute_observations(self):
        #use instead of step
        raise NotImplementedError()

    def compute_franka_reward(
        reset_buf, progress_buf, actions, cabinet_dof_pos,
        franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
        franka_lfinger_pos, franka_rfinger_pos,
        gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
        num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
        finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        raise NotImplementedError()
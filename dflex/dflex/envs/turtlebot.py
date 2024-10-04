# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# from envs.dflex_env import DFlexEnv
# import math
# import torch

# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import dflex as df

# import numpy as np
# np.set_printoptions(precision=5, linewidth=256, suppress=True)

# try:
#     from pxr import Usd
# except ModuleNotFoundError:
#     print("No pxr package")

# from utils import load_utils as lu
# from utils import torch_utils as tu
# from utils.common import *

import math
import os
import sys

import torch

from .dflex_env import DFlexEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dflex as df

import numpy as np
import dflex.envs.load_utils as lu
import dflex.envs.torch_utils as tu


np.set_printoptions(precision=5, linewidth=256, suppress=True)
import pdb


class TurtlebotEnv(DFlexEnv):

    def __init__(
            self, 
            render=False, 
            device='cuda:0', 
            num_envs=4096, 
            seed=0, 
            episode_length=1000, 
            no_grad=True, 
            stochastic_init=False, 
            MM_caching_frequency = 1, 
            early_termination = True,
            jacobian=False,
            contact_ke=4.0e4,
            contact_kd=None,  #  1.0e4,
            logdir=None,
            nan_state_fix=False,
            jacobian_norm=None,
            termination_height=0.27,
            action_penalty=0.0,
            action_strength = 20.0,
            joint_vel_obs_scaling=0.1,
            up_strength = 0.05,
            heading_strength = 0.08,
            lin_strength = 0.05,
            target_penalty = -0.3,
            target_strength = 0.2,
            ang_penalty = -0.00,
            obstacle_strength = 0.5,
            obstacle_penalty = -0.5,
            obst_radius = 0.75,
            survive_reward = 3,
            approach = 0,
            up_rew_scale=0.1,
        ):
        num_obs = 21 # correspond with self.obs_buf
        num_act = 2

        '''
        #################### obs table ############################
        self.obs_buf = torch.cat([torso_pos[:, :], # 0:3 
                                torso_rot, # 3:7
                                lin_vel, # 7:10
                                ang_vel, # 10:13
                                self.state.joint_q.view(self.num_envs, -1)[:, 7:], # 13:15
                                self.joint_vel_obs_scaling * self.state.joint_qd.view(self.num_envs, -1)[:, 6:], # 15:17
                                up_vec[:, 1:2], # 17
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 18
                                self.actions.clone()], # 19:21
                                dim = -1)
        '''
        
        super(TurtlebotEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.early_termination = early_termination
        self.contact_ke = contact_ke
        self.contact_kd = contact_kd if contact_kd is not None else contact_ke / 4.0


        self.init_sim()

        # navigation parameters
        target = (10.0, 0.1, 10.0)
        self.targets = tu.to_torch(list(target), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        
        obstacle = (6.0, 0.1, 4.5)
        
        self.obstacle = tu.to_torch(list(obstacle), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # obsacle to avoid

        # other parameters
        # self.termination_distance = 1000
        self.action_strength = action_strength
        self.joint_vel_obs_scaling = 0.1
        self.up_strength = joint_vel_obs_scaling
        self.heading_strength = heading_strength
        self.lin_strength = lin_strength
        self.action_penalty = action_penalty
        self.target_penalty = target_penalty
        self.target_strength = target_strength
        self.ang_penalty = ang_penalty
        self.obstacle_strength = obstacle_strength
        self.obstacle_penalty = obstacle_penalty
        self.obst_radius = obst_radius
        self.obst_threshold = 3 * obst_radius
        self.survive_reward = survive_reward
        self.approach = approach
        # 0 -- using target_penalty & obstacle_strength
        # 1 -- using obstacle_penalty & target_strength

        self.setup_visualizer(logdir)

        # pack for record
        self.hyper_parameter = None
        if self.approach == 0:
            self.hyper_parameter = {
                # "termination_distance": self.termination_distance,
                "action_strength": self.action_strength,
                "joint_vel_obs_scaling": self.joint_vel_obs_scaling,
                "up_strength": self.up_strength,
                "heading_strength": self.heading_strength,
                "lin_strength": self.lin_strength,
                "obstacle_strength": self.obstacle_strength,
                "action_penalty": self.action_penalty,
                "target_penalty": self.target_penalty,
                "ang_penalty": self.ang_penalty,
                "obst_threshold": self.obst_threshold,
                "obst_radius": self.obst_radius,
                "survive_reward": self.survive_reward,
            }
        else:
            self.hyper_parameter = {
                # "termination_distance": self.termination_distance,
                "action_strength": self.action_strength,
                "joint_vel_obs_scaling": self.joint_vel_obs_scaling,
                "up_strength": self.up_strength,
                "heading_strength": self.heading_strength,
                "lin_strength": self.lin_strength,
                "obstacle_penalty": self.obstacle_penalty,
                "action_penalty": self.action_penalty,
                "target_strength": self.target_strength,
                "ang_penalty": self.ang_penalty,
                "obst_threshold": self.obst_threshold,
                "obst_radius": self.obst_radius,
                "survive_reward": self.survive_reward,
            }

        # #-----------------------
        # # set up Usd renderer
        # if (self.visualize):
        #     self.stage = Usd.Stage.CreateNew("outputs/" + "Turtlebot_" + str(self.num_envs) + '_' + get_time_stamp() + ".usd")

        #     self.renderer = df.render.UsdRenderer(self.model, self.stage)
        #     self.renderer.draw_points = True
        #     self.renderer.draw_springs = True
        #     self.renderer.draw_shapes = True
        #     self.render_time = 0.0
        #
        #     # draw opstacles
        #     self.renderer.add_sphere(pos=obstacle, radius=obst_radius/2, name="obst", time=self.render_time)


    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 9 # joint(2) + position(3) + pose(4)
        # self.num_joint_qd = 9
        self.num_joint_qd = 8

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.y_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.start_pos = []
        self.start_joint_q = [0.0, 0.0] # wheel joint 
        self.start_joint_target = [0.0, 0.0]
        

        if self.visualize:
            self.env_dist = 2.5
        else:
            self.env_dist = 0. # set to zero for training for numerical consistency

        start_height = 0.1
        self.start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

        asset_folder = os.path.join(os.path.dirname(__file__), 'assets')
        for i in range(self.num_environments):
            
            start_pos_z = i*self.env_dist
            start_pos = [0.0, start_height, start_pos_z]
            self.start_pos.append(start_pos)
            lu.urdf_load(builder=self.builder, 
                            filename=os.path.join(asset_folder, 'turtlebot.urdf'),
                            xform=df.transform(start_pos, self.start_rot),
                            floating=True
                        )
            
            # pdb.set_trace()

            self.builder.joint_q[i*self.num_joint_q:i*self.num_joint_q + 3] = self.start_pos[-1]
            self.builder.joint_q[i*self.num_joint_q + 3:i*self.num_joint_q + 7] = self.start_rot

            # set wheel joint targets to rest 
            self.builder.joint_q[i*self.num_joint_q + 7:i*self.num_joint_q + 9] = [0.0, 0.0]
            self.builder.joint_target[i*self.num_joint_q + 7:i*self.num_joint_q + 9] = [0.0, 0.0]

        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_joint_q = tu.to_torch(self.start_joint_q, device=self.device)
        # self.start_joint_target = tu.to_torch(self.start_joint_target, device=self.device)
        

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()
        # pdb.set_trace()
        
        if (self.model.ground):
            self.model.collide(self.state)

    # def render(self, mode = 'human'):
    #     if self.visualize:
    #         self.render_time += self.dt
    #         self.renderer.update(self.state, self.render_time)

    #         render_interval = 1
    #         if (self.num_frames == render_interval):
    #             try:
    #                 self.stage.Save()
    #             except:
    #                 print("USD save error")

    #             self.num_frames -= render_interval

    # def step(self, actions):
    #     actions = actions.view((self.num_envs, self.num_actions))

    #     actions = torch.clip(actions, -1., 1.)
    #     # pdb.set_trace()

    #     self.actions = actions.clone()

    #     self.state.joint_act.view(self.num_envs, -1)[:, 6:] = actions * self.action_strength
        
    #     self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
    #     self.sim_time += self.sim_dt

    #     self.reset_buf = torch.zeros_like(self.reset_buf)

    #     self.progress_buf += 1
    #     self.num_frames += 1

    #     self.calculateObservations()
    #     self.calculateReward()

    #     env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

    #     if self.no_grad == False:
    #         self.obs_buf_before_reset = self.obs_buf.clone()
    #         self.extras = {
    #             'obs_before_reset': self.obs_buf_before_reset,
    #             'episode_end': self.termination_buf,
    #             }

    #     if len(env_ids) > 0:
    #        self.reset(env_ids)

    #     self.render()

    #     return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def unscale_act(self, action):
        return action * self.action_strength

    def set_act(self, action):
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = action

    # TODO: early termination add obstacle judgement
    def compute_termination(self, obs, act):
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # if self.early_termination:
        #     termination = obs[:, 0] < self.termination_height
        return termination
    
    def static_init_func(self, env_ids):
        xyz = self.start_pos[env_ids]
        quat = self.start_rotation.repeat(len(env_ids), 1)
        joints = self.start_joint_q.repeat(len(env_ids), 1)
        joint_q = torch.cat((xyz, quat, joints), dim=-1)
        joint_qd = torch.zeros((len(env_ids), self.num_joint_qd), device=self.device)
        return joint_q, joint_qd

    def stochastic_init_func(self, env_ids):
        """Method for computing stochastic init state"""
        xyz = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3]
            + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.0
        )
        angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
        axis = torch.nn.functional.normalize(
            torch.rand((len(env_ids), 3), device=self.device) - 0.5
        )
        quat = tu.quat_mul(
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7],
            tu.quat_from_angle_axis(angle, axis),
        )

        joints = (
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:]
            + 0.2
            * (
                torch.rand(
                    size=(len(env_ids), self.num_joint_q - 7),
                    device=self.device,
                )
                - 0.5
            )
            * 2.0
        )

        joint_q = torch.cat((xyz, quat, joints), dim=-1)
        joint_qd = 0.5 * (
            torch.rand(size=(len(env_ids), self.num_joint_qd), device=self.device) - 0.5
        )
        return joint_q, joint_qd

    def set_state_act(self, obs, act):
        # torso position
        self.state.joint_q.view(self.num_envs, -1)[:, 0:3] = obs[:, 0:3]
        # torso rotation
        self.state.joint_q.view(self.num_envs, -1)[:, 3:7] = obs[:, 3:7]
        # linear velocity
        self.state.joint_qd.view(self.num_envs, -1)[:, 3:6] = obs[:, 7:10]
        # angular velocity
        self.state.joint_qd.view(self.num_envs, -1)[:, 0:3] = obs[:, 10:13]
        # torque
        self.state.joint_q.view(self.num_envs, -1)[:, 7:] = obs[:, 13:15]
        # torque'
        self.state.joint_qd.view(self.num_envs, -1)[:, 6:] = obs[:, 15:17]
        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = act

    
    # def reset(self, env_ids = None, force_reset = True):
    #     if env_ids is None:
    #         if force_reset == True:
    #             env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    #     if env_ids is not None:
    #         # clone the state to avoid gradient error
    #         self.state.joint_q = self.state.joint_q.clone()
    #         self.state.joint_qd = self.state.joint_qd.clone()

    #         # fixed start state
    #         self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
    #         self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = self.start_rotation.clone()
    #         self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.start_joint_q.clone()
    #         self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.

    #         # randomization
    #         if self.stochastic_init:
    #             self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
    #             angle = (torch.rand(len(env_ids), device = self.device) - 0.5) * np.pi / 12.
    #             axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device = self.device) - 0.5)
    #             self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
    #             self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device = self.device) - 0.5) * 2.
    #             self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), 8), device=self.device) - 0.5)

    #         # clear action
    #         self.actions = self.actions.clone()
    #         self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

    #         self.progress_buf[env_ids] = 0

    #         self.calculateObservations()

    #     return self.obs_buf
    
    # '''
    # cut off the gradient from the current state to previous states
    # '''
    # def clear_grad(self, checkpoint = None):
    #     with torch.no_grad():
    #         if checkpoint is None:
    #             checkpoint = {}
    #             checkpoint['joint_q'] = self.state.joint_q.clone()
    #             checkpoint['joint_qd'] = self.state.joint_qd.clone()
    #             checkpoint['actions'] = self.actions.clone()
    #             checkpoint['progress_buf'] = self.progress_buf.clone()

    #         current_joint_q = checkpoint['joint_q'].clone()
    #         current_joint_qd = checkpoint['joint_qd'].clone()
    #         self.state = self.model.state()
    #         self.state.joint_q = current_joint_q
    #         self.state.joint_qd = current_joint_qd
    #         self.actions = checkpoint['actions'].clone()
    #         self.progress_buf = checkpoint['progress_buf'].clone()

    # '''
    # This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    # It has to be called every time the algorithm starts an episode and it returns the observation vectors
    # '''
    # def initialize_trajectory(self):
    #     self.clear_grad()
    #     self.calculateObservations()

    #     return self.obs_buf

    # def get_checkpoint(self):
    #     checkpoint = {}
    #     checkpoint['joint_q'] = self.state.joint_q.clone()
    #     checkpoint['joint_qd'] = self.state.joint_qd.clone()
    #     checkpoint['actions'] = self.actions.clone()
    #     checkpoint['progress_buf'] = self.progress_buf.clone()

    #     return checkpoint

    def observation_from_state(self, state):
        torso_pos = state.joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_rot = state.joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = state.joint_qd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_vel = state.joint_qd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)
        action = state.joint_act.view(self.num_envs, -1) / self.action_strength

        # print(torso_pos)
        # print(self.actions)

        # self.obs_buf = torch.cat([torso_pos[:, :], # 0:3 
        #                         torso_rot, # 3:7
        #                         lin_vel, # 7:10
        #                         ang_vel, # 10:13
        #                         state.joint_q.view(self.num_envs, -1)[:, 7:], # 13:15
        #                         self.joint_vel_obs_scaling * state.joint_qd.view(self.num_envs, -1)[:, 6:], # 15:17
        #                         up_vec[:, 1:2], # 17
        #                         (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 18
        #                         self.actions.clone()], # 19:21
        #                         dim = -1)

        return torch.cat(
            [
                torso_pos[:, :], # 0:3 
                torso_rot, # 3:7
                lin_vel, # 7:10
                ang_vel, # 10:13
                state.joint_q.view(self.num_envs, -1)[:, 7:], # 13:15
                self.joint_vel_obs_scaling * state.joint_qd.view(self.num_envs, -1)[:, 6:], # 15:17
                up_vec[:, 1:2], # 17
                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 18
                action.clone() # 19:21
            ], 
            dim = -1)

        
        # pdb.set_trace()

    def calculateReward(self, obs, act):
        up_reward = self.up_strength * obs[:, 17]
        heading_reward = self.heading_strength * obs[:, 18]
        # height_reward = obs[:, 0] - self.termination_height
        lin_vel_reward = self.lin_strength * torch.linalg.norm(obs[:, 7:10], dim=1) 
        ang_vel_penalty = torch.linalg.norm(obs[:, 10:13], dim=1) * self.ang_penalty
        action_penalty = torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        # self.rew_buf = None
        if self.approach == 0:
            target_dist = torch.linalg.norm(obs[:, 0:3] - self.targets, dim=1)
            target_penalty = target_dist * self.target_penalty
            obst_dist = torch.linalg.norm(obs[:, 0:3] - self.obstacle, dim=1)
            obst_reward = torch.where(obst_dist < self.obst_threshold, obst_dist*self.obstacle_strength, torch.zeros_like(obst_dist))
            rew_buf = self.survive_reward + obst_reward + ang_vel_penalty + lin_vel_reward + up_reward + heading_reward + target_penalty + action_penalty
            return rew_buf
        else:
            target_dist = torch.linalg.norm(obs[:, 0:3] - self.targets, dim=1)
            target_reward = torch.exp(1/(torch.ones_like(target_dist) + target_dist)) * self.target_strength
            obst_dist = torch.linalg.norm(obs[:, 0:3] - self.obstacle, dim=1)
            obst_penalty = torch.where(obst_dist < self.obst_threshold, torch.exp(1/(torch.ones_like(obst_dist) + obst_dist))*self.obstacle_penalty, torch.zeros_like(obst_dist))
            rew_buf = self.survive_reward + target_reward + ang_vel_penalty + lin_vel_reward + up_reward + heading_reward + obst_penalty + action_penalty
            return rew_buf

        # # reset agents
        # if self.early_termination:
        #     self.reset_buf = torch.where(obst_dist < (self.obst_radius), torch.ones_like(self.reset_buf), self.reset_buf)
        # self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.dflex_env import DFlexEnv
import math
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path


import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

from utils import load_utils as lu
from utils import torch_utils as tu
from utils.common import *
from utils.nerf import Nerf

import pdb
import matplotlib.pyplot as plt


class TurtlebotNerfEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = True):
        num_obs = 22 # correspond with self.obs_buf
        num_act = 2
    
        print('----------------------------', device)
        super(TurtlebotNerfEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.init_sim()
        self.episode_length = episode_length

        # navigation parameters
        # target = (10.0, 0.1, 10.0)
        target = (8.0, 0.1, 7.0)
        self.targets = tu.to_torch(list(target), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        
        # for training
        # obstacle_1 = (5.0, 0.1, 0)
        # obstacle_2 = (5.0, 0.1, 9.75)
        # obst_radius = 0.5

        # # for drawing
        # obstacle_1 = (2.0, 0.1, -1.5)
        # obstacle_2 = (2.0, 0.1, 1.5)

        # obstacle_1 = (2.0, 0.1, -3.0)
        # obstacle_2 = (2.0, 0.1, 3.0)
        # obstacle_3 = (7.0, 0.1, 0.0)

        obstacle_1 = (2.0, 0.1, -2.0)
        obstacle_2 = (2.0, 0.1, 2.6)
        obstacle_3 = (7.0, 0.1, 0.0)

        obst_radius = 0.4
        obst_ext = 0.4

        self.obstacle_1 = tu.to_torch(list(obstacle_1), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # obsacle to avoid

        self.obstacle_2 = tu.to_torch(list(obstacle_2), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # obsacle to avoid
        
        self.obstacle_3 = tu.to_torch(list(obstacle_3), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # obsacle to avoid
        
        # nerf
        self.data_dir = Path(__file__).parent.absolute().parent.absolute()
        nerf_path = self.data_dir / "envs/assets/nerf/data/trained_nerfs/column2/2024-01-17_083853/config.yml"
        data_path = self.data_dir / "envs/assets/nerf/data/nerf_training_data/columns2"
        render_camera_ints = {
        "fx": 636.76,
        "fy": 636.05,
        "cx": 646.82,
        "cy": 370.98,
        "w": 1280,
        "h": 720,
        "rescale": 0.25,
        }
        self.nerf = Nerf(nerf_path, data_path, render_camera_ints)
        
        
        # other parameters
        # self.termination_distance = 1000
        self.action_strength = 20.0
        self.joint_vel_obs_scaling = 0.1
        self.up_strength = 0.05
        self.heading_strength = 0.16
        self.lin_strength = 0.05
        self.action_penalty = -0.05
        self.target_penalty = -0.45
        self.target_strength = 0.2
        self.ang_penalty = -0.00
        self.obstacle_strength = 0.6
        self.obstacle_penalty = -0.5
        self.obst_radius = obst_radius
        self.obst_threshold = 1.5 * obst_radius
        self.survive_reward = 3
        self.depth_strength = 2.5
        self.approch = 0
        # 0 -- using target_penalty & obstacle_strength
        # 1 -- using obstacle_penalty & target_strength

        # TODO:
        # change to penalty obst and reward getting to target

        # pack for record
        self.hyper_parameter = None
        if self.approch == 0:
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
                "depth_strength": self.depth_strength,
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
                "depth_strength": self.depth_strength,
            }

        #-----------------------
        # set up Usd renderer
        self.time_stamp = get_time_stamp()
        self.episode_count = 0
        if (self.visualize):
            self.stage = Usd.Stage.CreateNew("outputs/" + "TurtlebotNerf_" + str(self.num_envs) + '_' + self.time_stamp + ".usd")

            self.renderer = df.render.UsdRenderer(self.model, self.stage)
            self.renderer.draw_points = True
            self.renderer.draw_springs = True
            self.renderer.draw_shapes = True
            self.render_time = 0.0
            # self.renderer.add_sphere(pos=obstacle_1, radius=obst_radius/2, name="obst1", time=self.render_time)
            # self.renderer.add_sphere(pos=obstacle_2, radius=obst_radius/2, name="obst2", time=self.render_time)
            
            # self.renderer.add_sphere(pos=obstacle_1, radius=obst_radius, name="obst1", time=self.render_time)
            # self.renderer.add_sphere(pos=obstacle_2, radius=obst_radius, name="obst2", time=self.render_time)
            # self.renderer.add_sphere(pos=obstacle_3, radius=obst_radius, name="obst3", time=self.render_time)

            # render obstacles
            self.renderer.add_box(pos=obstacle_1, extents=obst_ext, name="obst1", time=self.render_time)
            self.renderer.add_box(pos=obstacle_2, extents=obst_ext, name="obst2", time=self.render_time)
            self.renderer.add_box(pos=obstacle_3, extents=obst_ext, name="obst3", time=self.render_time)
            
            # render goal
            self.renderer.add_sphere(pos=(9.6, 0.1, 6.75), radius=0.25, name="goal", time=self.render_time)

        # recored x, y, depth for test
        self.x_record = []
        self.y_record = []
        self.depth_record = []

    def init_sim(self):
        self.builder = df.sim.ModelBuilder()

        self.dt = 1.0/60.0
        self.sim_substeps = 16
        self.sim_dt = self.dt

        self.ground = True

        self.num_joint_q = 9 # joint(2) + position(3) + pose(4)
        self.num_joint_qd = 9

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
        self.start_joint_target = tu.to_torch(self.start_joint_target, device=self.device)
        

        # finalize model
        self.model = self.builder.finalize(self.device)
        self.model.ground = self.ground
        self.model.gravity = torch.tensor((0.0, -9.81, 0.0), dtype=torch.float32, device=self.device)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.state = self.model.state()
        # pdb.set_trace()
        
        if (self.model.ground):
            self.model.collide(self.state)

    def render(self, mode = 'human'):
        if self.visualize:
            self.render_time += self.dt
            self.renderer.update(self.state, self.render_time)

            render_interval = 1
            if (self.num_frames == render_interval):
                try:
                    self.stage.Save()
                except:
                    print("USD save error")

                self.num_frames -= render_interval

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))

        actions = torch.clip(actions, -1., 1.)
        # pdb.set_trace()

        self.actions = actions.clone()

        self.state.joint_act.view(self.num_envs, -1)[:, 6:] = actions * self.action_strength
        
        # pdb.set_trace()
        self.state = self.integrator.forward(self.model, self.state, self.sim_dt, self.sim_substeps, self.MM_caching_frequency)
        self.sim_time += self.sim_dt

        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf,
                }

        if len(env_ids) > 0:
           self.reset(env_ids)

        self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # clone the state to avoid gradient error
            self.state.joint_q = self.state.joint_q.clone()
            self.state.joint_qd = self.state.joint_qd.clone()

            # fixed start state
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = self.start_rotation.clone()
            self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.start_joint_q.clone()
            self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.

            # randomization
            if self.stochastic_init:
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 0:3] + 0.1 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
                angle = (torch.rand(len(env_ids), device = self.device) - 0.5) * np.pi / 12.
                axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device = self.device) - 0.5)
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(self.state.joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.state.joint_q.view(self.num_envs, -1)[env_ids, 7:] + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device = self.device) - 0.5) * 2.
                self.state.joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), 8), device=self.device) - 0.5)

            # clear action
            self.actions = self.actions.clone()
            self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf
    
    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = self.state.joint_q.clone()
                checkpoint['joint_qd'] = self.state.joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            self.state = self.model.state()
            self.state.joint_q = current_joint_q
            self.state.joint_qd = current_joint_qd
            self.actions = checkpoint['actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()

    '''
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['joint_q'] = self.state.joint_q.clone()
        checkpoint['joint_qd'] = self.state.joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()

        return checkpoint
    
    def pose_transfer(self, position, quaternion):
        '''
        Get ROS style pose from position and quaternion
        '''
        qw, qx, qy, qz = quaternion
        R = torch.tensor([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        T = torch.zeros((3, 4))
        T[:3, :3] = R
        T[:, 3] = position
        return T

    def calculateObservations(self):
        torso_pos = self.state.joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_rot = self.state.joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = self.state.joint_qd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_vel = self.state.joint_qd.view(self.num_envs, -1)[:, 0:3]

        ros_pose = torch.zeros((self.num_envs, 3, 4))
        for i in range(self.num_envs):
            pos_i = torso_pos[i,:]
            rot_i = torso_rot[i,:]
            pose_i = self.pose_transfer(pos_i, rot_i)
            ros_pose[i, :, :] = pose_i

        depth_list = self.nerf.render_quad_poses(ros_pose)
        depth_list = self.depth_strength * torch.abs(depth_list)
        depth_list = depth_list.to(device='cuda')
        # pdb.set_trace()

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel = lin_vel - torch.cross(torso_pos, ang_vel, dim = -1)

        to_target = self.targets + self.start_pos - torso_pos
        to_target[:, 1] = 0.0
        
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        up_vec = tu.quat_rotate(torso_quat, self.basis_vec1)
        heading_vec = tu.quat_rotate(torso_quat, self.basis_vec0)

        # print(torso_pos)
        # print(self.actions)

        # Draw record plot for data visualization
        if self.visualize:
            pos_data = torso_pos.clone()
            pos_data = pos_data.detach().cpu().numpy()
            depth_data = depth_list.clone()
            depth_data = depth_data.detach().cpu().numpy()

            self.x_record.append(pos_data[0, 0])
            self.y_record.append(pos_data[0, 2])
            self.depth_record.append(depth_data[0]/2)

            self.episode_count += 1
            if self.episode_count == (self.episode_length-1):
                save_path = '/home/qianzhong/DiffRL_NeRF/examples'

                np.save(save_path + '/outputs/' + self.time_stamp + '_x.npy', np.array(self.x_record, dtype=object), allow_pickle=True)
                np.save(save_path + '/outputs/' + self.time_stamp + '_y.npy', np.array(self.y_record, dtype=object), allow_pickle=True)


                # figure x over time
                plt.figure()
                plt.plot(range(len(self.x_record)), self.x_record)
                plt.xlabel("Step")
                plt.ylabel("X/m")
                plt.savefig(save_path + '/outputs/' + self.time_stamp + '_x_plot.png')

                # figure y over time
                plt.figure()
                plt.plot(range(len(self.y_record)), self.y_record)
                plt.xlabel("Step")
                plt.ylabel("Y/m")
                plt.savefig(save_path + '/outputs/' + self.time_stamp + "_y_plot.png")

                # figure traj 
                plt.figure()
                plt.plot(self.x_record, self.y_record)
                plt.xlabel("X/m")
                plt.ylabel("Y/m")
                plt.savefig(save_path + '/outputs/' + self.time_stamp + "_traj_plot.png")

                # figure depth 
                plt.figure()
                plt.plot(range(len(self.depth_record)), self.depth_record)
                plt.xlabel("Step")
                plt.ylabel("depth/m")
                plt.savefig(save_path + '/outputs/' + self.time_stamp + "_depth_plot.png")

        # pdb.set_trace()
        self.obs_buf = torch.cat([torso_pos[:, :], # 0:3 
                                torso_rot, # 3:7
                                lin_vel, # 7:10
                                ang_vel, # 10:13
                                self.state.joint_q.view(self.num_envs, -1)[:, 7:], # 13:15
                                self.joint_vel_obs_scaling * self.state.joint_qd.view(self.num_envs, -1)[:, 6:], # 15:17
                                up_vec[:, 1:2], # 17
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 18
                                self.actions.clone(), # 19:21
                                depth_list], # 21
                                dim = -1)
        
        # pdb.set_trace()

    def calculateReward(self):
        up_reward = self.up_strength * self.obs_buf[:, 17]
        heading_reward = self.heading_strength * self.obs_buf[:, 18]
        # height_reward = self.obs_buf[:, 0] - self.termination_height
        lin_vel_reward = self.lin_strength * torch.linalg.norm(self.obs_buf[:, 7:10], dim=1) 
        ang_vel_penalty = torch.linalg.norm(self.obs_buf[:, 10:13], dim=1) * self.ang_penalty
        action_penalty = torch.sum(self.actions ** 2, dim = -1) * self.action_penalty

        self.rew_buf = None
        obst_dist = self.obs_buf[:, 21]
        if self.approch == 0:
            target_dist = torch.linalg.norm(self.obs_buf[:, 0:3] - self.targets, dim=1)
            target_penalty = target_dist * self.target_penalty
            # obst_dist = torch.linalg.norm(self.obs_buf[:, 0:3] - self.obstacle, dim=1)
            obst_reward = torch.where(obst_dist < self.obst_threshold, obst_dist*self.obstacle_strength, torch.zeros_like(obst_dist))
            self.rew_buf = self.survive_reward + obst_reward + ang_vel_penalty + lin_vel_reward + up_reward + heading_reward + target_penalty + action_penalty
        else:
            target_dist = torch.linalg.norm(self.obs_buf[:, 0:3] - self.targets, dim=1)
            target_reward = torch.exp(1/(torch.ones_like(target_dist) + target_dist)) * self.target_strength
            # obst_dist = torch.linalg.norm(self.obs_buf[:, 0:3] - self.obstacle, dim=1)
            obst_penalty = torch.where(obst_dist < self.obst_threshold, torch.exp(1/(torch.ones_like(obst_dist) + obst_dist))*self.obstacle_penalty, torch.zeros_like(obst_dist))
            self.rew_buf = self.survive_reward + target_reward + ang_vel_penalty + lin_vel_reward + up_reward + heading_reward + obst_penalty + action_penalty
        # print(self.rew_buf)

        # reset agents
        # if self.early_termination:
        #     self.reset_buf = torch.where(obst_dist < (self.obst_radius), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

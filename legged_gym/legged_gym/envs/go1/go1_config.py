# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 17 * 11
        n_priv = 3 + 3 + 3
        n_priv_latent = 4 + 1 + 12 + 12
        n_proprio = 2 + 3 + 3 + 3 + 36 + 1
        n_recon_num = 2 + 3 + 3 + 12 + 12 + 1
        history_len = 20

        num_observations = history_len * n_proprio 
        num_privileged_obs = 1630
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        obs_type = "og"
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2
        num_contact = 3

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 3

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            imu = 0.08
            base_ang_vel = 0.4
            gravity = 0.05
            dof_pos = 0.05
            dof_vel = 0.1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        hf2mesh_method = "grid"
        max_error = 0.1
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05 

        horizontal_scale = 0.05
        horizontal_scale_camera = 0.1

        vertical_scale = 0.005 
        border_size = 5 
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        max_stair_height = 0.15
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = True
        measure_horizontal_noise = 0.0

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8
        num_rows= 6
        num_cols = 6
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 1.5,
                        "rough slope down":1.5,
                        "stairs up": 3., 
                        "stairs down": 3., 
                        "discrete": 1.5, 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "smooth flat": 0.,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 0.,
                        "parkour_step": 0.,
                        "parkour_gap": 0,
                        "plane": 0,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        slope_treshold = 1.5
        origin_zero_z = False

        num_goals = 8

    class terrain_parkour(terrain):
        mesh_type = "trimesh"
        num_rows = 8
        num_cols = 6
        num_goals = 8
        selected = "BarrierTrack"
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.

        curriculum = True
        horizontal_scale = 0.025
        pad_unavailable_info = True

        BarrierTrack_kwargs = dict(
            options= [
                # "jump",
                "crawl",
                "tilt",
                "leap",
            ],
            track_width= 1.6,
            track_block_length= 1.6,
            wall_thickness= (0.04, 0.2),
            wall_height= -0.05,
            jump= dict(
                height= (0.1, 0.4),
                depth= (0.1, 0.8),
                fake_offset= 0.0,
                jump_down_prob= 0.,
            ),
            crawl= dict(
                height= (0.22, 0.5),
                depth= (0.1, 0.6),
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.27, 0.38),
                depth= (0.4, 1.),
                opening_angle= 0.0,
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.8, 1.2),
                depth= (-0.05, -0.4),
                height= 0.5,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
        )

        TerrainPerlin_kwargs = dict(
            zScale= 0.025,
            frequency= 10,
        )
    class contact_phase():
        num_contact_phase = 3


    class commands(LeggedRobotCfg.commands): 
        curriculum = False
        max_curriculum = 1.
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        forward_curriculum_threshold = 0.8
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        yaw_curriculum_threshold = 0.5
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        global_reference = False

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.2

        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-1.0, 1.0]
        ang_vel_yaw = [-1, 1]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]

        heading = [-3.14, 3.14]
        class curriculum_ranges:
            lin_vel_x = [0.5, 1]
            lin_vel_y = [-0, 0]
            ang_vel_yaw = [-0.1,0.1]
            heading = [-0.1, 0.1]

        class max_ranges:
            lin_vel_x = [-1.6, 1.6]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

        waypoint_delta = 0.7

        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            '2_FL_hip_joint': 0.1,
            '4_RL_hip_joint': 0.1,
            '1_FR_hip_joint': -0.1,
            '3_RR_hip_joint': -0.1,

            '2_FL_thigh_joint': 1.5,
            '4_RL_thigh_joint': 1.5,
            '1_FR_thigh_joint': 1,
            '3_RR_thigh_joint': 1.,

            '2_FL_calf_joint': -2.4,
            '4_RL_calf_joint': -2.4,
            '1_FR_calf_joint': -1.8,
            '3_RR_calf_joint': -1.8,

            'skateboard_joint_x': 0,
            'skateboard_joint_y': 0.9,
            'skateboard_joint_z': 0,

            'front_truck_roll_joint':0,
            'rear_truck_roll_joint':0,
            'front_left_wheel_joint':0,
            'front_right_wheel_joint':0,
            'rear_left_wheel_joint':0,
            'rear_right_wheel_joint':0
        }
        glide_default_pos = [
            0.3, 1.1, -2.4, 
            -0.3, 1.1, -2.4, 
            0.74, 1.13, -0.7,
            0, 0,
            0, 0,
            0, 0,
            0.3, 1.1, -2.4, 
            -0.3, 1.1,-2.4]

    class control(LeggedRobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        stiffness = {'joint': 40., 'skateboard':0, 'truck':100, 'wheel': 0}
        damping = {'joint': 1, 'skateboard':0, 'truck':10, 'wheel': 0 }
        action_scale = 0.25
        decimation = 4


    class asset(LeggedRobotCfg.asset):
        
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_skate/urdf/go1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]

        actuated_dof_names = ["1_FR_hip_joint", "1_FR_thigh_joint", "1_FR_calf_joint",
                              "2_FL_hip_joint", "2_FL_thigh_joint", "2_FL_calf_joint",
                              "3_RR_hip_joint", "3_RR_thigh_joint", "3_RR_calf_joint",
                              "4_RL_hip_joint", "4_RL_thigh_joint", "4_RL_calf_joint", ]

        underact_dof_names = ['front_truck_roll_joint',
                                'rear_truck_roll_joint',]

        undriven_dof_names = ['skateboard_joint_x',
                              'skateboard_joint_y',
                              'skateboard_joint_z',
                              'front_left_wheel_joint',
                              'front_right_wheel_joint',
                              'rear_left_wheel_joint',
                              'rear_right_wheel_joint']
        
        skateboard_dof_names = ['skateboard_joint_x',
                                'skateboard_joint_y',
                                'skateboard_joint_z']
        
        skateboard_link_name = ['skateboard_deck']

        wheel_link_names = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']

        wheel_dof_names = ['front_left_wheel_joint',
                           'front_right_wheel_joint',
                           'rear_left_wheel_joint',
                           'rear_right_wheel_joint']
        
        marker_link_names = [ "FR_f_marker", "FL_f_marker", "RR_f_marker", "RL_f_marker"]

        terminate_after_contacts_on = ["base", "thigh", "calf"]

        wheel_radius = 0.030
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 2.]
        randomize_base_mass = True
        added_mass_range = [0., 3.]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        action_buf_len = 8
        randomize_delay = True
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # ===GROUP1: GLIDE REWARD===
            glide_feet_on_board = 0.3
            glide_contact_num = 0.3      
            glide_feet_dis = 1.8
            glide_joint_pos = 1.2
            glide_hip_pos = 1.2

            # ===GROUP2: PUSH REWARD=== 
            push_tracking_lin_vel = 1.6
            push_tracking_ang_vel = 0.8            
            push_hip_pos = 0.6
            push_orientation = -2

            # ===GROUP3: REGULARIZATION===
            reg_wheel_contact_number = 0.8
            reg_board_body_z = 1
            reg_dof_acc = -2.5e-7
            reg_collision = -1.
            reg_action_rate = -0.22
            reg_delta_torques = -1.0e-7
            reg_torques = -0.00001
            reg_lin_vel_z = -0.1
            reg_ang_vel_xy = -0.01
            reg_orientation = -25

        cycle_time = 4
        only_positive_rewards = True
        tracking_sigma = 0.5
        tracking_sigma_yaw = 0.2
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.9
        max_contact_force = 70.
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class Go1CfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
 
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [256, 128]
        dha_hidden_dims = [256, 64, 32]
        num_modes = 3
        tsdyn_hidden_dims = [256, 128, 64]
        tsdyn_latent_dims = 20
        rnn_hidden_size = 512
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2.e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.9
        desired_kl = 0.01
        max_grad_norm = 1.
        glide_advantage_w = 0.35
        push_advantage_w = 0.4
        sim2real_advantage_w = 0.25
    
    class depth_encoder( LeggedRobotCfgPPO.depth_encoder ):
        if_depth = False    
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator( LeggedRobotCfgPPO.estimator ):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = Go1Cfg.env.n_priv
        num_prop = Go1Cfg.env.n_proprio
        num_scan = Go1Cfg.env.n_scan

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticMLP'
        algorithm_class_name = 'PPO_HDS'
        num_steps_per_env = 24
        max_iterations = 100000

        save_interval = 300
        experiment_name = 'go1'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
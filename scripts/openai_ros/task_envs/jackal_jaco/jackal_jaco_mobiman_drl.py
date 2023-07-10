#!/usr/bin/env python3

'''
LAST UPDATE: 2023.07.10

AUTHOR: Neset Unver Akmandor (NUA)

E-MAIL: akmandor.n@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:

NUA TODO:
'''

import rospy
import numpy as np
import time
import math
import cv2
import os
import csv
import random
import pathlib
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from squaternion import Quaternion

import tf
import tf2_ros
import tf2_msgs.msg
import roslaunch
import rospkg
from std_msgs.msg import Header, Bool, Float32MultiArray
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates
from octomap_msgs.msg import Octomap

from gym import spaces
from gym.envs.registration import register

from openai_ros.robot_envs import jackal_jaco_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest

from openai_ros.task_envs.jackal_jaco.mobiman_drl_config import *
#from tentabot.srv import *

#from imitation.data.types import TrajectoryWithRew
#from imitation.data import types

'''
DESCRIPTION: TODO...
'''
class JackalJacoMobimanDRL(jackal_jaco_env.JackalJacoEnv):

    '''
    DESCRIPTION: TODO...This Task Env is designed for having the JackalJaco in some kind of maze.
    It will learn how to move around the maze without crashing.
    '''
    def __init__(self, robot_id=0, data_folder_path=""):

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::__init__] START")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::__init__] data_folder_path: " + str(data_folder_path))

        self.listener = tf.TransformListener()

        ### Initialize Config Parameters
        self.config = Config(data_folder_path=data_folder_path)
        
        ### Initialize Variables
        #self.robot_namespace = "jackal_jaco_" + str(self.robot_id)
        self.init_flag = False
        self.init_goal_flag = False
        #self.robot_id = robot_id
        #self.previous_robot_id = self.robot_id
        self.step_num = 1
        self.total_step_num = 1
        self.total_collisions = 0
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.total_mean_episode_reward = 0.0
        self.goal_status = Bool()
        self.goal_status.data = False
        self.action_counter = 0
        self.observation_counter = 0

        self.init_robot_pose = {}
        self.robot_data = {}
        self.goal_data = {}
        self.arm_data = {}
        self.obs_data = {}
        #self.move_base_goal = PoseStamped()
        #self.move_base_flag = False
        self.training_data = []
        self.training_data.append(["episode_reward"])
        self.oar_data = []
        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])
        #self.time_old = time.time()

        ## Set Observation-Action-Reward data filename
        self.oar_data_file = data_folder_path + "oar_data.csv"

        # Subscriptions
        #rospy.Subscriber(self.config.goal_msg_name, MarkerArray, self.callback_goal)
        rospy.Subscriber(self.config.occgrid_msg_name, OccupancyGrid, self.callback_occgrid)
        #rospy.Subscriber(self.config.occgrid_msg_name, tf2_msgs.msg.TFMessage, self.callback_tf)

        #rospy.Subscriber(octomap_msg_name, OccupancyGrid, self.callback_octomap)
        #rospy.Subscriber("/" + str(self.robot_namespace) + "/scan", LaserScan, self.callback_laser_scan)
        #rospy.Subscriber("/" + str(self.robot_namespace) + "/laser_image", OccupancyGrid, self.callback_laser_image)
        #rospy.Subscriber("/" + str(self.robot_namespace) + "/laser_rings", Float32MultiArray, self.callback_laser_rings)

        # Services
        '''
        if  self.config.observation_space_type == "mobiman_FC" or \
            self.config.observation_space_type == "mobiman_1DCNN_FC" or \
            self.config.observation_space_type == "mobiman_2DCNN_FC" or \
            self.config.observation_space_type == "mobiman_2DCNN" or \
            self.config.observation_space_type == "mobiman_WP_FC":

            #rospy.wait_for_service('rl_step')
            #self.srv_rl_step = rospy.ServiceProxy('rl_step', rl_step, True)

            rospy.wait_for_service('update_goal')
            #self.srv_update_goal = rospy.ServiceProxy('update_goal', update_goal, True)

            #rospy.wait_for_service('reset_map_utility')
            #self.srv_reset_map_utility = rospy.ServiceProxy('reset_map_utility', reset_map_utility)

        if self.config.observation_space_type == "laser_WP_1DCNN_FC" or self.config.observation_space_type == "mobiman_WP_FC":
            
            rospy.wait_for_service('/move_base/make_plan')
            rospy.wait_for_service('/move_base/clear_costmaps')
            self.srv_move_base_get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan, True)
            # self.srv_clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty, True)
        '''

        # Publishers
        self.goal_status_pub = rospy.Publisher(self.config.goal_status_msg_name, Bool, queue_size=1)
        #self.filtered_laser_pub = rospy.Publisher(self.robot_namespace + '/laser/scan_filtered', LaserScan, queue_size=1)
        #self.debug_visu_pub = rospy.Publisher('/debug_visu', MarkerArray, queue_size=1)

        # Initialize OpenAI Gym Structure
        self.initialize_robot_pose(init_flag=False)

        super(JackalJacoMobimanDRL, self).__init__(initial_pose=self.init_robot_pose)

        self.update_robot_data()
        self.update_arm_data()
        self.update_goal_data()
        self.init_observation_action_space()

        self.reward_range = (-np.inf, np.inf)
        self.init_flag = True        

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::__init__] END")

    '''
    DESCRIPTION: TODO...Sets the Robot in its init pose NOTE: DELETE?
    '''
    def _set_init_pose(self):
        return True

    '''
    DESCRIPTION: TODO...Inits variables needed to be initialised each time we reset at the start
    of an episode.
    :return:
    '''
    def _init_env_variables(self):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_init_env_variables] START")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_init_env_variables] total_step_num: " + str(self.total_step_num))

        #self.previous_robot_id = self.robot_id
        self.episode_reward = 0.0
        self._episode_done = False
        self._reached_goal = False
        self.step_num = 1

        '''
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_init_env_variables] BEFORE client_reset_map_utility")
        # Reset Map
        success_reset_map_utility = self.client_reset_map_utility()
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_init_env_variables] AFTER client_reset_map_utility")
        '''

        self.initialize_robot_pose()

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triggered the done.
        time.sleep(1.0)

        self.update_robot_data()
        self.update_arm_data()
        self.update_goal_data()
        self.previous_base_distance2goal = self.get_base_distance2goal_2D()
        
        #self.previous_action = np.array([[self.config.init_lateral_speed, self.config.init_angular_speed]]).reshape(self.config.fc_obs_shape)

        #self.update_global_path_length()

        self.reinit_observation()
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_init_env_variables] END")
    
    '''
    DESCRIPTION: TODO...Here we define what sensor data defines our robots observations
    To know which Variables we have acces to, we need to read the
    TurtleBot3Env API DOCS
    :return:
    '''
    def _get_obs(self):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_get_obs] START")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_get_obs] total_step_num: " + str(self.total_step_num))

        # Update target observation
        self.update_observation()

        # Check if the goal is reached
        self.check_goal()

        # Check if the goal is reached
        self.check_collision()

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_get_obs] END")
        return self.obs

    '''
    DESCRIPTION: TODO...
    '''
    def _set_action(self, action):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_set_action] START")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_set_action] total_step_num: " + str(self.total_step_num))
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_set_action] action: " + str(action))
        
        # Run Action Server
        #success_rl_step = self.client_rl_step(1)
        #if not success_rl_step:
        #    rospy.logerr("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] OBSERVATION FAILURE!")

        #linear_speed = self.config.velocity_control_data[action, 0]
        #angular_speed = float(self.config.velocity_control_data[action, 1])

        #self.previous_action = np.array([[linear_speed, angular_speed]], dtype=np.float32).reshape(self.config.fc_obs_shape)
        #self.act = action

        # We tell TurtleBot3 the linear and angular speed to set to execute
        #self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_set_action] END")

    '''
    DESCRIPTION: TODO...
    '''
    def _is_done(self, observations):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] START")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] total_step_num: " + str(self.total_step_num))

        if self.step_num >= self.config.max_episode_steps:
            self._episode_done = True
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Too late...")

        if self._episode_done and (not self._reached_goal):
            rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Boooo! Episode done but not reached the goal...")
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Boooo! Episode done but not reached the goal...")
        elif self._episode_done and self._reached_goal:
            rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Gotcha! Episode done and reached the goal!")
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Gotcha! Episode done and reached the goal!")
        else:
            rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Not yet bro...")
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] Not yet bro...")

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_is_done] END")
        return self._episode_done

    '''
    DESCRIPTION: TODO...
    '''
    def _compute_reward(self, observations, done):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] START")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] done: " + str(done))
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_step_num: " + str(self.total_step_num))

        if self._episode_done and (not self._reached_goal):

            self.step_reward = self.config.penalty_terminal_fail
            self.goal_status.data = False
            self.goal_status_pub.publish(self.goal_status)

            if self.episode_num:
                #self.total_mean_episode_reward = round((self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num, self.config.mantissa_precision)
                self.total_mean_episode_reward = (self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num

            ## Add training data
            self.training_data.append([self.episode_reward])

            print("--------------")
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] robot_id: {}".format(self.robot_id))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] step_num: {}".format(self.step_num))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_step_num: {}".format(self.total_step_num))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] episode_num: {}".format(self.episode_num))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_collisions: {}".format(self.total_collisions))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] episode_reward: {}".format(self.episode_reward))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_mean_episode_reward: {}".format(self.total_mean_episode_reward))
            print("--------------")

        elif self._episode_done and self._reached_goal:

            #self.step_reward = self.config.reward_terminal_success + self.config.reward_terminal_mintime * (self.config.max_episode_steps - self.step_num) / self.config.max_episode_steps
            self.step_reward = self.config.reward_terminal_success
            self.goal_status.data = True
            self.goal_status_pub.publish(self.goal_status)

            if self.episode_num:
                #self.total_mean_episode_reward = round((self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num, self.config.mantissa_precision)
                self.total_mean_episode_reward = (self.total_mean_episode_reward * (self.episode_num - 1) + self.episode_reward) / self.episode_num

            ## Add training data
            self.training_data.append([self.episode_reward])

            print("--------------")
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] robot_id: {}".format(self.robot_id))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] step_num: {}".format(self.step_num))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_step_num: {}".format(self.total_step_num))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] episode_num: {}".format(self.episode_num))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_collisions: {}".format(self.total_collisions))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] episode_reward: {}".format(self.episode_reward))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_mean_episode_reward: {}".format(self.total_mean_episode_reward))
            print("--------------")

        else:

            '''
            if current_distance2goal > self.init_distance2goal:
                #penalty
                self.step_reward = -1 * self.config.reward_cumulative_step * current_distance2goal / (self.config.max_episode_steps * self.init_distance2goal)

            else:
                #reward
                self.step_reward = self.config.reward_cumulative_step * (self.init_distance2goal - current_distance2goal) / (self.config.max_episode_steps * self.init_distance2goal)
            '''

            ### NUA TODO: REVIEW!!!
            current_base_distance2goal = self.get_base_distance2goal_2D()
            penalty_step = self.config.penalty_cumulative_step / self.config.max_episode_steps
            rp_step = self.config.reward_step_scale * (self.previous_base_distance2goal - current_base_distance2goal)
            self.step_reward = penalty_step + rp_step
            self.previous_base_distance2goal = current_base_distance2goal

            self.step_num += 1

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] reward_step: " + str(reward_step))

            '''
            penalty_safety = 0
            if self.min_distance2obstacle < self.config.safety_range_threshold:
                
                penalty_safety = self.config.penalty_safety_scale * (self.config.safety_range_threshold / self.min_distance2obstacle)
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] penalty_safety: {}".format(penalty_safety))
            '''
            #self.step_reward = round(penalty_safety + reward_step, self.config.mantissa_precision)

            '''
            time_now = time.time()
            dt = time_now - self.time_old
            self.time_old = time_now
            
            print("----------------------")
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] current_distance2goal: " + str(current_distance2goal))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] init_distance2goal: " + str(self.init_distance2goal))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] max_episode_steps: " + str(self.config.max_episode_steps))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] reward_terminal_success: " + str(self.config.reward_terminal_success))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] reward_step_scale: " + str(self.config.reward_step_scale))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] penalty_terminal_fail: " + str(self.config.penalty_terminal_fail))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] penalty_cumulative_step: " + str(self.config.penalty_cumulative_step))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] penalty_step: " + str(penalty_step))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] rp_step: " + str(rp_step))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] step_reward: " + str(self.step_reward))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] dt: " + str(dt))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] max_lateral_speed: " + str(self.config.max_lateral_speed))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] max_step_reward: " + str(round(self.config.max_lateral_speed * dt, self.config.mantissa_precision)))
            print("----------------------")
            '''
        
        self.episode_reward += self.step_reward

        rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] step_reward: " + str(self.step_reward))
        rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] episode_reward: " + str(self.episode_reward))
        rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] total_step_num: " + str(self.total_step_num))

        '''
        print("**********************")
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] self.step_reward: " + str(self.step_reward))
        print("----------------------")
        '''

        '''
        # Save Observation-Action-Reward data into a file
        self.save_oar_data()

        if self._episode_done and (len(self.episode_oar_data['obs']) > 1):

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))

            if self.goal_status.data:
                info_data = np.ones(len(self.episode_oar_data['acts']))
            else:
                info_data = np.zeros(len(self.episode_oar_data['acts']))

            self.oar_data.append(TrajectoryWithRew( obs=np.array(self.episode_oar_data['obs']), 
                                                    acts=np.array(self.episode_oar_data['acts']),
                                                    infos=np.array(info_data),
                                                    terminal=True,
                                                    rews=np.array(self.episode_oar_data['rews']),))
        '''

        if self.total_step_num == self.config.training_timesteps:
            
            # Write Observation-Action-Reward data into a file
            #self.write_oar_data()

            ## Write training data
            write_data(self.config.data_folder_path + "training_data.csv", self.training_data)

        self.total_step_num += 1

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::_compute_reward] END")
        print("--------------------------------------------------")
        print("")
        return self.step_reward

    ################ Internal TaskEnv Methods ################

    '''
    DESCRIPTION: TODO...
    '''
    def save_oar_data(self):
        if  self.config.observation_space_type == "laser_FC" or \
            self.config.observation_space_type == "Tentabot_FC":
        
                #print("----------------------------------")
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] self.obs shape: " + str(self.obs.shape))
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] self.previous_action shape: " + str(self.previous_action.shape))
                #print("")

                obs_data = self.obs.reshape((-1)) # type: ignore

                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] obs_data shape: " + str(obs_data.shape))
                #print("----------------------------------")

                # Save Observation-Action-Reward Data
                self.episode_oar_data['obs'].append(obs_data) # type: ignore

                if not self._episode_done:

                    self.episode_oar_data['acts'].append(self.act) # type: ignore
                    #self.episode_oar_data['infos'].append()
                    #self.episode_oar_data['terminal'].append(self._episode_done)
                    self.episode_oar_data['rews'].append(self.step_reward) # type: ignore

                '''
                print("----------------------------------")
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs type: " + str(type(self.episode_oar_data['obs'])))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs len: " + str(len(self.episode_oar_data['obs'])))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data acts len: " + str(len(self.episode_oar_data['acts'])))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data: " + str(self.episode_oar_data))
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs: " + str(self.episode_oar_data.obs))
                print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] episode_oar_data obs shape: " + str(self.episode_oar_data.obs.shape))
                #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::save_oar_data] oar_data: " + str(self.oar_data))
                print("----------------------------------")
                '''

    '''
    DESCRIPTION: TODO...Save a sequence of Trajectories.

        Args:
            path: Trajectories are saved to this path.
            trajectories: The trajectories to save.
    '''
    def write_oar_data(self) -> None:
        path = self.config.data_folder_path + "oar_data.pkl"
        trajectories = self.oar_data
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        
        with open(tmp_path, "wb") as f:
            pickle.dump(trajectories, f)

        # Ensure atomic write
        os.replace(tmp_path, path)

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::write_oar_data] Written Observation-Action-Reward data!")

    '''
    DESCRIPTION: TODO...
    '''
    def callback_laser_scan(self, msg):
        self.laserscan_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_octomap(self, msg):
        self.octomap_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def callback_occgrid(self, msg):
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::callback_occgrid] INCOMING")
        self.occgrid_msg = msg

    '''
    DESCRIPTION: TODO...
    '''
    def initialize_occgrid_config(self):
        occgrid_msg = self.occgrid_msg
        self.config.set_occgrid_config(occgrid_msg)

    '''
    DESCRIPTION: TODO...
    '''
    def get_occgrid_image(self):
        occgrid_msg = self.occgrid_msg
        occgrid_image = np.array(occgrid_msg.data[0:self.config.occgrid_width])
        for i in range(1, self.config.occgrid_height):
            idx_from = i*self.config.occgrid_width
            idx_to = idx_from + self.config.occgrid_width
            occgrid_image_row = np.array(occgrid_msg.data[idx_from:idx_to])
            occgrid_image = np.vstack([occgrid_image, occgrid_image_row])

        if (self.config.occgrid_normalize_flag):
            max_scale = 1 / self.config.occgrid_occ_max
            self.occgrid_image = max_scale * occgrid_image
        else:
            self.occgrid_image = occgrid_image

        if self.step_num == 50:
            imi = (self.occgrid_image * 255).astype(np.uint8)
            im = Image.fromarray(imi)
            im = im.convert("L")
            im.save(self.config.data_folder_path + "occgrid_image.jpeg")
            np.savetxt(self.config.data_folder_path + "occgrid_image.txt", self.occgrid_image)

        #print("----------------------------------")
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] occgrid_width: " + str(self.config.occgrid_width))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] occgrid_height: " + str(self.config.occgrid_height))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] occgrid_resolution: " + str(self.config.occgrid_resolution))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] data type: " + str(type(occgrid_msg.data)))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] data len: " + str(len(data.data)))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] occgrid_image len: " + str(len(laser_image)))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] occgrid_image shape: " + str(occgrid_image.shape))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::get_occgrid_image] max_scale: " + str(max_scale))
        #print("----------------------------------")

        return occgrid_image

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def callback_move_base_global_plan(self, data):
        self.move_base_global_plan = data.poses
        self.move_base_flag = True
    '''

    '''
    DESCRIPTION: TODO... Update the odometry data
    '''
    def update_robot_data(self):

        try:
            (trans,rot) = self.listener.lookupTransform(self.config.world_frame_name, self.config.robot_frame_name, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] ERROR: No tf transform!")
        
        q = Quaternion(rot[3], rot[0], rot[1], rot[2]) # type: ignore
        e = q.to_euler(degrees=False)

        self.robot_data["x"] = trans[0] # type: ignore
        self.robot_data["y"] = trans[1] # type: ignore
        self.robot_data["z"] = trans[2] # type: ignore
        self.robot_data["qx"] = rot[0] # type: ignore
        self.robot_data["qy"] = rot[1] # type: ignore
        self.robot_data["qz"] = rot[2] # type: ignore
        self.robot_data["qw"] = rot[3] # type: ignore
        self.robot_data["roll"] = e[0]
        self.robot_data["pitch"] = e[1]
        self.robot_data["yaw"] = e[2]

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] x: " + str(self.robot_data["x"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] y: " + str(self.robot_data["y"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] z: " + str(self.robot_data["z"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] qx: " + str(self.robot_data["qx"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] qy: " + str(self.robot_data["qy"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] qz: " + str(self.robot_data["qz"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] qw: " + str(self.robot_data["qw"]))

    '''
    DESCRIPTION: TODO... Update the odometry data
    '''
    def update_arm_data(self):

        try:
            (trans,rot) = self.listener.lookupTransform(self.config.world_frame_name, self.config.ee_frame_name, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_robot_data] ERROR: No tf transform!")
        
        q = Quaternion(rot[3], rot[0], rot[1], rot[2]) # type: ignore
        e = q.to_euler(degrees=False)

        self.arm_data["x"] = trans[0] # type: ignore
        self.arm_data["y"] = trans[1] # type: ignore
        self.arm_data["z"] = trans[2] # type: ignore
        self.arm_data["qx"] = rot[0] # type: ignore
        self.arm_data["qy"] = rot[1] # type: ignore
        self.arm_data["qz"] = rot[2] # type: ignore
        self.arm_data["qw"] = rot[3] # type: ignore
        self.arm_data["roll"] = e[0]
        self.arm_data["pitch"] = e[1]
        self.arm_data["yaw"] = e[2] 

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] x: " + str(self.arm_data["x"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] y: " + str(self.arm_data["y"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] z: " + str(self.arm_data["z"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] qx: " + str(self.arm_data["qx"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] qy: " + str(self.arm_data["qy"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] qz: " + str(self.arm_data["qz"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_arm_data] qw: " + str(self.arm_data["qw"]))

    '''
    DESCRIPTION: TODO... Check if the goal is reached
    '''
    def check_goal(self):
        distance2goal_ee = self.get_arm_distance2goal_3D()
        if (distance2goal_ee < self.config.goal_range_min):
            self._episode_done = True
            self._reached_goal = True

    '''
    DESCRIPTION: TODO...Gets the initial location of the robot to reset
    '''
    def initialize_robot_pose(self, init_flag=True):
        robot0_init_yaw = 0.0
        if self.config.world_name == "conveyor":

            init_robot_pose_areas_x = []
            init_robot_pose_areas_x.extend(([-2.0,2.0], [-2.0,2.0], [-2.0,-2.0], [-2.0,2.0]))

            init_robot_pose_areas_y = []
            init_robot_pose_areas_y.extend(([-1.0,1.0], [-1.0,1.0], [-1.0,1.0], [-1.0,1.0]))

            area_idx = random.randint(0, len(init_robot_pose_areas_x)-1)
            self.robot_init_area_id = area_idx

            self.init_robot_pose["x"] = random.uniform(init_robot_pose_areas_x[area_idx][0], init_robot_pose_areas_x[area_idx][1])
            self.init_robot_pose["y"] = random.uniform(init_robot_pose_areas_y[area_idx][0], init_robot_pose_areas_y[area_idx][1])
            self.init_robot_pose["z"] = 0.0
            robot0_init_yaw = random.uniform(0.0, 2*math.pi)

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] x: " + str(self.init_robot_pose["x"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] y: " + str(self.init_robot_pose["y"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] z: " + str(self.init_robot_pose["z"]))

        robot0_init_quat = Quaternion.from_euler(0, 0, robot0_init_yaw)
        self.init_robot_pose["qx"] = robot0_init_quat.x
        self.init_robot_pose["qy"] = robot0_init_quat.y
        self.init_robot_pose["qz"] = robot0_init_quat.z
        self.init_robot_pose["qw"] = robot0_init_quat.w

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] Updated init_robot_pose x: " + str(self.init_robot_pose["x"]) + ", y: " + str(self.init_robot_pose["y"]))
        rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::initialize_robot_pose] Updated init_robot_pose x: " + str(self.init_robot_pose["x"]) + ", y: " + str(self.init_robot_pose["y"]))

        if init_flag:
            super(JackalJacoMobimanDRL, self).update_initial_pose(self.init_robot_pose)

    '''
    DESCRIPTION: TODO...
    '''
    def callback_tf(self, msg):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::callback_tf] INCOMING")

    '''
    DESCRIPTION: TODO...Gets the goal location for each robot
    '''
    def update_goal_data(self):
        try:
            (translation_wrt_world, rotation_wrt_world) = self.listener.lookupTransform(self.config.world_frame_name, self.config.goal_frame_name, rospy.Time(0))
            (translation_wrt_robot, rotation_wrt_robot) = self.listener.lookupTransform(self.config.robot_frame_name, self.config.goal_frame_name, rospy.Time(0))
            (translation_wrt_ee, rotation_wrt_ee) = self.listener.lookupTransform(self.config.ee_frame_name, self.config.goal_frame_name, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] ERROR: No tf transform!")

        self.goal_data["x"] = translation_wrt_world[0] # type: ignore
        self.goal_data["y"] = translation_wrt_world[1] # type: ignore
        self.goal_data["z"] = translation_wrt_world[2] # type: ignore
        self.goal_data["qx"] = rotation_wrt_world[0] # type: ignore
        self.goal_data["qy"] = rotation_wrt_world[1] # type: ignore
        self.goal_data["qz"] = rotation_wrt_world[2] # type: ignore
        self.goal_data["qw"] = rotation_wrt_world[3] # type: ignore

        self.goal_data["x_wrt_robot"] = translation_wrt_robot[0] # type: ignore
        self.goal_data["y_wrt_robot"] = translation_wrt_robot[1] # type: ignore
        self.goal_data["z_wrt_robot"] = translation_wrt_robot[2] # type: ignore
        self.goal_data["qx_wrt_robot"] = rotation_wrt_robot[0] # type: ignore
        self.goal_data["qy_wrt_robot"] = rotation_wrt_robot[1] # type: ignore
        self.goal_data["qz_wrt_robot"] = rotation_wrt_robot[2] # type: ignore
        self.goal_data["qw_wrt_robot"] = rotation_wrt_robot[3] # type: ignore

        self.goal_data["x_wrt_ee"] = translation_wrt_ee[0] # type: ignore
        self.goal_data["y_wrt_ee"] = translation_wrt_ee[1] # type: ignore
        self.goal_data["z_wrt_ee"] = translation_wrt_ee[2] # type: ignore
        self.goal_data["qx_wrt_ee"] = rotation_wrt_ee[0] # type: ignore
        self.goal_data["qy_wrt_ee"] = rotation_wrt_ee[1] # type: ignore
        self.goal_data["qz_wrt_ee"] = rotation_wrt_ee[2] # type: ignore
        self.goal_data["qw_wrt_ee"] = rotation_wrt_ee[3] # type: ignore
        
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] x: " + str(self.goal_data["x"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] y: " + str(self.goal_data["y"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] z: " + str(self.goal_data["z"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qx: " + str(self.goal_data["qx"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qy: " + str(self.goal_data["qy"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qz: " + str(self.goal_data["qz"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qw: " + str(self.goal_data["qw"]))

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] x_wrt_robot: " + str(self.goal_data["x_wrt_robot"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] y_wrt_robot: " + str(self.goal_data["y_wrt_robot"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] z_wrt_robot: " + str(self.goal_data["z_wrt_robot"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qx_wrt_robot: " + str(self.goal_data["qx_wrt_robot"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qy_wrt_robot: " + str(self.goal_data["qy_wrt_robot"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qz_wrt_robot: " + str(self.goal_data["qz_wrt_robot"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qw_wrt_robot: " + str(self.goal_data["qw_wrt_robot"]))

        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] x_wrt_ee: " + str(self.goal_data["x_wrt_ee"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] y_wrt_ee: " + str(self.goal_data["y_wrt_ee"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] z_wrt_ee: " + str(self.goal_data["z_wrt_ee"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qx_wrt_ee: " + str(self.goal_data["qx_wrt_ee"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qy_wrt_ee: " + str(self.goal_data["qy_wrt_ee"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qz_wrt_ee: " + str(self.goal_data["qz_wrt_ee"]))
        #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_goal_data] qw_wrt_ee: " + str(self.goal_data["qw_wrt_ee"]))

    '''
    DESCRIPTION: TODO...
    '''
    def get_euclidean_distance_2D(self, p1, p2={"x":0.0, "y":0.0}):
        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
    
    '''
    DESCRIPTION: TODO...
    '''
    def get_euclidean_distance_3D(self, p1, p2={"x":0.0, "y":0.0, "z":0.0}):
        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2 + (p1["z"] - p2["z"])**2)

    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_init_distance2goal_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.goal_data, self.init_robot_pose)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Get the initial distance to the goal
    '''
    def get_init_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.init_robot_pose)
        return distance2goal

    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_base_distance2goal_2D(self):
        distance2goal = self.get_euclidean_distance_2D(self.goal_data, self.robot_data)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_base_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.robot_data)
        return distance2goal
    
    '''
    DESCRIPTION: TODO...Gets the distance to the goal
    '''
    def get_arm_distance2goal_3D(self):
        distance2goal = self.get_euclidean_distance_3D(self.goal_data, self.arm_data)
        return distance2goal

    '''
    DESCRIPTION: TODO...
    value.
    '''
    def check_collision(self):
        ### NUA TODO: GET THE DATA FROM SUBSCRIBED TOPIC! 
        return False
        '''
        laserscan_msg = self.laserscan_msg
        self.min_distance2obstacle = min(laserscan_msg.ranges)
        for scan_range in laserscan_msg.ranges:
            if (self.config.collision_range_min > scan_range > 0):
                if not self._episode_done:
                    self.total_collisions += 1
                    #rospy.logdebug("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::check_collision] Hit me baby one more time!")
                    print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::check_collision] Hit me baby one more time!")
                self._episode_done = True
                return True
        return False
        '''

    '''
    DESCRIPTION: TODO...
    value.
    '''
    def get_obs_occgrid(self, image_flag=False):
        if image_flag:
            obs_occgrid = self.get_occgrid_image()
        else:
            occgrid_msg = self.occgrid_msg
            obs_occgrid = np.asarray(occgrid_msg.data)
            if self.config.occgrid_normalize_flag:
                max_scale = 1 / self.config.occgrid_occ_max
                obs_occgrid = max_scale * obs_occgrid
            obs_occgrid = obs_occgrid.reshape(self.config.fc_obs_shape)
        return obs_occgrid

    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_colspheredist(self):
        ### NUA TODO: GET THE DATA FROM SUBSCRIBED TOPIC! 
        obs_colspheredist = np.full((1, self.config.n_colsphere), self.config.collision_range_max).reshape(self.config.fc_obs_shape) # type: ignore
        return obs_colspheredist

    '''
    DESCRIPTION: TODO...
    '''
    def get_obs_goal(self):
        quat_goal_wrt_ee = Quaternion(self.goal_data["qw_wrt_ee"], self.goal_data["qx_wrt_ee"], self.goal_data["qy_wrt_ee"], self.goal_data["qz_wrt_ee"])
        euler_goal_wrt_ee = quat_goal_wrt_ee.to_euler(degrees=False)

        obs_goal = np.array([[self.goal_data["x_wrt_robot"], 
                              self.goal_data["y_wrt_robot"], 
                              self.goal_data["z_wrt_robot"],
                              self.goal_data["x_wrt_ee"], 
                              self.goal_data["y_wrt_ee"], 
                              self.goal_data["z_wrt_ee"],
                              euler_goal_wrt_ee[0],
                              euler_goal_wrt_ee[1],
                              euler_goal_wrt_ee[2]]]).reshape(self.config.fc_obs_shape) # type: ignore
        return obs_goal

    '''
    DESCRIPTION: TODO...
    '''
    def init_observation_action_space(self):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] START")
        self.initialize_occgrid_config()

        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        if self.config.observation_space_type == "mobiman_FC":
            
            # Occupancy (OccupancyGrid data)
            if self.config.occgrid_normalize_flag:   
                obs_occgrid_min = np.full((1, self.config.occgrid_data_size), 0.0).reshape(self.config.fc_obs_shape)
                obs_occgrid_max = np.full((1, self.config.occgrid_data_size), 1.0).reshape(self.config.fc_obs_shape)
            else:
                obs_occgrid_min = np.full((1, self.config.occgrid_data_size), self.config.occgrid_occ_min).reshape(self.config.fc_obs_shape)
                obs_occgrid_max = np.full((1, self.config.occgrid_data_size), self.config.occgrid_occ_max).reshape(self.config.fc_obs_shape)

            # Nearest collision distances (from spheres on robot body)
            obs_colspheredist_min = np.full((1, self.config.n_colsphere), self.config.collision_range_min).reshape(self.config.fc_obs_shape) # type: ignore
            obs_colspheredist_max = np.full((1, self.config.n_colsphere), self.config.collision_range_max).reshape(self.config.fc_obs_shape) # type: ignore

            # Goal (wrt. robot)
            obs_goal_min = np.array([[self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      -math.pi, 
                                      -math.pi, 
                                      -math.pi]]).reshape(self.config.fc_obs_shape) # type: ignore
            obs_goal_max = np.array([[self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      math.pi, 
                                      math.pi, 
                                      math.pi]]).reshape(self.config.fc_obs_shape) # type: ignore

            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_occgrid_min shape: " + str(obs_occgrid_min.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_colspheredist_min shape: " + str(obs_colspheredist_min.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_goal_min shape: " + str(obs_goal_min.shape))

            self.obs_data = {   "occgrid": np.vstack([obs_occgrid_min] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "colspheredist": np.vstack([obs_colspheredist_min] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))} # type: ignore

            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_data occgrid shape: " + str(self.obs_data["occgrid"].shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_data colspheredist shape: " + str(self.obs_data["colspheredist"].shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_data goal shape: " + str(self.obs_data["goal"].shape))

            obs_stacked_occgrid_min = np.hstack([obs_occgrid_min] * self.config.n_obs_stack) # type: ignore
            obs_stacked_occgrid_max = np.hstack([obs_occgrid_max] * self.config.n_obs_stack) # type: ignore

            obs_space_min = np.concatenate((obs_stacked_occgrid_min, obs_colspheredist_min, obs_goal_min), axis=0)
            obs_space_max = np.concatenate((obs_stacked_occgrid_max, obs_colspheredist_max, obs_goal_max), axis=0)

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_stacked_laser_low shape: " + str(obs_stacked_laser_low.shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_space_low shape: " + str(obs_space_low.shape))

            self.obs = obs_space_min
            self.observation_space = spaces.Box(obs_space_min, obs_space_max)
            self.action_space = spaces.Discrete(self.config.n_action)

        elif self.config.observation_space_type == "mobiman_2DCNN_FC":

            # Occupancy (OccupancyGrid image)
            obs_occgrid_image_min = np.full((1, self.config.occgrid_width), 0.0)
            obs_occgrid_image_min = np.vstack([obs_occgrid_image_min] * self.config.occgrid_height)
            obs_occgrid_image_min = np.expand_dims(obs_occgrid_image_min, axis=0)

            obs_occgrid_image_max = np.full((1, self.config.occgrid_width), 1.0)
            obs_occgrid_image_max = np.vstack([obs_occgrid_image_max] * self.config.occgrid_height)
            obs_occgrid_image_max = np.expand_dims(obs_occgrid_image_max, axis=0)

            # Nearest collision distances (from spheres on robot body)
            obs_colspheredist_min = np.full((1, self.config.n_colsphere), self.config.collision_range_min).reshape(self.config.fc_obs_shape) # type: ignore
            obs_colspheredist_max = np.full((1, self.config.n_colsphere), self.config.collision_range_max).reshape(self.config.fc_obs_shape) # type: ignore

            # Goal (wrt. robot)
            obs_goal_min = np.array([[self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      self.config.goal_range_min, 
                                      -math.pi, 
                                      -math.pi, 
                                      -math.pi]]).reshape(self.config.fc_obs_shape) # type: ignore
            obs_goal_max = np.array([[self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      self.config.goal_range_max, 
                                      math.pi, 
                                      math.pi, 
                                      math.pi]]).reshape(self.config.fc_obs_shape) # type: ignore

            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_occgrid_image_min shape: " + str(obs_occgrid_image_min.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_colspheredist_min shape: " + str(obs_colspheredist_min.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] obs_goal_min shape: " + str(obs_goal_min.shape))

            self.obs_data = {   "occgrid_image": np.vstack([obs_occgrid_image_min] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "colspheredist": np.vstack([obs_colspheredist_min] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal_min] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))} # type: ignore

            obs_space_occgrid_image_min = np.vstack([obs_occgrid_image_min] * self.config.n_obs_stack) # type: ignore
            obs_space_occgrid_image_max = np.vstack([obs_occgrid_image_max] * self.config.n_obs_stack) # type: ignore

            obs_space_colspheredist_goal_min = np.concatenate((obs_colspheredist_min, obs_goal_min), axis=0)
            obs_space_colspheredist_goal_max = np.concatenate((obs_colspheredist_max, obs_goal_max), axis=0)

            self.obs = {"occgrid_image": obs_space_occgrid_image_min, 
                        "colspheredist_goal": obs_space_colspheredist_goal_min}

            self.observation_space = spaces.Dict({  "occgrid_image": spaces.Box(obs_space_occgrid_image_min, obs_space_occgrid_image_max), 
                                                    "colspheredist_goal": spaces.Box(obs_space_colspheredist_goal_min, obs_space_colspheredist_goal_max)})

            self.action_space = spaces.Discrete(self.config.n_action)

        print()

        self.config.set_observation_shape(self.observation_space.shape)
        self.config.set_action_shape("Discrete, " + str(self.action_space.n))

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] observation_space shape: " + str(self.observation_space.shape))
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] action_space shape: " + str(self.action_space.shape))

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] observation_space: " + str(self.observation_space))
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] action_space: " + str(self.action_space))
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::init_observation_action_space] END")

    '''
    DESCRIPTION: TODO...
    '''
    def reinit_observation(self):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] START")

        self.episode_oar_data = dict(obs=[], acts=[], infos=None, terminal=[], rews=[])

        if self.config.observation_space_type == "mobiman_FC":
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] mobiman_FC")

            # Get OccGrid array observation
            obs_occgrid = self.get_obs_occgrid()

            # Get collision sphere distance observation
            obs_colspheredist = self.get_obs_colspheredist()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Stack observation data
            self.obs_data = {   "occgrid": np.vstack([obs_occgrid] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "colspheredist": np.vstack([obs_colspheredist] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))} # type: ignore

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_occgrid shape: " + str(obs_occgrid.shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_colspheredist shape: " + str(obs_colspheredist.shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_goal shape: " + str(obs_goal.shape))

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_data laser shape: " + str(self.obs_data["occgrid"].shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_data target shape: " + str(self.obs_data["colspheredist"].shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_data action shape: " + str(self.obs_data["goal"].shape))

            # Initialize observation
            obs_stacked_occgrid = np.hstack([obs_occgrid] * self.config.n_obs_stack) # type: ignore

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_stacked_laser shape: " + str(obs_stacked_laser.shape))

            self.obs = np.concatenate((obs_stacked_occgrid, obs_colspheredist, obs_goal), axis=0)

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "mobiman_2DCNN_FC":
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] mobiman_2DCNN_FC")

            # Get OccGrid image observation
            obs_occgrid_image = self.get_obs_occgrid(image_flag=True)
            obs_occgrid_image = np.expand_dims(obs_occgrid_image, axis=0)

            # Get collision sphere distance observation
            obs_colspheredist = self.get_obs_colspheredist()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Stack observation data
            self.obs_data = {   "occgrid_image": np.vstack([obs_occgrid_image] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "colspheredist": np.vstack([obs_colspheredist] * (self.config.n_obs_stack * self.config.n_skip_obs_stack)), # type: ignore
                                "goal": np.vstack([obs_goal] * (self.config.n_obs_stack * self.config.n_skip_obs_stack))} # type: ignore

            # Initialize observation                    
            obs_space_occgrid_image = np.vstack([obs_occgrid_image] * self.config.n_obs_stack) # type: ignore
            obs_space_colspheredist_goal = np.concatenate((obs_colspheredist, obs_goal), axis=0)

            print("---------------------")
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_occgrid_image shape: " + str(obs_occgrid_image.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_colspheredist shape: " + str(obs_colspheredist.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_goal shape: " + str(obs_goal.shape))

            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_data occgrid_image shape: " + str(self.obs_data["occgrid_image"].shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_data colspheredist shape: " + str(self.obs_data["colspheredist"].shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_space_occgrid_image shape: " + str(obs_space_occgrid_image.shape))
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] obs_space_colspheredist_goal shape: " + str(obs_space_colspheredist_goal.shape))
            print("---------------------")

            self.obs = {"occgrid_image": obs_space_occgrid_image,
                        "colspheredist_goal": obs_space_colspheredist_goal}
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::reinit_observation] END")
    
    '''
    DESCRIPTION: TODO...
    '''
    def update_observation(self):
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] START")
        if self.config.observation_space_type == "mobiman_FC":
            print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] mobiman_FC")

            # Get OccGrid array observation
            obs_occgrid = self.get_obs_occgrid()

            # Get collision sphere distance observation
            obs_colspheredist = self.get_obs_colspheredist()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Update observation data
            self.obs_data["occgrid"] = np.vstack((self.obs_data["occgrid"], obs_occgrid))
            self.obs_data["occgrid"] = np.delete(self.obs_data["occgrid"], np.s_[0], axis=0)

            self.obs_data["colspheredist"] = np.vstack((self.obs_data["colspheredist"], obs_colspheredist))
            self.obs_data["colspheredist"] = np.delete(self.obs_data["colspheredist"], np.s_[0], axis=0)

            self.obs_data["goal"] = np.vstack((self.obs_data["goal"], obs_goal))
            self.obs_data["goal"] = np.delete(self.obs_data["goal"], np.s_[0], axis=0)

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_data occgrid shape: " + str(self.obs_data["occgrid"].shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_data colspheredist shape: " + str(self.obs_data["colspheredist"].shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))

            # Update observation
            obs_stacked_occgrid = self.obs_data["occgrid"][-1,:].reshape(self.config.fc_obs_shape)

            if self.config.n_obs_stack > 1:
                latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                j = 0
                for i in range(latest_index-1, -1, -1): # type: ignore
                    j += 1
                    if j % self.config.n_skip_obs_stack == 0:
                        obs_stacked_occgrid = np.hstack((self.obs_data["occgrid"][i,:], obs_stacked_occgrid))

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_stacked_occgrid shape: " + str(obs_stacked_occgrid.shape))

            self.obs = np.concatenate((obs_stacked_occgrid, obs_colspheredist, obs_goal), axis=0)

            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs: " + str(self.obs.shape))

        elif self.config.observation_space_type == "mobiman_2DCNN_FC":

            # Get OccGrid image observation
            obs_occgrid_image = self.get_obs_occgrid(image_flag=True)
            obs_occgrid_image = np.expand_dims(obs_occgrid_image, axis=0)

            # Get collision sphere distance observation
            obs_colspheredist = self.get_obs_colspheredist()

            # Update goal observation
            obs_goal = self.get_obs_goal()

            # Update observation data
            self.obs_data["occgrid_image"] = np.vstack((self.obs_data["occgrid_image"], obs_occgrid_image))
            self.obs_data["occgrid_image"] = np.delete(self.obs_data["occgrid_image"], np.s_[0], axis=0)

            self.obs_data["colspheredist"] = np.vstack((self.obs_data["colspheredist"], obs_colspheredist))
            self.obs_data["colspheredist"] = np.delete(self.obs_data["colspheredist"], np.s_[0], axis=0)

            self.obs_data["goal"] = np.vstack((self.obs_data["goal"], obs_goal))
            self.obs_data["goal"] = np.delete(self.obs_data["goal"], np.s_[0], axis=0)

            # Update observation
            obs_space_occgrid_image = self.obs_data["occgrid_image"][-1,:,:]
            obs_space_occgrid_image = np.expand_dims(obs_space_occgrid_image, axis=0)

            if self.config.n_obs_stack > 1:
                if(self.config.n_skip_obs_stack > 1):
                    latest_index = (self.config.n_obs_stack * self.config.n_skip_obs_stack) - 1
                    j = 0
                    for i in range(latest_index-1, -1, -1): # type: ignore
                        j += 1
                        if j % self.config.n_skip_obs_stack == 0:

                            obs_space_occgrid_image_current = self.obs_data["occgrid_image"][i,:,:]
                            obs_space_occgrid_image_current = np.expand_dims(obs_space_occgrid_image_current, axis=0)
                            obs_space_occgrid_image = np.vstack([obs_space_occgrid_image_current, obs_space_occgrid_image])
                
                else:
                    obs_space_occgrid_image = self.obs_data["occgrid_image"]

            obs_space_colspheredist_goal = np.concatenate((obs_colspheredist, obs_goal), axis=0)

            #print("**************** " + str(self.step_num))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_data occgrid_image shape: " + str(self.obs_data["occgrid_image"].shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_data colspheredist shape: " + str(self.obs_data["colspheredist"].shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_data goal shape: " + str(self.obs_data["goal"].shape))
            ##print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_space_laser_image: ")
            ##print(obs_space_laser_image[0, 65:75])
            ##print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_target dist: " + str(obs_target[0,0]))
            ##print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_target angle: " + str(obs_target[0,1] * 180 / math.pi))
            ##print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] previous_action: " + str(self.previous_action))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_occgrid_image shape: " + str(obs_occgrid_image.shape))
            ##print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_space_laser_image type: " + str(type(obs_space_laser_image)))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_space_occgrid_image shape: " + str(obs_space_occgrid_image.shape))
            #print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] obs_space_colspheredist_goal shape: " + str(obs_space_colspheredist_goal.shape))
            #print("****************")

            self.obs["occgrid_image"] = obs_space_occgrid_image
            self.obs["colspheredist_goal"] = obs_space_colspheredist_goal
        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::update_observation] END")

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def publish_goal(self):
        
        goal_visu = MarkerArray()

        marker = Marker()
        marker.header.frame_id = self.config.world_frame_name
        marker.ns = ""
        marker.id = 1
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_pose["x"]
        marker.pose.position.y = self.goal_pose["y"]
        marker.pose.position.z = self.goal_pose["z"]

        marker.header.seq += 1
        marker.header.stamp = rospy.Time.now()

        goal_visu.markers.append(marker) # type: ignore

        self.goal_visu_pub.publish(goal_visu)
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def publish_move_base_goal(self):

        print("[jackal_jaco_mobiman_drl::JackalJacoMobimanDRL::publish_move_base_goal] x: " + str(self.goal_pose["x"]) + " y: " + str(self.goal_pose["y"]) + " z: " + str(self.goal_pose["z"]))

        self.move_base_goal.pose.position.x = self.goal_pose["x"]
        self.move_base_goal.pose.position.y = self.goal_pose["y"]
        self.move_base_goal.pose.position.z = self.goal_pose["z"]
        self.move_base_goal.pose.orientation.z = 0.0
        self.move_base_goal.pose.orientation.w = 1.0
        self.move_base_goal.header.seq += 1
        self.move_base_goal.header.frame_id = self.config.world_frame_name
        self.move_base_goal.header.stamp = rospy.Time.now()
        self.move_base_goal_pub.publish(self.move_base_goal) # type: ignore
    '''

    '''
    DESCRIPTION: TODO...
    '''
    def publish_debug_visu(self, debug_data):

        debug_visu = MarkerArray()

        for i, d in enumerate(debug_data):

            if d[0] < float("inf") or d[1] < float("inf"):
                marker = Marker()
                marker.header.frame_id = self.config.world_frame_name
                marker.ns = str(i)
                marker.id = i
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = d[0]
                marker.pose.position.y = d[1]
                marker.pose.position.z = 0

                debug_visu.markers.append(marker) # type: ignore
        
        if len(debug_visu.markers) > 0: # type: ignore
            
            for m in debug_visu.markers: # type: ignore
                m.header.seq += 1
                m.header.stamp = rospy.Time.now()
            
            self.debug_visu_pub.publish(debug_visu) # type: ignore

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def publish_wp_visu(self, full_data, obs_data):

        debug_visu = MarkerArray()

        #delete previous markers
        marker = Marker()
        marker.ns = str(0)
        marker.id = 0
        marker.action = marker.DELETEALL
        debug_visu.markers.append(marker) # type: ignore

        for i,d in enumerate(full_data):

            marker = Marker()
            marker.header.frame_id = self.config.world_frame_name
            marker.ns = str(i+1)
            marker.id = i+1
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = d.pose.position.x
            marker.pose.position.y = d.pose.position.y
            marker.pose.position.z = 0

            debug_visu.markers.append(marker) # type: ignore

        for j,d in enumerate(obs_data):

            marker = Marker()
            marker.header.frame_id = self.config.world_frame_name
            marker.ns = str(len(full_data)+j+1)
            marker.id = len(full_data)+j+1
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = d.pose.position.x
            marker.pose.position.y = d.pose.position.y
            marker.pose.position.z = 0

            debug_visu.markers.append(marker) # type: ignore

        if len(debug_visu.markers) > 0: # type: ignore
            
            for m in debug_visu.markers: # type: ignore
                m.header.seq += 1
                m.header.stamp = rospy.Time.now()
            
            self.debug_visu_pub.publish(debug_visu) # type: ignore
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def client_rl_step(self, parity):
        
        #rospy.wait_for_service('rl_step')
        try:
            #srv_rl_step = rospy.ServiceProxy('rl_step', rl_step, True)
            tentabot_client = self.srv_rl_step(parity)
            
            if self.config.observation_space_type == "Tentabot_FC":
                self.occupancy_set = (np.asarray(tentabot_client.occupancy_set)).reshape(self.config.fc_obs_shape)

            else:
                self.occupancy_set = (np.asarray(tentabot_client.occupancy_set)).reshape(self.config.cnn_obs_shape)

            print("--------------")
            print("turtlebot3_tentabot_drl::client_rl_step -> min id: " + str(np.argmin(self.occupancy_set)) + " val: " + str(np.min(self.occupancy_set)))
            print("turtlebot3_tentabot_drl::client_rl_step -> max id: " + str(np.argmax(self.occupancy_set)) + " val: " + str(np.max(self.occupancy_set)))
            print("turtlebot3_tentabot_drl::client_rl_step -> ")
            for i, val in enumerate(self.occupancy_set[0]):
                if 65 < i < 80:
                    print(str(i) + ": " + str(val))
            print("--------------")

            #self.tentabot_obs = occupancy_set
            #self.obs = np.stack((clearance_set, clutterness_set, closeness_set), axis=0)
            return True

        except rospy.ServiceException as e:
            print("turtlebot3_tentabot_drl::client_rl_step -> Service call failed: %s"%e)
            return False
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def client_update_goal(self):

        #rospy.wait_for_service('update_goal')
        try:
            #srv_update_goal = rospy.ServiceProxy('update_goal', update_goal, True)
            print("turtlebot3_tentabot_drl::get_goal_location -> Updated goal_pose x: " + str(self.goal_pose["x"]) + ", y: " + str(self.goal_pose["y"]))
            
            goalMsg = Pose()
            goalMsg.orientation.z = 0.0
            goalMsg.orientation.w = 1.0
            goalMsg.position.x = self.goal_pose["x"]
            goalMsg.position.y = self.goal_pose["y"]
            goalMsg.position.z = self.goal_pose["z"]

            success = self.srv_update_goal(goalMsg).success

            if(success):
                #print("turtlebot3_tentabot_drl::get_goal_location -> Updated goal_pose x: " + str(self.goal_pose["x"]) + ", y: " + str(self.goal_pose["y"]))
                rospy.logdebug("turtlebot3_tentabot_drl::client_update_goal -> Updated goal_pose x: " + str(self.goal_pose["x"]) + ", y: " + str(self.goal_pose["y"]))
            else:
                #print("turtlebot3_tentabot_drl::client_update_goal -> goal_pose is NOT updated!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_update_goal -> goal_pose is NOT updated!")

            return success

        except rospy.ServiceException as e:  
            print("turtlebot3_tentabot_drl::client_update_goal -> Service call failed: %s"%e)
            return False
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def client_reset_map_utility(self, parity):

        #rospy.wait_for_service('reset_map_utility')
        try:
            #srv_reset_map_utility = rospy.ServiceProxy('reset_map_utility', reset_map_utility, True)        
            
            success = self.srv_reset_map_utility(parity).success
            
            if(success):
                print("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is reset!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is reset!")

            else:
                print("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is NOT reset!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_reset_map_utility -> Map is NOT reset!")

            return success

        # Reset Robot Pose and Goal
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def client_move_base_get_plan(self):

        try:
            self.update_robot_data()

            start = PoseStamped()
            start.pose = self.odom_data.pose.pose
            start.header.seq += 1
            start.header.frame_id = self.config.world_frame_name
            start.header.stamp = rospy.Time.now()

            goal = PoseStamped()
            goal.pose.position.x = self.goal_pose["x"]
            goal.pose.position.y = self.goal_pose["y"]
            goal.pose.position.z = self.goal_pose["z"]
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 1.0
            goal.header.seq += 1
            goal.header.frame_id = self.config.world_frame_name
            goal.header.stamp = rospy.Time.now()

            tolerance = 0.5
            
            #self.srv_clear_costmap()
            self.move_base_global_plan = self.srv_move_base_get_plan(start, goal, tolerance).plan.poses
            
            if(len(self.move_base_global_plan)):
                
                #print("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                success = True

            else:
            
                print("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                rospy.logdebug("turtlebot3_tentabot_drl::client_move_base_get_plan -> move_base plan is received!")
                success = False

            return success

        # Reset Robot Pose and Goal
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def update_global_path_length(self):

        if self.client_move_base_get_plan():

            n_points_plan = len(self.move_base_global_plan)
            self.global_plan_length = 0

            p1 = {'x': self.move_base_global_plan[0].pose.position.x, 'y': self.move_base_global_plan[0].pose.position.y, 'z': self.move_base_global_plan[0].pose.position.z}
            for i in range(1, n_points_plan):

                p2 = {'x': self.move_base_global_plan[i].pose.position.x, 'y': self.move_base_global_plan[i].pose.position.y, 'z': self.move_base_global_plan[i].pose.position.z}
                self.global_plan_length += self.get_euclidean_distance(p1, p2)
                p1 = p2

            print("turtlebot3_tentabot_drl::update_global_path_length -> global_plan_length: " + str(self.global_plan_length))
    '''

    '''
    DESCRIPTION: TODO...
    '''
    '''
    def reset_pedsim(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.tentabot_path + "/launch/others/pedsim_ros/start_pedsim_validation.launch"])
        launch.start()
    '''
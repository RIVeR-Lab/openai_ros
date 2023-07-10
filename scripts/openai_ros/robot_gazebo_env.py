#!/usr/bin/env python3

'''
LAST UPDATE: 2023.06.30

AUTHOR:     OPENAI_ROS
            Neset Unver Akmandor (NUA)

E-MAIL: akmandor.n@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:
[1] 

NUA TODO:
'''

import rospy
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection

#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from openai_ros.msg import RLExperimentInfo

'''
DESCRIPTION: TODO...
'''
class RobotGazeboEnv(gym.Env):

    '''
    DESCRIPTION: TODO...
    '''
    def __init__(self, robot_namespace, controllers_list, reset_controls, start_init_physics_parameters=False, reset_world_or_sim="ROBOT", initial_pose={}):
        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::__init__] START")
        print("[robot_gazebo_env::RobotGazeboEnv::__init__] START")
        
        self.initial_pose = initial_pose
        self.gazebo = GazeboConnection(start_init_physics_parameters, reset_world_or_sim, robot_namespace=robot_namespace, initial_pose=self.initial_pose)
        self.controllers_object = ControllersConnection(namespace=robot_namespace, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # Set up ROS related variables
        self.episode_num = 1
        self.cumulated_episode_reward = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)

        # We Unpause the simulation and reset the controllers if needed
        """
        OPENAI_ROS: To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        """
        self.gazebo.unpauseSim()
        print("[robot_gazebo_env::RobotGazeboEnv::__init__] BEFORE reset_controllers")
        if self.reset_controls:
            self.controllers_object.reset_controllers()

        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::__init__] END")
        print("[robot_gazebo_env::RobotGazeboEnv::__init__] END")

    # Env methods

    '''
    DESCRIPTION: TODO...
    '''
    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    '''
    DESCRIPTION: TODO...Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action. Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        :param action:
        :return: obs, reward, done, info
    '''
    def step(self, action):
        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::step] START")

        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward

        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::step] END")
        return obs, reward, done, info

    '''
    DESCRIPTION: TODO...
    '''
    def reset(self):
        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::reset] START")
        #print("[robot_gazebo_env::RobotGazeboEnv::reset] START")

        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()

        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::reset] END")
        #print("[robot_gazebo_env::RobotGazeboEnv::reset] END")
        return obs

    '''
    DESCRIPTION: TODO...Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
    '''
    def close(self):
        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::close] Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("[robot_gazebo_env::RobotGazeboEnv::close] Closing RobotGazeboEnvironment")

    '''
    DESCRIPTION: TODO...Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
    '''
    def _update_episode(self):
        #self._publish_reward_topic(self.cumulated_episode_reward, self.episode_num)
        #rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::_update_episode] cumulated_episode_reward: " + str(self.cumulated_episode_reward))

        self.episode_num += 1
        #self.cumulated_episode_reward = 0

    '''
    DESCRIPTION: TODO...This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
    '''
    def _publish_reward_topic(self, reward, episode_number=1):

        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    '''
    DESCRIPTION: TODO...Resets a simulation
    '''
    def _reset_sim(self):
        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::_reset_sim] START")
        if self.reset_controls:
            rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::_reset_sim] RESET CONTROLLERS")
            print("[robot_gazebo_env::RobotGazeboEnv::_reset_sim] RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        else:
            rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::_reset_sim] DONT RESET CONTROLLERS")
            print("[robot_gazebo_env::RobotGazeboEnv::_reset_sim] DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("[robot_gazebo_env::RobotGazeboEnv::_reset_sim] END")
        return True

    '''
    DESCRIPTION: TODO...Sets the Robot in its init pose
    '''
    def _set_init_pose(self):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Checks that all the sensors, publishers and other simulation systems are
        operational.
    '''
    def _check_all_systems_ready(self):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Returns the observation.
    '''
    def _get_obs(self):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Inits variables needed to be initialised each time we reset at the start
        of an episode.
    '''
    def _init_env_variables(self):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Applies the given action to the simulation.
    '''
    def _set_action(self, action):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Indicates whether or not the episode is done ( the robot has fallen for example).
    '''
    def _is_done(self, observations):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Calculates the reward to give based on the observations given.
    '''
    def _compute_reward(self, observations, done):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
    '''
    def _env_setup(self, initial_qpos):
        raise NotImplementedError()

    '''
    DESCRIPTION: TODO...
    value.
    '''
    def update_initial_pose(self, initial_pose):
        self.initial_pose = initial_pose
        self.gazebo.update_initial_pose(self.initial_pose)


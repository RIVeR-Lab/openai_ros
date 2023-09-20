#!/usr/bin/env python3

'''
LAST UPDATE: 2023.09.04

AUTHOR:     OPENAI_ROS
            Neset Unver Akmandor (NUA)

E-MAIL: akmandor.n@northeastern.edu

DESCRIPTION: TODO...

REFERENCES:
[1] 

NUA TODO:
'''

import rospy
from std_srvs.srv import Empty, EmptyRequest
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SpawnModelRequest, SpawnModel, GetWorldProperties, GetWorldPropertiesRequest, GetPhysicsProperties, GetPhysicsPropertiesRequest, DeleteModel, DeleteModelRequest, SetPhysicsProperties, SetPhysicsPropertiesRequest, SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from controller_manager_msgs.srv import LoadController, LoadControllerRequest, ListControllers, ListControllersRequest, SwitchControllerRequest, SwitchController
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3, Pose
from mobiman_simulation.srv import resetMobiman, resetMobimanRequest
import subprocess
import os
import signal
from sensor_msgs.msg import JointState
import datetime
from tf.transformations import euler_from_quaternion
# import multiprocessing
import rospkg

'''
DESCRIPTION: TODO...
'''
class GazeboConnection():

    '''
    DESCRIPTION: TODO...
    '''
    def __init__(self, start_init_physics_parameters, reset_world_or_sim, max_retry = 20, robot_namespace='', initial_pose={}):
        
        print("[gazebo_connection::GazeboConnection::__init__] START")
        rospack = rospkg.RosPack()
        path = rospack.get_path('mobiman_simulation') + "/urdf/"
        self.jackal_jaco_urdf = path + "jackal_jaco.urdf"
        with open(self.jackal_jaco_urdf, 'r') as f:
            self.jackal_jacko_xml = f.read()
        self.joint_names = [f'j2n6s300_joint_{str(a)}' for a in range(1,7)]
        self._max_retry = max_retry
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_mobiman = rospy.ServiceProxy('/reset_mobiman', resetMobiman)
        #self.reset_robot = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #self.reset_robot_config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.robot_namespace = robot_namespace
        self.initial_pose = {}
        self.update_initial_pose(initial_pose)
        self.reset_done = False
        self.reset_counter = 0
        # Setup the Gravity Controle system
        service_name = '/gazebo/set_physics_properties'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))

        self.set_physics = rospy.ServiceProxy(service_name, SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()
        
        # We always pause the simulation, important for legged robots learning
        self.pauseSim()
        self.jackal_jaco_xml = None
        
        print("[gazebo_connection::GazeboConnection::__init__] END")

    '''
    DESCRIPTION: TODO...
    '''
    def pauseSim(self):

        rospy.logdebug("[gazebo_connection::GazeboConnection::pauseSim] START")
        #print("[gazebo_connection::GazeboConnection::pauseSim] START")
        
        paused_done = False
        counter = 0
        
        while not paused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    rospy.logdebug("[gazebo_connection::GazeboConnection::pauseSim] PAUSING service calling...")
                    self.pause()
                    paused_done = True
                    rospy.logdebug("[gazebo_connection::GazeboConnection::pauseSim] PAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("[gazebo_connection::GazeboConnection::pauseSim] ERROR: /gazebo/pause_physics service call failed")
            else:
                error_message = "[gazebo_connection::GazeboConnection::pauseSim] ERROR: Maximum retries done: " + str(self._max_retry) + ", please check Gazebo pause service"
                rospy.logerr(error_message)
                assert False, error_message

        rospy.logdebug("[gazebo_connection::GazeboConnection::pauseSim] END")
        #print("[gazebo_connection::GazeboConnection::pauseSim] END")

    '''
    DESCRIPTION: TODO...
    '''
    def unpauseSim(self):
        
        rospy.logdebug("[gazebo_connection::GazeboConnection::unpauseSim] START")
        unpaused_done = False
        counter = 0

        while not unpaused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    rospy.logdebug("[gazebo_connection::GazeboConnection::unpauseSim] UNPAUSING service calling...")
                    self.unpause()
                    unpaused_done = True
                    rospy.logdebug("[gazebo_connection::GazeboConnection::unpauseSim] UNPAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("[gazebo_connection::GazeboConnection::unpauseSim] ERROR: /gazebo/unpause_physics service call failed...Retrying " + str(counter))
            else:
                error_message = "[gazebo_connection::GazeboConnection::unpauseSim] ERROR: Maximum retries done" + str(self._max_retry) + ", please check Gazebo unpause service"
                rospy.logerr(error_message)
                assert False, error_message

        rospy.logdebug("[gazebo_connection::GazeboConnection::unpauseSim] END")

    '''
    DESCRIPTION: TODO...This was implemented because some simulations, when reseted the simulation
        the systems that work with TF break, and because sometime we wont be able to change them
        we need to reset world that ONLY resets the object position, not the entire simulation
        systems.
    '''
    def resetSim(self):

        if self.reset_world_or_sim == "SIMULATION":
            #rospy.logerr("[gazebo_connection::GazeboConnection::resetSim] SIMULATION")
            #print("[gazebo_connection::GazeboConnection::resetSim] SIMULATION")
            self.resetSimulation()

        elif self.reset_world_or_sim == "WORLD":
            #rospy.logerr("[gazebo_connection::GazeboConnection::resetSim] WORLD")
            #print("[gazebo_connection::GazeboConnection::resetSim] WORLD")
            self.resetWorld()
        
        elif self.reset_world_or_sim == "ROBOT":
            #rospy.logdebug("[gazebo_connection::GazeboConnection::resetSim] ROBOT")
            #print("[gazebo_connection::GazeboConnection::resetSim] ROBOT")
            self.reset_counter += 1
            #print("[gazebo_connection::GazeboConnection::resetSim] RESET COUNTER: ", self.reset_counter)
            self.resetRobot()
            # self.reset_world_or_sim == "ROBOT"
            rospy.sleep(1)
            # self.resetSim()
            # reset_thread = multiprocessing.Process(target=self.resetRobot)
            # reset_thread.daemon = True
            # reset_thread.start()
            # reset_thread.join(timeout=20)
            # if reset_thread.is_alive():
            #     get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
            #     phy_res = get_physics(GetPhysicsPropertiesRequest())
            #     get_model = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
            #     get_model_res = get_model(GetWorldPropertiesRequest())
            #     if phy_res.pause == False and 'mobiman' in get_model_res.model_names:
            #         print("BREAK")
            #     else:
            #         reset_thread.terminate()
            #         print("Terminating Thread")
            #         reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            #         delete_mobiman = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            #         rospy.wait_for_service('/gazebo/delete_model')
            #         rospy.wait_for_service('/gazebo/reset_world')
            #         try:
            #             delete_mobiman(DeleteModelRequest('mobiman'))
            #             reset_world(EmptyRequest())
            #             self.resetSim()
            #         except Exception as e:
            #             print("Exception: ", e)
            # self.reset_world_or_sim == "ROBOT"
            # print("[gazebo_connection::GazeboConnection::resetSim] Counter: ", self.reset_counter)
            # self.reset_counter 
            # self.resetSim()


        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logerr("[gazebo_connection::GazeboConnection::resetSim] NO_RESET_SIM")
            print("[gazebo_connection::GazeboConnection::resetSim] NO_RESET_SIM")
        
        else:
            rospy.logerr("[gazebo_connection::GazeboConnection::resetSim] ERROR: WRONG Reset Option:" + str(self.reset_world_or_sim))
            print("[gazebo_connection::GazeboConnection::resetSim] ERROR: WRONG Reset Option:" + str(self.reset_world_or_sim))

    '''
    DESCRIPTION: TODO...
    '''
    def resetSimulation(self):
        
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logdebug("[gazebo_connection::GazeboConnection::resetWorld] ERROR: /gazebo/reset_simulation service call failed!")


    '''
    DESCRIPTION: TODO...
    '''
    def resetWorld(self):
        
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logdebug("[gazebo_connection::GazeboConnection::resetWorld] ERROR: /gazebo/reset_world service call failed!")

    '''
    DESCRIPTION: TODO...
    '''
    def resetRobot(self):
        #print("[gazebo_connection::GazeboConnection::resetRobot] START")
        #robot_reset_request = SetModelStateRequest()
        #robot_reset_joint_request = SetModelConfigurationRequest()
        #robot_reset_link_request = SetLinkStateRequest()
        # reset_mobiman_request = resetMobimanRequest()
        
        if self.robot_namespace == "":
            rospy.logdebug("[gazebo_connection::GazeboConnection::resetRobot] ERROR: robot_namespace is not defined!")

        '''
        robot_reset_request.model_state.pose.position.x = self.initial_pose["x_init"]
        robot_reset_request.model_state.pose.position.y = self.initial_pose["y_init"]
        robot_reset_request.model_state.pose.position.z = self.initial_pose["z_init"]
        robot_reset_request.model_state.pose.orientation.x = self.initial_pose["x_rot_init"]
        robot_reset_request.model_state.pose.orientation.y = self.initial_pose["y_rot_init"]
        robot_reset_request.model_state.pose.orientation.z = self.initial_pose["z_rot_init"]
        robot_reset_request.model_state.pose.orientation.w = self.initial_pose["w_rot_init"]

        init_arm_joint_name_1 = 'j2n6s300_joint_1'
        init_arm_joint_name_2 = 'j2n6s300_joint_2'
        init_arm_joint_name_3 = 'j2n6s300_joint_3'
        init_arm_joint_name_4 = 'j2n6s300_joint_4'
        init_arm_joint_name_5 = 'j2n6s300_joint_5'
        init_arm_joint_name_6 = 'j2n6s300_joint_6'

        init_arm_joint_pos_1 = 0.0
        init_arm_joint_pos_2 = 2.9
        init_arm_joint_pos_3 = 1.3
        init_arm_joint_pos_4 = 4.2
        init_arm_joint_pos_5 = 1.4
        init_arm_joint_pos_6 = 0.0

        robot_reset_joint_request.urdf_param_name = "/robot_description"
        robot_reset_joint_request.joint_names = ['j2n6s300_jointsdasdas_2']
        robot_reset_joint_request.joint_positions = [2.9]
        '''
        
        '''
        robot_reset_joint_request.joint_names = [init_arm_joint_name_1, 
                                                 init_arm_joint_name_2, 
                                                 init_arm_joint_name_3, 
                                                 init_arm_joint_name_4, 
                                                 init_arm_joint_name_5, 
                                                 init_arm_joint_name_6]
        robot_reset_joint_request.joint_positions = [init_arm_joint_pos_1,
                                                     init_arm_joint_pos_2,
                                                     init_arm_joint_pos_3,
                                                     init_arm_joint_pos_4,
                                                     init_arm_joint_pos_5,
                                                     init_arm_joint_pos_6]
        '''
        robot_pose = Pose()
        robot_pose.position.x = self.initial_pose["x_init"]
        robot_pose.position.y = self.initial_pose["y_init"]
        robot_pose.position.z = self.initial_pose["z_init"]
        robot_pose.orientation.x = self.initial_pose["x_rot_init"]
        robot_pose.orientation.y = self.initial_pose["y_rot_init"]
        robot_pose.orientation.z = self.initial_pose["z_rot_init"]
        robot_pose.orientation.w = self.initial_pose["w_rot_init"]
        # reset_mobiman_request.x = self.initial_pose["x_init"]
        # reset_mobiman_request.y = self.initial_pose["y_init"]
        # reset_mobiman_request.z = self.initial_pose["z_init"]
        # reset_mobiman_request.quat_x = self.initial_pose["x_rot_init"]
        # reset_mobiman_request.quat_y = self.initial_pose["y_rot_init"]
        # reset_mobiman_request.quat_z = self.initial_pose["z_rot_init"]
        # reset_mobiman_request.quat_w = self.initial_pose["w_rot_init"]

        init_arm_joint_pos_1 = 0.0
        init_arm_joint_pos_2 = 2.9
        init_arm_joint_pos_3 = 1.3
        init_arm_joint_pos_4 = 4.2
        init_arm_joint_pos_5 = 1.4
        init_arm_joint_pos_6 = 0.0
        joint_positions = [init_arm_joint_pos_1, init_arm_joint_pos_2, init_arm_joint_pos_3, \
                            init_arm_joint_pos_4, init_arm_joint_pos_5, init_arm_joint_pos_6]
        # reset_mobiman_request.joint_1 = init_arm_joint_pos_1
        # reset_mobiman_request.joint_2 = init_arm_joint_pos_2
        # reset_mobiman_request.joint_3 = init_arm_joint_pos_3
        # reset_mobiman_request.joint_4 = init_arm_joint_pos_4
        # reset_mobiman_request.joint_5 = init_arm_joint_pos_5
        # reset_mobiman_request.joint_6 = init_arm_joint_pos_6

        #rospy.wait_for_service('/gazebo/set_model_state')
        #rospy.wait_for_service('/gazebo/set_model_configuration')
        # rospy.wait_for_service('/reset_mobiman')
        
        # try:
        pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        controller_list = ['arm_controller', 'joint_state_controller', 'jackal_velocity_controller']
        ### Pause Physics
        # '''
        #print("[gazebo_connection::GazeboConnection::resetRobot] Pause Physics")
        try:
            rospy.wait_for_service('/gazebo/pause_physics')
            pause_physics_client(EmptyRequest())
        except Exception as e:
            pass
        # '''
        ### Delete Model
        cdel = 0
        #print("[gazebo_connection::GazeboConnection::resetRobot] Delete Model")
        while True:
            try:
                delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
                rospy.wait_for_service('/gazebo/delete_model')
                delete_model(DeleteModelRequest('mobiman'))
            except Exception as e:
                pass
            rospy.wait_for_service('/gazebo/get_world_properties')
            get_model = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
            get_model_res = get_model(GetWorldPropertiesRequest())
            if 'mobiman' not in get_model_res.model_names:
                break
        ### Reset Simulation
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        try:
            rospy.wait_for_service('/gazebo/reset_world')
            reset_world(EmptyRequest())
        except Exception as e:
            print("[gazebo_connection::GazeboConnection::resetRobot] Error Resetting World")
        '''
        Subprocess for Spawn Model
        '''
        roll_,pitch_, yaw_ = euler_from_quaternion([self.initial_pose["x_rot_init"], self.initial_pose["y_rot_init"], self.initial_pose["z_rot_init"], self.initial_pose["w_rot_init"]])
        # print(roll_, pitch_, yaw_, '************')
        command = f'rosrun gazebo_ros spawn_model -urdf -file {self.jackal_jaco_urdf} -model mobiman -x {self.initial_pose["x_init"]} -y {self.initial_pose["y_init"]} -z {self.initial_pose["z_init"]} -R {roll_} -P {pitch_} -Y {yaw_} ' + \
                    f'-J {self.joint_names[0]} {init_arm_joint_pos_1} ' + \
                    f'-J {self.joint_names[1]} {init_arm_joint_pos_2} ' + \
                    f'-J {self.joint_names[2]} {init_arm_joint_pos_3} ' + \
                    f'-J {self.joint_names[3]} {init_arm_joint_pos_4} ' + \
                    f'-J {self.joint_names[4]} {init_arm_joint_pos_5} ' + \
                    f'-J {self.joint_names[5]} {init_arm_joint_pos_6} ' + '-reference_frame world'
        
        subprocess.call(command, shell=True, stderr=subprocess.STDOUT)
        #print(subprocess.STDOUT)
        '''
        ### Spawn Model
        cdel = 0
        print("[gazebo_connection::GazeboConnection::resetRobot] Spawning Model")
        while True:
            if cdel > 100:
                break
            cdel += 1
            try:
                rospy.wait_for_service('/gazebo/spawn_urdf_model')
                spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
                spawn_model(SpawnModelRequest(model_name='mobiman', model_xml=self.jackal_jacko_xml, robot_namespace='', initial_pose=robot_pose, reference_frame='world'))
            except Exception as e:
                pass
                # while res.status_message != True:
                #     res = spawn_model(SpawnModelRequest(model_name='mobiman', model_xml=self.jackal_jacko_xml, robot_namespace='', initial_pose=robot_pose, reference_frame='world'))
            rospy.wait_for_service('/gazebo/get_world_properties')
            get_model = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
            get_model_res = get_model(GetWorldPropertiesRequest())
            if 'mobiman' in get_model_res.model_names:
                break
            
        ### Set Config
        
        print("[gazebo_connection::GazeboConnection::resetRobot] Setting Configuration")
        try:
            rospy.wait_for_service('/gazebo/set_model_configuration')
            set_configuration = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
            for i in range(1, 40):
                rospy.wait_for_service('/gazebo/set_model_configuration')
                set_configuration(SetModelConfigurationRequest(model_name='mobiman', urdf_param_name='robot_description', joint_names=self.joint_names, joint_positions=joint_positions))
        except Exception as e:
            pass
        ### LOAD Controller
        '''
        #print("[gazebo_connection::GazeboConnection::resetRobot] Loading Controller")
        try:
            load_controller = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
            for controller in controller_list:
                rospy.wait_for_service('/controller_manager/load_controller')
                load_controller(LoadControllerRequest(controller))
        except Exception as e:
            pass
        ### Switch Controller
        #print("[gazebo_connection::GazeboConnection::resetRobot] Switching Controller")
        command = 'sleep 0.5 && rosservice call /gazebo/unpause_physics "{}"'
        unpause_proc = None
        try:
            switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            switch_controller_req = SwitchControllerRequest()
            switch_controller_req.start_asap = True
            switch_controller_req.strictness = switch_controller_req.STRICT
            for idx, controller in enumerate(controller_list):
                switch_controller_req.start_controllers = [controller]
                # if idx == 0:
                unpause_proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
                rospy.wait_for_service('/controller_manager/switch_controller')
                res = switch_controller(switch_controller_req)
        except Exception as e:
            pass
        '''
            # rospy.wait_for_service('/gazebo/spawn_urdf_model')
            #self.reset_robot(robot_reset_request)
            #res = self.reset_robot_config(robot_reset_joint_request)
            # suc = self.reset_mobiman(reset_mobiman_request)
            # print("[gazebo_connection::GazeboConnection::resetRobot] suc: " + str(suc))
            # self.reset_done = suc.success
        # except rospy.ServiceException as e:
        #     rospy.logdebug("[gazebo_connection::GazeboConnection::resetRobot] ERROR: /gazebo/set_model_state service call failed!")
        '''
        # try:
        # except Exception as e:
            # print("Error terminating process spawn")
        try:
            os.killpg(os.getpgid(unpause_proc.pid), signal.SIGTERM) # type: ignore
        except Exception as e:
            print("Error terminating process unpause")
        msg = None
        try:
            msg = rospy.wait_for_message('/joint_states', JointState, timeout=rospy.Duration(2.0))
        except Exception as e:
            ### So that if we get exception from ros for not finding the joint state message
            ### the program won't crash.
            pass
        if msg == None:
            self.reset_robot() # type: ignore
        
        # rospy.sleep(1)
        #print("[gazebo_connection::GazeboConnection::resetRobot] DEBUG INF")
        #while 1:
        #    continue

    '''
    DESCRIPTION: TODO...
    '''
    def init_values(self):

        self.resetSim()

        if self.start_init_physics_parameters:
            rospy.logdebug("[gazebo_connection::GazeboConnection::init_values] Initialising Simulation Physics Parameters")
            self.init_physics_parameters()
        else:
            rospy.logdebug("[gazebo_connection::GazeboConnection::init_values] ERROR: Not Initialising Simulation Physics Parameters!")

    '''
    DESCRIPTION: TODO...We initialise the physics parameters of the simulation, like gravity,
        friction coeficients and so on.
    '''
    def init_physics_parameters(self):

        self._time_step = Float64(0.001)
        self._max_update_rate = Float64(1000.0)

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()

    '''
    DESCRIPTION: TODO...
    '''
    def update_gravity_call(self):

        self.pauseSim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rospy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rospy.logdebug("[gazebo_connection::GazeboConnection::update_gravity_call] Gravity Update Result==" + str(result.success) + ",message==" + str(result.status_message))

        self.unpauseSim()

    '''
    DESCRIPTION: TODO...
    '''
    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self.update_gravity_call()

    '''
    DESCRIPTION: TODO...
    value.
    '''
    def update_initial_pose(self, initial_pose):
        self.initial_pose["x_init"] = initial_pose["x"]
        self.initial_pose["y_init"] = initial_pose["y"]
        self.initial_pose["z_init"] = initial_pose["z"]
        self.initial_pose["x_rot_init"] = initial_pose["qx"]
        self.initial_pose["y_rot_init"] = initial_pose["qy"]
        self.initial_pose["z_rot_init"] = initial_pose["qz"]
        self.initial_pose["w_rot_init"] = initial_pose["qw"]
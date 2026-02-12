#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Bool
from sensor_msgs.msg import JointState
try:
    from iiwa_msgs.msg import JointTorque
    HAS_IIWA_MSGS = True
except ImportError:
    HAS_IIWA_MSGS = False

import numpy as np
import socket
import time
import threading
from collections import deque
import json
import argparse
import os
import csv
from datetime import datetime
from scipy.interpolate import interp1d
from .trajectory_deformer import TrajectoryDeformer
from .promp import ProMP
from .stepwise_em_learner import StepwiseEMLearner

class StandaloneDeformationController(Node):
    def __init__(self):
        super().__init__('standalone_deformation_controller')
        
        # Parameters
        self.declare_parameter('kuka_ip', '172.31.1.147')
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('joint_torque_threshold_small', 0.5)
        self.declare_parameter('joint_torque_threshold_big', 3.0)
        self.declare_parameter('deformation_alpha', 0.1)
        self.declare_parameter('deformation_waypoints', 10)
        self.declare_parameter('promp_conditioning_sigma', 0.01)
        self.declare_parameter('trajectory_file', os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy'))
        self.declare_parameter('promp_file', 'promp_model.npy')
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('save_directory', '~/robot_demos')
        self.declare_parameter('demo_file', '')
        self.declare_parameter('trajectory_points', 100)
        self.declare_parameter('train_on_startup', True)
        self.declare_parameter('execute_after_training', True)
        # Note: Using TCP socket for joint torques (Java sends JOINT_TORQUE: messages)
        # ROS2 topics are not used - set use_ros2_joint_torques=False
        self.declare_parameter('external_joint_torque_topic', '/iiwa/jointExternalTorque')
        self.declare_parameter('use_ros2_joint_torques', False)  # Always False - use TCP socket
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.joint_torque_threshold_small = self.get_parameter('joint_torque_threshold_small').value
        self.joint_torque_threshold_big = self.get_parameter('joint_torque_threshold_big').value
        self.deformation_alpha = self.get_parameter('deformation_alpha').value
        self.deformation_waypoints = self.get_parameter('deformation_waypoints').value
        self.promp_conditioning_sigma = self.get_parameter('promp_conditioning_sigma').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.promp_file = self.get_parameter('promp_file').value
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        demo_file_param = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        self.train_on_startup = self.get_parameter('train_on_startup').value
        self.execute_after_training = self.get_parameter('execute_after_training').value
        self.external_joint_torque_topic = self.get_parameter('external_joint_torque_topic').value
        self.use_ros2_joint_torques = self.get_parameter('use_ros2_joint_torques').value
        
        # Set demo_file path
        if demo_file_param:
            if os.path.isabs(demo_file_param):
                self.demo_file = demo_file_param
            else:
                self.demo_file = demo_file_param if os.path.exists(demo_file_param) else os.path.join(self.save_directory, demo_file_param)
        else:
            self.demo_file = os.path.join(self.save_directory, 'all_demos.npy')
        
        # Components
        self.deformer = TrajectoryDeformer(alpha=self.deformation_alpha, n_waypoints=self.deformation_waypoints, energy_threshold=0.5)
        self.promp = None
        self.current_trajectory = None
        self.demos = []
        self.stepwise_em_learner = None
        
        # Normalization statistics
        self.demo_min = None
        self.demo_max = None
        self.demo_mean = None
        self.demo_std = None
        
        # State
        self.is_executing = False
        self.torque_data = deque(maxlen=1000)
        self.joint_torque_data = deque(maxlen=1000)
        
        # Execution tracking
        self.current_point_count = 0
        self.point_count_lock = threading.Lock()
        self.execution_interrupted = threading.Event()
        self.execution_interrupted.set()  # Start as set (not interrupted)
        
        # CSV logging
        self.execution_trajectory_log = []
        self.joint_torque_log = []
        self.external_torque_log = []
        self.result_directory = os.path.join(os.path.expanduser('~/result'), 'standalone_deformation_controller')
        
        # Thread lock for pybullet
        self.pybullet_lock = threading.Lock()
        
        # TCP communication
        self.kuka_socket = None
        self.torque_socket = None
        
        # Statistics
        self.deformation_count = 0
        self.conditioning_count = 0
        
        # Setup
        self.setup_communication()
        self.setup_ros_communication()
        
        # Train or load
        if self.train_on_startup:
            if self.train_and_generate_trajectory():
                self.get_logger().info('ProMP training completed')
            else:
                self.load_trajectory_and_promp()
        else:
            self.load_trajectory_and_promp()
        
        # Auto-execute
        if self.execute_after_training and self.current_trajectory is not None:
            threading.Timer(1.0, self.start_execution).start()
        
        self.get_logger().info('Standalone Deformation Controller initialized')
    
    def setup_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}:{self.kuka_port}...")
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.settimeout(5)
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            ready = self._receive_complete_message(self.kuka_socket, timeout=5.0)
            if ready and ready.strip() == "READY":
                self.get_logger().info('KUKA connection established')
            else:
                self.get_logger().error(f'Unexpected response: {ready}')
                self.kuka_socket = None
            
            # Setup torque data server
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', 30003))
            self.torque_socket.listen(1)
            
            self.torque_thread = threading.Thread(target=self.receive_torque_data)
            self.torque_thread.daemon = True
            self.torque_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup communication: {e}')
            self.kuka_socket = None
    
    def _receive_complete_message(self, sock, timeout=5.0, buffer_size=8192):
        """Receive complete message from socket - robust version matching train_and_execute.py"""
        try:
            original_timeout = sock.gettimeout()
            sock.settimeout(timeout)
            message_parts = []
            
            while True:
                try:
                    data = sock.recv(buffer_size)
                    if not data:
                        break
                    message_parts.append(data.decode('utf-8'))
                    if b'\n' in data:
                        break
                except socket.timeout:
                    # Timeout is expected - return what we have so far
                    break
                except OSError as e:
                    # Handle EAGAIN/EWOULDBLOCK (Resource temporarily unavailable)
                    if e.errno == 11:  # EAGAIN/EWOULDBLOCK
                        # This can happen with non-blocking sockets - just continue waiting
                        time.sleep(0.001)  # Small sleep to avoid busy wait
                        continue
                    else:
                        raise
            
            # Restore original timeout
            sock.settimeout(original_timeout)
            
            result = ''.join(message_parts)
            return result if result else None
            
        except socket.timeout:
            return None
        except Exception as e:
            # Don't log EAGAIN errors as errors - they're expected in some cases
            if not (isinstance(e, OSError) and e.errno == 11):
                self.get_logger().debug(f"Error receiving message: {e}")
            return None
    
    def setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers"""
        self.deformation_status_pub = self.create_publisher(String, 'deformation_status', 10)
        self.conditioning_status_pub = self.create_publisher(String, 'conditioning_status', 10)
        self.execution_status_pub = self.create_publisher(String, 'execution_status', 10)
        
        self.start_execution_sub = self.create_subscription(Bool, 'start_deformation_execution', self.start_execution_callback, 10)
        self.stop_execution_sub = self.create_subscription(Bool, 'stop_deformation_execution', self.stop_execution_callback, 10)
        
        # Note: Not using ROS2 topics for joint torques - using TCP socket instead
        # Joint torques are received via TCP socket from Java application (JOINT_TORQUE: messages)
        if self.use_ros2_joint_torques:
            self.get_logger().warn('use_ros2_joint_torques=True but ROS2 topics not used - using TCP socket instead')
            if HAS_IIWA_MSGS:
                self.external_joint_torque_sub = self.create_subscription(JointTorque, self.external_joint_torque_topic, self.external_joint_torque_callback, 10)
            else:
                self.external_joint_torque_sub = self.create_subscription(JointState, self.external_joint_torque_topic, self.external_joint_torque_callback_jointstate, 10)
        else:
            self.get_logger().info('Using TCP socket for joint torques (Java sends JOINT_TORQUE: messages on port 30003)')
    
    def cartesian_to_joint_python(self, cartesian_poses):
        """Convert Cartesian poses to joint positions using pybullet IK"""
        with self.pybullet_lock:
            try:
                import pybullet as p
                import pybullet_data
                
                # Check if already connected
                try:
                    p.getConnectionInfo()
                    p.disconnect()
                except:
                    pass
                
                physics_client = p.connect(p.DIRECT)
                if physics_client < 0:
                    self.get_logger().error('Failed to connect to pybullet')
                    return None
                
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                
                # Try to load KUKA URDF
                robot_id = None
                urdf_paths = [
                    "kuka_iiwa/model.urdf",
                    "kuka_lbr_iiwa_14_r820.urdf",
                    "/opt/ros/noetic/share/kuka_description/urdf/kuka_lbr_iiwa_14_r820.urdf",
                ]
                
                for urdf_path in urdf_paths:
                    try:
                        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
                        break
                    except:
                        continue
                
                if robot_id is None:
                    self.get_logger().warn('KUKA URDF not found, creating simplified model')
                    robot_id = self._create_simple_kuka_model(p)
                
                if robot_id is None:
                    p.disconnect()
                    return None
                
                num_joints = p.getNumJoints(robot_id)
                end_effector_link = num_joints - 1
                
                joint_positions = []
                initial_joints = [0.0, 0.7854, 0.0, -1.3962, 0.0, -0.6109, 0.0]
                current_joints = initial_joints.copy()
                
                for i, pose in enumerate(cartesian_poses):
                    x, y, z, alpha, beta, gamma = pose
                    target_pos = [x, y, z]
                    target_orn = p.getQuaternionFromEuler([alpha, beta, gamma])
                    
                    try:
                        joint_angles = p.calculateInverseKinematics(
                            robot_id, end_effector_link, target_pos, target_orn,
                            lowerLimits=[-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
                            upperLimits=[2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054],
                            jointRanges=[5.934, 4.188, 5.934, 4.188, 5.934, 4.188, 6.108],
                            restPoses=current_joints,
                            maxNumIterations=200,
                            residualThreshold=1e-5
                        )
                        
                        if joint_angles is not None and len(joint_angles) >= 7:
                            joint_angles_7 = list(joint_angles[:7])
                            limits = [(-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967),
                                     (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094), (-3.054, 3.054)]
                            valid = all(min_val <= angle <= max_val for angle, (min_val, max_val) in zip(joint_angles_7, limits))
                            
                            if valid:
                                joint_positions.append(joint_angles_7)
                                current_joints = joint_angles_7.copy()
                            else:
                                joint_positions.append(joint_positions[-1] if joint_positions else initial_joints)
                        else:
                            joint_positions.append(joint_positions[-1] if joint_positions else initial_joints)
                    except Exception as e:
                        joint_positions.append(joint_positions[-1] if joint_positions else initial_joints)
                
                p.disconnect()
                return np.array(joint_positions)
                
            except ImportError:
                self.get_logger().error('pybullet not installed')
                return None
            except Exception as e:
                self.get_logger().error(f'pybullet IK failed: {e}')
                try:
                    import pybullet as p
                    p.disconnect()
                except:
                    pass
                return None
    
    def _create_simple_kuka_model(self, p):
        """Create a simple 7-DOF KUKA model"""
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=base_visual, baseCollisionShapeIndex=base_collision)
    
    def load_demos(self):
        """Load demonstrations from file"""
        try:
            if not os.path.exists(self.demo_file):
                self.get_logger().error(f'Demo file not found: {self.demo_file}')
                return False
            
            loaded_data = np.load(self.demo_file, allow_pickle=True)
            demos_list = []
            
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                for demo in loaded_data:
                    arr = np.array(demo)
                    if arr.ndim == 2 and arr.shape[1] == 6:
                        demos_list.append(arr)
            elif isinstance(loaded_data, np.ndarray):
                if loaded_data.ndim == 3:
                    for i in range(loaded_data.shape[0]):
                        demos_list.append(loaded_data[i])
                elif loaded_data.ndim == 2 and loaded_data.shape[1] == 6:
                    demos_list.append(loaded_data)
            
            self.demos = demos_list
            self.get_logger().info(f'Loaded {len(self.demos)} demonstrations')
            return len(self.demos) > 0
            
        except Exception as e:
            self.get_logger().error(f'Error loading demonstrations: {e}')
            return False
    
    def normalize_demos(self):
        """Normalize demonstrations to same length and compute statistics"""
        if len(self.demos) == 0:
            return []
        
        target_length = self.trajectory_points
        normalized = []
        all_values = []
        
        for demo in self.demos:
            demo_array = np.array(demo)
            t_old = np.linspace(0, 1, len(demo_array))
            t_new = np.linspace(0, 1, target_length)
            
            normalized_demo = []
            for dim in range(6):
                try:
                    interp_func = interp1d(t_old, demo_array[:, dim], kind='cubic', fill_value='extrapolate')
                    normalized_demo.append(interp_func(t_new))
                except ValueError:
                    interp_func = interp1d(t_old, demo_array[:, dim], kind='linear', fill_value='extrapolate')
                    normalized_demo.append(interp_func(t_new))
            
            normalized_demo_array = np.column_stack(normalized_demo)
            normalized.append(normalized_demo_array)
            all_values.append(normalized_demo_array)
        
        all_values = np.concatenate(all_values, axis=0)
        self.demo_min = np.min(all_values, axis=0)
        self.demo_max = np.max(all_values, axis=0)
        self.demo_mean = np.mean(all_values, axis=0)
        self.demo_std = np.std(all_values, axis=0)
        self.demo_std = np.where(self.demo_std < 1e-10, 1.0, self.demo_std)
        
        normalized_scaled = []
        for demo in normalized:
            demo_normalized = (demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            normalized_scaled.append(demo_normalized)
        
        return normalized_scaled
    
    def train_promp(self):
        """Train ProMP on loaded demonstrations"""
        if len(self.demos) < 1:
            self.get_logger().error('No demonstrations available')
            return False
        
        try:
            normalized_demos = self.normalize_demos()
            
            self.promp = ProMP(num_basis=self.num_basis, sigma_noise=self.sigma_noise)
            self.promp.train(normalized_demos)
            
            self.stepwise_em_learner = StepwiseEMLearner(num_basis=self.num_basis, sigma_noise=self.sigma_noise, delta_N=0.1)
            
            if len(normalized_demos) > 0:
                self.stepwise_em_learner.initialize_from_first_demo(normalized_demos[0])
                for i in range(1, len(normalized_demos)):
                    self.stepwise_em_learner.stepwise_em_update(normalized_demos[i])
            
            if self.stepwise_em_learner.mean_weights is not None:
                self.promp.mean_weights = self.stepwise_em_learner.mean_weights.copy()
                self.promp.cov_weights = self.stepwise_em_learner.cov_weights.copy()
            
            self.get_logger().info('ProMP training completed')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error training ProMP: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def generate_trajectory(self):
        """Generate trajectory from trained ProMP"""
        if self.promp is None:
            self.get_logger().error('ProMP not trained')
            return None
        
        try:
            trajectory_normalized = self.promp.generate_trajectory(num_points=self.trajectory_points)
            
            if self.demo_min is not None and self.demo_max is not None:
                trajectory_denorm = trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                trajectory_denorm = np.clip(trajectory_denorm, self.demo_min, self.demo_max)
                self.current_trajectory = trajectory_denorm
            else:
                self.current_trajectory = trajectory_normalized
            
            return self.current_trajectory
            
        except Exception as e:
            self.get_logger().error(f'Error generating trajectory: {e}')
            return None
    
    def train_and_generate_trajectory(self):
        """Train ProMP and generate trajectory"""
        if not self.load_demos():
            return False
        if not self.train_promp():
            return False
        trajectory = self.generate_trajectory()
        if trajectory is None:
            return False
        self.deformer.set_trajectory(trajectory)
        return True
    
    def load_trajectory_and_promp(self):
        """Load trajectory and ProMP from files"""
        try:
            trajectory_paths = [
                self.trajectory_file,
                os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy'),
                os.path.join(os.path.expanduser('~/newfoldername'), 'learned_trajectory.npy'),
                'learned_trajectory.npy',
            ]
            
            for traj_path in trajectory_paths:
                if os.path.exists(traj_path):
                    try:
                        self.current_trajectory = np.load(traj_path)
                        self.deformer.set_trajectory(self.current_trajectory)
                        self.get_logger().info(f'Loaded trajectory from {traj_path}')
                        break
                    except Exception as e:
                        continue
            
            if self.current_trajectory is None:
                self.get_logger().error('Trajectory file not found')
                return
            
            promp_paths = [self.promp_file, os.path.join(os.path.expanduser('~/robotexecute'), 'promp_model.npy'), 'promp_model.npy']
            for promp_path in promp_paths:
                if os.path.exists(promp_path):
                    try:
                        promp_data = np.load(promp_path, allow_pickle=True).item()
                        self.promp = ProMP()
                        self.promp.mean_weights = promp_data['mean_weights']
                        self.promp.cov_weights = promp_data['cov_weights']
                        self.promp.basis_centers = promp_data['basis_centers']
                        self.promp.basis_width = promp_data['basis_width']
                        self.get_logger().info(f'Loaded ProMP from {promp_path}')
                        break
                    except Exception as e:
                        continue
                        
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory/ProMP: {e}')
    
    def start_execution_callback(self, msg):
        if msg.data:
            self.start_execution()
        else:
            self.stop_execution()
    
    def stop_execution_callback(self, msg):
        if msg.data:
            self.stop_execution()
    
    def external_joint_torque_callback(self, msg):
        try:
            if hasattr(msg, 'torque') and len(msg.torque) >= 7:
                joint_torques = list(msg.torque[:7])
            elif hasattr(msg, 'data') and len(msg.data) >= 7:
                joint_torques = list(msg.data[:7])
            else:
                return
            
            timestamp = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            self.joint_torque_data.append({'timestamp': timestamp, 'joint_torques': joint_torques})
        except Exception as e:
            self.get_logger().error(f'Error processing joint torque message: {e}')
    
    def external_joint_torque_callback_jointstate(self, msg):
        try:
            if not hasattr(msg, 'effort') or len(msg.effort) < 7:
                return
            
            joint_torques = list(msg.effort[:7])
            timestamp = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            self.joint_torque_data.append({'timestamp': timestamp, 'joint_torques': joint_torques})
        except Exception as e:
            self.get_logger().error(f'Error processing JointState message: {e}')
    
    def receive_torque_data(self):
        """Receive torque data from KUKA robot via TCP socket"""
        try:
            conn, addr = self.torque_socket.accept()
            self.get_logger().info(f'Torque data connection from {addr}')
            
            while True:
                data = conn.recv(2048)
                if not data:
                    break
                    
                lines = data.decode('utf-8').split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            parts = line.strip().split(',')
                            
                            # Parse JOINT_TORQUE messages from Java application (primary method - no ROS2 topics)
                            if line.startswith('JOINT_TORQUE:'):
                                joint_data = line.replace('JOINT_TORQUE:', '').strip()
                                values = [float(x) for x in joint_data.split(',')]
                                if len(values) >= 8:
                                    timestamp = values[0]
                                    joint_torques = values[1:8]
                                    self.joint_torque_data.append({'timestamp': timestamp, 'joint_torques': joint_torques})
                            elif len(parts) >= 7:
                                timestamp, fx, fy, fz, tx, ty, tz = map(float, parts[:7])
                                self.torque_data.append({
                                    'timestamp': timestamp,
                                    'force': [fx, fy, fz],
                                    'torque': [tx, ty, tz]
                                })
                        except ValueError:
                            continue
                            
        except Exception as e:
            self.get_logger().error(f'Error receiving torque data: {e}')
    
    def start_execution(self):
        """Start execution of the trajectory"""
        if self.is_executing:
            self.get_logger().warn('Execution already in progress')
            return
        if self.current_trajectory is None:
            self.get_logger().error('No trajectory loaded')
            return
        
        self.is_executing = True
        self.execution_thread = threading.Thread(target=self.execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        self.get_logger().info('Started trajectory execution')
    
    def execution_loop(self):
        """Main execution loop - execute trajectory and monitor torque"""
        trajectory = np.copy(self.current_trajectory)
        
        # Initialize CSV logging
        self.execution_trajectory_log = []
        self.joint_torque_log = []
        self.external_torque_log = []
        os.makedirs(self.result_directory, exist_ok=True)
        
        # Reset execution tracking
        with self.point_count_lock:
            self.current_point_count = 0
        self.execution_interrupted.set()  # Not interrupted initially
        
        self.get_logger().info(f'Starting trajectory execution ({len(trajectory)} points)...')
        self.execution_status_pub.publish(String(data='EXECUTION_STARTED'))
        
        # Start torque monitoring thread
        monitoring_thread = threading.Thread(target=self.monitor_torques_during_execution, args=(trajectory,))
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Execute trajectory using proven train_and_execute.py logic
        success = self.send_trajectory_to_kuka(trajectory)
        
        # Wait for monitoring thread to finish
        monitoring_thread.join(timeout=1.0)
        
        if not success:
            self.get_logger().error('Trajectory execution failed')
            self.is_executing = False
            return
        
        # Wait a bit for robot to finish moving
        time.sleep(10.0)
        
        self.is_executing = False
        self.get_logger().info('Trajectory execution finished')
        self.execution_status_pub.publish(String(data='EXECUTION_STOPPED'))
        
        # Save CSV files
        self.save_execution_data_to_csv()
    
    def monitor_torques_during_execution(self, trajectory):
        """Monitor joint torques during execution and trigger deformation/conditioning"""
        last_conditioning_time = 0
        conditioning_cooldown = 0.3
        
        while self.is_executing and self.execution_interrupted.is_set():
            if len(self.joint_torque_data) > 0:
                latest_joint_torques = self.joint_torque_data[-1]
                joint_torques = latest_joint_torques['joint_torques']
                max_joint_torque = max(abs(t) for t in joint_torques)
                
                current_time = time.time()
                
                # Check thresholds
                if (max_joint_torque > self.joint_torque_threshold_small and 
                    max_joint_torque < self.joint_torque_threshold_big and
                    (current_time - last_conditioning_time) > conditioning_cooldown):
                    
                    # Small to medium torque: Trigger ProMP conditioning
                    with self.point_count_lock:
                        current_idx = self.current_point_count
                    
                    t_current = current_idx / max(len(trajectory), 1)
                    t_current = np.clip(t_current, 0.0, 1.0)
                    
                    current_pose = self.get_current_robot_pose()
                    if current_pose is not None:
                        self.get_logger().info(f'ProMP conditioning triggered at t={t_current:.3f}, torque={max_joint_torque:.3f} Nm')
                        self.trigger_promp_conditioning_at_time(t_current, current_pose, trajectory)
                        last_conditioning_time = current_time
                
                elif max_joint_torque >= self.joint_torque_threshold_big and (current_time - last_conditioning_time) > conditioning_cooldown:
                    # Large torque: Trigger trajectory deformation and incremental learning
                    with self.point_count_lock:
                        current_idx = self.current_point_count
                    
                    t_current = current_idx / max(len(trajectory), 1)
                    t_current = np.clip(t_current, 0.0, 1.0)
                    
                    # Get human input for deformation
                    if len(self.torque_data) > 0:
                        latest_torque = self.torque_data[-1]
                        human_input = np.array(latest_torque['force'] + latest_torque['torque'])
                    else:
                        human_input = np.array([max_joint_torque, max_joint_torque, max_joint_torque, 0.0, 0.0, 0.0])
                    
                    self.get_logger().info(f'Trajectory deformation triggered at t={t_current:.3f}, torque={max_joint_torque:.3f} Nm')
                    self.handle_trajectory_deformation_and_incremental_learning(trajectory, human_input, current_idx, t_current)
                    last_conditioning_time = current_time
            
            time.sleep(0.01)  # 100 Hz monitoring
    
    def get_current_robot_pose(self):
        """Get current robot pose from KUKA"""
        if self.kuka_socket is None:
            return None
        
        try:
            self.kuka_socket.sendall(b"GET_POSE\n")
            self.kuka_socket.settimeout(1.0)
            pose_response = self._receive_complete_message(self.kuka_socket, timeout=1.0)
            self.kuka_socket.settimeout(None)
            
            if pose_response and pose_response.startswith("POSE:"):
                pose_str = pose_response.split("POSE:")[1].strip()
                current_pose = np.array([float(x) for x in pose_str.split(",")])
                return current_pose
            else:
                return None
        except Exception as e:
            self.get_logger().warn(f'Could not get current pose: {e}')
            return None
    
    def trigger_promp_conditioning_at_time(self, t_current, current_pose, current_trajectory):
        """Trigger ProMP conditioning at current time point"""
        if self.promp is None:
            self.get_logger().error('ProMP not available for conditioning')
            return
        
        try:
            # Normalize current pose
            if self.demo_min is not None and self.demo_max is not None:
                current_pose_normalized = (current_pose - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            else:
                current_pose_normalized = current_pose
            
            # Condition ProMP
            self.promp.condition_on_waypoint(t_current, current_pose_normalized, self.promp_conditioning_sigma)
            
            # Generate new trajectory
            num_points = len(current_trajectory)
            new_trajectory_normalized = self.promp.generate_trajectory(num_points=num_points)
            
            # Denormalize
            if self.demo_min is not None and self.demo_max is not None:
                new_trajectory = new_trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                new_trajectory = np.clip(new_trajectory, self.demo_min, self.demo_max)
            else:
                new_trajectory = new_trajectory_normalized
            
            # Merge: keep executed part, replace future part
            with self.point_count_lock:
                current_idx = self.current_point_count
            
            new_trajectory_start_idx = int(t_current * num_points)
            new_trajectory_start_idx = np.clip(new_trajectory_start_idx, 0, num_points - 1)
            
            merged_trajectory = np.vstack((
                current_trajectory[:current_idx],
                new_trajectory[new_trajectory_start_idx:]
            ))
            
            # Interrupt current execution
            self.execution_interrupted.clear()
            try:
                self.kuka_socket.sendall(b"STOP\n")
                self.kuka_socket.settimeout(2.0)
                response = self._receive_complete_message(self.kuka_socket, timeout=2.0)
                self.kuka_socket.settimeout(None)
            except Exception as e:
                self.get_logger().error(f'Error sending STOP: {e}')
            
            # Restart execution from current point
            trajectory_from_current = merged_trajectory[current_idx:]
            self.current_trajectory = merged_trajectory
            self.deformer.set_trajectory(merged_trajectory)
            
            # Restart execution
            self.execution_interrupted.set()
            threading.Thread(target=self.send_trajectory_to_kuka, args=(trajectory_from_current,), daemon=True).start()
            
            self.conditioning_count += 1
            self.get_logger().info(f'ProMP conditioning completed, restarting from point {current_idx}')
            
        except Exception as e:
            self.get_logger().error(f'Error during ProMP conditioning: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def handle_trajectory_deformation_and_incremental_learning(self, current_trajectory, human_input, current_idx, t_current):
        """Handle trajectory deformation and incremental learning"""
        try:
            # Deform trajectory
            self.deformer.set_trajectory(current_trajectory)
            deformed_traj, _, deformation_energy = self.deformer.deform(human_input)
            
            if deformed_traj is None:
                self.get_logger().error('Trajectory deformation failed')
                return
            
            # Normalize deformed trajectory
            deformed_demo = np.array(deformed_traj)
            if self.demo_min is not None and self.demo_max is not None:
                deformed_demo_normalized = (deformed_demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                if len(deformed_demo_normalized) != self.trajectory_points:
                    t_old = np.linspace(0, 1, len(deformed_demo_normalized))
                    t_new = np.linspace(0, 1, self.trajectory_points)
                    deformed_demo_normalized_resampled = []
                    for dim in range(6):
                        interp_func = interp1d(t_old, deformed_demo_normalized[:, dim], kind='cubic', fill_value='extrapolate')
                        deformed_demo_normalized_resampled.append(interp_func(t_new))
                    deformed_demo_normalized = np.column_stack(deformed_demo_normalized_resampled)
            else:
                deformed_demo_normalized = deformed_demo
            
            # Add as new demo
            self.demos.append(deformed_demo)
            
            # Initialize StepwiseEMLearner if needed
            if self.stepwise_em_learner is None:
                self.stepwise_em_learner = StepwiseEMLearner(num_basis=self.num_basis, sigma_noise=self.sigma_noise, delta_N=0.1)
                if len(self.demos) > 0:
                    if self.demo_min is not None:
                        first_demo_normalized = (np.array(self.demos[0]) - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                    else:
                        first_demo_normalized = np.array(self.demos[0])
                    self.stepwise_em_learner.initialize_from_first_demo(first_demo_normalized)
            
            # Update incrementally
            self.stepwise_em_learner.stepwise_em_update(deformed_demo_normalized)
            
            # Update ProMP parameters
            if self.stepwise_em_learner.mean_weights is not None:
                self.promp.mean_weights = self.stepwise_em_learner.mean_weights.copy()
                self.promp.cov_weights = self.stepwise_em_learner.cov_weights.copy()
            
            # Generate new trajectory
            num_points = len(current_trajectory)
            new_trajectory_normalized = self.promp.generate_trajectory(num_points=num_points)
            
            # Denormalize
            if self.demo_min is not None and self.demo_max is not None:
                new_trajectory = new_trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                new_trajectory = np.clip(new_trajectory, self.demo_min, self.demo_max)
            else:
                new_trajectory = new_trajectory_normalized
            
            # Update trajectory
            self.current_trajectory = new_trajectory
            self.deformer.set_trajectory(new_trajectory)
            
            # Interrupt and restart
            self.execution_interrupted.clear()
            try:
                self.kuka_socket.sendall(b"STOP\n")
                self.kuka_socket.settimeout(2.0)
                response = self._receive_complete_message(self.kuka_socket, timeout=2.0)
                self.kuka_socket.settimeout(None)
            except Exception as e:
                self.get_logger().error(f'Error sending STOP: {e}')
            
            # Restart from current point
            new_trajectory_start_idx = int(t_current * num_points)
            new_trajectory_start_idx = np.clip(new_trajectory_start_idx, 0, num_points - 1)
            trajectory_from_current = new_trajectory[new_trajectory_start_idx:]
            
            self.execution_interrupted.set()
            threading.Thread(target=self.send_trajectory_to_kuka, args=(trajectory_from_current,), daemon=True).start()
            
            self.deformation_count += 1
            self.get_logger().info(f'Trajectory deformation and incremental learning completed, restarting from point {new_trajectory_start_idx}')
            
        except Exception as e:
            self.get_logger().error(f'Error during trajectory deformation: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def send_trajectory_to_kuka(self, trajectory):
        """Send trajectory to KUKA robot - EXACT copy of train_and_execute.py logic"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            traj_array = np.array(trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            self.get_logger().info('Converting Cartesian trajectory to joint positions...')
            joint_trajectory = self.cartesian_to_joint_python(trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions')
                return False
            
            trajectory_str = ";".join([",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory])
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Wait for completion - EXACT copy of train_and_execute.py
            complete = False
            point_count = 0
            error_count = 0
            skipped_points = []
            
            while not complete:
                response = self._receive_complete_message(self.kuka_socket, timeout=30.0)
                if not response:
                    self.get_logger().warn('No response from KUKA')
                    break
                
                response = response.strip()
                
                if "TRAJECTORY_COMPLETE" in response:
                    self.get_logger().info(f'Trajectory execution completed. Points: {point_count}, Errors (skipped): {error_count}')
                    if skipped_points:
                        self.get_logger().warn(f'Skipped points: {skipped_points[:10]}' + ('...' if len(skipped_points) > 10 else ''))
                    complete = True
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    if point_count < len(trajectory):
                        skipped_points.append(point_count)
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    
                    # Update current point count for torque monitoring
                    with self.point_count_lock:
                        self.current_point_count = point_count
                    
                    # Check if execution was interrupted
                    if not self.execution_interrupted.is_set():
                        self.get_logger().info('Execution interrupted for trajectory update')
                        return False
                    
                    # Log executed trajectory point
                    if point_count <= len(trajectory):
                        self.execution_trajectory_log.append(trajectory[point_count - 1].tolist())
                    
                    # Log joint torques if available
                    if len(self.joint_torque_data) > 0:
                        latest_joint = self.joint_torque_data[-1]
                        self.joint_torque_log.append({
                            'timestamp': latest_joint['timestamp'],
                            'joint_torques': latest_joint['joint_torques'].copy()
                        })
                    
                    # Log external torques if available
                    if len(self.torque_data) > 0:
                        latest_torque = self.torque_data[-1]
                        self.external_torque_log.append({
                            'timestamp': latest_torque['timestamp'],
                            'force': latest_torque['force'].copy(),
                            'torque': latest_torque['torque'].copy()
                        })
                    
                    if point_count % 10 == 0:
                        self.get_logger().info(f'Progress: {point_count}/{len(trajectory)} points completed (errors skipped: {error_count})')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors (skipped): {error_count}')
                if point_count < len(trajectory):
                    self.execution_trajectory_log.extend(trajectory[:point_count].tolist())
                return point_count > 0
            else:
                self.execution_trajectory_log = trajectory.tolist()
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def stop_execution(self):
        """Stop trajectory execution"""
        self.is_executing = False
        if self.kuka_socket:
            try:
                self.kuka_socket.sendall(b"STOP\n")
            except:
                pass
        self.get_logger().info('Stopped trajectory execution')
        self.execution_status_pub.publish(String(data='EXECUTION_STOPPED'))
        self.save_execution_data_to_csv()
    
    def save_execution_data_to_csv(self):
        """Save execution trajectory, joint torques, and external torques to CSV files"""
        try:
            os.makedirs(self.result_directory, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if len(self.execution_trajectory_log) > 0:
                traj_file = os.path.join(self.result_directory, f'execution_trajectory_{timestamp}.csv')
                with open(traj_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['x_m', 'y_m', 'z_m', 'alpha_rad', 'beta_rad', 'gamma_rad'])
                    for point in self.execution_trajectory_log:
                        writer.writerow(point)
                self.get_logger().info(f'Saved execution trajectory to {traj_file} ({len(self.execution_trajectory_log)} points)')
            else:
                self.get_logger().warn('No execution trajectory data to save')
            
            if len(self.joint_torque_log) > 0:
                joint_torque_file = os.path.join(self.result_directory, f'joint_torques_{timestamp}.csv')
                with open(joint_torque_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp_s', 'joint1_Nm', 'joint2_Nm', 'joint3_Nm', 'joint4_Nm', 'joint5_Nm', 'joint6_Nm', 'joint7_Nm'])
                    for entry in self.joint_torque_log:
                        row = [entry['timestamp']] + entry['joint_torques']
                        writer.writerow(row)
                self.get_logger().info(f'Saved joint torques to {joint_torque_file} ({len(self.joint_torque_log)} samples)')
            
            if len(self.external_torque_log) > 0:
                external_torque_file = os.path.join(self.result_directory, f'external_torques_{timestamp}.csv')
                with open(external_torque_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp_s', 'force_x_N', 'force_y_N', 'force_z_N', 'torque_x_Nm', 'torque_y_Nm', 'torque_z_Nm'])
                    for entry in self.external_torque_log:
                        row = [entry['timestamp']] + entry['force'] + entry['torque']
                        writer.writerow(row)
                self.get_logger().info(f'Saved external torques to {external_torque_file} ({len(self.external_torque_log)} samples)')
                
        except Exception as e:
            self.get_logger().error(f'Error saving execution data to CSV: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Standalone Deformation Controller')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP')
    parser.add_argument('--trajectory-file', default=os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy'), help='Trajectory file path')
    args, _ = parser.parse_known_args()
    
    node = StandaloneDeformationController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

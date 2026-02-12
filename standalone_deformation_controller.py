#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Bool
from sensor_msgs.msg import JointState
try:
    # Try to import iiwa_msgs if available (ROS2 port of iiwa_stack)
    from iiwa_msgs.msg import JointTorque
    HAS_IIWA_MSGS = True
except ImportError:
    # Fallback to JointState if iiwa_msgs not available
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
from .trajectory_deformer import TrajectoryDeformer
from .promp import ProMP
from .stepwise_em_learner import StepwiseEMLearner
from .train_and_execute import TrainAndExecute

class StandaloneDeformationController(Node):
    def __init__(self):
        super().__init__('standalone_deformation_controller')
        
        # Parameters
        self.declare_parameter('kuka_ip', '172.31.1.147')  # Match train_and_execute.py
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('force_threshold', 10.0)
        self.declare_parameter('torque_threshold', 2.0)
        self.declare_parameter('energy_threshold', 0.5)
        # Joint torque thresholds for ProMP conditioning
        self.declare_parameter('joint_torque_threshold_small', 0.5)  # Small threshold to filter noise (Nm)
        self.declare_parameter('joint_torque_threshold_big', 3.0)  # Big threshold for ProMP conditioning (Nm)
        self.declare_parameter('deformation_alpha', 0.1)
        self.declare_parameter('deformation_waypoints', 10)
        self.declare_parameter('promp_conditioning_sigma', 0.01)
        # Default trajectory file location - check ~/robotexecute first (where train_and_execute.py saves it)
        default_trajectory = os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy')
        self.declare_parameter('trajectory_file', default_trajectory)
        self.declare_parameter('promp_file', 'promp_model.npy')
        self.declare_parameter('auto_start', True)  # Auto-start execution on initialization
        
        # Training parameters (from train_and_execute.py)
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('save_directory', '~/robot_demos')
        self.declare_parameter('demo_file', '')  # Empty means use save_directory/all_demos.npy
        self.declare_parameter('trajectory_points', 100)
        self.declare_parameter('train_on_startup', True)  # Train ProMP on startup
        self.declare_parameter('execute_after_training', True)  # Execute trajectory after training
        
        # ROS2 topic parameters for external joint torques (if using iiwa_stack)
        # Default: Use TCP socket (Java sends JOINT_TORQUE: messages)
        self.declare_parameter('external_joint_torque_topic', '/iiwa/jointExternalTorque')  # Only used if use_ros2_joint_torques=True
        self.declare_parameter('use_ros2_joint_torques', False)  # Default: False (use TCP socket from Java application)
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.force_threshold = self.get_parameter('force_threshold').value
        self.torque_threshold = self.get_parameter('torque_threshold').value
        self.energy_threshold = self.get_parameter('energy_threshold').value
        self.joint_torque_threshold_small = self.get_parameter('joint_torque_threshold_small').value
        self.joint_torque_threshold_big = self.get_parameter('joint_torque_threshold_big').value
        self.deformation_alpha = self.get_parameter('deformation_alpha').value
        self.deformation_waypoints = self.get_parameter('deformation_waypoints').value
        self.promp_conditioning_sigma = self.get_parameter('promp_conditioning_sigma').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.promp_file = self.get_parameter('promp_file').value
        self.auto_start = self.get_parameter('auto_start').value
        
        # Training parameters
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        demo_file_param = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        self.train_on_startup = self.get_parameter('train_on_startup').value
        self.execute_after_training = self.get_parameter('execute_after_training').value
        
        # ROS2 joint torque parameters
        self.external_joint_torque_topic = self.get_parameter('external_joint_torque_topic').value
        self.use_ros2_joint_torques = self.get_parameter('use_ros2_joint_torques').value
        
        # Set demo_file path (matching train_and_execute.py logic)
        if demo_file_param:
            if os.path.isabs(demo_file_param):
                self.demo_file = demo_file_param
            else:
                if os.path.exists(demo_file_param):
                    self.demo_file = demo_file_param
                else:
                    self.demo_file = os.path.join(self.save_directory, demo_file_param)
        else:
            self.demo_file = os.path.join(self.save_directory, 'all_demos.npy')
        
        # Components
        self.deformer = TrajectoryDeformer(
            alpha=self.deformation_alpha,
            n_waypoints=self.deformation_waypoints,
            energy_threshold=self.energy_threshold
        )
        self.promp = None
        self.current_trajectory = None
        self.current_joint_trajectory = None  # Joint space trajectory
        self.demos = []  # Demonstrations for training
        self.stepwise_em_learner = None  # Stepwise EM learner for incremental learning
        
        # Normalization statistics (for denormalizing generated trajectories)
        self.demo_min = None
        self.demo_max = None
        self.demo_mean = None
        self.demo_std = None
        
        # State
        self.is_executing = False
        self.is_monitoring = False
        self.execution_thread = None
        self.monitoring_thread = None
        self.torque_data = deque(maxlen=1000)  # External force/torque from sensor
        self.joint_torque_data = deque(maxlen=1000)  # Joint torques for each joint (7 joints)
        
        # Execution progress tracking
        self.trajectory_start_time = None  # Wall-clock time when trajectory started
        self.trajectory_duration = None  # Estimated duration of trajectory (seconds)
        self.current_trajectory_index = 0  # Current point index in trajectory
        
        # TCP communication
        self.kuka_socket = None
        self.torque_socket = None
        
        # Statistics
        self.deformation_count = 0
        self.conditioning_count = 0
        self.total_energy = 0.0
        
        # CSV logging data structures
        self.execution_trajectory_log = []  # Track final execution trajectory (Cartesian)
        self.joint_torque_log = []  # Track all joint torques during execution
        self.external_torque_log = []  # Track all external torques during execution
        self.result_directory = os.path.join(os.path.expanduser('~/result'), 'standalone_deformation_controller')
        
        # Thread lock for pybullet IK (pybullet is not thread-safe)
        self.pybullet_lock = threading.Lock()
        
        # Setup communication
        self.setup_communication()
        
        # Setup publishers and subscribers
        self.setup_ros_communication()
        
        # Step 1: Train ProMP from demonstrations (if enabled)
        if self.train_on_startup:
            self.get_logger().info('Training ProMP from demonstrations on startup...')
            if self.train_and_generate_trajectory():
                self.get_logger().info('ProMP training and trajectory generation completed')
            else:
                self.get_logger().warn('ProMP training failed, trying to load existing trajectory...')
                self.load_trajectory_and_promp()
        else:
            # Load existing trajectory and ProMP
            self.load_trajectory_and_promp()
        
        # Step 2: Execute trajectory (if enabled and trajectory is available)
        if self.execute_after_training and self.current_trajectory is not None:
            self.get_logger().info('Auto-execute enabled: Starting trajectory execution...')
            # Give a small delay to ensure everything is initialized
            threading.Timer(1.0, self.start_execution).start()
        elif self.auto_start and self.current_trajectory is not None:
            self.get_logger().info('Auto-start enabled: Starting trajectory execution...')
            # Give a small delay to ensure everything is initialized
            threading.Timer(1.0, self.start_execution).start()
        
        self.get_logger().info('Standalone Deformation Controller initialized')
        
    def setup_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            # Connect to KUKA for sending trajectories
            self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}:{self.kuka_port}...")
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.settimeout(5)  # 5 second timeout
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal (use robust message receiving)
            ready = self._receive_complete_message(self.kuka_socket, timeout=5.0)
            if ready and ready.strip() == "READY":
                self.get_logger().info('KUKA connection established - received READY signal')
            else:
                self.get_logger().error(f'Unexpected response from KUKA: {ready}')
                self.kuka_socket = None
            
            # Setup server for receiving torque data
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', 30003))
            self.torque_socket.listen(1)
            
            # Start torque data thread
            self.torque_thread = threading.Thread(target=self.receive_torque_data)
            self.torque_thread.daemon = True
            self.torque_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup communication: {e}')
            self.kuka_socket = None
    
    def _receive_complete_message(self, sock, timeout=5.0, buffer_size=8192):
        """
        Receive complete message from socket, handling multi-packet messages.
        Assumes messages end with newline character.
        Matches train_and_execute.py implementation.
        """
        try:
            sock.settimeout(timeout)
            message_parts = []
            
            while True:
                data = sock.recv(buffer_size)
                if not data:
                    break
                
                message_parts.append(data.decode('utf-8'))
                
                # Check if we received a complete line (ends with newline)
                if b'\n' in data:
                    break
            
            return ''.join(message_parts)
        except socket.timeout:
            self.get_logger().warn(f"Timeout waiting for response (>{timeout}s)")
            return None
        except Exception as e:
            self.get_logger().error(f"Error receiving message: {e}")
            return None
    
    def cartesian_to_joint_via_java(self, cartesian_poses):
        """Convert Cartesian poses to joint positions
        First tries Java IK, then falls back to Python pybullet IK
        Matches train_and_execute.py implementation"""
        # Try Java IK first (if available)
        # For now, fall back directly to Python IK (same as train_and_execute.py)
        return self.cartesian_to_joint_python(cartesian_poses)
    
    def cartesian_to_joint_python(self, cartesian_poses):
        """Convert Cartesian poses to joint positions using pybullet IK solver
        Matches train_and_execute.py implementation
        Thread-safe: uses lock to prevent concurrent pybullet access"""
        # Use lock to ensure thread-safe access to pybullet (pybullet is not thread-safe)
        with self.pybullet_lock:
            try:
                import pybullet as p
                import pybullet_data
                
                self.get_logger().info('Using pybullet for IK computation (most reliable for KUKA)')
                
                # Initialize pybullet in DIRECT mode (no GUI, faster)
                # Check if already connected (pybullet allows only one connection)
                try:
                    # Try to get connection info - if this fails, we're not connected
                    p.getConnectionInfo()
                    # Already connected - disconnect first to avoid conflicts
                    self.get_logger().warn('Pybullet already connected, disconnecting first...')
                    p.disconnect()
                except:
                    # Not connected, which is fine
                    pass
                
                physics_client = p.connect(p.DIRECT)
                if physics_client < 0:
                    self.get_logger().error('Failed to connect to pybullet physics server')
                    return None
                
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                
                # Try to load KUKA LBR iiwa URDF from common locations
                robot_id = None
                urdf_paths = [
                    "kuka_iiwa/model.urdf",  # pybullet_data
                    "kuka_lbr_iiwa_14_r820.urdf",  # Common name
                    "/opt/ros/noetic/share/kuka_description/urdf/kuka_lbr_iiwa_14_r820.urdf",  # ROS path
                ]
                
                for urdf_path in urdf_paths:
                    try:
                        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
                        self.get_logger().info(f'Loaded KUKA URDF from: {urdf_path}')
                        break
                    except:
                        continue
                
                # If URDF not found, create a simple 7-DOF model
                if robot_id is None:
                    self.get_logger().warn('KUKA URDF not found, creating simplified 7-DOF model')
                    robot_id = self._create_simple_kuka_model(p)
                
                if robot_id is None:
                    self.get_logger().error('Failed to create robot model')
                    p.disconnect()
                    return None
                
                # Get number of joints
                num_joints = p.getNumJoints(robot_id)
                end_effector_link = num_joints - 1
                
                joint_positions = []
                failed_count = 0
                
                # Initial joint configuration (seed for IK)
                initial_joints = [0.0, 0.7854, 0.0, -1.3962, 0.0, -0.6109, 0.0]
                current_joints = initial_joints.copy()
                
                self.get_logger().info(f'Computing IK for {len(cartesian_poses)} poses using pybullet...')
                
                for i, pose in enumerate(cartesian_poses):
                    x, y, z, alpha, beta, gamma = pose
                    target_pos = [x, y, z]
                    target_orn = p.getQuaternionFromEuler([alpha, beta, gamma])
                    
                    try:
                        # Verify connection is still active before computing IK
                        try:
                            p.getConnectionInfo()
                        except:
                            self.get_logger().error('Pybullet connection lost during IK computation')
                            raise RuntimeError('Pybullet connection lost')
                        
                        # Compute IK using pybullet's built-in solver
                        joint_angles = p.calculateInverseKinematics(
                            robot_id,
                            end_effector_link,
                            target_pos,
                            target_orn,
                            lowerLimits=[-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
                            upperLimits=[2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054],
                            jointRanges=[5.934, 4.188, 5.934, 4.188, 5.934, 4.188, 6.108],
                            restPoses=current_joints,
                            maxNumIterations=200,
                            residualThreshold=1e-5
                        )
                        
                        if joint_angles is not None and len(joint_angles) >= 7:
                            joint_angles_7 = list(joint_angles[:7])
                            
                            # Verify joint limits
                            valid = True
                            limits = [(-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967),
                                     (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094),
                                     (-3.054, 3.054)]
                            for j, (angle, (min_val, max_val)) in enumerate(zip(joint_angles_7, limits)):
                                if angle < min_val or angle > max_val:
                                    valid = False
                                    break
                            
                            if valid:
                                joint_positions.append(joint_angles_7)
                                current_joints = joint_angles_7.copy()
                            else:
                                if len(joint_positions) > 0:
                                    joint_positions.append(joint_positions[-1])
                                else:
                                    joint_positions.append(initial_joints)
                                failed_count += 1
                        else:
                            raise ValueError("IK returned invalid result")
                            
                    except Exception as e:
                        self.get_logger().warn(f'Pybullet IK failed for point {i}: {e}')
                        failed_count += 1
                        if len(joint_positions) > 0:
                            joint_positions.append(joint_positions[-1])
                        else:
                            joint_positions.append(initial_joints)
                    
                    if (i + 1) % 10 == 0:
                        self.get_logger().info(f'Pybullet IK progress: {i+1}/{len(cartesian_poses)} ({failed_count} failed)')
                
                # Disconnect pybullet before releasing lock
                try:
                    p.disconnect()
                except:
                    pass  # Ignore disconnect errors
                
                if failed_count == 0:
                    self.get_logger().info(f'Successfully computed IK for all {len(cartesian_poses)} poses using pybullet')
                else:
                    self.get_logger().warn(f'IK computed with {failed_count} failures (used previous/initial positions)')
                
                return np.array(joint_positions)
                
            except ImportError:
                self.get_logger().error('pybullet not installed. Install with: pip install pybullet')
                return None
            except Exception as e:
                self.get_logger().error(f'pybullet IK failed: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                # Ensure pybullet is disconnected even on error
                try:
                    import pybullet as p
                    p.disconnect()
                except:
                    pass
                return None
    
    def _create_simple_kuka_model(self, p):
        """Create a simple 7-DOF KUKA LBR iiwa model for IK"""
        # This is a simplified model - for best results, use actual URDF
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=base_visual, baseCollisionShapeIndex=base_collision)
        self.get_logger().warn('Simplified model created - accuracy may be reduced. Use actual URDF for best results.')
        return base_body
    
    def setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers"""
        # Publishers
        self.deformation_status_pub = self.create_publisher(String, 'deformation_status', 10)
        self.energy_pub = self.create_publisher(Float64, 'deformation_energy', 10)
        self.conditioning_status_pub = self.create_publisher(String, 'conditioning_status', 10)
        self.execution_status_pub = self.create_publisher(String, 'execution_status', 10)
        self.statistics_pub = self.create_publisher(String, 'deformation_statistics', 10)
        
        # Subscribers
        self.start_execution_sub = self.create_subscription(
            Bool, 'start_deformation_execution', self.start_execution_callback, 10)
        self.stop_execution_sub = self.create_subscription(
            Bool, 'stop_deformation_execution', self.stop_execution_callback, 10)
        self.load_trajectory_sub = self.create_subscription(
            String, 'load_trajectory', self.load_trajectory_callback, 10)
        
        # Subscribe to external joint torques from ROS2 topic (if iiwa_stack is available)
        # Otherwise, joint torques will be received via TCP socket from Java application
        if self.use_ros2_joint_torques:
            if HAS_IIWA_MSGS:
                # Use iiwa_msgs/JointTorque if available (matches iiwa_stack)
                self.external_joint_torque_sub = self.create_subscription(
                    JointTorque, self.external_joint_torque_topic, 
                    self.external_joint_torque_callback, 10)
                self.get_logger().info(f'Subscribed to external joint torques via ROS2 topic: {self.external_joint_torque_topic} (using iiwa_msgs/JointTorque)')
            else:
                # Fallback: Use sensor_msgs/JointState (standard ROS2 message)
                # Topic might be /iiwa/joint_states or similar
                self.external_joint_torque_sub = self.create_subscription(
                    JointState, self.external_joint_torque_topic,
                    self.external_joint_torque_callback_jointstate, 10)
                self.get_logger().info(f'Subscribed to external joint torques via ROS2 topic: {self.external_joint_torque_topic} (using sensor_msgs/JointState)')
                self.get_logger().info('Note: Ensure the topic publishes external joint torques in the effort field')
        else:
            self.get_logger().info('Using TCP socket for external joint torques (Java application will send JOINT_TORQUE: messages)')
            self.get_logger().info('Java application must be configured to send external joint torques via TCP port 30003')
    
    def load_demos(self):
        """Load demonstrations from file (matching train_and_execute.py)"""
        try:
            if not os.path.exists(self.demo_file):
                self.get_logger().error(f'Demo file not found: {self.demo_file}')
                self.get_logger().error('Please record demonstrations first using interactive_demo_recorder.py')
                return False
            
            loaded_data = np.load(self.demo_file, allow_pickle=True)
            demos_list = []
            
            # Handle object arrays (from interactive_demo_recorder.py)
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                for demo in loaded_data:
                    arr = np.array(demo)
                    if arr.ndim == 2 and arr.shape[1] == 6:
                        demos_list.append(arr)
                    elif arr.ndim == 1 and len(arr) == 6:
                        demos_list.append(arr.reshape(1, -1))
                    elif arr.ndim == 3:
                        for i in range(arr.shape[0]):
                            if arr[i].shape[1] == 6:
                                demos_list.append(arr[i])
            elif isinstance(loaded_data, np.ndarray):
                if loaded_data.ndim == 3:
                    for i in range(loaded_data.shape[0]):
                        demos_list.append(loaded_data[i])
                elif loaded_data.ndim == 2 and loaded_data.shape[1] == 6:
                    demos_list.append(loaded_data)
            
            self.demos = demos_list
            self.get_logger().info(f'Loaded {len(self.demos)} demonstrations from {self.demo_file}')
            return len(self.demos) > 0
            
        except Exception as e:
            self.get_logger().error(f'Error loading demonstrations: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def normalize_demos(self):
        """Normalize demonstrations to same length and compute statistics (matching train_and_execute.py)"""
        if len(self.demos) == 0:
            return []
        
        from scipy.interpolate import interp1d
        
        target_length = self.trajectory_points
        normalized = []
        all_values = []  # Collect all values for statistics
        
        # First, interpolate all demos to same length
        for demo in self.demos:
            demo_array = np.array(demo)
            t_old = np.linspace(0, 1, len(demo_array))
            t_new = np.linspace(0, 1, target_length)
            
            normalized_demo = []
            for dim in range(6):  # 6 DOF: x, y, z, alpha, beta, gamma
                try:
                    interp_func = interp1d(t_old, demo_array[:, dim], kind='cubic', fill_value='extrapolate')
                    normalized_demo.append(interp_func(t_new))
                except ValueError:
                    # Fallback to linear if cubic fails
                    interp_func = interp1d(t_old, demo_array[:, dim], kind='linear', fill_value='extrapolate')
                    normalized_demo.append(interp_func(t_new))
            
            normalized_demo_array = np.column_stack(normalized_demo)
            normalized.append(normalized_demo_array)
            all_values.append(normalized_demo_array)
        
        # Compute statistics across all demos for denormalization
        all_values = np.concatenate(all_values, axis=0)  # Stack all demos
        self.demo_min = np.min(all_values, axis=0)
        self.demo_max = np.max(all_values, axis=0)
        self.demo_mean = np.mean(all_values, axis=0)
        self.demo_std = np.std(all_values, axis=0)
        
        # Avoid division by zero
        self.demo_std = np.where(self.demo_std < 1e-10, 1.0, self.demo_std)
        
        self.get_logger().info(f'Demo statistics computed:')
        self.get_logger().info(f'  Min: {self.demo_min}')
        self.get_logger().info(f'  Max: {self.demo_max}')
        self.get_logger().info(f'  Mean: {self.demo_mean}')
        self.get_logger().info(f'  Std: {self.demo_std}')
        
        # Normalize values to [0, 1] range for ProMP training
        normalized_scaled = []
        for demo in normalized:
            # Normalize: (value - min) / (max - min)
            demo_normalized = (demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            normalized_scaled.append(demo_normalized)
        
        return normalized_scaled
    
    def train_promp(self):
        """Train ProMP on loaded demonstrations (matching train_and_execute.py)"""
        if len(self.demos) < 1:
            self.get_logger().error('No demonstrations available for training')
            return False
        
        try:
            self.get_logger().info('Normalizing demonstrations...')
            normalized_demos = self.normalize_demos()
            
            self.get_logger().info('Training ProMP...')
            self.promp = ProMP(num_basis=self.num_basis, sigma_noise=self.sigma_noise)
            self.promp.train(normalized_demos)
            
            # Initialize StepwiseEMLearner with initial demos for incremental learning
            self.get_logger().info('Initializing StepwiseEMLearner for incremental learning...')
            self.stepwise_em_learner = StepwiseEMLearner(
                num_basis=self.num_basis,
                sigma_noise=self.sigma_noise,
                delta_N=0.1  # Step size for incremental updates
            )
            
            # Initialize with first demo, then update with remaining demos
            if len(normalized_demos) > 0:
                self.stepwise_em_learner.initialize_from_first_demo(normalized_demos[0])
                self.get_logger().info(f'StepwiseEMLearner initialized with first demo')
                
                # Update with remaining demos incrementally
                for i in range(1, len(normalized_demos)):
                    self.stepwise_em_learner.stepwise_em_update(normalized_demos[i])
                    self.get_logger().info(f'StepwiseEMLearner updated with demo {i+1}/{len(normalized_demos)}')
            
            # Sync ProMP parameters from StepwiseEMLearner (they should match after initialization)
            if self.stepwise_em_learner.mean_weights is not None:
                self.promp.mean_weights = self.stepwise_em_learner.mean_weights.copy()
                self.promp.cov_weights = self.stepwise_em_learner.cov_weights.copy()
                self.get_logger().info('ProMP parameters synced with StepwiseEMLearner')
            
            self.get_logger().info('ProMP training and StepwiseEMLearner initialization completed successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error training ProMP: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def generate_trajectory(self):
        """Generate trajectory from trained ProMP and denormalize (matching train_and_execute.py)"""
        if self.promp is None:
            self.get_logger().error('ProMP not trained yet')
            return None
        
        try:
            trajectory_normalized = self.promp.generate_trajectory(num_points=self.trajectory_points)
            
            # Denormalize trajectory back to original scale
            if self.demo_min is not None and self.demo_max is not None:
                trajectory_denorm = trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                trajectory_denorm = np.clip(trajectory_denorm, self.demo_min, self.demo_max)
                self.current_trajectory = trajectory_denorm
            else:
                self.current_trajectory = trajectory_normalized
            
            self.get_logger().info(f'Generated trajectory shape: {self.current_trajectory.shape}')
            return self.current_trajectory
            
        except Exception as e:
            self.get_logger().error(f'Error generating trajectory: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def train_and_generate_trajectory(self):
        """Train ProMP and generate trajectory (complete training pipeline)"""
        # Step 1: Load demonstrations
        if not self.load_demos():
            return False
        
        # Step 2: Train ProMP
        if not self.train_promp():
            return False
        
        # Step 3: Generate trajectory
        trajectory = self.generate_trajectory()
        if trajectory is None:
            return False
        
        # Step 4: Set trajectory in deformer
        self.deformer.set_trajectory(trajectory)
        
        self.get_logger().info('Training and trajectory generation completed successfully')
        return True
    
    def load_trajectory_and_promp(self):
        """Load trajectory and ProMP from files
        Checks multiple common locations for trajectory file"""
        try:
            # Try to load trajectory - check multiple locations
            trajectory_loaded = False
            trajectory_paths = [
                self.trajectory_file,  # User-specified or default
                os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy'),  # train_and_execute.py default location
                os.path.join(os.path.expanduser('~/newfoldername'), 'learned_trajectory.npy'),  # Alternative location
                'learned_trajectory.npy',  # Current directory
            ]
            
            for traj_path in trajectory_paths:
                if os.path.exists(traj_path):
                    try:
                        self.current_trajectory = np.load(traj_path)
                        self.deformer.set_trajectory(self.current_trajectory)
                        self.get_logger().info(f'Loaded trajectory from {traj_path}')
                        trajectory_loaded = True
                        break
                    except Exception as e:
                        self.get_logger().warn(f'Failed to load trajectory from {traj_path}: {e}')
                        continue
            
            if not trajectory_loaded:
                self.get_logger().error(f'Trajectory file not found!')
                self.get_logger().error(f'Checked paths: {trajectory_paths}')
                self.get_logger().error('Please run train_and_execute.py first to generate a trajectory, or specify --trajectory-file')
                self.get_logger().error('Example: ros2 run kuka_promp_control standalone_deformation_controller --trajectory-file ~/robotexecute/learned_trajectory.npy')
                self.current_trajectory = None
                return
            
            # Load ProMP (if available) - also check multiple locations
            promp_paths = [
                self.promp_file,  # User-specified
                os.path.join(os.path.expanduser('~/robotexecute'), 'promp_model.npy'),
                'promp_model.npy',  # Current directory
            ]
            
            promp_loaded = False
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
                        promp_loaded = True
                        break
                    except Exception as e:
                        self.get_logger().warn(f'Failed to load ProMP from {promp_path}: {e}')
                        continue
            
            if not promp_loaded:
                self.get_logger().warn(f'ProMP file not found in any of: {promp_paths}')
                self.get_logger().info('ProMP is optional - deformation will use trajectory deformer instead')
                self.promp = None
                
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory/ProMP: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def start_execution_callback(self, msg):
        """Callback to start execution"""
        if msg.data:
            self.start_execution()
        else:
            self.stop_execution()
    
    def stop_execution_callback(self, msg):
        """Callback to stop execution"""
        if msg.data:
            self.stop_execution()
    
    def load_trajectory_callback(self, msg):
        """Callback to load new trajectory"""
        try:
            trajectory = np.load(msg.data)
            self.current_trajectory = trajectory
            self.deformer.set_trajectory(trajectory)
            self.get_logger().info(f'Loaded new trajectory from {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory: {e}')
    
    def external_joint_torque_callback(self, msg):
        """
        Callback for external joint torques from ROS2 topic (using iiwa_msgs/JointTorque)
        Similar to iiwa_stack's ExternalJointTorque class
        
        Expected message format (iiwa_msgs/JointTorque):
        - torque: float64[7] - external joint torques for 7 joints (Nm)
        - header: std_msgs/Header with timestamp
        """
        try:
            # Extract joint torques from message
            # iiwa_msgs/JointTorque typically has a 'torque' field with 7 values
            if hasattr(msg, 'torque') and len(msg.torque) >= 7:
                joint_torques = list(msg.torque[:7])  # Take first 7 values
            elif hasattr(msg, 'data') and len(msg.data) >= 7:
                joint_torques = list(msg.data[:7])
            else:
                self.get_logger().warn(f'Unexpected JointTorque message format: {type(msg)}')
                return
            
            # Get timestamp
            timestamp = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                # Convert ROS2 time to seconds
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            elif hasattr(msg, 'timestamp'):
                timestamp = msg.timestamp
            
            # Store external joint torques
            self.joint_torque_data.append({
                'timestamp': timestamp,
                'joint_torques': joint_torques  # [tau1, tau2, ..., tau7] in Nm
            })
            
        except Exception as e:
            self.get_logger().error(f'Error processing external joint torque message: {e}')
    
    def external_joint_torque_callback_jointstate(self, msg):
        """
        Callback for external joint torques from ROS2 topic (using sensor_msgs/JointState)
        Fallback when iiwa_msgs is not available
        
        Expected message format (sensor_msgs/JointState):
        - name: string[] - joint names
        - effort: float64[] - joint efforts (torques) - should contain external torques
        - header: std_msgs/Header with timestamp
        """
        try:
            # Extract joint torques from effort field
            if not hasattr(msg, 'effort') or len(msg.effort) < 7:
                self.get_logger().debug('JointState message does not have 7 joint efforts')
                return
            
            joint_torques = list(msg.effort[:7])  # Take first 7 joint torques
            
            # Get timestamp
            timestamp = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Store external joint torques
            self.joint_torque_data.append({
                'timestamp': timestamp,
                'joint_torques': joint_torques  # [tau1, tau2, ..., tau7] in Nm
            })
            
            # Log periodically
            if len(self.joint_torque_data) % 100 == 0:
                self.get_logger().debug(f'Received external joint torques via ROS2: {joint_torques}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing JointState message: {e}')
    
    def receive_torque_data(self):
        """Receive torque data from KUKA robot via TCP socket
        Receives external force/torque (from sensor) at flange
        
        Note: If use_ros2_joint_torques=True, external joint torques are received via ROS2 topic
        instead of TCP socket (cleaner approach, similar to iiwa_stack)
        """
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
                            
                            # Only parse external force/torque from TCP (joint torques come from ROS2 if enabled)
                            if not self.use_ros2_joint_torques and line.startswith('JOINT_TORQUE:'):
                                # Parse joint torque data from TCP (fallback mode)
                                joint_data = line.replace('JOINT_TORQUE:', '').strip()
                                values = [float(x) for x in joint_data.split(',')]
                                if len(values) >= 8:  # timestamp + 7 joints
                                    timestamp = values[0]
                                    joint_torques = values[1:8]  # 7 joint torques
                                    self.joint_torque_data.append({
                                        'timestamp': timestamp,
                                        'joint_torques': joint_torques
                                    })
                            elif len(parts) >= 7:
                                # Parse external force/torque data (format: timestamp,fx,fy,fz,tx,ty,tz)
                                timestamp, fx, fy, fz, tx, ty, tz = map(float, parts[:7])
                                self.torque_data.append({
                                    'timestamp': timestamp,
                                    'force': [fx, fy, fz],
                                    'torque': [tx, ty, tz]
                                })
                        except ValueError as e:
                            self.get_logger().debug(f'Error parsing torque data line: {line[:50]}... Error: {e}')
                            continue
                            
        except Exception as e:
            self.get_logger().error(f'Error receiving torque data: {e}')
    
    def detect_human_interaction(self):
        """Detect human interaction from torque data"""
        if len(self.torque_data) == 0:
            return None
        
        # Get latest torque data
        latest_data = self.torque_data[-1]
        
        force_magnitude = np.linalg.norm(latest_data['force'])
        torque_magnitude = np.linalg.norm(latest_data['torque'])
        
        if force_magnitude > self.force_threshold or torque_magnitude > self.torque_threshold:
            # Return human input vector (combine force and torque)
            human_input = np.array(latest_data['force'] + latest_data['torque'])
            return human_input
        
        return None
    
    def start_execution(self):
        """Start execution of the trajectory with real-time deformation"""
        if self.is_executing:
            self.get_logger().warn('Execution already in progress')
            return
        if self.current_trajectory is None:
            self.get_logger().error('No trajectory loaded for execution')
            return
        self.is_executing = True
        self.execution_thread = threading.Thread(target=self.execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        self.get_logger().info('Started trajectory execution with deformation monitoring')

    def execution_loop(self):
        """Main execution loop: execute trajectory first (like train_and_execute.py), 
        then monitor torque during execution and trigger deformation when needed"""
        trajectory = np.copy(self.current_trajectory)
        num_points = trajectory.shape[0]
        dt = 0.01  # 100 Hz monitoring rate
        
        # Initialize CSV logging data structures
        self.execution_trajectory_log = []
        self.joint_torque_log = []
        self.external_torque_log = []
        
        # Create result directory
        os.makedirs(self.result_directory, exist_ok=True)
        self.get_logger().info(f'CSV logging enabled. Results will be saved to: {self.result_directory}')
        
        # Step 1: Execute the initial trajectory first (like train_and_execute.py)
        # This ensures the robot actually starts moving
        self.get_logger().info('Starting trajectory execution...')
        self.execution_status_pub.publish(String(data='EXECUTION_STARTED'))
        
        # Track execution progress (shared between threads)
        point_count_lock = threading.Lock()
        point_count = [0]  # Use list to allow modification from nested function
        
        # Send trajectory and start execution in a separate thread that monitors progress
        # but allows interruption for deformation
        trajectory_executing = threading.Event()
        trajectory_executing.set()  # Set to True initially
        execution_thread = None
        
        def execute_trajectory_with_monitoring(traj):
            """Execute trajectory and monitor progress, can be interrupted"""
            try:
                # Use the method that tracks progress via point_count
                success = self.send_trajectory_to_kuka_with_interrupt_and_progress(
                    traj, trajectory_executing, point_count, point_count_lock)
                if success:
                    self.get_logger().info('Initial trajectory execution completed')
                else:
                    self.get_logger().warn('Initial trajectory execution had errors, but continuing monitoring...')
                    # Even if there were errors, continue monitoring - some points may have executed
                    # Set point_count to indicate we've started (even if partially)
                    with point_count_lock:
                        if point_count[0] == 0:
                            # No points executed, set to 1 to allow monitoring to continue
                            point_count[0] = 1
            except Exception as e:
                self.get_logger().error(f'Error in trajectory execution: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                # Set point_count to allow monitoring to continue even after error
                with point_count_lock:
                    if point_count[0] == 0:
                        point_count[0] = 1
            finally:
                # Only clear if trajectory actually finished completely
                # If interrupted, keep it set so monitoring can detect it
                if not trajectory_executing.is_set():
                    # Trajectory was interrupted, don't clear
                    pass
                else:
                    trajectory_executing.clear()  # Mark as finished
        
        # Start trajectory execution in background thread
        execution_thread = threading.Thread(target=execute_trajectory_with_monitoring, args=(trajectory,))
        execution_thread.daemon = True
        execution_thread.start()
        self.get_logger().info('Trajectory execution started, beginning torque monitoring...')
        
        # Step 2: Monitor torque during execution
        last_deformation_time = 0
        deformation_cooldown = 0.5  # Minimum time between deformations (seconds)
        last_conditioning_time = 0
        conditioning_cooldown = 0.3  # Minimum time between conditioning (seconds)
        
        while self.is_executing:
            # Check if trajectory execution thread is still alive
            if execution_thread is not None and not execution_thread.is_alive():
                # Thread finished (either completed or errored)
                # Check if trajectory_executing flag was cleared
                if not trajectory_executing.is_set():
                    # Trajectory finished or was interrupted
                    # Continue monitoring for a bit to allow for any final torque data
                    self.get_logger().info('Trajectory execution thread finished, continuing monitoring...')
                    # Don't break - continue monitoring loop
                else:
                    # Thread died but flag is still set - something went wrong
                    self.get_logger().warn('Trajectory execution thread died unexpectedly, but continuing monitoring...')
            
            # Check if trajectory is still executing
            if not trajectory_executing.is_set() and execution_thread is not None:
                # Trajectory finished or was interrupted, but continue monitoring
                # This allows the system to continue monitoring torques even after trajectory completes
                # or if there were errors
                pass
            
            # Monitor joint torques during execution and check thresholds
            if len(self.joint_torque_data) > 0:
                latest_joint_torques = self.joint_torque_data[-1]
                joint_torques = latest_joint_torques['joint_torques']
                max_joint_torque = max(abs(t) for t in joint_torques)
                
                # Log joint torques to CSV
                self.joint_torque_log.append({
                    'timestamp': latest_joint_torques['timestamp'],
                    'joint_torques': joint_torques.copy()
                })
                
                # Log joint torques periodically (every 50 samples)
                if len(self.joint_torque_data) % 50 == 0:
                    self.get_logger().info(f'Joint torques: J1={joint_torques[0]:.2f}, J2={joint_torques[1]:.2f}, '
                                          f'J3={joint_torques[2]:.2f}, J4={joint_torques[3]:.2f}, '
                                          f'J5={joint_torques[4]:.2f}, J6={joint_torques[5]:.2f}, '
                                          f'J7={joint_torques[6]:.2f} Nm (max={max_joint_torque:.2f} Nm)')
                
                # Check joint torque thresholds for ProMP conditioning
                current_time = time.time()
                if (max_joint_torque > self.joint_torque_threshold_small and 
                    max_joint_torque < self.joint_torque_threshold_big and
                    (current_time - last_conditioning_time) > conditioning_cooldown):
                    
                    # Small to medium torque: Trigger ProMP conditioning at current time t
                    self.get_logger().info(f'Joint torque {max_joint_torque:.3f} Nm exceeds small threshold '
                                          f'({self.joint_torque_threshold_small:.3f} Nm) - triggering ProMP conditioning')
                    
                    # Get current execution progress
                    with point_count_lock:
                        current_idx = point_count[0]
                    
                    # Calculate current time t in trajectory (normalized 0-1)
                    t_current = current_idx / max(num_points, 1)
                    t_current = np.clip(t_current, 0.0, 1.0)
                    
                    # Get current robot pose for conditioning
                    current_pose = self.get_current_robot_pose()
                    if current_pose is not None:
                        # Trigger ProMP conditioning at time t_current
                        self.trigger_promp_conditioning_at_time(t_current, current_pose, trajectory, 
                                                               trajectory_executing, execution_thread,
                                                               execute_trajectory_with_monitoring, point_count, point_count_lock)
                        last_conditioning_time = current_time
                    else:
                        self.get_logger().warn('Could not get current robot pose, skipping conditioning')
                
                elif max_joint_torque >= self.joint_torque_threshold_big:
                    # Large torque: Trigger trajectory deformation and incremental learning
                    current_time = time.time()
                    if (current_time - last_conditioning_time) > conditioning_cooldown:
                        self.get_logger().info(f'Joint torque {max_joint_torque:.3f} Nm exceeds big threshold '
                                              f'({self.joint_torque_threshold_big:.3f} Nm) - triggering trajectory deformation')
                        
                        # Get current execution progress to restart from this point
                        with point_count_lock:
                            current_idx = point_count[0]
                        
                        # Calculate current time t in trajectory (normalized 0-1)
                        t_current = current_idx / max(num_points, 1)
                        t_current = np.clip(t_current, 0.0, 1.0)
                        
                        # Get current force/torque for deformation
                        if len(self.torque_data) > 0:
                            latest_torque = self.torque_data[-1]
                            human_input = np.array(latest_torque['force'] + latest_torque['torque'])
                        else:
                            # Fallback: use joint torques converted to Cartesian (approximate)
                            # Sum joint torques as proxy for external force
                            human_input = np.array([max_joint_torque, max_joint_torque, max_joint_torque, 0.0, 0.0, 0.0])
                        
                        # Trigger trajectory deformation and incremental learning
                        # Pass current_idx and t_current to restart from this point
                        self.handle_trajectory_deformation_and_incremental_learning(
                            trajectory, human_input, trajectory_executing, execution_thread,
                            execute_trajectory_with_monitoring, point_count, point_count_lock,
                            current_idx, t_current)
                        last_conditioning_time = current_time
            
            # Check for external torque/force during execution (legacy support)
            if len(self.torque_data) > 0:
                latest = self.torque_data[-1]
                
                # Log external torque to CSV
                self.external_torque_log.append({
                    'timestamp': latest['timestamp'],
                    'force': latest['force'].copy(),
                    'torque': latest['torque'].copy()
                })
                
                max_force = max(abs(f) for f in latest['force'])
                max_torque = max(abs(t) for t in latest['torque'])
                
                # Check if force/torque exceeds threshold and cooldown has passed
                current_time = time.time()
                if (max_force > self.force_threshold or max_torque > self.torque_threshold) and \
                   (current_time - last_deformation_time) > deformation_cooldown:
                    
                    # Compute deformation energy
                    energy = np.linalg.norm(latest['force']) + np.linalg.norm(latest['torque'])
                    self.get_logger().info(f'Deformation detected! Energy: {energy:.4f}, Force: {max_force:.2f}N, Torque: {max_torque:.2f}Nm')
                    self.deformation_status_pub.publish(String(data='DEFORMATION_DETECTED'))
                    
                    # Interrupt current trajectory execution
                    trajectory_executing.clear()  # Signal to stop monitoring current trajectory
                    
                    # Send STOP to robot
                    try:
                        self.kuka_socket.sendall(b"STOP\n")
                        self.get_logger().info('Sent STOP to robot to interrupt current trajectory')
                        
                        # Wait for STOPPED response with timeout
                        self.kuka_socket.settimeout(2.0)
                        try:
                            response = self._receive_complete_message(self.kuka_socket, timeout=2.0)
                            if response and "STOPPED" in response:
                                self.get_logger().info('Robot stopped successfully')
                            else:
                                self.get_logger().warn('Did not receive STOPPED confirmation, proceeding anyway')
                        except socket.timeout:
                            self.get_logger().warn('Timeout waiting for STOPPED, proceeding with new trajectory')
                        finally:
                            self.kuka_socket.settimeout(None)
                    except Exception as e:
                        self.get_logger().error(f'Error sending STOP: {e}')
                    
                    # Get current robot position for conditioning
                    try:
                        self.kuka_socket.sendall(b"GET_POSE\n")
                        self.kuka_socket.settimeout(1.0)
                        pose_response = self._receive_complete_message(self.kuka_socket, timeout=1.0)
                        self.kuka_socket.settimeout(None)
                        
                        if pose_response and pose_response.startswith("POSE:"):
                            pose_str = pose_response.split("POSE:")[1].strip()
                            current_pose = np.array([float(x) for x in pose_str.split(",")])
                            t_condition = 0.5  # Use current time in trajectory (normalized)
                            y_condition = current_pose
                            self.get_logger().info(f'Got current pose for conditioning: {current_pose}')
                        else:
                            # Fallback: use mid-trajectory point
                            t_condition = 0.5
                            y_condition = trajectory[int(num_points * t_condition)]
                            self.get_logger().warn('Could not get current pose, using mid-trajectory point')
                    except Exception as e:
                        self.get_logger().warn(f'Could not get current pose: {e}, using mid-trajectory point')
                        t_condition = 0.5
                        y_condition = trajectory[int(num_points * t_condition)]
                    
                    # Generate new trajectory based on deformation energy
                    try:
                        new_traj = None
                        if self.promp is not None:
                            if energy < self.energy_threshold:
                                # Low energy: use ProMP conditioning
                                self.get_logger().info(f'Energy {energy:.4f} < threshold {self.energy_threshold} - Triggering ProMP conditioning')
                                self.promp.condition_on_waypoint(t_condition, y_condition)
                                new_traj = self.promp.generate_trajectory(num_points=num_points)
                                self.conditioning_count += 1
                                self.conditioning_status_pub.publish(String(data='CONDITIONING_COMPLETED'))
                            else:
                                # High energy: use stepwise EM update
                                self.get_logger().info(f'Energy {energy:.4f} >= threshold {self.energy_threshold} - Triggering stepwise EM update')
                                
                                # Deform trajectory first
                                human_input = np.array(latest['force'] + latest['torque'])
                                deformed_traj, _, deformation_energy = self.deformer.deform(human_input)
                                
                                if deformed_traj is not None:
                                    # Normalize deformed trajectory
                                    if self.demo_min is not None and self.demo_max is not None:
                                        deformed_demo_normalized = (np.array(deformed_traj) - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                                    else:
                                        deformed_demo_normalized = np.array(deformed_traj)
                                    
                                    # Initialize StepwiseEMLearner if needed
                                    if self.stepwise_em_learner is None:
                                        self.stepwise_em_learner = StepwiseEMLearner(
                                            num_basis=self.num_basis,
                                            sigma_noise=self.sigma_noise,
                                            delta_N=0.1
                                        )
                                        # Initialize with first demo if available
                                        if len(self.demos) > 0:
                                            if self.demo_min is not None:
                                                first_demo_normalized = (np.array(self.demos[0]) - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                                            else:
                                                first_demo_normalized = np.array(self.demos[0])
                                            self.stepwise_em_learner.initialize_from_first_demo(first_demo_normalized)
                                    
                                    # Update StepwiseEMLearner with deformed trajectory
                                    self.stepwise_em_learner.stepwise_em_update(deformed_demo_normalized)
                                    
                                    # Update ProMP parameters from StepwiseEMLearner
                                    if self.stepwise_em_learner.mean_weights is not None:
                                        self.promp.mean_weights = self.stepwise_em_learner.mean_weights.copy()
                                        self.promp.cov_weights = self.stepwise_em_learner.cov_weights.copy()
                                        self.promp.basis_centers = self.stepwise_em_learner.basis_centers.copy()
                                        self.promp.basis_width = self.stepwise_em_learner.basis_width
                                    
                                    # Generate new trajectory from updated ProMP
                                    num_points = len(trajectory)
                                    new_traj_normalized = self.promp.generate_trajectory(num_points=num_points)
                                    
                                    # Denormalize
                                    if self.demo_min is not None and self.demo_max is not None:
                                        new_traj = new_traj_normalized * (self.demo_max - self.demo_min) + self.demo_min
                                        new_traj = np.clip(new_traj, self.demo_min, self.demo_max)
                                    else:
                                        new_traj = new_traj_normalized
                                    
                                    # Add deformed trajectory as new demo
                                    self.demos.append(deformed_traj)
                                    self.get_logger().info(f'Deformed trajectory added as demonstration #{len(self.demos)}')
                                else:
                                    self.get_logger().warn('Deformation failed, keeping current trajectory')
                                    new_traj = trajectory
                                
                                self.deformation_status_pub.publish(String(data=f'HIGH_DEFORMATION:{energy:.4f}'))
                        else:
                            # No ProMP available - use trajectory deformer
                            self.get_logger().info('Using trajectory deformer (no ProMP available)')
                            human_input = np.array(latest['force'] + latest['torque'])
                            deformed_traj, _, _ = self.deformer.deform(human_input)
                            if deformed_traj is not None:
                                new_traj = deformed_traj
                            else:
                                self.get_logger().warn('Deformation failed, keeping current trajectory')
                                new_traj = trajectory
                        
                        if new_traj is not None:
                            # Update trajectory and send to robot
                            trajectory = new_traj
                            self.current_trajectory = new_traj
                            self.deformer.set_trajectory(new_traj)
                            
                            # Restart trajectory execution with new trajectory
                            trajectory_executing.set()  # Reset flag
                            execution_thread = threading.Thread(
                                target=execute_trajectory_with_monitoring, 
                                args=(new_traj,)
                            )
                            execution_thread.daemon = True
                            execution_thread.start()
                            self.get_logger().info('Sent new trajectory to robot after deformation')
                            
                            last_deformation_time = current_time
                            self.deformation_count += 1
                            self.total_energy += energy
                            self.energy_pub.publish(Float64(data=energy))
                        else:
                            self.get_logger().error('Failed to generate new trajectory')
                        
                    except Exception as e:
                        self.get_logger().error(f'Error generating new trajectory: {e}')
                        import traceback
                        self.get_logger().error(traceback.format_exc())
            
            time.sleep(dt)
        
        # Wait for execution thread to finish
        if execution_thread is not None:
            execution_thread.join(timeout=5.0)
        
        self.is_executing = False
        self.get_logger().info('Trajectory execution finished')
        self.execution_status_pub.publish(String(data='EXECUTION_STOPPED'))
        
        # Save CSV files
        self.save_execution_data_to_csv()
    
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
    
    def stop_execution(self):
        """Stop trajectory execution"""
        self.is_executing = False
        self.is_monitoring = False
        
        if self.kuka_socket:
            try:
                self.kuka_socket.sendall(b"STOP\n")
            except:
                pass
        
        self.get_logger().info('Stopped trajectory execution')
        self.execution_status_pub.publish(String(data='EXECUTION_STOPPED'))
        
        # Save CSV files
        self.save_execution_data_to_csv()
        
        # Publish final statistics
        self.publish_statistics()
    
    def monitoring_loop(self):
        """Deformation monitoring loop"""
        try:
            while self.is_monitoring:
                # Check for human interaction
                human_input = self.detect_human_interaction()
                
                if human_input is not None:
                    self.get_logger().info('Human interaction detected - applying deformation')
                    self.deformation_status_pub.publish(String(data='DEFORMATION_DETECTED'))
                    
                    # Apply deformation
                    deformed_traj, original_traj, energy = self.deformer.deform(human_input)
                    
                    if deformed_traj is not None:
                        self.deformation_count += 1
                        self.total_energy += energy
                        
                        # Publish energy
                        self.energy_pub.publish(Float64(data=energy))
                        
                        # Check energy threshold
                        if self.deformer.should_condition_promp():
                            self.get_logger().info(f'Energy {energy:.4f} < threshold - triggering ProMP conditioning')
                            self.trigger_promp_conditioning(deformed_traj)
                        else:
                            self.get_logger().info(f'Energy {energy:.4f} >= threshold - high deformation detected')
                            self.handle_high_deformation(deformed_traj, energy)
                
                time.sleep(0.01)  # 100 Hz monitoring
                
        except Exception as e:
            self.get_logger().error(f'Error in monitoring loop: {e}')
        finally:
            self.is_monitoring = False
    
    def trigger_promp_conditioning(self, deformed_trajectory):
        """Trigger ProMP conditioning based on deformation"""
        if self.promp is None:
            self.get_logger().warn('No ProMP available for conditioning')
            return
        
        try:
            # Get deformation region
            deform_region = self.deformer.get_deformation_region()
            
            # Create waypoints for conditioning
            t_conditions = []
            y_conditions = []
            
            for i, idx in enumerate(deform_region):
                # Normalize time to 0-1
                t_condition = idx / len(self.current_trajectory)
                y_condition = deformed_trajectory[idx]
                
                t_conditions.append(t_condition)
                y_conditions.append(y_condition)
            
            # Condition ProMP
            self.promp.condition_on_multiple_waypoints(
                t_conditions, y_conditions, self.promp_conditioning_sigma
            )
            
            # Generate new trajectory
            new_trajectory = self.promp.generate_trajectory(len(self.current_trajectory))
            
            # Update current trajectory
            self.current_trajectory = new_trajectory
            self.deformer.set_trajectory(new_trajectory)
            
            # Send updated trajectory to KUKA
            self.send_trajectory_to_kuka(new_trajectory)
            
            self.conditioning_count += 1
            self.get_logger().info('ProMP conditioning completed and new trajectory sent')
            self.conditioning_status_pub.publish(String(data='CONDITIONING_COMPLETED'))
            
        except Exception as e:
            self.get_logger().error(f'Error during ProMP conditioning: {e}')
    
    def handle_high_deformation(self, deformed_trajectory, energy):
        """Handle high deformation energy scenarios"""
        self.get_logger().warn(f'High deformation energy detected: {energy:.4f}')
        
        # For now, just log the high deformation
        # You can implement additional logic here (e.g., stop execution, switch to manual mode, etc.)
        self.deformation_status_pub.publish(String(data=f'HIGH_DEFORMATION:{energy:.4f}'))
    
    def trigger_promp_conditioning_at_time(self, t_current, current_pose, current_trajectory,
                                          trajectory_executing, execution_thread,
                                          execute_func, point_count, point_count_lock):
        """
        Trigger ProMP conditioning at the current execution time point
        
        Args:
            t_current: Current time in trajectory (normalized 0-1)
            current_pose: Current robot pose [x, y, z, alpha, beta, gamma]
            current_trajectory: Current trajectory being executed
            trajectory_executing: Event flag for trajectory execution
            execution_thread: Thread executing the trajectory
            execute_func: Function to execute trajectory
            point_count: List tracking current point index
            point_count_lock: Lock for thread-safe access to point_count
        """
        if self.promp is None:
            self.get_logger().error('ProMP not available for conditioning')
            return
        
        try:
            self.get_logger().info(f'Triggering ProMP conditioning at t={t_current:.3f} with pose: {current_pose}')
            
            # Normalize current pose for conditioning
            if self.demo_min is not None and self.demo_max is not None:
                current_pose_normalized = (current_pose - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            else:
                current_pose_normalized = current_pose
            
            # Condition ProMP at time t_current with current pose
            self.promp.condition_on_waypoint(t_current, current_pose_normalized, self.promp_conditioning_sigma)
            
            # Generate new trajectory from conditioned ProMP
            num_points = len(current_trajectory)
            new_trajectory_normalized = self.promp.generate_trajectory(num_points=num_points)
            
            # Denormalize new trajectory
            if self.demo_min is not None and self.demo_max is not None:
                new_trajectory = new_trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                new_trajectory = np.clip(new_trajectory, self.demo_min, self.demo_max)
            else:
                new_trajectory = new_trajectory_normalized
            
            self.get_logger().info(f'Generated new trajectory from conditioned ProMP (shape: {new_trajectory.shape})')
            
            # Get current execution progress
            with point_count_lock:
                current_idx = point_count[0]
            
            # Merge new trajectory: keep executed part, replace future part
            # The new trajectory starts from t=0, so we need to extract the part from t_current onwards
            new_trajectory_start_idx = int(t_current * num_points)
            new_trajectory_start_idx = np.clip(new_trajectory_start_idx, 0, num_points - 1)
            
            # Merge: executed part + new conditioned part
            merged_trajectory = np.vstack((
                current_trajectory[:current_idx],
                new_trajectory[new_trajectory_start_idx:]
            ))
            
            # Interrupt current execution
            trajectory_executing.clear()  # Signal to stop current trajectory
            
            # Send STOP to robot
            try:
                self.kuka_socket.sendall(b"STOP\n")
                self.get_logger().info('Sent STOP to robot to apply conditioned trajectory')
                
                # Wait for STOPPED response
                self.kuka_socket.settimeout(2.0)
                try:
                    response = self._receive_complete_message(self.kuka_socket, timeout=2.0)
                    if response and "STOPPED" in response:
                        self.get_logger().info('Robot stopped successfully, restarting with conditioned trajectory')
                    else:
                        self.get_logger().warn('Did not receive STOPPED confirmation, proceeding anyway')
                except socket.timeout:
                    self.get_logger().warn('Timeout waiting for STOPPED, proceeding with new trajectory')
                finally:
                    self.kuka_socket.settimeout(None)
            except Exception as e:
                self.get_logger().error(f'Error sending STOP: {e}')
            
            # Restart trajectory execution from current point with merged trajectory
            trajectory_from_current = merged_trajectory[current_idx:]
            
            execution_thread = threading.Thread(
                target=execute_func,
                args=(trajectory_from_current,)
            )
            execution_thread.daemon = True
            execution_thread.start()
            trajectory_executing.set()  # Reset flag
            
            # Update point count to reflect the restart point
            with point_count_lock:
                point_count[0] = current_idx
            
            # Update current trajectory
            self.current_trajectory = merged_trajectory
            
            self.conditioning_count += 1
            self.get_logger().info(f'ProMP conditioning completed and execution restarted from point {current_idx} (t={t_current:.3f})')
            self.conditioning_status_pub.publish(String(data=f'CONDITIONING_COMPLETED:t={t_current:.3f}'))
            
        except Exception as e:
            self.get_logger().error(f'Error during ProMP conditioning: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def handle_trajectory_deformation_and_incremental_learning(self, current_trajectory, human_input,
                                                               trajectory_executing, execution_thread,
                                                               execute_func, point_count, point_count_lock,
                                                               current_idx, t_current):
        """
        Handle trajectory deformation when big threshold is exceeded:
        1. Deform current trajectory using trajectory_deformer
        2. Treat deformed trajectory as new demonstration
        3. Train incrementally using StepwiseEMLearner
        4. Generate new trajectory from updated ProMP
        5. Continue execution with new trajectory from the time point where force was applied
        
        Args:
            current_trajectory: Current trajectory being executed
            human_input: Human force/torque input [fx, fy, fz, tx, ty, tz]
            trajectory_executing: Event flag for trajectory execution
            execution_thread: Thread executing the trajectory
            execute_func: Function to execute trajectory
            point_count: List tracking current point index
            point_count_lock: Lock for thread-safe access to point_count
            current_idx: Current point index where force was applied
            t_current: Current time in trajectory (normalized 0-1) where force was applied
        """
        try:
            self.get_logger().info('Starting trajectory deformation and incremental learning...')
            
            # Step 1: Deform current trajectory
            self.deformer.set_trajectory(current_trajectory)
            deformed_traj, original_traj, deformation_energy = self.deformer.deform(human_input)
            
            if deformed_traj is None:
                self.get_logger().error('Trajectory deformation failed')
                return
            
            self.get_logger().info(f'Trajectory deformed successfully. Energy: {deformation_energy:.4f}')
            self.deformation_status_pub.publish(String(data=f'DEFORMATION_COMPLETED:{deformation_energy:.4f}'))
            
            # Step 2: Normalize deformed trajectory to match demo format
            # Deformed trajectory is already in the same format as demos (N x 6)
            deformed_demo = np.array(deformed_traj)
            
            # Normalize deformed demo using same statistics as original demos
            if self.demo_min is not None and self.demo_max is not None:
                # Normalize: (value - min) / (max - min)
                deformed_demo_normalized = (deformed_demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                # Ensure same length as other demos
                if len(deformed_demo_normalized) != self.trajectory_points:
                    from scipy.interpolate import interp1d
                    t_old = np.linspace(0, 1, len(deformed_demo_normalized))
                    t_new = np.linspace(0, 1, self.trajectory_points)
                    deformed_demo_normalized_resampled = []
                    for dim in range(6):
                        interp_func = interp1d(t_old, deformed_demo_normalized[:, dim], 
                                             kind='cubic', fill_value='extrapolate')
                        deformed_demo_normalized_resampled.append(interp_func(t_new))
                    deformed_demo_normalized = np.column_stack(deformed_demo_normalized_resampled)
            else:
                self.get_logger().warn('Demo normalization statistics not available, using raw deformed trajectory')
                deformed_demo_normalized = deformed_demo
            
            # Step 3: Add deformed trajectory as new demonstration
            self.demos.append(deformed_demo)  # Store in original scale
            demo_count = len(self.demos)
            self.get_logger().info(f'Deformed trajectory added as demonstration #{demo_count}')
            
            # Step 4: Initialize StepwiseEMLearner if not already initialized
            if self.stepwise_em_learner is None:
                self.get_logger().info('Initializing StepwiseEMLearner...')
                self.stepwise_em_learner = StepwiseEMLearner(
                    num_basis=self.num_basis,
                    sigma_noise=self.sigma_noise,
                    delta_N=0.1
                )
                
                # Initialize with first demo if available
                if len(self.demos) > 0:
                    # Normalize first demo
                    if self.demo_min is not None:
                        first_demo_normalized = (np.array(self.demos[0]) - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                    else:
                        first_demo_normalized = np.array(self.demos[0])
                    self.stepwise_em_learner.initialize_from_first_demo(first_demo_normalized)
                    self.get_logger().info('StepwiseEMLearner initialized with first demo')
            
            # Step 5: Update StepwiseEMLearner incrementally with deformed trajectory
            self.get_logger().info('Updating ProMP incrementally with deformed trajectory using StepwiseEM...')
            self.stepwise_em_learner.stepwise_em_update(deformed_demo_normalized)
            
            # Step 6: Update ProMP parameters from StepwiseEMLearner
            if self.stepwise_em_learner.mean_weights is not None:
                self.promp.mean_weights = self.stepwise_em_learner.mean_weights.copy()
                self.promp.cov_weights = self.stepwise_em_learner.cov_weights.copy()
                self.promp.basis_centers = self.stepwise_em_learner.basis_centers.copy()
                self.promp.basis_width = self.stepwise_em_learner.basis_width
                self.get_logger().info('ProMP parameters updated from StepwiseEMLearner')
            
            # Step 7: Generate new trajectory from updated ProMP
            num_points = len(current_trajectory)
            new_trajectory_normalized = self.promp.generate_trajectory(num_points=num_points)
            
            # Denormalize new trajectory
            if self.demo_min is not None and self.demo_max is not None:
                new_trajectory = new_trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                new_trajectory = np.clip(new_trajectory, self.demo_min, self.demo_max)
            else:
                new_trajectory = new_trajectory_normalized
            
            self.get_logger().info(f'Generated new trajectory from updated ProMP (shape: {new_trajectory.shape})')
            
            # Step 8: Update current trajectory and deformer
            self.current_trajectory = new_trajectory
            self.deformer.set_trajectory(new_trajectory)
            
            # Step 9: Interrupt current execution and restart with new trajectory
            trajectory_executing.clear()  # Signal to stop current trajectory
            
            # Send STOP to robot
            try:
                self.kuka_socket.sendall(b"STOP\n")
                self.get_logger().info('Sent STOP to robot to apply new trajectory from incremental learning')
                
                # Wait for STOPPED response
                self.kuka_socket.settimeout(2.0)
                try:
                    response = self._receive_complete_message(self.kuka_socket, timeout=2.0)
                    if response and "STOPPED" in response:
                        self.get_logger().info('Robot stopped successfully, restarting with new trajectory')
                    else:
                        self.get_logger().warn('Did not receive STOPPED confirmation, proceeding anyway')
                except socket.timeout:
                    self.get_logger().warn('Timeout waiting for STOPPED, proceeding with new trajectory')
                finally:
                    self.kuka_socket.settimeout(None)
            except Exception as e:
                self.get_logger().error(f'Error sending STOP: {e}')
            
            # Step 10: Extract trajectory segment from current time point onwards
            # Calculate the index in the new trajectory corresponding to t_current
            new_trajectory_start_idx = int(t_current * num_points)
            new_trajectory_start_idx = np.clip(new_trajectory_start_idx, 0, num_points - 1)
            
            # Get the remaining trajectory from the current time point
            trajectory_from_current = new_trajectory[new_trajectory_start_idx:]
            
            self.get_logger().info(f'Restarting execution from point {new_trajectory_start_idx}/{num_points} '
                                  f'(t={t_current:.3f}) with {len(trajectory_from_current)} remaining points')
            
            # Restart trajectory execution with remaining trajectory from current point
            execution_thread = threading.Thread(
                target=execute_func,
                args=(trajectory_from_current,)
            )
            execution_thread.daemon = True
            execution_thread.start()
            trajectory_executing.set()  # Reset flag
            
            # Set point count to the starting index (for tracking purposes)
            # Note: The actual execution will start from index 0 of trajectory_from_current,
            # but we track the absolute position in the full trajectory
            with point_count_lock:
                point_count[0] = new_trajectory_start_idx
            
            self.get_logger().info(f'Restarted trajectory execution from time point t={t_current:.3f} '
                                  f'with new trajectory from incremental learning (demo #{demo_count})')
            self.deformation_status_pub.publish(String(data=f'INCREMENTAL_LEARNING_COMPLETED:{demo_count}:t={t_current:.3f}'))
            
        except Exception as e:
            self.get_logger().error(f'Error during trajectory deformation and incremental learning: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def send_trajectory_to_kuka(self, trajectory):
        """Send full trajectory to KUKA robot and wait for completion (blocking)
        Converts Cartesian to joint positions using pybullet IK to avoid workspace errors
        Matches train_and_execute.py implementation"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Validate trajectory format
            traj_array = np.array(trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            # Convert Cartesian to joint positions using pybullet IK (matches train_and_execute.py)
            self.get_logger().info('Converting Cartesian trajectory to joint positions using pybullet IK...')
            joint_trajectory = self.cartesian_to_joint_via_java(trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions')
                self.get_logger().error('Please install pybullet: pip install pybullet')
                return False
            
            # Store joint trajectory
            self.current_joint_trajectory = joint_trajectory
            
            # Format joint trajectory for KUKA: j1,j2,j3,j4,j5,j6,j7 separated by semicolons
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
            ])
            
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.get_logger().info('Using joint positions avoids workspace errors - all points should be reachable')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Wait for completion using robust message receiving
            complete = False
            point_count = 0
            error_count = 0
            
            while not complete:
                response = self._receive_complete_message(self.kuka_socket, timeout=30.0)
                if not response:
                    self.get_logger().warn('No response from KUKA')
                    break
                
                response = response.strip()
                
                if "TRAJECTORY_COMPLETE" in response:
                    self.get_logger().info(f'Trajectory execution completed. Points: {point_count}, Errors (skipped): {error_count}')
                    complete = True
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    if point_count % 10 == 0:
                        self.get_logger().info(f'Progress: {point_count}/{len(joint_trajectory)} points completed')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors: {error_count}')
                return point_count > 0  # Return True if some progress made
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def send_trajectory_to_kuka_with_interrupt_and_progress(self, trajectory, interrupt_event, 
                                                           point_count, point_count_lock):
        """Send trajectory to KUKA robot and monitor execution, can be interrupted
        Also tracks execution progress via point_count"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Validate trajectory format
            traj_array = np.array(trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            # Convert Cartesian to joint positions using pybullet IK
            self.get_logger().info('Converting Cartesian trajectory to joint positions using pybullet IK...')
            joint_trajectory = self.cartesian_to_joint_via_java(trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions')
                return False
            
            # Store joint trajectory
            self.current_joint_trajectory = joint_trajectory
            
            # Format joint trajectory for KUKA
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
            ])
            
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Monitor execution progress, checking for interrupt and tracking point count
            complete = False
            error_count = 0
            
            with point_count_lock:
                point_count[0] = 0  # Reset point count
            
            while not complete and interrupt_event.is_set():
                # Use shorter timeout to check interrupt more frequently
                response = self._receive_complete_message(self.kuka_socket, timeout=1.0)
                if not response:
                    # Timeout - check if we should continue
                    if not interrupt_event.is_set():
                        self.get_logger().info('Trajectory execution interrupted by conditioning request')
                        return False
                    continue
                
                response = response.strip()
                
                if "TRAJECTORY_COMPLETE" in response:
                    self.get_logger().info(f'Trajectory execution completed. Points: {point_count[0]}, Errors (skipped): {error_count}')
                    complete = True
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                elif "POINT_COMPLETE" in response:
                    with point_count_lock:
                        point_count[0] += 1
                    if point_count[0] % 10 == 0:
                        self.get_logger().info(f'Progress: {point_count[0]}/{len(joint_trajectory)} points completed')
            
            if not complete and not interrupt_event.is_set():
                self.get_logger().info('Trajectory execution interrupted')
                return False
            elif not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count[0]}, Errors: {error_count}')
                return point_count[0] > 0
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def send_trajectory_to_kuka_with_interrupt(self, trajectory, interrupt_event):
        """Send trajectory to KUKA robot and monitor execution, can be interrupted
        Similar to send_trajectory_to_kuka but checks interrupt_event to allow stopping"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Validate trajectory format
            traj_array = np.array(trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            # Convert Cartesian to joint positions using pybullet IK
            self.get_logger().info('Converting Cartesian trajectory to joint positions using pybullet IK...')
            joint_trajectory = self.cartesian_to_joint_via_java(trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions')
                self.get_logger().error('Please install pybullet: pip install pybullet')
                return False
            
            # Store joint trajectory
            self.current_joint_trajectory = joint_trajectory
            
            # Format joint trajectory for KUKA
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
            ])
            
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Monitor execution progress, checking for interrupt
            complete = False
            point_count = 0
            error_count = 0
            
            while not complete and interrupt_event.is_set():
                # Use shorter timeout to check interrupt more frequently
                response = self._receive_complete_message(self.kuka_socket, timeout=1.0)
                if not response:
                    # Timeout - check if we should continue
                    if not interrupt_event.is_set():
                        self.get_logger().info('Trajectory execution interrupted by deformation request')
                        return False
                    continue
                
                response = response.strip()
                
                if "TRAJECTORY_COMPLETE" in response:
                    self.get_logger().info(f'Trajectory execution completed. Points: {point_count}, Errors (skipped): {error_count}')
                    complete = True
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    if point_count % 10 == 0:
                        self.get_logger().info(f'Progress: {point_count}/{len(joint_trajectory)} points completed')
            
            if not complete and not interrupt_event.is_set():
                self.get_logger().info('Trajectory execution interrupted')
                return False
            elif not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors: {error_count}')
                return point_count > 0
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def send_trajectory_to_kuka_async(self, trajectory):
        """Send trajectory to KUKA robot without blocking (for real-time updates)
        Converts Cartesian to joint positions using pybullet IK"""
        def send_and_monitor():
            if self.kuka_socket is None:
                self.get_logger().error('No connection to KUKA robot')
                return
            
            try:
                # Validate trajectory format
                traj_array = np.array(trajectory)
                if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                    self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                    return
                
                # Convert Cartesian to joint positions using pybullet IK
                self.get_logger().info('Converting Cartesian trajectory to joint positions (async)...')
                joint_trajectory = self.cartesian_to_joint_via_java(trajectory)
                
                if joint_trajectory is None or len(joint_trajectory) == 0:
                    self.get_logger().error('Failed to convert trajectory to joint positions')
                    return
                
                # Store joint trajectory
                self.current_joint_trajectory = joint_trajectory
                
                # Format joint trajectory for KUKA
                trajectory_str = ";".join([
                    ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
                ])
                
                command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
                self.kuka_socket.sendall(command.encode('utf-8'))
                self.get_logger().info(f'Sent joint trajectory to KUKA ({len(joint_trajectory)} points) - monitoring in background')
                
                # Monitor responses in background (non-blocking for main loop)
                error_count = 0
                point_count = 0
                while True:
                    try:
                        response = self._receive_complete_message(self.kuka_socket, timeout=0.1)
                        if not response:
                            continue  # Timeout is expected - continue monitoring
                        
                        response = response.strip()
                        
                        if "TRAJECTORY_COMPLETE" in response:
                            self.get_logger().info(f'Trajectory completed: {point_count} points, {error_count} errors')
                            break
                        elif "ERROR" in response:
                            error_count += 1
                            self.get_logger().warn(f'Trajectory execution error: {response}')
                        elif "POINT_COMPLETE" in response:
                            point_count += 1
                    except socket.timeout:
                        # Timeout is expected - continue monitoring
                        continue
                    except Exception as e:
                        self.get_logger().error(f'Error monitoring trajectory: {e}')
                        break
            except Exception as e:
                self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
        
        # Start trajectory sending in separate thread
        thread = threading.Thread(target=send_and_monitor)
        thread.daemon = True
        thread.start()
    
    def publish_statistics(self):
        """Publish execution statistics"""
        stats = {
            'deformation_count': self.deformation_count,
            'conditioning_count': self.conditioning_count,
            'total_energy': self.total_energy,
            'average_energy': self.total_energy / max(self.deformation_count, 1)
        }
        
        self.statistics_pub.publish(String(data=json.dumps(stats)))
        self.get_logger().info(f'Execution statistics: {stats}')
    
    def save_promp(self, filename=None):
        """Save current ProMP state"""
        if self.promp is None:
            self.get_logger().warn('No ProMP to save')
            return
        
        if filename is None:
            filename = self.promp_file
        
        try:
            promp_data = {
                'mean_weights': self.promp.mean_weights,
                'cov_weights': self.promp.cov_weights,
                'basis_centers': self.promp.basis_centers,
                'basis_width': self.promp.basis_width
            }
            np.save(filename, promp_data)
            self.get_logger().info(f'ProMP saved to {filename}')
        except Exception as e:
            self.get_logger().error(f'Error saving ProMP: {e}')
    
    def save_execution_data_to_csv(self):
        """Save execution trajectory, joint torques, and external torques to CSV files"""
        try:
            # Create result directory if it doesn't exist
            os.makedirs(self.result_directory, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save execution trajectory
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
            
            # Save joint torques
            if len(self.joint_torque_log) > 0:
                joint_torque_file = os.path.join(self.result_directory, f'joint_torques_{timestamp}.csv')
                with open(joint_torque_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp_s', 'joint1_Nm', 'joint2_Nm', 'joint3_Nm', 'joint4_Nm', 'joint5_Nm', 'joint6_Nm', 'joint7_Nm'])
                    for entry in self.joint_torque_log:
                        row = [entry['timestamp']] + entry['joint_torques']
                        writer.writerow(row)
                self.get_logger().info(f'Saved joint torques to {joint_torque_file} ({len(self.joint_torque_log)} samples)')
            else:
                self.get_logger().warn('No joint torque data to save')
            
            # Save external torques
            if len(self.external_torque_log) > 0:
                external_torque_file = os.path.join(self.result_directory, f'external_torques_{timestamp}.csv')
                with open(external_torque_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp_s', 'force_x_N', 'force_y_N', 'force_z_N', 'torque_x_Nm', 'torque_y_Nm', 'torque_z_Nm'])
                    for entry in self.external_torque_log:
                        row = [entry['timestamp']] + entry['force'] + entry['torque']
                        writer.writerow(row)
                self.get_logger().info(f'Saved external torques to {external_torque_file} ({len(self.external_torque_log)} samples)')
            else:
                self.get_logger().warn('No external torque data to save')
                
        except Exception as e:
            self.get_logger().error(f'Error saving execution data to CSV: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Standalone Deformation Controller')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP (matches train_and_execute.py)')
    parser.add_argument('--trajectory-file', default=os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy'), help='Trajectory file path (default: ~/robotexecute/learned_trajectory.npy)')
    parser.add_argument('--promp-file', default='promp_model.npy', help='ProMP file path')
    parser.add_argument('--energy-threshold', type=float, default=0.5, help='Energy threshold for conditioning')
    parser.add_argument('--force-threshold', type=float, default=10.0, help='Force threshold for deformation')
    parser.add_argument('--torque-threshold', type=float, default=2.0, help='Torque threshold for deformation')
    
    args, _ = parser.parse_known_args()
    
    # Create node with parameters
    node = StandaloneDeformationController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
        node.save_promp()  # Save ProMP state before shutting down
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
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
from .airl import AIRL

class AIRLController(Node):
    def __init__(self):
        super().__init__('airl_controller')
        
        # Parameters
        self.declare_parameter('kuka_ip', '172.31.1.147')  # Match train_and_execute.py
        self.declare_parameter('kuka_port', 30002)
        # Default trajectory file location
        default_trajectory = os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy')
        self.declare_parameter('trajectory_file', default_trajectory)
        self.declare_parameter('airl_model_file', 'airl_model.npy')
        self.declare_parameter('auto_start', True)  # Auto-start execution on initialization
        
        # Training parameters
        self.declare_parameter('save_directory', '~/robot_demos')
        self.declare_parameter('demo_file', '')  # Empty means use save_directory/all_demos.npy
        self.declare_parameter('trajectory_points', 100)
        self.declare_parameter('train_on_startup', True)  # Train AIRL on startup
        self.declare_parameter('execute_after_training', True)  # Execute trajectory after training
        
        # AIRL training parameters
        self.declare_parameter('airl_hidden_dim', 64)
        self.declare_parameter('airl_learning_rate', 0.001)
        self.declare_parameter('airl_gamma', 0.99)
        self.declare_parameter('airl_num_iterations', 1000)
        self.declare_parameter('airl_batch_size', 32)
        self.declare_parameter('use_torch', True)  # Use PyTorch if available
        
        # ROS2 topic parameters for external joint torques (if using iiwa_stack)
        self.declare_parameter('external_joint_torque_topic', '/iiwa/jointExternalTorque')
        self.declare_parameter('use_ros2_joint_torques', False)  # Default: False (use TCP socket)
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.airl_model_file = self.get_parameter('airl_model_file').value
        self.auto_start = self.get_parameter('auto_start').value
        
        # Training parameters
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        demo_file_param = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        self.train_on_startup = self.get_parameter('train_on_startup').value
        self.execute_after_training = self.get_parameter('execute_after_training').value
        
        # AIRL parameters
        self.airl_hidden_dim = self.get_parameter('airl_hidden_dim').value
        self.airl_learning_rate = self.get_parameter('airl_learning_rate').value
        self.airl_gamma = self.get_parameter('airl_gamma').value
        self.airl_num_iterations = self.get_parameter('airl_num_iterations').value
        self.airl_batch_size = self.get_parameter('airl_batch_size').value
        self.use_torch = self.get_parameter('use_torch').value
        
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
        self.airl = None
        self.current_trajectory = None
        self.current_joint_trajectory = None  # Joint space trajectory
        self.demos = []  # Demonstrations for training
        
        # Normalization statistics (for denormalizing generated trajectories)
        self.demo_min = None
        self.demo_max = None
        self.demo_mean = None
        self.demo_std = None
        
        # State
        self.is_executing = False
        self.execution_thread = None
        self.joint_torque_data = deque(maxlen=1000)  # Joint torques for each joint (7 joints) - for logging only
        
        # TCP communication
        self.kuka_socket = None
        self.torque_socket = None
        
        # CSV logging data structures
        self.execution_trajectory_log = []  # Track final execution trajectory (Cartesian)
        self.joint_torque_log = []  # Track all joint torques during execution (for logging only)
        self.result_directory = os.path.join(os.path.expanduser('~/result'), 'airl_controller')
        
        # Thread lock for pybullet IK (pybullet is not thread-safe)
        self.pybullet_lock = threading.Lock()
        
        # Setup communication
        self.setup_communication()
        
        # Setup publishers and subscribers
        self.setup_ros_communication()
        
        # Step 1: Train AIRL from demonstrations (if enabled)
        if self.train_on_startup:
            self.get_logger().info('Training AIRL from demonstrations on startup...')
            if self.train_and_generate_trajectory():
                self.get_logger().info('AIRL training and trajectory generation completed')
            else:
                self.get_logger().warn('AIRL training failed, trying to load existing trajectory...')
                self.load_trajectory_and_airl()
        else:
            # Load existing trajectory and AIRL model
            self.load_trajectory_and_airl()
        
        # Step 2: Execute trajectory (if enabled and trajectory is available)
        if self.execute_after_training and self.current_trajectory is not None:
            self.get_logger().info('Auto-execute enabled: Starting trajectory execution...')
            # Give a small delay to ensure everything is initialized
            threading.Timer(1.0, self.start_execution).start()
        elif self.auto_start and self.current_trajectory is not None:
            self.get_logger().info('Auto-start enabled: Starting trajectory execution...')
            # Give a small delay to ensure everything is initialized
            threading.Timer(1.0, self.start_execution).start()
        
        self.get_logger().info('AIRL Controller initialized')
    
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
            
            # Setup server for receiving joint torque data (for logging only)
            # Note: AIRL controller doesn't use torques for trajectory updates, only for logging
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', 30003))
            self.torque_socket.listen(1)
            
            # Start joint torque data thread (for logging only)
            self.torque_thread = threading.Thread(target=self.receive_joint_torque_data)
            self.torque_thread.daemon = True
            self.torque_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup communication: {e}')
            self.kuka_socket = None
    
    def _receive_complete_message(self, sock, timeout=5.0, buffer_size=8192):
        """
        Receive complete message from socket, handling multi-packet messages.
        Assumes messages end with newline character.
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
        self.execution_status_pub = self.create_publisher(String, 'execution_status', 10)
        self.training_status_pub = self.create_publisher(String, 'training_status', 10)
        
        # Subscribers
        self.start_execution_sub = self.create_subscription(
            Bool, 'start_execution', self.start_execution_callback, 10)
        self.stop_execution_sub = self.create_subscription(
            Bool, 'stop_execution', self.stop_execution_callback, 10)
        
        # Subscribe to joint torques from ROS2 topic (for logging only - not used for trajectory updates)
        if self.use_ros2_joint_torques:
            if HAS_IIWA_MSGS:
                self.external_joint_torque_sub = self.create_subscription(
                    JointTorque, self.external_joint_torque_topic, 
                    self.external_joint_torque_callback, 10)
                self.get_logger().info(f'Subscribed to joint torques via ROS2 topic: {self.external_joint_torque_topic} (for logging only)')
            else:
                self.external_joint_torque_sub = self.create_subscription(
                    JointState, self.external_joint_torque_topic,
                    self.external_joint_torque_callback_jointstate, 10)
                self.get_logger().info(f'Subscribed to joint torques via ROS2 topic: {self.external_joint_torque_topic} (for logging only)')
        else:
            self.get_logger().info('Using TCP socket for joint torques (for logging only - Java application will send JOINT_TORQUE: messages)')
    
    def external_joint_torque_callback(self, msg):
        """Callback for external joint torques from ROS2 topic (using iiwa_msgs/JointTorque)"""
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
            
            self.joint_torque_data.append({
                'timestamp': timestamp,
                'joint_torques': joint_torques
            })
        except Exception as e:
            self.get_logger().error(f'Error processing external joint torque message: {e}')
    
    def external_joint_torque_callback_jointstate(self, msg):
        """Callback for external joint torques from ROS2 topic (using sensor_msgs/JointState)"""
        try:
            if not hasattr(msg, 'effort') or len(msg.effort) < 7:
                return
            
            joint_torques = list(msg.effort[:7])
            timestamp = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            self.joint_torque_data.append({
                'timestamp': timestamp,
                'joint_torques': joint_torques
            })
        except Exception as e:
            self.get_logger().error(f'Error processing JointState message: {e}')
    
    def receive_joint_torque_data(self):
        """Receive joint torque data from KUKA robot via TCP socket (for logging only)
        AIRL controller doesn't use torques for trajectory updates"""
        try:
            conn, addr = self.torque_socket.accept()
            self.get_logger().info(f'Joint torque data connection from {addr}')
            
            while True:
                data = conn.recv(2048)
                if not data:
                    break
                    
                lines = data.decode('utf-8').split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            # Only parse joint torque data (format: JOINT_TORQUE:timestamp,j1,j2,j3,j4,j5,j6,j7)
                            if line.startswith('JOINT_TORQUE:'):
                                joint_data = line.replace('JOINT_TORQUE:', '').strip()
                                values = [float(x) for x in joint_data.split(',')]
                                if len(values) >= 8:  # timestamp + 7 joints
                                    timestamp = values[0]
                                    joint_torques = values[1:8]  # 7 joint torques
                                    self.joint_torque_data.append({
                                        'timestamp': timestamp,
                                        'joint_torques': joint_torques
                                    })
                            # Ignore external force/torque data - not needed for AIRL controller
                        except ValueError as e:
                            self.get_logger().debug(f'Error parsing joint torque data line: {line[:50]}... Error: {e}')
                            continue
                            
        except Exception as e:
            self.get_logger().error(f'Error receiving joint torque data: {e}')
    
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
        """Normalize demonstrations to same length and compute statistics"""
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
        
        return normalized
    
    def train_airl(self):
        """Train AIRL on loaded demonstrations"""
        if len(self.demos) < 1:
            self.get_logger().error('No demonstrations available for training')
            return False
        
        try:
            self.get_logger().info('Normalizing demonstrations...')
            normalized_demos = self.normalize_demos()
            
            self.get_logger().info('Training AIRL...')
            self.airl = AIRL(
                state_dim=6,  # Cartesian pose: x, y, z, alpha, beta, gamma
                action_dim=6,  # Cartesian velocities
                hidden_dim=self.airl_hidden_dim,
                learning_rate=self.airl_learning_rate,
                gamma=self.airl_gamma,
                use_torch=self.use_torch
            )
            
            # Set logger for AIRL
            self.airl.logger = self.get_logger()
            
            # Train AIRL on normalized demonstrations
            self.airl.train(
                normalized_demos,
                num_iterations=self.airl_num_iterations,
                batch_size=self.airl_batch_size
            )
            
            self.get_logger().info('AIRL training completed successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error training AIRL: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def generate_trajectory(self):
        """Generate trajectory from trained AIRL and denormalize"""
        if self.airl is None:
            self.get_logger().error('AIRL not trained yet')
            return None
        
        try:
            # Get initial state from first demo (or use mean)
            if len(self.demos) > 0:
                initial_state = self.demos[0][0]  # First point of first demo
            else:
                initial_state = self.demo_mean if self.demo_mean is not None else np.zeros(6)
            
            # Normalize initial state if needed
            if self.demo_min is not None and self.demo_max is not None:
                initial_state_normalized = (initial_state - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            else:
                initial_state_normalized = initial_state
            
            # Generate trajectory using AIRL (pass demonstrations for better generation)
            normalized_demos = []
            if len(self.demos) > 0:
                # Normalize demos for trajectory generation
                for demo in self.demos:
                    if self.demo_min is not None and self.demo_max is not None:
                        demo_normalized = (demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
                    else:
                        demo_normalized = demo
                    normalized_demos.append(demo_normalized)
            
            trajectory_normalized = self.airl.generate_trajectory(
                initial_state_normalized,
                num_points=self.trajectory_points,
                dt=0.01,
                demonstrations=normalized_demos if len(normalized_demos) > 0 else None
            )
            
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
        """Train AIRL and generate trajectory (complete training pipeline)"""
        # Step 1: Load demonstrations
        if not self.load_demos():
            return False
        
        # Step 2: Train AIRL
        if not self.train_airl():
            return False
        
        # Step 3: Generate trajectory
        trajectory = self.generate_trajectory()
        if trajectory is None:
            return False
        
        self.get_logger().info('Training and trajectory generation completed successfully')
        return True
    
    def load_trajectory_and_airl(self):
        """Load trajectory and AIRL model from files"""
        try:
            # Try to load trajectory - check multiple locations
            trajectory_loaded = False
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
                        self.get_logger().info(f'Loaded trajectory from {traj_path}')
                        trajectory_loaded = True
                        break
                    except Exception as e:
                        self.get_logger().warn(f'Failed to load trajectory from {traj_path}: {e}')
                        continue
            
            if not trajectory_loaded:
                self.get_logger().error(f'Trajectory file not found!')
                self.get_logger().error('Please run training first or specify --trajectory-file')
                self.current_trajectory = None
                return
            
            # Load AIRL model (if available)
            airl_paths = [
                self.airl_model_file,
                os.path.join(os.path.expanduser('~/robotexecute'), 'airl_model.npy'),
                'airl_model.npy',
            ]
            
            airl_loaded = False
            for airl_path in airl_paths:
                if os.path.exists(airl_path):
                    try:
                        # Note: AIRL model loading would need to be implemented based on the model format
                        self.get_logger().info(f'AIRL model file found at {airl_path} (loading not yet implemented)')
                        airl_loaded = True
                        break
                    except Exception as e:
                        self.get_logger().warn(f'Failed to load AIRL model from {airl_path}: {e}')
                        continue
            
            if not airl_loaded:
                self.get_logger().warn('AIRL model file not found - will need to retrain')
                self.airl = None
                
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory/AIRL: {e}')
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
    
    def start_execution(self):
        """Start execution of the trajectory"""
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
        self.get_logger().info('Started trajectory execution')
    
    def execution_loop(self):
        """Main execution loop: execute trajectory"""
        trajectory = np.copy(self.current_trajectory)
        
        # Initialize CSV logging data structures
        self.execution_trajectory_log = []
        self.joint_torque_log = []
        
        # Create result directory
        os.makedirs(self.result_directory, exist_ok=True)
        self.get_logger().info(f'CSV logging enabled. Results will be saved to: {self.result_directory}')
        
        self.get_logger().info('Starting trajectory execution...')
        self.execution_status_pub.publish(String(data='EXECUTION_STARTED'))
        
        # Monitor joint torques during execution in a separate thread (for logging only)
        def monitor_joint_torques():
            while self.is_executing:
                # Log joint torques (for logging purposes only - not used for trajectory updates)
                if len(self.joint_torque_data) > 0:
                    latest = self.joint_torque_data[-1]
                    self.joint_torque_log.append({
                        'timestamp': latest['timestamp'],
                        'joint_torques': latest['joint_torques'].copy()
                    })
                
                time.sleep(0.01)  # 100 Hz
        
        monitor_thread = threading.Thread(target=monitor_joint_torques)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Send trajectory to robot
        success = self.send_trajectory_to_kuka(trajectory)
        
        if success:
            self.get_logger().info('Trajectory execution completed successfully')
            self.execution_status_pub.publish(String(data='EXECUTION_COMPLETED'))
        else:
            self.get_logger().warn('Trajectory execution had errors')
            self.execution_status_pub.publish(String(data='EXECUTION_FAILED'))
        
        self.is_executing = False
        
        # Save CSV files
        self.save_execution_data_to_csv()
    
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
        
        # Save CSV files
        self.save_execution_data_to_csv()
    
    def send_trajectory_to_kuka(self, trajectory):
        """Send full trajectory to KUKA robot and wait for completion (blocking)
        Converts Cartesian to joint positions using pybullet IK to avoid workspace errors"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Validate trajectory format
            traj_array = np.array(trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            # Log trajectory info before IK conversion
            self.get_logger().info(f'Trajectory shape: {traj_array.shape}')
            self.get_logger().info(f'Trajectory range - X: [{traj_array[:, 0].min():.3f}, {traj_array[:, 0].max():.3f}], '
                                  f'Y: [{traj_array[:, 1].min():.3f}, {traj_array[:, 1].max():.3f}], '
                                  f'Z: [{traj_array[:, 2].min():.3f}, {traj_array[:, 2].max():.3f}]')
            self.get_logger().info(f'First point: {traj_array[0]}')
            self.get_logger().info(f'Last point: {traj_array[-1]}')
            
            # Convert Cartesian to joint positions using pybullet IK
            self.get_logger().info('Converting Cartesian trajectory to joint positions using pybullet IK...')
            joint_trajectory = self.cartesian_to_joint_via_java(trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions')
                self.get_logger().error('Please install pybullet: pip install pybullet')
                return False
            
            # Log joint trajectory info
            self.get_logger().info(f'Joint trajectory shape: {joint_trajectory.shape}')
            self.get_logger().info(f'Joint trajectory range - J1: [{joint_trajectory[:, 0].min():.3f}, {joint_trajectory[:, 0].max():.3f}], '
                                  f'J2: [{joint_trajectory[:, 1].min():.3f}, {joint_trajectory[:, 1].max():.3f}]')
            self.get_logger().info(f'First joint position: {joint_trajectory[0]}')
            self.get_logger().info(f'Last joint position: {joint_trajectory[-1]}')
            
            # Check if joint trajectory has variation (not all same values)
            joint_variation = np.std(joint_trajectory, axis=0)
            if np.any(joint_variation < 1e-6):
                self.get_logger().warn(f'Warning: Some joints have very little variation: {joint_variation}')
                self.get_logger().warn('This might indicate IK conversion issues or trajectory generation problems')
            
            # Store joint trajectory
            self.current_joint_trajectory = joint_trajectory
            
            # Format joint trajectory for KUKA: j1,j2,j3,j4,j5,j6,j7 separated by semicolons
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
            ])
            
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.get_logger().info('Using joint positions avoids workspace errors - all points should be reachable')
            self.get_logger().info(f'Command length: {len(command)} bytes')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Wait for completion - handle fragmented responses and skip errors (matches train_and_execute.py)
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
                    # Extract point number if available in error message
                    if point_count < len(joint_trajectory):
                        skipped_points.append(point_count)
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                    # Continue execution - skip this point and move to next
                    # Don't return False, just log and continue
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    # Log executed trajectory point
                    if point_count <= len(trajectory):
                        self.execution_trajectory_log.append(trajectory[point_count - 1].tolist())
                    if point_count % 10 == 0:  # Log every 10 points
                        self.get_logger().info(f'Progress: {point_count}/{len(joint_trajectory)} points completed (errors skipped: {error_count})')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors (skipped): {error_count}')
                # Log partial trajectory
                if point_count < len(trajectory):
                    self.execution_trajectory_log.extend(trajectory[:point_count].tolist())
                # Return True if we made some progress, False if nothing worked
                if point_count > 0:
                    self.get_logger().info(f'Partial execution completed with {point_count} successful points')
                    return True
                else:
                    return False
            else:
                # Log complete trajectory
                self.execution_trajectory_log = trajectory.tolist()
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def save_airl_model(self, filename=None):
        """Save current AIRL model"""
        if self.airl is None:
            self.get_logger().warn('No AIRL model to save')
            return
        
        if filename is None:
            filename = self.airl_model_file
        
        try:
            # Save AIRL model (simplified - would need proper serialization for PyTorch models)
            if self.airl.use_torch:
                self.get_logger().warn('PyTorch model saving not fully implemented - saving basic parameters')
                model_data = {
                    'state_dim': self.airl.state_dim,
                    'action_dim': self.airl.action_dim,
                    'hidden_dim': self.airl.hidden_dim,
                    'learning_rate': self.airl.learning_rate,
                    'gamma': self.airl.gamma,
                    'use_torch': True
                }
            else:
                model_data = {
                    'reward_weights': self.airl.reward_weights,
                    'state_dim': self.airl.state_dim,
                    'action_dim': self.airl.action_dim,
                    'use_torch': False
                }
            
            np.save(filename, model_data)
            self.get_logger().info(f'AIRL model saved to {filename}')
        except Exception as e:
            self.get_logger().error(f'Error saving AIRL model: {e}')
    
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
            
                
        except Exception as e:
            self.get_logger().error(f'Error saving execution data to CSV: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AIRL Controller')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP')
    parser.add_argument('--trajectory-file', default=os.path.join(os.path.expanduser('~/robotexecute'), 'learned_trajectory.npy'), help='Trajectory file path')
    parser.add_argument('--airl-model-file', default='airl_model.npy', help='AIRL model file path')
    
    args, _ = parser.parse_known_args()
    
    # Create node
    node = AIRLController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
        node.save_airl_model()  # Save AIRL model before shutting down
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

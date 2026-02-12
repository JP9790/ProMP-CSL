#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import socket
import time
import matplotlib.pyplot as plt
import os
from .airl import AIRL
import argparse
import math
import csv
from datetime import datetime
from collections import deque
import threading

class AIRLExecute(Node):
    def __init__(self):
        super().__init__('airl_execute')
        
        # Parameters
        # Note: kuka_ip should match the IP where Java app is running (typically robot controller IP)
        self.declare_parameter('kuka_ip', '172.31.1.147')  # Default matches controller_ip from data.xml
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('save_directory', '~/robot_demos')  # Default matches interactive_demo_recorder.py
        self.declare_parameter('demo_file', '')  # Empty means use save_directory/all_demos.npy
        self.declare_parameter('trajectory_points', 100)
        
        # AIRL training parameters
        self.declare_parameter('airl_num_iterations', 1000)
        self.declare_parameter('airl_batch_size', 32)
        self.declare_parameter('airl_learning_rate', 0.001)
        self.declare_parameter('airl_gamma', 0.99)
        self.declare_parameter('use_torch', True)  # Use PyTorch if available
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        demo_file_param = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        
        # AIRL parameters
        self.airl_num_iterations = self.get_parameter('airl_num_iterations').value
        self.airl_batch_size = self.get_parameter('airl_batch_size').value
        self.airl_learning_rate = self.get_parameter('airl_learning_rate').value
        self.airl_gamma = self.get_parameter('airl_gamma').value
        self.use_torch = self.get_parameter('use_torch').value
        
        # Set demo_file path - use save_directory if demo_file is empty or relative
        if demo_file_param:
            if os.path.isabs(demo_file_param):
                # Absolute path provided
                self.demo_file = demo_file_param
            else:
                # Relative path - check if it exists in current dir, otherwise use save_directory
                if os.path.exists(demo_file_param):
                    self.demo_file = demo_file_param
                else:
                    self.demo_file = os.path.join(self.save_directory, demo_file_param)
        else:
            # Default: use save_directory/all_demos.npy
            self.demo_file = os.path.join(self.save_directory, 'all_demos.npy')
        
        self.get_logger().info(f'Demo file path: {self.demo_file}')
        self.get_logger().info(f'Save directory: {self.save_directory}')
        
        # AIRL and trajectory
        self.airl = None
        self.learned_trajectory = None
        self.learned_joint_trajectory = None  # Joint space trajectory
        self.demos = []
        
        # Normalization statistics for denormalizing generated trajectories
        self.demo_min = None
        self.demo_max = None
        self.demo_mean = None
        self.demo_std = None
        
        # Trajectory save directory
        self.trajectory_save_dir = os.path.expanduser('~/newfoldername')
        
        # TCP communication
        self.kuka_socket = None
        self.torque_socket = None
        
        # CSV logging data structures
        self.execution_trajectory_log = []  # Track final execution trajectory (Cartesian)
        self.joint_torque_log = []  # Track all joint torques during execution
        self.external_torque_log = []  # Track all external torques during execution
        self.result_directory = os.path.join(os.path.expanduser('~/result'), 'airl_execute')
        
        # Torque data storage
        self.torque_data = deque(maxlen=1000)  # External force/torque from sensor
        self.joint_torque_data = deque(maxlen=1000)  # Joint torques for each joint (7 joints)
        
        # Setup communication
        self.setup_kuka_communication()
        
        # Setup torque data receiving (if Java app sends it)
        self.setup_torque_data_receiving()
        
        # Load demonstrations
        self.load_demos()
        
    def setup_kuka_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}:{self.kuka_port}...")
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.settimeout(5)  # 5 second timeout
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal
            ready = self._receive_complete_message(self.kuka_socket, timeout=5.0)
            if ready and ready.strip() == "READY":
                self.get_logger().info('KUKA connection established - received READY signal')
            else:
                self.get_logger().error(f'Unexpected response from KUKA: {ready}')
                self.kuka_socket = None
            
        except Exception as e:
            self.get_logger().error(f'Failed to connect to KUKA: {e}')
            self.kuka_socket = None
    
    def setup_torque_data_receiving(self):
        """Setup server for receiving torque data from Java application"""
        try:
            # Setup server for receiving torque data (Java sends to port 30003)
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', 30003))
            self.torque_socket.listen(1)
            
            # Start torque data thread
            self.torque_thread = threading.Thread(target=self.receive_torque_data)
            self.torque_thread.daemon = True
            self.torque_thread.start()
            self.get_logger().info('Torque data receiving server started on port 30003')
        except Exception as e:
            self.get_logger().warn(f'Failed to setup torque data receiving: {e}')
            self.torque_socket = None
    
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
                            
                            # Parse joint torque data from TCP (format: JOINT_TORQUE:timestamp,j1,j2,j3,j4,j5,j6,j7)
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
        First tries Java IK, then falls back to Python pybullet IK"""
        # Try Java IK first (if available)
        # For now, fall back directly to Python IK
        return self.cartesian_to_joint_python(cartesian_poses)
    
    def cartesian_to_joint_python(self, cartesian_poses):
        """Convert Cartesian poses to joint positions using pybullet IK solver"""
        try:
            import pybullet as p
            import pybullet_data
            
            self.get_logger().info('Using pybullet for IK computation (most reliable for KUKA)')
            
            # Initialize pybullet in DIRECT mode (no GUI, faster)
            physics_client = p.connect(p.DIRECT)
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
            
            p.disconnect()
            
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
            return None
    
    def _create_simple_kuka_model(self, p):
        """Create a simple 7-DOF KUKA LBR iiwa model for IK"""
        # This is a simplified model - for best results, use actual URDF
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=base_visual, baseCollisionShapeIndex=base_collision)
        self.get_logger().warn('Simplified model created - accuracy may be reduced. Use actual URDF for best results.')
        return base_body
    
    def load_demos(self):
        """Load demonstrations from file"""
        try:
            if not os.path.exists(self.demo_file):
                self.get_logger().error(f'Demo file not found: {self.demo_file}')
                self.get_logger().error('Please record demonstrations first using interactive_demo_recorder.py')
                self.get_logger().info('You can specify a different path using --demo-file argument')
                self.demos = []
                return
            
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
            elif isinstance(loaded_data, np.ndarray):
                if loaded_data.ndim == 3:
                    for i in range(loaded_data.shape[0]):
                        demos_list.append(loaded_data[i])
                elif loaded_data.ndim == 2 and loaded_data.shape[1] == 6:
                    demos_list.append(loaded_data)
            
            self.demos = demos_list
            self.get_logger().info(f'Loaded {len(self.demos)} demonstrations from {self.demo_file}')
            
        except Exception as e:
            self.get_logger().error(f'Error loading demos: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.demos = []
    
    def normalize_demos(self):
        """Normalize demonstrations to same length and compute statistics for denormalization"""
        if len(self.demos) == 0:
            return []
        
        target_length = self.trajectory_points
        normalized = []
        all_values = []  # Collect all values for statistics
        
        # First, interpolate all demos to same length
        for demo in self.demos:
            demo_array = np.array(demo)
            t_old = np.linspace(0, 1, len(demo))
            t_new = np.linspace(0, 1, target_length)
            
            normalized_demo = []
            for i in range(demo_array.shape[1]):  # For each dimension
                from scipy.interpolate import interp1d
                try:
                    interp_func = interp1d(t_old, demo_array[:, i], kind='cubic')
                    normalized_demo.append(interp_func(t_new))
                except ValueError:
                    # Fallback to linear if cubic fails
                    interp_func = interp1d(t_old, demo_array[:, i], kind='linear')
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
        
        # Normalize values to [0, 1] range for AIRL training
        normalized_scaled = []
        for demo in normalized:
            # Normalize: (value - min) / (max - min)
            demo_normalized = (demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            normalized_scaled.append(demo_normalized)
        
        # Debug: print shape of each normalized demo
        for i, demo in enumerate(normalized_scaled):
            self.get_logger().debug(f"Normalized demo {i} shape: {demo.shape}, range: [{np.min(demo):.3f}, {np.max(demo):.3f}]")
        
        return normalized_scaled
    
    def train_airl(self):
        """Train AIRL on loaded demonstrations"""
        if len(self.demos) < 1:
            self.get_logger().error('No demonstrations available for training')
            return False
        
        try:
            self.get_logger().info('Normalizing demonstrations...')
            normalized_demos = self.normalize_demos()
            
            if len(normalized_demos) == 0:
                self.get_logger().error('No normalized demonstrations available')
                return False
            
            self.get_logger().info('Training AIRL...')
            
            # Initialize AIRL
            self.airl = AIRL(
                state_dim=6,  # Cartesian pose: x, y, z, alpha, beta, gamma
                action_dim=6,  # Cartesian velocities
                hidden_dim=64,
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
            
            # Generate trajectory using AIRL
            # Note: generate_trajectory doesn't accept demonstrations parameter in current version
            # It will use the demonstrations stored during training if available
            trajectory_normalized = self.airl.generate_trajectory(
                initial_state_normalized,
                num_points=self.trajectory_points,
                dt=0.01
            )
            
            # Denormalize trajectory back to original scale
            if self.demo_min is not None and self.demo_max is not None:
                trajectory_denorm = trajectory_normalized * (self.demo_max - self.demo_min) + self.demo_min
                trajectory_denorm = np.clip(trajectory_denorm, self.demo_min, self.demo_max)
                self.learned_trajectory = trajectory_denorm
            else:
                self.learned_trajectory = trajectory_normalized
            
            self.get_logger().info(f'Generated trajectory shape: {self.learned_trajectory.shape}')
            return self.learned_trajectory
            
        except Exception as e:
            self.get_logger().error(f'Error generating trajectory: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def visualize_trajectories(self):
        """Visualize learned trajectory and demonstrations"""
        if self.learned_trajectory is None:
            self.get_logger().warn('No learned trajectory to visualize')
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            labels = ['X (m)', 'Y (m)', 'Z (m)', 'Alpha (rad)', 'Beta (rad)', 'Gamma (rad)']
            
            # Plot learned trajectory
            t_learned = np.linspace(0, 1, len(self.learned_trajectory))
            for i in range(6):
                axes[i].plot(t_learned, self.learned_trajectory[:, i], 'b-', linewidth=2, label='Learned (AIRL)')
            
            # Plot demonstrations
            for j, demo in enumerate(self.demos):
                demo_array = np.array(demo)
                t_demo = np.linspace(0, 1, len(demo_array))
                for i in range(6):
                    axes[i].plot(t_demo, demo_array[:, i], 'r--', alpha=0.3, linewidth=1)
            
            # Format plots
            for i in range(6):
                y_min = min(np.min(self.learned_trajectory[:, i]), 
                           min([np.min(demo[:, i]) for demo in self.demos] if self.demos else [0]))
                y_max = max(np.max(self.learned_trajectory[:, i]), 
                           max([np.max(demo[:, i]) for demo in self.demos] if self.demos else [0]))
                axes[i].set_title(f'{labels[i]} (range: [{y_min:.3f}, {y_max:.3f}])')
                axes[i].set_xlabel('Time Steps')
                axes[i].set_ylabel(labels[i])
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f'Error visualizing trajectories: {e}')
    
    def send_trajectory_to_kuka(self):
        """Send learned trajectory to KUKA robot (using joint positions to avoid workspace errors)"""
        if self.learned_trajectory is None:
            self.get_logger().error('No learned trajectory available')
            return False
        
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Validate trajectory format
            traj_array = np.array(self.learned_trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            # Convert Cartesian to joint positions using Python IK solver (pybullet)
            # This prevents workspace errors by computing valid joint positions before sending
            self.get_logger().info('Converting Cartesian trajectory to joint positions using pybullet IK...')
            joint_trajectory = self.cartesian_to_joint_via_java(self.learned_trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions - cannot proceed')
                self.get_logger().error('Please install pybullet: pip install pybullet')
                return False
            
            # Store joint trajectory
            self.learned_joint_trajectory = joint_trajectory
            
            # Format joint trajectory for KUKA: j1,j2,j3,j4,j5,j6,j7 separated by semicolons
            # Format: "JOINT_TRAJECTORY:j1_1,j2_1,j3_1,j4_1,j5_1,j6_1,j7_1;j1_2,j2_2,..."
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
            ])
            
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.get_logger().info('Using joint positions avoids workspace errors - all points should be reachable')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Initialize CSV logging
            self.execution_trajectory_log = []
            self.joint_torque_log = []
            self.external_torque_log = []
            
            # Create result directory
            os.makedirs(self.result_directory, exist_ok=True)
            self.get_logger().info(f'CSV logging enabled. Results will be saved to: {self.result_directory}')
            
            # Wait for completion - handle fragmented responses and skip errors
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
                    
                    # Log complete trajectory
                    self.execution_trajectory_log = self.learned_trajectory.tolist()
                    
                    # Save CSV files after completion
                    self.save_execution_data_to_csv()
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    # Extract point number if available in error message
                    if point_count < len(self.learned_trajectory):
                        skipped_points.append(point_count)
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                    # Continue execution - skip this point and move to next
                    # Don't return False, just log and continue
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    
                    # Log executed trajectory point
                    if point_count <= len(self.learned_trajectory):
                        self.execution_trajectory_log.append(self.learned_trajectory[point_count - 1].tolist())
                    
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
                    
                    if point_count % 10 == 0:  # Log every 10 points
                        self.get_logger().info(f'Progress: {point_count}/{len(self.learned_trajectory)} points completed (errors skipped: {error_count})')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors (skipped): {error_count}')
                
                # Log partial trajectory if execution was interrupted
                if point_count < len(self.learned_trajectory):
                    self.execution_trajectory_log.extend(self.learned_trajectory[:point_count].tolist())
                
                # Save CSV files even if execution didn't complete
                self.save_execution_data_to_csv()
                
                # Return True if we made some progress, False if nothing worked
                if point_count > 0:
                    self.get_logger().info(f'Partial execution completed with {point_count} successful points')
                    return True
                else:
                    return False
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Save CSV files even on error
            self.save_execution_data_to_csv()
            return False
    
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
    
    def save_learned_trajectory(self, filename=None):
        """Save learned trajectory to ~/newfoldername"""
        if self.learned_trajectory is None:
            self.get_logger().warn('No learned trajectory to save')
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            
            # Save Cartesian trajectory
            if filename is None:
                cartesian_file = os.path.join(self.trajectory_save_dir, 'learned_trajectory_airl.npy')
            else:
                cartesian_file = os.path.join(self.trajectory_save_dir, filename)
            
            np.save(cartesian_file, self.learned_trajectory)
            self.get_logger().info(f'Learned Cartesian trajectory saved to {cartesian_file}')
            
            # Save joint trajectory if available
            if self.learned_joint_trajectory is not None:
                joint_file = os.path.join(self.trajectory_save_dir, 'learned_joint_trajectory_airl.npy')
                np.save(joint_file, self.learned_joint_trajectory)
                self.get_logger().info(f'Learned joint trajectory saved to {joint_file}')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error saving learned trajectory: {e}')
            return False
    
    def run_complete_pipeline(self):
        """Run complete pipeline: train AIRL and execute"""
        self.get_logger().info('Starting complete AIRL pipeline...')
        
        # Step 1: Train AIRL
        if not self.train_airl():
            self.get_logger().error('AIRL training failed')
            return False
        
        # Step 2: Generate trajectory
        trajectory = self.generate_trajectory()
        if trajectory is None:
            self.get_logger().error('Trajectory generation failed')
            return False
        
        # Step 3: Visualize
        self.visualize_trajectories()
        
        # Step 4: Save trajectory
        self.save_learned_trajectory()
        
        # Step 5: Execute on KUKA
        success = self.send_trajectory_to_kuka()
        
        if success:
            self.get_logger().info('Complete pipeline executed successfully')
        else:
            self.get_logger().error('Pipeline failed at execution step')
        
        return success

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AIRL Execute')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP')
    parser.add_argument('--kuka-port', type=int, default=30002, help='KUKA robot port')
    parser.add_argument('--num-basis', type=int, default=50, help='Number of basis functions')
    parser.add_argument('--sigma-noise', type=float, default=0.01, help='Observation noise')
    parser.add_argument('--save-directory', default='~/robot_demos', help='Directory for saving/loading demos')
    parser.add_argument('--demo-file', default='', help='Demo file path (empty = use save_directory/all_demos.npy)')
    parser.add_argument('--trajectory-points', type=int, default=100, help='Number of trajectory points')
    parser.add_argument('--airl-num-iterations', type=int, default=1000, help='Number of AIRL training iterations')
    parser.add_argument('--airl-batch-size', type=int, default=32, help='Batch size for AIRL training')
    parser.add_argument('--airl-learning-rate', type=float, default=0.001, help='Learning rate for AIRL')
    parser.add_argument('--airl-gamma', type=float, default=0.99, help='Discount factor for AIRL')
    parser.add_argument('--no-torch', action='store_true', help='Disable PyTorch (use numpy fallback)')
    parser.add_argument('--no-visualize', action='store_true', help='Skip trajectory visualization')
    parser.add_argument('--no-execute', action='store_true', help='Skip robot execution')
    
    args, _ = parser.parse_known_args()
    
    # Create node
    node = AIRLExecute()
    
    # Override parameters from command line if provided
    if args.kuka_ip:
        node.kuka_ip = args.kuka_ip
    if args.kuka_port:
        node.kuka_port = args.kuka_port
    if args.num_basis:
        node.num_basis = args.num_basis
    if args.sigma_noise:
        node.sigma_noise = args.sigma_noise
    if args.save_directory:
        node.save_directory = os.path.expanduser(args.save_directory)
    if args.demo_file:
        node.demo_file = args.demo_file
    if args.trajectory_points:
        node.trajectory_points = args.trajectory_points
    if args.airl_num_iterations:
        node.airl_num_iterations = args.airl_num_iterations
    if args.airl_batch_size:
        node.airl_batch_size = args.airl_batch_size
    if args.airl_learning_rate:
        node.airl_learning_rate = args.airl_learning_rate
    if args.airl_gamma:
        node.airl_gamma = args.airl_gamma
    if args.no_torch:
        node.use_torch = False
    
    try:
        # Run complete pipeline
        if not args.no_execute:
            success = node.run_complete_pipeline()
        else:
            # Just train and generate, don't execute
            if node.train_airl():
                node.generate_trajectory()
                if not args.no_visualize:
                    node.visualize_trajectories()
                node.save_learned_trajectory()
                success = True
            else:
                success = False
        
        if success:
            node.get_logger().info('AIRL pipeline completed successfully')
        else:
            node.get_logger().error('AIRL pipeline failed')
        
        # Keep node alive for a bit to ensure CSV files are saved
        time.sleep(1.0)
        
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
        # Save CSV files before exiting
        node.save_execution_data_to_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import socket
import time
import matplotlib.pyplot as plt
import os
from .promp import ProMP
import argparse
import math
import csv
from datetime import datetime
from collections import deque
import threading

class TrainAndExecuteCartesian(Node):
    def __init__(self):
        super().__init__('train_and_execute_cartesian')
        
        # Parameters
        self.declare_parameter('kuka_ip', '172.31.1.147')
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('save_directory', '~/robot_demos')
        self.declare_parameter('demo_file', '')
        self.declare_parameter('trajectory_points', 100)
        self.declare_parameter('validate_with_ik', True)  # Validate Cartesian poses with IK before sending
        self.declare_parameter('skip_invalid_points', True)  # Skip points that exceed axis limits
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        demo_file_param = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        self.validate_with_ik = self.get_parameter('validate_with_ik').value
        self.skip_invalid_points = self.get_parameter('skip_invalid_points').value
        
        # Set demo_file path
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
        
        self.get_logger().info(f'Demo file path: {self.demo_file}')
        self.get_logger().info(f'Save directory: {self.save_directory}')
        
        # ProMP and trajectory
        self.promp = None
        self.learned_trajectory = None
        self.demos = []
        
        # Normalization statistics
        self.demo_min = None
        self.demo_max = None
        self.demo_mean = None
        self.demo_std = None
        
        # Trajectory save directory
        self.trajectory_save_dir = os.path.expanduser('~/robotexecute')
        
        # TCP communication
        self.kuka_socket = None
        self.torque_socket = None
        
        # CSV logging
        self.execution_trajectory_log = []
        self.joint_torque_log = []
        self.external_torque_log = []
        self.result_directory = os.path.join(os.path.expanduser('~/result'), 'train_and_execute_cartesian')
        
        # Torque data storage
        self.torque_data = deque(maxlen=1000)
        self.joint_torque_data = deque(maxlen=1000)
        
        # Thread lock for pybullet IK validation
        self.pybullet_lock = threading.Lock()
        
        # Setup communication
        self.setup_kuka_communication()
        self.setup_torque_data_receiving()
        
        # Load demonstrations
        self.load_demos()
    
    def setup_kuka_communication(self):
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
        except Exception as e:
            self.get_logger().error(f'Failed to connect to KUKA: {e}')
            self.kuka_socket = None
    
    def setup_torque_data_receiving(self):
        """Setup server for receiving torque data"""
        try:
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', 30003))
            self.torque_socket.listen(1)
            
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
    
    def _receive_complete_message(self, sock, timeout=5.0, buffer_size=8192):
        """Receive complete message from socket"""
        try:
            sock.settimeout(timeout)
            message_parts = []
            while True:
                data = sock.recv(buffer_size)
                if not data:
                    break
                message_parts.append(data.decode('utf-8'))
                if b'\n' in data:
                    break
            return ''.join(message_parts)
        except socket.timeout:
            return None
        except Exception as e:
            self.get_logger().debug(f"Error receiving message: {e}")
            return None
    
    def validate_cartesian_trajectory_with_ik(self, cartesian_trajectory):
        """Validate Cartesian trajectory using IK - filter out points that exceed axis limits"""
        if not self.validate_with_ik:
            return cartesian_trajectory, []
        
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
                    self.get_logger().warn('Failed to connect to pybullet for validation - sending trajectory without validation')
                    return cartesian_trajectory, []
                
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
                    self.get_logger().warn('KUKA URDF not found - sending trajectory without validation')
                    p.disconnect()
                    return cartesian_trajectory, []
                
                num_joints = p.getNumJoints(robot_id)
                end_effector_link = num_joints - 1
                
                validated_trajectory = []
                invalid_indices = []
                initial_joints = [0.0, 0.7854, 0.0, -1.3962, 0.0, -0.6109, 0.0]
                current_joints = initial_joints.copy()
                
                # Joint limits for KUKA LBR iiwa 7 DOF
                joint_limits = [
                    (-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967),
                    (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094), (-3.054, 3.054)
                ]
                
                self.get_logger().info(f'Validating {len(cartesian_trajectory)} Cartesian poses with IK...')
                
                for i, pose in enumerate(cartesian_trajectory):
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
                            
                            # Check joint limits
                            valid = True
                            for j, (angle, (min_val, max_val)) in enumerate(zip(joint_angles_7, joint_limits)):
                                if angle < min_val or angle > max_val:
                                    valid = False
                                    break
                            
                            if valid:
                                validated_trajectory.append(pose)
                                current_joints = joint_angles_7.copy()
                            else:
                                invalid_indices.append(i)
                                if not self.skip_invalid_points:
                                    # Use previous valid joint configuration for next IK
                                    if len(validated_trajectory) > 0:
                                        validated_trajectory.append(validated_trajectory[-1])
                                    else:
                                        validated_trajectory.append(pose)  # Keep first point even if invalid
                        else:
                            invalid_indices.append(i)
                            if not self.skip_invalid_points:
                                if len(validated_trajectory) > 0:
                                    validated_trajectory.append(validated_trajectory[-1])
                                else:
                                    validated_trajectory.append(pose)
                    except Exception as e:
                        invalid_indices.append(i)
                        if not self.skip_invalid_points:
                            if len(validated_trajectory) > 0:
                                validated_trajectory.append(validated_trajectory[-1])
                            else:
                                validated_trajectory.append(pose)
                
                p.disconnect()
                
                if invalid_indices:
                    self.get_logger().warn(f'Found {len(invalid_indices)} invalid poses (exceed axis limits): {invalid_indices[:10]}' + ('...' if len(invalid_indices) > 10 else ''))
                    if self.skip_invalid_points:
                        self.get_logger().info(f'Filtered trajectory: {len(validated_trajectory)}/{len(cartesian_trajectory)} valid poses')
                    else:
                        self.get_logger().info(f'Kept invalid poses (replaced with previous valid pose): {len(validated_trajectory)} poses')
                else:
                    self.get_logger().info(f'All {len(cartesian_trajectory)} poses are valid (within axis limits)')
                
                return np.array(validated_trajectory), invalid_indices
                
            except ImportError:
                self.get_logger().warn('pybullet not installed - sending trajectory without validation')
                return cartesian_trajectory, []
            except Exception as e:
                self.get_logger().warn(f'IK validation failed: {e} - sending trajectory without validation')
                try:
                    import pybullet as p
                    p.disconnect()
                except:
                    pass
                return cartesian_trajectory, []
    
    def load_demos(self):
        """Load demonstrations from file"""
        try:
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
            
        except Exception as e:
            self.get_logger().error(f'Error loading demonstrations: {e}')
            self.demos = []
    
    def normalize_demos(self):
        """Normalize demonstrations to same length and compute statistics"""
        if len(self.demos) == 0:
            return []
        
        from scipy.interpolate import interp1d
        
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
            self.get_logger().error('No demonstrations available for training')
            return False
        
        try:
            normalized_demos = self.normalize_demos()
            
            self.promp = ProMP(num_basis=self.num_basis, sigma_noise=self.sigma_noise)
            self.promp.train(normalized_demos)
            
            self.get_logger().info('ProMP training completed')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error training ProMP: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def generate_trajectory(self, num_samples=1):
        """Generate trajectory from trained ProMP"""
        if self.promp is None:
            self.get_logger().error('ProMP not trained yet')
            return None
        
        try:
            if num_samples == 1:
                trajectory_normalized = self.promp.generate_trajectory(num_points=self.trajectory_points)
            else:
                trajectories = []
                for _ in range(num_samples):
                    traj = self.promp.generate_trajectory(num_points=self.trajectory_points)
                    trajectories.append(traj)
                trajectory_normalized = np.mean(trajectories, axis=0)
            
            # Denormalize trajectory
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
    
    def send_trajectory_to_kuka(self):
        """Send learned Cartesian trajectory to KUKA robot (validates with IK to avoid axis limits)"""
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
            
            # Validate Cartesian trajectory with IK to filter out points exceeding axis limits
            validated_trajectory, invalid_indices = self.validate_cartesian_trajectory_with_ik(traj_array)
            
            if len(validated_trajectory) == 0:
                self.get_logger().error('No valid poses after IK validation - cannot execute')
                return False
            
            # Format Cartesian trajectory for KUKA: x,y,z,alpha,beta,gamma separated by semicolons
            # Format: "TRAJECTORY:x1,y1,z1,alpha1,beta1,gamma1;x2,y2,z2,alpha2,beta2,gamma2;..."
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in validated_trajectory
            ])
            
            command = f"TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending Cartesian trajectory to KUKA ({len(validated_trajectory)} points)...')
            if invalid_indices:
                self.get_logger().info(f'Note: {len(invalid_indices)} invalid poses were filtered out')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Initialize CSV logging
            self.execution_trajectory_log = []
            self.joint_torque_log = []
            self.external_torque_log = []
            
            # Create result directory
            os.makedirs(self.result_directory, exist_ok=True)
            self.get_logger().info(f'CSV logging enabled. Results will be saved to: {self.result_directory}')
            
            # Wait for completion
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
                    
                    # Save CSV files after completion
                    self.save_execution_data_to_csv()
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    if point_count < len(validated_trajectory):
                        skipped_points.append(point_count)
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    
                    # Log executed trajectory point
                    if point_count <= len(validated_trajectory):
                        self.execution_trajectory_log.append(validated_trajectory[point_count - 1].tolist())
                    
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
                        self.get_logger().info(f'Progress: {point_count}/{len(validated_trajectory)} points completed (errors skipped: {error_count})')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors (skipped): {error_count}')
                if point_count < len(validated_trajectory):
                    self.execution_trajectory_log.extend(validated_trajectory[:point_count].tolist())
                
                self.save_execution_data_to_csv()
                
                if point_count > 0:
                    self.get_logger().info(f'Partial execution completed with {point_count} successful points')
                    return True
                else:
                    return False
            else:
                self.execution_trajectory_log = validated_trajectory.tolist()
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.save_execution_data_to_csv()
            return False
    
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
    
    def save_learned_trajectory(self):
        """Save learned trajectory to file"""
        if self.learned_trajectory is None:
            self.get_logger().warn('No learned trajectory to save')
            return
        
        try:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            traj_file = os.path.join(self.trajectory_save_dir, 'learned_trajectory_cartesian.npy')
            np.save(traj_file, self.learned_trajectory)
            self.get_logger().info(f'Saved learned Cartesian trajectory to {traj_file}')
        except Exception as e:
            self.get_logger().error(f'Error saving trajectory: {e}')
    
    def run_complete_pipeline(self):
        """Run complete pipeline: train ProMP, generate trajectory, validate, and execute"""
        self.get_logger().info('Starting complete pipeline...')
        
        # Train ProMP
        if not self.train_promp():
            self.get_logger().error('ProMP training failed')
            return False
        
        # Generate trajectory
        trajectory = self.generate_trajectory()
        if trajectory is None:
            self.get_logger().error('Trajectory generation failed')
            return False
        
        # Save learned trajectory
        self.save_learned_trajectory()
        
        # Execute trajectory
        success = self.send_trajectory_to_kuka()
        
        if success:
            self.get_logger().info('Complete pipeline finished successfully')
        else:
            self.get_logger().error('Trajectory execution failed')
        
        return success

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Train ProMP and Execute Cartesian Trajectory')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP')
    parser.add_argument('--kuka-port', type=int, default=30002, help='KUKA robot port')
    parser.add_argument('--demo-file', default='', help='Demo file path')
    parser.add_argument('--save-directory', default='~/robot_demos', help='Save directory for demos')
    parser.add_argument('--num-basis', type=int, default=50, help='Number of basis functions')
    parser.add_argument('--sigma-noise', type=float, default=0.01, help='Noise sigma')
    parser.add_argument('--trajectory-points', type=int, default=100, help='Number of trajectory points')
    parser.add_argument('--validate-with-ik', action='store_true', default=True, help='Validate trajectory with IK before sending')
    parser.add_argument('--skip-invalid-points', action='store_true', default=True, help='Skip points that exceed axis limits')
    parser.add_argument('--no-execute', action='store_true', help='Train and generate but do not execute')
    
    args, _ = parser.parse_known_args()
    
    node = TrainAndExecuteCartesian()
    
    try:
        if args.no_execute:
            # Just train and generate
            if node.train_promp():
                node.generate_trajectory()
                node.save_learned_trajectory()
                node.get_logger().info('Training and generation completed (no execution)')
        else:
            # Run complete pipeline
            node.run_complete_pipeline()
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

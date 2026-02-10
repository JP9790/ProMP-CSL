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

class TrainAndExecute(Node):
    def __init__(self):
        super().__init__('train_and_execute')
        
        # Parameters
        # Note: kuka_ip should match the IP where Java app is running (typically robot controller IP)
        self.declare_parameter('kuka_ip', '172.31.1.147')  # Default matches controller_ip from data.xml
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('save_directory', '~/robot_demos')  # Default matches interactive_demo_recorder.py
        self.declare_parameter('demo_file', '')  # Empty means use save_directory/all_demos.npy
        self.declare_parameter('trajectory_points', 100)
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        demo_file_param = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        
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
        
        # ProMP and trajectory
        self.promp = None
        self.learned_trajectory = None
        self.demos = []
        
        # Normalization statistics for denormalizing generated trajectories
        self.demo_min = None
        self.demo_max = None
        self.demo_mean = None
        self.demo_std = None
        
        # TCP communication
        self.kuka_socket = None
        
        # Setup communication
        self.setup_kuka_communication()
        
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
    
    def load_demos(self):
        """Load demonstrations from file and flatten any 3D arrays into a list of 2D arrays.
        
        Handles formats from interactive_demo_recorder.py:
        - Object array of lists (each demo is a list of [x,y,z,alpha,beta,gamma] lists)
        - Each demo should be shape (N, 6) where N is number of poses
        """
        try:
            loaded_data = np.load(self.demo_file, allow_pickle=True)
            demos_list = []
            
            # Handle object arrays (from interactive_demo_recorder.py)
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                for demo in loaded_data:
                    arr = np.array(demo)
                    if arr.ndim == 2 and arr.shape[1] == 6:
                        # Valid demo: (N, 6) array
                        demos_list.append(arr)
                    elif arr.ndim == 1 and len(arr) == 6:
                        # Single pose, wrap in array
                        demos_list.append(arr.reshape(1, -1))
                    elif arr.ndim == 3:
                        # Flatten 3D array into list of 2D arrays
                        for i in range(arr.shape[0]):
                            if arr[i].shape[1] == 6:
                                demos_list.append(arr[i])
                    else:
                        self.get_logger().warn(f'Skipping demo with unexpected shape: {arr.shape}')
                self.demos = demos_list
            elif isinstance(loaded_data, np.ndarray):
                if loaded_data.ndim == 3:
                    # 3D array: (num_demos, num_points, 6)
                    for i in range(loaded_data.shape[0]):
                        demos_list.append(loaded_data[i])
                    self.demos = demos_list
                elif loaded_data.ndim == 2:
                    # Single demo: (num_points, 6)
                    if loaded_data.shape[1] == 6:
                        self.demos = [loaded_data]
                    else:
                        self.get_logger().error(f'Unexpected demo shape: {loaded_data.shape}')
                        self.demos = []
                else:
                    self.demos = [loaded_data]
            elif isinstance(loaded_data, list):
                # Already a list
                self.demos = [np.array(demo) for demo in loaded_data if len(np.array(demo).shape) == 2]
            else:
                self.get_logger().error(f'Unexpected demo file format: {type(loaded_data)}')
                self.demos = []
            
            self.get_logger().info(f'Loaded {len(self.demos)} demonstrations from {self.demo_file}')
            # Debug: print shape of each demo
            for i, demo in enumerate(self.demos):
                demo_arr = np.array(demo)
                if demo_arr.ndim == 2 and demo_arr.shape[1] == 6:
                    self.get_logger().info(f"Demo {i}: {demo_arr.shape[0]} poses, shape: {demo_arr.shape}")
                else:
                    self.get_logger().warn(f"Demo {i} has unexpected shape: {demo_arr.shape}")
                    
        except FileNotFoundError:
            self.get_logger().error(f'Demo file not found: {self.demo_file}')
            self.get_logger().info('Make sure you have recorded demos using interactive_demo_recorder.py first')
            self.get_logger().info(f'Expected location: {self.demo_file}')
            self.get_logger().info(f'Save directory: {self.save_directory}')
            self.get_logger().info('You can specify a different path using --demo-file argument')
            self.demos = []
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
        
        # Normalize values to [0, 1] range for ProMP training
        normalized_scaled = []
        for demo in normalized:
            # Normalize: (value - min) / (max - min)
            demo_normalized = (demo - self.demo_min) / (self.demo_max - self.demo_min + 1e-10)
            normalized_scaled.append(demo_normalized)
        
        # Debug: print shape of each normalized demo
        for i, demo in enumerate(normalized_scaled):
            self.get_logger().debug(f"Normalized demo {i} shape: {demo.shape}, range: [{np.min(demo):.3f}, {np.max(demo):.3f}]")
        
        return normalized_scaled
    
    def denormalize_trajectory(self, trajectory):
        """Denormalize trajectory from [0,1] range back to original demo range"""
        if self.demo_min is None or self.demo_max is None:
            self.get_logger().error('Cannot denormalize: demo statistics not computed')
            return trajectory
        
        # Denormalize: value * (max - min) + min
        trajectory_denorm = trajectory * (self.demo_max - self.demo_min) + self.demo_min
        return trajectory_denorm
    
    def train_promp(self):
        """Train ProMP on loaded demonstrations"""
        if len(self.demos) < 1:
            self.get_logger().error('No demonstrations available for training')
            return False
        
        try:
            self.get_logger().info('Normalizing demonstrations...')
            normalized_demos = self.normalize_demos()
            
            self.get_logger().info('Training ProMP...')
            self.promp = ProMP(num_basis=self.num_basis, sigma_noise=self.sigma_noise)
            self.promp.train(normalized_demos)
            
            self.get_logger().info('ProMP training completed successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error training ProMP: {e}')
            return False
    
    def generate_trajectory(self, num_samples=1):
        """Generate trajectory from trained ProMP and denormalize to original scale"""
        if self.promp is None:
            self.get_logger().error('ProMP not trained yet')
            return None
        
        try:
            if num_samples == 1:
                trajectory_normalized = self.promp.generate_trajectory(num_points=self.trajectory_points)
                self.get_logger().info(f'Generated single trajectory (normalized), shape: {trajectory_normalized.shape}')
            else:
                # Generate multiple trajectories for visualization
                trajectories = []
                for _ in range(num_samples):
                    traj = self.promp.generate_trajectory(num_points=self.trajectory_points)
                    trajectories.append(traj)
                trajectory_normalized = np.mean(trajectories, axis=0)
                self.get_logger().info(f'Generated {num_samples} trajectories and computed mean')
            
            # Denormalize trajectory back to original scale
            self.learned_trajectory = self.denormalize_trajectory(trajectory_normalized)
            
            self.get_logger().info(f'Trajectory denormalized, range:')
            self.get_logger().info(f'  X: [{np.min(self.learned_trajectory[:, 0]):.3f}, {np.max(self.learned_trajectory[:, 0]):.3f}] m')
            self.get_logger().info(f'  Y: [{np.min(self.learned_trajectory[:, 1]):.3f}, {np.max(self.learned_trajectory[:, 1]):.3f}] m')
            self.get_logger().info(f'  Z: [{np.min(self.learned_trajectory[:, 2]):.3f}, {np.max(self.learned_trajectory[:, 2]):.3f}] m')
            self.get_logger().info(f'  Alpha: [{np.min(self.learned_trajectory[:, 3]):.3f}, {np.max(self.learned_trajectory[:, 3]):.3f}] rad')
            self.get_logger().info(f'  Beta: [{np.min(self.learned_trajectory[:, 4]):.3f}, {np.max(self.learned_trajectory[:, 4]):.3f}] rad')
            self.get_logger().info(f'  Gamma: [{np.min(self.learned_trajectory[:, 5]):.3f}, {np.max(self.learned_trajectory[:, 5]):.3f}] rad')
            
            return self.learned_trajectory
            
        except Exception as e:
            self.get_logger().error(f'Error generating trajectory: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def visualize_trajectories(self):
        """Visualize demonstrations and learned trajectory"""
        if len(self.demos) == 0:
            self.get_logger().warn('No demonstrations to visualize')
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Alpha (rad)', 'Beta (rad)', 'Gamma (rad)']
        
        # Plot demonstrations (original scale)
        for i in range(6):
            for j, demo in enumerate(self.demos):
                demo_array = np.array(demo)
                # Interpolate demo to same length for visualization
                if len(demo_array) != self.trajectory_points:
                    from scipy.interpolate import interp1d
                    t_old = np.linspace(0, 1, len(demo_array))
                    t_new = np.linspace(0, 1, self.trajectory_points)
                    interp_func = interp1d(t_old, demo_array[:, i], kind='linear')
                    demo_interp = interp_func(t_new)
                else:
                    demo_interp = demo_array[:, i]
                axes[i].plot(demo_interp, 'b-', alpha=0.3, label='Demos' if j == 0 else "")
            
            # Plot learned trajectory (denormalized, original scale)
            if self.learned_trajectory is not None:
                axes[i].plot(self.learned_trajectory[:, i], 'r-', linewidth=2, label='Learned')
                # Show range
                y_min = np.min(self.learned_trajectory[:, i])
                y_max = np.max(self.learned_trajectory[:, i])
                axes[i].set_title(f'{labels[i]} (range: [{y_min:.3f}, {y_max:.3f}])')
            
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel(labels[i])
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def send_trajectory_to_kuka(self):
        """Send learned trajectory to KUKA robot"""
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
            
            # Format trajectory for KUKA: x,y,z,alpha,beta,gamma separated by semicolons
            # Format: "TRAJECTORY:x1,y1,z1,a1,b1,g1;x2,y2,z2,a2,b2,g2;..."
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in self.learned_trajectory
            ])
            
            command = f"TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending trajectory to KUKA ({len(self.learned_trajectory)} points)...')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
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
                    if point_count % 10 == 0:  # Log every 10 points
                        self.get_logger().info(f'Progress: {point_count}/{len(self.learned_trajectory)} points completed (errors skipped: {error_count})')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors (skipped): {error_count}')
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
            return False
    
    def save_learned_trajectory(self, filename='learned_trajectory.npy'):
        """Save learned trajectory to file"""
        if self.learned_trajectory is not None:
            np.save(filename, self.learned_trajectory)
            self.get_logger().info(f'Learned trajectory saved to {filename}')
    
    def load_learned_trajectory(self, filename='learned_trajectory.npy'):
        """Load learned trajectory from file"""
        try:
            self.learned_trajectory = np.load(filename)
            self.get_logger().info(f'Learned trajectory loaded from {filename}')
            return True
        except FileNotFoundError:
            self.get_logger().warn(f'No learned trajectory file found: {filename}')
            return False
    
    def run_complete_pipeline(self):
        """Run complete pipeline: train ProMP and execute"""
        self.get_logger().info('Starting complete ProMP pipeline...')
        
        # Step 1: Train ProMP
        if not self.train_promp():
            self.get_logger().error('ProMP training failed')
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
    parser = argparse.ArgumentParser(description='Train ProMP and execute trajectory')
    parser.add_argument('--demo-file', default='', help='Demo file path (default: ~/robot_demos/all_demos.npy from interactive_demo_recorder)')
    parser.add_argument('--save-directory', default='~/robot_demos', help='Directory where demos are saved (default: ~/robot_demos)')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP (default: 172.31.1.147)')
    parser.add_argument('--train-only', action='store_true', help='Only train ProMP, do not execute')
    parser.add_argument('--execute-only', action='store_true', help='Only execute, do not train')
    parser.add_argument('--visualize', action='store_true', help='Show trajectory visualization')
    parser.add_argument('--load-trajectory', default='', help='Load trajectory from file instead of training')
    
    args, _ = parser.parse_known_args()
    
    # Create node with parameters
    node = TrainAndExecute()
    
    # Override demo_file and save_directory if provided via command line
    if args.save_directory:
        node.save_directory = os.path.expanduser(args.save_directory)
        # Update demo_file path if it was using default
        if not args.demo_file:
            node.demo_file = os.path.join(node.save_directory, 'all_demos.npy')
    
    if args.demo_file:
        # Use provided demo file path
        if os.path.isabs(args.demo_file):
            node.demo_file = args.demo_file
        else:
            # Try current directory first, then save_directory
            if os.path.exists(args.demo_file):
                node.demo_file = args.demo_file
            else:
                node.demo_file = os.path.join(node.save_directory, args.demo_file)
        node.get_logger().info(f'Using demo file from command line: {node.demo_file}')
    
    try:
        if args.load_trajectory:
            # Load pre-trained trajectory
            if node.load_learned_trajectory(args.load_trajectory):
                if args.visualize:
                    node.visualize_trajectories()
                if not args.train_only:
                    node.send_trajectory_to_kuka()
        elif args.execute_only:
            # Load and execute without training
            if node.load_learned_trajectory():
                if args.visualize:
                    node.visualize_trajectories()
                node.send_trajectory_to_kuka()
        elif args.train_only:
            # Train only
            if node.train_promp():
                node.generate_trajectory()
                node.save_learned_trajectory()
                if args.visualize:
                    node.visualize_trajectories()
        else:
            # Run complete pipeline
            node.run_complete_pipeline()
            
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
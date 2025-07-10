#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import socket
import time
import matplotlib.pyplot as plt
from .promp import ProMP
import argparse

class TrainAndExecute(Node):
    def __init__(self):
        super().__init__('train_and_execute')
        
        # Parameters
        self.declare_parameter('kuka_ip', '192.170.10.25')
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('demo_file', 'demos.npy')
        self.declare_parameter('trajectory_points', 100)
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.demo_file = self.get_parameter('demo_file').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        
        # ProMP and trajectory
        self.promp = None
        self.learned_trajectory = None
        self.demos = []
        
        # TCP communication
        self.kuka_socket = None
        
        # Setup communication
        self.setup_kuka_communication()
        
        # Load demonstrations
        self.load_demos()
        
    def setup_kuka_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal
            ready = self.kuka_socket.recv(1024).decode('utf-8')
            self.get_logger().info(f'KUKA connection established: {ready}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to connect to KUKA: {e}')
            self.kuka_socket = None
    
    def load_demos(self):
        """Load demonstrations from file and flatten any 3D arrays into a list of 2D arrays."""
        try:
            self.demos = np.load(self.demo_file, allow_pickle=True)
            demos_list = []
            if isinstance(self.demos, np.ndarray) and self.demos.dtype == object:
                for demo in self.demos:
                    arr = np.array(demo)
                    if arr.ndim == 3:
                        # Flatten 3D array into list of 2D arrays
                        for i in range(arr.shape[0]):
                            demos_list.append(arr[i])
                    else:
                        demos_list.append(arr)
                self.demos = demos_list
            elif isinstance(self.demos, np.ndarray):
                if self.demos.ndim == 3:
                    for i in range(self.demos.shape[0]):
                        demos_list.append(self.demos[i])
                    self.demos = demos_list
                else:
                    self.demos = [self.demos]
            self.get_logger().info(f'Loaded {len(self.demos)} demonstrations from {self.demo_file}')
            # Debug: print shape of each demo
            for i, demo in enumerate(self.demos):
                print(f"Demo {i} type: {type(demo)}, shape: {np.array(demo).shape}")
        except FileNotFoundError:
            self.get_logger().error(f'Demo file not found: {self.demo_file}')
            self.demos = []
        except Exception as e:
            self.get_logger().error(f'Error loading demos: {e}')
            self.demos = []
    
    def normalize_demos(self):
        """Normalize demonstrations to same length using interpolation"""
        if len(self.demos) == 0:
            return []
        
        target_length = self.trajectory_points
        normalized = []
        
        for demo in self.demos:
            demo_array = np.array(demo)
            t_old = np.linspace(0, 1, len(demo))
            t_new = np.linspace(0, 1, target_length)
            
            normalized_demo = []
            for i in range(demo_array.shape[1]):  # For each dimension
                from scipy.interpolate import interp1d
                interp_func = interp1d(t_old, demo_array[:, i], kind='cubic')
                normalized_demo.append(interp_func(t_new))
            
            normalized.append(np.column_stack(normalized_demo))
        
        # Debug: print shape of each normalized demo
        for i, demo in enumerate(normalized):
            print(f"Normalized demo {i} shape: {demo.shape}")
        return normalized
    
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
        """Generate trajectory from trained ProMP"""
        if self.promp is None:
            self.get_logger().error('ProMP not trained yet')
            return None
        
        try:
            if num_samples == 1:
                self.learned_trajectory = self.promp.generate_trajectory()
                self.get_logger().info('Generated single trajectory')
            else:
                # Generate multiple trajectories for visualization
                trajectories = []
                for _ in range(num_samples):
                    traj = self.promp.generate_trajectory()
                    trajectories.append(traj)
                self.learned_trajectory = np.mean(trajectories, axis=0)
                self.get_logger().info(f'Generated {num_samples} trajectories and computed mean')
            
            return self.learned_trajectory
            
        except Exception as e:
            self.get_logger().error(f'Error generating trajectory: {e}')
            return None
    
    def visualize_trajectories(self):
        """Visualize demonstrations and learned trajectory"""
        if len(self.demos) == 0:
            self.get_logger().warn('No demonstrations to visualize')
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Alpha (rad)', 'Beta (rad)', 'Gamma (rad)']
        
        # Plot demonstrations
        for i in range(6):
            for j, demo in enumerate(self.demos):
                demo_array = np.array(demo)
                axes[i].plot(demo_array[:, i], 'b-', alpha=0.3, label='Demos' if j == 0 else "")
            
            # Plot learned trajectory
            if self.learned_trajectory is not None:
                axes[i].plot(self.learned_trajectory[:, i], 'r-', linewidth=2, label='Learned')
            
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
            # Format trajectory for KUKA
            trajectory_str = ";".join([
                ",".join(map(str, point)) for point in self.learned_trajectory
            ])
            
            command = f"TRAJECTORY:{trajectory_str}"
            self.get_logger().info('Sending trajectory to KUKA...')
            self.kuka_socket.sendall((command + "\n").encode('utf-8'))
            
            # Wait for completion
            while True:
                response = self.kuka_socket.recv(1024).decode('utf-8')
                self.get_logger().info(f'KUKA response: {response}')
                
                if "TRAJECTORY_COMPLETE" in response:
                    self.get_logger().info('Trajectory executed successfully on KUKA')
                    return True
                elif "ERROR" in response:
                    self.get_logger().error(f'KUKA execution error: {response}')
                    return False
                elif "POINT_COMPLETE" in response:
                    self.get_logger().debug('Trajectory point completed')
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
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
    parser.add_argument('--demo-file', default='demos.npy', help='Demo file path')
    parser.add_argument('--kuka-ip', default='192.170.10.25', help='KUKA robot IP')
    parser.add_argument('--train-only', action='store_true', help='Only train ProMP, do not execute')
    parser.add_argument('--execute-only', action='store_true', help='Only execute, do not train')
    parser.add_argument('--visualize', action='store_true', help='Show trajectory visualization')
    parser.add_argument('--load-trajectory', default='', help='Load trajectory from file instead of training')
    
    args, _ = parser.parse_known_args()
    
    # Create node with parameters
    node = TrainAndExecute()
    
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
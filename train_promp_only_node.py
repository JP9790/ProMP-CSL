#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from promp import ProMP

class ProMPTrainerNode(Node):
    def __init__(self):
        super().__init__('promp_trainer_node')
        # Declare parameters
        self.declare_parameter('demo_file', 'all_demos.npy')
        self.declare_parameter('trajectory_file', 'learned_trajectory.npy')
        self.declare_parameter('promp_file', 'promp_model.npy')
        self.declare_parameter('num_basis', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('trajectory_points', 100)
        # Get parameters
        self.demo_file = self.get_parameter('demo_file').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.promp_file = self.get_parameter('promp_file').value
        self.num_basis = self.get_parameter('num_basis').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.trajectory_points = self.get_parameter('trajectory_points').value
        # Run training
        self.train_and_visualize()

    def train_and_visualize(self):
        # Load demos
        demos = np.load(self.demo_file, allow_pickle=True)
        demos_list = []
        if isinstance(demos, np.ndarray) and demos.dtype == object:
            for demo in demos:
                arr = np.array(demo)
                if arr.ndim == 3:
                    for i in range(arr.shape[0]):
                        demos_list.append(arr[i])
                else:
                    demos_list.append(arr)
            demos = demos_list
        elif isinstance(demos, np.ndarray):
            if demos.ndim == 3:
                for i in range(demos.shape[0]):
                    demos_list.append(demos[i])
                demos = demos_list
            else:
                demos = [demos]
        self.get_logger().info(f"Loaded {len(demos)} demos.")
        # Normalize demos
        target_length = self.trajectory_points
        normalized = []
        for demo in demos:
            demo_array = np.array(demo)
            t_old = np.linspace(0, 1, len(demo_array))
            t_new = np.linspace(0, 1, target_length)
            normalized_demo = []
            for i in range(demo_array.shape[1]):
                from scipy.interpolate import interp1d
                interp_func = interp1d(t_old, demo_array[:, i], kind='cubic')
                normalized_demo.append(interp_func(t_new))
            normalized.append(np.column_stack(normalized_demo))
        normalized = np.stack(normalized, axis=0)  # (num_demos, target_length, dof)
        self.get_logger().info(f"Normalized demos shape: {normalized.shape}")
        # Train ProMP
        promp = ProMP(num_basis=self.num_basis, sigma_noise=self.sigma_noise)
        promp.train(normalized)
        # Generate learned trajectory
        learned_traj = promp.generate_trajectory(num_points=target_length)
        self.get_logger().info(f"Learned trajectory shape: {learned_traj.shape}")
        # Save learned trajectory
        np.save(self.trajectory_file, learned_traj)
        self.get_logger().info(f"Saved learned trajectory to {self.trajectory_file}")
        # Save ProMP parameters
        promp_data = {
            'mean_weights': promp.mean_weights,
            'cov_weights': promp.cov_weights,
            'basis_centers': promp.basis_centers,
            'basis_width': promp.basis_width
        }
        np.save(self.promp_file, promp_data)
        self.get_logger().info(f"Saved ProMP model to {self.promp_file}")
        # Visualization
        self.visualize(normalized, learned_traj, promp)

    def visualize(self, demos, learned_traj, promp):
        num_demos, T, dof = demos.shape
        t = np.linspace(0, 1, T)
        fig, axes = plt.subplots(dof, 1, figsize=(10, 2 * dof), sharex=True)
        if dof == 1:
            axes = [axes]
        for d in range(dof):
            # Plot all demos
            for i in range(num_demos):
                axes[d].plot(t, demos[i, :, d], color='gray', alpha=0.4)
            # Plot learned trajectory
            axes[d].plot(t, learned_traj[:, d], color='blue', label='Learned Trajectory', linewidth=2)
            # Plot mean and std (distribution)
            # Sample many trajectories to estimate mean/std
            samples = []
            for _ in range(100):
                sample = promp.generate_trajectory(num_points=T)
                samples.append(sample[:, d])
            samples = np.stack(samples, axis=0)
            mean = np.mean(samples, axis=0)
            std = np.std(samples, axis=0)
            axes[d].plot(t, mean, color='red', linestyle='--', label='ProMP Mean')
            axes[d].fill_between(t, mean - std, mean + std, color='red', alpha=0.2, label='ProMP ±1 std')
            axes[d].set_ylabel(f'Dim {d+1}')
            axes[d].legend()
            axes[d].grid(True)
        axes[-1].set_xlabel('Normalized Time')
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = ProMPTrainerNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
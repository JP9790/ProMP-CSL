#!/usr/bin/env python3

import numpy as np
import copy
import matplotlib.pyplot as plt

class TrajectoryDeformer:
    def __init__(self, alpha=0.1, n_waypoints=10, energy_threshold=0.5):
        """
        Initialize trajectory deformer
        
        Args:
            alpha: Deformation sensitivity parameter
            n_waypoints: Number of waypoints to deform
            energy_threshold: Threshold for deformation energy
        """
        self.alpha = alpha
        self.n = n_waypoints
        self.energy_threshold = energy_threshold
        
        # Current state
        self.curr_waypt_idx = 0
        self.waypts = None
        self.original_trajectory = None
        self.deformed_trajectory = None
        self.num_waypts = 0
        
        # Deformation matrix D (minimum jerk) - computed from finite differencing matrix A
        # D = (A^T A)^(-1) where A is the finite differencing matrix
        self.D = None  # Will be computed when trajectory is set
        
        # Energy tracking
        self.deformation_energy = 0.0
        self.energy_history = []
        
    def _create_finite_differencing_matrix(self, n):
        """
        Create finite differencing matrix A for minimum jerk trajectories.
        
        A is a (N+3) × N matrix where:
        - For column j: A[j, j] = 1, A[j+1, j] = -3, A[j+2, j] = 3, A[j+3, j] = -1
        - All other elements are 0
        
        This matrix represents third-order finite differences.
        
        Args:
            n: Number of waypoints (N)
            
        Returns:
            A: Finite differencing matrix of shape (n+3, n)
        """
        A = np.zeros((n + 3, n))
        
        # Fill the matrix according to the pattern: 1, -3, 3, -1
        for j in range(n):
            if j < n:
                A[j, j] = 1.0
            if j + 1 < n + 3:
                A[j + 1, j] = -3.0
            if j + 2 < n + 3:
                A[j + 2, j] = 3.0
            if j + 3 < n + 3:
                A[j + 3, j] = -1.0
        
        return A
    
    def _compute_deformation_matrix_D(self, n):
        """
        Compute deformation matrix D = (A^T A)^(-1) for minimum jerk trajectories.
        
        Args:
            n: Number of waypoints
            
        Returns:
            D: Deformation matrix of shape (n, n)
        """
        # Create finite differencing matrix A
        A = self._create_finite_differencing_matrix(n)
        
        # Compute D = (A^T A)^(-1)
        ATA = np.dot(A.T, A)
        
        # Check if matrix is invertible
        try:
            D = np.linalg.inv(ATA)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            print('Warning: ATA matrix is singular, using pseudo-inverse')
            D = np.linalg.pinv(ATA)
        
        return D
    
    def set_trajectory(self, trajectory):
        """Set the trajectory to be deformed"""
        self.waypts = copy.deepcopy(trajectory)
        self.original_trajectory = copy.deepcopy(trajectory)
        self.num_waypts = len(trajectory)
        self.curr_waypt_idx = 0
        self.deformed_trajectory = None
        self.deformation_energy = 0.0
        self.energy_history = []
        
        # Compute deformation matrix D for the deformation region size
        # D will be recomputed if n_waypoints changes, but typically it's fixed
        self.D = self._compute_deformation_matrix_D(self.n)
        
    def deform(self, u_h):
        """
        Deform the next n waypoints based on human force input
        
        Args:
            u_h: Human force/torque input [fx, fy, fz, tx, ty, tz] or [x, y, z, alpha, beta, gamma]
            
        Returns:
            tuple: (deformed_trajectory, original_trajectory, deformation_energy)
        """
        deform_waypt_idx = self.curr_waypt_idx + 1
        
        # Check if we can deform (enough waypoints remaining)
        if (deform_waypt_idx + self.n) > self.num_waypts:
            return (None, None, 0.0)
        
        # Store original trajectory
        waypts_prev = copy.deepcopy(self.waypts)
        waypts_deform = copy.deepcopy(self.waypts)
        
        # Ensure u_h has 6 dimensions (cartesian + orientation)
        if len(u_h) != 6:
            raise ValueError("Human input u_h must have 6 dimensions")
        
        # Ensure D matrix is computed
        if self.D is None:
            self.D = self._compute_deformation_matrix_D(self.n)
        
        # Apply deformation formula: S*_d = S_d + μ * D * f_h(t_c)
        # where μ = self.alpha, D is the minimum jerk matrix, f_h = u_h
        # D is (n × n), f_h is (6,), result is (n × 6)
        # For each dimension, compute: gamma[:, dim] = alpha * D * (f_h[dim] * ones(n))
        # This applies the force f_h[dim] uniformly across the n waypoints, then transforms via D
        gamma = np.zeros((self.n, 6))
        ones_vector = np.ones(self.n)  # Vector of ones for uniform force application
        for dim in range(6):
            # D is (n × n), f_h[dim] * ones(n) is (n,), so D * (f_h[dim] * ones(n)) gives (n,)
            # Then multiply by alpha: gamma[:, dim] = alpha * D * (f_h[dim] * ones(n))
            gamma[:, dim] = self.alpha * np.dot(self.D, u_h[dim] * ones_vector)
        
        # Apply deformation to the region
        waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx, :] += gamma
        
        # Calculate deformation energy
        energy = self._calculate_deformation_energy(
            waypts_prev[deform_waypt_idx : self.n + deform_waypt_idx],
            waypts_deform[deform_waypt_idx : self.n + deform_waypt_idx]
        )
        
        # Store deformed trajectory
        self.deformed_trajectory = waypts_deform
        self.deformation_energy = energy
        self.energy_history.append(energy)
        
        return (waypts_deform, waypts_prev, energy)
    
    def _calculate_deformation_energy(self, original_region, deformed_region):
        """
        Calculate deformation energy as area between original and deformed trajectories
        
        Args:
            original_region: Original trajectory region
            deformed_region: Deformed trajectory region
            
        Returns:
            float: Deformation energy
        """
        if len(original_region) != len(deformed_region):
            return 0.0
        
        # Calculate area between curves for each dimension
        total_energy = 0.0
        time_steps = np.linspace(0, 1, len(original_region))
        
        for dim in range(6):
            # Calculate area between original and deformed curves using numpy's trapz
            area = np.trapz(
                np.abs(deformed_region[:, dim] - original_region[:, dim]),
                time_steps
            )
            total_energy += area
        
        return total_energy
    
    def should_condition_promp(self):
        """Check if ProMP conditioning should be triggered"""
        return self.deformation_energy < self.energy_threshold
    
    def get_deformation_region(self):
        """Get the current deformation region indices"""
        deform_waypt_idx = self.curr_waypt_idx + 1
        return range(deform_waypt_idx, min(deform_waypt_idx + self.n, self.num_waypts))
    
    def advance_waypoint(self):
        """Advance to next waypoint"""
        self.curr_waypt_idx += 1
        if self.curr_waypt_idx >= self.num_waypts:
            return False
        return True
    
    def reset_deformation(self):
        """Reset deformation state"""
        self.deformed_trajectory = None
        self.deformation_energy = 0.0
        self.energy_history = []
    
    def visualize_deformation(self, save_path=None):
        """Visualize original vs deformed trajectory"""
        if self.deformed_trajectory is None:
            print("No deformation to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Alpha (rad)', 'Beta (rad)', 'Gamma (rad)']
        
        for i in range(6):
            # Plot original trajectory
            axes[i].plot(self.original_trajectory[:, i], 'b-', label='Original', linewidth=2)
            
            # Plot deformed trajectory
            axes[i].plot(self.deformed_trajectory[:, i], 'r--', label='Deformed', linewidth=2)
            
            # Highlight deformation region
            deform_region = self.get_deformation_region()
            if len(deform_region) > 0:
                axes[i].axvspan(deform_region[0], deform_region[-1], alpha=0.2, color='yellow', label='Deformation Region')
            
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel(labels[i])
            axes[i].legend()
            axes[i].grid(True)
        
        plt.suptitle(f'Deformation Energy: {self.deformation_energy:.4f}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
#!/usr/bin/env python3

import numpy as np
import copy
from scipy.integrate import trapz
from scipy.spatial.distance import cdist
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
        
        # Deformation matrix H (smooth transition)
        self.H = self._create_deformation_matrix()
        
        # Energy tracking
        self.deformation_energy = 0.0
        self.energy_history = []
        
    def _create_deformation_matrix(self):
        """Create smooth deformation matrix H"""
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            # Smooth transition: start small, peak in middle, end small
            if i < self.n // 2:
                H[i, i] = (i + 1) / (self.n // 2)
            else:
                H[i, i] = (self.n - i) / (self.n // 2)
        return H
    
    def set_trajectory(self, trajectory):
        """Set the trajectory to be deformed"""
        self.waypts = copy.deepcopy(trajectory)
        self.original_trajectory = copy.deepcopy(trajectory)
        self.num_waypts = len(trajectory)
        self.curr_waypt_idx = 0
        self.deformed_trajectory = None
        self.deformation_energy = 0.0
        self.energy_history = []
        
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
        
        # Calculate deformation for each dimension
        gamma = np.zeros((self.n, 6))
        for dim in range(6):
            gamma[:, dim] = self.alpha * np.dot(self.H, u_h[dim])
        
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
            # Calculate area between original and deformed curves
            area = trapz(
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
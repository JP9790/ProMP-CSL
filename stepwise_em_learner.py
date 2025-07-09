#!/usr/bin/env python3

import numpy as np
import copy
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class StepwiseEMLearner:
    def __init__(self, num_basis=50, sigma_noise=0.01, learning_rate=0.1):
        """
        Initialize Stepwise EM Learner
        
        Args:
            num_basis: Number of basis functions for ProMP
            sigma_noise: Noise parameter for ProMP
            learning_rate: Learning rate for EM updates
        """
        self.num_basis = num_basis
        self.sigma_noise = sigma_noise
        self.learning_rate = learning_rate
        
        # ProMP parameters
        self.mean_weights = None
        self.cov_weights = None
        self.basis_centers = None
        self.basis_width = None
        
        # EM state
        self.current_u = None  # Current trajectory weights
        self.demonstrations = []  # All previous demonstrations
        self.positive_demos = []  # Positive demonstrations
        self.negative_demos = []  # Negative demonstrations
        
        # Statistics
        self.em_iterations = 0
        self.convergence_history = []
        
    def _generate_basis_functions(self, t):
        """Generate RBF basis functions"""
        if self.basis_centers is None:
            self.basis_centers = np.linspace(0, 1, self.num_basis)
            self.basis_width = 1.0 / (self.num_basis - 1)
        
        # Calculate distances from centers
        distances = cdist(t.reshape(-1, 1), self.basis_centers.reshape(-1, 1))
        
        # Generate RBF basis functions
        basis = np.exp(-0.5 * (distances / self.basis_width) ** 2)
        
        return basis
    
    def _compute_weights(self, trajectory):
        """Compute weights for a given trajectory"""
        t = np.linspace(0, 1, len(trajectory))
        basis = self._generate_basis_functions(t)
        
        # Solve for weights using least squares
        weights = np.linalg.lstsq(basis, trajectory, rcond=None)[0]
        return weights
    
    def _trajectory_to_weights(self, trajectory):
        """Convert trajectory to weight representation"""
        return self._compute_weights(trajectory)
    
    def _weights_to_trajectory(self, weights, num_points=100):
        """Convert weights back to trajectory"""
        t = np.linspace(0, 1, num_points)
        basis = self._generate_basis_functions(t)
        trajectory = basis @ weights
        return trajectory
    
    def initialize_from_demonstrations(self, demonstrations):
        """Initialize ProMP from previous demonstrations"""
        if len(demonstrations) == 0:
            raise ValueError("No demonstrations provided for initialization")
        
        # Compute weights for each demonstration
        all_weights = []
        for demo in demonstrations:
            weights = self._compute_weights(demo)
            all_weights.append(weights)
        
        # Compute mean and covariance of weights
        all_weights = np.array(all_weights)
        self.mean_weights = np.mean(all_weights, axis=0)
        self.cov_weights = np.cov(all_weights.T)
        
        # Add noise to covariance
        self.cov_weights += self.sigma_noise * np.eye(self.cov_weights.shape[0])
        
        # Initialize current u as mean of all demonstrations
        self.current_u = copy.deepcopy(self.mean_weights)
        self.demonstrations = demonstrations
        
        print(f'Initialized from {len(demonstrations)} demonstrations')
    
    def create_positive_negative_demos(self, original_trajectory, deformed_trajectory, 
                                     deformation_region, energy_threshold):
        """
        Create positive and negative demonstrations from deformation
        
        Args:
            original_trajectory: Original learned trajectory
            deformed_trajectory: Deformed trajectory
            deformation_region: Indices of deformation region
            energy_threshold: Energy threshold for triggering EM learning
        """
        if len(deformation_region) == 0:
            return None, None
        
        # Calculate deformation energy
        energy = self._calculate_deformation_energy(
            original_trajectory[deformation_region],
            deformed_trajectory[deformation_region]
        )
        
        if energy < energy_threshold:
            return None, None  # Don't create demos if energy is below threshold
        
        # Create negative demonstration (before deformation region)
        negative_demo = original_trajectory[:deformation_region[0]]
        
        # Create positive demonstration (deformed region + after deformation region)
        positive_demo = np.vstack([
            deformed_trajectory[deformation_region],  # Deformed region
            original_trajectory[deformation_region[-1]+1:]  # After deformation region
        ])
        
        return positive_demo, negative_demo
    
    def _calculate_deformation_energy(self, original_region, deformed_region):
        """Calculate deformation energy between two trajectory regions"""
        if len(original_region) != len(deformed_region):
            return 0.0
        
        # Calculate area between curves for each dimension
        total_energy = 0.0
        time_steps = np.linspace(0, 1, len(original_region))
        
        for dim in range(6):
            # Calculate area between original and deformed curves
            area = np.trapz(
                np.abs(deformed_region[:, dim] - original_region[:, dim]),
                time_steps
            )
            total_energy += area
        
        return total_energy
    
    def stepwise_em_update(self, positive_demo, negative_demo, delta_N=0.1):
        """
        Perform stepwise EM update using positive and negative demonstrations
        
        Args:
            positive_demo: Positive demonstration trajectory
            negative_demo: Negative demonstration trajectory
            delta_N: Learning rate for EM update
        """
        if self.current_u is None:
            raise ValueError("EM learner not initialized")
        
        # Convert demonstrations to weight space
        u_pos = self._trajectory_to_weights(positive_demo)
        u_neg = self._trajectory_to_weights(negative_demo)
        
        # Stepwise EM update: u = (1-δ_N)u + δ_N(u_pos - u_neg)
        self.current_u = (1 - delta_N) * self.current_u + delta_N * (u_pos - u_neg)
        
        # Update ProMP parameters
        self._update_promp_parameters()
        
        self.em_iterations += 1
        print(f'EM update completed (iteration {self.em_iterations})')
    
    def _update_promp_parameters(self):
        """Update ProMP mean and covariance from current u"""
        if self.current_u is None:
            return
        
        # Update mean weights
        if self.mean_weights is None:
            self.mean_weights = copy.deepcopy(self.current_u)
        else:
            # Smooth update of mean
            self.mean_weights = 0.9 * self.mean_weights + 0.1 * self.current_u
        
        # Update covariance (simplified approach)
        if self.cov_weights is None:
            self.cov_weights = self.sigma_noise * np.eye(len(self.current_u))
        else:
            # Add some uncertainty to covariance
            self.cov_weights += 0.01 * self.sigma_noise * np.eye(self.cov_weights.shape[0])
    
    def generate_updated_trajectory(self, num_points=100):
        """Generate updated trajectory from current EM state"""
        if self.current_u is None:
            raise ValueError("EM learner not initialized")
        
        return self._weights_to_trajectory(self.current_u, num_points)
    
    def get_convergence_metric(self):
        """Calculate convergence metric for EM algorithm"""
        if len(self.demonstrations) == 0:
            return 0.0
        
        # Calculate average distance from current u to all demonstration weights
        distances = []
        for demo in self.demonstrations:
            demo_weights = self._trajectory_to_weights(demo)
            distance = np.linalg.norm(self.current_u - demo_weights)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        self.convergence_history.append(avg_distance)
        
        return avg_distance
    
    def check_convergence(self, tolerance=1e-4, min_iterations=5):
        """Check if EM algorithm has converged"""
        if len(self.convergence_history) < min_iterations:
            return False
        
        # Check if the change in convergence metric is below tolerance
        recent_changes = np.diff(self.convergence_history[-min_iterations:])
        return np.all(np.abs(recent_changes) < tolerance)
    
    def visualize_em_progress(self, save_path=None):
        """Visualize EM learning progress"""
        if len(self.convergence_history) == 0:
            print("No convergence history to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot convergence metric
        ax1.plot(self.convergence_history, 'b-', linewidth=2)
        ax1.set_xlabel('EM Iterations')
        ax1.set_ylabel('Convergence Metric')
        ax1.set_title('EM Convergence Progress')
        ax1.grid(True)
        
        # Plot current trajectory vs original demonstrations
        if len(self.demonstrations) > 0:
            current_traj = self.generate_updated_trajectory()
            
            # Plot one dimension as example
            ax2.plot(current_traj[:, 0], 'r-', linewidth=2, label='Updated Trajectory')
            
            for i, demo in enumerate(self.demonstrations[:3]):  # Show first 3 demos
                ax2.plot(demo[:, 0], 'b-', alpha=0.3, label=f'Demo {i+1}' if i == 0 else "")
            
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('X Position')
            ax2.set_title('Trajectory Comparison')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_em_state(self, filename='em_state.npy'):
        """Save EM learning state"""
        state = {
            'current_u': self.current_u,
            'mean_weights': self.mean_weights,
            'cov_weights': self.cov_weights,
            'em_iterations': self.em_iterations,
            'convergence_history': self.convergence_history,
            'basis_centers': self.basis_centers,
            'basis_width': self.basis_width
        }
        np.save(filename, state)
        print(f'EM state saved to {filename}')
    
    def load_em_state(self, filename='em_state.npy'):
        """Load EM learning state"""
        try:
            state = np.load(filename, allow_pickle=True).item()
            self.current_u = state['current_u']
            self.mean_weights = state['mean_weights']
            self.cov_weights = state['cov_weights']
            self.em_iterations = state['em_iterations']
            self.convergence_history = state['convergence_history']
            self.basis_centers = state['basis_centers']
            self.basis_width = state['basis_width']
            print(f'EM state loaded from {filename}')
        except FileNotFoundError:
            print(f'EM state file not found: {filename}')
        except Exception as e:
            print(f'Error loading EM state: {e}')
    
    def get_statistics(self):
        """Get EM learning statistics"""
        return {
            'em_iterations': self.em_iterations,
            'convergence_metric': self.get_convergence_metric(),
            'converged': self.check_convergence(),
            'num_demonstrations': len(self.demonstrations),
            'num_positive_demos': len(self.positive_demos),
            'num_negative_demos': len(self.negative_demos)
        }
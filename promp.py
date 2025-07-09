import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

class ProMP:
    def __init__(self, num_basis=50, sigma_noise=0.01):
        self.num_basis = num_basis
        self.sigma_noise = sigma_noise
        self.weights = None
        self.basis_centers = None
        self.basis_width = None
        self.mean_weights = None
        self.cov_weights = None
        
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
    
    def train(self, demonstrations):
        """Train ProMP on multiple demonstrations"""
        if len(demonstrations) == 0:
            raise ValueError("No demonstrations provided")
        
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
        
        print(f'ProMP trained on {len(demonstrations)} demonstrations')
    
    def generate_trajectory(self, num_points=100):
        """Generate trajectory from learned ProMP"""
        if self.mean_weights is None:
            raise ValueError("ProMP not trained yet")
        
        # Sample weights from distribution
        weights = np.random.multivariate_normal(self.mean_weights, self.cov_weights)
        
        # Generate trajectory
        t = np.linspace(0, 1, num_points)
        basis = self._generate_basis_functions(t)
        
        trajectory = basis @ weights
        
        return trajectory
    
    def condition_on_waypoint(self, t_condition, y_condition, sigma_condition=0.01):
        """
        Condition ProMP on a waypoint
        
        Args:
            t_condition: Time point (0-1) where conditioning occurs
            y_condition: Desired position at conditioning point
            sigma_condition: Uncertainty of conditioning point
        """
        if self.mean_weights is None:
            raise ValueError("ProMP not trained yet")
        
        # Generate basis at conditioning point
        basis_condition = self._generate_basis_functions(np.array([t_condition]))
        
        # Compute conditional distribution
        K = self.cov_weights @ basis_condition.T @ np.linalg.inv(
            basis_condition @ self.cov_weights @ basis_condition.T + sigma_condition * np.eye(1)
        )
        
        # Update mean and covariance
        self.mean_weights = self.mean_weights + K @ (y_condition - basis_condition @ self.mean_weights)
        self.cov_weights = self.cov_weights - K @ basis_condition @ self.cov_weights
        
        print(f'ProMP conditioned on waypoint at t={t_condition:.3f}')
    
    def condition_on_multiple_waypoints(self, t_conditions, y_conditions, sigma_condition=0.01):
        """
        Condition ProMP on multiple waypoints
        
        Args:
            t_conditions: List of time points (0-1)
            y_conditions: List of desired positions
            sigma_condition: Uncertainty of conditioning points
        """
        if self.mean_weights is None:
            raise ValueError("ProMP not trained yet")
        
        for t_cond, y_cond in zip(t_conditions, y_conditions):
            self.condition_on_waypoint(t_cond, y_cond, sigma_condition)
    
    def get_logger(self):
        """Get logger for compatibility"""
        import logging
        return logging.getLogger(__name__)
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
        self.dof = None
        
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
        """Compute weights for a given trajectory (T, dof)"""
        t = np.linspace(0, 1, len(trajectory))
        basis = self._generate_basis_functions(t)
        # Solve for weights using least squares for each dof
        weights = np.linalg.lstsq(basis, trajectory, rcond=None)[0]  # (num_basis, dof)
        return weights
    
    def train(self, demonstrations):
        """Train ProMP on multiple 3D demonstrations: (num_demos, T, dof) or list of (T, dof)"""
        if isinstance(demonstrations, list):
            demonstrations = np.array(demonstrations)
        if demonstrations.ndim == 2:
            demonstrations = demonstrations[None, ...]  # Single demo
        if demonstrations.ndim != 3:
            raise ValueError(f"Expected 3D array (num_demos, T, dof), got shape {demonstrations.shape}")
        num_demos, T, dof = demonstrations.shape
        self.dof = dof
        # Compute weights for each demonstration
        all_weights = []
        for demo in demonstrations:
            weights = self._compute_weights(demo)  # (num_basis, dof)
            all_weights.append(weights)
        all_weights = np.stack(all_weights, axis=0)  # (num_demos, num_basis, dof)
        # Compute mean and covariance for each dof
        self.mean_weights = np.mean(all_weights, axis=0)  # (num_basis, dof)
        self.cov_weights = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            self.cov_weights[d] = np.cov(all_weights[:, :, d].T) + self.sigma_noise * np.eye(self.num_basis)
        print(f'ProMP trained on {num_demos} demonstrations, dof={dof}')
    
    def generate_trajectory(self, num_points=100):
        """Generate trajectory from learned ProMP (returns (num_points, dof))"""
        if self.mean_weights is None:
            raise ValueError("ProMP not trained yet")
        dof = self.mean_weights.shape[1]
        weights = np.zeros((self.num_basis, dof))
        for d in range(dof):
            weights[:, d] = np.random.multivariate_normal(self.mean_weights[:, d], self.cov_weights[d])
        t = np.linspace(0, 1, num_points)
        basis = self._generate_basis_functions(t)
        trajectory = basis @ weights  # (num_points, dof)
        return trajectory
    
    def condition_on_waypoint(self, t_condition, y_condition, sigma_condition=0.01):
        """
        Condition ProMP on a waypoint (supports multi-dof)
        Args:
            t_condition: Time point (0-1) where conditioning occurs
            y_condition: Desired position at conditioning point (dof,)
            sigma_condition: Uncertainty of conditioning point
        """
        if self.mean_weights is None:
            raise ValueError("ProMP not trained yet")
        basis_condition = self._generate_basis_functions(np.array([t_condition]))  # (1, num_basis)
        for d in range(self.dof):
            K = self.cov_weights[d] @ basis_condition.T @ np.linalg.inv(
                basis_condition @ self.cov_weights[d] @ basis_condition.T + sigma_condition * np.eye(1)
            )
            self.mean_weights[:, d] = self.mean_weights[:, d] + (K @ (y_condition[d] - basis_condition @ self.mean_weights[:, d])).flatten()
            self.cov_weights[d] = self.cov_weights[d] - K @ basis_condition @ self.cov_weights[d]
        print(f'ProMP conditioned on waypoint at t={t_condition:.3f}')
    
    def condition_on_multiple_waypoints(self, t_conditions, y_conditions, sigma_condition=0.01):
        """
        Condition ProMP on multiple waypoints
        Args:
            t_conditions: List of time points (0-1)
            y_conditions: List of desired positions (each (dof,))
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
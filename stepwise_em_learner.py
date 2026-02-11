#!/usr/bin/env python3
"""
Stepwise EM ProMP Learner
Implements stepwise EM learning with sufficient statistics according to:
- Posterior distribution computation
- Expected Sufficient Statistics (ESS)
- MAP estimation with hyperparameters
- Observation noise update
"""

import numpy as np
import copy
try:
    from scipy.spatial.distance import cdist
except ImportError:
    # Fallback if scipy not available
    def cdist(X, Y):
        """Simple distance computation fallback"""
        X = np.array(X)
        Y = np.array(Y)
        distances = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                distances[i, j] = np.abs(x - y)
        return distances
import matplotlib.pyplot as plt

class StepwiseEMLearner:
    def __init__(self, num_basis=50, sigma_noise=0.01, delta_N=0.1, 
                 m_0=None, k_0=1.0, S_0=None, v_0=None):
        """
        Initialize Stepwise EM Learner for ProMP
        
        Args:
            num_basis: Number of basis functions for ProMP
            sigma_noise: Initial observation noise (Σ_y)
            delta_N: Step size for ESS update (δ_N)
            m_0: Prior mean hyperparameter (default: zero vector)
            k_0: Prior precision hyperparameter (default: 1.0)
            S_0: Prior covariance hyperparameter (default: identity * sigma_noise)
            v_0: Prior degrees of freedom hyperparameter (default: num_basis + 1)
        """
        self.num_basis = num_basis
        self.sigma_noise = sigma_noise  # Initial Σ_y
        self.delta_N = delta_N
        
        # Hyperparameters for MAP estimation
        self.m_0 = m_0  # Prior mean
        self.k_0 = k_0  # Prior precision
        self.S_0 = S_0  # Prior covariance
        self.v_0 = v_0 if v_0 is not None else num_basis + 1  # Prior degrees of freedom
        
        # ProMP parameters (will be updated via MAP)
        self.mean_weights = None  # μ_w
        self.cov_weights = None   # Σ_w
        self.basis_centers = None
        self.basis_width = None
        
        # Sufficient statistics u = [u₁, u₂, u₃]
        self.u_1_curr = None  # Current u₁ (m_w)
        self.u_2_curr = None  # Current u₂ (m_w m_w^T + S_w)
        self.u_3_curr = None  # Current u₃ (observation statistics)
        
        # EM state
        self.demonstrations = []  # All demonstrations
        self.N = 0  # Number of demonstrations seen
        self.T_total = 0  # Total effective time steps across all demonstrations
        self.eta = 0.0  # Normalization factor
        
        # Statistics
        self.em_iterations = 0
        self.convergence_history = []
    
    def _generate_basis_functions(self, t):
        """Generate RBF basis functions Φ_t"""
        if self.basis_centers is None:
            self.basis_centers = np.linspace(0, 1, self.num_basis)
            self.basis_width = max(1.0 / (self.num_basis - 1), 1e-6)  # Ensure non-zero width
        
        # Ensure t is 1D array
        t = np.array(t).flatten()
        
        # Calculate distances from centers
        try:
            distances = cdist(t.reshape(-1, 1), self.basis_centers.reshape(-1, 1))
        except:
            # Fallback if cdist fails
            t_2d = t.reshape(-1, 1)
            centers_2d = self.basis_centers.reshape(-1, 1)
            distances = np.abs(t_2d - centers_2d.T)
        
        # Generate RBF basis functions with numerical stability
        # Clamp distances to avoid overflow
        distances = np.clip(distances, 0, 10.0 * self.basis_width)
        exponent = -0.5 * (distances / max(self.basis_width, 1e-6)) ** 2
        exponent = np.clip(exponent, -50, 50)  # Prevent overflow
        basis = np.exp(exponent)
        
        # Ensure basis is finite
        basis = np.nan_to_num(basis, nan=0.0, posinf=1.0, neginf=0.0)
        
        return basis
    
    def _compute_posterior(self, trajectory, mu_w_prior, sigma_w_prior, sigma_y):
        """
        Compute posterior distribution over weights for a single trajectory
        
        Args:
            trajectory: Trajectory data (T, dof)
            mu_w_prior: Prior mean μ_w
            sigma_w_prior: Prior covariance Σ_w
            sigma_y: Observation noise Σ_y
            
        Returns:
            m_w: Posterior mean
            S_w: Posterior covariance
        """
        T, dof = trajectory.shape
        
        # Normalize time to [0, 1]
        t = np.linspace(0, 1, T)
        Phi = self._generate_basis_functions(t)  # (T, num_basis)
        
        # Initialize posterior parameters
        m_w = np.zeros((self.num_basis, dof))
        S_w = np.zeros((dof, self.num_basis, self.num_basis))
        
        # Compute posterior for each DOF independently
        for d in range(dof):
            y_d = trajectory[:, d]  # (T,)
            mu_w_d = mu_w_prior[:, d] if mu_w_prior is not None else np.zeros(self.num_basis)
            sigma_w_d = sigma_w_prior[d] if sigma_w_prior is not None else np.eye(self.num_basis) * self.sigma_noise
            
            # Compute S_w = (Σ_w^(-1) + Σ_t Φ_t^T Σ_y^(-1) Φ_t)^(-1)
            sigma_w_inv = np.linalg.inv(sigma_w_d)
            sigma_y_inv = 1.0 / sigma_y  # Scalar for now (assuming isotropic noise)
            
            # Sum over time: Σ_t Φ_t^T Σ_y^(-1) Φ_t
            sum_term = np.zeros((self.num_basis, self.num_basis))
            for t_idx in range(T):
                phi_t = Phi[t_idx, :]  # (num_basis,)
                sum_term += np.outer(phi_t, phi_t) * sigma_y_inv
            
            S_w_d = np.linalg.inv(sigma_w_inv + sum_term)
            
            # Compute m_w = S_w(Σ_w^(-1) μ_w + Σ_t Φ_t^T Σ_y^(-1) y_t)
            sum_y_term = np.zeros(self.num_basis)
            for t_idx in range(T):
                phi_t = Phi[t_idx, :]  # (num_basis,)
                sum_y_term += phi_t * sigma_y_inv * y_d[t_idx]
            
            m_w_d = S_w_d @ (sigma_w_inv @ mu_w_d + sum_y_term)
            
            m_w[:, d] = m_w_d
            S_w[d] = S_w_d
        
        return m_w, S_w
    
    def _compute_sufficient_statistics(self, trajectory, m_w, S_w):
        """
        Compute Expected Sufficient Statistics (ESS) for a trajectory
        
        Args:
            trajectory: Trajectory data (T, dof)
            m_w: Posterior mean m_w
            S_w: Posterior covariance S_w
            
        Returns:
            u_1: First sufficient statistic (m_w)
            u_2: Second sufficient statistic (m_w m_w^T + S_w)
            u_3: Third sufficient statistic (observation statistics)
        """
        T, dof = trajectory.shape
        
        # Normalize time to [0, 1]
        t = np.linspace(0, 1, T)
        Phi = self._generate_basis_functions(t)  # (T, num_basis)
        
        # u₁ = m_w
        u_1 = m_w  # (num_basis, dof)
        
        # u₂ = m_w m_w^T + S_w
        u_2 = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            m_w_d = m_w[:, d]  # (num_basis,)
            u_2[d] = np.outer(m_w_d, m_w_d) + S_w[d]  # (num_basis, num_basis)
        
        # u₃ = Σ_t (y_t y_t^T - 2 y_t m_w^T Φ_t^T + Φ_t (m_w m_w^T + S_w) Φ_t^T)
        u_3 = np.zeros((dof, dof))
        for t_idx in range(T):
            phi_t = Phi[t_idx, :]  # (num_basis,)
            y_t = trajectory[t_idx, :]  # (dof,)
            
            for d1 in range(dof):
                for d2 in range(dof):
                    m_w_d1 = m_w[:, d1]
                    m_w_d2 = m_w[:, d2]
                    
                    # y_t y_t^T term
                    term1 = y_t[d1] * y_t[d2]
                    
                    # -2 y_t m_w^T Φ_t^T term
                    term2 = -2 * y_t[d1] * (m_w_d2 @ phi_t)
                    
                    # Φ_t (m_w m_w^T + S_w) Φ_t^T term
                    mw_mwT_plus_S = np.outer(m_w_d1, m_w_d2) + S_w[d1] if d1 == d2 else np.outer(m_w_d1, m_w_d2)
                    term3 = phi_t @ mw_mwT_plus_S @ phi_t
                    
                    u_3[d1, d2] += term1 + term2 + term3
        
        return u_1, u_2, u_3
    
    def initialize_from_demonstrations(self, demonstrations):
        """Initialize ProMP from previous demonstrations"""
        if len(demonstrations) == 0:
            raise ValueError("No demonstrations provided for initialization")
        
        self.demonstrations = demonstrations
        self.N = len(demonstrations)
        
        # Initialize prior hyperparameters if not set
        if self.m_0 is None:
            # Estimate DOF from first demonstration
            dof = demonstrations[0].shape[1] if demonstrations[0].ndim > 1 else 1
            self.m_0 = np.zeros((self.num_basis, dof))
        
        if self.S_0 is None:
            dof = demonstrations[0].shape[1] if demonstrations[0].ndim > 1 else 1
            self.S_0 = np.array([np.eye(self.num_basis) * self.sigma_noise for _ in range(dof)])
        
        # Compute initial mean and covariance from demonstrations
        all_weights = []
        for demo in self.demonstrations:
            demo_array = np.array(demo)
            T = len(demo_array)
            t = np.linspace(0, 1, T)
            Phi = self._generate_basis_functions(t)
            # Solve for weights using least squares
            weights = np.linalg.lstsq(Phi, demo_array, rcond=None)[0]  # (num_basis, dof)
            all_weights.append(weights)
        
        all_weights = np.array(all_weights)  # (N, num_basis, dof)
        
        # Initialize mean and covariance
        dof = all_weights.shape[2]
        self.mean_weights = np.mean(all_weights, axis=0)  # (num_basis, dof)
        self.cov_weights = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            self.cov_weights[d] = np.cov(all_weights[:, :, d].T) + np.eye(self.num_basis) * self.sigma_noise
        
        # Initialize sufficient statistics from all demonstrations
        self._initialize_sufficient_statistics()
        
        print(f'Initialized from {len(demonstrations)} demonstrations')
    
    def _initialize_sufficient_statistics(self):
        """Initialize sufficient statistics from current ProMP parameters"""
        if self.mean_weights is None or self.cov_weights is None:
            return
        
        dof = self.mean_weights.shape[1]
        
        # u₁ = m_w (current mean)
        self.u_1_curr = self.mean_weights.copy()  # (num_basis, dof)
        
        # u₂ = m_w m_w^T + S_w
        self.u_2_curr = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            m_w_d = self.mean_weights[:, d]
            self.u_2_curr[d] = np.outer(m_w_d, m_w_d) + self.cov_weights[d]
        
        # Initialize u₃ from demonstrations
        self.u_3_curr = np.zeros((dof, dof))
        self.T_total = 0
        
        for demo in self.demonstrations:
            T = len(demo)
            self.T_total += T
            
            # Compute posterior for this demo
            m_w_demo, S_w_demo = self._compute_posterior(
                demo, self.mean_weights, self.cov_weights, self.sigma_noise
            )
            
            # Compute ESS for this demo
            _, _, u_3_demo = self._compute_sufficient_statistics(demo, m_w_demo, S_w_demo)
            self.u_3_curr += u_3_demo
        
        # Initialize normalization factor
        self.eta = self.N
    
    def initialize_from_first_demo(self, trajectory):
        """
        Initialize from a single demonstration (for incremental learning)
        This allows starting with just one demo and learning incrementally
        """
        trajectory = np.array(trajectory)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        
        self.demonstrations = [trajectory]
        self.N = 1
        
        # Initialize prior hyperparameters if not set
        dof = trajectory.shape[1]
        if self.m_0 is None:
            self.m_0 = np.zeros((self.num_basis, dof))
        
        if self.S_0 is None:
            self.S_0 = np.array([np.eye(self.num_basis) * self.sigma_noise for _ in range(dof)])
        
        # Compute initial mean and covariance from first demo
        T = len(trajectory)
        t = np.linspace(0, 1, T)
        Phi = self._generate_basis_functions(t)
        weights = np.linalg.lstsq(Phi, trajectory, rcond=None)[0]  # (num_basis, dof)
        
        self.mean_weights = weights  # (num_basis, dof)
        self.cov_weights = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            self.cov_weights[d] = np.eye(self.num_basis) * self.sigma_noise
        
        # Initialize sufficient statistics from first demo
        m_w_demo, S_w_demo = self._compute_posterior(
            trajectory, self.mean_weights, self.cov_weights, self.sigma_noise
        )
        
        u_1, u_2, u_3 = self._compute_sufficient_statistics(trajectory, m_w_demo, S_w_demo)
        
        self.u_1_curr = u_1
        self.u_2_curr = u_2
        self.u_3_curr = u_3
        self.T_total = T
        self.eta = 1.0
        
        # Update ProMP parameters via MAP
        self._update_promp_parameters_map()
        self._update_observation_noise()
        
        print(f'Initialized from first demonstration (T={T}, dof={dof})')
    
    def stepwise_em_update(self, trajectory):
        """
        Perform stepwise EM update using a new demonstration
        
        Args:
            trajectory: New demonstration trajectory (T, dof)
        """
        # Convert to numpy array and ensure 2D
        trajectory = np.array(trajectory)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        
        # Initialize from first demo if not already initialized
        if self.mean_weights is None or self.cov_weights is None:
            print("Not initialized, initializing from first demonstration...")
            self.initialize_from_first_demo(trajectory)
            return
        
        T = len(trajectory)
        dof = trajectory.shape[1]
        
        # Compute posterior distribution for this trajectory
        m_w_new, S_w_new = self._compute_posterior(
            trajectory, self.mean_weights, self.cov_weights, self.sigma_noise
        )
        
        # Compute sufficient statistics for this trajectory
        u_1_new, u_2_new, u_3_new = self._compute_sufficient_statistics(trajectory, m_w_new, S_w_new)
        
        # Update sufficient statistics: u^curr = (1-δ_N)u^curr + δ_N u'
        if self.u_1_curr is None:
            # First update: initialize
            self.u_1_curr = u_1_new.copy()
            self.u_2_curr = u_2_new.copy()
            self.u_3_curr = u_3_new.copy()
            self.T_total = T
            self.eta = 1.0
        else:
            # Update: u^curr = (1-δ_N)u^curr + δ_N u'
            self.u_1_curr = (1 - self.delta_N) * self.u_1_curr + self.delta_N * u_1_new
            self.u_2_curr = (1 - self.delta_N) * self.u_2_curr + self.delta_N * u_2_new
            self.u_3_curr = (1 - self.delta_N) * self.u_3_curr + self.delta_N * u_3_new
            
            # Update effective time steps
            self.T_total = (1 - self.delta_N) * self.T_total + self.delta_N * T
            
            # Update normalization factor
            self.eta = (1 - self.delta_N) * self.eta + self.delta_N
        
        # Update number of demonstrations
        self.N += 1
        
        # Update ProMP parameters via MAP estimation
        self._update_promp_parameters_map()
        
        # Update observation noise
        self._update_observation_noise()
        
        # Store demonstration
        self.demonstrations.append(trajectory)
        
        self.em_iterations += 1
        print(f'EM update completed (iteration {self.em_iterations}, N={self.N})')
    
    def _update_promp_parameters_map(self):
        """
        Update ProMP parameters using MAP estimation
        
        μ_w* = (1/η)u₁^curr
        μ_w = (1/(N+k₀))(k₀m₀ + Nμ_w*)
        
        Σ_w* = (1/η)(u₂^curr + S_w) - μ_w μ_w^T
        Σ_w = (S₀ + NΣ_w* + (k₀N/(k₀+N))(μ_w* - m₀)(μ_w* - m₀)^T) / (N + v₀ + KD + 2)
        """
        if self.u_1_curr is None or self.eta == 0:
            return
        
        dof = self.u_1_curr.shape[1]
        KD = self.num_basis * dof
        
        # Compute μ_w* = (1/η)u₁^curr
        # Ensure u_1_curr is finite
        u_1_safe = np.nan_to_num(self.u_1_curr, nan=0.0, posinf=1e6, neginf=-1e6)
        eta_safe = max(self.eta, 1e-10)
        mu_w_star = u_1_safe / eta_safe  # (num_basis, dof)
        
        # Ensure mu_w_star is finite
        mu_w_star = np.nan_to_num(mu_w_star, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Compute μ_w = (1/(N+k₀))(k₀m₀ + Nμ_w*)
        if self.m_0 is not None:
            m_0_safe = np.nan_to_num(self.m_0, nan=0.0, posinf=1e6, neginf=-1e6)
            denominator = max(self.N + self.k_0, 1e-10)
            self.mean_weights = (self.k_0 * m_0_safe + self.N * mu_w_star) / denominator
        else:
            self.mean_weights = mu_w_star
        
        # Ensure mean_weights is finite
        self.mean_weights = np.nan_to_num(self.mean_weights, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Compute Σ_w* = (1/η)u₂^curr - μ_w μ_w^T
        sigma_w_star = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            mu_w_d = self.mean_weights[:, d]
            # Ensure u_2_curr is finite
            u_2_safe = np.nan_to_num(self.u_2_curr[d], nan=0.0, posinf=1e6, neginf=-1e6)
            sigma_w_star[d] = u_2_safe / max(self.eta, 1e-10) - np.outer(mu_w_d, mu_w_d)
            # Ensure positive semi-definite
            sigma_w_star[d] = (sigma_w_star[d] + sigma_w_star[d].T) / 2
            # Add small regularization to ensure positive definiteness
            sigma_w_star[d] += np.eye(self.num_basis) * 1e-6
        
        # Compute Σ_w = (S₀ + NΣ_w* + (k₀N/(k₀+N))(μ_w* - m₀)(μ_w* - m₀)^T) / (N + v₀ + KD + 2)
        self.cov_weights = np.zeros((dof, self.num_basis, self.num_basis))
        for d in range(dof):
            mu_w_star_d = mu_w_star[:, d]
            mu_w_d = self.mean_weights[:, d]
            
            if self.m_0 is not None:
                m_0_d = self.m_0[:, d]
                diff = mu_w_star_d - m_0_d
                prior_term = (self.k_0 * self.N / (self.k_0 + self.N)) * np.outer(diff, diff)
            else:
                prior_term = np.zeros((self.num_basis, self.num_basis))
            
            if self.S_0 is not None:
                S_0_d = self.S_0[d]
            else:
                S_0_d = np.eye(self.num_basis) * self.sigma_noise
            
            numerator = S_0_d + self.N * sigma_w_star[d] + prior_term
            denominator = max(self.N + self.v_0 + KD + 2, 1e-10)
            
            self.cov_weights[d] = numerator / denominator
            
            # Ensure covariance is positive semi-definite and finite
            self.cov_weights[d] = (self.cov_weights[d] + self.cov_weights[d].T) / 2
            self.cov_weights[d] = np.nan_to_num(self.cov_weights[d], nan=1e-6, posinf=1e6, neginf=1e-6)
            # Ensure minimum eigenvalue is positive
            eigvals = np.linalg.eigvals(self.cov_weights[d])
            min_eigval = np.min(eigvals)
            if min_eigval < 1e-6:
                self.cov_weights[d] += np.eye(self.num_basis) * (1e-6 - min_eigval)
    
    def _update_observation_noise(self):
        """
        Update observation noise: Σ_y = (1/T)u₃^curr
        """
        if self.u_3_curr is None or self.T_total == 0:
            return
        
        # For isotropic noise, use trace or average diagonal
        # Σ_y = (1/T)u₃^curr
        dof = self.u_3_curr.shape[0]
        
        # Ensure u_3_curr is finite
        u_3_safe = np.nan_to_num(self.u_3_curr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Compute sigma_y_new with safety checks
        trace_val = np.trace(u_3_safe)
        if trace_val > 0 and np.isfinite(trace_val):
            sigma_y_new = trace_val / (self.T_total * dof)
            # Clamp to reasonable range
            sigma_y_new = np.clip(sigma_y_new, 1e-6, 1.0)
        else:
            sigma_y_new = self.sigma_noise  # Keep current value
        
        # Update with smoothing
        self.sigma_noise = (1 - self.delta_N) * self.sigma_noise + self.delta_N * sigma_y_new
    
    def generate_updated_trajectory(self, num_points=100):
        """Generate updated trajectory from current ProMP parameters"""
        if self.mean_weights is None:
            raise ValueError("ProMP not initialized")
        
        dof = self.mean_weights.shape[1]
        t = np.linspace(0, 1, num_points)
        Phi = self._generate_basis_functions(t)
        
        # Use mean weights directly (deterministic) instead of sampling
        # This avoids numerical issues and provides consistent results
        weights = self.mean_weights.copy()  # (num_basis, dof)
        
        # Clamp weights to reasonable range to prevent overflow
        weights = np.clip(weights, -1e3, 1e3)
        
        # Ensure weights are finite
        weights = np.nan_to_num(weights, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Ensure Phi is finite
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Compute trajectory with safe matrix multiplication
        try:
            trajectory = Phi @ weights  # (num_points, dof)
        except Exception as e:
            # Fallback: element-wise multiplication if matrix multiplication fails
            print(f"Warning: Matrix multiplication failed ({e}), using fallback")
            trajectory = np.zeros((Phi.shape[0], weights.shape[1]))
            for i in range(Phi.shape[0]):
                for j in range(weights.shape[1]):
                    trajectory[i, j] = np.dot(Phi[i, :], weights[:, j])
        
        # Ensure trajectory is finite and clamp to reasonable range
        trajectory = np.clip(trajectory, -10.0, 10.0)
        trajectory = np.nan_to_num(trajectory, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return trajectory
    
    def get_convergence_metric(self):
        """Calculate convergence metric for EM algorithm"""
        if len(self.demonstrations) < 2:
            return 0.0
        
        # Compute distance between current and previous mean weights
        if len(self.convergence_history) > 0:
            # Use change in mean weights as convergence metric
            prev_mean = self.convergence_history[-1]
            if isinstance(prev_mean, np.ndarray):
                current_metric = np.linalg.norm(self.mean_weights - prev_mean)
            else:
                current_metric = np.linalg.norm(self.mean_weights)
        else:
            current_metric = np.linalg.norm(self.mean_weights)
        
        self.convergence_history.append(self.mean_weights.copy())
        return current_metric
    
    def check_convergence(self, tolerance=1e-4, min_iterations=5):
        """Check if EM algorithm has converged"""
        if len(self.convergence_history) < min_iterations + 1:
            return False
        
        # Check if the change in mean weights is below tolerance
        recent_changes = []
        for i in range(len(self.convergence_history) - min_iterations, len(self.convergence_history) - 1):
            change = np.linalg.norm(self.convergence_history[i+1] - self.convergence_history[i])
            recent_changes.append(change)
        
        return np.all(np.array(recent_changes) < tolerance)
    
    def visualize_em_progress(self, save_path=None):
        """Visualize EM learning progress"""
        if len(self.convergence_history) == 0:
            print("No convergence history to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot convergence metric
        metrics = []
        for i in range(1, len(self.convergence_history)):
            change = np.linalg.norm(self.convergence_history[i] - self.convergence_history[i-1])
            metrics.append(change)
        
        ax1.plot(metrics, 'b-', linewidth=2)
        ax1.set_xlabel('EM Iterations')
        ax1.set_ylabel('Change in Mean Weights')
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
            'mean_weights': self.mean_weights,
            'cov_weights': self.cov_weights,
            'u_1_curr': self.u_1_curr,
            'u_2_curr': self.u_2_curr,
            'u_3_curr': self.u_3_curr,
            'sigma_noise': self.sigma_noise,
            'N': self.N,
            'T_total': self.T_total,
            'eta': self.eta,
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
            self.mean_weights = state['mean_weights']
            self.cov_weights = state['cov_weights']
            self.u_1_curr = state.get('u_1_curr')
            self.u_2_curr = state.get('u_2_curr')
            self.u_3_curr = state.get('u_3_curr')
            self.sigma_noise = state.get('sigma_noise', self.sigma_noise)
            self.N = state.get('N', 0)
            self.T_total = state.get('T_total', 0)
            self.eta = state.get('eta', 0.0)
            self.em_iterations = state.get('em_iterations', 0)
            self.convergence_history = state.get('convergence_history', [])
            self.basis_centers = state.get('basis_centers')
            self.basis_width = state.get('basis_width')
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
            'num_demonstrations': self.N,
            'sigma_noise': self.sigma_noise,
            'T_total': self.T_total,
            'eta': self.eta
        }

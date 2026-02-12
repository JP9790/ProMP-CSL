#!/usr/bin/env python3
"""
Adversarial Inverse Reinforcement Learning (AIRL)
Simplified implementation for trajectory learning from demonstrations
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. AIRL will use simplified numpy implementation.")


class AIRL:
    """
    Adversarial Inverse Reinforcement Learning for learning reward functions from demonstrations
    """
    def __init__(self, state_dim=6, action_dim=6, hidden_dim=64, learning_rate=0.001, 
                 gamma=0.99, use_torch=True):
        """
        Initialize AIRL
        
        Args:
            state_dim: Dimension of state space (6 for Cartesian pose: x, y, z, alpha, beta, gamma)
            action_dim: Dimension of action space (6 for Cartesian velocities)
            hidden_dim: Hidden dimension for neural networks
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            use_torch: Whether to use PyTorch (if available) or numpy implementation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_torch = use_torch and HAS_TORCH
        
        # Reward function (learned from demonstrations)
        self.reward_function = None
        self.discriminator = None
        
        # Policy (learned from reward function)
        self.policy = None
        
        # Training history
        self.training_losses = []
        self.reward_history = []
        
        # For numpy fallback: use simple linear reward function
        if not self.use_torch:
            self.reward_weights = np.random.randn(state_dim + action_dim)
            print("Using simplified numpy-based AIRL (no PyTorch)")
        
        # Logger (will be set by controller)
        self.logger = None
    
    def _create_reward_network(self):
        """Create neural network for reward function"""
        if not self.use_torch:
            return None
        
        class RewardNetwork(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, 1)
                self.relu = nn.ReLU()
                
            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        return RewardNetwork(self.state_dim, self.action_dim, self.hidden_dim)
    
    def _create_discriminator(self):
        """Create discriminator network (distinguishes expert vs policy)"""
        if not self.use_torch:
            return None
        
        class Discriminator(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.sigmoid(self.fc3(x))
        
        return Discriminator(self.state_dim, self.action_dim, self.hidden_dim)
    
    def compute_reward(self, state, action):
        """
        Compute reward for given state-action pair
        
        Args:
            state: State vector (state_dim,)
            action: Action vector (action_dim,)
            
        Returns:
            float: Reward value
        """
        if self.use_torch and self.reward_function is not None:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.FloatTensor(action).unsqueeze(0)
            with torch.no_grad():
                reward = self.reward_function(state_t, action_t).item()
            return reward
        else:
            # Simplified numpy implementation
            sa = np.concatenate([state, action])
            return np.dot(self.reward_weights, sa)
    
    def train(self, demonstrations, num_iterations=1000, batch_size=32):
        """
        Train AIRL on demonstrations
        
        Args:
            demonstrations: List of demonstration trajectories, each is (T, state_dim) array
            num_iterations: Number of training iterations
            batch_size: Batch size for training
        """
        if len(demonstrations) == 0:
            raise ValueError("No demonstrations provided")
        
        self.get_logger().info(f'Training AIRL on {len(demonstrations)} demonstrations...')
        
        # Get logger to pass to training methods
        logger = self.get_logger()
        
        if self.use_torch:
            self._train_torch(demonstrations, num_iterations, batch_size, logger)
        else:
            self._train_numpy(demonstrations, num_iterations, logger)
    
    def _train_torch(self, demonstrations, num_iterations, batch_size, logger):
        """Train AIRL using PyTorch"""
        # Store demonstrations for trajectory generation
        self.demonstrations = demonstrations
        
        # Create networks
        self.reward_function = self._create_reward_network()
        self.discriminator = self._create_discriminator()
        
        # Optimizers
        reward_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.learning_rate)
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        # Prepare expert data
        expert_states = []
        expert_actions = []
        for demo in demonstrations:
            states = demo[:, :self.state_dim]
            # Actions are differences between consecutive states
            actions = np.diff(states, axis=0)
            # Pad with zero action for last state
            actions = np.vstack([actions, np.zeros((1, self.action_dim))])
            
            expert_states.append(states)
            expert_actions.append(actions)
        
        expert_states = np.concatenate(expert_states, axis=0)
        expert_actions = np.concatenate(expert_actions, axis=0)
        
        # Training loop
        for iteration in range(num_iterations):
            # Sample batch from expert data
            indices = np.random.choice(len(expert_states), batch_size, replace=False)
            expert_s_batch = torch.FloatTensor(expert_states[indices])
            expert_a_batch = torch.FloatTensor(expert_actions[indices])
            
            # Generate policy data (simplified: use random for now)
            # In full AIRL, this would come from policy rollouts
            policy_s_batch = torch.FloatTensor(np.random.randn(batch_size, self.state_dim))
            policy_a_batch = torch.FloatTensor(np.random.randn(batch_size, self.action_dim))
            
            # Train discriminator
            disc_optimizer.zero_grad()
            
            expert_logits = self.discriminator(expert_s_batch, expert_a_batch)
            policy_logits = self.discriminator(policy_s_batch, policy_a_batch)
            
            # Discriminator loss: maximize log(D(expert)) + log(1 - D(policy))
            disc_loss = -torch.mean(torch.log(expert_logits + 1e-8)) - torch.mean(torch.log(1 - policy_logits + 1e-8))
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train reward function (simplified: use discriminator output as reward signal)
            reward_optimizer.zero_grad()
            reward = self.reward_function(expert_s_batch, expert_a_batch)
            reward_loss = -torch.mean(reward)  # Maximize reward for expert data
            reward_loss.backward()
            reward_optimizer.step()
            
            if iteration % 100 == 0:
                logger.info(f'Iteration {iteration}/{num_iterations}, Disc Loss: {disc_loss.item():.4f}, Reward Loss: {reward_loss.item():.4f}')
                self.training_losses.append(disc_loss.item())
        
        logger.info('AIRL training completed')
    
    def _train_numpy(self, demonstrations, num_iterations, logger):
        """Simplified numpy-based training (fallback when PyTorch not available)"""
        logger.info('Using simplified numpy-based AIRL training...')
        
        # Collect all state-action pairs from demonstrations
        all_states = []
        all_actions = []
        
        for demo in demonstrations:
            states = demo[:, :self.state_dim]
            # Actions are differences between consecutive states
            actions = np.diff(states, axis=0)
            actions = np.vstack([actions, np.zeros((1, self.action_dim))])
            
            all_states.append(states)
            all_actions.append(actions)
        
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        # Store demonstrations for trajectory generation
        self.demonstrations = demonstrations
        
        # Simple reward learning: fit linear reward function to expert data
        # Reward = w^T * [state; action]
        sa_pairs = np.hstack([all_states, all_actions])
        
        # Compute target rewards: higher reward for actions that follow demonstration patterns
        # Use distance to mean action as reward signal
        mean_action = np.mean(all_actions, axis=0)
        action_distances = np.linalg.norm(all_actions - mean_action, axis=1)
        # Inverse distance: closer to mean = higher reward
        max_dist = np.max(action_distances) + 1e-10
        target_rewards = 1.0 - (action_distances / max_dist)
        target_rewards = np.clip(target_rewards, 0.1, 1.0)  # Keep rewards positive
        
        # Simple linear regression for reward weights
        # Minimize: ||w^T * sa - target_rewards||^2
        def objective(w):
            predictions = np.dot(sa_pairs, w)
            return np.mean((predictions - target_rewards) ** 2)
        
        result = minimize(objective, self.reward_weights, method='BFGS')
        self.reward_weights = result.x
        
        logger.info(f'Simplified AIRL training completed. Reward weights shape: {self.reward_weights.shape}')
        logger.info(f'Reward weight range: [{np.min(self.reward_weights):.4f}, {np.max(self.reward_weights):.4f}]')
    
    def generate_trajectory(self, initial_state, num_points=100, dt=0.01, demonstrations=None):
        """
        Generate trajectory using learned reward function
        
        Args:
            initial_state: Initial state (state_dim,)
            num_points: Number of trajectory points
            dt: Time step
            demonstrations: Optional list of demonstrations to guide trajectory generation (if None, uses self.demonstrations from training)
            
        Returns:
            trajectory: Generated trajectory (num_points, state_dim)
        """
        if self.reward_function is None and self.reward_weights is None:
            raise ValueError("AIRL not trained yet. Call train() first.")
        
        trajectory = np.zeros((num_points, self.state_dim))
        trajectory[0] = initial_state
        
        # Use provided demonstrations or fall back to stored demonstrations from training
        if demonstrations is None:
            demonstrations = getattr(self, 'demonstrations', None)
        
        # If demonstrations are available, use them as a guide
        if demonstrations is not None and len(demonstrations) > 0:
            # Use mean demonstration trajectory as baseline
            mean_demo = np.mean([demo for demo in demonstrations], axis=0)
            
            # Interpolate mean demo to match num_points
            if len(mean_demo) != num_points:
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, len(mean_demo))
                t_new = np.linspace(0, 1, num_points)
                mean_demo_interp = np.zeros((num_points, self.state_dim))
                for dim in range(self.state_dim):
                    interp_func = interp1d(t_old, mean_demo[:, dim], kind='cubic', fill_value='extrapolate')
                    mean_demo_interp[:, dim] = interp_func(t_new)
                mean_demo = mean_demo_interp
            
            # Start from initial state, then blend with mean demo
            # Use reward function to adjust the trajectory
            trajectory = mean_demo.copy()
            trajectory[0] = initial_state
            
            # Refine trajectory using reward function (gradient ascent on reward)
            for iteration in range(5):  # Multiple refinement passes
                for i in range(1, num_points):
                    current_state = trajectory[i-1]
                    target_state = trajectory[i]
                    
                    # Compute action as difference
                    base_action = (target_state - current_state) / dt
                    
                    # Try variations around base action to maximize reward
                    best_action = base_action.copy()
                    best_reward = self.compute_reward(current_state, base_action)
                    
                    # Try gradient-based improvement
                    for _ in range(20):
                        # Add small perturbation
                        perturbation = np.random.randn(self.action_dim) * 0.05
                        candidate_action = base_action + perturbation
                        
                        # Ensure action doesn't cause too large state changes
                        max_action_norm = np.linalg.norm(base_action) * 2.0
                        if np.linalg.norm(candidate_action) > max_action_norm:
                            candidate_action = candidate_action / np.linalg.norm(candidate_action) * max_action_norm
                        
                        reward = self.compute_reward(current_state, candidate_action)
                        if reward > best_reward:
                            best_reward = reward
                            best_action = candidate_action
                    
                    # Update trajectory point
                    trajectory[i] = current_state + best_action * dt
            
            return trajectory
        
        # Fallback: generate trajectory from scratch using reward function
        current_state = initial_state.copy()
        
        for i in range(1, num_points):
            # Use larger action search space
            best_action = np.zeros(self.action_dim)
            best_reward = float('-inf')
            
            # Try more actions with larger scale
            for _ in range(50):
                # Scale action based on progress through trajectory
                progress = i / num_points
                action_scale = 0.5 * (1.0 - progress * 0.5)  # Start larger, decrease over time
                action = np.random.randn(self.action_dim) * action_scale
                
                reward = self.compute_reward(current_state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
            
            # Use larger dt for more meaningful movement
            effective_dt = dt * 10.0  # Scale up time step
            next_state = current_state + best_action * effective_dt
            trajectory[i] = next_state
            current_state = next_state
        
        return trajectory
    
    def get_logger(self):
        """Get logger for compatibility"""
        if self.logger is not None:
            return self.logger
        import logging
        return logging.getLogger(__name__)

#!/usr/bin/env python3
"""Simple test for stepwise EM learner"""

import numpy as np
from stepwise_em_learner import StepwiseEMLearner

# Generate simple demo
t = np.linspace(0, 1, 100)
demo1 = np.column_stack([
    0.3 * t,
    0.2 * np.sin(np.pi * t),
    0.5 + 0.1 * t
])

print("Creating learner...")
learner = StepwiseEMLearner(num_basis=20, sigma_noise=0.01, delta_N=0.2)

print("Initializing from first demo...")
learner.initialize_from_first_demo(demo1)

print("Generating trajectory...")
traj1 = learner.generate_updated_trajectory(num_points=100)
print(f"Trajectory shape: {traj1.shape}")
print(f"Trajectory range X: [{np.min(traj1[:, 0]):.4f}, {np.max(traj1[:, 0]):.4f}]")

print("Adding second demo...")
demo2 = np.column_stack([
    0.3 * t + 0.05,
    0.2 * np.sin(np.pi * t) + 0.05,
    0.5 + 0.1 * t + 0.05
])
learner.stepwise_em_update(demo2)

print("Generating trajectory after 2 demos...")
traj2 = learner.generate_updated_trajectory(num_points=100)
print(f"Trajectory shape: {traj2.shape}")
print(f"Trajectory range X: [{np.min(traj2[:, 0]):.4f}, {np.max(traj2[:, 0]):.4f}]")

print("Test completed successfully!")

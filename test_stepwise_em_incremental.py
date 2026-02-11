#!/usr/bin/env python3
"""
Test script for Stepwise EM ProMP Learner - Incremental Learning
Tests incremental learning with 5 demonstrations, showing progress after 2, 4, and 5 demos
"""

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualization")

from stepwise_em_learner import StepwiseEMLearner

def generate_demo_trajectory(demo_id, duration=5.0, dt=0.01, noise_level=0.02):
    """
    Generate a demonstration trajectory with some variation
    
    Args:
        demo_id: Demo number (0-4)
        duration: Duration in seconds
        dt: Time step
        noise_level: Noise level for variation
        
    Returns:
        trajectory: (T, 6) array [x, y, z, alpha, beta, gamma]
    """
    t = np.arange(0, duration, dt)
    n_points = len(t)
    
    # Base trajectory: moving in 3D space
    # Add variation based on demo_id to create different demonstrations
    variation = demo_id * 0.05  # Small variation between demos
    
    x = 0.3 * (t / duration) + variation * np.sin(2 * np.pi * t / duration) + np.random.normal(0, noise_level, n_points)
    y = 0.2 * np.sin(np.pi * t / duration) + variation * np.cos(2 * np.pi * t / duration) + np.random.normal(0, noise_level, n_points)
    z = 0.5 + 0.1 * (t / duration) + 0.05 * np.sin(2 * np.pi * t / duration) + variation * np.sin(np.pi * t / duration) + np.random.normal(0, noise_level, n_points)
    
    # Orientation: minimal rotation
    alpha = np.zeros(n_points) + np.random.normal(0, noise_level * 0.1, n_points)
    beta = np.zeros(n_points) + np.random.normal(0, noise_level * 0.1, n_points)
    gamma = np.pi / 8 * (t / duration) + variation * 0.1 + np.random.normal(0, noise_level * 0.1, n_points)
    
    trajectory = np.column_stack([x, y, z, alpha, beta, gamma])
    
    return trajectory

def test_incremental_learning():
    """Test incremental stepwise EM learning with 5 demonstrations"""
    
    print("="*70)
    print("STEPWISE EM PROMP INCREMENTAL LEARNING TEST")
    print("="*70)
    print()
    
    # Generate 5 demonstration trajectories
    print("Generating 5 demonstration trajectories...")
    demos = []
    for i in range(5):
        demo = generate_demo_trajectory(i, duration=5.0, dt=0.01)
        demos.append(demo)
        print(f"  Demo {i+1}: shape {demo.shape}")
    
    # Initialize StepwiseEMLearner
    print("\nInitializing StepwiseEMLearner...")
    em_learner = StepwiseEMLearner(
        num_basis=30,
        sigma_noise=0.01,
        delta_N=0.2  # Step size for incremental updates
    )
    
    # Initialize from first demo
    print("\nInitializing from first demonstration...")
    em_learner.initialize_from_first_demo(demos[0])
    
    # Track trajectories at different stages
    trajectories_at_stages = {}
    
    # Learn incrementally: add demos one by one
    print("\n" + "="*70)
    print("INCREMENTAL LEARNING PROCESS")
    print("="*70)
    
    for i in range(1, 5):  # Start from demo 2 (demo 1 already used for initialization)
        print(f"\n--- Learning from Demo {i+1} ---")
        print(f"Current state: N={em_learner.N}, iterations={em_learner.em_iterations}")
        
        # Perform stepwise EM update with new demo
        em_learner.stepwise_em_update(demos[i])
        
        # Generate learned trajectory
        learned_traj = em_learner.generate_updated_trajectory(num_points=500)
        
        # Store trajectory at key stages
        if i == 1:  # After 2 demos total (initialized from 1, learned from 1)
            trajectories_at_stages[2] = learned_traj.copy()
            print(f"✓ Learned trajectory after 2 demos: shape {learned_traj.shape}")
        elif i == 3:  # After 4 demos total
            trajectories_at_stages[4] = learned_traj.copy()
            print(f"✓ Learned trajectory after 4 demos: shape {learned_traj.shape}")
        elif i == 4:  # After 5 demos total
            trajectories_at_stages[5] = learned_traj.copy()
            print(f"✓ Learned trajectory after 5 demos: shape {learned_traj.shape}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    stats = em_learner.get_statistics()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Visualize results
    if HAS_MATPLOTLIB:
        print("\nGenerating visualization...")
        visualize_incremental_learning(demos, trajectories_at_stages)
    else:
        print("\nSkipping visualization (matplotlib not available)")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70)

def visualize_incremental_learning(demos, trajectories_at_stages):
    """Visualize incremental learning progress"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create time axis
    time_2 = np.linspace(0, 1, len(trajectories_at_stages[2]))
    time_4 = np.linspace(0, 1, len(trajectories_at_stages[4]))
    time_5 = np.linspace(0, 1, len(trajectories_at_stages[5]))
    time_demos = np.linspace(0, 1, len(demos[0]))
    
    # Plot 3D trajectories
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
    
    # Plot all demonstrations
    colors_demo = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for i, demo in enumerate(demos):
        ax_3d.plot(demo[:, 0], demo[:, 1], demo[:, 2], 
                  color=colors_demo[i], alpha=0.3, linewidth=1, 
                  label=f'Demo {i+1}' if i < 3 else "")
    
    # Plot learned trajectories at different stages
    ax_3d.plot(trajectories_at_stages[2][:, 0], trajectories_at_stages[2][:, 1], trajectories_at_stages[2][:, 2],
               'b--', linewidth=2, label='After 2 demos', alpha=0.7)
    ax_3d.plot(trajectories_at_stages[4][:, 0], trajectories_at_stages[4][:, 1], trajectories_at_stages[4][:, 2],
               'g--', linewidth=2, label='After 4 demos', alpha=0.7)
    ax_3d.plot(trajectories_at_stages[5][:, 0], trajectories_at_stages[5][:, 1], trajectories_at_stages[5][:, 2],
               'r-', linewidth=3, label='After 5 demos (final)', alpha=0.9)
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory Comparison: Incremental Learning')
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    
    # Plot individual dimensions (X, Y, Z)
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    positions = [2, 3, 5]
    
    for dim_idx, (label, pos) in enumerate(zip(labels, positions)):
        ax = fig.add_subplot(2, 3, pos)
        
        # Plot all demonstrations
        for i, demo in enumerate(demos):
            ax.plot(time_demos, demo[:, dim_idx], 
                   color=colors_demo[i], alpha=0.2, linewidth=1)
        
        # Plot learned trajectories at different stages
        ax.plot(time_2, trajectories_at_stages[2][:, dim_idx], 
               'b--', linewidth=2, label='After 2 demos', alpha=0.7)
        ax.plot(time_4, trajectories_at_stages[4][:, dim_idx], 
               'g--', linewidth=2, label='After 4 demos', alpha=0.7)
        ax.plot(time_5, trajectories_at_stages[5][:, dim_idx], 
               'r-', linewidth=3, label='After 5 demos (final)', alpha=0.9)
        
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel(label)
        ax.set_title(f'{label} - Incremental Learning Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Stepwise EM ProMP Incremental Learning: Demonstrations vs Learned Trajectories', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = 'stepwise_em_incremental_learning.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Close figure to free memory
    plt.close()
    
    # Print trajectory statistics
    print("\n" + "="*70)
    print("TRAJECTORY STATISTICS")
    print("="*70)
    for stage, traj in trajectories_at_stages.items():
        print(f"\nAfter {stage} demos:")
        print(f"  X range: [{np.min(traj[:, 0]):.4f}, {np.max(traj[:, 0]):.4f}] m")
        print(f"  Y range: [{np.min(traj[:, 1]):.4f}, {np.max(traj[:, 1]):.4f}] m")
        print(f"  Z range: [{np.min(traj[:, 2]):.4f}, {np.max(traj[:, 2]):.4f}] m")
        print(f"  Mean position: ({np.mean(traj[:, 0]):.4f}, {np.mean(traj[:, 1]):.4f}, {np.mean(traj[:, 2]):.4f}) m")

if __name__ == '__main__':
    try:
        test_incremental_learning()
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

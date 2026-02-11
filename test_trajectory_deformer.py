#!/usr/bin/env python3
"""
Test script for TrajectoryDeformer
Generates a trajectory from 0 to 5s, applies 3N force at 2s, and plots results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trajectory_deformer import TrajectoryDeformer

def generate_test_trajectory(duration=5.0, dt=0.01):
    """
    Generate a 3D test trajectory from 0 to duration seconds
    
    Args:
        duration: Total duration in seconds
        dt: Time step in seconds
        
    Returns:
        trajectory: Array of shape (N, 6) with [x, y, z, alpha, beta, gamma]
                   where x, y, z are 3D positions, orientation is kept minimal
    """
    t = np.arange(0, duration, dt)
    n_points = len(t)
    
    # Generate a 3D trajectory: moving in 3D space
    # Start at (0, 0, 0.5), move to (0.3, 0.2, 0.6) with smooth 3D motion
    x = 0.3 * (t / duration)  # Linear motion in x: 0 -> 0.3m
    y = 0.2 * np.sin(10*np.pi * t / duration)  # Sinusoidal motion in y: 0 -> 0.2m -> 0
    z = 0.5 + 0.1 * (t / duration) + 0.05 * np.sin(20 * np.pi * t / duration)  # Combined linear + sinusoidal in z
    
    # Keep orientation minimal (zeros) to focus on 3D position deformation
    alpha = np.zeros_like(t)  # Roll = 0
    beta = np.zeros_like(t)   # Pitch = 0
    gamma = np.zeros_like(t)  # Yaw = 0
    
    trajectory = np.column_stack([x, y, z, alpha, beta, gamma])
    
    return trajectory, t

def test_trajectory_deformer():
    """Test the TrajectoryDeformer with a force applied at 2s"""
    
    # Generate trajectory
    print("Generating test trajectory (0 to 5s)...")
    trajectory, time_steps = generate_test_trajectory(duration=5.0, dt=0.01)
    n_points = len(trajectory)
    print(f"Generated trajectory with {n_points} points")
    
    # Create deformer
    alpha = 0.0000001  # Deformation sensitivity parameter μ (reduced for smaller deformation)
    n_waypoints = 50  # Number of waypoints to deform
    energy_threshold = 0.5
    
    deformer = TrajectoryDeformer(
        alpha=alpha,
        n_waypoints=n_waypoints,
        energy_threshold=energy_threshold
    )
    
    # Set trajectory
    deformer.set_trajectory(trajectory)
    print(f"Deformer initialized with alpha={alpha}, n_waypoints={n_waypoints}")
    
    # Find the index corresponding to 2 seconds
    target_time = 2.0
    time_idx = int(target_time / (5.0 / n_points))
    if time_idx >= n_points:
        time_idx = n_points - 1
    
    # Set current waypoint index to just before 2s
    deformer.curr_waypt_idx = max(0, time_idx - 1)
    print(f"Applying force at t={target_time}s (index {deformer.curr_waypt_idx})")
    
    # Apply force in 3 directions simultaneously: 3N in x, 2N in y, 1N in z
    # Force vector: [fx, fy, fz, tx, ty, tz] = [3N, 2N, 1N, 0, 0, 0]
    force_vector = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    print(f"Applying 3D force: fx={force_vector[0]}N, fy={force_vector[1]}N, fz={force_vector[2]}N")
    print(f"Full force vector: {force_vector}")
    
    # Deform trajectory
    deformed_traj, original_traj, energy = deformer.deform(force_vector)
    
    if deformed_traj is None:
        print("ERROR: Deformation failed!")
        return
    
    print(f"Deformation energy: {energy:.6f}")
    print(f"Deformation region: indices {deformer.get_deformation_region()}")
    
    # Plot results - focus on 3D position
    fig = plt.figure(figsize=(18, 12))
    
    # Create 3D trajectory plot
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
    
    # Plot 3D trajectories
    ax_3d.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], 
               'b-', label='Original 3D Trajectory', linewidth=2, alpha=0.7)
    ax_3d.plot(deformed_traj[:, 0], deformed_traj[:, 1], deformed_traj[:, 2], 
               'r--', label='Deformed 3D Trajectory', linewidth=2, alpha=0.7)
    
    # Mark force application point
    deform_region = list(deformer.get_deformation_region())
    if len(deform_region) > 0:
        force_idx = deformer.curr_waypt_idx
        ax_3d.scatter(original_traj[force_idx, 0], original_traj[force_idx, 1], original_traj[force_idx, 2],
                     color='green', s=100, marker='*', label='Force Applied', zorder=5)
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory Comparison')
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    
    # Plot individual dimensions (x, y, z)
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    colors = ['red', 'green', 'blue']
    positions = [2, 3, 5]  # Subplot positions
    
    for i in range(3):
        ax = fig.add_subplot(2, 3, positions[i])
        
        # Plot original trajectory
        ax.plot(time_steps, original_traj[:, i], 'b-', label='Original', linewidth=2, alpha=0.7)
        
        # Plot deformed trajectory
        ax.plot(time_steps, deformed_traj[:, i], 'r--', label='Deformed', linewidth=2, alpha=0.7)
        
        # Highlight deformation region
        if len(deform_region) > 0:
            deform_start_time = time_steps[deform_region[0]]
            deform_end_time = time_steps[deform_region[-1]]
            ax.axvspan(deform_start_time, deform_end_time, alpha=0.2, color='yellow', label='Deformation Region')
        
        # Mark the force application time (2s)
        ax.axvline(x=target_time, color='green', linestyle=':', linewidth=2, label='Force Applied')
        
        # Add force vector annotation
        force_magnitude = force_vector[i]
        if force_magnitude > 0:
            ax.text(0.02, 0.98, f'Force: {force_magnitude}N', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{labels[i]} Position')
    
    # Plot orientation (should be minimal/zero)
    ax_orient = fig.add_subplot(2, 3, 6)
    for i in range(3, 6):
        ax_orient.plot(time_steps, original_traj[:, i], '--', alpha=0.5, 
                      label=['Alpha', 'Beta', 'Gamma'][i-3])
    ax_orient.set_xlabel('Time (s)')
    ax_orient.set_ylabel('Orientation (rad)')
    ax_orient.set_title('Orientation (should be ~0)')
    ax_orient.legend()
    ax_orient.grid(True, alpha=0.3)
    
    plt.suptitle(f'3D Trajectory Deformation Test: Force Applied at t={target_time}s\n'
                 f'Force: fx={force_vector[0]}N, fy={force_vector[1]}N, fz={force_vector[2]}N | '
                 f'α={alpha}, n_waypoints={n_waypoints}, Energy={energy:.6f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = 'trajectory_deformation_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("DEFORMATION STATISTICS")
    print("="*60)
    print(f"Original trajectory shape: {original_traj.shape}")
    print(f"Deformed trajectory shape: {deformed_traj.shape}")
    print(f"Deformation energy: {energy:.6f}")
    print(f"Deformation region: {deform_region}")
    print(f"\n3D Position Deformations:")
    print(f"  Max deformation (X): {np.max(np.abs(deformed_traj[:, 0] - original_traj[:, 0])):.6f} m")
    print(f"  Max deformation (Y): {np.max(np.abs(deformed_traj[:, 1] - original_traj[:, 1])):.6f} m")
    print(f"  Max deformation (Z): {np.max(np.abs(deformed_traj[:, 2] - original_traj[:, 2])):.6f} m")
    
    # Calculate 3D displacement magnitude
    position_diff = deformed_traj[:, :3] - original_traj[:, :3]
    displacement_magnitude = np.linalg.norm(position_diff, axis=1)
    max_displacement = np.max(displacement_magnitude)
    print(f"  Max 3D displacement magnitude: {max_displacement:.6f} m")
    
    # Show deformation at force application point
    if len(deform_region) > 0:
        force_idx = deformer.curr_waypt_idx
        print(f"\nDeformation at force application point (t={target_time}s, idx={force_idx}):")
        print(f"  Original position: ({original_traj[force_idx, 0]:.6f}, {original_traj[force_idx, 1]:.6f}, {original_traj[force_idx, 2]:.6f}) m")
        print(f"  Deformed position: ({deformed_traj[force_idx, 0]:.6f}, {deformed_traj[force_idx, 1]:.6f}, {deformed_traj[force_idx, 2]:.6f}) m")
        print(f"  Displacement: ({deformed_traj[force_idx, 0] - original_traj[force_idx, 0]:.6f}, "
              f"{deformed_traj[force_idx, 1] - original_traj[force_idx, 1]:.6f}, "
              f"{deformed_traj[force_idx, 2] - original_traj[force_idx, 2]:.6f}) m")
    
    # Check if ProMP conditioning should be triggered
    if deformer.should_condition_promp():
        print(f"Energy ({energy:.6f}) < threshold ({energy_threshold}) - ProMP conditioning recommended")
    else:
        print(f"Energy ({energy:.6f}) >= threshold ({energy_threshold}) - High deformation detected")

if __name__ == '__main__':
    print("="*60)
    print("TRAJECTORY DEFORMER TEST")
    print("="*60)
    print()
    
    try:
        test_trajectory_deformer()
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

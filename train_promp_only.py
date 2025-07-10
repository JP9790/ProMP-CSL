#!/usr/bin/env python3

import numpy as np
from promp import ProMP
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train ProMP and save learned trajectory (no execution)')
    parser.add_argument('--demo-file', default='all_demos.npy', help='Path to demos file (npy, 3D or object array)')
    parser.add_argument('--trajectory-file', default='learned_trajectory.npy', help='Where to save the learned trajectory')
    parser.add_argument('--promp-file', default='promp_model.npy', help='Where to save the trained ProMP parameters')
    parser.add_argument('--num-basis', type=int, default=50, help='Number of ProMP basis functions')
    parser.add_argument('--sigma-noise', type=float, default=0.01, help='ProMP noise parameter')
    parser.add_argument('--trajectory-points', type=int, default=100, help='Number of points in generated trajectory')
    args = parser.parse_args()

    # Load demos
    demos = np.load(args.demo_file, allow_pickle=True)
    demos_list = []
    if isinstance(demos, np.ndarray) and demos.dtype == object:
        for demo in demos:
            arr = np.array(demo)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    demos_list.append(arr[i])
            else:
                demos_list.append(arr)
        demos = demos_list
    elif isinstance(demos, np.ndarray):
        if demos.ndim == 3:
            for i in range(demos.shape[0]):
                demos_list.append(demos[i])
            demos = demos_list
        else:
            demos = [demos]
    print(f"Loaded {len(demos)} demos.")

    # Normalize demos
    target_length = args.trajectory_points
    normalized = []
    for demo in demos:
        demo_array = np.array(demo)
        t_old = np.linspace(0, 1, len(demo_array))
        t_new = np.linspace(0, 1, target_length)
        normalized_demo = []
        for i in range(demo_array.shape[1]):
            from scipy.interpolate import interp1d
            interp_func = interp1d(t_old, demo_array[:, i], kind='cubic')
            normalized_demo.append(interp_func(t_new))
        normalized.append(np.column_stack(normalized_demo))
    normalized = np.stack(normalized, axis=0)  # (num_demos, target_length, dof)
    print(f"Normalized demos shape: {normalized.shape}")

    # Train ProMP
    promp = ProMP(num_basis=args.num_basis, sigma_noise=args.sigma_noise)
    promp.train(normalized)

    # Generate learned trajectory
    learned_traj = promp.generate_trajectory(num_points=target_length)
    print(f"Learned trajectory shape: {learned_traj.shape}")

    # Save learned trajectory
    np.save(args.trajectory_file, learned_traj)
    print(f"Saved learned trajectory to {args.trajectory_file}")

    # Save ProMP parameters
    promp_data = {
        'mean_weights': promp.mean_weights,
        'cov_weights': promp.cov_weights,
        'basis_centers': promp.basis_centers,
        'basis_width': promp.basis_width
    }
    np.save(args.promp_file, promp_data)
    print(f"Saved ProMP model to {args.promp_file}")

if __name__ == '__main__':
    main() 
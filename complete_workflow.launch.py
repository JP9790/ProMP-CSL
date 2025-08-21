from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # KUKA Communication Parameters
        DeclareLaunchArgument(
            'kuka_ip',
            default_value='172.31.1.147',
            description='IP address of KUKA robot (all-in-one Java app)'
        ),
        
        DeclareLaunchArgument(
            'kuka_port',
            default_value='30002',
            description='Port for KUKA communication'
        ),
        
        DeclareLaunchArgument(
            'save_directory',
            default_value='~/robot_demos',
            description='Directory to save demo files'
        ),
        
        # File Paths
        DeclareLaunchArgument(
            'trajectory_file',
            default_value='learned_trajectory.npy',
            description='Path to learned trajectory file'
        ),
        
        DeclareLaunchArgument(
            'promp_file',
            default_value='promp_model.npy',
            description='Path to ProMP model file'
        ),
        
        # Deformation Parameters
        DeclareLaunchArgument(
            'energy_threshold',
            default_value='0.5',
            description='Energy threshold for ProMP conditioning'
        ),
        
        DeclareLaunchArgument(
            'force_threshold',
            default_value='10.0',
            description='Force threshold for deformation detection'
        ),
        
        DeclareLaunchArgument(
            'torque_threshold',
            default_value='2.0',
            description='Torque threshold for deformation detection'
        ),
        
        # ProMP Parameters
        DeclareLaunchArgument(
            'num_basis_functions',
            default_value='50',
            description='Number of basis functions for ProMP'
        ),
        
        DeclareLaunchArgument(
            'sigma_noise',
            default_value='0.01',
            description='Noise parameter for ProMP'
        ),
        
        # All-in-One Controller Node (for demo recording and basic control)
        Node(
            package='kuka_promp_control',
            executable='control_script',
            name='all_in_one_controller',
            parameters=[{
                'kuka_ip': LaunchConfiguration('kuka_ip'),
                'kuka_port': LaunchConfiguration('kuka_port'),
                'save_directory': LaunchConfiguration('save_directory'),
            }],
            output='screen'
        ),
        
        # ProMP Training Node (for training on recorded demos)
        Node(
            package='kuka_promp_control',
            executable='train_promp_only_node',
            name='promp_trainer',
            parameters=[{
                'demo_file': 'all_demos.npy',
                'trajectory_file': LaunchConfiguration('trajectory_file'),
                'promp_file': LaunchConfiguration('promp_file'),
                'num_basis': LaunchConfiguration('num_basis_functions'),
                'sigma_noise': LaunchConfiguration('sigma_noise'),
                'trajectory_points': 100,
            }],
            output='screen'
        ),
        
        # Deformation Controller Node (for execution with deformation)
        Node(
            package='kuka_promp_control',
            executable='standalone_deformation_controller',
            name='deformation_controller',
            parameters=[{
                # KUKA Communication
                'kuka_ip': LaunchConfiguration('kuka_ip'),
                'kuka_port': LaunchConfiguration('kuka_port'),
                'torque_port': 30003,
                
                # File Paths
                'trajectory_file': LaunchConfiguration('trajectory_file'),
                'promp_file': LaunchConfiguration('promp_file'),
                
                # Deformation Parameters
                'energy_threshold': LaunchConfiguration('energy_threshold'),
                'force_threshold': LaunchConfiguration('force_threshold'),
                'torque_threshold': LaunchConfiguration('torque_threshold'),
                'deformation_alpha': 0.1,
                'deformation_waypoints': 10,
                
                # ProMP Parameters
                'promp_conditioning_sigma': 0.01,
                'num_basis_functions': LaunchConfiguration('num_basis_functions'),
                'sigma_noise': LaunchConfiguration('sigma_noise'),
                
                # EM Learning Parameters
                'em_learning_rate': 0.1,
                'em_convergence_tolerance': 1e-4,
                'em_min_iterations': 5,
            }],
            output='screen'
        )
    ]) 
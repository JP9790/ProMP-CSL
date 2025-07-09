from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # KUKA Communication Parameters
        DeclareLaunchArgument(
            'kuka_ip',
            default_value='172.31.1.25',
            description='IP address of KUKA robot'
        ),
        
        DeclareLaunchArgument(
            'kuka_port',
            default_value='30002',
            description='Port for KUKA communication'
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
        
        DeclareLaunchArgument(
            'deformation_alpha',
            default_value='0.1',
            description='Deformation sensitivity parameter'
        ),
        
        DeclareLaunchArgument(
            'deformation_waypoints',
            default_value='10',
            description='Number of waypoints to deform'
        ),
        
        # ProMP Parameters
        DeclareLaunchArgument(
            'promp_conditioning_sigma',
            default_value='0.01',
            description='Sigma for ProMP conditioning'
        ),
        
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
        
        # EM Learning Parameters
        DeclareLaunchArgument(
            'em_learning_rate',
            default_value='0.1',
            description='Learning rate for EM algorithm'
        ),
        
        DeclareLaunchArgument(
            'em_convergence_tolerance',
            default_value='1e-4',
            description='Convergence tolerance for EM algorithm'
        ),
        
        DeclareLaunchArgument(
            'em_min_iterations',
            default_value='5',
            description='Minimum iterations for EM convergence check'
        ),
        
        # Node
        Node(
            package='kuka_promp_control',
            executable='standalone_deformation_controller',
            name='deformation_controller',
            parameters=[{
                # KUKA Communication
                'kuka_ip': LaunchConfiguration('kuka_ip'),
                'kuka_port': LaunchConfiguration('kuka_port'),
                
                # File Paths
                'trajectory_file': LaunchConfiguration('trajectory_file'),
                'promp_file': LaunchConfiguration('promp_file'),
                
                # Deformation Parameters
                'energy_threshold': LaunchConfiguration('energy_threshold'),
                'force_threshold': LaunchConfiguration('force_threshold'),
                'torque_threshold': LaunchConfiguration('torque_threshold'),
                'deformation_alpha': LaunchConfiguration('deformation_alpha'),
                'deformation_waypoints': LaunchConfiguration('deformation_waypoints'),
                
                # ProMP Parameters
                'promp_conditioning_sigma': LaunchConfiguration('promp_conditioning_sigma'),
                'num_basis_functions': LaunchConfiguration('num_basis_functions'),
                'sigma_noise': LaunchConfiguration('sigma_noise'),
                
                # EM Learning Parameters
                'em_learning_rate': LaunchConfiguration('em_learning_rate'),
                'em_convergence_tolerance': LaunchConfiguration('em_convergence_tolerance'),
                'em_min_iterations': LaunchConfiguration('em_min_iterations'),
            }],
            output='screen'
        )
    ])
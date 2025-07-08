from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'kuka_ip',
            default_value='192.168.1.50',
            description='IP address of KUKA robot'
        ),
        
        DeclareLaunchArgument(
            'kuka_port',
            default_value='30002',
            description='Port for KUKA communication'
        ),
        
        Node(
            package='kuka_promp_control',
            executable='demo_recorder',
            name='demo_recorder',
            parameters=[{
                'kuka_ip': LaunchConfiguration('kuka_ip'),
                'kuka_port': LaunchConfiguration('kuka_port'),
                'record_frequency': 100.0,
                'demo_duration': 10.0,
                'num_basis_functions': 50,
                'sigma_noise': 0.01
            }],
            output='screen'
        )
    ])
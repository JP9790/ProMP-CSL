from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'kuka_ip',
            default_value='192.170.10.25',  # KUKA robot IP (same network as PC)
            description='IP address of KUKA robot'
        ),
        
        DeclareLaunchArgument(
            'ros2_pc_ip',
            default_value='192.170.10.1',  # ROS2 PC IP (what KUKA connects to)
            description='IP address of ROS2 PC (what KUKA connects to)'
        ),
        
        DeclareLaunchArgument(
            'kuka_port',
            default_value='30002',
            description='Port for KUKA trajectory communication'
        ),
        
        DeclareLaunchArgument(
            'torque_port',
            default_value='30003',
            description='Port for receiving torque data from KUKA'
        ),
        
        DeclareLaunchArgument(
            'save_directory',
            default_value='~/robot_demos',
            description='Directory to save individual and combined demo files'
        ),
        
        DeclareLaunchArgument(
            'demo_duration',
            default_value='10.0',
            description='Duration of each demo recording in seconds'
        ),
        
        DeclareLaunchArgument(
            'record_frequency',
            default_value='100.0',
            description='Recording frequency in Hz'
        ),
        
        DeclareLaunchArgument(
            'auto_save',
            default_value='true',
            description='Automatically save individual demos after recording'
        ),
        
        Node(
            package='kuka_promp_control',
            executable='interactive_demo_recorder',
            name='interactive_demo_recorder',
            parameters=[{
                'kuka_ip': LaunchConfiguration('kuka_ip'),
                'kuka_port': LaunchConfiguration('kuka_port'),
                'torque_port': LaunchConfiguration('torque_port'),
                'ros2_pc_ip': LaunchConfiguration('ros2_pc_ip'),
                'save_directory': LaunchConfiguration('save_directory'),
                'demo_duration': LaunchConfiguration('demo_duration'),
                'record_frequency': LaunchConfiguration('record_frequency'),
                'auto_save': LaunchConfiguration('auto_save'),
                'num_basis_functions': 50,
                'sigma_noise': 0.01,
                'force_threshold': 10.0,
                'torque_threshold': 2.0,
                'enable_human_interaction': True,
                'send_torque_data': True
            }],
            output='screen'
        )
    ]) 
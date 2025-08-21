from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'kuka_ip',
            default_value='172.31.1.147',  # KUKA robot IP (same network as PC)
            description='IP address of KUKA robot'
        ),
        
        DeclareLaunchArgument(
            'ros2_pc_ip',
            default_value='172.31.1.100',  # ROS2 PC IP (what KUKA connects to)
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
        

        
        Node(
            package='kuka_promp_control',
            executable='demo_recorder',
            name='demo_recorder',
            parameters=[{
                'kuka_ip': LaunchConfiguration('kuka_ip'),
                'kuka_port': LaunchConfiguration('kuka_port'),
                'torque_port': LaunchConfiguration('torque_port'),
                'ros2_pc_ip': LaunchConfiguration('ros2_pc_ip'),
                'record_frequency': 100.0,
                'demo_duration': 10.0,
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
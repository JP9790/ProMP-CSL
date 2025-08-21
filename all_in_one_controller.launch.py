from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
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
        
        # All-in-One Controller Node
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
        )
    ]) 
# ProMP-CSL
Balancing short&amp; long term adaptation in ProMPs
# Usage
1. Create package:
    ```shell
    ros2 pkg create --build-type ament_python kuka_promp_control
    cd kuka_promp_control
    ```
2. Add package dependencies:
    ```shell
    <depend>rclpy</depend>
    <depend>geometry_msgs</depend>
    <depend>std_msgs</depend>
    <depend>sensor_msgs</depend>
    <depend>tf2_ros</depend>
    <depend>tf2_geometry_msgs</depend>
    <depend>numpy</depend>
    <depend>scipy</depend>
    <depend>matplotlib</depend>
    ```
3. Build and Run
    ```shell
    # Build the package
    colcon build --packages-select kuka_promp_control

    # Source the workspace
    source install/setup.bash

    # Launch the demo recorder
    ros2 launch kuka_promp_control demo_recorder.launch.py kuka_ip:=192.168.1.50
    ```
4. Record Demo
    ```shell
    # In another terminal, run the control script
    ros2 run kuka_promp_control control_script
    ```
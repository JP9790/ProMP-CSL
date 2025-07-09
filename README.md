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
4. Initial Record 
    ```shell
    # In another terminal, run the control script
    ros2 run kuka_promp_control control_script
    ```
5. Manual Record
    ```shell
    # Start recording
    ros2 topic pub /start_recording std_msgs/msg/Bool "data: true"

    # Stop recording
    ros2 topic pub /start_recording std_msgs/msg/Bool "data: false"
    ```
6. Train ProMP
6.1. Update `setup.py` to include the new script
    ```shell    
    entry_points={
        'console_scripts': [
            'demo_recorder = kuka_promp_control.demo_recorder:main',
            'control_script = kuka_promp_control.control_script:main',
            'train_and_execute = kuka_promp_control.train_and_execute:main',
            'standalone_deformation_controller = kuka_promp_control.standalone_deformation_controller:main',  # Add this line
        ],
    },      
    ```
6.2. Execution
- Complete Pipeline
    ```shell
    ros2 run kuka_promp_control train_and_execute
    ```
- Train Only (with visualization)
    ```shell
    ros2 run kuka_promp_control train_and_execute --train-only --visualize
    ```
- Execute Only (load pre-trained trajectory)
    ```shell
    ros2 run kuka_promp_control train_and_execute --execute-only
    ```
-  Load Specific Trajectory File
    ```shell
    ros2 run kuka_promp_control train_and_execute --load-trajectory my_trajectory.npy--visualize
    ```
7. Deformation:
7.1 Basic Usage (with default parameters)
    ```shell
    # Build the package
    colcon build --packages-select kuka_promp_control

    # Source the workspace
    source install/setup.bash

    # Launch with default parameters
    ros2 launch kuka_promp_control deformation_controller.launch.py
    ```
7.2 Custom Deformation Parameters
    ```shell
    ros2 launch kuka_promp_control deformation_controller.launch.py \
        energy_threshold:=0.3 \
        force_threshold:=8.0 \
        torque_threshold:=1.5 \
        deformation_alpha:=0.15
    ```
7.3 Custom EM Learning Parameters
    ```shell
    ros2 launch kuka_promp_control deformation_controller.launch.py \
        em_learning_rate:=0.2 \
        em_convergence_tolerance:=1e-5 \
        em_min_iterations:=10
    ```
7.3 Monitor Status:
    ```shell
    # Monitor all status topics
    ros2 topic echo /deformation_status
    ros2 topic echo /deformation_energy
    ros2 topic echo /conditioning_status
    ros2 topic echo /execution_status
    ros2 topic echo /deformation_statistics
    ```

8. Work Flow
    ```shell
    ros2 run kuka_promp_control demo_recorder
    ros2 run kuka_promp_control control_script
    ros2 run kuka_promp_control train_and_execute --train-only
    ros2 launch kuka_promp_control deformation_controller.launch.py kuka_ip:=192.168.1.50
    ros2 topic pub /start_deformation_execution std_msgs/msg/Bool "data: true"


    ```
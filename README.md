# ProMP-CSL
Balancing short&amp; long term adaptation in ProMPs

# Usage

# 1. Create package:
    ```shell
    ros2 pkg create --build-type ament_python kuka_promp_control
    cd kuka_promp_control
    ```

# 2. Setup package structure:
    ```shell
    # Create required directories
    mkdir -p kuka_promp_control
    mkdir -p launch
    mkdir -p resource
    mkdir -p test
    
    # Create __init__.py
    touch kuka_promp_control/__init__.py
    
    # Create resource file
    echo "kuka_promp_control" > resource/kuka_promp_control
    ```

# 3. Add package dependencies to `package.xml`:
    ```xml
    <?xml version="1.0"?>
    <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
    <package format="3">
      <name>kuka_promp_control</name>
      <version>0.0.1</version>
      <description>ProMP-CSL: Balancing short & long term adaptation in ProMPs</description>
      <maintainer email="your_email@example.com">your_name</maintainer>
      <license>MIT</license>

      <depend>rclpy</depend>
      <depend>geometry_msgs</depend>
      <depend>std_msgs</depend>
      <depend>sensor_msgs</depend>
      <depend>tf2_ros</depend>
      <depend>tf2_geometry_msgs</depend>

      <test_depend>ament_copyright</test_depend>
      <test_depend>ament_flake8</test_depend>
      <test_depend>ament_pep257</test_depend>
      <test_depend>python3-pytest</test_depend>

      <export>
        <build_type>ament_python</build_type>
      </export>
    </package>
    ```

# 4. Create `setup.py`:
    ```python
    from setuptools import setup
    import os
    from glob import glob

    package_name = 'kuka_promp_control'

    setup(
        name=package_name,
        version='0.0.1',
        packages=[package_name],
        data_files=[
            ('share/ament_index/resource_index/packages',
                ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='your_name',
        maintainer_email='your_email@example.com',
        description='ProMP-CSL: Balancing short & long term adaptation in ProMPs',
        license='MIT',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'demo_recorder = kuka_promp_control.demo_recorder:main',
                'control_script = kuka_promp_control.control_script:main',
                'train_and_execute = kuka_promp_control.train_and_execute:main',
                'standalone_deformation_controller = kuka_promp_control.standalone_deformation_controller:main',
            ],
        },
    )
    ```

# 5. Copy Python files:
    ```shell
    # Copy all Python files to kuka_promp_control/ directory
    cp demo_recorder.py kuka_promp_control/
    cp control_script.py kuka_promp_control/
    cp train_and_execute.py kuka_promp_control/
    cp standalone_deformation_controller.py kuka_promp_control/
    cp promp.py kuka_promp_control/
    cp trajectory_deformer.py kuka_promp_control/
    cp stepwise_em_learner.py kuka_promp_control/
    
    # Copy launch files
    cp demo_recorder.launch.py launch/
    cp deformation_controller.launch.py launch/
    ```

# 6. Install Python dependencies:
    ```shell
    pip3 install numpy scipy matplotlib
    ```

# 7. Build and Run
    ```shell
    # Build the package
    colcon build --packages-select kuka_promp_control

    # Source the workspace
    source install/setup.bash

    # Launch the demo recorder
    ros2 launch kuka_promp_control demo_recorder.launch.py kuka_ip:=192.170.1.100
    ```

# 8. Initial Record 
    ```shell
    # In another terminal, run the control script
    ros2 run kuka_promp_control control_script
    ```

# 9. Manual Record
    ```shell
    # Start recording
    ros2 topic pub /start_recording std_msgs/msg/Bool "data: true"

    # Stop recording
    ros2 topic pub /start_recording std_msgs/msg/Bool "data: false"
    ```

# 10. Check Recording Status
    ```shell
    # Monitor recording status
    ros2 topic echo /demo_status
    
    # List available topics
    ros2 topic list
    ```

# 11. Train ProMP

## 11.1. Execution
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
- Load Specific Trajectory File
    ```shell
    ros2 run kuka_promp_control train_and_execute --load-trajectory my_trajectory.npy --visualize
    ```

# 12. Deformation:
## 12.1 Basic Usage (with default parameters)
    ```shell
    # Build the package
    colcon build --packages-select kuka_promp_control

    # Source the workspace
    source install/setup.bash

    # Launch with default parameters
    ros2 launch kuka_promp_control deformation_controller.launch.py
    ```

## 12.2 Custom Deformation Parameters
    ```shell
    ros2 launch kuka_promp_control deformation_controller.launch.py \
        energy_threshold:=0.3 \
        force_threshold:=8.0 \
        torque_threshold:=1.5 \
        deformation_alpha:=0.15
    ```

## 12.3 Custom EM Learning Parameters
    ```shell
    ros2 launch kuka_promp_control deformation_controller.launch.py \
        em_learning_rate:=0.2 \
        em_convergence_tolerance:=1e-5 \
        em_min_iterations:=10
    ```

## 12.4 Monitor Status:
    ```shell
    # Monitor all status topics
    ros2 topic echo /deformation_status
    ros2 topic echo /deformation_energy
    ros2 topic echo /conditioning_status
    ros2 topic echo /execution_status
    ros2 topic echo /deformation_statistics
    ```

## 12.5 Control Deformation Execution:
    ```shell
    # Start deformation execution
    ros2 topic pub /start_deformation_execution std_msgs/msg/Bool "data: true"
    
    # Stop deformation execution
    ros2 topic pub /stop_deformation_execution std_msgs/msg/Bool "data: true"
    
    # Load new trajectory
    ros2 topic pub /load_trajectory std_msgs/msg/String "data: 'new_trajectory.npy'"
    ```

# 13. Troubleshooting:
    ```shell
    # Check if package is built correctly
    ros2 pkg list | grep kuka_promp_control
    
    # Check if nodes are available
    ros2 node list
    
    # Check if topics are available
    ros2 topic list
    
    # Check node info
    ros2 node info /demo_recorder
    
    # Check parameter values
    ros2 param list /demo_recorder
    ros2 param get /demo_recorder kuka_ip
    ```

# 14. Work Flow
    ```shell
    # Terminal 1: Start demo recorder
    ros2 launch kuka_promp_control demo_recorder.launch.py kuka_ip:=192.170.1.100
    
    # Terminal 2: Run control script for automatic recording
    ros2 run kuka_promp_control control_script
    
    # Terminal 3: Train ProMP
    ros2 run kuka_promp_control train_and_execute --train-only --visualize
    
    # Terminal 4: Start deformation controller
    ros2 launch kuka_promp_control deformation_controller.launch.py kuka_ip:=192.170.1.100
    
    # Terminal 5: Start deformation execution
    ros2 topic pub /start_deformation_execution std_msgs/msg/Bool "data: true"
    ```

# 15. Java Application Setup:
    ```shell
    # Copy Java files to KUKA Sunrise Workbench project
    # 1. Create new KUKA Sunrise Project
    # 2. Copy cartesianimpedance.java to src/application/
    # 3. Copy RoboticsAPI.data.xml to project root
    # 4. Build and deploy to KUKA robot
    # 5. Ensure robot IP matches configuration (192.170.1.100)
    ```
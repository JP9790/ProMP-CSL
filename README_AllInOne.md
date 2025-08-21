# KUKA ProMP Control - All-in-One Workflow

This README describes the **all-in-one workflow** using the `FlexibleCartesianImpedance.java` application for KUKA robot control with ProMP learning and real-time deformation adaptation.

## 🎯 Overview

The all-in-one workflow consolidates demo recording, ProMP training, and deformation execution into a single Java application (`FlexibleCartesianImpedance.java`) that communicates with Python ROS2 nodes for machine learning and trajectory adaptation.

### Key Features:
- ✅ **Single Java Application** - All robot control in one place
- ✅ **Impedance Control** - Robot is compliant and responds to user interaction
- ✅ **Real-time Deformation** - Adapts trajectory based on external forces
- ✅ **Interactive Demo Recording** - Record multiple demonstrations
- ✅ **ProMP Learning** - Learn from demonstrations
- ✅ **Launch File Support** - Easy one-command startup

## 🏗️ Architecture

```
┌─────────────────┐    TCP/IP    ┌─────────────────┐    ROS2 Topics    ┌─────────────────┐
│   KUKA Robot    │◄────────────►│  Java App       │◄─────────────────►│  Python Nodes   │
│                 │   Port 30002 │FlexibleCartesian│                   │                 │
│ - Impedance     │              │Impedance.java   │                   │ - Demo Recorder │
│ - Force Sensing │              │                 │                   │ - ProMP Trainer │
│ - Motion Control│              │ - Demo Recording│                   │ - Deformation   │
└─────────────────┘              │ - Trajectory    │                   │   Controller    │
                                 │   Execution     │                   └─────────────────┘
                                 │ - Force Sending │
                                 └─────────────────┘
```

## 📋 Prerequisites

### Hardware:
- KUKA iiwa robot with Sunrise Workbench
- Force/torque sensor (optional, for deformation detection)
- Network connection between robot and PC

### Software:
- Java 1.7+ (for KUKA Robotics API)
- Python 3.8+ with ROS2
- Required Python packages: `numpy`, `scipy`, `rclpy`

### Network Configuration:
- **KUKA Robot IP**: `172.31.1.147` (default)
- **ROS2 PC IP**: `172.31.1.100` (default)
- **Communication Port**: `30002` (Java-Python)
- **Force Data Port**: `30003` (Java-Python)

## 🚀 Quick Start

### 1. Start the Java Application

Deploy and start `FlexibleCartesianImpedance.java` on the KUKA robot:

```java
// The robot will automatically:
// - Move to initial position (pointing away from wall)
// - Start TCP server on port 30002
// - Enable impedance control mode
// - Wait for Python commands
```

### 2. Launch the Complete Workflow

```bash
# Launch everything at once
ros2 launch kuka_promp_control complete_workflow.launch.py \
  kuka_ip:=172.31.1.147 \
  save_directory:=~/robot_demos
```

### 3. Record Demonstrations

```bash
# Start recording a demo
ros2 topic pub /record_demo std_msgs/msg/Bool "data: true"

# Physically guide the robot through the desired motion
# The robot will be compliant and record your movements

# Stop recording
ros2 topic pub /record_demo std_msgs/msg/Bool "data: false"

# Repeat for multiple demonstrations
```

### 4. Retrieve and Train ProMP

```bash
# Retrieve all recorded demos from Java app
ros2 topic pub /get_demos std_msgs/msg/Bool "data: true"

# ProMP training happens automatically
# Check status:
ros2 topic echo /controller_status
```

### 5. Execute with Deformation

```bash
# Execute learned trajectory with deformation adaptation
ros2 topic pub /execute_trajectory std_msgs/msg/String "data: 'learned_trajectory.npy'"

# During execution, you can:
# - Push/pull the robot to deform the trajectory
# - The robot will adapt in real-time
# - Low energy: ProMP conditioning
# - High energy: Stepwise EM learning
```

## 📁 File Structure

```
ProMP-CSL/
├── application/
│   ├── FlexibleCartesianImpedance.java    # All-in-one Java app
│   └── cartesianimpedance.java            # Reference (don't modify)
├── kuka_promp_control/
│   ├── control_script.py                  # All-in-one Python controller
│   ├── interactive_demo_recorder.py       # Demo recording (updated)
│   ├── standalone_deformation_controller.py # Deformation execution
│   ├── promp.py                          # ProMP implementation
│   └── train_promp_only_node.py          # ProMP training
├── all_in_one_controller.launch.py       # Simple controller launch
├── complete_workflow.launch.py           # Complete workflow launch
└── README_AllInOne.md                    # This file
```

## 🔧 Configuration

### Java Application Parameters

```java
// Network settings (matching cartesianimpedance.java)
private String ros2PCIP = "172.31.1.100";
private int serverPort = 30002;
private int forceDataPort = 30003;

// Initial position (pointing away from wall)
private static final double[] INITIAL_POSITION = {
    Math.PI, -0.7854, 0.0, 1.3962, 0.0, 0.6109, 0.0
};

// Impedance parameters
private double stiffnessX = 1000.0;
private double stiffnessY = 1000.0;
private double stiffnessZ = 1000.0;
private double stiffnessRot = 100.0;
private double damping = 0.7;

// Force thresholds
private double forceThreshold = 10.0;
private double torqueThreshold = 2.0;
```

### Python Parameters

```python
# KUKA Communication
kuka_ip = "172.31.1.147"
kuka_port = 30002
torque_port = 30003

# File paths
save_directory = "~/robot_demos"
trajectory_file = "learned_trajectory.npy"
promp_file = "promp_model.npy"

# ProMP parameters
num_basis_functions = 50
sigma_noise = 0.01

# Deformation parameters
energy_threshold = 0.5
force_threshold = 10.0
torque_threshold = 2.0
```

## 📡 Communication Protocol

### Java → Python Commands

| Command | Description | Response |
|---------|-------------|----------|
| `START_DEMO_RECORDING` | Start recording demo | `OK` |
| `STOP_DEMO_RECORDING` | Stop recording demo | `OK` |
| `GET_DEMOS` | Retrieve all demos | `DEMOS:DEMO_0:x,y,z,a,b,g;|DEMO_1:...` |
| `CLEAR_DEMOS` | Clear all demos | `OK` |
| `GET_POSE` | Get current pose | `POSE:x,y,z,a,b,g` |
| `TRAJECTORY:data` | Execute trajectory | `POINT_COMPLETE` (per point) |
| `STOP` | Stop execution | `OK` |
| `SET_IMPEDANCE:Kx,Ky,Kz,D` | Set impedance | `OK` |

### Python → Java Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/record_demo` | `std_msgs/Bool` | Start/stop demo recording |
| `/get_demos` | `std_msgs/Bool` | Retrieve demos from Java |
| `/clear_demos` | `std_msgs/Bool` | Clear all demos |
| `/execute_trajectory` | `std_msgs/String` | Execute trajectory file |
| `/controller_status` | `std_msgs/String` | Status feedback |

## 🎮 Usage Examples

### Example 1: Simple Demo Recording

```bash
# 1. Start Java app on KUKA robot
# 2. Launch controller
ros2 launch kuka_promp_control all_in_one_controller.launch.py

# 3. Record demo
ros2 topic pub /record_demo std_msgs/msg/Bool "data: true"
# ... guide robot ...
ros2 topic pub /record_demo std_msgs/msg/Bool "data: false"

# 4. Save demos
ros2 topic pub /get_demos std_msgs/msg/Bool "data: true"
```

### Example 2: Complete Workflow

```bash
# 1. Start Java app on KUKA robot
# 2. Launch complete workflow
ros2 launch kuka_promp_control complete_workflow.launch.py

# 3. Record multiple demos
for i in {1..5}; do
  echo "Recording demo $i"
  ros2 topic pub /record_demo std_msgs/msg/Bool "data: true"
  sleep 10  # Record for 10 seconds
  ros2 topic pub /record_demo std_msgs/msg/Bool "data: false"
  sleep 2
done

# 4. Retrieve and train
ros2 topic pub /get_demos std_msgs/msg/Bool "data: true"

# 5. Execute with deformation
ros2 topic pub /execute_trajectory std_msgs/msg/String "data: 'learned_trajectory.npy'"
```

### Example 3: Custom Parameters

```bash
# Launch with custom parameters
ros2 launch kuka_promp_control complete_workflow.launch.py \
  kuka_ip:=192.168.1.100 \
  save_directory:=/home/user/custom_demos \
  force_threshold:=5.0 \
  energy_threshold:=0.3 \
  num_basis_functions:=100
```

## 🔍 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Error: Could not connect to KUKA robot
   ```
   - Check IP addresses in Java and Python
   - Verify network connectivity
   - Ensure Java app is running on robot

2. **"Illegal number of joint points"**
   ```
   Error: cannot convert ptp to xxxxxx, illegal number of joint points, should be 7
   ```
   - This is fixed in the current version
   - Java app now uses LIN motion for Cartesian data

3. **Robot Not Compliant**
   ```
   Robot is rigid during execution
   ```
   - Check impedance parameters
   - Verify `CartesianImpedanceControlMode` is active
   - Ensure force thresholds are appropriate

4. **Demo Recording Issues**
   ```
   No demos recorded
   ```
   - Check `START_DEMO_RECORDING`/`STOP_DEMO_RECORDING` commands
   - Verify `recordDemoData()` thread is running
   - Check demo data format

### Debug Commands

```bash
# Check connection
ros2 topic echo /controller_status

# Monitor force data
ros2 topic echo /force_data

# Check robot pose
ros2 topic pub /get_pose std_msgs/msg/Bool "data: true"

# Test impedance settings
ros2 topic pub /set_impedance std_msgs/msg/String "data: '1000,1000,1000,0.7'"
```

## 📊 Performance Tips

1. **Demo Recording**
   - Record at least 5-10 demonstrations
   - Keep demonstrations consistent
   - Use smooth, continuous motions

2. **ProMP Training**
   - Use 50-100 basis functions for complex motions
   - Adjust `sigma_noise` based on demonstration quality
   - Normalize trajectories for better learning

3. **Deformation Execution**
   - Start with low force thresholds
   - Gradually increase for more responsive deformation
   - Monitor energy levels for appropriate adaptation method

4. **Network Performance**
   - Use dedicated network for robot communication
   - Minimize network latency
   - Monitor TCP connection stability

## 🔄 Workflow Summary

1. **Setup**: Deploy Java app on KUKA robot
2. **Launch**: Start Python workflow with launch file
3. **Record**: Collect multiple demonstrations
4. **Train**: Learn ProMP model from demonstrations
5. **Execute**: Run trajectory with real-time deformation
6. **Adapt**: Robot responds to external forces

## 📚 References

- [KUKA Robotics API Documentation](https://www.kuka.com/en-us/products/robotics-systems/software/robotics-api)
- [ProMP Paper](https://arxiv.org/abs/1503.07619)
- [ROS2 Launch System](https://docs.ros.org/en/humble/Tutorials/Launch-Files/Creating-Launch-Files.html)

## 🤝 Contributing

This all-in-one workflow is designed to be:
- **Modular**: Easy to modify individual components
- **Configurable**: Parameter-driven behavior
- **Robust**: Error handling and recovery
- **User-friendly**: Simple launch file interface

For questions or issues, please refer to the main project documentation or create an issue in the repository. 
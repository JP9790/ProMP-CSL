#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Bool
import numpy as np
import socket
import time
import threading
from collections import deque
import json
import argparse
import os
from .trajectory_deformer import TrajectoryDeformer
from .promp import ProMP
from .stepwise_em_learner import StepwiseEMLearner

class StandaloneDeformationController(Node):
    def __init__(self):
        super().__init__('standalone_deformation_controller')
        
        # Parameters
        self.declare_parameter('kuka_ip', '172.31.1.147')  # Match train_and_execute.py
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('force_threshold', 10.0)
        self.declare_parameter('torque_threshold', 2.0)
        self.declare_parameter('energy_threshold', 0.5)
        self.declare_parameter('deformation_alpha', 0.1)
        self.declare_parameter('deformation_waypoints', 10)
        self.declare_parameter('promp_conditioning_sigma', 0.01)
        self.declare_parameter('trajectory_file', 'learned_trajectory.npy')
        self.declare_parameter('promp_file', 'promp_model.npy')
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.force_threshold = self.get_parameter('force_threshold').value
        self.torque_threshold = self.get_parameter('torque_threshold').value
        self.energy_threshold = self.get_parameter('energy_threshold').value
        self.deformation_alpha = self.get_parameter('deformation_alpha').value
        self.deformation_waypoints = self.get_parameter('deformation_waypoints').value
        self.promp_conditioning_sigma = self.get_parameter('promp_conditioning_sigma').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.promp_file = self.get_parameter('promp_file').value
        
        # Components
        self.deformer = TrajectoryDeformer(
            alpha=self.deformation_alpha,
            n_waypoints=self.deformation_waypoints,
            energy_threshold=self.energy_threshold
        )
        self.promp = None
        self.current_trajectory = None
        self.current_joint_trajectory = None  # Joint space trajectory
        
        # State
        self.is_executing = False
        self.is_monitoring = False
        self.execution_thread = None
        self.monitoring_thread = None
        self.torque_data = deque(maxlen=1000)
        
        # TCP communication
        self.kuka_socket = None
        self.torque_socket = None
        
        # Statistics
        self.deformation_count = 0
        self.conditioning_count = 0
        self.total_energy = 0.0
        
        # Setup communication
        self.setup_communication()
        
        # Setup publishers and subscribers
        self.setup_ros_communication()
        
        # Load trajectory and ProMP
        self.load_trajectory_and_promp()
        
        self.get_logger().info('Standalone Deformation Controller initialized')
        
    def setup_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            # Connect to KUKA for sending trajectories
            self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}:{self.kuka_port}...")
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.settimeout(5)  # 5 second timeout
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal (use robust message receiving)
            ready = self._receive_complete_message(self.kuka_socket, timeout=5.0)
            if ready and ready.strip() == "READY":
                self.get_logger().info('KUKA connection established - received READY signal')
            else:
                self.get_logger().error(f'Unexpected response from KUKA: {ready}')
                self.kuka_socket = None
            
            # Setup server for receiving torque data
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', 30003))
            self.torque_socket.listen(1)
            
            # Start torque data thread
            self.torque_thread = threading.Thread(target=self.receive_torque_data)
            self.torque_thread.daemon = True
            self.torque_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup communication: {e}')
            self.kuka_socket = None
    
    def _receive_complete_message(self, sock, timeout=5.0, buffer_size=8192):
        """
        Receive complete message from socket, handling multi-packet messages.
        Assumes messages end with newline character.
        Matches train_and_execute.py implementation.
        """
        try:
            sock.settimeout(timeout)
            message_parts = []
            
            while True:
                data = sock.recv(buffer_size)
                if not data:
                    break
                
                message_parts.append(data.decode('utf-8'))
                
                # Check if we received a complete line (ends with newline)
                if b'\n' in data:
                    break
            
            return ''.join(message_parts)
        except socket.timeout:
            self.get_logger().warn(f"Timeout waiting for response (>{timeout}s)")
            return None
        except Exception as e:
            self.get_logger().error(f"Error receiving message: {e}")
            return None
    
    def cartesian_to_joint_python(self, cartesian_poses):
        """Convert Cartesian poses to joint positions using pybullet IK solver
        Matches train_and_execute.py implementation"""
        try:
            import pybullet as p
            import pybullet_data
            
            self.get_logger().info('Using pybullet for IK computation (most reliable for KUKA)')
            
            # Initialize pybullet in DIRECT mode (no GUI, faster)
            physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Try to load KUKA LBR iiwa URDF from common locations
            robot_id = None
            urdf_paths = [
                "kuka_iiwa/model.urdf",  # pybullet_data
                "kuka_lbr_iiwa_14_r820.urdf",  # Common name
                "/opt/ros/noetic/share/kuka_description/urdf/kuka_lbr_iiwa_14_r820.urdf",  # ROS path
            ]
            
            for urdf_path in urdf_paths:
                try:
                    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
                    self.get_logger().info(f'Loaded KUKA URDF from: {urdf_path}')
                    break
                except:
                    continue
            
            # If URDF not found, create a simple 7-DOF model
            if robot_id is None:
                self.get_logger().warn('KUKA URDF not found, creating simplified 7-DOF model')
                robot_id = self._create_simple_kuka_model(p)
            
            if robot_id is None:
                self.get_logger().error('Failed to create robot model')
                p.disconnect()
                return None
            
            # Get number of joints
            num_joints = p.getNumJoints(robot_id)
            end_effector_link = num_joints - 1
            
            joint_positions = []
            failed_count = 0
            
            # Initial joint configuration (seed for IK)
            initial_joints = [0.0, 0.7854, 0.0, -1.3962, 0.0, -0.6109, 0.0]
            current_joints = initial_joints.copy()
            
            self.get_logger().info(f'Computing IK for {len(cartesian_poses)} poses using pybullet...')
            
            for i, pose in enumerate(cartesian_poses):
                x, y, z, alpha, beta, gamma = pose
                target_pos = [x, y, z]
                target_orn = p.getQuaternionFromEuler([alpha, beta, gamma])
                
                try:
                    # Compute IK using pybullet's built-in solver
                    joint_angles = p.calculateInverseKinematics(
                        robot_id,
                        end_effector_link,
                        target_pos,
                        target_orn,
                        lowerLimits=[-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
                        upperLimits=[2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054],
                        jointRanges=[5.934, 4.188, 5.934, 4.188, 5.934, 4.188, 6.108],
                        restPoses=current_joints,
                        maxNumIterations=200,
                        residualThreshold=1e-5
                    )
                    
                    if joint_angles is not None and len(joint_angles) >= 7:
                        joint_angles_7 = list(joint_angles[:7])
                        
                        # Verify joint limits
                        valid = True
                        limits = [(-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967),
                                 (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094),
                                 (-3.054, 3.054)]
                        for j, (angle, (min_val, max_val)) in enumerate(zip(joint_angles_7, limits)):
                            if angle < min_val or angle > max_val:
                                valid = False
                                break
                        
                        if valid:
                            joint_positions.append(joint_angles_7)
                            current_joints = joint_angles_7.copy()
                        else:
                            if len(joint_positions) > 0:
                                joint_positions.append(joint_positions[-1])
                            else:
                                joint_positions.append(initial_joints)
                            failed_count += 1
                    else:
                        raise ValueError("IK returned invalid result")
                        
                except Exception as e:
                    self.get_logger().warn(f'Pybullet IK failed for point {i}: {e}')
                    failed_count += 1
                    if len(joint_positions) > 0:
                        joint_positions.append(joint_positions[-1])
                    else:
                        joint_positions.append(initial_joints)
                
                if (i + 1) % 10 == 0:
                    self.get_logger().info(f'Pybullet IK progress: {i+1}/{len(cartesian_poses)} ({failed_count} failed)')
            
            p.disconnect()
            
            if failed_count == 0:
                self.get_logger().info(f'Successfully computed IK for all {len(cartesian_poses)} poses using pybullet')
            else:
                self.get_logger().warn(f'IK computed with {failed_count} failures (used previous/initial positions)')
            
            return np.array(joint_positions)
            
        except ImportError:
            self.get_logger().error('pybullet not installed. Install with: pip install pybullet')
            return None
        except Exception as e:
            self.get_logger().error(f'pybullet IK failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def _create_simple_kuka_model(self, p):
        """Create a simple 7-DOF KUKA LBR iiwa model for IK"""
        # This is a simplified model - for best results, use actual URDF
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        base_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=base_visual, baseCollisionShapeIndex=base_collision)
        self.get_logger().warn('Simplified model created - accuracy may be reduced. Use actual URDF for best results.')
        return base_body
    
    def setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers"""
        # Publishers
        self.deformation_status_pub = self.create_publisher(String, 'deformation_status', 10)
        self.energy_pub = self.create_publisher(Float64, 'deformation_energy', 10)
        self.conditioning_status_pub = self.create_publisher(String, 'conditioning_status', 10)
        self.execution_status_pub = self.create_publisher(String, 'execution_status', 10)
        self.statistics_pub = self.create_publisher(String, 'deformation_statistics', 10)
        
        # Subscribers
        self.start_execution_sub = self.create_subscription(
            Bool, 'start_deformation_execution', self.start_execution_callback, 10)
        self.stop_execution_sub = self.create_subscription(
            Bool, 'stop_deformation_execution', self.stop_execution_callback, 10)
        self.load_trajectory_sub = self.create_subscription(
            String, 'load_trajectory', self.load_trajectory_callback, 10)
    
    def load_trajectory_and_promp(self):
        """Load trajectory and ProMP from files"""
        try:
            # Load trajectory
            self.current_trajectory = np.load(self.trajectory_file)
            self.deformer.set_trajectory(self.current_trajectory)
            self.get_logger().info(f'Loaded trajectory from {self.trajectory_file}')
            
            # Load ProMP (if available)
            try:
                promp_data = np.load(self.promp_file, allow_pickle=True).item()
                self.promp = ProMP()
                self.promp.mean_weights = promp_data['mean_weights']
                self.promp.cov_weights = promp_data['cov_weights']
                self.promp.basis_centers = promp_data['basis_centers']
                self.promp.basis_width = promp_data['basis_width']
                self.get_logger().info(f'Loaded ProMP from {self.promp_file}')
            except FileNotFoundError:
                self.get_logger().warn(f'ProMP file not found: {self.promp_file}')
                self.promp = None
                
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory/ProMP: {e}')
    
    def start_execution_callback(self, msg):
        """Callback to start execution"""
        if msg.data:
            self.start_execution()
        else:
            self.stop_execution()
    
    def stop_execution_callback(self, msg):
        """Callback to stop execution"""
        if msg.data:
            self.stop_execution()
    
    def load_trajectory_callback(self, msg):
        """Callback to load new trajectory"""
        try:
            trajectory = np.load(msg.data)
            self.current_trajectory = trajectory
            self.deformer.set_trajectory(trajectory)
            self.get_logger().info(f'Loaded new trajectory from {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory: {e}')
    
    def receive_torque_data(self):
        """Receive torque data from KUKA robot"""
        try:
            conn, addr = self.torque_socket.accept()
            self.get_logger().info(f'Torque data connection from {addr}')
            
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                    
                lines = data.decode('utf-8').split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            timestamp, fx, fy, fz, tx, ty, tz = map(float, line.strip().split(','))
                            self.torque_data.append({
                                'timestamp': timestamp,
                                'force': [fx, fy, fz],
                                'torque': [tx, ty, tz]
                            })
                        except ValueError:
                            continue
                            
        except Exception as e:
            self.get_logger().error(f'Error receiving torque data: {e}')
    
    def detect_human_interaction(self):
        """Detect human interaction from torque data"""
        if len(self.torque_data) == 0:
            return None
        
        # Get latest torque data
        latest_data = self.torque_data[-1]
        
        force_magnitude = np.linalg.norm(latest_data['force'])
        torque_magnitude = np.linalg.norm(latest_data['torque'])
        
        if force_magnitude > self.force_threshold or torque_magnitude > self.torque_threshold:
            # Return human input vector (combine force and torque)
            human_input = np.array(latest_data['force'] + latest_data['torque'])
            return human_input
        
        return None
    
    def start_execution(self):
        """Start execution of the trajectory with real-time deformation"""
        if self.is_executing:
            self.get_logger().warn('Execution already in progress')
            return
        if self.current_trajectory is None:
            self.get_logger().error('No trajectory loaded for execution')
            return
        self.is_executing = True
        self.execution_thread = threading.Thread(target=self.execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        self.get_logger().info('Started trajectory execution with deformation monitoring')

    def execution_loop(self):
        """Main execution loop: send trajectory to KUKA, monitor torque, and deform in real time"""
        trajectory = np.copy(self.current_trajectory)
        num_points = trajectory.shape[0]
        dt = 0.01  # 100 Hz
        
        # Send the full trajectory at the start (non-blocking)
        self.send_trajectory_to_kuka_async(trajectory)
        self.get_logger().info('Started monitoring for deformation during execution')
        
        # Track current execution state
        trajectory_executing = True
        last_deformation_time = 0
        deformation_cooldown = 0.5  # Minimum time between deformations (seconds)
        
        while self.is_executing:
            # Check for external torque/force
            if len(self.torque_data) > 0:
                latest = self.torque_data[-1]
                max_force = max(abs(f) for f in latest['force'])
                max_torque = max(abs(t) for t in latest['torque'])
                
                # Check if force/torque exceeds threshold and cooldown has passed
                current_time = time.time()
                if (max_force > self.force_threshold or max_torque > self.torque_threshold) and \
                   (current_time - last_deformation_time) > deformation_cooldown:
                    
                    # Compute deformation energy
                    energy = np.linalg.norm(latest['force']) + np.linalg.norm(latest['torque'])
                    self.get_logger().info(f'Deformation detected! Energy: {energy:.4f}, Force: {max_force:.2f}N, Torque: {max_torque:.2f}Nm')
                    
                    # Send STOP to robot (non-blocking with timeout)
                    try:
                        self.kuka_socket.sendall(b"STOP\n")
                        self.get_logger().info('Sent STOP to robot')
                        
                        # Wait for STOPPED response with timeout
                        self.kuka_socket.settimeout(2.0)  # 2 second timeout
                        try:
                            response = self.kuka_socket.recv(1024).decode('utf-8')
                            self.get_logger().info(f'KUKA response: {response}')
                            if "STOPPED" not in response:
                                self.get_logger().warn('Did not receive STOPPED confirmation, proceeding anyway')
                        except socket.timeout:
                            self.get_logger().warn('Timeout waiting for STOPPED, proceeding with new trajectory')
                        finally:
                            self.kuka_socket.settimeout(None)  # Reset timeout
                    except Exception as e:
                        self.get_logger().error(f'Error sending STOP: {e}')
                    
                    # Get current robot position for conditioning (if available)
                    # For now, use mid-trajectory point as condition
                    try:
                        # Try to get current pose from robot
                        self.kuka_socket.sendall(b"GET_POSE\n")
                        self.kuka_socket.settimeout(1.0)
                        pose_response = self.kuka_socket.recv(1024).decode('utf-8')
                        self.kuka_socket.settimeout(None)
                        
                        if pose_response.startswith("POSE:"):
                            # Parse current pose
                            pose_str = pose_response.split("POSE:")[1].strip()
                            current_pose = np.array([float(x) for x in pose_str.split(",")])
                            t_condition = 0.5  # Use current time in trajectory (normalized)
                            y_condition = current_pose
                        else:
                            # Fallback: use mid-trajectory point
                            t_condition = 0.5
                            y_condition = trajectory[int(num_points * t_condition)]
                    except Exception as e:
                        self.get_logger().warn(f'Could not get current pose: {e}, using mid-trajectory point')
                        t_condition = 0.5
                        y_condition = trajectory[int(num_points * t_condition)]
                    
                    # Generate new trajectory based on deformation energy
                    try:
                        if self.promp is not None:
                            if energy < self.energy_threshold:
                                self.get_logger().info('Triggering ProMP conditioning (low energy)')
                                self.promp.condition_on_waypoint(t_condition, y_condition)
                                new_traj = self.promp.generate_trajectory(num_points=num_points)
                            else:
                                self.get_logger().info('Triggering stepwise EM update (high energy)')
                                em_learner = StepwiseEMLearner(self.promp)
                                new_traj = em_learner.update_and_generate(trajectory, latest)
                        else:
                            # No ProMP available - use trajectory deformer
                            self.get_logger().info('Using trajectory deformer (no ProMP available)')
                            human_input = np.array(latest['force'] + latest['torque'])
                            deformed_traj, _, _ = self.deformer.deform(human_input)
                            if deformed_traj is not None:
                                new_traj = deformed_traj
                            else:
                                self.get_logger().warn('Deformation failed, keeping current trajectory')
                                new_traj = trajectory
                        
                        # Update trajectory and send to robot
                        trajectory = new_traj
                        self.current_trajectory = new_traj
                        self.deformer.set_trajectory(new_traj)
                        
                        # Send new trajectory (non-blocking)
                        self.send_trajectory_to_kuka_async(new_traj)
                        self.get_logger().info('Sent new trajectory to robot after deformation')
                        
                        last_deformation_time = current_time
                        self.deformation_count += 1
                        self.total_energy += energy
                        
                    except Exception as e:
                        self.get_logger().error(f'Error generating new trajectory: {e}')
                        import traceback
                        self.get_logger().error(traceback.format_exc())
            
            time.sleep(dt)
        
        self.is_executing = False
        self.get_logger().info('Trajectory execution finished')
    
    def stop_execution(self):
        """Stop trajectory execution"""
        self.is_executing = False
        self.is_monitoring = False
        
        if self.kuka_socket:
            try:
                self.kuka_socket.sendall(b"STOP\n")
            except:
                pass
        
        self.get_logger().info('Stopped trajectory execution')
        self.execution_status_pub.publish(String(data='EXECUTION_STOPPED'))
        
        # Publish final statistics
        self.publish_statistics()
    
    def monitoring_loop(self):
        """Deformation monitoring loop"""
        try:
            while self.is_monitoring:
                # Check for human interaction
                human_input = self.detect_human_interaction()
                
                if human_input is not None:
                    self.get_logger().info('Human interaction detected - applying deformation')
                    self.deformation_status_pub.publish(String(data='DEFORMATION_DETECTED'))
                    
                    # Apply deformation
                    deformed_traj, original_traj, energy = self.deformer.deform(human_input)
                    
                    if deformed_traj is not None:
                        self.deformation_count += 1
                        self.total_energy += energy
                        
                        # Publish energy
                        self.energy_pub.publish(Float64(data=energy))
                        
                        # Check energy threshold
                        if self.deformer.should_condition_promp():
                            self.get_logger().info(f'Energy {energy:.4f} < threshold - triggering ProMP conditioning')
                            self.trigger_promp_conditioning(deformed_traj)
                        else:
                            self.get_logger().info(f'Energy {energy:.4f} >= threshold - high deformation detected')
                            self.handle_high_deformation(deformed_traj, energy)
                
                time.sleep(0.01)  # 100 Hz monitoring
                
        except Exception as e:
            self.get_logger().error(f'Error in monitoring loop: {e}')
        finally:
            self.is_monitoring = False
    
    def trigger_promp_conditioning(self, deformed_trajectory):
        """Trigger ProMP conditioning based on deformation"""
        if self.promp is None:
            self.get_logger().warn('No ProMP available for conditioning')
            return
        
        try:
            # Get deformation region
            deform_region = self.deformer.get_deformation_region()
            
            # Create waypoints for conditioning
            t_conditions = []
            y_conditions = []
            
            for i, idx in enumerate(deform_region):
                # Normalize time to 0-1
                t_condition = idx / len(self.current_trajectory)
                y_condition = deformed_trajectory[idx]
                
                t_conditions.append(t_condition)
                y_conditions.append(y_condition)
            
            # Condition ProMP
            self.promp.condition_on_multiple_waypoints(
                t_conditions, y_conditions, self.promp_conditioning_sigma
            )
            
            # Generate new trajectory
            new_trajectory = self.promp.generate_trajectory(len(self.current_trajectory))
            
            # Update current trajectory
            self.current_trajectory = new_trajectory
            self.deformer.set_trajectory(new_trajectory)
            
            # Send updated trajectory to KUKA
            self.send_trajectory_to_kuka(new_trajectory)
            
            self.conditioning_count += 1
            self.get_logger().info('ProMP conditioning completed and new trajectory sent')
            self.conditioning_status_pub.publish(String(data='CONDITIONING_COMPLETED'))
            
        except Exception as e:
            self.get_logger().error(f'Error during ProMP conditioning: {e}')
    
    def handle_high_deformation(self, deformed_trajectory, energy):
        """Handle high deformation energy scenarios"""
        self.get_logger().warn(f'High deformation energy detected: {energy:.4f}')
        
        # For now, just log the high deformation
        # You can implement additional logic here (e.g., stop execution, switch to manual mode, etc.)
        self.deformation_status_pub.publish(String(data=f'HIGH_DEFORMATION:{energy:.4f}'))
    
    def send_trajectory_to_kuka(self, trajectory):
        """Send full trajectory to KUKA robot and wait for completion (blocking)
        Converts Cartesian to joint positions using pybullet IK to avoid workspace errors
        Matches train_and_execute.py implementation"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Validate trajectory format
            traj_array = np.array(trajectory)
            if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                return False
            
            # Convert Cartesian to joint positions using pybullet IK (matches train_and_execute.py)
            self.get_logger().info('Converting Cartesian trajectory to joint positions using pybullet IK...')
            joint_trajectory = self.cartesian_to_joint_python(trajectory)
            
            if joint_trajectory is None or len(joint_trajectory) == 0:
                self.get_logger().error('Failed to convert trajectory to joint positions')
                self.get_logger().error('Please install pybullet: pip install pybullet')
                return False
            
            # Store joint trajectory
            self.current_joint_trajectory = joint_trajectory
            
            # Format joint trajectory for KUKA: j1,j2,j3,j4,j5,j6,j7 separated by semicolons
            trajectory_str = ";".join([
                ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
            ])
            
            command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
            self.get_logger().info(f'Sending joint trajectory to KUKA ({len(joint_trajectory)} points)...')
            self.get_logger().info('Using joint positions avoids workspace errors - all points should be reachable')
            self.kuka_socket.sendall(command.encode('utf-8'))
            
            # Wait for completion using robust message receiving
            complete = False
            point_count = 0
            error_count = 0
            
            while not complete:
                response = self._receive_complete_message(self.kuka_socket, timeout=30.0)
                if not response:
                    self.get_logger().warn('No response from KUKA')
                    break
                
                response = response.strip()
                
                if "TRAJECTORY_COMPLETE" in response:
                    self.get_logger().info(f'Trajectory execution completed. Points: {point_count}, Errors (skipped): {error_count}')
                    complete = True
                    return True
                elif "ERROR" in response:
                    error_count += 1
                    self.get_logger().warn(f'Point execution error (skipping): {response}')
                elif "POINT_COMPLETE" in response:
                    point_count += 1
                    if point_count % 10 == 0:
                        self.get_logger().info(f'Progress: {point_count}/{len(joint_trajectory)} points completed')
            
            if not complete:
                self.get_logger().warn(f'Trajectory execution did not complete normally. Points: {point_count}, Errors: {error_count}')
                return point_count > 0  # Return True if some progress made
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def send_trajectory_to_kuka_async(self, trajectory):
        """Send trajectory to KUKA robot without blocking (for real-time updates)
        Converts Cartesian to joint positions using pybullet IK"""
        def send_and_monitor():
            if self.kuka_socket is None:
                self.get_logger().error('No connection to KUKA robot')
                return
            
            try:
                # Validate trajectory format
                traj_array = np.array(trajectory)
                if traj_array.ndim != 2 or traj_array.shape[1] != 6:
                    self.get_logger().error(f'Invalid trajectory shape: {traj_array.shape}. Expected (N, 6)')
                    return
                
                # Convert Cartesian to joint positions using pybullet IK
                self.get_logger().info('Converting Cartesian trajectory to joint positions (async)...')
                joint_trajectory = self.cartesian_to_joint_python(trajectory)
                
                if joint_trajectory is None or len(joint_trajectory) == 0:
                    self.get_logger().error('Failed to convert trajectory to joint positions')
                    return
                
                # Store joint trajectory
                self.current_joint_trajectory = joint_trajectory
                
                # Format joint trajectory for KUKA
                trajectory_str = ";".join([
                    ",".join([f"{val:.6f}" for val in point]) for point in joint_trajectory
                ])
                
                command = f"JOINT_TRAJECTORY:{trajectory_str}\n"
                self.kuka_socket.sendall(command.encode('utf-8'))
                self.get_logger().info(f'Sent joint trajectory to KUKA ({len(joint_trajectory)} points) - monitoring in background')
                
                # Monitor responses in background (non-blocking for main loop)
                error_count = 0
                point_count = 0
                while True:
                    try:
                        response = self._receive_complete_message(self.kuka_socket, timeout=0.1)
                        if not response:
                            continue  # Timeout is expected - continue monitoring
                        
                        response = response.strip()
                        
                        if "TRAJECTORY_COMPLETE" in response:
                            self.get_logger().info(f'Trajectory completed: {point_count} points, {error_count} errors')
                            break
                        elif "ERROR" in response:
                            error_count += 1
                            self.get_logger().warn(f'Trajectory execution error: {response}')
                        elif "POINT_COMPLETE" in response:
                            point_count += 1
                    except socket.timeout:
                        # Timeout is expected - continue monitoring
                        continue
                    except Exception as e:
                        self.get_logger().error(f'Error monitoring trajectory: {e}')
                        break
            except Exception as e:
                self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
        
        # Start trajectory sending in separate thread
        thread = threading.Thread(target=send_and_monitor)
        thread.daemon = True
        thread.start()
    
    def publish_statistics(self):
        """Publish execution statistics"""
        stats = {
            'deformation_count': self.deformation_count,
            'conditioning_count': self.conditioning_count,
            'total_energy': self.total_energy,
            'average_energy': self.total_energy / max(self.deformation_count, 1)
        }
        
        self.statistics_pub.publish(String(data=json.dumps(stats)))
        self.get_logger().info(f'Execution statistics: {stats}')
    
    def save_promp(self, filename=None):
        """Save current ProMP state"""
        if self.promp is None:
            self.get_logger().warn('No ProMP to save')
            return
        
        if filename is None:
            filename = self.promp_file
        
        try:
            promp_data = {
                'mean_weights': self.promp.mean_weights,
                'cov_weights': self.promp.cov_weights,
                'basis_centers': self.promp.basis_centers,
                'basis_width': self.promp.basis_width
            }
            np.save(filename, promp_data)
            self.get_logger().info(f'ProMP saved to {filename}')
        except Exception as e:
            self.get_logger().error(f'Error saving ProMP: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Standalone Deformation Controller')
    parser.add_argument('--kuka-ip', default='172.31.1.147', help='KUKA robot IP (matches train_and_execute.py)')
    parser.add_argument('--trajectory-file', default='learned_trajectory.npy', help='Trajectory file path')
    parser.add_argument('--promp-file', default='promp_model.npy', help='ProMP file path')
    parser.add_argument('--energy-threshold', type=float, default=0.5, help='Energy threshold for conditioning')
    parser.add_argument('--force-threshold', type=float, default=10.0, help='Force threshold for deformation')
    parser.add_argument('--torque-threshold', type=float, default=2.0, help='Torque threshold for deformation')
    
    args, _ = parser.parse_known_args()
    
    # Create node with parameters
    node = StandaloneDeformationController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
        node.save_promp()  # Save ProMP state before shutting down
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
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
from .trajectory_deformer import TrajectoryDeformer
from .promp import ProMP

class StandaloneDeformationController(Node):
    def __init__(self):
        super().__init__('standalone_deformation_controller')
        
        # Parameters
        self.declare_parameter('kuka_ip', '192.170.10.25')
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
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal
            ready = self.kuka_socket.recv(1024).decode('utf-8')
            self.get_logger().info(f'KUKA connection established: {ready}')
            
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
        """Start trajectory execution with deformation monitoring"""
        if self.current_trajectory is None:
            self.get_logger().error('No trajectory loaded for execution')
            return False
        
        if self.is_executing:
            self.get_logger().warn('Execution already in progress')
            return False
        
        self.is_executing = True
        self.is_monitoring = True
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self.execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.get_logger().info('Started trajectory execution with deformation monitoring')
        self.execution_status_pub.publish(String(data='EXECUTION_STARTED'))
        
        return True
    
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
    
    def execution_loop(self):
        """Main execution loop"""
        try:
            # Send initial trajectory to KUKA
            self.send_trajectory_to_kuka(self.current_trajectory)
            
            # Monitor execution progress
            while self.is_executing:
                time.sleep(0.1)  # 10 Hz monitoring
                
        except Exception as e:
            self.get_logger().error(f'Error in execution loop: {e}')
        finally:
            self.is_executing = False
    
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
        """Send trajectory to KUKA robot"""
        if self.kuka_socket is None:
            self.get_logger().error('No connection to KUKA robot')
            return False
        
        try:
            # Format trajectory for KUKA
            trajectory_str = ";".join([
                ",".join(map(str, point)) for point in trajectory
            ])
            
            command = f"TRAJECTORY:{trajectory_str}"
            self.kuka_socket.sendall((command + "\n").encode('utf-8'))
            
            # Wait for acknowledgment
            response = self.kuka_socket.recv(1024).decode('utf-8')
            if "TRAJECTORY_COMPLETE" in response or "POINT_COMPLETE" in response:
                return True
            else:
                self.get_logger().warn(f'Unexpected KUKA response: {response}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            return False
    
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
    parser.add_argument('--kuka-ip', default='192.170.10.25', help='KUKA robot IP')
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
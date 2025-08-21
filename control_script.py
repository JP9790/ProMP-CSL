#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import socket
import time
import numpy as np
import os

class AllInOneController(Node):
    def __init__(self):
        super().__init__('all_in_one_controller')
        
        # Parameters
        self.declare_parameter('kuka_ip', '172.31.1.147')
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('save_directory', '~/robot_demos')
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.save_directory = self.get_parameter('save_directory').value
        
        # Expand home directory
        self.save_directory = os.path.expanduser(self.save_directory)
        os.makedirs(self.save_directory, exist_ok=True)
        
        # TCP connection to Java app
        self.kuka_socket = None
        self.kuka_connected = False
        
        # Demo data
        self.demos = []
        
        # Setup connection
        self.setup_connection()
        
        # Setup publishers and subscribers
        self.setup_ros_communication()
        
        self.get_logger().info('All-in-One Controller initialized')
        self.get_logger().info(f'Connected to Java app: {self.kuka_connected}')
    
    def setup_connection(self):
        """Setup TCP connection to Java app"""
        try:
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal
            ready = self.kuka_socket.recv(1024).decode('utf-8').strip()
            if ready == "READY":
                self.kuka_connected = True
                self.get_logger().info('Connected to Java app - received READY signal')
            else:
                self.get_logger().error(f'Unexpected response from Java app: {ready}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Java app: {e}')
    
    def setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers"""
        # Publishers
        self.status_pub = self.create_publisher(String, 'controller_status', 10)
        
        # Subscribers
        self.record_demo_sub = self.create_subscription(
            Bool, 'record_demo', self.record_demo_callback, 10)
        self.get_demos_sub = self.create_subscription(
            Bool, 'get_demos', self.get_demos_callback, 10)
        self.clear_demos_sub = self.create_subscription(
            Bool, 'clear_demos', self.clear_demos_callback, 10)
        self.execute_trajectory_sub = self.create_subscription(
            String, 'execute_trajectory', self.execute_trajectory_callback, 10)
    
    def record_demo_callback(self, msg):
        """Callback to start/stop demo recording"""
        if msg.data:
            self.start_demo_recording()
        else:
            self.stop_demo_recording()
    
    def get_demos_callback(self, msg):
        """Callback to retrieve demos from Java app"""
        if msg.data:
            self.retrieve_demos()
    
    def clear_demos_callback(self, msg):
        """Callback to clear demos"""
        if msg.data:
            self.clear_demos()
    
    def execute_trajectory_callback(self, msg):
        """Callback to execute trajectory"""
        self.execute_trajectory(msg.data)
    
    def start_demo_recording(self):
        """Start demo recording on Java app"""
        if not self.kuka_connected:
            self.get_logger().error("Not connected to Java app")
            return
        
        try:
            self.kuka_socket.sendall(b'START_DEMO_RECORDING\n')
            response = self.kuka_socket.recv(1024).decode('utf-8').strip()
            if response == "DEMO_RECORDING_STARTED":
                self.get_logger().info("Demo recording started")
                self.status_pub.publish(String(data='DEMO_RECORDING_STARTED'))
            else:
                self.get_logger().error(f"Failed to start demo recording: {response}")
        except Exception as e:
            self.get_logger().error(f"Error starting demo recording: {e}")
    
    def stop_demo_recording(self):
        """Stop demo recording on Java app"""
        if not self.kuka_connected:
            self.get_logger().error("Not connected to Java app")
            return
        
        try:
            self.kuka_socket.sendall(b'STOP_DEMO_RECORDING\n')
            response = self.kuka_socket.recv(1024).decode('utf-8').strip()
            if response == "DEMO_RECORDING_STOPPED":
                self.get_logger().info("Demo recording stopped")
                self.status_pub.publish(String(data='DEMO_RECORDING_STOPPED'))
            else:
                self.get_logger().error(f"Failed to stop demo recording: {response}")
        except Exception as e:
            self.get_logger().error(f"Error stopping demo recording: {e}")
    
    def retrieve_demos(self):
        """Retrieve demos from Java app"""
        if not self.kuka_connected:
            self.get_logger().error("Not connected to Java app")
            return
        
        try:
            self.kuka_socket.sendall(b'GET_DEMOS\n')
            response = self.kuka_socket.recv(8192).decode('utf-8').strip()
            
            if response.startswith('DEMOS:'):
                # Parse demos and save to file
                demos_data = response[6:]
                demo_sections = demos_data.split('|')
                
                self.demos.clear()
                
                for section in demo_sections:
                    if section.strip() and section.startswith('DEMO_'):
                        demo_lines = section.split(';')
                        demo_data = []
                        
                        for line in demo_lines:
                            if line.strip() and ',' in line:
                                pose_values = [float(x) for x in line.split(',')]
                                if len(pose_values) >= 6:
                                    demo_data.append(pose_values[:6])
                        
                        if demo_data:
                            self.demos.append(demo_data)
                
                # Save demos to file
                filename = os.path.join(self.save_directory, 'all_demos.npy')
                np.save(filename, np.array(self.demos, dtype=object))
                self.get_logger().info(f"Retrieved and saved {len(self.demos)} demos to {filename}")
                self.status_pub.publish(String(data=f'DEMOS_RETRIEVED:{len(self.demos)}'))
            else:
                self.get_logger().error(f"Failed to retrieve demos: {response}")
                
        except Exception as e:
            self.get_logger().error(f"Error retrieving demos: {e}")
    
    def clear_demos(self):
        """Clear demos on Java app"""
        if not self.kuka_connected:
            self.get_logger().error("Not connected to Java app")
            return
        
        try:
            self.kuka_socket.sendall(b'CLEAR_DEMOS\n')
            response = self.kuka_socket.recv(1024).decode('utf-8').strip()
            if response == "DEMOS_CLEARED":
                self.get_logger().info("Demos cleared")
                self.demos.clear()
                self.status_pub.publish(String(data='DEMOS_CLEARED'))
            else:
                self.get_logger().error(f"Failed to clear demos: {response}")
        except Exception as e:
            self.get_logger().error(f"Error clearing demos: {e}")
    
    def execute_trajectory(self, trajectory_data):
        """Execute trajectory on Java app"""
        if not self.kuka_connected:
            self.get_logger().error("Not connected to Java app")
            return
        
        try:
            command = f"TRAJECTORY:{trajectory_data}\n"
            self.kuka_socket.sendall(command.encode('utf-8'))
            self.get_logger().info("Trajectory execution started")
            self.status_pub.publish(String(data='TRAJECTORY_EXECUTING'))
        except Exception as e:
            self.get_logger().error(f"Error executing trajectory: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AllInOneController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        if node.kuka_socket:
            node.kuka_socket.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
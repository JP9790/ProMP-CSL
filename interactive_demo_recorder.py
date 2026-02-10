#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
import numpy as np
import socket
import json
import time
from collections import deque
import threading
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import datetime

# Import ProMP class
try:
    from .promp import ProMP
except ImportError:
    from promp import ProMP

class InteractiveDemoRecorder(Node):
    def __init__(self):
        super().__init__('interactive_demo_recorder')
        
        # Parameters
        # Note: kuka_ip should match the IP where Java app is running (typically robot controller IP)
        # ros2_pc_ip should match the IP configured in Java's RoboticsAPI.data.xml
        self.declare_parameter('kuka_ip', '172.31.1.147')  # Default matches controller_ip from data.xml
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('torque_port', 30003)
        self.declare_parameter('ros2_pc_ip', '172.31.1.25')  # Default matches ros2_pc_ip from data.xml
        self.declare_parameter('record_frequency', 100.0)  # Hz
        self.declare_parameter('demo_duration', 10.0)  # seconds
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('force_threshold', 10.0)
        self.declare_parameter('torque_threshold', 2.0)
        self.declare_parameter('enable_human_interaction', True)
        self.declare_parameter('send_torque_data', True)
        self.declare_parameter('save_directory', '~/robot_demos')
        self.declare_parameter('auto_save', True)
        
        # Get parameters
        self.kuka_ip = self.get_parameter('kuka_ip').value
        self.kuka_port = self.get_parameter('kuka_port').value
        self.torque_port = self.get_parameter('torque_port').value
        self.ros2_pc_ip = self.get_parameter('ros2_pc_ip').value
        self.record_freq = self.get_parameter('record_frequency').value
        self.demo_duration = self.get_parameter('demo_duration').value
        self.num_basis = self.get_parameter('num_basis_functions').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.force_threshold = self.get_parameter('force_threshold').value
        self.torque_threshold = self.get_parameter('torque_threshold').value
        self.enable_human_interaction = self.get_parameter('enable_human_interaction').value
        self.send_torque_data = self.get_parameter('send_torque_data').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        self.auto_save = self.get_parameter('auto_save').value
        
        # Data storage
        self.demos = []
        self.current_demo = []
        self.is_recording = False
        self.recording_thread = None
        self.demo_counter = 0
        
        # TCP communication with KUKA
        self.kuka_socket = None
        self.torque_socket = None
        self.torque_data = deque(maxlen=1000)
        self.kuka_connected = False
        
        # ProMP parameters
        self.promp = None
        self.learned_trajectory = None
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Setup publishers and subscribers
        self.setup_communication()
        
        # Setup services
        self.setup_services()
        
        # Setup timers
        self.setup_timers()
        
        self.get_logger().info(f'Interactive Demo Recorder initialized')
        self.get_logger().info(f'Demo save directory: {self.save_directory}')
        
    def setup_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}:{self.kuka_port} ...")
            # Connect to KUKA for sending trajectories
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.kuka_socket.settimeout(5)  # 5 second timeout
            self.kuka_socket.connect((self.kuka_ip, self.kuka_port))
            
            # Wait for READY signal from Java application
            ready = self.kuka_socket.recv(1024).decode('utf-8').strip()
            if ready == "READY":
                self.kuka_connected = True
                self.get_logger().info('KUKA connection established - received READY signal')
            else:
                self.get_logger().error(f'Unexpected response from KUKA: {ready}')
                return
            
            # Setup server for receiving torque data
            self.torque_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.torque_socket.bind(('0.0.0.0', self.torque_port))
            self.torque_socket.listen(1)
            
            # Start torque data thread
            self.torque_thread = threading.Thread(target=self.receive_torque_data)
            self.torque_thread.daemon = True
            self.torque_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup communication: {e}')
            # Do not return here; allow the rest of the node to run
    
    def setup_services(self):
        """Setup ROS2 services for control"""
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'demo_status', 10)
        self.trajectory_pub = self.create_publisher(PoseStamped, 'learned_trajectory', 10)
        
        # Subscribers
        self.record_sub = self.create_subscription(
            Bool, 'start_recording', self.record_callback, 10)
        
    def setup_timers(self):
        """Setup timers for periodic tasks"""
        self.timer = self.create_timer(1.0, self.timer_callback)
        
    def record_callback(self, msg):
        """Callback for starting/stopping recording"""
        if msg.data and not self.is_recording:
            self.start_recording()
        elif not msg.data and self.is_recording:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording a new demonstration"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.current_demo = []
        self.demo_counter += 1
        self.get_logger().info(f'Starting demonstration recording #{self.demo_counter}...')
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_demo_thread)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.status_pub.publish(String(data=f'RECORDING_DEMO_{self.demo_counter}'))
    
    def stop_recording(self):
        """Stop recording current demonstration"""
        if not self.is_recording:
            self.get_logger().warn("No recording in progress")
            return
            
        self.get_logger().info("Stopping demo recording...")
        self.is_recording = False
        
        # Wait for recording thread to finish stopping Java recording
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self.recording_thread.is_alive():
                self.get_logger().warn("Recording thread did not finish in time")
        
        # Retrieve demos from Java app (Java records internally)
        if self.kuka_connected:
            self.get_logger().info("Retrieving demos from Java app...")
            if self.retrieve_demos_from_java():
                self.get_logger().info(f"Successfully retrieved {len(self.demos)} demos from Java")
            else:
                self.get_logger().error("Failed to retrieve demos from Java app")
        
        # Save demos if we have any
        if len(self.demos) > 0:
            latest_demo = self.demos[-1]
            self.get_logger().info(f'Demo #{self.demo_counter} recorded with {len(latest_demo)} points')
            
            # Auto-save individual demo
            if self.auto_save:
                self.save_individual_demo(latest_demo, self.demo_counter)
            
            # Save all demos
            self.save_all_demos()
        else:
            self.get_logger().warn("No demos retrieved from Java app")
        
        self.status_pub.publish(String(data=f'DEMO_{self.demo_counter}_COMPLETED'))
    
    def record_demo_thread(self):
        """Thread for recording demonstration data using the new Java app commands"""
        # Start demo recording on the Java app
        if self.kuka_connected:
            try:
                self.kuka_socket.sendall(b'START_DEMO_RECORDING\n')
                # Read response - may need multiple recv calls for complete message
                response = self._receive_complete_message(self.kuka_socket, timeout=5.0)
                if response and response.strip() == "DEMO_RECORDING_STARTED":
                    self.get_logger().info("Demo recording started on Java app")
                else:
                    self.get_logger().error(f"Failed to start demo recording: {response}")
                    self.is_recording = False
                    return
            except Exception as e:
                self.get_logger().error(f"Error starting demo recording: {e}")
                self.is_recording = False
                return
        else:
            self.get_logger().error("Not connected to KUKA - cannot start recording")
            self.is_recording = False
            return
        
        # Wait while recording - Java app records internally
        # Recording continues until is_recording is set to False (by stop_recording() or user)
        dt = 1.0 / self.record_freq
        while self.is_recording:
            try:
                # The Java app is now recording the demo data internally
                # We just need to wait and monitor the recording
                time.sleep(dt)
                
            except Exception as e:
                self.get_logger().error(f'Error during demo recording: {e}')
                break
        
        # Stop demo recording on the Java app
        if self.kuka_connected:
            try:
                self.kuka_socket.sendall(b'STOP_DEMO_RECORDING\n')
                response = self._receive_complete_message(self.kuka_socket, timeout=5.0)
                if response and response.strip() == "DEMO_RECORDING_STOPPED":
                    self.get_logger().info("Demo recording stopped on Java app")
                else:
                    self.get_logger().error(f"Failed to stop demo recording: {response}")
            except Exception as e:
                self.get_logger().error(f"Error stopping demo recording: {e}")
    
    def retrieve_demos_from_java(self):
        """Retrieve all recorded demos from the Java app"""
        if not self.kuka_connected:
            self.get_logger().error("Not connected to Java app")
            return False
        
        try:
            # Request all demos from Java app
            self.kuka_socket.sendall(b'GET_DEMOS\n')
            # Use larger buffer and receive complete message (demos can be large)
            response = self._receive_complete_message(self.kuka_socket, timeout=10.0, buffer_size=65536)
            if not response:
                self.get_logger().error("No response received from Java app")
                return False
            
            response = response.strip()
            
            if response.startswith('DEMOS:'):
                # Parse the demos from Java app
                demos_data = response[6:]  # Remove 'DEMOS:' prefix
                demo_sections = demos_data.split('|')
                
                self.demos.clear()  # Clear existing demos
                
                for section in demo_sections:
                    if section.strip() and section.startswith('DEMO_'):
                        # Parse individual demo
                        # Remove "DEMO_X:" prefix first
                        colon_idx = section.find(':')
                        if colon_idx == -1:
                            continue
                        poses_str = section[colon_idx + 1:]  # Get everything after "DEMO_X:"
                        
                        demo_lines = poses_str.split(';')
                        demo_data = []
                        
                        for line in demo_lines:
                            if line.strip() and ',' in line:
                                try:
                                    pose_values = [float(x) for x in line.split(',')]
                                    if len(pose_values) >= 6:
                                        demo_data.append(pose_values[:6])  # x, y, z, alpha, beta, gamma
                                except ValueError:
                                    # Skip invalid pose data
                                    continue
                        
                        if demo_data:
                            self.demos.append(demo_data)
                
                self.get_logger().info(f"Retrieved {len(self.demos)} demos from Java app")
                return True
            else:
                self.get_logger().error(f"Failed to retrieve demos: {response}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error retrieving demos from Java app: {e}")
            return False
    
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
    
    def save_individual_demo(self, demo_data, demo_number):
        """Save individual demonstration to .npy and .csv files (overwrite if exists)"""
        npy_filename = os.path.join(self.save_directory, f'demo_{demo_number:03d}.npy')
        csv_filename = os.path.join(self.save_directory, f'demo_{demo_number:03d}.csv')
        try:
            np.save(npy_filename, np.array(demo_data))
            np.savetxt(csv_filename, np.array(demo_data), delimiter=',')
            self.get_logger().info(f'Individual demo saved to: {npy_filename} and {csv_filename}')
            return npy_filename, csv_filename
        except Exception as e:
            self.get_logger().error(f'Error saving individual demo: {e}')
            return None, None

    def save_all_demos(self, filename=None):
        """Save all demonstrations to all_demos.npy (overwrite)"""
        # Demos should already be retrieved in stop_recording(), but retrieve again to be sure
        if self.kuka_connected and len(self.demos) == 0:
            self.get_logger().info("No demos in memory, retrieving from Java app...")
            if not self.retrieve_demos_from_java():
                self.get_logger().error("Failed to retrieve demos from Java app")
                return None
        
        if len(self.demos) == 0:
            self.get_logger().warn("No demos to save")
            return None
            
        if filename is None:
            filename = os.path.join(self.save_directory, 'all_demos.npy')
        try:
            np.save(filename, np.array(self.demos, dtype=object))
            self.get_logger().info(f'All demos saved to: {filename}')
            return filename
        except Exception as e:
            self.get_logger().error(f'Error saving all demos: {e}')
            return None
    
    def load_demos(self, filename):
        """Load demonstrations from file"""
        try:
            loaded_demos = np.load(filename, allow_pickle=True).tolist()
            self.demos.extend(loaded_demos)
            self.get_logger().info(f'Loaded {len(loaded_demos)} demos from {filename}')
            self.demo_counter = len(self.demos)
            return True
        except FileNotFoundError:
            self.get_logger().warn(f'No demo file found: {filename}')
            return False
        except Exception as e:
            self.get_logger().error(f'Error loading demos: {e}')
            return False
    
    def get_demo_info(self):
        """Get information about recorded demos"""
        info = {
            'total_demos': len(self.demos),
            'demo_counter': self.demo_counter,
            'save_directory': self.save_directory,
            'demo_lengths': [len(demo) for demo in self.demos] if self.demos else []
        }
        return info
    
    def clear_demos(self):
        """Clear all recorded demonstrations"""
        self.demos = []
        self.demo_counter = 0
        self.get_logger().info('All demos cleared')
    
    def list_saved_demos(self):
        """List all saved demo files in the save directory"""
        try:
            demo_files = [f for f in os.listdir(self.save_directory) if f.endswith('.npy')]
            demo_files.sort()
            return demo_files
        except Exception as e:
            self.get_logger().error(f'Error listing demo files: {e}')
            return []
    
    # Remove or comment out ProMP training and execution methods
    # def train_promp(self): ...
    # def normalize_demos(self): ...
    # etc.

    def _receive_complete_message(self, sock, timeout=5.0, buffer_size=8192):
        """
        Receive complete message from socket, handling multi-packet messages.
        Assumes messages end with newline character.
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
    
    def timer_callback(self):
        """Periodic timer callback"""
        # Publish status
        status_msg = f'Demos: {len(self.demos)}, Recording: {self.is_recording}, Connected: {self.kuka_connected}'
        self.get_logger().debug(status_msg)
    
    def plot_demos(self):
        """Plot recorded demonstrations"""
        if len(self.demos) == 0:
            self.get_logger().warn('No demonstrations to plot')
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Alpha (rad)', 'Beta (rad)', 'Gamma (rad)']
        
        for i in range(6):
            for j, demo in enumerate(self.demos):
                demo_array = np.array(demo)
                axes[i].plot(demo_array[:, i], label=f'Demo {j+1}')
            
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel(labels[i])
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveDemoRecorder()
    
    print("\n" + "="*60)
    print("INTERACTIVE DEMO RECORDER")
    print("="*60)
    print(f"Save directory: {node.save_directory}")
    print(f"Demo duration: {node.demo_duration} seconds")
    print(f"Record frequency: {node.record_freq} Hz")
    print("\nCommands:")
    print("  'start' - Start recording a new demo")
    print("  'stop' - Stop current recording")
    print("  'info' - Show demo information")
    print("  'list' - List saved demo files")
    print("  'plot' - Plot recorded demos")
    print("  'clear' - Clear all demos")
    print("  'quit' - Exit the program")
    print("="*60)
    
    try:
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'start':
                if node.is_recording:
                    print("Recording already in progress. Use 'stop' to stop recording.")
                else:
                    print("Starting demo recording... (Type 'stop' to stop recording)")
                    node.start_recording()
                
            elif command == 'stop':
                if not node.is_recording:
                    print("No recording in progress.")
                else:
                    print("Stopping demo recording...")
                    node.stop_recording()
                    print("Demo recording stopped. Demos retrieved and saved.")
                
            elif command == 'info':
                info = node.get_demo_info()
                print(f"\nDemo Information:")
                print(f"  Total demos: {info['total_demos']}")
                print(f"  Demo counter: {info['demo_counter']}")
                print(f"  Save directory: {info['save_directory']}")
                if info['demo_lengths']:
                    print(f"  Demo lengths: {info['demo_lengths']}")
                
            elif command == 'list':
                demo_files = node.list_saved_demos()
                print(f"\nSaved demo files in {node.save_directory}:")
                for file in demo_files:
                    print(f"  {file}")
                
            elif command == 'plot':
                print("Plotting recorded demos...")
                node.plot_demos()
                
            elif command == 'clear':
                confirm = input("Are you sure you want to clear all demos? (y/n): ")
                if confirm.lower() == 'y':
                    node.clear_demos()
                    print("All demos cleared!")
                    
            elif command == 'quit':
                print("Exiting...")
                break
                
            else:
                print("Unknown command. Available commands: start, stop, info, list, plot, clear, quit")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Save all demos before shutting down
        if len(node.demos) > 0:
            node.save_all_demos()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
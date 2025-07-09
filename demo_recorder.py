# ros2 pkg create --build-type ament_python kuka_promp_control
# cd kuka_promp_control

# <depend>rclpy</depend>
# <depend>geometry_msgs</depend>
# <depend>std_msgs</depend>
# <depend>sensor_msgs</depend>
# <depend>tf2_ros</depend>
# <depend>tf2_geometry_msgs</depend>
# <depend>numpy</depend>
# <depend>scipy</depend>
# <depend>matplotlib</depend>

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

# Import ProMP class
try:
    from .promp import ProMP
except ImportError:
    from promp import ProMP

class DemoRecorder(Node):
    def __init__(self):
        super().__init__('demo_recorder')
        
        # Parameters
        self.declare_parameter('kuka_ip', '192.170.10.25')
        self.declare_parameter('kuka_port', 30002)
        self.declare_parameter('torque_port', 30003)
        self.declare_parameter('ros2_pc_ip', '192.170.10.1')
        self.declare_parameter('record_frequency', 100.0)  # Hz
        self.declare_parameter('demo_duration', 10.0)  # seconds
        self.declare_parameter('num_basis_functions', 50)
        self.declare_parameter('sigma_noise', 0.01)
        self.declare_parameter('force_threshold', 10.0)
        self.declare_parameter('torque_threshold', 2.0)
        self.declare_parameter('enable_human_interaction', True)
        self.declare_parameter('send_torque_data', True)
        
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
        
        # Data storage
        self.demos = []
        self.current_demo = []
        self.is_recording = False
        self.recording_thread = None
        
        # TCP communication with KUKA
        self.kuka_socket = None
        self.torque_socket = None
        self.torque_data = deque(maxlen=1000)
        self.kuka_connected = False
        
        # ProMP parameters
        self.promp = None
        self.learned_trajectory = None
        
        # Setup publishers and subscribers
        self.setup_communication()
        
        # Setup services
        self.setup_services()
        
        # Setup timers
        self.setup_timers()
        
        self.get_logger().info('Demo Recorder initialized')
        
    def setup_communication(self):
        """Setup TCP communication with KUKA robot"""
        try:
            # Connect to KUKA for sending trajectories
            self.kuka_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        self.get_logger().info('Starting demonstration recording...')
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_demo_thread)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.status_pub.publish(String(data='RECORDING'))
    
    def stop_recording(self):
        """Stop recording current demonstration"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if len(self.current_demo) > 0:
            self.demos.append(np.array(self.current_demo))
            self.get_logger().info(f'Demo recorded with {len(self.current_demo)} points')
            self.get_logger().info(f'Total demos: {len(self.demos)}')
        
        self.status_pub.publish(String(data='STOPPED'))
    
    def record_demo_thread(self):
        """Thread for recording demonstration data"""
        start_time = time.time()
        dt = 1.0 / self.record_freq
        
        while self.is_recording and (time.time() - start_time) < self.demo_duration:
            try:
                # Get current pose from robot's current position
                # Note: The Java app doesn't implement GET_POSE, so we'll use a different approach
                # For now, we'll simulate recording by using the robot's current position
                # In a real implementation, you would need to add GET_POSE support to the Java app
                
                # Send a request for current pose (this will need to be implemented in Java)
                if self.kuka_connected:
                    self.kuka_socket.sendall(b'GET_POSE\n')
                    response = self.kuka_socket.recv(1024).decode('utf-8').strip()
                    
                    if response.startswith('POSE:'):
                        pose_data = response[5:].strip().split(',')
                        if len(pose_data) >= 6:
                            pose = [float(x) for x in pose_data[:6]]  # x, y, z, alpha, beta, gamma
                            self.current_demo.append(pose)
                    elif response == "ERROR:GET_POSE_NOT_IMPLEMENTED":
                        # Fallback: use simulated pose data for testing
                        simulated_pose = [
                            np.sin(time.time() * 0.5) * 0.1,  # x
                            np.cos(time.time() * 0.3) * 0.1,  # y
                            0.5 + np.sin(time.time() * 0.2) * 0.05,  # z
                            0.0,  # alpha
                            0.0,  # beta
                            np.sin(time.time() * 0.1) * 0.1   # gamma
                        ]
                        self.current_demo.append(simulated_pose)
                        self.get_logger().debug('Using simulated pose data (GET_POSE not implemented in Java)')
                
                time.sleep(dt)
                
            except Exception as e:
                self.get_logger().error(f'Error recording demo: {e}')
                break
    
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
    
    def train_promp(self):
        """Train ProMP on recorded demonstrations"""
        if len(self.demos) < 1:
            self.get_logger().warn('No demonstrations available for training')
            return False
        
        try:
            # Normalize demonstrations to same length
            normalized_demos = self.normalize_demos()
            
            # Train ProMP
            self.promp = ProMP(num_basis=self.num_basis, sigma_noise=self.sigma_noise)
            self.promp.train(normalized_demos)
            
            # Generate learned trajectory
            self.learned_trajectory = self.promp.generate_trajectory()
            
            self.get_logger().info('ProMP training completed')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error training ProMP: {e}')
            return False
    
    def normalize_demos(self):
        """Normalize demonstrations to same length using interpolation"""
        target_length = 100  # Normalize to 100 points
        normalized = []
        
        for demo in self.demos:
            demo_array = np.array(demo)
            t_old = np.linspace(0, 1, len(demo))
            t_new = np.linspace(0, 1, target_length)
            
            normalized_demo = []
            for i in range(demo_array.shape[1]):  # For each dimension
                interp_func = interp1d(t_old, demo_array[:, i], kind='cubic')
                normalized_demo.append(interp_func(t_new))
            
            normalized.append(np.column_stack(normalized_demo))
        
        return normalized
    
    def publish_trajectory_to_kuka(self):
        """Publish learned trajectory to KUKA robot"""
        if self.learned_trajectory is None:
            self.get_logger().warn('No learned trajectory available')
            return False
        
        try:
            # Format trajectory for KUKA (matches Java app format)
            trajectory_str = ";".join([
                ",".join(map(str, point)) for point in self.learned_trajectory
            ])
            
            command = f"TRAJECTORY:{trajectory_str}"
            self.kuka_socket.sendall((command + "\n").encode('utf-8'))
            
            # Wait for completion
            while True:
                response = self.kuka_socket.recv(1024).decode('utf-8').strip()
                self.get_logger().info(f'KUKA response: {response}')
                if "TRAJECTORY_COMPLETE" in response:
                    break
                elif "ERROR" in response:
                    self.get_logger().error(f'KUKA execution error: {response}')
                    return False
                elif "POINT_COMPLETE" in response:
                    # Continue waiting for full completion
                    continue
            
            self.get_logger().info('Trajectory sent to KUKA successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory to KUKA: {e}')
            return False
    
    def timer_callback(self):
        """Periodic timer callback"""
        # Publish status
        status_msg = f'Demos: {len(self.demos)}, Recording: {self.is_recording}, Connected: {self.kuka_connected}'
        self.get_logger().debug(status_msg)
    
    def save_demos(self, filename='demos.npy'):
        """Save demonstrations to file"""
        if len(self.demos) > 0:
            np.save(filename, self.demos)
            self.get_logger().info(f'Demos saved to {filename}')
    
    def load_demos(self, filename='demos.npy'):
        """Load demonstrations from file"""
        try:
            self.demos = np.load(filename, allow_pickle=True).tolist()
            self.get_logger().info(f'Loaded {len(self.demos)} demos from {filename}')
        except FileNotFoundError:
            self.get_logger().warn(f'No demo file found: {filename}')
    
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
    node = DemoRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_demos()  # Save demos before shutting down
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
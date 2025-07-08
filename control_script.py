#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import time

class ControlScript(Node):
    def __init__(self):
        super().__init__('control_script')
        
        # Publishers
        self.record_pub = self.create_publisher(Bool, 'start_recording', 10)
        
        # Subscribers
        self.status_sub = self.create_subscription(
            String, 'demo_status', self.status_callback, 10)
        
        self.status = 'IDLE'
        
    def status_callback(self, msg):
        self.status = msg.data
        self.get_logger().info(f'Status: {self.status}')
    
    def record_demo(self, duration=10):
        """Record a demonstration"""
        self.get_logger().info(f'Starting demo recording for {duration} seconds...')
        self.record_pub.publish(Bool(data=True))
        
        # Wait for recording to complete
        time.sleep(duration + 2)
        
        self.record_pub.publish(Bool(data=False))
        self.get_logger().info('Demo recording completed')
    
    def run_demo_session(self, num_demos=3):
        """Run a complete demo session"""
        self.get_logger().info(f'Starting demo session with {num_demos} demonstrations')
        
        for i in range(num_demos):
            self.get_logger().info(f'Recording demo {i+1}/{num_demos}')
            self.record_demo()
            time.sleep(2)  # Wait between demos
        
        self.get_logger().info('Demo session completed')

def main(args=None):
    rclpy.init(args=args)
    node = ControlScript()
    
    # Run demo session
    node.run_demo_session(num_demos=3)
    
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
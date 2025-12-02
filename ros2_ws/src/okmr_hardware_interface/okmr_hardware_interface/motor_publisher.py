import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from okmr_msgs.msg import MotorThrottle  

class MotorTestPublisher(Node):

    def __init__(self):
        super().__init__('motor_test_publisher')

        # Use topic name expected by subscriber (change if I'm using the wrong name)
        self.publisher_ = self.create_publisher(MotorThrottle, '/motor_throttle', 10)

        # Timer to send a message every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = MotorThrottle()

        # Fill in throttle values (oscillating between -1.0 and 1.0 for this test publisher, change when necessary)
        value = (-1.0 if self.i % 2 == 0 else 1.0)
        msg.fli = value
        msg.fri = value
        msg.bli = value
        msg.bri = value
        msg.flo = value
        msg.fro = value
        msg.blo = value
        msg.bro = value

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing motor values: {value}')
        self.i += 1 #this is what changes it from -1.0 to 1.0


def main(args=None):
    rclpy.init(args=args)
    try:
        node = MotorTestPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

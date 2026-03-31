#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from grasp_interfaces.msg import Grasp, GraspArray


class DummyGraspPublisher(Node):
    def __init__(self):
        super().__init__("dummy_grasp_publisher")

        self.declare_parameter("x", -0.047)
        self.declare_parameter("y", 0.119)
        self.declare_parameter("z", 0.097)
        self.declare_parameter("qx", 0.851)
        self.declare_parameter("qy", 0.160)
        self.declare_parameter("qz", 0.092)
        self.declare_parameter("qw", 0.491)

        self.grasp_array_pub = self.create_publisher(GraspArray, "target_grasps", 10)

        self.timer = self.create_timer(1.0, self.publish_grasp)

        self.get_logger().info(
            "DummyGraspPublisher node has been started. Publishing grasp at 1 Hz."
        )

    def publish_grasp(self):
        x = self.get_parameter("x").get_parameter_value().double_value
        y = self.get_parameter("y").get_parameter_value().double_value
        z = self.get_parameter("z").get_parameter_value().double_value

        qx = self.get_parameter("qx").get_parameter_value().double_value
        qy = self.get_parameter("qy").get_parameter_value().double_value
        qz = self.get_parameter("qz").get_parameter_value().double_value
        qw = self.get_parameter("qw").get_parameter_value().double_value

        # Set the grasp pose and wrench
        g = Grasp()
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = z
        g.pose.orientation.x = qx
        g.pose.orientation.y = qy
        g.pose.orientation.z = qz
        g.pose.orientation.w = qw
        g.wrench.force.x = float(0.0)
        g.wrench.force.y = float(0.0)
        g.wrench.force.z = float(1.0)
        g.wrench.torque.x = float(0.0)
        g.wrench.torque.y = float(0.0)
        g.wrench.torque.z = float(0.0)

        array_msg = GraspArray()
        array_msg.header.stamp = self.get_clock().now().to_msg()
        array_msg.header.frame_id = "object_frame"
        array_msg.grasps.append(g)
        self.grasp_array_pub.publish(array_msg)


def main(args=None):

    rclpy.init(args=args)

    node = DummyGraspPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
Example usage:

ros2 run planner dummy.py --ros-args -p x:=1.5 -p y:=0.5 -p qz:=1.0
"""

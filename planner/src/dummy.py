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
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("fz", 5.0)
        self.declare_parameter("tx", 0.0)
        self.declare_parameter("ty", 0.0)
        self.declare_parameter("tz", 0.0)

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

        fx = self.get_parameter("fx").get_parameter_value().double_value
        fy = self.get_parameter("fy").get_parameter_value().double_value
        fz = self.get_parameter("fz").get_parameter_value().double_value

        tx = self.get_parameter("tx").get_parameter_value().double_value
        ty = self.get_parameter("ty").get_parameter_value().double_value
        tz = self.get_parameter("tz").get_parameter_value().double_value

        # Set the grasp pose and wrench
        g = Grasp()
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = z
        g.pose.orientation.x = qx
        g.pose.orientation.y = qy
        g.pose.orientation.z = qz
        g.pose.orientation.w = qw
        g.wrench.force.x = fx
        g.wrench.force.y = fy
        g.wrench.force.z = fz
        g.wrench.torque.x = tx
        g.wrench.torque.y = ty
        g.wrench.torque.z = tz

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

Grasp1
- position: [-0.284, -0.127, 0.095] orientation: [-0.054, 0.864, 0.499, -0.031]
- force: [0.90, -0.03, 0.43] torque: [-0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=0.016 -p y:=-0.127 -p z:=0.095 -p qx:=-0.054 -p qy:=0.864 -p qz:=0.499 -p qw:=-0.031 -p fx:=0.90 -p fy:=-0.03 -p fz:=0.43 -p tx:=-0.00 -p ty:=-0.00 -p tz:=-0.00
"""

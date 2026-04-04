#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from grasp_interfaces.msg import Grasp, GraspArray

F = 20.0


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
        g.wrench.force.x = F * fx
        g.wrench.force.y = F * fy
        g.wrench.force.z = F * fz
        g.wrench.torque.x = tx
        g.wrench.torque.y = ty
        g.wrench.torque.z = tz

        array_msg = GraspArray()
        array_msg.header.stamp = self.get_clock().now().to_msg()
        array_msg.header.frame_id = "object_frame"
        array_msg.grasps.append(g)
        self.grasp_array_pub.publish(array_msg)

        self.get_logger().info(
            f"Published grasp: position=({x:.3f}, {y:.3f}, {z:.3f}), orientation=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}), "
            f"force=({fx:.2f}, {fy:.2f}, {fz:.2f}), torque=({tx:.2f}, {ty:.2f}, {tz:.2f})"
        )


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
- position: [-0.130, 0.064, 0.069] orientation: [-0.374, 0.600, -0.600, 0.374]
- force: [-0.80, 0.02, -0.60] torque: [-0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.13 -p y:=0.064 -p z:=0.069 -p qx:=-0.374 -p qy:=0.600 -p qz:=-0.600 -p qw:=0.374 -p fx:=-0.80 -p fy:=0.02 -p fz:=-0.60 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp2
- position: [-0.145, -0.006, 0.070] orientation: []
- force: [-0.98, -0.20, -0.05] torque: [-0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.145 -p y:=-0.006 -p z:=0.070 -p qx:=-0.510 -p qy:=0.490 -p qz:=-0.490 -p qw:=0.510 -p fx:=-0.98 -p fy:=-0.20 -p fz:=-0.05 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp3
- position: [-0.141, -0.042, 0.075] orientation: [0.423, 0.567, 0.567, 0.423]
- force: [-0.72, -0.25, -0.64] torque: [0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.141 -p y:=-0.042 -p z:=0.075 -p qx:=0.423 -p qy:=0.567 -p qz:=0.567 -p qw:=0.423 -p fx:=-0.72 -p fy:=-0.25 -p fz:=-0.64 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp4
- position: [-0.130, 0.063, 0.080] orientation: [0.600, 0.375, 0.375, 0.600]
- force: [-0.73, 0.47, -0.50] torque: [0.00, -0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.130 -p y:=0.063 -p z:=0.080 -p qx:=0.600 -p qy:=0.375 -p qz:=0.375 -p qw:=0.600 -p fx:=-0.73 -p fy:=0.47 -p fz:=-0.50 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp5
- position: [-0.145, 0.030, 0.074] orientation: [0.446, -0.549, 0.549, -0.446]
- force: [-0.95, 0.23, 0.19] torque: [0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.145 -p y:=0.030 -p z:=0.074 -p qx:=0.446 -p qy:=-0.549 -p qz:=0.549 -p qw:=-0.446 -p fx:=-0.95 -p fy:=0.23 -p fz:=0.19 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp6
- position: [-0.143, 0.030, 0.073] orientation: [0.549, 0.446, 0.446, 0.549]
- force: [-0.83, 0.13, -0.55] torque: [0.00, 0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.143 -p y:=0.030 -p z:=0.073 -p qx:=0.549 -p qy:=0.446 -p qz:=0.446 -p qw:=0.549 -p fx:=-0.83 -p fy:=0.13 -p fz:=-0.55 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 7
- position: [-0.018, -0.145, 0.078] orientation: [0.044, 0.706, 0.706, 0.044]
- force: [-0.03, -1.00, -0.05] torque: [-0.00, -0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.018 -p y:=-0.145 -p z:=0.078 -p qx:=0.044 -p qy:=0.706 -p qz:=0.706 -p qw:=0.044 -p fx:=-0.03 -p fy:=-1.00 -p fz:=-0.05 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp8
- position: [-0.136, 0.040, 0.087] orientation: [0.500, -0.500, 0.500, -0.500]
- force: [-0.27, 0.21, 0.94] torque: [-0.00, -0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.136 -p y:=0.040 -p z:=0.087 -p qx:=0.500 -p qy:=-0.500 -p qz:=0.500 -p qw:=-0.500 -p fx:=-0.27 -p fy:=0.21 -p fz:=0.94 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp9
- position: [-0.000, -0.155, 0.042] orientation: [0.000, 0.707, 0.707, -0.000]
- force: [-0.18, -0.95, 0.24] torque: [-0.00, -0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.000 -p y:=-0.155 -p z:=0.042 -p qx:=0.000 -p qy:=0.707 -p qz:=0.707 -p qw:=-0.000 -p fx:=-0.18 -p fy:=-0.95 -p fz:=0.24 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 10
- position: [-0.000, -0.133, 0.050] orientation: [-0.707, 0.000, 0.000, 0.707]
- force: [-0.07, 0.98, -0.21] torque: [-0.00, 0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.000 -p y:=-0.133 -p z:=0.050 -p qx:=-0.707 -p qy:=0.000 -p qz:=0.000 -p qw:=0.707 -p fx:=-0.07 -p fy:=0.98 -p fz:=-0.21 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0


L-Bracket 4
----------------------------------------------------------------------------------
Grasp 1
- position: [-0.144, 0.040, 0.079] orientation: [0.500, -0.500, 0.500, -0.500]
- force: [-0.19, -0.24, 0.95] torque: [0.00, -0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.144 -p y:=0.040 -p z:=0.079 -p qx:=0.500 -p qy:=-0.500 -p qz:=0.500 -p qw:=-0.500 -p fx:=-0.19 -p fy:=-0.24 -p fz:=0.95 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 2
- position: [-0.160, 0.040, 0.071] orientation: [0.500, -0.500, 0.500, -0.500]
- force: [-0.63, 0.51, 0.58] torque: [-0.00, -0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.160 -p y:=0.040 -p z:=0.071 -p qx:=0.500 -p qy:=-0.500 -p qz:=0.500 -p qw:=-0.500 -p fx:=-0.63 -p fy:=0.51 -p fz:=0.58 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 3
- position: [-0.160, 0.040, 0.062] orientation: [0.500, 0.500, 0.500, 0.500]
- force: [0.36, -0.67, -0.65] torque: [0.00, 0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.160 -p y:=0.040 -p z:=0.062 -p qx:=0.500 -p qy:=0.500 -p qz:=0.500 -p qw:=0.500 -p fx:=0.36 -p fy:=-0.67 -p fz:=-0.65 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 4
- position: [0.0, -0.121, 0.041] orientation: [-0.707, 0.000, 0.000, 0.707]
- force: [0.36, -0.36, 0.86] torque: [-0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=0.0 -p y:=-0.121 -p z:=0.041 -p qx:=-0.707 -p qy:=0.000 -p qz:=0.000 -p qw:=0.707 -p fx:=0.36 -p fy:=-0.36 -p fz:=0.86 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

T-Bracket 4
----------------------------------------------------------------------------------
Grasp 1
- position: [-0.122, -0.000, 0.101] orientation: [0.612, 0.612, 0.354, 0.354]
- force: [-0.81, 0.11, -0.57] torque: [-0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.122 -p y:=0.000 -p z:=0.101 -p qx:=0.612 -p qy:=0.612 -p qz:=0.354 -p qw:=0.354 -p fx:=-0.81 -p fy:=0.11 -p fz:=-0.57 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 2
- position: [-0.1277, -0.000, 0.099] orientation: [0.612, 0.612, 0.354, 0.354]
- force: [-0.92, 0.01, 0.40] torque: [-0.00, 0.00, 0.00]
ros2 run planner dummy.py --ros-args -p x:=-0.1277 -p y:=0.000 -p z:=0.099 -p qx:=0.612 -p qy:=0.612 -p qz:=0.354 -p qw:=0.354 -p fx:=-0.92 -p fy:=0.01 -p fz:=0.40 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0

Grasp 3
- position: [0.0, -0.149, 0.041] orientation: [0.000, 0.707, 0.707, -0.000]
- force: [-0.08, 0.89, -0.45] torque: [0.00, -0.00, -0.00]
ros2 run planner dummy.py --ros-args -p x:=0.0 -p y:=-0.149 -p z:=0.041 -p qx:=0.000 -p qy:=0.707 -p qz:=0.707 -p qw:=-0.000 -p fx:=-0.08 -p fy:=0.89 -p fz:=-0.45 -p tx:=0.0 -p ty:=0.0 -p tz:=0.0
"""

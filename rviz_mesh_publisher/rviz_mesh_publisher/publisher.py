import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker


class RvizMeshPublisher(Node):
    def __init__(self):
        super().__init__("rviz_mesh_publisher")
        self.publisher_ = self.create_publisher(Marker, "visualization_marker", 10)
        self.timer = self.create_timer(1.0, self.publish_mesh)

    def publish_mesh(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "rviz_mesh_publisher"
        marker.id = 0

        # Set the marker type to MESH_RESOURCE
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        marker.mesh_resource = "package://rviz_mesh_publisher/meshes/L-Bracket4.stl"

        # Set position and orientation
        marker.pose.position.x = -0.15
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale (1.0 means original size)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Set color (Setting a, r, g, b to 0 lets the mesh keep its native textures if using .dae)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.publisher_.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RvizMeshPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

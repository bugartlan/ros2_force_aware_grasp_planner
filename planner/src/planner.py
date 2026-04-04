#!/usr/bin/env python3
import os

import meshio
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from config import ROBOTIQ_HANDE_GRIPPER
from geometry_msgs.msg import Point, Quaternion, Vector3
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import PoseArray as PoseArrayMsg
from geometry_msgs.msg import Wrench as WrenchMsg
from meshgraphnet.nets import EncodeProcessDecode
from meshgraphnet.normalizer import Normalizer
from optimizer import GNNBasedGraspOptimizer
from rclpy.node import Node

from grasp_interfaces.msg import Grasp, GraspArray

NUM_PUBLISHED_GRASPS = 20  # Number of grasps to publish in the GraspArray response


class GraspPlannerNode(Node):
    def __init__(self):
        super().__init__("grasp_planner_node")

        # Create publishers for the pose and wrench
        self.pose_pub = self.create_publisher(PoseMsg, "target_grasp_pose", 10)
        self.wrench_pub = self.create_publisher(WrenchMsg, "target_grasp_wrench", 10)
        self.grasp_array_pub = self.create_publisher(GraspArray, "target_grasps", 10)
        self.object_frame_id = "object_frame"

        self.package = get_package_share_directory("planner")
        self.checkpoint_path = os.path.join(
            self.package, "assets", "checkpoints", "Model0.pth"
        )

        self.group = "ur_manipulator"
        self.singularity_threshold = 0.01

        # Mode: 0 = force only, 1 = torque only, 2 = full wrench
        self.declare_parameter("mode", 0)
        self.get_logger().info(f"Mode: {self.get_parameter('mode').value}")

        self.get_logger().info("Grasp Planner Node initialized.")

    def optimize_grasp(self):
        self.get_logger().info("Starting grasp optimization...")

        msh_path = os.path.join(self.package, "assets", "meshes", "L-Bracket4_cg1.msh")

        msh = meshio.read(msh_path)
        gripper = ROBOTIQ_HANDE_GRIPPER()

        checkpoint = torch.load(
            self.checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        model_state_dict = checkpoint["model_state_dict"]
        params = checkpoint["params"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalizer = Normalizer(
            num_features=params["node_dim"],
            num_categorical=params["num_categorical"],
            device=device,
            stats=checkpoint["stats"],
        )
        model = EncodeProcessDecode(
            node_dim=params["node_dim"],
            edge_dim=params["edge_dim"],
            output_dim=params["output_dim"],
            latent_dim=params["latent_dim"],
            message_passing_steps=params["message_passing_steps"],
            use_layer_norm=params["use_layer_norm"],
        ).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        optimizer = GNNBasedGraspOptimizer(gripper, model, normalizer, device=device)
        grasps = optimizer.optimize(msh, mu=0.01, k=100)

        if grasps:
            self.get_logger().info(f"{len(grasps)} valid grasps found!")
            self.get_logger().info("Valid grasp found! Publishing results...")
            self.publish_grasp_array(grasps[: min(NUM_PUBLISHED_GRASPS, len(grasps))])
        else:
            self.get_logger().error("Failed to find a valid grasp.")

    def publish_grasp_array(self, grasps: list):
        array_msg = GraspArray()
        array_msg.header.stamp = self.get_clock().now().to_msg()
        array_msg.header.frame_id = self.object_frame_id

        for score, grasp in grasps:
            g = Grasp()

            # Pack Pose
            g.pose.position.x = float(grasp.pose.pos[0])
            g.pose.position.y = float(grasp.pose.pos[1])
            g.pose.position.z = float(grasp.pose.pos[2])
            g.pose.orientation.x = float(grasp.pose.quat[0])
            g.pose.orientation.y = float(grasp.pose.quat[1])
            g.pose.orientation.z = float(grasp.pose.quat[2])
            g.pose.orientation.w = float(grasp.pose.quat[3])
            g.score = score

            # Pack Wrench
            if grasp.wrench is not None:
                g.wrench.force.x = float(grasp.wrench[0])
                g.wrench.force.y = float(grasp.wrench[1])
                g.wrench.force.z = float(grasp.wrench[2])
                g.wrench.torque.x = float(grasp.wrench[3])
                g.wrench.torque.y = float(grasp.wrench[4])
                g.wrench.torque.z = float(grasp.wrench[5])

            array_msg.grasps.append(g)

        self.grasp_array_pub.publish(array_msg)
        self.get_logger().info(f"Published {len(array_msg.grasps)} paired grasps.")

    def publish_pose(self, pose):
        pose_msg = PoseMsg()
        pose_msg.position = Point(
            x=float(pose.pos[0]), y=float(pose.pos[1]), z=float(pose.pos[2])
        )
        pose_msg.orientation = Quaternion(
            x=float(pose.quat[0]),
            y=float(pose.quat[1]),
            z=float(pose.quat[2]),
            w=float(pose.quat[3]),
        )

        self.pose_pub.publish(pose_msg)
        self.get_logger().info("Target pose published successfully.")

    def publish_pose_array(self, poses: list):
        pose_array = PoseArrayMsg()
        for pose in poses:
            pose_msg = PoseMsg()
            pose_msg.position = Point(
                x=float(pose.pos[0]),
                y=float(pose.pos[1]),
                z=float(pose.pos[2]),
            )
            pose_msg.orientation = Quaternion(
                x=float(pose.quat[0]),
                y=float(pose.quat[1]),
                z=float(pose.quat[2]),
                w=float(pose.quat[3]),
            )
            pose_array.poses.append(pose_msg)
            self.pose_pub.publish(pose_array)

    def publish_wrench(self, wrench):
        wrench_msg = WrenchMsg()
        if wrench is not None:
            wrench_msg.force = Vector3(
                x=float(wrench[0]), y=float(wrench[1]), z=float(wrench[2])
            )
            wrench_msg.torque = Vector3(
                x=float(wrench[3]), y=float(wrench[4]), z=float(wrench[5])
            )

        self.wrench_pub.publish(wrench_msg)
        self.get_logger().info("Target wrench published successfully.")


def main(args=None):

    rclpy.init(args=args)
    node = GraspPlannerNode()
    node.optimize_grasp()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

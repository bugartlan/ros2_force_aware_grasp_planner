#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>

geometry_msgs::msg::Pose createPose(double x, double y, double z, double qx,
                                    double qy, double qz, double qw) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.x = qx;
  pose.orientation.y = qy;
  pose.orientation.z = qz;
  pose.orientation.w = qw;
  return pose;
}

geometry_msgs::msg::Pose createPose(double x, double y, double z, double roll,
                                    double pitch, double yaw) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  pose.orientation = tf2::toMsg(q);
  return pose;
}

int main(int argc, char **argv) {
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);

  auto const node = std::make_shared<rclcpp::Node>(
      "moveit",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
          true));

  auto logger = rclcpp::get_logger("moveit");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  using moveit::planning_interface::MoveGroupInterface;
  // Group name specified in the SRDF (in moveit_config)
  auto move_group_interface = MoveGroupInterface(node, "ur_manipulator");
  auto kinematic_state = move_group_interface.getCurrentState();
  auto *joint_model_group =
      kinematic_state->getJointModelGroup("ur_manipulator");

  auto pose = createPose(-0.4, 0.3, 0.2, 3.14, 0.0, 0.0);
  bool success =
      kinematic_state->setFromIK(joint_model_group, pose, "tool0", 0.1);

  if (!success) {
    RCLCPP_ERROR(logger, "Failed to solve IK");
  } else {
    RCLCPP_INFO(logger, "IK solution found");
  }

  Eigen::MatrixXd jacobian;
  kinematic_state->getJacobian(joint_model_group,
                               kinematic_state->getLinkModel("tool0"),
                               Eigen::Vector3d::Zero(), jacobian);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
  double manipulability = svd.singularValues().prod();
  RCLCPP_INFO(logger, "Manipulability: %.6f", manipulability);

  rclcpp::shutdown();
  spinner.join();

  return 0;
}
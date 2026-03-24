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

  std::vector<double> start_joint_positions = {0.0,   -1.57, -1.57,
                                               -1.57, 1.57,  0.0};

  using moveit::planning_interface::MoveGroupInterface;
  // Group name specified in the SRDF (in moveit_config)
  auto move_group_interface = MoveGroupInterface(node, "ur_manipulator");
  move_group_interface.setPlannerId("RRTStar");
  move_group_interface.setMaxVelocityScalingFactor(0.1);
  move_group_interface.setMaxAccelerationScalingFactor(0.1);

  // Initialize MoveIt Visual Tools
  auto moveit_visual_tools = moveit_visual_tools::MoveItVisualTools(
      node, "base_link", rviz_visual_tools::RVIZ_MARKER_TOPIC,
      move_group_interface.getRobotModel());
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.loadRemoteControl();

  // Visualization helper functions
  auto const draw_title = [&moveit_visual_tools](std::string const &title) {
    auto const text_pose = [] {
      auto pose = Eigen::Isometry3d::Identity();
      pose.translation().z() = 1.0;
      return pose;
    }();
    moveit_visual_tools.publishText(text_pose, title, rviz_visual_tools::WHITE,
                                    rviz_visual_tools::XLARGE);
    moveit_visual_tools.trigger();
  };

  auto const prompt = [&moveit_visual_tools](auto text) {
    moveit_visual_tools.prompt(text);
  };

  auto const draw_trajectory_tool_path =
      [&moveit_visual_tools,
       jmg = move_group_interface.getRobotModel()->getJointModelGroup(
           "ur_manipulator")](auto const &trajectory) {
        moveit_visual_tools.publishTrajectoryLine(trajectory, jmg);
      };

  // First, move to the start position
  RCLCPP_INFO(logger, "Moving to start position...");
  move_group_interface.setJointValueTarget(start_joint_positions);
  move_group_interface.move();

  auto home_pose = move_group_interface.getCurrentPose();

  prompt("Press 'next' in the RvizVisualToolsGui window to plan");
  draw_title("Planning");
  moveit_visual_tools.trigger();

  // Now, plan a path to the target
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(createPose(-0.3, 0.1, 0.2, 3.14, 0.0, 0.0));
  waypoints.push_back(createPose(-0.3, 0.3, 0.2, 3.14, 0.0, 0.0));
  waypoints.push_back(createPose(-0.4, 0.3, 0.2, 3.14, 0.0, 0.0));
  waypoints.push_back(createPose(-0.4, -0.1, 0.2, 3.14, 0.0, 0.0));
  waypoints.push_back(createPose(-0.3, -0.1, 0.2, 3.14, 0.0, 0.0));
  waypoints.push_back(createPose(-0.3, 0.1, 0.2, 3.14, 0.0, 0.0));
  waypoints.push_back(home_pose.pose);

  const double eef_step = 0.01; // 1 cm
  const double jump_threshold = 0.0;
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group_interface.computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory);

  RCLCPP_INFO(logger, "Trajectory plan %.2f%% achieved", fraction * 100.0);

  if (fraction > 0.99) {
    draw_trajectory_tool_path(trajectory);
    moveit_visual_tools.trigger();
    prompt("Press 'next' in the RvizVisualToolsGui window to execute");
    draw_title("Executing");
    moveit_visual_tools.trigger();
    move_group_interface.execute(trajectory);
  } else {
    RCLCPP_ERROR(logger, "Planning failed");
  }

  rclcpp::shutdown();
  spinner.join();

  return 0;
}
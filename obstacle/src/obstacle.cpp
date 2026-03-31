#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <rclcpp/rclcpp.hpp>

using moveit::planning_interface::MoveGroupInterface;

moveit_msgs::msg::CollisionObject
loadObjectFromStl(const std::string &stl_path, geometry_msgs::msg::Pose &pose,
                  const std::string &frame_id) {
  moveit_msgs::msg::CollisionObject obj;
  obj.id = "obstacle";
  obj.header.frame_id = frame_id;

  shape_msgs::msg::Mesh mesh_msg;
  shapes::Mesh *mesh = shapes::createMeshFromResource(stl_path);
  shapes::ShapeMsg mesh_msg_temp;
  shapes::constructMsgFromShape(mesh, mesh_msg_temp);
  mesh_msg = boost::get<shape_msgs::msg::Mesh>(mesh_msg_temp);
  obj.meshes.push_back(mesh_msg);
  obj.mesh_poses.push_back(pose);

  return obj;
}

geometry_msgs::msg::Pose makePoseMsg(double x, double y, double z, double roll,
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
  rclcpp::init(argc, argv);

  auto const node = std::make_shared<rclcpp::Node>(
      "obstacle_avoidance",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
          true));

  auto logger = rclcpp::get_logger("obstacle_avoidance");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  std::vector<double> start_joint_positions = {0.0,   -1.57, -1.57,
                                               -1.57, 1.57,  0.0};

  auto move_group_interface = MoveGroupInterface(node, "ur_manipulator");
  move_group_interface.startStateMonitor();
  move_group_interface.setPlannerId("RRTstarkConfigDefault");
  move_group_interface.setMaxVelocityScalingFactor(0.1);
  move_group_interface.setMaxAccelerationScalingFactor(0.1);

  const auto end_effector_link = move_group_interface.getEndEffectorLink();
  if (end_effector_link.empty()) {
    RCLCPP_WARN(
        logger,
        "No default end effector link is configured for this move group.");
  } else {
    RCLCPP_INFO(logger, "Default end effector link: %s",
                end_effector_link.c_str());
  }

  auto moveit_visual_tools = moveit_visual_tools::MoveItVisualTools(
      node, "base_link", rviz_visual_tools::RVIZ_MARKER_TOPIC,
      move_group_interface.getRobotModel());
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.loadRemoteControl();

  auto planning_scene_interface =
      moveit::planning_interface::PlanningSceneInterface();

  // Object pose
  auto pose = makePoseMsg(-0.3, 0.0, 0.0, 0.0, 0.0, 0.0);
  planning_scene_interface.applyCollisionObject(loadObjectFromStl(
      "package://rviz_mesh_publisher/meshes/Bushing3.stl", pose, "base_link"));

  // Allow collision between fingers and obstacle
  moveit_msgs::msg::PlanningScene planning_scene_msg;
  planning_scene_msg.is_diff = true;
  planning_scene_interface.applyPlanningScene(planning_scene_msg);

  // Home position
  move_group_interface.setJointValueTarget(start_joint_positions);
  move_group_interface.move();

  // Target pose
  auto target_pose = makePoseMsg(-0.3, 0.0, 0.2, 3.14, 0.0, 1.57);
  move_group_interface.setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (move_group_interface.plan(plan) ==
                  moveit::core::MoveItErrorCode::SUCCESS);
  if (success) {
    RCLCPP_INFO(logger, "Planning successful, executing the plan...");
    move_group_interface.execute(plan);
  } else {
    RCLCPP_ERROR(logger, "Planning failed");
  }

  return 0;
}
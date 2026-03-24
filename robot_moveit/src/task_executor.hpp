#ifndef TASK_EXECUTOR_HPP_
#define TASK_EXECUTOR_HPP_

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <grasp_interfaces/msg/grasp_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ur_msgs/srv/set_force_mode.hpp>

enum class State { IDLE, STARTUP, MOVING, GRASPING, FORCE, DONE, ERROR };

class TaskExecutor : public rclcpp::Node {
public:
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using SwitchController = controller_manager_msgs::srv::SwitchController;
  using SetForceMode = ur_msgs::srv::SetForceMode;
  using Trigger = std_srvs::srv::Trigger;

  explicit TaskExecutor(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  void init();
  void execute(const geometry_msgs::msg::Pose &pose,
               const geometry_msgs::msg::Wrench &wrench);

private:
  void transitionTo(State next_state);
  void returnHome();
  void onStartup();
  void onMoving();
  void onGrasping();
  void onForce();
  void onDone();
  void onError();
  void switchControllers(const std::vector<std::string> &activate,
                         const std::vector<std::string> &deactivate,
                         std::function<void()> callback);

  void poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg);
  void wrenchCallback(const geometry_msgs::msg::Wrench::SharedPtr msg);
  void
  graspArrayCallback(const grasp_interfaces::msg::GraspArray::SharedPtr msg);
  void checkAndExecute();

  State state_{State::IDLE};
  geometry_msgs::msg::Pose target_pose_;
  geometry_msgs::msg::Wrench target_wrench_;
  std::vector<double> target_joint_values_;
  grasp_interfaces::msg::GraspArray target_grasps_;
  bool pose_received_{false};
  bool wrench_received_{false};
  bool grasps_received_{false};

  tf2::Transform tf_base_to_object_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>
      move_group_interface_;
  std::shared_ptr<moveit_visual_tools::MoveItVisualTools> moveit_visual_tools_;
  std::vector<double> home_positions = {0.0, -1.57, -1.57, -1.57, 1.57, 0.0};

  // ROS Interfaces
  rclcpp::Client<SwitchController>::SharedPtr switch_client_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr gripper_client_;
  rclcpp::Client<SetForceMode>::SharedPtr force_start_client_;
  rclcpp::Client<Trigger>::SharedPtr force_stop_client_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_pub_;

  // Subscribers for the Python planner
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr wrench_sub_;
  rclcpp::Subscription<grasp_interfaces::msg::GraspArray>::SharedPtr
      grasp_array_sub_;
  rclcpp::TimerBase::SharedPtr wrench_timer_;
  rclcpp::TimerBase::SharedPtr duration_timer_;

  // Callback groups for multi-threading
  rclcpp::CallbackGroup::SharedPtr client_cb_group_;

  // Helpers
  geometry_msgs::msg::Pose transformPose(const geometry_msgs::msg::Pose &pose,
                                         const tf2::Transform &tf);

  // Constants
  static constexpr double MANIPULABILITY_THRESHOLD = 0.015;
};

#endif
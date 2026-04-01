#ifndef TASK_EXECUTOR_HPP_
#define TASK_EXECUTOR_HPP_

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <grasp_interfaces/msg/grasp_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ur_msgs/srv/set_force_mode.hpp>

enum class State {
  IDLE,
  STARTUP,
  READY,
  PREGRASP,
  GRASPING,
  FORCE,
  DONE,
  ERROR
};

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
  std::string stateToString(State s);
  void returnHome();
  void onStartup();
  void onReady();
  void onPregrasp();
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
  shape_msgs::msg::Mesh loadObjectMesh(const std::string &path);
  void visualizePose(const geometry_msgs::msg::Pose &pose,
                     const std::string &id);
  void visualizePose(const std::vector<double> &joint_values,
                     const std::string &text = "");
  void openGripper();
  void closeGripper();
  void findValidGrasp();
  bool computePregrasp(const geometry_msgs::msg::Pose &pose);
  bool isValid(const geometry_msgs::msg::Pose &pose, int max_attempts = 3);
  std::vector<double> getCurrentJointValues();

  State state_{State::IDLE};

  // Messages received from the planner
  geometry_msgs::msg::Pose pregrasp_pose_;
  geometry_msgs::msg::Pose target_pose_;
  geometry_msgs::msg::Wrench target_wrench_;
  std::vector<double> pregrasp_joint_values_;
  std::vector<double> target_joint_values_;
  grasp_interfaces::msg::GraspArray target_grasps_;

  // Flags to track if we've received the necessary inputs
  bool pose_received_{false};
  bool wrench_received_{false};
  bool grasps_received_{false};
  bool busy_{false};

  // Transformation: [x, y, z, qx, qy, qz, qw]
  tf2::Transform tf_base_to_object_;

  // Collision object
  moveit_msgs::msg::CollisionObject collision_object_;

  // MoveIt interfaces
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>
      move_group_interface_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface>
      planning_scene_interface;
  std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor> psm_;
  std::shared_ptr<moveit_visual_tools::MoveItVisualTools> moveit_visual_tools_;
  const moveit::core::JointModelGroup *joint_model_group_{nullptr};
  const moveit::core::LinkModel *end_effector_link_{nullptr};
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
  static constexpr double kManipulabilityThreshold = 0.01;
  static constexpr double kMotionVelocityScale = 0.1;
  static constexpr double kMotionAccelScale = 0.1;
  static constexpr int kGraspSettleMs = 1000;
  static constexpr int kWrenchPublishIntervalMs = 8;
  static constexpr const char *kGroupName = "ur_manipulator";
};

#endif
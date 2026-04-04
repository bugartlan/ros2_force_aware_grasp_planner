#include "task_executor.hpp"
#include "task_executor_utils.hpp"
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <shape_msgs/msg/mesh.hpp>

struct Params {
  double force_duration_sec;
  double force_magnitude;
  double motion_timeout_sec;
  std::string base_link;
  std::vector<double> tf_base_to_object;
};

Params loadParams(rclcpp::Node &node) {
  return {node.declare_parameter("force_duration_sec", 5.0),
          node.declare_parameter("force_magnitude", 10.0),
          node.declare_parameter("motion_timeout_sec", 15.0),
          node.declare_parameter("base_link", "ur5e_base_link"),
          node.declare_parameter(
              "tf_base_to_object",
              std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})};
}

moveit_msgs::msg::CollisionObject
loadObjectFromStl(const std::string &stl_path, geometry_msgs::msg::Pose pose,
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

TaskExecutor::TaskExecutor(const rclcpp::NodeOptions &options)
    : Node("task_executor", options) {
  // Parameters
  loadParams(*this);

  // Service Clients
  switch_client_ =
      create_client<SwitchController>("/controller_manager/switch_controller");
  gripper_client_ = create_client<std_srvs::srv::SetBool>("gripper_command");
  force_start_client_ =
      create_client<SetForceMode>("/force_mode_controller/start_force_mode");
  force_stop_client_ =
      create_client<Trigger>("/force_mode_controller/stop_force_mode");

  // Publishers
  wrench_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/force_mode_controller/wrench_command", 10);

  // Subscribers
  grasp_array_sub_ = create_subscription<grasp_interfaces::msg::GraspArray>(
      "/target_grasps", 10,
      std::bind(&TaskExecutor::graspArrayCallback, this,
                std::placeholders::_1));

  tf_base_to_object_ = task_executor_utils::transformFromParams(
      this->get_parameter("tf_base_to_object").as_double_array());

  RCLCPP_INFO(get_logger(), "Task Executor Node Created.");
}

void TaskExecutor::init() {
  // Initialize MoveIt interfaces
  move_group_interface_ =
      std::make_shared<moveit::planning_interface::MoveGroupInterface>(
          shared_from_this(), "ur_manipulator");
  move_group_interface_->startStateMonitor(1.0);
  move_group_interface_->setPlanningTime(
      this->get_parameter("motion_timeout_sec").as_double());
  move_group_interface_->setPlannerId("PRMkConfigDefault");
  move_group_interface_->setNumPlanningAttempts(10);
  move_group_interface_->setMaxVelocityScalingFactor(kMotionVelocityScale);
  move_group_interface_->setMaxAccelerationScalingFactor(kMotionAccelScale);

  psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      shared_from_this(), "robot_description");
  if (psm_->getPlanningScene()) {
    psm_->startSceneMonitor();
    psm_->startWorldGeometryMonitor();
    psm_->startStateMonitor();
    RCLCPP_INFO(get_logger(), "PlanningSceneMonitor initialized successfully.");
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to initialize PlanningSceneMonitor.");
  }

  moveit_visual_tools_ =
      std::make_shared<moveit_visual_tools::MoveItVisualTools>(
          shared_from_this(), "base_link", rviz_visual_tools::RVIZ_MARKER_TOPIC,
          move_group_interface_->getRobotModel());
  moveit_visual_tools_->deleteAllMarkers();
  moveit_visual_tools_->loadRemoteControl();
  moveit_visual_tools_->loadRobotStatePub("/display_robot_state");

  joint_model_group_ =
      move_group_interface_->getRobotModel()->getJointModelGroup(kGroupName);
  end_effector_link_ =
      move_group_interface_->getRobotModel()->getLinkModel("tool0");

  planning_scene_interface =
      std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

  collision_object_ = loadObjectFromStl(
      "package://rviz_mesh_publisher/meshes/Bushing3.stl",
      task_executor_utils::transformToPoseMsg(tf_base_to_object_),
      move_group_interface_->getPlanningFrame());
  RCLCPP_INFO(get_logger(), "Initialization complete.");

  transitionTo(State::STARTUP);
}

void TaskExecutor::transitionTo(State next_state) {
  RCLCPP_INFO(get_logger(), "State: %s → %s", stateToString(state_).c_str(),
              stateToString(next_state).c_str());

  state_ = next_state;

  switch (state_) {
  case State::STARTUP:
    onStartup();
    break;
  case State::READY:
    onReady();
    break;
  case State::PREGRASP:
    onPregrasp();
    break;
  case State::GRASPING:
    onGrasping();
    break;
  case State::FORCE:
    onForce();
    break;
  case State::DONE:
    onDone();
    break;
  case State::ERROR:
    onError();
    break;
  default:
    break;
  }
}

std::string TaskExecutor::stateToString(State s) {
  switch (s) {
  case State::IDLE:
    return "IDLE";
  case State::STARTUP:
    return "STARTUP";
  case State::READY:
    return "READY";
  case State::PREGRASP:
    return "PREGRASP";
  case State::GRASPING:
    return "GRASPING";
  case State::FORCE:
    return "FORCE";
  case State::DONE:
    return "DONE";
  case State::ERROR:
    return "ERROR";
  default:
    return "UNKNOWN";
  }
}

void TaskExecutor::returnHome() {
  RCLCPP_INFO(get_logger(), "Returning to home position...");
  move_group_interface_->setJointValueTarget(home_positions);
  move_group_interface_->move();
}

void TaskExecutor::openGripper() {
  if (!gripper_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Gripper service not available!");
    transitionTo(State::ERROR);
    return;
  }

  auto gripper_request = std::make_shared<std_srvs::srv::SetBool::Request>();
  gripper_request->data = true; // Open gripper
  gripper_client_->async_send_request(gripper_request);
  rclcpp::sleep_for(std::chrono::milliseconds(1000));
}

void TaskExecutor::closeGripper() {
  if (!gripper_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Gripper service not available!");
    transitionTo(State::ERROR);
    return;
  }

  auto gripper_request = std::make_shared<std_srvs::srv::SetBool::Request>();
  gripper_request->data = false; // Close gripper
  gripper_client_->async_send_request(gripper_request);
  rclcpp::sleep_for(std::chrono::milliseconds(1000));
}

void TaskExecutor::onStartup() {
  openGripper();

  RCLCPP_INFO(get_logger(), "Activating controllers");
  switchControllers(
      {"scaled_joint_trajectory_controller"},
      {"force_mode_controller", "joint_trajectory_controller"}, [this]() {
        returnHome();
        RCLCPP_INFO(get_logger(), "Startup complete, ready to execute tasks");
        transitionTo(State::READY);
      });
}

void TaskExecutor::onReady() {
  RCLCPP_INFO(get_logger(), "Robot is in READY state.");
  if (grasps_received_) {
    RCLCPP_INFO(get_logger(), "Grasp data already available.");
    std::thread([this]() { findValidGrasp(); }).detach();
  } else {
    RCLCPP_INFO(get_logger(), "No grasp data yet. Waiting ...");
  }
}

void TaskExecutor::onPregrasp() {
  // Add collision object to the scene
  planning_scene_interface->applyCollisionObject(collision_object_);
  RCLCPP_INFO(
      get_logger(),
      "\033[1;34mTarget grasp - position: [%.3f, %.3f, %.3f] orientation: [%.3f, "
      "%.3f, %.3f, %.3f]\033[0m",
      target_pose_.position.x, target_pose_.position.y, target_pose_.position.z,
      target_pose_.orientation.x, target_pose_.orientation.y,
      target_pose_.orientation.z, target_pose_.orientation.w);
  RCLCPP_INFO(get_logger(), "Moving to pregrasp position...");
  visualizePose(pregrasp_joint_values_, "Pregrasp_Pose");

  move_group_interface_->setStartStateToCurrentState();
  // move_group_interface_->setStartState(current_state);
  move_group_interface_->setJointValueTarget(pregrasp_joint_values_);
  moveit::planning_interface::MoveGroupInterface::Plan plan;

  // Remove collision object before grasping to avoid interference
  planning_scene_interface->removeCollisionObjects({"obstacle"});
  if (move_group_interface_->plan(plan) ==
      moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_INFO(get_logger(), "Plan found! Visualizing trajectory...");
    moveit_visual_tools_->deleteAllMarkers();
    moveit_visual_tools_->publishTrajectoryLine(
        plan.trajectory_, end_effector_link_, joint_model_group_);
    moveit_visual_tools_->trigger();
    moveit_visual_tools_->prompt(
        "Press 'Next' in the RvizVisualToolsGui window to execute");

    // Debugging: log the planned joint values
    auto first_point = plan.trajectory_.joint_trajectory.points[0].positions;
    RCLCPP_INFO(get_logger(),
                "Planned joint values for pregrasp: [%.3f, %.3f, %.3f, "
                "%.3f, %.3f, %.3f]",
                first_point[0], first_point[1], first_point[2], first_point[3],
                first_point[4], first_point[5]);

    RCLCPP_INFO(get_logger(), "Executing...");
    move_group_interface_->execute(plan);
    transitionTo(State::GRASPING);
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to compute a valid trajectory");
    transitionTo(State::ERROR);
  }
}

void TaskExecutor::onGrasping() {

  RCLCPP_INFO(get_logger(), "Starting grasping phase...");
  // moveit::core::RobotState
  // start_state(*move_group_interface_->getCurrentState());
  // start_state.setJointGroupPositions(joint_model_group_,
  // pregrasp_joint_values_); move_group_interface_->setStartState(start_state);

  move_group_interface_->setStartStateToCurrentState();
  move_group_interface_->setJointValueTarget(target_joint_values_);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  if (move_group_interface_->plan(plan) ==
      moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_INFO(get_logger(), "Plan found! Visualizing trajectory...");
    moveit_visual_tools_->deleteAllMarkers();
    moveit_visual_tools_->publishTrajectoryLine(
        plan.trajectory_, end_effector_link_, joint_model_group_);
    moveit_visual_tools_->trigger();
    moveit_visual_tools_->prompt(
        "Press 'Next' in the RvizVisualToolsGui window to execute");

    RCLCPP_INFO(get_logger(), "Executing...");
    move_group_interface_->execute(plan);

    auto gripper_request = std::make_shared<std_srvs::srv::SetBool::Request>();
    gripper_request->data = false; // Close gripper
    gripper_client_->async_send_request(gripper_request);

    // Wait for some time
    rclcpp::sleep_for(std::chrono::milliseconds(kGraspSettleMs));
    switchControllers(
        {"force_mode_controller"},
        {"scaled_joint_trajectory_controller", "joint_trajectory_controller"},
        [this]() { transitionTo(State::FORCE); });
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to compute a valid trajectory");
    transitionTo(State::ERROR);
    return;
  }
}

void TaskExecutor::onForce() {
  // Check if force start service is available
  if (!force_start_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Force start service not available!");
    transitionTo(State::ERROR);
    return;
  }

  RCLCPP_INFO(get_logger(), "Applying wrench");
  RCLCPP_INFO(get_logger(),
              "\033[1;34mWrench - force: [%.2f, %.2f, %.2f] torque: [%.2f, "
              "%.2f, %.2f]\033[0m",
              target_wrench_.force.x, target_wrench_.force.y,
              target_wrench_.force.z, target_wrench_.torque.x,
              target_wrench_.torque.y, target_wrench_.torque.z);
  auto request = std::make_shared<SetForceMode::Request>();
  request->task_frame.header.frame_id =
      this->get_parameter("base_link").as_string();
  request->task_frame.pose.orientation.w = 1.0;

  request->selection_vector_x = 1;
  request->selection_vector_y = 1;
  request->selection_vector_z = 1;
  request->selection_vector_rx = 0;
  request->selection_vector_ry = 0;
  request->selection_vector_rz = 0;
  request->wrench = target_wrench_;
  // request->type = SetForceMode::Request::NO_TRANSFORM;
  request->speed_limits.linear.x = 0.1;
  request->speed_limits.linear.y = 0.1;
  request->speed_limits.linear.z = 0.1;
  request->speed_limits.angular.x = 0.1;
  request->speed_limits.angular.y = 0.1;
  request->speed_limits.angular.z = 0.1;

  RCLCPP_INFO(get_logger(), "Sending force mode start request...");
  force_start_client_->async_send_request(
      request, [this](rclcpp::Client<SetForceMode>::SharedFuture future) {
        try {
          auto response = future.get();
          if (response->success) {
            RCLCPP_INFO(this->get_logger(), "Force mode started successfully");
            // Start wrench publisher timer
            wrench_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(kWrenchPublishIntervalMs), [this]() {
                  geometry_msgs::msg::WrenchStamped msg;
                  msg.header.stamp = this->now();
                  msg.header.frame_id =
                      this->get_parameter("base_link").as_string();
                  msg.wrench = target_wrench_;
                  wrench_pub_->publish(msg);
                });

            // Stop after duration
            duration_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(
                    this->get_parameter("force_duration_sec").as_double()),
                [this]() {
                  duration_timer_->cancel();
                  // Stop force mode
                  if (wrench_timer_)
                    wrench_timer_->cancel();
                  force_stop_client_->async_send_request(
                      std::make_shared<Trigger::Request>(),
                      [this](rclcpp::Client<Trigger>::SharedFuture future) {
                        try {
                          auto response = future.get();
                          if (response->success) {
                            RCLCPP_INFO(this->get_logger(),
                                        "Force mode successfully stopped.");
                            transitionTo(State::DONE);
                          } else {
                            RCLCPP_ERROR(this->get_logger(),
                                         "Failed to stop force mode.");
                            transitionTo(State::ERROR);
                          }
                        } catch (const std::exception &e) {
                          RCLCPP_ERROR(this->get_logger(),
                                       "Stop service call failed: %s",
                                       e.what());
                          transitionTo(State::ERROR);
                        }
                      });
                });
          } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to start force mode");
            transitionTo(State::ERROR);
          }
        } catch (const std::exception &e) {
          RCLCPP_ERROR(this->get_logger(), "Service call failed: %s", e.what());
          transitionTo(State::ERROR);
        }
      });
}

void TaskExecutor::onDone() {
  RCLCPP_INFO(get_logger(), "Task completed successfully!");
  rclcpp::shutdown();
}

void TaskExecutor::onError() {
  RCLCPP_ERROR(get_logger(), "An error occurred during task execution.");
  rclcpp::shutdown();
}

void TaskExecutor::switchControllers(const std::vector<std::string> &activate,
                                     const std::vector<std::string> &deactivate,
                                     std::function<void()> callback) {
  // Check if service is available
  if (!switch_client_->wait_for_service(std::chrono::seconds(3))) {
    RCLCPP_ERROR(this->get_logger(), "Controller Manager service not found!");
    transitionTo(State::ERROR);
    return;
  }

  auto request = std::make_shared<SwitchController::Request>();
  request->activate_controllers = activate;
  request->deactivate_controllers = deactivate;
  request->strictness = SwitchController::Request::BEST_EFFORT; // STRICT

  auto future = switch_client_->async_send_request(
      request,
      [this, callback](rclcpp::Client<SwitchController>::SharedFuture future) {
        try {
          auto response = future.get();
          if (response->ok) {
            RCLCPP_INFO(this->get_logger(),
                        "Controllers switched successfully");
            callback();
          } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to switch controllers");
            transitionTo(State::ERROR);
          }
        } catch (const std::exception &e) {
          RCLCPP_ERROR(this->get_logger(), "Service call failed: %s", e.what());
          transitionTo(State::ERROR);
        }
      });
}

void TaskExecutor::graspArrayCallback(
    const grasp_interfaces::msg::GraspArray::SharedPtr msg) {
  if (!grasps_received_) {
    RCLCPP_INFO(get_logger(), "Received grasp array with %lu grasps",
                msg->grasps.size());
    grasps_received_ = true;
    target_grasps_ = *msg;
    if (state_ == State::READY) {
      std::thread([this]() {
        findValidGrasp();
        grasps_received_ = false;
      }).detach();
    }
  }
}

bool TaskExecutor::isValid(const geometry_msgs::msg::Pose &pose,
                           int max_attempts) {
  // Check if the given pose is reachable and not near singularity
  auto current_state = move_group_interface_->getCurrentState();
  auto kinematic_state =
      std::make_shared<moveit::core::RobotState>(*current_state);
  auto *joint_model_group = kinematic_state->getJointModelGroup(kGroupName);
  for (int i = 0; i < max_attempts; ++i) {
    if (kinematic_state->setFromIK(joint_model_group, pose, "tool0", 0.1)) {
      kinematic_state->update();

      collision_detection::CollisionRequest collision_request;
      collision_detection::CollisionResult collision_result;
      collision_request.group_name = kGroupName;
      {
        planning_scene_monitor::LockedPlanningSceneRO planning_scene(psm_);
        planning_scene->checkCollision(collision_request, collision_result,
                                       *kinematic_state);
      }
      if (collision_result.collision)
        continue;

      RCLCPP_INFO(get_logger(), "Pose is reachable (IK solution found)");
      std::vector<double> joint_values;
      kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);

      // Log the joint values for debugging
      std::stringstream ss;
      ss << "IK Solution found: [";
      for (size_t j = 0; j < joint_values.size(); ++j) {
        ss << joint_values[j] << (j == joint_values.size() - 1 ? "" : ", ");
      }
      ss << "]";

      RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

      Eigen::MatrixXd jacobian;
      if (!kinematic_state->getJacobian(joint_model_group,
                                        kinematic_state->getLinkModel("tool0"),
                                        Eigen::Vector3d::Zero(), jacobian)) {
        RCLCPP_ERROR(get_logger(), "Failed to get Jacobian");
        return false;
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
      double manipulability = svd.singularValues().prod();
      RCLCPP_INFO(get_logger(), "-> Manipulability: %.6f", manipulability);
      if (manipulability >= kManipulabilityThreshold) {
        kinematic_state->copyJointGroupPositions(joint_model_group,
                                                 target_joint_values_);
        // visualizePose(target_joint_values_);
        if (computePregrasp(pose))
          return true;
      }
    }
    RCLCPP_WARN(get_logger(), "Attempt %d: Failed. Retrying...", i + 1);
    kinematic_state->setToRandomPositionsNearBy(joint_model_group,
                                                *current_state, 0.05);
  }
  return false;
}

bool TaskExecutor::computePregrasp(const geometry_msgs::msg::Pose &pose) {
  // Compute a pre-grasp pose slightly away from the target pose
  auto start_state = move_group_interface_->getCurrentState();
  const auto *joint_model_group = start_state->getJointModelGroup(kGroupName);
  start_state->setJointGroupPositions(joint_model_group, target_joint_values_);
  move_group_interface_->setStartState(*start_state);

  tf2::Transform tf_pregrasp;
  tf_pregrasp.setOrigin(tf2::Vector3(0, 0, -0.05)); // 5cm back along z-axis
  tf_pregrasp.setRotation(tf2::Quaternion(0, 0, 0, 1));
  pregrasp_pose_ =
      task_executor_utils::applyLocalTransformToPose(pose, tf_pregrasp);
  RCLCPP_INFO(get_logger(),
              "Pregrasp - position: [%.3f, %.3f, %.3f] orientation: [%.3f, "
              "%.3f, %.3f, %.3f]",
              pregrasp_pose_.position.x, pregrasp_pose_.position.y,
              pregrasp_pose_.position.z, pregrasp_pose_.orientation.x,
              pregrasp_pose_.orientation.y, pregrasp_pose_.orientation.z,
              pregrasp_pose_.orientation.w);

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(pregrasp_pose_);

  moveit_msgs::msg::RobotTrajectory traj;
  double fraction =
      move_group_interface_->computeCartesianPath(waypoints, 0.01, 0.0, traj);
  RCLCPP_INFO(get_logger(), "Pre-grasp IK solution fraction: %.2f%%",
              fraction * 100.0);

  // Restore default behavior so later plans start from the monitored current
  // state.
  move_group_interface_->setStartStateToCurrentState();

  if (fraction > 0.99) {
    RCLCPP_INFO(get_logger(), "Pre-grasp pose is valid and reachable");
    pregrasp_joint_values_ = traj.joint_trajectory.points.back().positions;
  }

  return fraction > 0.99;
}

void TaskExecutor::findValidGrasp() {
  if (busy_ || state_ != State::READY)
    return;

  busy_ = true;
  for (auto &grasp : target_grasps_.grasps) {
    auto pose = task_executor_utils::applyWorldTransformToPose(
        grasp.pose, tf_base_to_object_);
    RCLCPP_INFO(get_logger(),
                "Grasp - position: [%.3f, %.3f, %.3f] orientation: [%.3f, "
                "%.3f, %.3f, %.3f]",
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z,
                pose.orientation.w);

    if (isValid(pose, 10)) {
      target_pose_ = pose;
      target_wrench_ = grasp.wrench;
      RCLCPP_INFO(get_logger(), "\033[1;34mGrasp score: %.2f\033[0m",
                  grasp.score);
      visualizePose(target_joint_values_, "Target_Grasp_Pose");
      transitionTo(State::PREGRASP);
      busy_ = false;
      return;
    }
    RCLCPP_WARN(get_logger(), "-> Grasp is invalid. Skipping.");
  }

  busy_ = false;
  RCLCPP_ERROR(get_logger(), "No feasible grasp found.");
  transitionTo(State::ERROR);
}

void TaskExecutor::visualizePose(const std::vector<double> &joint_values,
                                 const std::string &text) {
  RCLCPP_INFO(get_logger(), "Visualizing pose...");
  RCLCPP_INFO(get_logger(),
              "Joint values: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
              joint_values[0], joint_values[1], joint_values[2],
              joint_values[3], joint_values[4], joint_values[5]);
  // Visualize target pose for debugging
  moveit_visual_tools_->deleteAllMarkers();
  moveit::core::RobotStatePtr goal_state =
      move_group_interface_->getCurrentState();
  goal_state->setJointGroupPositions(
      goal_state->getJointModelGroup("ur_manipulator"), joint_values);
  moveit_visual_tools_->publishRobotState(goal_state, rviz_visual_tools::GREEN);

  if (!text.empty()) {
    auto text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 0.5;
    moveit_visual_tools_->publishText(text_pose, text, rviz_visual_tools::BLUE,
                                      rviz_visual_tools::XLARGE);
  }

  moveit_visual_tools_->trigger();
  moveit_visual_tools_->prompt("Press 'Next' to continue.");
}

void TaskExecutor::visualizePose(const geometry_msgs::msg::Pose &pose,
                                 const std::string &id) {
  RCLCPP_INFO(get_logger(), "Visualizing pose...");
  // Visualize target pose for debugging
  moveit_visual_tools_->deleteAllMarkers();
  moveit_visual_tools_->publishAxisLabeled(pose, id);
  moveit::core::RobotStatePtr goal_state =
      move_group_interface_->getCurrentState(1.0);
  if (!goal_state) {
    RCLCPP_WARN(get_logger(),
                "Could not get robot state for visualization, skipping.");
    return;
  }
  goal_state->setFromIK(goal_state->getJointModelGroup("ur_manipulator"), pose,
                        "tool0", 0.1);
  moveit_visual_tools_->publishRobotState(goal_state, rviz_visual_tools::GREEN);

  moveit_visual_tools_->trigger();
  moveit_visual_tools_->prompt("Press 'Next' to continue.");
}

std::vector<double> TaskExecutor::getCurrentJointValues() {
  // Get current joint values clipped to [-pi, pi]
  auto current_state = move_group_interface_->getCurrentState();
  std::vector<double> joint_values;
  current_state->copyJointGroupPositions(joint_model_group_, joint_values);
  for (size_t i = 0; i < joint_values.size(); ++i) {
    // RCLCPP_INFO(get_logger(), "Raw joint value %lu: %.3f", i,
    // joint_values[i]);
    joint_values[i] =
        std::fmod(std::fmod(joint_values[i], 2 * M_PI) + M_PI, 2 * M_PI) - M_PI;
    // RCLCPP_INFO(get_logger(), "Clipped joint value %lu: %.3f", i,
    //             joint_values[i]);
  }
  return joint_values;
}
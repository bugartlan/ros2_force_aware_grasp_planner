#include "task_executor.hpp"

TaskExecutor::TaskExecutor(const rclcpp::NodeOptions &options)
    : Node("task_executor", options) {
  // Parameters
  this->declare_parameter("force_duration_sec", 5.0);
  this->declare_parameter("force_magnitude", 10.0);
  this->declare_parameter("motion_timeout_sec", 15.0);
  this->declare_parameter("base_link", "ur5e_base_link");

  // [x, y, z, qx, qy, qz, qw]
  this->declare_parameter(
      "tf_base_to_object",
      std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});

  // Service Clients
  switch_client_ =
      create_client<SwitchController>("/controller_manager/switch_controller");
  gripper_client_ =
      create_client<std_srvs::srv::SetBool>("gripper_command");
  force_start_client_ =
      create_client<SetForceMode>("/force_mode_controller/start_force_mode");
  force_stop_client_ =
      create_client<Trigger>("/force_mode_controller/stop_force_mode");

  // Publishers
  wrench_pub_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/force_mode_controller/wrench_command", 10);

  // Subscribers
  pose_sub_ = create_subscription<geometry_msgs::msg::Pose>(
      "/target_grasp_pose", 10,
      std::bind(&TaskExecutor::poseCallback, this, std::placeholders::_1));
  wrench_sub_ = create_subscription<geometry_msgs::msg::Wrench>(
      "/target_grasp_wrench", 10,
      std::bind(&TaskExecutor::wrenchCallback, this, std::placeholders::_1));
  grasp_array_sub_ = create_subscription<grasp_interfaces::msg::GraspArray>(
      "/target_grasps", 10,
      std::bind(&TaskExecutor::graspArrayCallback, this,
                std::placeholders::_1));

  auto tf_base_to_object_param =
      this->get_parameter("tf_base_to_object").as_double_array();
  tf_base_to_object_.setOrigin(tf2::Vector3(tf_base_to_object_param[0],
                                            tf_base_to_object_param[1],
                                            tf_base_to_object_param[2]));
  tf_base_to_object_.setRotation(
      tf2::Quaternion(tf_base_to_object_param[3], tf_base_to_object_param[4],
                      tf_base_to_object_param[5], tf_base_to_object_param[6]));

  RCLCPP_INFO(get_logger(),
              "Task Executor Node Created. Listening for planner...");
}

void TaskExecutor::init() {
  move_group_interface_ =
      std::make_shared<moveit::planning_interface::MoveGroupInterface>(
          shared_from_this(), "ur_manipulator");
  move_group_interface_->startStateMonitor(2.0);
  moveit_visual_tools_ =
      std::make_shared<moveit_visual_tools::MoveItVisualTools>(
          shared_from_this(), "base_link", rviz_visual_tools::RVIZ_MARKER_TOPIC,
          move_group_interface_->getRobotModel());
  moveit_visual_tools_->deleteAllMarkers();
  moveit_visual_tools_->loadRemoteControl();
  transitionTo(State::STARTUP);
}

void TaskExecutor::execute(const geometry_msgs::msg::Pose &pose,
                           const geometry_msgs::msg::Wrench &wrench) {
  RCLCPP_INFO(get_logger(), "Executing task with target pose and wrench");
  target_pose_ = pose;
  target_wrench_ = wrench;
  transitionTo(State::MOVING);
}

void TaskExecutor::transitionTo(State next_state) {
  state_ = next_state;
  switch (state_) {
  case State::STARTUP:
    onStartup();
    break;
  case State::MOVING:
    onMoving();
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

void TaskExecutor::returnHome() {
  RCLCPP_INFO(get_logger(), "Returning to home position...");
  move_group_interface_->setJointValueTarget(home_positions);
  move_group_interface_->move();
}

void TaskExecutor::onStartup() {
  // Open gripper at startup
  if (!gripper_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Gripper service not available!");
    transitionTo(State::ERROR);
    return;
  }
  auto gripper_request = std::make_shared<std_srvs::srv::SetBool::Request>();
  gripper_request->data = true; // Open gripper
  gripper_client_->async_send_request(gripper_request);

  // Activate passthrough trajectory and force mode controllers at startup
  RCLCPP_INFO(get_logger(), "Activating controllers");
  switchControllers(
      {"scaled_joint_trajectory_controller"},
      {"force_mode_controller", "joint_trajectory_controller"}, [this]() {
        returnHome();
        RCLCPP_INFO(get_logger(),
                    "Initialization complete, ready to execute tasks");
      });
}

void TaskExecutor::onMoving() {
  RCLCPP_INFO(get_logger(), "Starting motion to target pose...");
  move_group_interface_->setPlannerId("RRTStar");
  move_group_interface_->setMaxVelocityScalingFactor(0.05);
  move_group_interface_->setMaxAccelerationScalingFactor(0.05);

  // Direct cartesian target
  // move_group_interface_->setPoseTarget(target_pose_);

  // Joint space target from IK
  move_group_interface_->setJointValueTarget(target_joint_values_);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  if (move_group_interface_->plan(plan) ==
      moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_INFO(get_logger(), "Plan found! Visualizing trajectory...");
    moveit_visual_tools_->deleteAllMarkers();
    const moveit::core::JointModelGroup *jmg =
        move_group_interface_->getRobotModel()->getJointModelGroup(
            "ur_manipulator");
    moveit_visual_tools_->publishTrajectoryLine(
        plan.trajectory_,
        move_group_interface_->getRobotModel()->getLinkModel("tool0"), jmg);
    moveit_visual_tools_->trigger();
    moveit_visual_tools_->prompt(
        "Press 'Next' in the RvizVisualToolsGui window to execute");

    RCLCPP_INFO(get_logger(), "Executing...");
    move_group_interface_->execute(plan);
    switchControllers(
        {"force_mode_controller"},
        {"scaled_joint_trajectory_controller", "joint_trajectory_controller"},
        [this]() { transitionTo(State::GRASPING); });
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to compute a valid trajectory");
    transitionTo(State::ERROR);
  }

  // moveit_msgs::msg::RobotTrajectory trajectory;
  // double fraction = move_group_interface_->computeCartesianPath(
  //     {target_pose_}, 0.01, 0.0, trajectory);
  // RCLCPP_INFO(get_logger(), "Cartesian path computed with %.2f%% success",
  //             fraction * 100.0);
  // if (fraction > 0.99) {
  //   move_group_interface_->execute(trajectory);
  //   switchControllers(
  //       {"force_mode_controller"},
  //       {"scaled_joint_trajectory_controller",
  //       "joint_trajectory_controller"}, [this]() {
  //       transitionTo(State::GRASPING); });
  // } else {
  //   RCLCPP_ERROR(get_logger(), "Failed to compute a valid trajectory");
  //   transitionTo(State::ERROR);
  // }
}

void TaskExecutor::onGrasping() {
  RCLCPP_INFO(get_logger(), "Starting grasping phase...");
  auto gripper_request = std::make_shared<std_srvs::srv::SetBool::Request>();
  gripper_request->data = false; // Close gripper
  gripper_client_->async_send_request(gripper_request);

  // Wait for .5 second
  rclcpp::sleep_for(std::chrono::milliseconds(1000));
  transitionTo(State::FORCE);
}

void TaskExecutor::onForce() {
  // Check if force start service is available
  if (!force_start_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Force start service not available!");
    transitionTo(State::ERROR);
    return;
  }

  RCLCPP_INFO(get_logger(), "Applying wrench");
  auto request = std::make_shared<SetForceMode::Request>();
  request->task_frame.header.frame_id =
      this->get_parameter("base_link").as_string();
  request->task_frame.pose.orientation.w = 1.0;

  request->selection_vector_x = 1;
  request->selection_vector_y = 1;
  request->selection_vector_z = 1;
  request->selection_vector_rx = 1;
  request->selection_vector_ry = 1;
  request->selection_vector_rz = 1;
  request->wrench = target_wrench_;
  request->type = SetForceMode::Request::NO_TRANSFORM;
  request->speed_limits.linear.x = 0.1;
  request->speed_limits.linear.y = 0.1;
  request->speed_limits.linear.z = 0.1;
  request->speed_limits.angular.x = 0.1;
  request->speed_limits.angular.y = 0.1;
  request->speed_limits.angular.z = 0.1;
  force_start_client_->async_send_request(
      request, [this](rclcpp::Client<SetForceMode>::SharedFuture future) {
        try {
          auto response = future.get();
          if (response->success) {
            RCLCPP_INFO(this->get_logger(), "Force mode started successfully");
            // Start wrench publisher timer
            wrench_timer_ =
                this->create_wall_timer(std::chrono::milliseconds(8), [this]() {
                  geometry_msgs::msg::WrenchStamped msg;
                  msg.header.stamp = this->now();
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
                      std::make_shared<Trigger::Request>());
                  transitionTo(State::DONE);
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
        }
      });
}

void TaskExecutor::poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "Received target pose from planner");
  RCLCPP_INFO(get_logger(),
              "Pose - position: [%.3f, %.3f, %.3f] orientation: [%.3f, %.3f, "
              "%.3f, %.3f]",
              msg->position.x, msg->position.y, msg->position.z,
              msg->orientation.x, msg->orientation.y, msg->orientation.z,
              msg->orientation.w);
  target_pose_ = transformPose(*msg, tf_base_to_object_);
  RCLCPP_INFO(get_logger(),
              "Transformed Pose - position: [%.3f, %.3f, %.3f] orientation: "
              "[%.3f, %.3f, %.3f, %.3f]",
              target_pose_.position.x, target_pose_.position.y,
              target_pose_.position.z, target_pose_.orientation.x,
              target_pose_.orientation.y, target_pose_.orientation.z,
              target_pose_.orientation.w);

  pose_received_ = true;
  checkAndExecute();
}

void TaskExecutor::wrenchCallback(
    const geometry_msgs::msg::Wrench::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "Received target wrench from planner");
  RCLCPP_INFO(get_logger(),
              "Wrench - force: [%.3f, %.3f, %.3f] torque: [%.3f, %.3f, %.3f]",
              msg->force.x, msg->force.y, msg->force.z, msg->torque.x,
              msg->torque.y, msg->torque.z);
  target_wrench_ = *msg;
  wrench_received_ = true;
  checkAndExecute();
}

void TaskExecutor::graspArrayCallback(
    const grasp_interfaces::msg::GraspArray::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "Received target grasps from planner");
  target_grasps_ = *msg;
  grasps_received_ = true;
  std::thread([this]() { checkAndExecute(); }).detach();
}

void TaskExecutor::checkAndExecute() {
  if (grasps_received_) {
    // Reset flags so we don't double-trigger
    grasps_received_ = false;

    moveit::core::RobotStatePtr kinematic_state =
        move_group_interface_->getCurrentState();
    if (!kinematic_state) {
      RCLCPP_ERROR(get_logger(), "Failed to get current robot state");
      transitionTo(State::ERROR);
      return;
    }

    const moveit::core::JointModelGroup *joint_model_group =
        kinematic_state->getJointModelGroup("ur_manipulator");

    bool found_feasible_grasp = false;
    for (auto &grasp : target_grasps_.grasps) {
      RCLCPP_INFO(get_logger(),
                  "Checking grasp feasibility and singularity...");
      auto pose = transformPose(grasp.pose, tf_base_to_object_);
      RCLCPP_INFO(get_logger(),
                  "Grasp - position: [%.3f, %.3f, %.3f] orientation: [%.3f, "
                  "%.3f, %.3f, %.3f]",
                  pose.position.x, pose.position.y, pose.position.z,
                  pose.orientation.x, pose.orientation.y, pose.orientation.z,
                  pose.orientation.w);
      kinematic_state = move_group_interface_->getCurrentState();
      if (kinematic_state->setFromIK(joint_model_group, pose, "tool0", 0.1)) {
        RCLCPP_INFO(get_logger(), "-> Grasp is reachable (IK solution found)");
        RCLCPP_INFO(get_logger(), "-> Checking for manipulability...");
        Eigen::MatrixXd jacobian;
        if (!kinematic_state->getJacobian(
                joint_model_group, kinematic_state->getLinkModel("tool0"),
                Eigen::Vector3d::Zero(), jacobian)) {
          RCLCPP_ERROR(get_logger(), "Failed to get Jacobian");
          transitionTo(State::ERROR);
          continue;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
        double manipulability = svd.singularValues().prod();
        RCLCPP_INFO(get_logger(), "-> Manipulability: %.6f", manipulability);
        if (manipulability < MANIPULABILITY_THRESHOLD) {
          RCLCPP_WARN(get_logger(), "-> Grasp is near singularity.");
          continue;
        }
        target_pose_ = pose;
        target_wrench_ = grasp.wrench;
        kinematic_state->copyJointGroupPositions(joint_model_group,
                                                 target_joint_values_);
        found_feasible_grasp = true;
        break;
      }
      RCLCPP_WARN(get_logger(),
                  "-> Grasp is unreachable (No IK solution). Skipping.");
    }
    if (!found_feasible_grasp) {
      RCLCPP_ERROR(get_logger(), "No feasible grasp found.");
      transitionTo(State::ERROR);
    } else {
      transitionTo(State::MOVING);
    }
  }
}

geometry_msgs::msg::Pose
TaskExecutor::transformPose(const geometry_msgs::msg::Pose &pose,
                            const tf2::Transform &tf) {
  tf2::Transform pose_tf(
      tf2::Quaternion(pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w),
      tf2::Vector3(pose.position.x, pose.position.y, pose.position.z));

  tf2::Transform result_tf = tf * pose_tf;

  geometry_msgs::msg::Pose transformed_pose;
  transformed_pose.position.x = result_tf.getOrigin().x();
  transformed_pose.position.y = result_tf.getOrigin().y();
  transformed_pose.position.z = result_tf.getOrigin().z();
  transformed_pose.orientation.x = result_tf.getRotation().x();
  transformed_pose.orientation.y = result_tf.getRotation().y();
  transformed_pose.orientation.z = result_tf.getRotation().z();
  transformed_pose.orientation.w = result_tf.getRotation().w();

  return transformed_pose;
}

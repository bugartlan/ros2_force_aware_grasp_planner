#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <ur_msgs/srv/set_force_mode.hpp>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto const node = std::make_shared<rclcpp::Node>(
      "force_control_node",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
          true));

  auto logger = rclcpp::get_logger("force_control_node");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  auto switch_controller_client =
      node->create_client<controller_manager_msgs::srv::SwitchController>(
          "/controller_manager/switch_controller");
  switch_controller_client->wait_for_service();

  auto request = std::make_shared<
      controller_manager_msgs::srv::SwitchController::Request>();
  request->activate_controllers = {"force_mode_controller"};
  request->deactivate_controllers = {"scaled_joint_trajectory_controller"};
  RCLCPP_INFO(logger, "Sending request to switch controllers...");
  auto future = switch_controller_client->async_send_request(request);

  auto set_force_mode_client = node->create_client<ur_msgs::srv::SetForceMode>(
      "/force_mode_controller/start_force_mode");
  set_force_mode_client->wait_for_service();

  auto force_request = std::make_shared<ur_msgs::srv::SetForceMode::Request>();
  force_request->task_frame.header.frame_id = "base_link";
  force_request->task_frame.pose.orientation.w = 1.0;

  geometry_msgs::msg::Wrench wrench;
  wrench.force.x = 0.0;
  wrench.force.y = 0.0;
  wrench.force.z = 1.0;
  wrench.torque.x = 0.0;
  wrench.torque.y = 0.0;
  wrench.torque.z = 0.0;

  force_request->selection_vector_x = 1;
  force_request->selection_vector_y = 1;
  force_request->selection_vector_z = 1;
  force_request->selection_vector_rx = 1;
  force_request->selection_vector_ry = 1;
  force_request->selection_vector_rz = 1;
  force_request->wrench = wrench;
  force_request->type = ur_msgs::srv::SetForceMode::Request::NO_TRANSFORM;
  force_request->speed_limits.linear.x = 0.1;
  force_request->speed_limits.linear.y = 0.1;
  force_request->speed_limits.linear.z = 0.1;
  force_request->speed_limits.angular.x = 0.1;
  force_request->speed_limits.angular.y = 0.1;
  force_request->speed_limits.angular.z = 0.1;



RCLCPP_INFO(logger, "Sending request to start force mode...");
  auto force_future = set_force_mode_client->async_send_request(force_request, [&](rclcpp::Client<ur_msgs::srv::SetForceMode>::SharedFuture future) {
    try {
          auto response = future.get();
          if (response->success) {
            RCLCPP_INFO(logger, "Force mode started successfully");
            // Start wrench publisher timer
            // auto wrench_timer_ = node->create_wall_timer(
            //     std::chrono::milliseconds(100), [&]() {
            //       geometry_msgs::msg::WrenchStamped msg;
            //       msg.header.stamp = node->now();
            //       msg.wrench = wrench;
            //       wrench_pub_->publish(msg);
            //     });

            // Stop after duration
            // duration_timer_ = this->create_wall_timer(
            //     std::chrono::duration<double>(
            //         this->get_parameter("force_duration_sec").as_double()),
            //     [this]() {
            //       duration_timer_->cancel();
            //       // Stop force mode
            //       if (wrench_timer_)
            //         wrench_timer_->cancel();
            //       force_stop_client_->async_send_request(
            //           std::make_shared<Trigger::Request>());
            //     });
          } else {
            RCLCPP_ERROR(logger, "Failed to start force mode");
          }
        } catch (const std::exception &e) {
          RCLCPP_ERROR(logger, "Service call failed: %s", e.what());
        }
  });

  return 0;
}
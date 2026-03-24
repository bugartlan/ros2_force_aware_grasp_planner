#include "task_executor.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  std::thread spin_thread([&executor]() { executor.spin(); });

  rclcpp::NodeOptions options;
  options.parameter_overrides({
      rclcpp::Parameter("force_duration_sec", 5.0),
      rclcpp::Parameter("force_magnitude", 10.0),
      rclcpp::Parameter("motion_timeout_sec", 15.0),
      rclcpp::Parameter("base_link", "base_link"),
      rclcpp::Parameter(
          "tf_base_to_object",
          std::vector<double>{-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}),
  });

  auto node = std::make_shared<TaskExecutor>(options);
  executor.add_node(node);
  node->init();
  spin_thread.join();

  rclcpp::shutdown();
  return 0;
}
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Transform.h>

namespace task_executor_utils {

geometry_msgs::msg::Pose
applyWorldTransformToPose(const geometry_msgs::msg::Pose &pose,
                          const tf2::Transform &tf);
geometry_msgs::msg::Pose
applyLocalTransformToPose(const geometry_msgs::msg::Pose &pose,
                          const tf2::Transform &tf);

geometry_msgs::msg::Pose transformToPoseMsg(const tf2::Transform &tf);

tf2::Transform transformFromParams(const std::vector<double> &p);
} // namespace task_executor_utils
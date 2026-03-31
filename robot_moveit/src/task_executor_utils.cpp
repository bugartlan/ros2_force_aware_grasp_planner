#include "task_executor_utils.hpp"

geometry_msgs::msg::Pose task_executor_utils::applyWorldTransformToPose(
    const geometry_msgs::msg::Pose &pose, const tf2::Transform &tf) {
  // Pre-multiply transform (world-frame): result = tf * pose.
  tf2::Transform pose_tf(
      tf2::Quaternion(pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w),
      tf2::Vector3(pose.position.x, pose.position.y, pose.position.z));
  tf2::Transform result_tf = tf * pose_tf;
  return transformToPoseMsg(result_tf);
}

geometry_msgs::msg::Pose task_executor_utils::applyLocalTransformToPose(
    const geometry_msgs::msg::Pose &pose, const tf2::Transform &tf) {
  // Post-multiply transform (local-frame): result = pose * tf.
  tf2::Transform pose_tf(
      tf2::Quaternion(pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w),
      tf2::Vector3(pose.position.x, pose.position.y, pose.position.z));
  tf2::Transform result_tf = pose_tf * tf;
  return transformToPoseMsg(result_tf);
}

geometry_msgs::msg::Pose
task_executor_utils::transformToPoseMsg(const tf2::Transform &tf) {
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position.x = tf.getOrigin().x();
  pose_msg.position.y = tf.getOrigin().y();
  pose_msg.position.z = tf.getOrigin().z();
  pose_msg.orientation.x = tf.getRotation().x();
  pose_msg.orientation.y = tf.getRotation().y();
  pose_msg.orientation.z = tf.getRotation().z();
  pose_msg.orientation.w = tf.getRotation().w();
  return pose_msg;
}

tf2::Transform
task_executor_utils::transformFromParams(const std::vector<double> &p) {
  if (p.size() != 7) {
    throw std::runtime_error(
        "Expected 7 parameters for transform (x, y, z, qx, qy, qz, qw)");
  }
  tf2::Transform tf;
  tf.setOrigin(tf2::Vector3(p[0], p[1], p[2]));
  tf.setRotation(tf2::Quaternion(p[3], p[4], p[5], p[6]));
  return tf;
}
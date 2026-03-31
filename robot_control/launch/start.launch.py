from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")
    kinematics_params_file = LaunchConfiguration("kinematics_params_file")
    show_object = LaunchConfiguration("show_object")

    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.56.101",  # put your robot's IP address here
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz", default_value="false", description="Launch RViz?"
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware", default_value="false", description="Use fake hardware?"
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "kinematics_params_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("ur_description"),
                    "config",
                    "ur3e",
                    "default_kinematics.yaml",
                ]
            ),
            description="Path to the kinematics parameters file.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "show_object",
            default_value="false",
            description="Whether to show the object mesh in RViz.",
        )
    )

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare("robot_description"),
            "rviz",
            "moveit.rviz",
        ]
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    rviz_mesh_publisher_node = Node(
        package="rviz_mesh_publisher",
        condition=IfCondition(show_object),
        executable="publisher",
        name="rviz_mesh_publisher",
        output="log",
    )

    gripper_controller_node = Node(
        package="gripper_control",
        executable="gripper_control",
        name="gripper_controller",
        output="log",
    )

    return LaunchDescription(
        declared_arguments
        + [
            rviz_node,  # spawn RViz node with my custom config
            rviz_mesh_publisher_node,  # spawn RViz mesh publisher node with my custom config
            gripper_controller_node,  # spawn gripper controller node
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare("ur_robot_driver"),
                                "launch",
                                "ur_control.launch.py",
                            ]
                        )
                    ]
                ),
                launch_arguments={
                    "ur_type": "ur3e",
                    "robot_ip": robot_ip,
                    "tf_prefix": "",
                    "description_package": "robot_description",
                    "description_file": "robot.urdf.xacro",
                    "use_fake_hardware": use_fake_hardware,
                    "launch_rviz": "false",  # RViz is launched separately above
                    "kinematics_params_file": kinematics_params_file,
                }.items(),
            ),
        ]
    )

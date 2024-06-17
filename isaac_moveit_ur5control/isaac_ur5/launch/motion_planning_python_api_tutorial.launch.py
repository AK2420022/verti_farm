
"""
A launch file for running the motion planning python api tutorial
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS 2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="ur5_robot", package_name="moveit_resources_ur5_moveit_config"
        )
        .robot_description(file_path="config/ur5_robot.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },)
        .robot_description_semantic(file_path="config/ur5_robot.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .moveit_cpp(
            file_path="config/motion_planning_python.yaml"
        )
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner","chomp"]
        )
        #.sensors_3d("config/sensors_realsense.yaml")
        .to_moveit_configs()
    )

    example_file = DeclareLaunchArgument(
        "example_file",
        default_value="motion_planning_python.py",
        description="Python API tutorial file name",
    )

    moveit_py_node = Node(
        name="moveit_py",
        package="moveit_resources_ur5_moveit_config",
        executable=LaunchConfiguration("example_file"),
        output="both",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
    )

    rviz_config_file = os.path.join(
        get_package_share_directory("moveit_resources_ur5_moveit_config"),
        "config",
        "motion_planning_python_api_tutorial.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
    )

    # Static TF
    world2robot_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )
    # Static TF
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        parameters=[moveit_config.robot_description,{"use_sim_time": True},],
        arguments=["--frame-id", "wrist_3_link", "--child-frame-id", "camera"],
    )
    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description,{"use_sim_time": True},],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_ur5_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path, {"use_sim_time": True},],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )
    load_controllers = []
    for controller in [
        "manipulator_controller",
    ]:
        load_controllers += [
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner {}".format(controller)],
                shell=True,
                output="log",
            )
        ]
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name = 'joint_state_publisher',
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["manipulator_controller", "-c", "/controller_manager"],
    )
    return LaunchDescription(
        [
            ros2_control_hardware_type,
            robot_state_publisher,
            example_file,
	    world2robot_tf_node,
	    static_tf_node,
            ros2_control_node,
            rviz_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            moveit_py_node,

        ]

    )


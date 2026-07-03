"""Launch the staged, safety-gated WBC runner for model_15200.

Milestone 1 (dry-run, standalone with planner_imitate):
    # terminal A - fake planner:
    ros2 launch hope_planner planner_imitate.launch.py dry_run:=false level:=2
    # terminal B - WBC runner (dry-run, logs action/joint targets, NO hardware):
    ros2 launch hope_wbc_runner wbc_runner.launch.py \
        onnx_path:=/abs/path/to/exported/policy.onnx mode:=dry_run csv_path:=/tmp/wbc_runner.csv

Note: planner_imitate dry_run:=false only means it PUBLISHES /racket/command; the
WBC runner mode=dry_run is what guarantees no joint commands are sent.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("hope_wbc_runner")) / "config" / "wbc_runner.yaml"
    args = [
        DeclareLaunchArgument("mode", default_value="dry_run",
                              description="dry_run (log only) | shadow (predict on real state) | hardware"),
        DeclareLaunchArgument("onnx_path", default_value="",
                              description="REQUIRED: path to exported model_15200 policy.onnx"),
        DeclareLaunchArgument("hardware_enable", default_value="false",
                              description="must be true (and estop false) to publish in hardware mode"),
        DeclareLaunchArgument("state_source", default_value="synthetic",
                              description="synthetic (dry-run) | ros (shadow/hardware: subscribe joints+imu)"),
        DeclareLaunchArgument("joint_state_topic", default_value="/a3/joint_states",
                              description="robot joint feedback topic (shadow/hardware)"),
        DeclareLaunchArgument("joint_state_type", default_value="sensor_msgs",
                              description="'sensor_msgs' (std JointState) | 'joint_msgs' (a3 sim JointState)"),
        DeclareLaunchArgument("imu_topic", default_value="/a3/imu",
                              description="sensor_msgs/Imu topic"),
        DeclareLaunchArgument("base_pose_topic", default_value="/P1/pose",
                              description="geometry_msgs/PoseStamped pelvis world pose from mocap relay; "
                                          "set '' to use nominal height fallback (standalone dry-run without mocap)"),
        DeclareLaunchArgument("csv_path", default_value="",
                              description="CSV log path ('' disables)"),
    ]
    node = Node(
        package="hope_wbc_runner",
        executable="wbc_runner_node",
        name="wbc_runner",
        output="screen",
        parameters=[
            str(config),
            {
                "mode": LaunchConfiguration("mode"),
                "onnx_path": LaunchConfiguration("onnx_path"),
                "hardware_enable": LaunchConfiguration("hardware_enable"),
                "state_source": LaunchConfiguration("state_source"),
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "joint_state_type": LaunchConfiguration("joint_state_type"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "base_pose_topic": LaunchConfiguration("base_pose_topic"),
                "csv_path": LaunchConfiguration("csv_path"),
            },
        ],
    )
    return LaunchDescription(args + [node])

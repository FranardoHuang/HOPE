"""Launch the HOPE model-based planner node with the default config."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("hope_planner")) / "config" / "hope_planner.yaml"
    task_revision_config = (
        Path(get_package_share_directory("hope_planner"))
        / "config"
        / "hope_planner.task_revision.yaml"
    )
    task_revision = LaunchConfiguration("task_revision")
    return LaunchDescription([
        DeclareLaunchArgument(
            "task_revision",
            default_value="false",
            description=(
                "Opt in to schema-4 task revisions. When true, the task-revision "
                "overlay is loaded after the base planner config."
            ),
        ),
        Node(
            package="hope_planner",
            executable="hope_planner_node",
            name="hope_planner",
            output="screen",
            parameters=[str(config)],
            condition=UnlessCondition(task_revision),
        ),
        Node(
            package="hope_planner",
            executable="hope_planner_node",
            name="hope_planner",
            output="screen",
            parameters=[str(config), str(task_revision_config)],
            condition=IfCondition(task_revision),
        ),
    ])

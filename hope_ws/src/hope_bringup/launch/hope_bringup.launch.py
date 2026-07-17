"""Generic HOPE PingPong bringup: motion capture -> planner.

Starts the racket planner and its ball source. By default it launches the
``vrpn_mocap`` VRPN client (configurable server address, default ``localhost``)
so the planner runs against a real motion-capture stream. For testing without
mocap, set ``use_fake_ball:=true`` to publish a synthetic ``/poses`` stream
instead.

The planner subscribes to ``poses_topic`` (a ``geometry_msgs/PoseArray`` with
the ball at ``ball_pose_index``). When using a real tracker, map/relay your
mocap system's ball pose onto that topic; ``fake_ball_publisher`` already
publishes it in the expected form.

Examples::

    # Real mocap on this machine:
    ros2 launch hope_bringup hope_bringup.launch.py mocap_server:=localhost

    # Real mocap on another host:
    ros2 launch hope_bringup hope_bringup.launch.py mocap_server:=mocap.local mocap_port:=3883

    # No mocap, synthetic ball for a smoke test:
    ros2 launch hope_bringup hope_bringup.launch.py use_fake_ball:=true
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mocap_server = LaunchConfiguration("mocap_server")
    mocap_port = LaunchConfiguration("mocap_port")
    use_fake_ball = LaunchConfiguration("use_fake_ball")

    planner_config = Path(get_package_share_directory("hope_planner")) / "config" / "hope_planner.yaml"

    vrpn_client = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("vrpn_mocap"), "launch", "client.launch.yaml"])
        ),
        launch_arguments={"server": mocap_server, "port": mocap_port}.items(),
        condition=UnlessCondition(use_fake_ball),
    )

    fake_ball = Node(
        package="hope_bringup",
        executable="fake_ball_publisher",
        name="fake_ball_publisher",
        output="screen",
        condition=IfCondition(use_fake_ball),
    )

    planner = Node(
        package="hope_planner",
        executable="hope_planner_node",
        name="hope_planner",
        output="screen",
        parameters=[str(planner_config)],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "mocap_server", default_value="localhost",
            description="VRPN motion-capture server IP/hostname."),
        DeclareLaunchArgument(
            "mocap_port", default_value="3883",
            description="VRPN motion-capture server port."),
        DeclareLaunchArgument(
            "use_fake_ball", default_value="false",
            description="Publish a synthetic /poses ball stream instead of starting vrpn_mocap."),
        vrpn_client,
        fake_ball,
        planner,
    ])

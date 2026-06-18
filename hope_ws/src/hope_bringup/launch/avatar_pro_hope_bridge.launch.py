"""Full Avatar-Pro (Chingmu VRPN) -> HOPE mocap bringup.

Chain:  Avatar-Pro / CMTracker (MCServer)  --VRPN-->  vrpn_mocap client_node
        --> /vrpn_mocap/<object>/pose  --> avatar_pro_vrpn_relay
        --> /poses, /tf, /ball/point, /{table,P1,P2}/pose  --> hope_planner / WBC

Also starts the static HOPE world frame (table landmarks + mocap->base_link).

TODO before running on hardware (see reimplement.md "Values you must fill in"):
  * server   -> the Avatar-Pro / CMTracker (MCServer) PC IP on the arena LAN.
  * port     -> the Chingmu VRPN port (default 3883).
  * update_freq -> match the mocap camera rate (>= 240-360 Hz for the ball).
  * object names + mocap->base_link offsets are set in the config files.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    config_path = launch_dir.parent / "config" / "avatar_pro_vrpn.yaml"

    server = LaunchConfiguration("server")
    port = LaunchConfiguration("port")
    update_freq = LaunchConfiguration("update_freq")
    start_vrpn_client = LaunchConfiguration("start_vrpn_client")
    start_world = LaunchConfiguration("start_world")

    return LaunchDescription([
        DeclareLaunchArgument(
            "server", default_value="192.168.1.100",
            description="TODO: Avatar-Pro / Chingmu VRPN server (MCServer PC) IP on the arena LAN.",
        ),
        DeclareLaunchArgument("port", default_value="3883", description="Chingmu VRPN port."),
        DeclareLaunchArgument(
            "update_freq", default_value="360.0",
            description="VRPN client poll rate (Hz); match the mocap camera rate.",
        ),
        DeclareLaunchArgument("start_vrpn_client", default_value="true"),
        DeclareLaunchArgument("start_world", default_value="true"),

        # Static HOPE world frame: table landmarks + mocap->base_link offsets.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_dir / "hope_world.launch.py")),
            condition=IfCondition(start_world),
        ),

        # VRPN client: Avatar-Pro server -> /vrpn_mocap/<object>/pose (+ /tf).
        Node(
            package="vrpn_mocap",
            executable="client_node",
            namespace="vrpn_mocap",
            name="client_node",
            output="screen",
            condition=IfCondition(start_vrpn_client),
            parameters=[{
                "server": server,
                "port": ParameterValue(port, value_type=int),
                "frame_id": "world",
                "update_freq": ParameterValue(update_freq, value_type=float),
                "multi_sensor": False,
            }],
        ),

        # Relay: /vrpn_mocap/* -> HOPE-standard topics.
        Node(
            package="hope_bringup",
            executable="avatar_pro_vrpn_relay",
            name="avatar_pro_vrpn_relay",
            output="screen",
            parameters=[str(config_path)],
        ),
    ])

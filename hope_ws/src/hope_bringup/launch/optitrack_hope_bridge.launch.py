"""Full OptiTrack (Motive/NatNet) -> HOPE mocap bringup.

Chain:  Motive (NatNet UDP, cmd port 1510)  -->  motion_capture_tracking_node
        (vendored, namespace /optitrack; poses/tf remapped to /optitrack/*)
        --> optitrack_mct_relay
        --> /poses, /tf, /ball/point, /{table,P1,P2}/pose  --> hope_planner

OptiTrack sibling of the VRPN path (``vrpn_mocap`` + ``pose_to_posearray``,
wired by ``hope_bringup.launch.py``); both feed the identical ``/poses``
contract (ball at index 0), so everything downstream is unchanged. This launch
starts the mocap side only — run the planner via ``hope_bringup.launch.py``
(``mocap_backend:=optitrack`` includes this file), or separately.
Also starts the static HOPE world frame (table landmarks + mocap->base_link).

The remaps are LOAD-BEARING, not cosmetic:
  * the driver's `poses` topic is motion_capture_tracking_interfaces/
    NamedPoseArray -- on the bare /poses name it would collide with the HOPE
    /poses contract (geometry_msgs/PoseArray) as a DDS type mismatch;
  * the driver broadcasts /tf with raw body names (P1/Table/...) which would
    fight the relay's world->P1/Table transforms and the hope_world statics.

Before running against a live rig (see docs/OPTITRACK.md):
  * hostname -> the Motive PC IP on the arena LAN.
  * Motive streaming pane per the mocap reference doc §6.3 (Z-up, Rigid
    Bodies ON, NatNet enabled; marker streaming per the ball MODE notes in
    config/optitrack_mct.yaml).
  * rigid-body names + ball tracker are set in config/optitrack_mct.yaml and
    config/optitrack_relay.yaml (standardized: P1/P2/Table assets, ball 'Ball').
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
    mct_config_path = launch_dir.parent / "config" / "optitrack_mct.yaml"
    relay_config_path = launch_dir.parent / "config" / "optitrack_relay.yaml"

    hostname = LaunchConfiguration("hostname")
    mocap_type = LaunchConfiguration("mocap_type")
    start_mocap_node = LaunchConfiguration("start_mocap_node")
    start_world = LaunchConfiguration("start_world")
    position_scale = LaunchConfiguration("position_scale")

    return LaunchDescription([
        DeclareLaunchArgument(
            "hostname",
            description="REQUIRED: Motive PC IP (NatNet server) on the mocap "
                        "LAN, e.g. hostname:=192.168.1.100. NatNet uses the "
                        "command port (default 1510) and auto-negotiates the "
                        "data port / unicast-vs-multicast from the server.",
        ),
        DeclareLaunchArgument(
            "mocap_type", default_value="optitrack",
            description="libmotioncapture backend: 'optitrack' (live Motive) or "
                        "'mock' (no-hardware smoke test; streams the yaml "
                        "rigid_bodies statically).",
        ),
        DeclareLaunchArgument(
            "start_mocap_node", default_value="true",
            description="Set false when replaying a recorded /optitrack/poses bag.",
        ),
        DeclareLaunchArgument("start_world", default_value="true"),
        DeclareLaunchArgument(
            "position_scale", default_value="1.0",
            description="Uniform position conversion applied by the relay. Motive "
                        "streams metres -> 1.0 (use 0.001 only for a millimetre feed).",
        ),

        # Static HOPE world frame: table landmarks + mocap->base_link offsets.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_dir / "hope_world.launch.py")),
            condition=IfCondition(start_world),
        ),

        # Vendored NatNet driver: Motive -> /optitrack/poses (NamedPoseArray),
        # /optitrack/tf, /optitrack/pointCloud. Motive-native rigid bodies
        # (P1/P2/Table) and the librigidbodytracker single-marker ball ('Ball',
        # from optitrack_mct.yaml) arrive in the same NamedPoseArray.
        Node(
            package="motion_capture_tracking",
            executable="motion_capture_tracking_node",
            namespace="optitrack",
            name="motion_capture_tracking_node",
            output="screen",
            condition=IfCondition(start_mocap_node),
            parameters=[
                str(mct_config_path),
                {"hostname": hostname, "type": mocap_type},
            ],
            remappings=[
                # `poses`/`pointCloud` are relative and already resolve under
                # /optitrack; /tf and /tf_static are absolute and must be
                # remapped explicitly (see module docstring).
                ("/tf", "/optitrack/tf"),
                ("/tf_static", "/optitrack/tf_static"),
            ],
        ),

        # Relay: /optitrack/poses -> HOPE-standard topics (the relay is the
        # only /tf authority for ball/Table/P1/P2).
        Node(
            package="hope_bringup",
            executable="optitrack_mct_relay",
            name="optitrack_mct_relay",
            output="screen",
            parameters=[
                str(relay_config_path),
                {"position_scale": ParameterValue(position_scale, value_type=float)},
            ],
        ),
    ])

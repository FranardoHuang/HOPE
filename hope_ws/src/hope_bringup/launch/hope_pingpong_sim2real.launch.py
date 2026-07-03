"""End-to-end HOPE ping-pong sim2real bring-up: mocap -> planner -> WBC -> robot.

One launch for the whole chain:

  AvatarPro/CMTracker (MCServer, VRPN)
    --> vrpn_mocap client_node        /vrpn_mocap/<obj>/pose[_id_N]
    --> avatar_pro_vrpn_relay         /poses (ball-synced), /P1/pose, /ball/point   [TABLE frame]
    --> hope_planner_node             /racket/command (frame_id="world")            [TABLE frame]
    --> wbc_runner                    (table->policy frame conversion inside)
    --> /a3/joint_command             (hardware mode + enable + !estop only)
    --> agibot_hardware_bridge        /body_drive/*_joint_command  --> AGI backend

Staged usage (SAFETY: default mode is dry_run — nothing is ever published):

  # 1. dry-run against live mocap (no robot needed; verifies mocap->planner->obs):
  ros2 launch hope_bringup hope_pingpong_sim2real.launch.py \
      server:=192.168.10.100 ball_tracking_mode:=rigid_body ball_object:=Ball \
      onnx_path:=/abs/path/policy.onnx mode:=dry_run

  # 2. shadow on the robot/sim (real joints+IMU+mocap feed the obs, still no output):
  ...same...  mode:=shadow state_source:=ros start_hw_bridge:=true

  # 3. hardware (publishes ONLY after `ros2 topic pub /hope/hardware_enable ... true`):
  ...same...  mode:=hardware state_source:=ros start_hw_bridge:=true

Prerequisites for shadow/hardware: source the AGI overlay for joint_msgs, and run
the AGI backend with an AimRT config that enables the ros2 backend for the
/body_drive/(.*) topics (the default sim/hardware configs are iceoryx-ONLY —
see a3_aimrt_config.ros2.yaml; without it the bridge hears silence).
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_launch = Path(__file__).resolve().parent
    planner_cfg = Path(get_package_share_directory("hope_planner")) / "config" / "hope_planner.yaml"
    runner_cfg = Path(get_package_share_directory("hope_wbc_runner")) / "config" / "wbc_runner.yaml"

    args = [
        # --- mocap (forwarded to avatar_pro_hope_bridge.launch.py) ---
        DeclareLaunchArgument("server", default_value="192.168.10.100",
                              description="AvatarPro / CMTracker (MCServer) PC IP."),
        DeclareLaunchArgument("port", default_value="3883"),
        DeclareLaunchArgument("update_freq", default_value="300.0",
                              description="VRPN poll rate; match the camera rate."),
        DeclareLaunchArgument("ball_tracking_mode", default_value="rigid_body",
                              description="'rigid_body' (ball_object names the CMTracker body; "
                                          "preferred) or 'auto' (motion-based marker lock)."),
        DeclareLaunchArgument("ball_object", default_value="Ball",
                              description="Ball rigid-body name in CMTracker."),
        DeclareLaunchArgument("start_mocap", default_value="true",
                              description="false = mocap bridge already running elsewhere."),
        # --- planner ---
        DeclareLaunchArgument("start_planner", default_value="true"),
        # --- WBC runner ---
        DeclareLaunchArgument("onnx_path", default_value="",
                              description="REQUIRED: exported policy.onnx (with clip metadata)."),
        DeclareLaunchArgument("mode", default_value="dry_run",
                              description="dry_run | shadow | hardware (see wbc_runner)."),
        DeclareLaunchArgument("state_source", default_value="synthetic",
                              description="synthetic | ros (shadow/hardware need ros)."),
        DeclareLaunchArgument("csv_path", default_value="",
                              description="wbc_runner CSV log ('' disables)."),
        # --- hardware bridge (only for shadow/hardware against the AGI backend) ---
        DeclareLaunchArgument("start_hw_bridge", default_value="false",
                              description="true = bridge /a3/* <-> /body_drive/* (needs joint_msgs)."),
    ]

    mocap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(bringup_launch / "avatar_pro_hope_bridge.launch.py")),
        condition=IfCondition(LaunchConfiguration("start_mocap")),
        launch_arguments={
            "server": LaunchConfiguration("server"),
            "port": LaunchConfiguration("port"),
            "update_freq": LaunchConfiguration("update_freq"),
            "ball_tracking_mode": LaunchConfiguration("ball_tracking_mode"),
            "ball_object": LaunchConfiguration("ball_object"),
        }.items(),
    )

    planner = Node(
        package="hope_planner",
        executable="hope_planner_node",
        name="hope_planner",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_planner")),
        parameters=[str(planner_cfg)],
    )

    runner = Node(
        package="hope_wbc_runner",
        executable="wbc_runner_node",
        name="wbc_runner",
        output="screen",
        parameters=[
            str(runner_cfg),
            {
                "onnx_path": LaunchConfiguration("onnx_path"),
                "mode": LaunchConfiguration("mode"),
                "state_source": LaunchConfiguration("state_source"),
                "csv_path": LaunchConfiguration("csv_path"),
            },
        ],
    )

    hw_bridge = Node(
        package="agibot_hardware_bridge",
        executable="agibot_hardware_bridge",
        name="agibot_hardware_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_hw_bridge")),
        parameters=[
            str(Path(get_package_share_directory("agibot_hardware_bridge"))
                / "config" / "agibot_bridge.yaml"),
        ],
    )

    return LaunchDescription(args + [mocap, planner, runner, hw_bridge])

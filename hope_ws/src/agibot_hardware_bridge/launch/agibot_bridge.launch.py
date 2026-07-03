"""Launch the AGI A3 hardware bridge node.

This bridge adapts the WBC runner's single merged joint topics to/from the
AGI body-drive multi-topic interface. Launch BEFORE starting wbc_runner in
shadow or hardware mode.

REQUIRES: joint_msgs on the ROS2 overlay path. In the hope distrobox run:
    source <agi_sim_dir>/setup_ros2_msgs.bash
before ros2 launch.

Usage (shadow mode bring-up sequence):
    # Terminal 1: AGI sim (inside hope distrobox)
    bash scripts/run_sim.sh

    # Terminal 2: AvatarPro bridge + planner (inside hope distrobox)
    ros2 launch hope_bringup avatar_pro_hope_bridge.launch.py \\
        server:=192.168.10.100 port:=3883 ball_tracking_mode:=rigid_body ball_object:=Ball

    ros2 launch hope_planner hope_planner.launch.py

    # Terminal 3: this bridge (inside hope distrobox)
    ros2 launch agibot_hardware_bridge agibot_bridge.launch.py

    # Terminal 4: WBC runner in shadow mode
    ros2 launch hope_wbc_runner wbc_runner.launch.py \\
        onnx_path:=/path/to/policy.onnx \\
        mode:=shadow \\
        state_source:=ros \\
        joint_state_type:=joint_msgs \\
        base_pose_topic:=/P1/pose \\
        csv_path:=/tmp/shadow.csv
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = (
        Path(get_package_share_directory("agibot_hardware_bridge"))
        / "config"
        / "agibot_bridge.yaml"
    )
    return LaunchDescription([
        Node(
            package="agibot_hardware_bridge",
            executable="agibot_hardware_bridge",
            name="agibot_hardware_bridge",
            output="screen",
            parameters=[str(config)],
        ),
    ])

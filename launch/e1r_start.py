from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_share = Path(__file__).resolve().parents[1]
    default_config = str(repo_share / "config" / "e1r_rslidar_sdk.yaml")

    config_arg = DeclareLaunchArgument(
        "config_path",
        default_value=default_config,
        description="Absolute path to the RoboSense E1R rslidar_sdk YAML config.",
    )

    rslidar_node = Node(
        namespace="rslidar_sdk",
        package="rslidar_sdk",
        executable="rslidar_sdk_node",
        name="rslidar_sdk_node",
        output="screen",
        parameters=[{"config_path": LaunchConfiguration("config_path")}],
    )

    return LaunchDescription([config_arg, rslidar_node])

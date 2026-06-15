#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RVIZ_CONFIG="${RVIZ_CONFIG:-${REPO_ROOT}/rviz/e1r_pointcloud.rviz}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-${HOME}/e1r_ros2_ws/install/setup.bash}"

echo "[E1R] Setting Jetson local display environment"
export DISPLAY=:1
export XAUTHORITY=/run/user/1002/gdm/Xauthority
export XDG_RUNTIME_DIR=/run/user/1002

echo "[E1R] Sourcing ROS2 Humble"
source /opt/ros/humble/setup.bash

if [ -f "${WORKSPACE_SETUP}" ]; then
  echo "[E1R] Sourcing workspace: ${WORKSPACE_SETUP}"
  source "${WORKSPACE_SETUP}"
else
  echo "[E1R] Workspace setup not found, continuing with system ROS only: ${WORKSPACE_SETUP}"
fi

echo "[E1R] Starting RViz2 with config: ${RVIZ_CONFIG}"
rviz2 -d "${RVIZ_CONFIG}"

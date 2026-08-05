#!/bin/bash
# Launch the full pipeline in one command.
#   ./launch_pipeline.sh          -> pipeline + rviz
#   ./launch_pipeline.sh --gazebo -> also start Gazebo
set -e
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/inoruske/surgical_twin_ws/config/cyclonedds.xml
source /opt/ros/jazzy/setup.bash
source ~/surgical_twin_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HOME/surgical_twin_ws/models

LOG=~/surgical_twin_ws/logs; mkdir -p "$LOG"
PIDS=()
cleanup() { echo; echo "shutting down..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

start() {  # start <name> <delay>
  echo "  -> $1"
  ros2 run surgical_perception "$1" > "$LOG/$1.log" 2>&1 &
  PIDS+=($!); sleep "$2"
}

if [[ "$1" == "--gazebo" ]]; then
  echo "starting Gazebo..."; gz sim -v 2 empty.sdf > "$LOG/gazebo.log" 2>&1 &
  PIDS+=($!); sleep 8
fi

echo "starting pipeline..."
start perception_node   8     # model load takes longest
start stereo_depth_node 3
start pose_estimator    2
start twin_sync_node    2
start rviz_markers      2
start video_publisher   2

echo; echo "running. logs -> $LOG"
echo "  rviz2 -d ~/surgical_twin_ws/config/surgical.rviz"
echo "Ctrl-C to stop everything."
wait

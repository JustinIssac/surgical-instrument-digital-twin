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

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA driver on kernel $(uname -r)."
  echo "Reboot and select 6.17.0-22-generic (Advanced options for Ubuntu)."
  exit 1
fi

LOG=~/surgical_twin_ws/logs; mkdir -p "$LOG"
PIDS=()
cleanup() { echo; echo "shutting down..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

LOOP=${LOOP:-true}
MODE=${MODE:-auto}   # auto | mono | stereo_pair | sbs
SRC=${SRC:-}

start() {  # start <name> <delay>
  echo "  -> $1"
  ros2 run surgical_perception "$1" > "$LOG/$1.log" 2>&1 &
  PIDS+=($!); sleep "$2"
}

if [[ "$1" == "--gazebo" ]]; then
  echo "starting Gazebo..."; gz sim -v 2 ~/surgical_twin_ws/models/surgical_scene.sdf > "$LOG/gazebo.log" 2>&1 &
  PIDS+=($!); sleep 8
fi

# reconstructed tissue surface, if built
if [ -f "$HOME/surgical_twin_ws/models/tissue/model.sdf" ] && [ "$1" == "--gazebo" ]; then
  gz service -s /world/empty/create --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean --timeout 4000 \
    --req 'sdf_filename: "'$HOME'/surgical_twin_ws/models/tissue/model.sdf", name: "tissue"' \
    > /dev/null 2>&1 && echo "  -> tissue surface"
fi

echo "starting pipeline..."
start perception_node   8     # model load takes longest
start stereo_depth_node 3
start mono_depth_node   6   # idles unless stereo is unavailable
start pose_estimator    2
start twin_sync_node    2
start rviz_markers      2
echo "  -> video_publisher (loop=$LOOP)"
VP_ARGS="-p loop:=$LOOP -p stereo_mode:=$MODE"
[ -n "$SRC" ] && VP_ARGS="$VP_ARGS -p source:=$SRC"
ros2 run surgical_perception video_publisher \
    --ros-args $VP_ARGS > "$LOG/video_publisher.log" 2>&1 &
PIDS+=($!); sleep 2

echo; echo "running. logs -> $LOG"
echo "  rviz2 -d ~/surgical_twin_ws/config/surgical.rviz"
echo "Ctrl-C to stop everything."
wait

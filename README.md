# Surgical Instrument Tracking & Digital Twin Synchronisation

> AI-driven real-time surgical instrument detection, stereo depth estimation, and Digital Twin synchronisation for Minimally Invasive Surgery.

---

## Overview

This project builds a complete software pipeline that:
1. **Detects and segments** surgical instruments in endoscopic video using YOLOv11
2. **Estimates 3D position** of each instrument using stereo camera geometry and real calibration data
3. **Estimates the jaws (tip) position** via mask PCA, validated at 2.81 mm median error against ground-truth jaw labels
4. **Filters trajectories** using a Kalman filter for smooth, physically plausible tracking
4. **Synchronises a Digital Twin** in Gazebo Sim in real time, mirroring instrument movements
5. **Tracks Economy of Motion** — cumulative path length per instrument for surgical skill assessment

The system runs entirely on ROS 2 Jazzy and achieves **95.8% mask mAP50** across 8 surgical instrument classes at **83 FPS** inference speed.

---

## Demo
```
[Surgical Video] → [YOLOv11 Detection] → [Stereo Depth] → [Kalman Filter] → [Gazebo Digital Twin]
```

**ROS 2 Node Graph:**
```
/video_publisher → /camera/image_raw → /perception_node
                                     → /stereo_depth_node ← /camera/right/image_raw
/perception_node → /instrument_detections → /stereo_depth_node
/stereo_depth_node → /instrument_detections_3d → /pose_estimator
/pose_estimator → /instrument_poses_3d → /twin_sync_node
/twin_sync_node → /instrument_poses_filtered
               → Gazebo Digital Twin (via gz.transport)
```

---

## Results

> **Evaluation protocol.** Results below are on a custom **temporal split**
> of the EndoVis 2017 *training* sequences (frames 0-160 train, 161-168
> discarded as a guard gap, 169-224 validation). This is **not** the official
> challenge test set, and the metric is **mAP50** (instance segmentation),
> not the **IoU** reported by the challenge leaderboard. Figures here are
> therefore **not directly comparable** to published EndoVis 2017 results.
>
> Two classes (`Maryland_Bipolar_Forceps`, `Grasping_Retractor_Right`) occur
> in a single contiguous block within a single sequence and cannot be
> temporally validated without leakage; they are reported as train-only.

### Segmentation Model (YOLOv11s-seg)

| Class | Instances | Mask mAP50 | Mask mAP50-95 |
|-------|-----------|-----------|---------------|
| Large_Needle_Driver_Left | 134 | 0.972 | 0.788 |
| Large_Needle_Driver_Right | 103 | 0.984 | 0.871 |
| Prograsp_Forceps_Left | 86 | 0.952 | 0.730 |
| Prograsp_Forceps_Right | 78 | 0.973 | 0.811 |
| Maryland_Bipolar_Forceps | 18 | 0.920 | 0.710 |
| Bipolar_Forceps | 114 | 0.972 | 0.803 |
| Monopolar_Curved_Scissors | 71 | 0.982 | 0.911 |
| Grasping_Retractor_Right | 23 | 0.909 | 0.804 |
| **Overall** | **627** | **0.958** | **0.803** |

### System Performance

| Metric | Value |
|--------|-------|
| Inference speed | 12.1ms per frame |
| Real-time capability | ~83 FPS |
| Model size | 20.5MB |
| Stereo depth method | SGBM (Semi-Global Block Matching) |
| Stereo baseline | 4.2773 mm (EndoVis calibration)
| Rectified focal length | 1125.55 px |
| Active image region | 1272x1016 at offset (328, 32) | |
| Kalman filter state | [x, y, z, vx, vy, vz] |

---

## System Requirements

### Hardware
- NVIDIA GPU (tested on GTX 1650 4GB — minimum recommended 8GB for Isaac Sim)
- Stereo camera or stereo surgical scope (tested with MICCAI EndoVis 2017 dataset)

### Software
- Ubuntu 24.04 LTS
- ROS 2 Jazzy Jalisco
- Gazebo Sim 8.10.0 (Gazebo Harmonic)
- Python 3.12
- CUDA 12.1+

---

## Installation

### 1. Install ROS 2 Jazzy
```bash
sudo apt install software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions
sudo apt install -y ros-jazzy-cv-bridge ros-jazzy-vision-opencv
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Gazebo Harmonic
```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable \
    $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update
sudo apt install -y gz-harmonic
```

### 3. Clone this repository
```bash
git clone git@github.com:JustinIssac/-surgical-instrument-digital-twin.git
cd surgical-instrument-digital-twin
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt --break-system-packages
pip install "numpy<2" --break-system-packages --force-reinstall
```

### 5. Build the ROS 2 workspace
```bash
mkdir -p ~/surgical_twin_ws/src
cp -r ros2_ws/src/surgical_perception ~/surgical_twin_ws/src/
cd ~/surgical_twin_ws
colcon build --packages-select surgical_perception
source install/setup.bash
echo "source ~/surgical_twin_ws/install/setup.bash" >> ~/.bashrc
```

### 6. Download the trained model weights

The model weights are not included in this repository due to file size.
Download `best.pt` from the [Releases](https://github.com/JustinIssac/-surgical-instrument-digital-twin/releases) page and place it at:
```
~/surgical_twin_ws/models/best.pt
```

### 7. Download the dataset

The MICCAI EndoVis 2017 dataset requires free registration at:
[https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org)

Download and extract into `~/surgical_twin_ws/test_data/` (left frames) and `~/surgical_twin_ws/test_data_right/` (right frames).

---

## Running the Pipeline

Start each node in a separate terminal **in this exact order**:
```bash
# Terminal 1 — Gazebo simulation
gz sim -v 4 empty.sdf

# Terminal 2 — Video publisher (streams left + right frames)
cd ~/surgical_twin_ws && source install/setup.bash
ros2 run surgical_perception video_publisher

# Terminal 3 — Perception node (YOLOv11 inference)
# Wait for: "Perception node ready, waiting for frames..."
cd ~/surgical_twin_ws && source install/setup.bash
ros2 run surgical_perception perception_node

# Terminal 4 — Stereo depth node
cd ~/surgical_twin_ws && source install/setup.bash
ros2 run surgical_perception stereo_depth_node

# Terminal 5 — Pose estimator
cd ~/surgical_twin_ws && source install/setup.bash
ros2 run surgical_perception pose_estimator

# Terminal 6 — Twin sync node (Kalman filter + Gazebo sync)
cd ~/surgical_twin_ws && source install/setup.bash
ros2 run surgical_perception twin_sync_node

# Optional Terminal 7 — Visualise node graph
rqt_graph
```

---

## Project Structure
```
surgical-instrument-digital-twin/
├── ros2_ws/
│   └── src/
│       └── surgical_perception/
│           ├── surgical_perception/
│           │   ├── perception_node.py      # YOLOv11 inference node
│           │   ├── video_publisher.py      # Stereo frame publisher
│           │   ├── stereo_depth_node.py    # SGBM stereo depth estimation
│           │   ├── pose_estimator.py       # 2D → 3D coordinate lifting
│           │   └── twin_sync_node.py       # Kalman filter + Gazebo sync
│           ├── package.xml
│           └── setup.py
├── models/
│   └── gazebo/
│       └── surgical_instrument/
│           ├── model.sdf                   # Instrument 3D model
│           └── model.config
├── data/
│   └── config/
│       └── camera_calibration.txt          # EndoVis real camera parameters
├── launch/
│   └── start_pipeline.sh                   # Quick-start reference script
├── requirements.txt
├── verify_detections.py                    # Pipeline verification script
└── README.md
```

---

## ROS 2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Left camera frames |
| `/camera/right/image_raw` | `sensor_msgs/Image` | Right camera frames |
| `/instrument_detections` | `std_msgs/String` | YOLOv11 detections (JSON) |
| `/disparity_image` | `sensor_msgs/Image` | SGBM disparity map |
| `/instrument_detections_3d` | `std_msgs/String` | Detections + stereo depth |
| `/instrument_poses_3d` | `std_msgs/String` | 3D poses in camera frame |
| `/instrument_poses_filtered` | `std_msgs/String` | Kalman-filtered poses + path lengths |
| `/annotated_frame` | `sensor_msgs/Image` | YOLOv11 annotated video |

---

## Instrument Classes

| ID | Class | Training Coverage |
|----|-------|------------------|
| 0 | Large_Needle_Driver_Left | 100% |
| 1 | Large_Needle_Driver_Right | 95.1% |
| 2 | Prograsp_Forceps_Left | 78.0% |
| 3 | Prograsp_Forceps_Right | 88.7% |
| 4 | Maryland_Bipolar_Forceps | 33.3% |
| 5 | Bipolar_Forceps | 99.8% |
| 6 | Monopolar_Curved_Scissors | 56.0% |
| 7 | Grasping_Retractor_Right | 10.2% |

---

## Known Limitations

- **Hardware:** GTX 1650 (4GB VRAM) limits simultaneous Gazebo + inference speed. Recommended 8GB+ GPU for production.
- **Stereo depth:** Highly reflective instrument surfaces reduce stereo matching reliability. System gracefully falls back to assumed depth.
- **Instrument classes:** Only the 8 EndoVis 2017 classes are supported. New instruments require retraining (~3-4 hours).
- **Danger zones:** Keep-Out Zones require manual pre-operative definition by a clinical expert. Automatic anatomy segmentation is future work.
- **Simulator:** Currently using Gazebo Harmonic. Migration to NVIDIA Isaac Sim pending GPU server access.

---

## Pending Work

- [ ] Quantitative evaluation of 3D pose estimation (reprojection error)
- [ ] Robustness testing under occlusion and lighting variation
- [ ] Keep-Out Zone safety feature in Gazebo
- [ ] AR screen overlay visualiser
- [ ] NVIDIA Isaac Sim migration (pending)

---

## Dataset

This project uses the **MICCAI EndoVis 2017 Robotic Instrument Segmentation** dataset.

- 8 surgical sequences, 225 frames each (1,800 total)
- 1920×1080 resolution, part-level segmentation masks (10=shaft, 20=wrist, 30=jaws)
- Stereo camera with calibration parameters
- License: Creative Commons Non-Commercial

> Please register at the Grand Challenge platform to access the dataset.

---

## Citation

If you use this work, please cite the EndoVis 2017 dataset:
```bibtex
@inproceedings{allan2019,
  title={2017 robotic instrument segmentation challenge},
  author={Allan, Max and others},
  journal={arXiv preprint arXiv:1902.06426},
  year={2019}
}
```

---

## Author

**Justin Varghese Issac**
Masters in Intelligent Robotics — University of Galway
Supervisor: Liam Kilmartin

---

## License

This project is licensed under the MIT License.
The MICCAI EndoVis 2017 dataset is licensed under Creative Commons Non-Commercial — see dataset terms for usage restrictions.

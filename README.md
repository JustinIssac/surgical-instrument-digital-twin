source ~/.bashrc
cd ~/surgical_twin_ws
./launch_pipeline.sh --gazebo



rviz2 -d ~/surgical_twin_ws/config/surgical.rviz

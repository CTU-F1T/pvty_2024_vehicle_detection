PVTY detekce
# LIDAR startup
- catkin clean
- catkin build
- source devel/setup.bash
- roslaunch racecar teleop.launch

# links
- camera ros https://github.com/IntelRealSense/realsense-ros
- camera datasheet: https://dev.intelrealsense.com/docs/lidar-camera-l515-datasheet

# Setting up the environment and running the script:
  - **(Windows specific!)** Download WSL - Ubuntu 20.04 LTS 
  - Install ros noetic by following this tutorial: https://foxglove.dev/blog/installing-ros1-noetic-on-ubuntu
  - Clone the repository 
    - `git clone https://github.com/CTU-F1T/pvty_2024_vehicle_detection.git`
  - Download the rosbag from the Gdisk folder: https://drive.google.com/drive/u/1/folders/1NrswYhbbgiJ4l5COxxrjti3AzJohsMWV
  - Create `/src/data` folder in the git repo and save the rosbag there
  - go to `/src` folder
  - Create python .venv and install the requirements.txt 
    - `pip install -r requirements.txt`
    - **you might have to link the ros from your Linux distro to the .venv using this command:
      - `ln -s /opt/ros/noetic/lib/python3/dist-packages/* .venv/lib/python3.8/site-packages/`
  - Then do these three steps in three separated terminals: 
    - 1. `roscore` 
    - 2. `rosbag play src/data/01_test_rec.bag`
    - 3. `python3 camera_test_sub.py`
  

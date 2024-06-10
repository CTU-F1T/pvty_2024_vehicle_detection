# F1tenth Opponent Vehicle Detection
Welcome to the F1tenth Opponent Vehicle Detection project wiki. This project was undertaken in the summer term of 2024 at CTU Prague for the project PVTY by a dedicated team to develop and implement methods for detecting opponent vehicles in an autonomous racing platform.

### LIDAR Startup

1. **Clean and Build the Catkin Workspace**:
    ```bash
    catkin clean
    catkin build
    ```

2. **Source the Setup File**:
    ```bash
    source devel/setup.bash
    ```

3. **Launch the Teleoperation Node**:
    ```bash
    roslaunch racecar teleop.launch
    ```

4. **Launch roscore**:
    ```bash
    roscore
    ```

5. **Launch the camera from the scripts folder**:
    ```
    ./scripts/start_camera
    ```

6. **Launch main.py in the /src/detection folder**:
    - there could be a possibility that the main.py is not working properly, if that happens please use the `april_tag_demo.py` in the /src/detection/demos folder
    ```bash
    python3 main.py
    ```

### Additional Resources

- **Camera ROS Package**: [Intel RealSense ROS](https://github.com/IntelRealSense/realsense-ros)
- **Camera Datasheet**: [LIDAR Camera L515 Datasheet](https://dev.intelrealsense.com/docs/lidar-camera-l515-datasheet)

### Running the Script

Before running the script, set the following environment variable:
```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```  

# For Full Information 
Please refer to the [Wiki](https://github.com/CTU-F1T/pvty_2024_vehicle_detection/wiki) page for full information about the project, including detection methods, experimental results, and future work.
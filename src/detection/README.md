# Setting Up the Environment and Running the Script on local machine utilizing pre-recorded rosbags 

## Prerequisites

### For Windows Users
- **Download and Install WSL**: Install Ubuntu 20.04 LTS via Windows Subsystem for Linux (WSL).
- **Install ROS Noetic**: Follow the tutorial to install ROS Noetic on Ubuntu 20.04 LTS [here](https://foxglove.dev/blog/installing-ros1-noetic-on-ubuntu).

### For macOS and Linux Users
- **Install ROS Noetic**: Follow the official ROS installation guide for your operating system. For Ubuntu, you can follow the tutorial [here](https://foxglove.dev/blog/installing-ros1-noetic-on-ubuntu).

## Setting Up the Environment

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/CTU-F1T/pvty_2024_vehicle_detection.git
    ```

2. **Download ROS Bags**: Download the necessary ROS bags from the Gdisk folder [here](https://drive.google.com/drive/folders/1DuLVWwxOwGOYZqpi_2dEGsDMjf_Gl4vp) and place them in a directory named `src/data` within your project directory.

3. **Navigate to the `/src` Folder**:
    ```bash
    cd pvty_2024_vehicle_detection/src/detection
    ```

4. **Create a Python Virtual Environment and Install Dependencies**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

5. **Link ROS Python Packages to the Virtual Environment** (if necessary):
    ```bash
    ln -s /opt/ros/noetic/lib/python3/dist-packages/* .venv/lib/python3.8/site-packages/
    ```

## Running the Script

Follow these steps in three separate terminals:

1. **Start ROS Master**:
    ```bash
    roscore
    ```

2. **Play the ROS Bag**:
    ```bash
    rosbag play -l src/data/angle.bag
    ```

3. **Activate the Virtual Environment and Run the Python Script**:
    ```bash
    source .venv/bin/activate
    python3 main.py
    ```

## Visualizing Data in RViz
The script is publishing the following topics 
- **/detection/marker**: Publishes markers for visualizing detections.
- **/gradient/merged_grad**: Publishes merged gradient images.
- **/gradient/image_grad**: Publishes gradient images.
- **/gradient/depth_grad**: Publishes depth gradient images.
- **/detection/focus**: Publishes focus area images.
- **/detection/april_tags**: Publishes AprilTag detection images.
- **/detection/at_marker**: Publishes markers for AprilTag detections.
- **/detection/pose_estimate**: Publishes pose estimations of detected objects.

To visualize the published topics in RViz, follow these steps:

1. **Start RViz**:
    ```bash
    rviz
    ```

2. **Add the Relevant Topics**:
    - Click the "Add" button in the bottom left.
    - Select the appropriate type (e.g., Marker, Image, Pose) based on the topic.
    - Add the following topics:
      - `/detection/marker` (Marker)
      - `/gradient/merged_grad` (Image)
      - `/gradient/image_grad` (Image)
      - `/gradient/depth_grad` (Image)
      - `/detection/focus` (Image)
      - `/detection/april_tags` (Image)
      - `/detection/at_marker` (Marker)
      - `/detection/pose_estimate` (Pose)

3. **Customize the RViz Layout**:
    - Arrange the display panels in RViz to suit your workflow.
    - Use the "Views" panel to save different layout configurations.

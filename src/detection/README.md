# Setting Up the Environment and Running the Script

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

2. **Download ROS Bags**: Download the necessary ROS bags from the Gdisk folder [here](https://drive.google.com/drive/u/1/folders/1NrswYhbbgiJ4l5COxxrjti3AzJohsMWV) and place them in a directory named `src/data` within your project directory.

3. **Navigate to the `/src` Folder**:
    ```bash
    cd pvty_2024_vehicle_detection/src
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
    rosbag play -l src/data/01_test_rec.bag
    ```

3. **Activate the Virtual Environment and Run the Python Script**:
    ```bash
    source .venv/bin/activate
    python3 camera_test_sub.py
    ```

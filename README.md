# Adaptive Controller for Jackal Robot Navigation

## Description

This project aims to improve the accuracy of autonomous mobile robot navigation by creating an adaptive controller that reduces the discrepancies between expected and actual actions. The project uses reinforcement learning to add a correction term to the original planning twist command. The Jackal robot is used in this project, and the framework is based on the Benchmark Autonomous Robot Navigation (BARN) Challenge.

## Contents
- [Background](#background)
- [Research Purpose](#research-purpose)
- [Methodology](#methodology)
- [System Setup](#system-setup)
- [Training Process](#training-process)
- [Training Results](#training-results)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Background

- The navigation of mobile robots relies on precision in following planned trajectories to ensure accuracy. However, discrepancies in control outputs can lead to significant deviations from planned paths.
- The project employs the Jackal robot, equipped with 2D LiDAR, to navigate both pre-generated and novel environments from the BARN dataset.
- Various navigation strategies are employed, including Dynamic Window Approach (DWA), Fast DWA, and reinforcement learning-based methods like APPLR.

## Research Purpose

- **Main Goal:** Develop an adaptive controller to enhance trajectory accuracy by separating planning and control processes.
- **Strategy:** Train a neural network using reinforcement learning to add a correction term to the original planning twist command, reducing the deviation between expected and actual actions.
- **Expected Outcome:** Improved alignment between planned and actual vehicle paths.

## Methodology

- **Inputs:** Velocity commands, LiDAR data, and target waypoints.
- **Outputs:** Adjusted velocities to correct deviations.
- **Neural Network:** The correction network consists of fully connected layers, with an input layer, two hidden layers (64 neurons each), and an output layer. The ReLU activation function is used in the hidden layers.
- **Reinforcement Learning Framework:** The TD3 algorithm is used for training.

## System Setup

- **Simulation Environment:** Utilizes ROS and Gazebo to simulate real-world robotics scenarios. A custom script based on the BARN Challenge is used for testing.
- **ROS Communication:** ROS publishers and subscribers handle movement commands and sensor data integration.
- **Integration with OpenAI Gym:** The simulation environment is compatible with OpenAI Gym for training and testing the adaptive controller.

## Training Process

- **Feedback Loop:** Robot actions are adjusted in real-time based on sensor feedback and neural network predictions.
- **Adaptive Learning:** The system adapts its strategy based on accumulated experience, improving navigation performance over time.
- **Reward Design:** The reward function penalizes large deviations and rewards improvements in local progress toward the goal.

## Training Results

| Method              | Success Rate | Average Navigation Time (seconds) |
|---------------------|--------------|-----------------------------------|
| TEB                 | 70.98%       | 88.64                            |
| TEB with Controller | 73.45%       | 85.03                            |
| DWA                 | 72%          | 76.64                            |
| DWA with Controller | 89.47%       | 50.92                            |

The adaptive controller demonstrated significant improvements in navigation performance, particularly under the DWA algorithm.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zcccc-keven/adaptive-controller-for-jackal.git
    ```

2. Install any additional dependencies and make:
   ```bash
   pip install -r requirements.txt
   catkin_make
   source devel/setup.bash
    ```
## Usage

1. Launch the simulation:
   ```bash
   git clone https://github.com/zcccc-keven/adaptive-controller-for-jackal.git
    ```
   
2. Train the controller using reinforcement learning:
   ```bash
   cd /TD3
   python train_td3.py --env CustomEnv2-v0 --policy TD3 --seed 0 --start_timesteps 10000 --eval_freq 5000 --max_timesteps 1000000 --save_model
    ```
   
3. Test the trained controller:
    ```bash
   git clone https://github.com/zcccc-keven/adaptive-controller-for-jackal.git
    ```   

## Acknowledgements

This project builds on several existing frameworks and repositories:

    BARN Challenge: The project heavily relies on the framework and setup provided by the BARN Challenge repository, which focuses on testing and improving autonomous navigation systems.
    TD3 (Twin Delayed Deep Deterministic Policy Gradient): The TD3 algorithm used for reinforcement learning in this project is based on the implementation from [sfujim/TD3](https://github.com/sfujim/TD3). The TD3 framework provides the reinforcement learning backbone for training the adaptive controller in the simulation environment.

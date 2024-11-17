# Adaptive Controller for Jackal Robot Navigation

## Description

This project aims to improve the accuracy of autonomous mobile robot navigation by creating an adaptive controller that reduces the discrepancies between expected and actual actions. The project uses reinforcement learning to add a correction term to the original planning twist command. The Jackal robot is used in this project, and the framework is based on the Benchmark Autonomous Robot Navigation (BARN) Challenge.

## Purpose

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

## Navigation Simulation

- Start ![[start.jpg]]
- In process
- Goal reached

## Training Results

| Method              | Success Rate | Average Navigation Time (seconds) |
|---------------------|--------------|-----------------------------------|
| TEB                 | 70.98%       | 88.64                            |
| TEB with Controller | 73.45%       | 85.03                            |
| DWA                 | 72%          | 76.64                            |
| DWA with Controller | 89.47%       | 50.92                            |

The adaptive controller demonstrated significant improvements in navigation performance, particularly under the DWA algorithm.

## Installation

1. Follow the instruction at:
   ```bash
   https://github.com/Daffan/ros_jackal/tree/main
    ```
2. Clone the repository:
   ```bash
   git clone https://github.com/zzczjh9/adaptive-controller-for-jackal.git
    ```

3. Install any additional dependencies and make:
   ```bash
   pip install -r requirements.txt
   catkin_make
   source devel/setup.bash
    ```
   
## Usage

1. Launch the simulation:
   ```bash
   cd ~/jackal_ws/src/the-barn-challenge
   source ../../devel/setup.bash
   python3 run_test62.py --laser
    ```
   
2. Train the controller using reinforcement learning:
   ```bash
   source venv/bin/activate
   source ~/jackal_ws1/devel/setup.bash
   cd ~/jackal_ws1/src/the-barn-challenge/TD3
   python train_td3.py --env CustomEnv2-v0 --policy TD3 --seed 0 --start_timesteps 10000 --eval_freq 5000 --max_timesteps 1000000 --save_model
    ```
   
3. Test the trained controller:
   ```bash
   python3 model_test.py
    ```
## Acknowledgements

This project builds on several existing frameworks and repositories:

    BARN Challenge: The project heavily relies on the framework and setup provided by the BARN Challenge repository, which focuses on testing and improving autonomous navigation systems.
    TD3 (Twin Delayed Deep Deterministic Policy Gradient): The TD3 algorithm used for reinforcement learning in this project is based on the implementation from [sfujim/TD3](https://github.com/sfujim/TD3). The TD3 framework provides the reinforcement learning backbone for training the adaptive controller in the simulation environment.

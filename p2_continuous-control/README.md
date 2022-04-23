# Project2 Continuous Control

## Task Reacher
This task tries to keep a double-jointed arm in the goal location. If the arm is in the goal, a reward +0.1 is provided for each step.

The observation space is 33 dimensions which includes position, rotation, velocity, and angular velocities of the arm. The agent is required to select a four-dimensional action, of which each coordinate is a number between -1 and 1.

## Dependencies
### 1. Setup the Python Environment According to README.md at the Parent Directory
### 2. Download the Unity Environment given by Udacity
Download the Unity environment depending on your operating system.
- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip

Unzip the file and place it in this directory.

## How to Run
```
python train.py
```
You can change the settings by modifying the settings.py.
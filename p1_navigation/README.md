# Project1 Navigation

## Task Banana
This task tries to get yellow bananas, while avoiding black bananas. If the agent collects a yellow banana, the agent gets a reward +1. If the agent collects a black banana, the agent gets a reward -1.

The state space is 37 dimensions which includes the agent's velocity. The agent has four actions, which are moving forward, moving backward, turning left and turning right.

## Dependencies
### 1. Setup the Python Environment According to README.md at the Parent Directory
### 2. Download the Unity Environment given by Udacity
Download the Unity environment depending on your operating system.
- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
Unzip the file and place it in this directory.

## How to Run
```
python train.py
```
You can change the settings by modifying the settings.py.
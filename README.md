# LunarLander

## Overview
This repository contains an implementation of the policy gradient method to solve the LunarLander-v2 task from OpenAI's Gym. This project demonstrates how reinforcement learning algorithms can train an agent to safely land a spacecraft on the lunar surface.

## Implementation Details
- **Environment**: The `LunarLander-v2` environment simulates a landing pad and requires the agent to make a soft landing using discrete actions controlling the thrusters.
- **Algorithm**: We use a policy gradient technique, specifically the REINFORCE algorithm, to optimize the agent's decisions. The model learns to choose actions that maximize cumulative rewards over each episode.
- **Network Architecture**: The agent's decision-making policy is determined by a neural network with two hidden layers, each having 64 neurons and using the ReLU activation function. The output layer utilizes a softmax function to provide a probability distribution over possible actions.

## Getting Started

### Prerequisites
To run this project, you will need:
- Python 3.8 or newer
- OpenAI Gym
- NumPy
- PyTorch

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/Pangao11/LunarLander.git
2. Running the Program
    ```bash
    python Lunar.py

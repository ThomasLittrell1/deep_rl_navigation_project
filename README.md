# Navigation Project

## Project Details

This repo contains the code to solve the Navigation Project, which is part of the Udacity Deep Reinforcement Learning nanodegree. The goal of the project is to teach a robot to navigate around an environment and collect yellow bananas while avoiding the blue bananas. The details of the MDP are:

- States: The states are 37-dimensional continuous vectors simulating sensor input to the robot about its movement and environment
- Actions: There are 4 discrete actions (moving left, right, forward, and back)
- Rewards: The agent recieves a reward of +1 for every yellow banana that it collects and a reward of -1 for every blue banana

The goal is to have the agent get an average score of +13 for at least 100 consecutive rounds.

## Getting Started

You will need three things to run the project:
1. The appropriate python packages
2. The `Banana.app` Unity environment supplied by Udacity
3. The additional `python` code provided by Udacity for running the Unity environment

### Installing the python packages

Create a new conda environment (or virtual environment of your choice) and activate it. Install the dependencies using `python -m pip install -r requirements.txt`

### Installing Udacity dependencies

Udacity provides specific helper code to run the project and environment. I have not included their `Banana.app` file or their `python` folder with additional code for the environemnt in this repository to protect their IP. Udacity provides specific instructions for installing the unity environment. You can download the `python` folder of additional dependencies from your Udacity workspace and install it using `python -m pip -q install python` or by running `pip -q install ./python` in the Navigation.ipynb notebook.

## Instructions

To train an agent, create a kernel from your conda environment and run all cells in the Navigation.ipynb Jupyter notebook. If the agent completes the task successfully, the saved weights will be put into a checkpoint.pth file.

All code for the agent, the Q-network, and the training code is kept in the src directory.

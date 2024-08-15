# SnakeGameAI

This project features a Snake game controlled by an AI agent using reinforcement learning, specifically Q-learning. Below is an overview of the files involved in this project, please read those. To run the Game with the agent, please execute the file `AgentModel.py`.

## Files Overview

1. **Environment.py**  
   This file contains the code for the game environment where the Snake operates. It defines the rules and logic of the game.

2. **AgentModel.py**  
   This file contains the code for the AI agent, including the states, actions, and Q-table. The training phase of the agent is also implemented here. To run the Snake game, execute this file.

3. **Evaluation_Trained_Data.py**  
   This is a modified version of `AgentModel.py`, where the agent no longer trains but instead uses a pre-trained Q-table with a greedy policy. You need to import a trained Q-table in this file.

4. **Live_plotter.py**  
   This script plots the training graph during execution and updates it after each game, allowing you to track game scores in real time.

5. **Result_plotter.py**  
   This script allows you to visualize data such as scores and mean scores, which are obtained from `AgentModel.py`. Various agents are compared for analysis.

6. **Qtable_from_results.npy**  
   This file contains the Q-table after 500 runs without using a seed (policy = greedy, discount factor = 0.9, learning rate = 0.3).

7. **Qtable_seed24_from_results.npy**  
   Similar to the previous file, but with the use of seed 24.

8. **Font.ttf**  
   This is the font used for the PyGame display.

## Configuration

- **Speed**  
  To change the game speed, modify line 28 of the `Environment.py` file. During analysis, the speed was set to a very high value.

- **Random Seed Setting**  
  The random seed setting is commented out by default. To enable it, check line 7 of the `Environment.py` file.

- **Reinforcement Learning Variables**  
  You can adjust the RL variables between lines 8 and 13 in the `AgentModel.py` file.

- **Policy**  
  The default policy is greedy. To use the epsilon-greedy policy, comment out lines 91 to 96 and uncomment lines 74 to 88 in `AgentModel.py`.

- **Saving Data**  
  To save data and plots, check lines 151 to 155 of the `AgentModel.py` file.

- **Evaluation of Trained Data**  
  The Q-table is loaded at line 16 in the `Evaluation_Trained_Data.py` file.

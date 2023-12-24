# Implementation of the paper "Decentralized Counterfactual Value with Threat Detection in Multi-Agent Mixed Cooperative and Competitive Environments"

This is the code for the paper "Decentralized Counterfactual Value with Threat Detection in Multi-Agent Mixed Cooperative and Competitive Environments".

## Environment
1. Grid Examples: The environment contains two basic grid environment (2-room and 4-room environments), which are implemented in the `GRID/ENV` file.
2. Classical Scenarios: Job Scheduling, Matthew Effect and Manufacturing Plant. All scenarios have limited resources, thus, the agents encounter the mixed cooperative and competitive relationship with others under the general-sum rewards.
3. SSD Environments: The environment is a 2D grid game with the partially observable state with the picture of $15 \times 15 \times 3$. The action space is a discrete space that includes $7$ basic motions: move up, move down, move left, move right, stay, rotate clockwise and rotate counterclockwise. Each agent intends to collect more apples in the map and each apple responds with a $+1$ reward. 
4. [MAgent](https://github.com/geek-ai/MAgent): MAgent is a research platform for many-agent reinforcement learning. Unlike previous research platforms that focus on reinforcement learning research with a single agent or only few agents, MAgent aims at supporting reinforcement learning research that scales up from hundreds to millions of agents.

## Quick start
Please follow the instruction of 'README.md' file in different environments to install Python requirements.

# Implementation of the paper "Decentralized Counterfactual Value with Threat Detection for Multi-Agent Reinforcement Learning in Mixed Cooperative and Competitive Environments"

This is the code for the paper "Decentralized Counterfactual Value with Threat Detection in Multi-Agent Mixed Cooperative and Competitive Environments".

## Environment
1. Grid Examples: The environment contains two basic grid environment (2-room and 4-room environments), which are implemented in the `GRID/ENV` file.
2. Classical Scenarios: Job Scheduling, Matthew Effect and Manufacturing Plant. All scenarios have limited resources, thus, the agents encounter the mixed cooperative and competitive relationship with others under the general-sum rewards.
3. SSD Environments: The environment is a 2D grid game with the partially observable state with the picture of $15 \times 15 \times 3$. The action space is a discrete space that includes $7$ basic motions: move up, move down, move left, move right, stay, rotate clockwise and rotate counterclockwise. Each agent intends to collect more apples in the map and each apple responds with a $+1$ reward. 
4. [MAgent](https://github.com/geek-ai/MAgent): MAgent is a research platform for many-agent reinforcement learning. Unlike previous research platforms that focus on reinforcement learning research with a single agent or only few agents, MAgent aims at supporting reinforcement learning research that scales up from hundreds to millions of agents.

## Quick start
Please follow the instruction of 'README.md' file in different environments to install Python requirements.

## Cite our paper
```
@article{DCVTD,
  author       = {Shaokang Dong and
                  Chao Li and
                  Shangdong Yang and
                  Wenbin Li and
                  Yang Gao},
  title        = {Decentralized Counterfactual Value with Threat Detection for Multi-Agent
                  Reinforcement Learning in mixed cooperative and competitive environments},
  journal      = {Expert Syst. Appl.},
  volume       = {257},
  pages        = {125116},
  year         = {2024},
  url          = {https://doi.org/10.1016/j.eswa.2024.125116},
  doi          = {10.1016/J.ESWA.2024.125116}
}
```

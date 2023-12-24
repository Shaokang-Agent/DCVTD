# WToE_Battle_Game

### Environment
The environment consists of two teams, namely the red team and the blue team, each comprising $64$ agents. The objective of each team is to eliminate all opposing agents. Agents have the ability to perform actions such as movement and attacking nearby enemies. The associated rewards for these actions are as follows: -0.005 for a movement action, 0.2 for attacking an enemy, 5 for eliminating an enemy, -0.1 for attacking an empty space, and -0.1 for being attacked or eliminated.

### Requirement
```shell
$ conda create -n Exp4 python=3.6.13
$ conda activate Exp4
$ pip install -r requirements.txt
```

### Multi-agent training 
IAC: 
```shell
$ python train_battle.py --algo ac
```
IAC with DCVTD: 
```shell
$ python train_battle.py --algo ac_dcvtd
```
IQL: 
```shell
$ python train_battle.py --algo il
```
IQL with DCVTD: 
```shell
$ python train_battle.py --algo il_dcvtd
```
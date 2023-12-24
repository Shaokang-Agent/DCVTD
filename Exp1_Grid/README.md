# Exp1_GRID

### Environment
The environment contains two basic grid environment (2-rooms and 4-rooms environment), which are implemented in the `ENV` file.

### Requirement
$ conda create -n Exp1 python=3.7.9
$ conda activate Exp1
$ pip install -r requirements.txt

### Algorithm
We impletent the DCVTD, DCV, SOM, EITI, Maximum Entropy Q-learning, \epislon-greedy algorithm.
Training usage: ```python Two_rooms/obstacle_dcvtd.py``` or ```python Four_rooms/DCVTD.py```
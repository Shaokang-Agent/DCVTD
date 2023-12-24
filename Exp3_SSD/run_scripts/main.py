import numpy as np
import pandas as pd
import argparse
import sys
sys.path.append('./')
import torch
from run_scripts.runner import Runner


parser = argparse.ArgumentParser()

parser.add_argument("--algorithm", type=str, default="DQN")
parser.add_argument("--env", type=str, default="Cleanup")
parser.add_argument("--num_agents", type=int, default=5)
parser.add_argument("--num_episodes", type=int, default=5000)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--evaluate_cycle", type=int, default=10)
parser.add_argument("--train_steps", type=int, default=1)
parser.add_argument("--evaluate_epi", type=int, default=1)
parser.add_argument("--buffer_size", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--round", type=int, default=1)
parser.add_argument("--control_num_agents", type=int, default=1)
parser.add_argument("--save", type=bool, default=False)
parser.add_argument("--load", type=bool, default=False)

args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
if args.algorithm == "DQN":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_episode = 3000
    args.epsilon_final = 0.99
    args.target_update_iter = 100
    args.double_dqn = False
    args.grad_norm_clip = 5

elif args.algorithm == "DCVTD":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_episode = 3000
    args.epsilon_final = 0.99
    args.target_update_iter = 100
    args.double_dqn = True #False
    args.grad_norm_clip = 5
    args.hidden_dim = 32
    args.heads = 4
    args.dim = 4
    if args.env == "Cleanup":
        args.env_alpha_initial = 0.7
        args.env_alpha_final = 0.9
        args.internal_scale = 0.25
    else:
        args.env_alpha_initial = 0.6
        args.env_alpha_final = 0.9
        args.internal_scale = 0.25

elif args.algorithm == "SOCIAL":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_episode = 3000
    args.epsilon_final = 0.99
    args.target_update_iter = 100
    args.double_dqn = True
    args.grad_norm_clip = 10
    args.r_in_scale = 0.005
    args.env_alpha_initial = 0.7
    args.env_alpha_final = 0.9
    if args.num_agents == 5:
        if args.env == "Harvest":
            args.r_in_scale = 0.005
            args.env_alpha_initial = 0.8
            args.env_alpha_final = 0.99
        else:
            args.r_in_scale = 0.005
            args.env_alpha_initial = 0.8
            args.env_alpha_final = 0.95

elif args.algorithm == "MADDPG":
    args.actor_lr = 5e-4
    args.critic_lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_episode = 3000
    args.epsilon_final = 0.99
    args.tau = 0.99
    args.replace_param = 100
    args.grad_norm_clip = 5

elif args.algorithm == "QMIX":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_episode = 3000
    args.epsilon_final = 0.99
    args.two_hyper_layers = False
    args.qmix_hidden_dim = 32
    args.hyper_hidden_dim = 32
    args.tau = 0.99
    args.replace_param = 100
    args.grad_norm_clip = 10

else:
    args.actor_lr = 5e-4
    args.critic_lr = 5e-4
    args.clip_param = 0.05
    args.epsilon_init = 0.5
    args.epsilon_final = 0.99
    args.epsilon_episode = 3000


print("Environment: {}, agent-number: {}, algorithm: {}, cuda: {}".format(args.env, args.num_agents , args.algorithm, args.cuda))
for i in range(args.round):
    runner = Runner(args)
    runner.run(i+1)
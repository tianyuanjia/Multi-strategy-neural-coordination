# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:20:56 2021

@author: 98703
"""
from environment import MazeEnv
import numpy as np
import torch
from next_model import Model2D
from algorithm import NEXT_plan, RRTS_plan
from config import set_random_seed
from utils.plot2 import plot_tree



def visual(seed, env, index, model, t_max=1000):
    set_random_seed(seed)

    UCB_type = 'kde'
    # Sample a problem from the environment
    pb_idx = index # 0 - 2999
    pb = env.init_new_problem(pb_idx)
    model.set_problem(pb)

    # search_tree, done, n_sample = NEXT_plan(
    #     env=env,
    #     model=model,
    #     T=t_max,
    #     g_explore_eps=0.1,
    #     stop_when_success=True,
    #     UCB_type=UCB_type
    # )
    # # 画图
    # plot_tree(
    #     states=search_tree.states,
    #     parents=search_tree.parents,
    #     problem=env.get_problem()
    # )



    search_tree, done_rrts,n_sample = RRTS_plan(
        env=env,
        T=1000,
        stop_when_success=True
    )
    plot_tree(
        states = search_tree.states,
        parents = search_tree.parents,
        problem = env.get_problem()
    )


if __name__ == '__main__':
    seed = 1234
    set_random_seed(seed)  # 每次运行，生成指定随机数相同
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False
    dim = 2
    env = MazeEnv(dim=2, map_file ='maze_files/mazes_hard.npz')
    model = Model2D(env=env, cuda = cuda, dim = dim)
    model_file = './my_trained_models/weight_2d.pt'
    model.net.load_state_dict(torch.load(model_file, map_location=device))
    pb = 1
    visual(seed=seed, env=env, index=pb, model=model, t_max=1000)

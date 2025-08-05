import os
import time
import json

import torch

from cfg_communicate import get_cfg
from environment.env import *
from agent.network import *
from agent.heuristics import *

if __name__=="__main__":
    cfg = get_cfg()

    model_path = cfg.model_path
    param_path = cfg.param_path

    # data_dir = ["./input/test/v2/25-60/",
    #             "./input/test/v2/25-70/",
    #             "./input/test/v2/25-80/",
    #             "./input/test/v2/25-90/",
    #             "./input/test/v2/25-100/"]
    # res_dir = ["./output/test/v2/25-60/",     # 아마도 Result의 약자인 듯
    #            "./output/test/v2/25-70/",
    #            "./output/test/v2/25-80/",
    #            "./output/test/v2/25-90/",
    #            "./output/test/v2/25-100/"]
    #

    use_gnn = bool(cfg.use_gnn)
    use_added_info = bool(cfg.use_added_info)
    encoding = cfg.encoding
    restriction = bool(cfg.restriction)
    algorithm = cfg.algorithm
    random_seed = cfg.random_seed

    sequencing = ["SPT", "MOR", "MWKR"]
    assignment = ["MF", "LU", "HP"]
    PDR = []
    for i in sequencing:
        for j in assignment:
            PDR.append(i + "-" + j)

    data_dir = ""  # TODO: Unity로 입력받도록 코드 구성
    res_dir = ""  # TODO: Unity로 전송되도록 코드 구성

    test_paths = os.listdir(data_dir)
    index = ["P%d" % i for i in range(1, len(test_paths) + 1)] + ["avg"]
    columns = ["RL"] + PDR
    df_delay = pd.DataFrame(index=index, columns=columns)
    df_move = pd.DataFrame(index=index, columns=columns)
    df_priority = pd.DataFrame(index=index, columns=columns)
    df_delay_cost = pd.DataFrame(index=index, columns=columns)
    df_move_cost = pd.DataFrame(index=index, columns=columns)
    df_loss_cost = pd.DataFrame(index=index, columns=columns)
    df_computing_time = pd.DataFrame(index=index, columns=columns)

    for name in columns:
        progress = 0
        list_delay = []
        list_move = []
        list_priority = []
        list_delay_cost = []
        list_move_cost = []
        list_loss_cost = []
        list_computing_time = []

        for prob, path in zip(index, test_paths):
            random.seed(random_seed)

            delay = 0.0
            move = 0.0
            priority_ratio = 0.0
            delay_cost = 0.0
            move_cost = 0.0
            loss_cost = 0.0
            computing_time = 0.0

            env = QuayScheduling(data_dir + path, algorithm=name,
                                 state_encoding=encoding, restriction=restriction,
                                 record_events=False, device=torch.device('cpu'))

            embed_dim = cfg.embed_dim
            num_heads = cfg.num_heads
            num_HGT_layers = cfg.num_HGT_layers
            num_actor_layers = cfg.num_actor_layers
            num_critic_layers = cfg.num_critic_layers

            agent = Scheduler(env.meta_data, env.state_size, env.num_nodes,
                              int(embed_dim),
                              int(num_heads),
                              int(num_HGT_layers),
                              int(num_actor_layers),
                              int(num_critic_layers),
                              use_gnn=use_gnn, use_added_info=use_added_info).to(torch.device('cpu'))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            agent.load_state_dict(checkpoint['model_state_dict'])

            start = time.time()
            state, mask, current_ops, added_info = env.reset()
            done = False

            while not done:
                action, _, _ = agent.act(state, mask, current_ops, added_info, greedy=False)

                next_state, reward, done, next_mask, next_current_ops, next_added_info = env.step(action)

                state = next_state
                mask = next_mask
                current_ops = next_current_ops
                added_info = next_added_info

                if done:
                    finish = time.time()
                    delay = sum(env.monitor.delay.values()) / len(env.monitor.delay.values())
                    move = sum(env.monitor.move.values()) / len(env.monitor.move.values())
                    priority_ratio = sum(env.monitor.priority_ratio.values()) / len(env.monitor.priority_ratio.values())
                    delay_cost = 4000 * sum(env.monitor.delay.values())
                    move_cost = 4000 * sum(env.monitor.move.values())
                    loss_cost = 12 * sum(env.monitor.loss.values())
                    computing_time = finish - start
                    break
            list_delay.append(delay)
            list_move.append(move)
            list_priority.append(priority_ratio)
            list_delay_cost.append(delay_cost)
            list_move_cost.append(move_cost)
            list_loss_cost.append(loss_cost)
            list_computing_time.append(computing_time)

            progress += 1

        df_delay[name] = list_delay + [sum(list_delay) / len(list_delay)]
        df_move[name] = list_move + [sum(list_move) / len(list_move)]
        df_priority[name] = list_priority + [sum(list_priority) / len(list_priority)]
        df_delay_cost[name] = list_delay_cost + [sum(list_delay_cost) / len(list_delay_cost)]
        df_move_cost[name] = list_move_cost + [sum(list_move_cost) / len(list_move_cost)]
        df_loss_cost[name] = list_loss_cost + [sum(list_loss_cost) / len(list_loss_cost)]
        df_computing_time[name] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
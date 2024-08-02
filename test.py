import os
import time
import json

import torch
# import vessl

from environment.env import *
from agent.network import *
from agent.heuristics import *
from cfg_test import *

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    cfg = get_cfg()

    model_path = cfg.model_path
    param_path = cfg.param_path
    # data_dir = [cfg.data_dir]
    # res_dir = [cfg.res_dir]
    # sim_dir = cfg.sim_dir

    # data_dir = ["./input/test/28-100/", "./input/test/28-80/", "./input/test/28-60/",
    #             "./input/test/40-120/", "./input/test/40-100/", "./input/test/40-80/",
    #             "./input/test/35-100/", "./input/test/35-80/", "./input/test/35-60/",
    #             "./input/test/25-100/", "./input/test/25-80/", "./input/test/25-60/",
    #             "./input/test/20-80/", "./input/test/20-60/", "./input/test/20-40/"]
    # res_dir = ["./output/test/28-100/", "./output/test/28-80/", "./output/test/28-60/",
    #            "./output/test/40-120/", "./output/test/40-100/", "./output/test/40-80/",
    #            "./output/test/35-100/", "./output/test/35-80/", "./output/test/35-60/",
    #            "./output/test/25-100/", "./output/test/25-80/", "./output/test/25-60/",
    #            "./output/test/20-80/", "./output/test/20-60/", "./output/test/20-40/"]

    data_dir = ["./input/test/v2/25-60/",
                "./input/test/v2/25-70/",
                "./input/test/v2/25-80/",
                "./input/test/v2/25-90/",
                "./input/test/v2/25-100/"]
    res_dir = ["./output/test/v2/25-60/",
               "./output/test/v2/25-70/",
               "./output/test/v2/25-80/",
               "./output/test/v2/25-90/",
               "./output/test/v2/25-100/"]

    use_gnn = bool(cfg.use_gnn)
    use_added_info = bool(cfg.use_added_info)
    encoding = cfg.encoding
    restriction = bool(cfg.restriction)
    algorithm = cfg.algorithm
    random_seed = cfg.random_seed

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    sequencing = ["SPT", "MOR", "MWKR"]
    assignment = ["MF", "LU", "HP"]
    PDR = []
    for i in sequencing:
        for j in assignment:
            PDR.append(i + "-" + j)

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        test_paths = os.listdir(data_dir_temp)
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

                env = QuayScheduling(data_dir_temp + path, algorithm=name,
                                     state_encoding=encoding, restriction=restriction,
                                     record_events=False, device=torch.device('cpu'))

                if name == "RL":
                    with open(param_path, 'r') as f:
                        parameters = json.load(f)

                    agent = Scheduler(env.meta_data, env.state_size, env.num_nodes,
                                      int(parameters['embed_dim']),
                                      int(parameters['num_heads']),
                                      int(parameters['num_HGT_layers']),
                                      int(parameters['num_actor_layers']),
                                      int(parameters['num_critic_layers']),
                                      use_gnn=use_gnn, use_added_info=use_added_info).to(torch.device('cpu'))
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    agent.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent = Heuristic(env.num_of_ships, env.num_of_quays)

                start = time.time()
                state, mask, current_ops, added_info = env.reset()
                done = False

                while not done:
                    if name == "RL":
                        action, _, _ = agent.act(state, mask, current_ops, added_info, greedy=False)
                    else:
                        action = agent.act(state)

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
                print("%d/%d test for %s done" % (progress, len(index) - 1, name))

            df_delay[name] = list_delay + [sum(list_delay) / len(list_delay)]
            df_move[name] = list_move + [sum(list_move) / len(list_move)]
            df_priority[name] = list_priority + [sum(list_priority) / len(list_priority)]
            df_delay_cost[name] = list_delay_cost + [sum(list_delay_cost) / len(list_delay_cost)]
            df_move_cost[name] = list_move_cost + [sum(list_move_cost) / len(list_move_cost)]
            df_loss_cost[name] = list_loss_cost + [sum(list_loss_cost) / len(list_loss_cost)]
            df_computing_time[name] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
            print("==========test for %s finished==========" % name)

        writer = pd.ExcelWriter(res_dir_temp + 'test_results.xlsx')
        df_delay.to_excel(writer, sheet_name="delay")
        df_move.to_excel(writer, sheet_name="move")
        df_priority.to_excel(writer, sheet_name="priority")
        df_delay_cost.to_excel(writer, sheet_name="delay_cost")
        df_move_cost.to_excel(writer, sheet_name="move_cost")
        df_loss_cost.to_excel(writer, sheet_name="loss_cost")
        df_computing_time.to_excel(writer, sheet_name="computing_time")
        writer.close()
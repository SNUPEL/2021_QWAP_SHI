import os
import time
import json

import torch

from cfg_communicate import get_cfg
from environment.env import *
from agent.network import *
from agent.heuristics import *
from pathlib import Path
import sys
import argparse

if __name__=="__main__":
    cfg = get_cfg()

    model_path = cfg.model_path
    param_path = cfg.param_path

    use_gnn = bool(cfg.use_gnn)
    use_added_info = bool(cfg.use_added_info)
    encoding = cfg.encoding
    restriction = bool(cfg.restriction)
    algorithm = cfg.algorithm
    random_seed = cfg.random_seed

    PDR = ["SPT-MF", "MOR-MF", "MWKR-MF"]

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, required=True, help="데이터 폴더(또는 파일) 경로")
        parser.add_argument("--res_path", type=str, required=True, help="결과 저장 폴더 경로")
        args = parser.parse_args()

        # model_path = Path(args.model_path).resolve()
        data_dir = args.data_path
        res_dir = args.res_path
        # data_dir = sys.argv[1]  # TODO: Unity로 입력받도록 코드 구성
        # res_dir = sys.argv[2]  # TODO: Unity로 전송되도록 코드 구성
    else:
        data_dir = cfg.data_path
        res_dir = cfg.res_path

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
            file_path = os.path.join(data_dir, path)
            env = QuayScheduling(file_path, algorithm=name,
                                 state_encoding=encoding, restriction=restriction,
                                 record_events=True, device=torch.device('cpu'))


            if name == "RL":
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

                next_state, _, done, next_mask, next_current_ops, next_added_info = env.step(action)

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

        log_df = env.get_logs()
        log_df.to_csv(res_dir + f'\\log-{name}.csv', header=False, index=False, encoding="utf-8-sig")
        # log_df.astype(int).to_csv(res_dir + f'\\log-{name}.csv', header=False, index=False, encoding="utf-8-sig")

        # writer = pd.ExcelWriter(res_dir + f'\\log-{name}.xlsx')
        # df_delay.to_excel(writer, sheet_name="delay")
        # df_move.to_excel(writer, sheet_name="move")
        # df_priority.to_excel(writer, sheet_name="priority")
        # df_delay_cost.to_excel(writer,sheet_name="delay_cost")
        # df_move_cost.to_excel(writer, sheet_name="move_cost")
        # df_loss_cost.to_excel(writer, sheet_name="loss_cost")
        # df_computing_time.to_excel(writer, sheet_name="computing_time")
        # env.get_logs().to_excel(writer, sheet_name="logs")
        # writer.close()

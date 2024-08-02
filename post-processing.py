import os
import numpy as np
import pandas as pd


def get_ideal_cost():
    data_dir = ["./input/test/28-100/", "./input/test/28-90/", "./input/test/28-80/", "./input/test/28-70/",
                "./input/test/28-60/",
                "./input/test/40-120/", "./input/test/40-100/", "./input/test/40-80/",
                "./input/test/35-100/", "./input/test/35-80/", "./input/test/35-60/",
                "./input/test/25-100/", "./input/test/25-80/", "./input/test/25-60/",
                "./input/test/20-80/", "./input/test/20-60/", "./input/test/20-40/"]

    res_dir = ["./output/test/28-100/", "./output/test/28-90/", "./output/test/28-80/", "./output/test/28-70/",
               "./output/test/28-60/",
               "./output/test/40-120/", "./output/test/40-100/", "./output/test/40-80/",
               "./output/test/35-100/", "./output/test/35-80/", "./output/test/35-60/",
               "./output/test/25-100/", "./output/test/25-80/", "./output/test/25-60/",
               "./output/test/20-80/", "./output/test/20-60/", "./output/test/20-40/"]

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        file_paths = os.listdir(data_dir_temp)
        index = ["P%d" % i for i in range(1, len(file_paths) + 1)] + ["avg"]
        columns = ["Ideal"]
        costs = []

        df_ideal = pd.DataFrame(index=index, columns=columns)
        for prob, path in zip(index, file_paths):
            df_scenario = pd.read_excel(data_dir_temp + path, sheet_name="ship", engine='openpyxl')
            df_quay = df_scenario[(df_scenario["Operation_Type"] != "시운전") & (df_scenario["Operation_Type"] != "G/T")]
            df_sea = df_scenario[(df_scenario["Operation_Type"] == "시운전") | (df_scenario["Operation_Type"] == "G/T")]

            delay_cost = 0
            moving_cost = 4000 * (2 * len(df_sea) + 2)
            loss_cost = 12 * 5 * (df_quay["Duration"].sum())

            ideal_cost = delay_cost + moving_cost + loss_cost
            costs.append(ideal_cost)

        df_ideal["Ideal"] = costs + [sum(costs) / len(costs)]

        writer = pd.ExcelWriter(res_dir_temp + 'baseline.xlsx')
        df_ideal.to_excel(writer, sheet_name="ideal")
        writer.close()


def summary(file_dir):
    ideal_cost = pd.read_excel(file_dir + "baseline.xlsx", index_col=0, engine='openpyxl').squeeze()

    file_list = ["(RL-BG) test_results.xlsx",
                 "(RL-DG) test_results.xlsx",
                 "(RL-MLP) test_results.xlsx",
                 "(RL-restricted) test_results.xlsx",
                 "(RL-wo-added info) test_results.xlsx",
                 "(Heuristics) test_results.xlsx"]

    df_list = []
    for temp in file_list:
        delay_cost = pd.read_excel(file_dir + temp, sheet_name="delay_cost", index_col=0, engine="openpyxl")
        moving_cost = pd.read_excel(file_dir + temp, sheet_name="move_cost", index_col=0, engine="openpyxl")
        loss_cost = pd.read_excel(file_dir + temp, sheet_name="loss_cost", index_col=0, engine="openpyxl")
        total_cost = pd.DataFrame(0.0, columns=delay_cost.columns, index=delay_cost.index)

        total_cost = total_cost.add(delay_cost)
        total_cost = total_cost.add(moving_cost)
        total_cost = total_cost.add(loss_cost)

        if "RL" in temp:
            total_cost = total_cost.rename(columns={"RL": temp.split()[0][1:-1]})

        df_list.append(total_cost)

    df = pd.concat(df_list, axis=1)
    df = df.div(ideal_cost, axis="index")
    df = df.mul(100)

    df.loc["avg"] = df.apply(np.mean, axis=0)

    writer = pd.ExcelWriter(file_dir + 'summary.xlsx')
    df.to_excel(writer, sheet_name="results")
    writer.close()


if __name__ == "__main__":
    file_dir = ["./output/test/28-100/", "./output/test/28-90/", "./output/test/28-80/",
                "./output/test/28-70/", "./output/test/28-60/",
               "./output/test/40-120/", "./output/test/40-100/", "./output/test/40-80/",
               "./output/test/35-100/", "./output/test/35-80/", "./output/test/35-60/",
               "./output/test/25-100/", "./output/test/25-80/", "./output/test/25-60/",
               "./output/test/20-80/", "./output/test/20-60/", "./output/test/20-40/"]

    for temp in file_dir:
        summary(temp)
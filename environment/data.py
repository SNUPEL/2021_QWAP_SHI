import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, n_ships, file_path, docks_info=None):
        self.n_ships = n_ships
        self.df_quay = pd.read_excel(file_path, sheet_name="quay", engine="openpyxl")
        self.df_ship = pd.read_excel(file_path, sheet_name="ship", engine="openpyxl")
        self.df_operation = pd.read_excel(file_path, sheet_name="operation", engine="openpyxl")

        if "선종" in self.df_quay.columns and "작업" in self.df_quay.columns:
            self.df_quay = self.df_quay.set_index(["선종", "작업"])

        if "선종" in self.df_operation.columns and "작업" in self.df_operation.columns:
            self.df_operation = self.df_operation.set_index(["선종", "작업"])

        if docks_info is not None:
            self.docks_info = docks_info
        else:
            # self.docks_info = {"Dock1": {"iat": 30, "num": 2},
            #                    "Dock2": {"iat": 30, "num": 1},
            #                    "Dock3": {"iat": 50, "num": 1},
            #                    "Dock4": {"iat": 50, "num": 1},
            #                    "Dock5": {"iat": 50, "num": 1}}
            self.docks_info = {"Dock1": {"iat": 45, "num": 2},
                               "Dock2": {"iat": 45, "num": 1},
                               "Dock3": {"iat": 75, "num": 1},
                               "Dock4": {"iat": 75, "num": 1},
                               "Dock5": {"iat": 75, "num": 1}}

    def generate(self, file_path=None):
        columns = ["Ship_Name", "Ship_Index", "Ship_Type", "Category", "Launching_Date", "Delivery_Date",
                   "Operation_Name", "Operation_Index", "Operation_Type", "Order",
                   "Start_Date", "Finish_Date", "Duration", "Interruption", "Fixed_Duration"]
        df_scenario = pd.DataFrame(columns=columns)
        df_quay = self.df_quay

        # 진수일 생성
        launching_dates = np.array([])
        for dock, info in self.docks_info.items():
            iat = np.random.exponential(info["iat"], self.n_ships).astype("int")
            launching_dates = np.concatenate([launching_dates, np.repeat(np.cumsum(iat), info["num"])])
        launching_dates = np.sort(launching_dates)
        launching_dates = launching_dates - np.min(launching_dates)

        # 선박 데이터 생성
        ship_idx = 0
        operation_idx = 0
        for i in range(self.n_ships):
            ship_name = "J-%d" % i
            ship_type = np.random.choice(list(self.df_ship["선종"]), p=list(self.df_ship["비율"]))

            ship_info = self.df_ship[self.df_ship["선종"] == ship_type].iloc[0].to_dict()
            category = ship_info["구분"]

            launching_date = launching_dates[i]
            total_duration = np.random.randint(ship_info["최소"], ship_info["최대"] + 1)
            delivery_date = launching_date + total_duration

            operation_info = self.df_operation.loc[ship_type]
            start_date = launching_date
            for operation_type, row in operation_info.iterrows():
                order = row["순번"]
                operation_name = "O-%d-%d" % (i, order)
                interruption = row["자르기"]
                fixed_duration = row["필수기간"]

                if order == len(operation_info) - 1:
                    duration = delivery_date - start_date
                    finish_date = delivery_date
                else:
                    duration = np.ceil(total_duration * ((row["종료(%)"] - row["착수(%)"]) / 100))
                    finish_date = start_date + duration

                temp = {"Ship_Name": ship_name, "Ship_Index": ship_idx,
                        "Ship_Type": ship_type, "Category": category,
                        "Launching_Date": launching_date, "Delivery_Date": delivery_date,
                        "Operation_Name": operation_name, "Operation_Index": operation_idx,
                        "Operation_Type": operation_type, "Order": order,
                        "Start_Date": start_date, "Finish_Date": finish_date, "Duration": duration,
                        "Interruption": interruption, "Fixed_Duration": fixed_duration}
                temp = pd.DataFrame(temp, index=[0])
                df_scenario = pd.concat([df_scenario, temp], ignore_index=True)

                start_date = finish_date
                operation_idx += 1

            ship_idx += 1

        # 초기 배치 데이터 생성
        min_finish_date = df_scenario["Finish_Date"].min()
        offset_operation = df_scenario[df_scenario["Finish_Date"] == min_finish_date]
        offset = np.random.randint(offset_operation["Start_Date"], offset_operation["Finish_Date"])[0]
        df_initial = df_scenario[(df_scenario["Start_Date"] <= offset) & (df_scenario["Finish_Date"] >= offset)]
        df_initial = df_initial.sort_values(by=["Start_Date"])
        df_initial = df_initial.reset_index(drop=True)

        occupied_quays = []
        for i, row in df_initial.iterrows():
            quay_list = df_quay.loc[(row["Ship_Type"], row["Operation_Type"])]
            quay_list = quay_list[(quay_list != "N") & ~(quay_list.index.isin(occupied_quays))]

            if len(quay_list) == 0:
                quay = "S"
            else:
                quay_list = quay_list[~quay_list.index.isin(occupied_quays)]
                if len(quay_list) == 0:
                    quay = "Buffer"
                else:
                    quay = quay_list.sample(n=1).index.to_numpy()[0]
                    occupied_quays.append(quay)

            df_initial.loc[i, "Initial_Quay"] = quay

        if file_path is not None:
            with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                df_scenario.to_excel(writer, sheet_name="ship", index=False)
                df_initial.to_excel(writer, sheet_name="initial", index=False)
                df_quay.reset_index().to_excel(writer, sheet_name="quay", index=False)

        return df_scenario, df_initial, df_quay


def get_load_graph(df_scenario, graph=False, filepath=None):
    start = df_scenario["Start_Date"].min()
    finish = df_scenario["Finish_Date"].max()
    timeline = np.arange(start, finish + 1)
    load = np.zeros(int(finish - start) + 1)

    for i, row in df_scenario.iterrows():
        load[int(row["Start_Date"]):int(row["Finish_Date"]) + 1] += 1

    if graph:
        fig, ax = plt.subplots(1, figsize=(16, 6))
        ax.plot(timeline, load)
        plt.show()
        if filepath is not None:
            plt.savefig(filepath)
        plt.close()

    return np.max(load)


if __name__ == "__main__":
    file_path = "../input/configurations/v2/config (m=25).xlsx"

    # # validation data generation
    # n_ships = 80
    # data_src = DataGenerator(n_ships, file_path)
    # file_dir = "../input/validation/v2/25-%d/" % n_ships
    #
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
    #
    # iteration = 10
    # for i in range(1, iteration + 1):
    #     flag = True
    #     while flag:
    #         file_path = file_dir + "instance-{0}.xlsx".format(i)
    #         df_scenario, df_initial, df_quay = data_src.generate(file_path)
    #         max_load = get_load_graph(df_scenario)
    #         if len(df_quay.columns) * 0.8 <= max_load <= len(df_quay.columns) * 1.2:
    #             flag = False

    # test data generation
    for n_ships in [60, 70, 80, 90, 100]:
        data_src = DataGenerator(n_ships, file_path)
        test_dir = "../input/test/v2/25-%s/" % str(n_ships)

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        iteration = 50
        for i in range(1, iteration + 1):
            flag = True
            while flag:
                test_file_path = test_dir + "instance-{0}.xlsx".format(i)
                df_scenario, df_initial, df_quay = data_src.generate(file_path=test_file_path)
                max_load = get_load_graph(df_scenario)
                if len(df_quay.columns) * 0.8 <= max_load <= len(df_quay.columns) * 1.2:
                    flag = False
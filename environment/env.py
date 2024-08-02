import os
import simpy
import numpy as np
import torch

from torch_geometric.data import HeteroData
from collections import OrderedDict
from environment.simulation import *
from environment.data import DataGenerator, get_load_graph


class QuayScheduling:
    def __init__(self, data_src, look_ahead=3, w_delay=0.0, w_move=1.0, w_priority=1.0,
                 algorithm="RL", state_encoding="DG", restriction=False, record_events=True, device=None):
        self.data_src = data_src
        self.look_ahead = look_ahead
        self.w_delay = w_delay
        self.w_move = w_move
        self.w_priority = w_priority
        self.algorithm = algorithm
        self.state_encoding = state_encoding
        self.restriction = restriction
        self.record_events = record_events
        self.device = device

        if type(self.data_src) is DataGenerator:
            flag = True
            while flag:
                self.df_scenario, self.df_initial, self.df_quay = self.data_src.generate()
                max_load = get_load_graph(self.df_scenario)
                if len(self.df_quay.columns) * 0.8 <= max_load <= len(self.df_quay.columns) * 1.2:
                    flag = False
        else:
            self.df_scenario = pd.read_excel(data_src, sheet_name="ship", engine='openpyxl')
            self.df_initial = pd.read_excel(data_src, sheet_name="initial", engine='openpyxl')
            self.df_quay = pd.read_excel(data_src, sheet_name="quay", engine='openpyxl').set_index(["선종", "작업"])

        self.df_scenario = self.df_scenario.sort_values(by=["Operation_Index"])
        self.df_quay = self.df_quay.sort_index(axis=1)

        if not self.df_quay.iloc[0].isin(["A", "B", "C", "D", "E", "N"]).any():
            self.numerical_encoding = True
        else:
            self.numerical_encoding = False

        self.num_of_ships = len(self.df_scenario["Ship_Name"].unique())
        self.num_of_ops = len(self.df_scenario)
        self.num_of_quays = len(self.df_quay.columns)

        self.meta_data, self.state_size, self.num_nodes, self.quay_ids, self.ship_ids = self._initialize()

        self.actions_done = []
        self.decision_time = 0.0

    def step(self, action):
        quay_id = action % self.num_nodes["quay"]
        ship_id = action // self.num_nodes["quay"]
        done = False

        self.actions_done.append(quay_id)

        ship = self.monitor.remove_queue(ship_id)
        current_quay = ship.current_quay
        next_quay = self.quay_ids[quay_id] if self.quay_ids.get(quay_id) is not None else "Buffer"

        if current_quay is None:
            self.model["Source"].calling_event[ship.name].succeed(next_quay)
        else:
            self.model[current_quay].calling_event[ship.name].succeed(next_quay)

        while True:
            while True:
                if self.monitor.scheduling:
                    while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                        self.sim_env.step()
                    break
                if len(self.monitor.operations_done) == len(self.df_scenario):
                    done = True
                    break
                self.sim_env.step()

            if self.decision_time != self.sim_env.now:
                self.actions_done = []

            if self.algorithm == "RL":
                next_state, current_ops, added_info = self._get_state_for_RL()
                mask = self._get_mask()
            else:
                next_state, current_ops, added_info = self._get_state_for_heuristics()
                mask = self._get_mask()

            if done:
                break

            if not mask.any():
                new_ships = []
                for ship in self.monitor.ships_in_queue.values():
                    operation = ship.get_current_operation()
                    if not operation.id in self.monitor.operations_in_buffer.keys():
                        new_ships.append(ship.id)

                for ship_id in new_ships:
                    ship = self.monitor.remove_queue(ship_id)
                    current_quay = ship.current_quay
                    next_quay = "Buffer"

                    if current_quay is None:
                        self.model["Source"].calling_event[ship.name].succeed(next_quay)
                    else:
                        self.model[current_quay].calling_event[ship.name].succeed(next_quay)
                if self.monitor.scheduling:
                    self.monitor.scheduling = False
            else:
                break

        reward = self._calculate_reward()

        if self.decision_time != self.sim_env.now:
            self.decision_time = self.sim_env.now

        return next_state, reward, done, mask, current_ops, added_info

    def reset(self):
        self.sim_env, self.ships, self.model, self.monitor = self._modeling()

        while True:
            while True:
                if self.monitor.scheduling:
                    while self.sim_env in [event[0] for event in self.sim_env._queue]:
                        self.sim_env.step()
                    break
                self.sim_env.step()

            if self.decision_time != self.sim_env.now:
                self.actions_done = []
                self.decision_time = self.sim_env.now

            if self.algorithm == "RL":
                initial_state, current_ops, added_info = self._get_state_for_RL()
                mask = self._get_mask()
            else:
                initial_state, current_ops, added_info = self._get_state_for_heuristics()
                mask = self._get_mask()

            if not mask.any():
                new_ships = []
                for ship in self.monitor.ships_in_queue.values():
                    operation = ship.get_current_operation()
                    if not operation.id in self.monitor.operations_in_buffer.keys():
                        new_ships.append(ship.id)

                for ship_id in new_ships:
                    ship = self.monitor.remove_queue(ship_id)
                    current_quay = ship.current_quay
                    next_quay = "Buffer"

                    if current_quay is None:
                        self.model["Source"].calling_event[ship.name].succeed(next_quay)
                    else:
                        self.model[current_quay].calling_event[ship.name].succeed(next_quay)
            else:
                self.decision_time = self.sim_env.now
                break

        return initial_state, mask, current_ops, added_info

    def get_logs(self, path=None):
        log = self.model["Sink"].monitor.get_logs(path)
        return log

    def generate_new_instance(self):
        if not type(self.data_src) is DataGenerator:
            print("Invalid data source")
        else:
            flag = True
            while flag:
                self.df_scenario, self.df_initial, self.df_quay = self.data_src.generate()
                max_load = get_load_graph(self.df_scenario)
                if len(self.df_quay.columns) * 0.8 <= max_load <= len(self.df_quay.columns) * 1.2:
                    flag = False

            self.df_scenario = self.df_scenario.sort_values(by=["Operation_Index"])
            self.df_quay = self.df_quay.sort_index(axis=1)

            self.num_of_ships = len(self.df_scenario["Ship_Name"].unique())
            self.num_of_ops = len(self.df_scenario)
            self.num_of_quays = len(self.df_quay.columns)

            self.meta_data, self.state_size, self.num_nodes, self.quay_ids, self.ship_ids = self._initialize()

    def _initialize(self):
        if self.state_encoding == "BG":
            meta_data = (["quay", "ship"],
                         [("ship", "good", "quay"), ("quay", "good_inv", "ship"),
                          ("ship", "bad", "quay"), ("quay", "bad_inv", "ship")])
            state_size = {"quay": 4, "ship": 6 * self.look_ahead}
            num_nodes = {"quay": self.num_of_quays, "ship": self.num_of_ships}
        elif self.state_encoding == "DG":
            meta_data = (["quay", "operation"],
                         [("operation", "predecessor", "operation"),
                          ("operation", "good", "quay"), ("quay", "good_inv", "operation"),
                          ("operation", "bad", "quay"), ("quay", "bad_inv", "operation")])
            state_size = {"quay": 4, "operation": 8}
            num_nodes = {"quay": self.num_of_quays, "operation": self.num_of_ops}

        quay_ids = OrderedDict()
        for i, quay in enumerate(self.df_quay.columns):
            quay_ids[i] = quay

        ship_ids = OrderedDict()
        for i, row in self.df_scenario.iterrows():
            if not row["Ship_Index"] in ship_ids.keys():
                ship_ids[int(row["Ship_Index"])] = row["Ship_Name"]

        return meta_data, state_size, num_nodes, quay_ids, ship_ids

    def _get_mask(self):
        mask = np.zeros((self.num_of_quays, self.num_of_ships), dtype=bool)
        if self.restriction:
            mask_restricted = np.zeros((self.num_of_quays, self.num_of_ships), dtype=bool)

        for ship in self.monitor.ships_in_queue.values():
            operation = ship.get_current_operation()

            for quay_id, name in self.quay_ids.items():
                quay = self.model[name]

                priority_score = operation.get_priority_score(quay_id)
                occupied, interruption = quay.check_status(self.sim_env.now)

                if (priority_score != "N") and (not quay_id in self.actions_done):
                    if (not occupied) or (occupied and interruption):
                        mask[quay_id, ship.id] = 1
                    if self.restriction:
                        if not occupied:
                            mask_restricted[quay_id, ship.id] = 1

        if self.restriction:
            if mask_restricted.any():
                mask = mask_restricted

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return mask

    def _get_state_for_RL(self):
        state = HeteroData()

        if self.state_encoding == "DG":
            X_ops = np.zeros((self.num_of_ops, self.state_size["operation"]))
        elif self.state_encoding == "BG":
            X_ships = np.zeros((self.num_of_ships, self.state_size["ship"]))
        X_quays = np.zeros((self.num_of_quays, self.state_size["quay"]))
        added_info = np.zeros((self.num_of_ships, self.num_of_quays, 2))

        if self.state_encoding == "DG":
            edge_pre = [[], []]
            # edge_pre, edge_suc = [[], []], [[], []]
        edge_good, edge_bad = [[], []], [[], []]
        edge_good_inv, edge_bad_inv = [[], []], [[], []]

        current_ops = np.zeros(self.num_of_ships)

        for ship_idx, ship_name in self.ship_ids.items():
            if ship_idx in self.monitor.ships_before_LC.keys():
                ship = self.monitor.ships_before_LC[ship_idx]
                current_ops[ship.id] = ship.operations[0].id
            elif ship_idx in self.monitor.ships_in_process.keys():
                ship = self.monitor.ships_in_process[ship_idx]
                current_ops[ship.id] = ship.operations[ship.step].id
            else:
                ship = self.monitor.ships_after_DL[ship_idx]
                current_ops[ship.id] = ship.operations[-1].id

            for i, operation in enumerate(ship.operations):
                if self.state_encoding == "DG":
                    if operation.id in self.monitor.operations_in_quay.keys():
                        operation = self.monitor.operations_in_quay[operation.id]
                        X_ops[operation.id, 0:3] = [0, 1, 0]
                    elif operation.id in self.monitor.operations_interrupted.keys():
                        operation = self.monitor.operations_interrupted[operation.id]
                        X_ops[operation.id, 0:3] = [0, 1, 0]
                    elif operation.id in self.monitor.operations_in_buffer.keys():
                        operation = self.monitor.operations_in_buffer[operation.id]
                        X_ops[operation.id, 0:3] = [0, 1, 0]
                    elif operation.id in self.monitor.operations_done:
                        operation = self.monitor.operations_done[operation.id]
                        X_ops[operation.id, 0:3] = [0, 0, 1]
                    else:
                        operation = self.monitor.operations_unscheduled[operation.id]
                        X_ops[operation.id, 0:3] = [1, 0, 0]

                    X_ops[operation.id, 3] = int(operation.duration) / self.df_scenario["Duration"].max()

                    if operation.interruption == "S":
                        X_ops[operation.id, 4:7] = [1, 0, 0]
                        X_ops[operation.id, 7] = operation.fixed_duration / operation.duration
                    elif operation.interruption == "F":
                        X_ops[operation.id, 4:7] = [0, 1, 0]
                        X_ops[operation.id, 7] = operation.fixed_duration / operation.duration
                    else:
                        X_ops[operation.id, 4:7] = [0, 0, 1]
                        X_ops[operation.id, 7] = 0

                    if i > 0:
                        edge_pre[0].append(operation.id - 1)
                        edge_pre[1].append(operation.id)
                    # if i < len(ship.operations) - 1:
                    #     edge_suc[0].append(operation.id + 1)
                    #     edge_suc[1].append(operation.id)

                    if self.numerical_encoding:
                        def transform(score):
                            if score >= 1:
                                grade = "A"
                            elif score >= 0.7:
                                grade = "B"
                            elif score >= 0.3:
                                grade = "C"
                            elif score >= 0.1:
                                grade = "D"
                            elif score > 0.0:
                                grade = "E"
                            else:
                                grade = "N"
                            return grade
                        priority_score = self.df_quay.loc[(ship.type, operation.type)].apply(transform).to_numpy()
                    else:
                        priority_score = self.df_quay.loc[(ship.type, operation.type)].to_numpy()

                    if not (priority_score == "N").all():
                        if not operation.id in self.monitor.operations_done.keys():
                            for j in range(len(priority_score)):
                                if priority_score[j] == "A" or priority_score[j] == "B":
                                    edge_good[0].append(operation.id)
                                    edge_good[1].append(j)
                                    edge_good_inv[0].append(j)
                                    edge_good_inv[1].append(operation.id)
                                elif priority_score[j] == "C" or priority_score[j] == "D" or priority_score[j] == "E":
                                    edge_bad[0].append(operation.id)
                                    edge_bad[1].append(j)
                                    edge_bad_inv[0].append(j)
                                    edge_bad_inv[1].append(operation.id)
                        else:
                            current_quay = self.model[operation.current_quay]
                            if priority_score[current_quay.id] == "A" or priority_score[current_quay.id] == "B":
                                edge_good[0].append(operation.id)
                                edge_good[1].append(current_quay.id)
                                edge_good_inv[0].append(current_quay.id)
                                edge_good_inv[1].append(operation.id)
                            elif priority_score[current_quay.id] == "C" or priority_score[current_quay.id] == "D" or \
                                    priority_score[current_quay.id] == "E":
                                edge_bad[0].append(operation.id)
                                edge_bad[1].append(current_quay.id)
                                edge_bad_inv[0].append(current_quay.id)
                                edge_bad_inv[1].append(operation.id)

                elif self.state_encoding == "BG":
                    if ship.step <= i < ship.step + self.look_ahead:
                        X_ships[ship_idx, 6 * (i - ship.step)] = (i + 1) / len(ship.operations)
                        X_ships[ship_idx, 6 * (i - ship.step) + 1] = operation.duration / self.df_scenario["Duration"].max()
                        if operation.interruption == "S":
                            X_ships[ship_idx, (6 * (i - ship.step) + 2):(6 * (i - ship.step) + 5)] = [1, 0, 0]
                            X_ships[ship_idx, 6 * (i - ship.step) + 5] = operation.fixed_duration / operation.duration
                        elif operation.interruption == "F":
                            X_ships[ship_idx, (6 * (i - ship.step) + 2):(6 * (i - ship.step) + 5)] = [0, 1, 0]
                            X_ships[ship_idx, 6 * (i - ship.step) + 5] = operation.fixed_duration / operation.duration
                        else:
                            X_ships[ship_idx, (6 * (i - ship.step) + 2):(6 * (i - ship.step) + 5)] = [0, 0, 1]

                    if i == ship.step:
                        priority_score = self.df_quay.loc[(ship.type, operation.type)].to_numpy()
                        for j in range(len(priority_score)):
                            if priority_score[j] == "A" or priority_score[j] == "B":
                                edge_good[0].append(ship_idx)
                                edge_good[1].append(j)
                                edge_good_inv[0].append(j)
                                edge_good_inv[1].append(ship_idx)
                            elif priority_score[j] == "C" or priority_score[j] == "D" or priority_score[j] == "E":
                                edge_bad[0].append(ship_idx)
                                edge_bad[1].append(j)
                                edge_bad_inv[0].append(j)
                                edge_bad_inv[1].append(ship_idx)

        # quay node에 대한 feature 계산
        for quay_id, name in self.quay_ids.items():
            quay = self.model[name]

            occupied, interruption = quay.check_status(self.sim_env.now)
            if not occupied:
                X_quays[quay_id, 0:3] = [1, 0, 0]
                X_quays[quay_id, 3] = 0
            else:
                if interruption:
                    X_quays[quay_id, 0:3] = [0, 1, 0]
                    X_quays[quay_id, 3] = 0
                else:
                    X_quays[quay_id, 0:3] = [0, 0, 1]
                    remaining_duration = float('inf')
                    max_duration = 0
                    for operation in quay.operations_in_working.values():
                        remaining_duration = min(remaining_duration, operation.duration -
                                                 (operation.progress + self.sim_env.now - operation.working_start))
                        max_duration = max(max_duration, operation.duration)
                    X_quays[quay_id, 3] = remaining_duration / max_duration

        for ship_idx, ship in self.monitor.ships_in_queue.items():
            operation = ship.get_current_operation()
            for quay_idx, name in self.quay_ids.items():
                quay = self.model[name]
                occupied, interruption = quay.check_status(self.sim_env.now)
                priority_score = operation.get_priority_score(quay_idx)

                candidate_flag = not quay_idx in self.actions_done
                if self.numerical_encoding:
                    candidate_flag = candidate_flag and priority_score > 0.0
                else:
                    candidate_flag = candidate_flag and priority_score != "N"

                if candidate_flag:
                    info = []
                    if not occupied:
                        info.append(1.0)
                    else:
                        info.append(0.0)

                    if self.numerical_encoding:
                        info.append(priority_score / 1.5)
                    else:
                        if priority_score in ["A", "B"]:
                            info.append(1.0)
                        else:
                            info.append(0.0)

                    added_info[ship_idx, quay_idx] = info

        if self.state_encoding == "DG":
            X_ops = torch.tensor(X_ops, dtype=torch.float32).to(self.device)
            state["operation"].x = X_ops
        elif self.state_encoding == "BG":
            X_ships = torch.tensor(X_ships, dtype=torch.float32).to(self.device)
            state["ship"].x = X_ships

        X_quays = torch.tensor(X_quays, dtype=torch.float32).to(self.device)
        state["quay"].x = X_quays

        added_info = torch.tensor(added_info, dtype=torch.float32).to(self.device)

        if self.state_encoding == "DG":
            edge_pre = torch.from_numpy(np.array(edge_pre)).type(torch.long).to(self.device)
            state["operation", "predecessor", "operation"].edge_index = edge_pre
            # edge_suc = torch.from_numpy(np.array(edge_suc)).type(torch.long).to(self.device)
            # state["operation", "successor", "operation"].edge_index = edge_suc

        edge_good = torch.from_numpy(np.array(edge_good)).type(torch.long).to(self.device)
        edge_good_inv = torch.from_numpy(np.array(edge_good_inv)).type(torch.long).to(self.device)
        edge_bad = torch.from_numpy(np.array(edge_bad)).type(torch.long).to(self.device)
        edge_bad_inv = torch.from_numpy(np.array(edge_bad_inv)).type(torch.long).to(self.device)

        if self.state_encoding == "DG":
            state["operation", "good", "quay"].edge_index = edge_good
            state["quay", "good_inv", "operation"].edge_index = edge_good_inv
            state["operation", "bad", "quay"].edge_index = edge_bad
            state["quay", "bad_inv", "operation"].edge_index = edge_bad_inv
        elif self.state_encoding == "BG":
            state["ship", "good", "quay"].edge_index = edge_good
            state["quay", "good_inv", "ship"].edge_index = edge_good_inv
            state["ship", "bad", "quay"].edge_index = edge_bad
            state["quay", "bad_inv", "ship"].edge_index = edge_bad_inv

        current_ops = torch.tensor(current_ops, dtype=torch.long).to(self.device)

        return state, current_ops, added_info

    def _get_state_for_heuristics(self):
        state = {}
        sequencing, assignment = self.algorithm.split("-")

        for ship in self.monitor.ships_in_queue.values():
            if sequencing == "SPT":
                operation = ship.get_current_operation()
                duration = operation.get_duration()
                priority_idx = 1 / duration
            elif sequencing == "MOR":
                num_of_remaining_ops = len(ship.operations) - ship.step
                priority_idx = num_of_remaining_ops
            elif sequencing == "MWKR":
                remaining_duration = 0
                for operation in ship.operations[ship.step:]:
                    duration = operation.get_duration()
                    remaining_duration += duration
                priority_idx = remaining_duration
            elif sequencing == "FIFO":
                operation = ship.get_current_operation()
                waiting_start = operation.waiting_start
                priority_idx = 1 / waiting_start if waiting_start > 0 else float('inf')
            else:
                print("invalid algorithm name")
            state[ship.id] = [priority_idx]

            temp = {}
            for name in self.df_quay.columns:
                quay_wall = self.model[name]
                operation = ship.get_current_operation()
                priority_score = operation.get_priority_score(quay_wall.id)

                if (not quay_wall.id in self.actions_done) and (priority_score != "N"):
                    occupied, interruption = quay_wall.check_status(self.sim_env.now)
                    if (not occupied) or (occupied and interruption):
                        if assignment == "MF":
                            flexibility = 0
                            for ship_temp in self.monitor.ships_in_queue.values():
                                operation_temp = ship_temp.get_current_operation()
                                priority_score_temp = operation_temp.get_priority_score(quay_wall.id)
                                eligible = 1 if priority_score_temp != "N" else 0
                                flexibility += eligible
                            priority_idx = flexibility / len(self.monitor.ships_in_queue)
                        elif assignment == "LU":
                            utilization = quay_wall.utilization / self.sim_env.now
                            priority_idx = 1 - utilization
                        elif assignment == "HP":
                            if self.numerical_encoding:
                                preference = priority_score / 1.5
                            else:
                                if priority_score in ["A", "B"]:
                                    preference = 1
                                else:
                                    preference = 0.5
                            priority_idx = preference

                    if not occupied:
                        temp[quay_wall.id] = priority_idx * 1.5
                    elif occupied and interruption:
                        temp[quay_wall.id] = priority_idx

            state[ship.id].append(temp)

        return state, None, None

    def _calculate_reward(self):
        reward_delay = 0
        reward_move = 0
        reward_priority = 0

        if self.sim_env.now - self.decision_time > 0:
            for ship_id, waiting_start in self.monitor.reward_delay.items():
                reward_delay += - (self.sim_env.now - waiting_start) # / (self.sim_env.now - self.decision_time)
                self.monitor.reward_delay[ship_id] = self.sim_env.now

        for interruption in self.monitor.reward_move.values():
            if interruption:
                reward_move += -1
        self.monitor.reward_move = {}

        for priority_score, duration in self.monitor.reward_priority.values():
            if self.numerical_encoding:
                reward_priority += (priority_score / 1.5 - 1.0)
            else:
                if priority_score == "A":
                    reward_priority += 0
                elif priority_score == "B":
                    reward_priority += 0
                elif priority_score == "C":
                    reward_priority += -1 * duration
                elif priority_score == "D":
                    reward_priority += -1 * duration
                elif priority_score == "E":
                    reward_priority += -1 * duration
        self.monitor.reward_priority = {}

        reward_total = self.w_delay * reward_delay + self.w_move * reward_move + self.w_priority * reward_priority

        return reward_total

    def _modeling(self):
        sim_env = simpy.Environment()
        monitor = Monitor(self.record_events)

        ships = []
        df_scenario_group = self.df_scenario.groupby(by=["Ship_Name", "Ship_Index", "Ship_Type", "Category",
                                                         "Launching_Date", "Delivery_Date"])

        for idx, group in df_scenario_group:
            ship_name, ship_index, ship_type, category, launching_date, delivery_date = idx

            operations = []
            for i, row in group.iterrows():
                priority_scores = self.df_quay.loc[(row["Ship_Type"], row["Operation_Type"])].to_list()
                operation = Operation(row["Operation_Name"], row["Operation_Index"], row["Operation_Type"],
                                      row["Start_Date"], row["Finish_Date"], row["Duration"],
                                      row["Interruption"], min(row["Fixed_Duration"], row["Duration"]), priority_scores)
                operations.append(operation)

            initial_step = 0
            initial_quay = None
            if ship_name in self.df_initial["Ship_Name"].tolist():
                initial_ship = self.df_initial[self.df_initial["Ship_Name"] == ship_name]
                initial_operation = initial_ship["Operation_Type"].to_list()[0]
                initial_step = initial_ship["Order"].tolist()[0] - 1
                initial_quay = initial_ship["Initial_Quay"].tolist()[0]

                if initial_quay == "S":
                    if self.numerical_encoding:
                        priority_score = self.df_quay.loc[(ship_type, initial_operation)].to_numpy()
                        waiting = not (np.sum(priority_score) == 0.0)
                    else:
                        priority_score = self.df_quay.loc[(ship_type, initial_operation)].to_numpy()
                        waiting = not (priority_score == "N").all()

                    if waiting:
                        initial_quay = "Buffer"

            ship = Ship(ship_name, ship_index, ship_type, category, launching_date, delivery_date,
                        operations, initial_step, initial_quay)
            ships.append(ship)

        ships = sorted(ships, key=lambda x: x.operations[x.step].start_planned)

        model = {}
        model["Source"] = Source(sim_env, "Source", ships, model, monitor)
        quay_id = 0
        for i in self.df_quay.columns:
            quay = Quay(sim_env, i, quay_id, model, monitor, capacity=1)
            model[i] = quay
            quay_id += 1
        model["S"] = Quay(sim_env, "S", quay_id, model, monitor, capacity=float('inf'))
        model["Buffer"] = Buffer(sim_env, "Buffer", model, monitor)
        model["Sink"] = Sink(sim_env, "Sink", monitor)

        return sim_env, ships, model, monitor


if __name__ == "__main__":

    from data import *
    from agent.network import *

    log_dir = '../result/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_src = "../input/temp.xlsx"
    w_delay = 1.0
    w_move = 0.5
    w_priority = 0.5
    record_events = True

    embed_dim = 128
    num_heads = 4
    num_HGT_layers = 2
    num_actor_layers = 3
    num_critic_layers = 3

    env = QuayScheduling(data_src, w_delay=w_delay, w_move=w_move, w_priority=w_priority, record_events=record_events)
    scheduler = Scheduler(env.meta_data, env.state_size, env.num_nodes, embed_dim, num_heads,
                          num_HGT_layers, num_actor_layers, num_critic_layers)

    done = False
    state, current_ops, mask = env.reset()
    r = []

    while not done:
        action = scheduler.act(state, current_ops, mask)

        next_state, reward, done, current_ops, mask = env.step(action)
        r.append(reward)
        state = next_state

        print(reward)
        #print(next_state)
        if done:
            break
    _ = env.get_logs("../temp.csv")
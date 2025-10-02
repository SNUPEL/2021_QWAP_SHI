import os
import time
import json
from collections import defaultdict

import torch
import pandas as pd

from agent.ppo import Agent
from cfg_communicate import get_cfg
from environment.env import *
from agent.network import *
from agent.heuristics import *
from pathlib import Path
import sys
import argparse


class CumulativeLogGenerator:
    def __init__(self):
        # 열로 쓸 대상 로그 이름(요구한 순서)
        self.desired_keys = [
            "MOR-MF",
            "MWKR-MF",
            "RL",
            "SPT-MF"
        ]
        self.instance_path = 'input/test/v2/28-70/instance-1.xlsx'
        self.log_path = dict()
        self.log_path['MOR-MF'] = 'Quay_Planning_Grid/Assets/Data/log-MOR-MF.xlsx'
        self.log_path['RL'] = 'Quay_Planning_Grid/Assets/Data/log-RL.xlsx'
        self.log_path['MWKR-MF'] = 'Quay_Planning_Grid/Assets/Data/log-MWKR-MF.xlsx'
        self.log_path['SPT-MF'] = 'Quay_Planning_Grid/Assets/Data/log-SPT-MF.xlsx'
        # log_path 안의 모든 파일로부터 Log 객체 생성
        self.logs = {}
        for key, path in self.log_path.items():
            try:
                self.logs[key] = Log(path, self.instance_path)
            except Exception as e:
                print(f"[ERROR] {key}: {e}")

        print("파일을 읽어들였습니다.")


    def _generate_cost_log(self, move_log_path, delay_log_path, priority_log_path):

        # 1) 기존 로그 로드 (index='time')
        move_df = pd.read_excel(move_log_path, index_col=0)
        delay_df = pd.read_excel(delay_log_path, index_col=0)
        priority_df = pd.read_excel(priority_log_path, index_col=0)

        # 2) 인덱스(시간) 정규화: 세 DF의 index 합집합으로 맞추고 누적로그 특성상 ffill
        all_idx = move_df.index.union(delay_df.index).union(priority_df.index).unique().sort_values()

        move_df = move_df.reindex(all_idx).ffill().fillna(0.0)
        delay_df = delay_df.reindex(all_idx).ffill().fillna(0.0)
        priority_df = priority_df.reindex(all_idx).ffill().fillna(0.0)

        # 3) 컬럼 정렬(세 DF 공통 컬럼만 사용; 없으면 0으로 보강하고 싶으면 reindex로 확장 가능)
        common_cols = sorted(set(move_df.columns) & set(delay_df.columns) & set(priority_df.columns))
        if common_cols:
            move_df = move_df[common_cols]
            delay_df = delay_df[common_cols]
            priority_df = priority_df[common_cols]

        # 4) 가중치 적용: delay:move:priority = 2:1:1
        w_delay, w_move, w_priority = 2.0, 1.0, -1.0
        cost_df = w_delay * delay_df + w_move * move_df + w_priority * priority_df
        cost_df.index.name = "time"

        # 5) 저장
        cost_df.to_excel("CostLog.xlsx")

        print("Saved: CostLog.xlsx")

    def ensure_delay_priority(self, logs, instance_path):
        """delay/priority 타임로그가 없으면 생성"""
        for k, lg in logs.items():
            # ship 정보 붙이기 (이미 붙어있어도 재호출 OK)
            try:
                lg.attach_instance(instance_path)
            except Exception:
                # attach_instance가 이미 되어 있거나 ship sheet가 불완전해도 무시하고 진행
                pass

            # delay
            if not getattr(lg, "delay_timelog", None):
                try:
                    lg.generate_delay_timelog(instance_path)
                except Exception as e:
                    print(f"[delay skip] {k}: {e}")

            # priority
            if not getattr(lg, "priority_timelog", None):
                try:
                    lg.generate_priority_timelog()
                except Exception as e:
                    print(f"[priority skip] {k}: {e}")

    def build_log_matrix(self, logs: dict, key_list: list, attr_name: str) -> pd.DataFrame:
        """
        logs[키].<attr_name> (dict: {time: value})를 모아
        행=0..max_t, 열=key_list 형태 DataFrame 반환.
        없는 키/시간은 0으로 채우고 누적 시계열은 ffill로 이어붙임.
        """
        # 전체 max_t 계산
        max_t = 0
        for k in key_list:
            tl = getattr(logs.get(k, None), attr_name, None)
            if tl:
                try:
                    max_t = max(max_t, max(tl.keys()))
                except ValueError:
                    pass
        idx = pd.RangeIndex(0, max_t + 1)

        # 컬럼별 시리즈 구성
        cols = {}
        for k in key_list:
            tl = getattr(logs.get(k, None), attr_name, None)
            if not tl:
                # 해당 로그가 없거나 타임로그가 없으면 0으로 채움
                cols[k] = pd.Series(0.0, index=idx)
                continue
            s = pd.Series(tl, dtype=float)
            s = s.reindex(idx).ffill().fillna(0.0)  # 누적로그 특성상 앞 값 유지
            cols[k] = s

        df = pd.DataFrame(cols, index=idx)
        df.index.name = "time"
        return df

    def save_logs(self):
        # 1) delay/priority 필요 시 생성
        self.ensure_delay_priority(self.logs, self.instance_path)

        # 2) 각 매트릭스 생성
        move_df = self.build_log_matrix(self.logs, self.desired_keys, "move_timelog")
        delay_df = self.build_log_matrix(self.logs, self.desired_keys, "delay_timelog")
        priority_df = self.build_log_matrix(self.logs, self.desired_keys, "priority_timelog")

        # 3) 엑셀 저장
        move_df.to_excel("MoveLog.xlsx")
        delay_df.to_excel("DelayLog.xlsx")
        priority_df.to_excel("PriorityLog.xlsx")

        print("Saved: MoveLog.xlsx, DelayLog.xlsx, PriorityLog.xlsx")
        self._generate_cost_log("MoveLog.xlsx", "DelayLog.xlsx", "PriorityLog.xlsx")


class Log:
    """
    Excel 로그 파일을 받아서
    - name: 파일명에서 자동 추출 (log-<NAME>.xlsx 형태 우선)
    - filepath: 원본 경로
    - move_timelog: {t: cum_count} 형태의 누적 이동 횟수 사전
    - maxtime: move_timelog의 최대 키 값
    """

    def __init__(self, filepath: str, instancepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {filepath}")

        self.filepath = filepath
        self.name = self._infer_name_from_path(filepath)
        self._df = pd.read_excel(filepath)  # 원본 보관(필요 시)
        self.move_timelog = {}
        self.delay_timelog = {}
        self.priority_timelog = {}
        self.maxtime = 0

        # 추가 필드

        self._priority_increased_dict = {}
        self._delay_increased_dict = {}  # {time: delta} 원천 기록

        self.ship_df = None
        self._op_duration_map = None  # {op: duration}
        self.attach_instance(instancepath)

        # 생성 즉시 누적 이동 로그 구축
        self.generate_move_timelog()
        self.generate_delay_timelog(instancepath)
        self.generate_priority_timelog()

    @staticmethod
    def _infer_name_from_path(filepath: str) -> str:
        """
        파일명이 'log-XXX.xlsx' 형태라면 XXX를 name으로 사용.
        그렇지 않으면 확장자 없는 파일명을 name으로 사용.
        """
        stem = os.path.splitext(os.path.basename(filepath))[0]  # e.g., 'log-RL'
        if stem.lower().startswith('log-') and len(stem) > 4:
            return stem[4:]  # 'RL'
        return stem

    def attach_instance(self, instance_path: str, ship_sheet: str = 'ship'):
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"인스턴스 파일을 찾을 수 없습니다: {instance_path}")

        ship_df = pd.read_excel(instance_path, sheet_name=ship_sheet)
        required_cols = {'Operation_Name', 'Duration', 'Finish_Date'}
        if not required_cols.issubset(ship_df.columns):
            missing = required_cols - set(ship_df.columns)
            raise ValueError(f"'ship' 시트에 필요한 컬럼이 없습니다: {missing}")

        # 숫자형 변환
        ship_df['Duration'] = pd.to_numeric(ship_df['Duration'], errors='coerce')
        ship_df['Finish_Date'] = pd.to_numeric(ship_df['Finish_Date'], errors='coerce')

        # 대표 duration/due를 맵으로 (여러 행이면 첫 유효값 사용)
        op_duration_map = {}
        for op_name, grp in ship_df.groupby('Operation_Name'):
            dur_vals = grp['Duration'].dropna()
            if not dur_vals.empty:
                op_duration_map[str(op_name)] = float(dur_vals.iloc[0])

        self.ship_df = ship_df
        self._op_duration_map = op_duration_map  # {op: duration}

    def generate_move_timelog(self):
        """
        - 'Event' == 'Ship Moved' 인 행들의 'Time'을 사용
        - Time을 정수(time_floor = int(Time))로 깎아서,
          각 정수 시각에 발생한 'Ship Moved' 개수 누적
        - 0부터 max_int_time까지 {t: 누적횟수} 생성
        예) (2, 4)에 'Ship Moved'가 있으면
            {0:0, 1:0, 2:1, 3:1, 4:2}
        """
        if 'Event' not in self._df.columns or 'Time' not in self._df.columns:
            raise ValueError("로그 파일에 'Event' 또는 'Time' 컬럼이 없습니다.")

        # 숫자형으로 변환
        times = pd.to_numeric(self._df['Time'], errors='coerce')
        events = self._df['Event'].astype(str)

        # Ship Moved만 추출 후 정수 시간으로 바꿈
        moved_times = times[events.eq('Ship Moved')].dropna().astype(float)
        # 정수 시각(내림)으로 변환: 2.9 -> 2
        moved_int_times = moved_times.astype(int)

        if moved_int_times.empty:
            # 이동 이벤트가 없는 경우: 0 시각만 0으로
            self.move_timelog = {0: 0}
            self.maxtime = 0
            return

        max_t = int(max(moved_int_times.max(), 0))
        # 각 정수 시각별 발생 건수
        count_by_t = defaultdict(int)
        for t in moved_int_times:
            count_by_t[int(t)] += 1

        # 누적합 구성
        cumulative = 0
        timelog = {}
        for t in range(0, max_t + 1):
            cumulative += count_by_t.get(t, 0)
            timelog[t] = cumulative

        self.move_timelog = timelog
        self.maxtime = max_t

    def generate_delay_timelog(self, instance_path: str, ship_sheet_name: str = 'ship'):
        """
        instance_path의 'ship' 시트에서
          - operation_name: Operation_Name의 unique 값
          - due_date: Finish_Date
        을 읽어와 map을 구성하고,
        self._df에서 해당 operation의 'Working Finished'가
        due_date보다 늦으면 그 지연량 delta를 해당 시각 t=int(Time)에 기록.
        이후 0..max_t까지 누적으로 합산하여 delay_timelog를 만든다.
        """
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"인스턴스 파일을 찾을 수 없습니다: {instance_path}")

        ship_df = pd.read_excel(instance_path, sheet_name=ship_sheet_name)

        # 필수 컬럼 체크
        required_cols = {'Operation_Name', 'Finish_Date'}
        if not required_cols.issubset(set(ship_df.columns)):
            missing = required_cols - set(ship_df.columns)
            raise ValueError(f"'ship' 시트에 필요한 컬럼이 없습니다: {missing}")

        # Operation_Name -> due_date (수치) 매핑
        # 여러 행이 있을 경우 첫 값 또는 대표값 사용(여기서는 첫 유효 수치)
        op_due_map = {}
        for op_name, grp in ship_df.groupby('Operation_Name'):
            # Finish_Date를 수치로 변환(동일 단위 가정)
            due_vals = pd.to_numeric(grp['Finish_Date'], errors='coerce').dropna()
            if not due_vals.empty:
                op_due_map[str(op_name)] = float(due_vals.iloc[0])  # 대표 due
            # 변환 실패하면 해당 op는 스킵

        # 로그에서 'Working Finished' & Operation 매칭
        if 'Event' not in self._df.columns or 'Time' not in self._df.columns or 'Operation' not in self._df.columns:
            raise ValueError("로그 파일에 'Event', 'Time', 'Operation' 컬럼이 필요합니다.")

        log_times = pd.to_numeric(self._df['Time'], errors='coerce')
        log_events = self._df['Event'].astype(str)
        log_ops = self._df['Operation'].astype(str)

        finished_mask = log_events.eq('Working Finished') & log_times.notna()
        finished_df = self._df.loc[finished_mask, ['Time', 'Operation']].copy()
        finished_df['Time'] = pd.to_numeric(finished_df['Time'], errors='coerce')
        finished_df = finished_df[finished_df['Time'].notna()]
        if finished_df.empty:
            # 완료 이벤트 없으면 0으로 초기화만
            max_t = self.maxtime if self.move_timelog else int(
                pd.to_numeric(self._df['Time'], errors='coerce').max() or 0)
            self.delay_timelog = {t: 0.0 for t in range(0, max(max_t, 0) + 1)}
            self._delay_increased_dict = {}
            return

        # delay 증가가 발생한 시점별 delta 누적
        delay_increased_dict = defaultdict(float)

        for _, row in finished_df.iterrows():
            op = str(row['Operation'])
            t_val = float(row['Time'])
            if op not in op_due_map:
                continue  # due를 모르면 스킵
            due = float(op_due_map[op])
            if t_val > due:
                t_int = int(t_val)  # 시간 버킷
                delta = t_val - due
                delay_increased_dict[t_int] += float(delta)

        # 최대 시간 결정: 기존 maxtime이 있으면 그걸 이용, 없으면 로그 기반
        if self.move_timelog:
            max_t = self.maxtime
        else:
            max_t = int(pd.to_numeric(self._df['Time'], errors='coerce').max() or 0)
            max_t = max(0, max_t)

        # 초기화(정적 누적 형식)
        delay_timelog = {}
        cumulative_delay = 0.0
        for t in range(0, max_t + 1):
            cumulative_delay += delay_increased_dict.get(t, 0.0)
            delay_timelog[t] = float(cumulative_delay)

        self._delay_increased_dict = dict(delay_increased_dict)  # 원본 증가치 기록
        self.delay_timelog = delay_timelog

    def generate_priority_timelog(self):
        """
        각 Operation의 'Working Started' 시작시각들을 오름차순 정렬.
        - 시작시각이 1개뿐이면 weight = 1.0
        - 시작시각이 2개 이상이면
            w2 = (t2 - t1) / duration,
            w3 = (t3 - t2) / duration, ... 해당 시점에 부여
          그리고 첫 시작시각의 가중치 w1 = max(0, 1 - sum(w2..wn))
        각 행의 earned_point = grade_point[Info] * weight
        를 t=int(Time) 시각에 누적한다.
        """
        # 사전 조건
        if not hasattr(self, '_op_duration_map'):
            raise RuntimeError("먼저 attach_instance(instance_path)를 호출해 'ship' 정보를 붙이세요.")

        required_cols = {'Operation', 'Event', 'Time', 'Info'}
        if not required_cols.issubset(self._df.columns):
            missing = required_cols - set(self._df.columns)
            raise ValueError(f"로그에 필요한 컬럼이 없습니다: {missing}")

        grade_point = {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 0}

        df = self._df.copy()
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time'])
        df['Event'] = df['Event'].astype(str)
        df['Operation'] = df['Operation'].astype(str)
        df['Info'] = df['Info'].astype(str).str.strip().str.upper()

        started_df = df[df['Event'].eq('Working Started')]
        if started_df.empty:
            max_t = self.maxtime if self.move_timelog else int(
                pd.to_numeric(self._df['Time'], errors='coerce').max() or 0)
            self.priority_timelog = {t: 0.0 for t in range(0, max(max_t, 0) + 1)}
            self._priority_increased_dict = {}
            return

        # {time: earned_point} 증가 기록
        priority_increased_dict = defaultdict(float)

        # Operation 단위로 처리
        for op, grp in started_df.groupby('Operation'):
            # duration 가져오기 (없으면 스킵)
            duration = self._op_duration_map.get(op, None)
            if duration is None or duration <= 0:
                continue

            # 시작시각/Info 정렬
            g = grp[['Time', 'Info']].sort_values('Time').reset_index(drop=True)
            times = g['Time'].astype(float).tolist()
            infos = g['Info'].tolist()
            n = len(times)

            if n == 1:
                # 단일 시작: weight=1.0
                info = infos[0]
                if info in grade_point:
                    t_int = int(times[0])
                    earned = grade_point[info] * 1.0
                    priority_increased_dict[t_int] += float(earned)
                continue

            # n >= 2: 연속 차이 기반 가중치
            # w1..wn-1 먼저 계산
            weights = []
            for i in range(1, n):
                dt = max(0.0, times[i] - times[i - 1])  # 음수 방지
                w = dt / duration
                weights.append(w)

            # 첫 시작의 가중치는 남는 몫으로 설정(과잉 방지)
            wn = max(0.0, 1.0 - sum(weights))
            # print(op, ' | w:',weights, wn)
            # 첫 시작 시점 가점
            if infos[-1] in grade_point and wn > 0:
                t_int = int(times[-1])
                earned = grade_point[infos[-1]] * wn
                priority_increased_dict[t_int] += float(earned)

                # print(f'Point earned! Time:{t_int}, grade:{grade_point[infos[-1]]} weight:{wn}')

            # 이후 시작 시점 가점
            for i in range(n - 1):
                info_i = infos[i]
                if info_i not in grade_point:
                    continue
                w_i = weights[i]
                if w_i <= 0:
                    continue
                t_int = int(times[i])
                earned = grade_point[info_i] * w_i
                priority_increased_dict[t_int] += float(earned)
                # print(f'Point earned! Time:{t_int}, grade:{grade_point[info_i]} weight:{w_i}')

        # 누적 타임로그 생성
        if self.move_timelog:
            max_t = self.maxtime
        else:
            max_t = int(pd.to_numeric(self._df['Time'], errors='coerce').max() or 0)
            max_t = max(0, max_t)

        cumulative = 0.0
        priority_timelog = {}
        for t in range(0, max_t + 1):
            cumulative += priority_increased_dict.get(t, 0.0)
            priority_timelog[t] = float(cumulative)

        self._priority_increased_dict = dict(priority_increased_dict)
        self.priority_timelog = priority_timelog

class AgentCommunicator():
    def __init__(self):
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
                        priority_ratio = sum(env.monitor.priority_ratio.values()) / len(
                            env.monitor.priority_ratio.values())
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
            writer = pd.ExcelWriter(res_dir + f'\\log-{name}.xlsx')
            # df_delay.to_excel(writer, sheet_name="delay")
            # df_move.to_excel(writer, sheet_name="move")
            # df_priority.to_excel(writer, sheet_name="priority")
            # df_delay_cost.to_excel(writer,sheet_name="delay_cost")
            # df_move_cost.to_excel(writer, sheet_name="move_cost")
            # df_loss_cost.to_excel(writer, sheet_name="loss_cost")
            # df_computing_time.to_excel(writer, sheet_name="computing_time")
            env.get_logs().to_excel(writer, sheet_name="logs")
            writer.close()

if __name__=="__main__":

    # agentcommunicator = AgentCommunicator()

    cumulativeLogGenerator = CumulativeLogGenerator()
    cumulativeLogGenerator.save_logs()




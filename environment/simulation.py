import simpy
import pandas as pd


class Operation:
    def __init__(self, name, id, type, start_planned, finish_planned, duration, interruption, fixed_duration, priority_scores):
        self.name = name
        self.id = id
        self.type = type
        self.start_planned = start_planned
        self.finish_planned = finish_planned
        self.duration = duration
        self.interruption = interruption
        self.fixed_duration = fixed_duration
        self.priority_scores = priority_scores

        self.progress = 0.0
        self.start_actual = None
        self.finish_actual = None
        self.working_start = -1.0
        self.waiting_start = -1.0
        self.waiting_time_cum = 0.0
        self.current_quay = None

    def get_duration(self):
        if self.interruption == "S" or self.interruption == "F":
            if self.fixed_duration >= self.duration:
                duration = self.duration
            else:
                duration = self.duration - self.progress
        else:
            duration = self.duration

        return duration

    def get_priority_score(self, quay_id):
        priority_score = self.priority_scores[quay_id]
        return priority_score

    def check_interruption(self, time):
        if self.interruption == "S":
            if time - self.working_start + self.progress < self.duration - self.fixed_duration:
                possible = True
            else:
                possible = False
        elif self.interruption == "F":
            if time - self.working_start + self.progress > self.fixed_duration:
                possible = True
            else:
                possible = False
        else:
            possible = False

        return possible


class Ship:
    def __init__(self, name, id, type, category, launching_date, delivery_date, operations, initial_step, initial_quay=None):
        self.name = name
        self.id = id
        self.type = type
        self.category = category
        self.launching_date = launching_date
        self.delivery_date = delivery_date
        self.operations = operations
        self.initial_step = initial_step
        self.initial_quay = initial_quay

        self.step = initial_step
        self.current_quay = initial_quay

    def get_current_operation(self):
        if len(self.operations) == self.step:
            operation = None
        else:
            operation = self.operations[self.step]
        return operation


class Source:
    def __init__(self, env, name, ships, model, monitor):
        self.env = env
        self.name = name
        self.ships = ships
        self.model = model
        self.monitor = monitor

        self.calling_event = {}
        self.action = env.process(self.run())

    def run(self):
        for ship in self.ships:
            self.monitor.ships_before_LC[ship.id] = ship
            self.monitor.delay[ship.id] = 0
            self.monitor.move[ship.id] = 0
            self.monitor.priority_ratio[ship.id] = 0
            self.monitor.loss[ship.id] = 0
            self.monitor.sea[ship.id] = 0
            for i, operation in enumerate(ship.operations):
                if i < ship.step:
                    self.monitor.operations_done[operation.id] = operation
                else:
                    self.monitor.operations_unscheduled[operation.id] = operation

        while True:
            ship = self.ships.pop(0)
            operation = ship.get_current_operation()

            IAT = operation.start_planned - self.env.now
            if IAT > 0:
                yield self.env.timeout(IAT)

            if ship.initial_quay is not None:
                self.model[ship.initial_quay].put(ship)
                if ship.initial_quay == "Buffer":
                    self.monitor.reward_delay[ship.id] = self.env.now
                del self.monitor.operations_unscheduled[operation.id]
                del self.monitor.ships_before_LC[ship.id]
                self.monitor.ships_in_process[ship.id] = ship
            else:
                self.env.process(self.launching(ship))

            if len(self.ships) == 0:
                break

    def launching(self, ship):
        operation = ship.get_current_operation()
        self.monitor.put_queue(ship)
        self.calling_event[ship.name] = self.env.event()
        next_quay = yield self.calling_event[ship.name]
        next_quay_id = self.model[next_quay].id if next_quay != "Buffer" else None

        interruption = False
        if next_quay != "Buffer":
            occupied, interruption = self.model[next_quay].check_status(self.env.now)
            if interruption:
                for process in self.model[next_quay].processes.values():
                    process.interrupt()
        duration = operation.get_duration()
        if next_quay_id is not None:
            priority_score = operation.get_priority_score(next_quay_id)
        else:
            if type(operation.priority_scores[0]) != str:
                priority_score = 0.0
            else:
                priority_score = "N"

        if next_quay == "Buffer":
            self.monitor.reward_delay[ship.id] = self.env.now
        self.monitor.reward_move[ship.id] = interruption
        self.monitor.reward_priority[ship.id] = (priority_score, duration)

        # self.monitor.move[ship.id] += 1

        self.model[next_quay].put(ship)
        del self.calling_event[ship.name]
        del self.monitor.operations_unscheduled[operation.id]
        del self.monitor.ships_before_LC[ship.id]
        self.monitor.ships_in_process[ship.id] = ship

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, ship=ship.name, event="Ship Launched")
            self.monitor.record(self.env.now, ship=ship.name, event="Ship Moved")


class Quay:
    def __init__(self, env, name, id, model, monitor, capacity=1):
        self.env = env
        self.name = name
        self.id = id
        self.model = model
        self.monitor = monitor
        self.capacity = capacity

        self.calling_event = {}
        self.operations_in_working = {}
        self.processes = {}

        self.utilization = 0.0

    def put(self, ship):
        operation = ship.get_current_operation()
        ship.current_quay = self.name
        operation.current_quay = self.name
        self.processes[operation.id] = self.env.process(self.working(ship))
        self.operations_in_working[operation.id] = operation
        self.monitor.operations_in_quay[operation.id] = operation

    def working(self, ship):
        operation = ship.get_current_operation()
        operation.waiting_start = -1

        if self.name != "S":
            priority_score = operation.get_priority_score(self.id)
        else:
            priority_score = None
        duration = operation.get_duration()
        working_start = self.env.now
        operation.working_start = working_start
        if operation.start_actual is None:
            operation.start_actual = working_start

        try:
            if self.monitor.record_events:
                self.monitor.record(self.env.now, location=self.name, ship=ship.name,
                                    operation=operation.name, event="Working Started", info=priority_score)
            yield self.env.timeout(duration)
        except simpy.Interrupt as i:
            if self.monitor.record_events:
                self.monitor.record(self.env.now, location=self.name, ship=ship.name,
                                    operation=operation.name, event="Working Interrupted", info=priority_score)
            self.utilization += (self.env.now - working_start)
            operation.progress += (self.env.now - working_start)
            self.monitor.operations_interrupted[operation.id] = operation
            interrupted = True
        else:
            if self.monitor.record_events:
                self.monitor.record(self.env.now, location=self.name, ship=ship.name,
                                    operation=operation.name, event="Working Finished", info=priority_score)
            self.utilization += duration
            operation.progress += duration
            ship.step += 1
            if operation.finish_actual is None:
                operation.finish_actual = self.env.now
            self.monitor.operations_done[operation.id] = operation
            interrupted = False

        del self.processes[operation.id]
        del self.operations_in_working[operation.id]
        del self.monitor.operations_in_quay[operation.id]

        if self.name != "S":
            if (type(priority_score) == str and (priority_score == "A" or priority_score == "B")) or \
                    (type(priority_score) != str and priority_score >= 0.7):
                self.monitor.priority_ratio[ship.id] += (self.env.now - working_start)
            else:
                self.monitor.loss[ship.id] += (self.env.now - working_start)
        else:
            self.monitor.sea[ship.id] += (self.env.now - working_start)

        if ship.step == len(ship.operations):
            self.model["Sink"].put(ship)
            # self.monitor.move[ship.id] += 1
            if len(self.monitor.operations_in_buffer) > 0:
                self.monitor.scheduling = True
            if self.monitor.record_events:
                self.monitor.record(self.env.now, ship=ship.name, event="Ship Moved")
        else:
            operation = ship.get_current_operation()
            # if operation.type == "시운전" or operation.type == "G/T":
            if "시운전" in operation.type or "G/T" in operation.type:
                self.model["S"].put(ship)
                # self.monitor.move[ship.id] += 1
                if len(self.monitor.operations_in_buffer) > 0:
                    self.monitor.scheduling = True
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, ship=ship.name, event="Ship Moved")
            else:
                decision_making = False
                if self.name == "S":
                    decision_making = True
                elif interrupted:
                    decision_making = True
                else:
                    priority_score = operation.get_priority_score(self.id)
                    if ((type(priority_score) == str and priority_score == "N")
                            or (type(priority_score) != str and priority_score == 0.0)):
                        decision_making = True
                    else:
                        if len(self.monitor.operations_in_buffer) > 0:
                            decision_making = True

                if decision_making:
                    self.monitor.put_queue(ship)
                    self.calling_event[ship.name] = self.env.event()
                    next_quay = yield self.calling_event[ship.name]
                    next_quay_id = self.model[next_quay].id if next_quay != "Buffer" else None

                    interruption = False
                    if next_quay != "Buffer":
                        occupied, interruption = self.model[next_quay].check_status(self.env.now)
                        if interruption:
                            for process in self.model[next_quay].processes.values():
                                process.interrupt()
                    duration = operation.get_duration()
                    if next_quay_id is not None:
                        priority_score = operation.get_priority_score(next_quay_id)
                    else:
                        if type(operation.priority_scores[0]) != str:
                            priority_score = 0.0
                        else:
                            priority_score = "N"

                    if next_quay == "Buffer":
                        self.monitor.reward_delay[ship.id] = self.env.now
                    self.monitor.reward_move[ship.id] = interruption
                    self.monitor.reward_priority[ship.id] = (priority_score, duration)

                    if self.id != next_quay_id and self.name != "S":
                        self.monitor.move[ship.id] += 1

                    del self.calling_event[ship.name]
                    self.model[next_quay].put(ship)

                    if self.monitor.record_events:
                        if self.id != next_quay_id:
                            self.monitor.record(self.env.now, ship=ship.name, event="Ship Moved")
                else:
                    self.model[self.name].put(ship)

            if interrupted:
                del self.monitor.operations_interrupted[operation.id]
            else:
                del self.monitor.operations_unscheduled[operation.id]

    def check_status(self, time):
        if len(self.operations_in_working) == self.capacity:
            occupied = True
            interruption = False
            for operation in self.operations_in_working.values():
                if operation.check_interruption(time):
                    interruption = True
        else:
            occupied = False
            interruption = False

        return occupied, interruption


class Buffer:
    def __init__(self, env, name, model, monitor, capacity=float('inf')):
        self.env = env
        self.name = name
        self.model = model
        self.monitor = monitor
        self.capacity = capacity

        self.calling_event = {}
        self.operations_in_waiting = {}
        self.processes = {}

    def put(self, ship):
        operation = ship.get_current_operation()
        ship.current_quay = self.name
        operation.current_quay = self.name
        self.processes[operation.id] = self.env.process(self.waiting(ship))
        self.operations_in_waiting[operation.id] = operation
        self.monitor.operations_in_buffer[operation.id] = operation

    def waiting(self, ship):
        operation = ship.get_current_operation()

        self.monitor.put_queue(ship, from_buffer=True)
        waiting_start = self.env.now
        operation.waiting_start = waiting_start
        if operation.start_actual is None:
            operation.start_actual = waiting_start

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name,
                                ship=ship.name, operation=operation.name, event="Waiting Started")
        self.calling_event[ship.name] = self.env.event()
        next_quay = yield self.calling_event[ship.name]
        next_quay_id = self.model[next_quay].id if next_quay != "Buffer" else None

        self.monitor.sea[ship.id] += (self.env.now - waiting_start)

        interruption = False
        if next_quay != "Buffer":
            occupied, interruption = self.model[next_quay].check_status(self.env.now)
            if interruption:
                for process in self.model[next_quay].processes.values():
                    process.interrupt()
        duration = operation.get_duration()
        if next_quay_id is not None:
            priority_score = operation.get_priority_score(next_quay_id)
        else:
            if type(operation.priority_scores[0]) != str:
                priority_score = 0.0
            else:
                priority_score = "N"

        if next_quay == "Buffer":
            self.monitor.reward_delay[ship.id] = self.env.now
        else:
            del self.monitor.reward_delay[ship.id]
        self.monitor.reward_move[ship.id] = interruption
        self.monitor.reward_priority[ship.id] = (priority_score, duration)

        self.monitor.delay[ship.id] += (self.env.now - operation.waiting_start)
        if next_quay_id is not None:
            self.monitor.move[ship.id] += 1

        del self.calling_event[ship.name]
        del self.processes[operation.id]
        del self.operations_in_waiting[operation.id]
        del self.monitor.operations_in_buffer[operation.id]
        operation.waiting_time_cum += self.env.now - operation.waiting_start

        self.model[next_quay].put(ship)

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name,
                                ship=ship.name, operation=operation.name, event="Waiting Finished")

            if next_quay_id is not None:
                self.monitor.record(self.env.now, ship=ship.name, event="Ship Moved")


class Sink:
    def __init__(self, env, name, monitor):
        self.env = env
        self.name = name
        self.monitor = monitor

        self.event = env.event()
        self.ships = []

    def put(self, ship):
        self.ships.append(ship)
        self.monitor.ships_after_DL[ship.id] = ship
        del self.monitor.ships_in_process[ship.id]

        self.monitor.priority_ratio[ship.id] /= (self.env.now - ship.launching_date - self.monitor.sea[ship.id])

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, ship=ship.name, event="Ship Delivered")


class Monitor:
    def __init__(self, record_events=True):
        self.record_events = record_events

        self.ships_in_queue = {}
        self.scheduling = False

        # self.reward_info = {"move": False, "priority": "A", "delay": False}
        self.reward_delay = {}
        self.reward_move = {}
        self.reward_priority = {}

        self.delay = {}
        self.move = {}
        self.priority_ratio = {}
        self.loss = {}
        self.sea = {}

        # information used for constructing the states
        self.operations_unscheduled = {}
        self.operations_in_quay = {}
        self.operations_in_buffer = {}
        self.operations_interrupted = {}
        self.operations_done = {}

        self.ships_before_LC = {}
        self.ships_in_process = {}
        self.ships_after_DL = {}

        self.time = []
        self.location = []
        self.ship = []
        self.operation = []
        self.event = []
        self.info = []

    def put_queue(self, ship, from_buffer=False):
        if not self.scheduling and not from_buffer:
            self.scheduling = True
        self.ships_in_queue[ship.id] = ship

    def remove_queue(self, ship_id):
        ship = self.ships_in_queue[ship_id]
        del self.ships_in_queue[ship_id]

        operations_for_decision = []
        for temp in self.ships_in_queue.values():
            operation = temp.get_current_operation()
            if not operation.id in self.operations_in_buffer.keys():
                operations_for_decision.append(operation.id)

        if len(operations_for_decision) == 0:
            self.scheduling = False

        return ship

    def record(self, time, location=None, ship=None, operation=None, event=None, info=None):
        self.time.append(time)
        self.location.append(location)
        self.ship.append(ship)
        self.operation.append(operation)
        self.event.append(event)
        self.info.append(info)

    def get_logs(self, file_path=None):
        df_log = pd.DataFrame(columns=['Time', 'Location', 'Ship', 'Operation', 'Event', 'Info'])
        df_log['Time'] = self.time
        df_log['Location'] = self.location
        df_log['Ship'] = self.ship
        df_log['Operation'] = self.operation
        df_log['Event'] = self.event
        df_log['Info'] = self.info

        if file_path is not None:
            df_log.to_excel(file_path, index=False)

        return df_log
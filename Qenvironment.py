import math
import torch
from missile import Missile
import random
from Qtorch import QLearning
import numpy as np
MISSILE_STATUS = {0: "Flying", 1: "Locked", 2: "Intercepted", 3: "Escaped"}
max_lock_missiles = 10 # 每轮锁定导弹的上限

class Environment:
    def __init__(self, missile_list, q_learning=None):
        self.missile_list = missile_list
        self.current_round = 0
        self.intercepted_missiles = []
        self.locked_missiles = []
        self.escaped_missiles = []
        self.num_states = len(missile_list)
        self.total_reward = 0  # 用于存储累计奖励
        self.q_learning = q_learning

    def get_missile_list(self):
        return self.missile_list
    def get_current_round(self):
        return self.current_round
    def set_current_round(self, round):
        self.current_round = round

    def launch_missiles(self):
        game_duration = 16000
        rewards = []  # 创建一个奖励列表
        total_reward = 0
        for elapsed_time in range(game_duration):
            self.update_missile_positions()
            self.update_missile_statuses()
            self.lock_missiles()

             # 在每个回合的开始时检查导弹是否被拦截
            for missile in self.missile_list:
                if missile.get_status() == 1 and self.current_round >= missile.get_intercept_round():
                    distance_to_origin = math.sqrt(missile.x ** 2 + missile.y ** 2)
                    if distance_to_origin < 2:
                        missile.set_status(3)
                    else:
                        missile.set_status(2)
            # Q-learning 策略选择动作
            if self.q_learning is not None and elapsed_time % 8 == 7:
                self.lock_missiles()
                state = [self.current_round] + [m.get_status() for m in self.missile_list]
                actions = [str(m.number) for m in self.missile_list if m.get_status() == 0]
                action = self.q_learning.choose_action(state, actions)
                if action is None:  # 检查 action 是否为 None
                    continue  # 如果为 None，则跳过当前迭代
                action_missile = next((m for m in self.missile_list if m.number == int(action) and m.get_status() == 0), None)
                if action_missile is not None:
                    distance_to_origin = math.sqrt(action_missile.x ** 2 + action_missile.y ** 2)
                    if distance_to_origin < 2:
                        action_missile.set_status(3)
                        q_reward = self.calculate_reward(action_missile) # 使用 calculate_reward 方法计算奖励
                    else:
                        if action_missile.height == 1:
                            q_success_rate = 0.7
                        elif action_missile.height == 2:
                            q_success_rate = 0.8 
                        else:
                            q_success_rate = 0.9
                        if random.random() < q_success_rate:
                            action_missile.set_status(2)
                            if action_missile not in self.intercepted_missiles:
                                self.intercepted_missiles.append(action_missile)
                        else:
                            action_missile.set_status(0)  # 导弹拦截失败，状态转为0
                        q_reward = self.calculate_reward(action_missile) # 使用 calculate_reward 方法计算奖励
                    rewards.append(q_reward)  # 将单个导弹的奖励添加到列表中
                    total_reward += q_reward  # 更新总奖励
                    print(f"Round {self.current_round}: Total reward = {total_reward}")  # 在每个回合打印总奖励
                    next_state = [self.current_round + 1] + [m.get_status() for m in self.missile_list]
                    next_available_actions = [str(m.number) for m in self.missile_list if m.get_status() == 0]
                    self.q_learning.update_table(state, action, q_reward, next_state, next_available_actions=next_available_actions)
            self.current_round += 1
            # 判断游戏是否结束
            if self.is_end():
                break 
        return rewards  # 游戏结束时返回 rewards 列表

    def get_total_reward(self):
        return self.total_reward
    
    def calculate_reward(self, missile):
        reward = 0
        height = missile.get_height()
        status = missile.get_status()
        if status == 2 and height == 1:
            reward += 3
        elif status == 2 and height == 2:
            reward += 2
        elif status == 2 and height == 3:
            reward += 1
        elif status == 3:
            reward -= 10
        else:  # 拦截失败
            reward -= 1
        return reward
    
    def update_missile_positions(self):
        active_missiles = [missile for missile in self.missile_list if missile.get_status() in [0, 1] and missile.launch_round <= self.current_round]
        if not active_missiles:
            return
        positions = torch.tensor([[float(m.x), float(m.y)] for m in self.missile_list], dtype=torch.float)
        positions = positions.float()
        distances = torch.norm(positions, dim=1).float()
        directions = - positions / distances.view(-1, 1)
        new_positions = positions + directions * torch.tensor([float(m.speed) for m in self.missile_list]).view(-1, 1)
        for i, missile in enumerate(self.missile_list):
            if missile.get_status() in [0, 1] and missile.launch_round <= self.current_round:
                missile.set_position(new_positions[i][0].item(), new_positions[i][1].item())
                missile.distance = distances[i].item()

    def lock_missiles(self, actions=None):
        active_missiles = [missile for missile in self.missile_list if
                           missile.get_status() == 0 and missile.launch_round <= self.current_round]
        if not active_missiles:
            return
        if self.q_learning is not None and actions is not None:
            if not actions:
                return
            num_missiles_to_lock = min(len(actions), max_lock_missiles)
            chosen_actions = []
            for _ in range(num_missiles_to_lock):
                state = (self.current_round, len(active_missiles), len(actions))
                action = self.q_learning.choose_action(state, actions)
                chosen_actions.append(action)
                actions.remove(action)

            for action in chosen_actions:
                missile_number = int(action)
                missile = next(
                    (m for m in self.missile_list if m.number == missile_number), None)
                if missile is not None and missile.get_status() == 0:
                    distance_to_origin = math.sqrt(
                        missile.x ** 2 + missile.y ** 2)
                    if distance_to_origin < 2:
                        missile.set_status(3)
                    else:
                        missile.set_status(1)
                    intercept_time = distance_to_origin / (2 * missile.speed)
                    intercept_round = int(intercept_time)
                    missile.set_intercept_round(
                        self.current_round + intercept_round)
                    
                    # 更新Q值
                    if missile.get_status() == 1:
                        state = (self.current_round, len(active_missiles), len(actions) + 1)
                        next_state = (self.current_round, len(active_missiles) - 1, len(actions))
                        next_available_actions = actions.copy()
                        next_available_actions.append(action)
                        reward = self.get_intercept_success_rate(missile) - 1
                        self.q_learning.update_table(state, action, reward, next_state, next_available_actions)

    def update_missile_statuses(self):
        for missile in self.missile_list:
            if missile.get_status() == 0:
                distance_to_origin = math.sqrt(missile.x ** 2 + missile.y ** 2)
                if distance_to_origin < 2:
                    missile.set_status(3)
            elif missile.get_status() == 1:
                distance_to_origin = math.sqrt(missile.x ** 2 + missile.y ** 2)
                intercept_round = int(distance_to_origin / (2 * missile.speed))
                missile.set_intercept_round(intercept_round)
                if self.current_round >= missile.launch_round + intercept_round:
                    if random.random() < self.get_intercept_success_rate(missile):
                        missile.set_status(2)
                        if missile not in self.intercepted_missiles:
                            self.intercepted_missiles.append(missile)
                    else:
                        missile.set_status(3)
                elif missile.get_status() == 2:
                    if missile not in self.intercepted_missiles:
                        self.intercepted_missiles.append(missile)

    def get_intercept_success_rate(self, missile):
        if missile.height == 1:
            return 0.7
        elif missile.height == 2:
            return 0.8
        else:
            return 0.9

    def remove_missiles(self, q_learning):
        intercepted_missiles = []
        escaped_missiles = []
        locked_missiles = []
        for missile in self.missile_list:
            if missile.get_status() == 2:  # Intercepted
                intercepted_missiles.append(missile)
            elif missile.get_status() == 3:  # Escaped
                escaped_missiles.append(missile)
            elif missile.get_status() == 1:  # Locked
                locked_missiles.append(missile)
        return self.missile_list, intercepted_missiles, escaped_missiles, locked_missiles

    def is_end(self):
        if all(missile.get_status() in [2, 3] for missile in self.missile_list):
            return True
        if self.get_current_round() > 20:
            return True
        return False

    def reset(self, missile_list):
        self.missile_list = missile_list
        self.score = 0
        self.locked_missiles = []
        self.intercepted_missiles = []
        self.escaped_missiles = []
        self.current_round = 0
        return self.get_state()

    def get_state(self):
        state = []
        for missile in self.missile_list:
            state.append(missile.status)
        return tuple(state)

    def get_intercepted_missiles(self):
        return self.intercepted_missiles

    def get_available_actions(self):
        actions = []
        for missile in self.missile_list:
            if missile.get_status() == 0:
                actions.append(str(missile.number))
        return actions
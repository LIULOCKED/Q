import csv
import math
import time
import numpy as np
from Qenvironment import Environment
from missile import Missile
from Qtorch import QLearning 
import torch
import datetime
from missile import Missile, generate_missile_list, save_missile_list_to_csv

MISSILE_STATUS = {0: "Flying", 1: "Locked", 2: "Intercepted", 3: "Escaped"}
max_lock_missiles = 10  # 每轮锁定导弹的上限
MAX_ROUNDS = 20  # 最大游戏轮次
LEARNING_RATE = 0.1  # 学习率
DISCOUNT_FACTOR = 0.9  # 折扣因子
EXPLORATION_RATE = 0.2  # 探索率

# 从CSV文件中读取导弹列表
def read_missile_list(file_path):
    missile_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头行
        for row in reader:
            missile = Missile.from_csv_row(row)
            missile_list.append(missile)
    return missile_list

# 训练函数，包括生成导弹列表、初始化环境、Q学习实例等
def train(loop_index, env):
    # 检查是否使用GPU加速
    if torch.cuda.is_available():
        print("Using GPU acceleration.")
        device = torch.device("cuda")
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    num_actions = len(env.get_available_actions())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPLORATION_MIN = 0.1  # 添加 EXPLORATION_MIN
    EXPLORATION_MAX = 1.0  # 添加 EXPLORATION_MAX
    q_learning = QLearning(num_actions, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_MIN, EXPLORATION_MAX, device)  # 添加 device 参数

    # 开始训练循环
    for i in range(MAX_ROUNDS):
        print(f"Training round {i + 1}/{MAX_ROUNDS}")
        # 发射导弹并获取总奖励
        rewards = env.launch_missiles()
        available_actions = env.get_available_actions()
        # 修改状态表示以包含导弹位置、高度和状态
        state = [env.current_round, 0, 0]
        # 在循环之前初始化 action 变量
        action = None
        for missile in env.missile_list:
            if missile.get_status() == 0:
                state[1] = missile.distance
                state[2] = missile.height
                action = q_learning.choose_action(state, available_actions)
                missile_index = int(action)
                missile.set_number(missile_index)
                missile.set_status(1)
                available_actions.remove(action)
        # 修改下一个状态表示以包含导弹位置、高度和状态
        next_state = [env.current_round + 1, 0, 0]
        for missile in env.missile_list:
            if missile.get_status() == 0:
                next_state[1] = missile.distance
                next_state[2] = missile.height
                break
        if action is not None:
            q_learning.update_table(state, action, sum(rewards), next_state, env.get_available_actions())
        # 更新游戏轮次
        env.set_current_round(env.current_round + 1)
        rewards = env.launch_missiles()
    # 训练结束，输出最终结果
    missile_list, intercepted_missiles, escaped_missiles, locked_missiles = env.remove_missiles(q_learning)  # 修改为 q_learning 实例
    score = env.get_total_reward()
    
    print(f"Final score: {score}")
    print(f"missile_list: {[missile.status for missile in missile_list]}")
    print(f"Intercepted missiles: {[missile.number for missile in intercepted_missiles]}")
    print(f"Escaped missiles: {[missile.number for missile in escaped_missiles]}")
    print(f"locked_missiles: {[missile.number for missile in locked_missiles]}")
    # 保存模型
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_file_path = f"model/model_{env.get_current_round()}_{current_time}.pt"
    torch.save(q_learning.q_network.state_dict(), model_file_path)
    print(f"Model saved to {model_file_path}")

if __name__ == '__main__':
    filename = None
    env = Environment([])  # 在训练循环之前创建 env 实例
    for i in range(20):
        print(f"Training loop {i + 1}")

        # 每100次训练更换一次CSV文件
        if i % 10 == 0:
            num_groups = 5
            num_missiles = 120
            missile_list = generate_missile_list(num_missiles, num_groups, i // 100)  # 更新file_index的计算
            filename = f'missiles_{i // 10 + 1}.csv'  # 修改文件名生成规则
            save_missile_list_to_csv(missile_list, filename, i // 100)  # 更新file_index的计算
            print(f"Using CSV file: {filename}")  # 将打印使用的CSV文件名移动到此处

        missile_list_copy = [missile.clone() for missile in missile_list]  # 创建导弹列表的深拷贝
        env.reset(missile_list_copy)  # 重置环境状态
        train(i, env)
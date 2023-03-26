import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
MAX_ROUNDS = 100  # 最大游戏轮次
class QNetwork(nn.Module):
    # 包含三个线性层的简单前馈神经网络。
    # 它有一个输入层（大小为 num_inputs，本例中为1），
    # 一个输出层（大小为 num_actions，表示每个动作的 Q 值），
    # 以及两个隐藏层（大小分别为64和32）。
    # ReLU激活函数用于输入层和第一个隐藏层之间以及第一个和第二个隐藏层之间。

    def __init__(self, num_inputs, num_actions, device):
        super(QNetwork, self).__init__()
        self.device = device  # 将 device 参数存储为实例变量
        self.fc1 = nn.Linear(num_inputs, 64).to(self.device)  # 将 device 参数传递给每个层
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, num_actions).to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearning:
    def __init__(self, num_actions, learning_rate, discount_factor, exploration_min, exploration_max, device):
        self.device = device #用于运行神经网络的设备（CPU或GPU）
        self.num_actions = num_actions #可执行的动作数量
        self.learning_rate = learning_rate #神经网络的学习率
        self.discount_factor = discount_factor #折扣因子，用于计算未来奖励的折扣值
        self.exploration_rate = exploration_max #探索率的最大值
        self.exploration_min = exploration_min #探索率的最小值
        self.exploration_decay = (exploration_max - exploration_min) / 500 #探索衰减值
        num_inputs = 3  # 根据您的环境设置输入数量 设置为1，表示输入向量的大小
        self.q_network = QNetwork(num_inputs, num_actions, device).to(device)  # 将 device 参数传递给 QNetwork 类
        self.target_network = QNetwork(num_inputs, num_actions, device).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=5000)
        self.alpha = 0.1
        self.gamma = 0.9
        self.loss_fn = nn.MSELoss().to(self.device)  # 定义损失函数为均方损失，并将其分配给设备

    def choose_action(self, state, available_actions):
        if len(available_actions) == 0:  # 检查 available_actions 是否为空
            return None  # 如果为空，则返回 None 作为默认动作
        if random.random() < self.exploration_rate:
            action = random.choice(available_actions)
        else:
            state_tensor = torch.tensor([state[0], state[1], state[2]], dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            available_actions_tensor = torch.tensor([int(a) for a in available_actions if int(a) < self.num_actions])
            action_values = q_values[:, available_actions_tensor]
            max_actions = available_actions_tensor[action_values == torch.max(action_values)].cpu().numpy()
            action = str(np.random.choice(max_actions))
        return action

    def update_table(self, state, action, reward, next_state, next_available_actions):
        state_tensor = torch.tensor([state[0], state[1], state[2]], dtype=torch.float32).to(self.device).unsqueeze(0)
        action_tensor = torch.tensor(int(action), dtype=torch.int64).to(self.device).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor([next_state[0], next_state[1], next_state[2]], dtype=torch.float32).to(self.device).unsqueeze(0)
        next_available_actions_tensor = torch.tensor([int(a) for a in next_available_actions if int(a) < self.num_actions])
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)
        
        # 计算目标 Q 值
        max_next_q_value = torch.max(next_q_values[:, next_available_actions_tensor])
        target_q_value = reward_tensor + self.gamma * max_next_q_value
        target_q_values = q_values.clone().detach()
        target_q_values[0, action_tensor] = target_q_value

        # 计算损失并更新 Q-network
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
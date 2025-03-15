import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import math
import os
from agent import QLearningAgent

# 定义经验元组，用于优先经验回放
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class NoisyLinear(nn.Module):
    """噪声线性层，用于实现噪声网络探索"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习的参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """生成缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class RainbowDQNNetwork(nn.Module):
    """Rainbow DQN网络，支持分布式DQN（C51）和噪声网络"""
    def __init__(self, input_size, hidden_size, output_size, atom_size=51, support_min=-10, support_max=10, noisy=True):
        super(RainbowDQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.atom_size = atom_size
        
        # 设置分布式DQN的支持范围
        self.support = torch.linspace(support_min, support_max, atom_size)
        
        # 根据是否使用噪声网络选择不同的线性层
        if noisy:
            self.fc1 = NoisyLinear(input_size, hidden_size)
            self.fc2 = NoisyLinear(hidden_size, hidden_size)
            self.fc3 = NoisyLinear(hidden_size, output_size * atom_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size * atom_size)
        
        self.noisy = noisy
    
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 重塑输出以获取每个动作的分布
        x = x.view(-1, self.output_size, self.atom_size)
        x = F.softmax(x, dim=2)  # 对每个动作的分布应用softmax
        
        return x
    
    def reset_noise(self):
        """重置所有噪声层的噪声"""
        if self.noisy:
            self.fc1.reset_noise()
            self.fc2.reset_noise()
            self.fc3.reset_noise()

class PrioritizedReplayBuffer:
    """优先经验回FFER区"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样指数
        self.beta_increment = beta_increment  # beta的增量
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # 初始最大优先级
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # 新经验的优先级设为最大优先级
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """根据优先级采样经验"""
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities
        
        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # 计算重要性权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        weights = torch.FloatTensor(weights)
        
        # 获取经验
        experiences = [self.buffer[idx] for idx in indices]
        
        # 使用numpy.array()预处理列表，避免慢速的逐个转换
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class RainbowDQNAgent(QLearningAgent):
    """Rainbow DQN代理，继承自QLearningAgent"""
    def __init__(self, game_size=4, learning_rate=0.0003, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        # 不调用父类的__init__，而是重新实现
        self.game_size = game_size
        self.state_size = game_size * game_size
        self.action_size = 4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.batch_size = 64
        self.model_path = 'rainbow_dqn_agent_model.pth'
        
        # Rainbow DQN特有参数
        self.n_step = 3  # 多步学习步数
        self.atom_size = 51  # 分布式DQN的原子数
        self.v_min = -10  # 价值分布的最小值
        self.v_max = 10  # 价值分布的最大值
        # 检查是否有可用的GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建支持范围并移动到正确的设备上
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)  # 支持范围
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)  # 原子间距
        self.use_noisy_nets = True  # 是否使用噪声网络
        self.target_update_freq = 10  # 目标网络更新频率
        
        # 创建神经网络模型
        self.model = RainbowDQNNetwork(
            self.state_size, 128, self.action_size, 
            self.atom_size, self.v_min, self.v_max, self.use_noisy_nets
        ).to(self.device)
        
        self.target_model = RainbowDQNNetwork(
            self.state_size, 128, self.action_size, 
            self.atom_size, self.v_min, self.v_max, self.use_noisy_nets
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
        # 使用优先经验回放
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        
        # 多步学习缓冲区
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # 如果存在已保存的模型，则加载
        if os.path.exists(self.model_path):
            self.load_model()
    
    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_state(self, state):
        """预处理状态为神经网络输入"""
        # 对数变换，处理大数值
        log_state = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])
        # 归一化
        if np.max(log_state) > 0:
            log_state = log_state / 11.0  # 2048 = 2^11
        return log_state.flatten()
    
    def _get_n_step_info(self):
        """获取多步学习的信息"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            
            reward = r + self.discount_factor * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return reward, next_state, done
    
    def remember(self, state, action, reward, next_state, done):
        """将经验存储到回FFER区（支持多步学习）"""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        
        # 将经验添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区已满，处理多步学习
        if len(self.n_step_buffer) == self.n_step:
            # 获取多步学习的奖励和下一个状态
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            state = self.n_step_buffer[0][0]  # 初始状态
            action = self.n_step_buffer[0][1]  # 初始动作
            
            # 将多步经验添加到优先经验回放缓冲区
            self.memory.add(state, action, n_step_reward, n_step_next_state, n_step_done)
    
    def choose_action(self, state, available_actions):
        """选择动作（探索或利用）"""
        if not available_actions:
            return None
        
        # 如果使用噪声网络，不需要额外的探索策略
        if self.use_noisy_nets:
            # 直接使用当前策略选择动作
            state = self.preprocess_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # 获取分布式DQN的动作分布
                dist = self.model(state_tensor)
                # 计算每个动作的期望值
                q_values = torch.sum(dist * self.support.expand_as(dist), dim=2).cpu().numpy()[0]
            
            # 屏蔽不可用的动作
            masked_q_values = q_values.copy()
            for i in range(4):
                if i not in available_actions:
                    masked_q_values[i] = float('-inf')
            
            return np.argmax(masked_q_values)
        else:
            # 使用传统的epsilon-greedy策略
            if random.random() < self.exploration_rate:
                return random.choice(available_actions)
            
            state = self.preprocess_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist = self.model(state_tensor)
                q_values = torch.sum(dist * self.support.expand_as(dist), dim=2).cpu().numpy()[0]
            
            # 屏蔽不可用的动作
            masked_q_values = q_values.copy()
            for i in range(4):
                if i not in available_actions:
                    masked_q_values[i] = float('-inf')
            
            return np.argmax(masked_q_values)
    
    def _projection_distribution(self, next_dist, rewards, dones):
        """计算分布式DQN的目标分布（向量化实现）"""
        batch_size = rewards.size(0)
        
        # 支持范围
        support = self.support.expand_as(next_dist)
        
        # 计算目标值 r + gamma * z
        target_z = rewards.unsqueeze(1) + self.discount_factor * support * (1 - dones).unsqueeze(1)
        target_z = target_z.clamp(min=self.v_min, max=self.v_max)  # 裁剪到支持范围内
        
        # 计算投影的bin索引
        b = (target_z - self.v_min) / self.delta_z
        l = b.floor().long()  # 下界
        u = b.ceil().long()   # 上界
        
        # 处理边界情况
        l[(l > 0) * (l == u)] -= 1
        u[(u < (self.atom_size - 1)) * (l == u)] += 1
        
        # 初始化投影目标
        proj_dist = torch.zeros(batch_size, self.action_size, self.atom_size).to(self.device)
        
        # 使用向量化操作替代嵌套循环
        # 如果next_dist只有一个动作维度，则复制到所有动作
        if next_dist.size(1) == 1:
            next_dist = next_dist.expand(-1, self.action_size, -1)
        
        # 对每个批次样本处理
        for i in range(batch_size):
            # 计算投影权重并直接更新proj_dist
            for j in range(self.action_size):
                # 计算上下界的权重
                weight_l = (u[i, j] - b[i, j])
                weight_u = (b[i, j] - l[i, j])
                
                # 获取索引并确保它们是标量
                for k in range(self.atom_size):
                    # 获取当前原子的索引
                    idx_l = l[i, j, k].item()
                    idx_u = u[i, j, k].item()
                    
                    # 确保索引在有效范围内
                    if 0 <= idx_l < self.atom_size:
                        # 更新下界
                        proj_dist[i, j, idx_l] += next_dist[i, j, k] * weight_l[k]
                    if 0 <= idx_u < self.atom_size:
                        # 更新上界
                        proj_dist[i, j, idx_u] += next_dist[i, j, k] * weight_u[k]
        
        return proj_dist
    
    def replay(self):
        """从经验回FFER区中批量学习"""
        if len(self.memory.buffer) < self.batch_size:
            return
        
        # 从优先经验回FFER区中采样
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # 将数据转移到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # 计算当前分布
        current_dist = self.model(states)
        # 获取每个状态对应动作的分布
        log_p = torch.log(current_dist[range(self.batch_size), actions.long()])
        
        with torch.no_grad():
            # 双重DQN：使用当前网络选择动作，使用目标网络评估
            next_dist = self.target_model(next_states)
            next_action = torch.sum(next_dist * self.support.expand_as(next_dist), dim=2).argmax(1)
            next_dist = next_dist[range(self.batch_size), next_action]
            
            # 计算目标分布
            target_dist = self._projection_distribution(next_dist.unsqueeze(1), rewards, dones)
        
        # 获取每个状态对应动作的目标分布
        target_dist = target_dist[range(self.batch_size), actions.long()]
        
        # 计算KL散度损失
        loss = -torch.sum(target_dist * log_p, dim=1)
        
        # 应用重要性采样权重
        weighted_loss = (loss * weights).mean()
        
        # 更新优先级
        priorities = loss.detach().cpu().numpy() + 1e-6  # 添加小值防止优先级为0
        self.memory.update_priorities(indices, priorities)
        
        # 优化模型
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        # 重置噪声网络的噪声
        if self.use_noisy_nets:
            self.model.reset_noise()
            self.target_model.reset_noise()
        
        return weighted_loss.item()
    
    def decay_exploration(self):
        """衰减探索率（仅在不使用噪声网络时有效）"""
        if not self.use_noisy_nets and self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def save_model(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate
        }, self.model_path)
    
    def load_model(self):
        """加载模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.exploration_rate = checkpoint['exploration_rate']
            self.update_target_model()
            print(f"模型已加载，探索率: {self.exploration_rate:.4f}")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    def train(self, episodes=1000, max_steps=10000, save_interval=100, target_update_interval=10, callback=None):
        """训练代理"""
        import os
        from game import Game2048
        
        game = Game2048(self.game_size)
        scores = []
        max_tiles = []
        losses = []
        rewards = []
        
        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0
            episode_loss = []
            episode_rewards = []
            
            for step in range(max_steps):
                available_actions = game.get_available_actions()
                if not available_actions:
                    break
                
                action = self.choose_action(state, available_actions)
                next_state, reward, done, info = game.move(action)
                
                # 存储经验
                self.remember(state, action, reward, next_state, done)
                
                # 从经验回放中学习
                if len(self.memory.buffer) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                episode_rewards.append(reward)
                
                if done:
                    break
            
            # 记录结果
            scores.append(game.score)
            max_tiles.append(game.get_max_tile())
            if episode_loss:
                losses.append(np.mean(episode_loss))
            if episode_rewards:
                rewards.append(np.mean(episode_rewards))
            
            # 衰减探索率（仅在不使用噪声网络时有效）
            self.decay_exploration()
            
            # 更新目标网络
            if episode % target_update_interval == 0:
                self.update_target_model()
            
            # 打印进度
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_max_tile = np.mean(max_tiles[-10:])
                avg_loss = np.mean(losses[-10:]) if losses else 0
                avg_reward = np.mean(rewards[-10:]) if rewards else 0
                print(f"Episode: {episode}, 平均分数: {avg_score:.2f}, 平均最大数字: {avg_max_tile:.2f}, 平均损失: {avg_loss:.4f}, 平均奖励: {avg_reward:.4f}, 探索率: {self.exploration_rate:.4f}")
                
                # 调用回调函数进行可视化
                if callback:
                    callback(episode, avg_score, avg_max_tile, avg_loss, avg_reward)
            
            # 保存模型
            if episode % save_interval == 0:
                self.save_model()
        
        # 训练结束后保存模型
        self.save_model()
        print("训练完成！")
        
        return scores, max_tiles
    
    def play_move(self, game, return_q_values=False):
        """在给定游戏中执行一步"""
        state = game.get_state()
        available_actions = game.get_available_actions()
        
        if not available_actions:
            return None if not return_q_values else (None, [0, 0, 0, 0])
        
        # 选择动作（无探索）
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.model(state_tensor)
            q_values = torch.sum(dist * self.support.expand_as(dist), dim=2).cpu().numpy()[0]
        
        # 从可用动作中选择Q值最大的
        masked_q_values = q_values.copy()
        for i in range(4):
            if i not in available_actions:
                masked_q_values[i] = float('-inf')
        
        action = np.argmax(masked_q_values)
        
        if return_q_values:
            return action, q_values
        return action
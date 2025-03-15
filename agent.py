import numpy as np
import random
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from game import Game2048

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QLearningAgent:
    def __init__(self, game_size=4, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.game_size = game_size
        self.state_size = game_size * game_size
        self.action_size = 4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.batch_size = 64
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区
        self.model_path = 'dqn_agent_model.pth'
        
        # 检查是否有可用的GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建神经网络模型
        self.model = DQNNetwork(self.state_size, 128, self.action_size).to(self.device)
        self.target_model = DQNNetwork(self.state_size, 128, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
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
    
    def remember(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区"""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, available_actions):
        """选择动作（探索或利用）"""
        if not available_actions:
            return None
        
        # 探索：随机选择动作
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # 利用：选择Q值最大的动作
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # 屏蔽不可用的动作
        masked_q_values = q_values.copy()
        for i in range(4):
            if i not in available_actions:
                masked_q_values[i] = float('-inf')
        
        return np.argmax(masked_q_values)
    
    def replay(self):
        """从经验回放缓冲区中批量学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机抽取批次样本
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值和目标Q值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # 计算损失并优化
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_exploration(self):
        """衰减探索率"""
        if self.exploration_rate > self.exploration_min:
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
                if len(self.memory) >= self.batch_size:
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
            
            # 衰减探索率
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
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # 从可用动作中选择Q值最大的
        masked_q_values = q_values.copy()
        for i in range(4):
            if i not in available_actions:
                masked_q_values[i] = float('-inf')
        
        action = np.argmax(masked_q_values)
        
        if return_q_values:
            return action, q_values
        return action
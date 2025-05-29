import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from go_game_env import GoGameEnv

class GoDQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GoDQNNetwork, self).__init__()
        # 使用更深的网络结构
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_size * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 调整输入形状为(batch_size, channels, height, width)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, int(np.sqrt(x.size(1))), int(np.sqrt(x.size(1))))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GoQLearningAgent:
    def __init__(self, board_size=19, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.board_size = board_size
        self.state_size = board_size * board_size
        self.action_size = board_size * board_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.model_path = 'go_dqn_agent_model.pth'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = GoDQNNetwork(self.state_size, 512, self.action_size).to(self.device)
        self.target_model = GoDQNNetwork(self.state_size, 512, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_state(self, state):
        return state.flatten().astype(np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        
        if random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # 屏蔽无效动作
        masked_q_values = q_values.copy()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[i] = float('-inf')
        
        return np.argmax(masked_q_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_exploration(self):
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate
        }, self.model_path)
    
    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.exploration_rate = checkpoint['exploration_rate']
            self.update_target_model()
            print(f"模型已加载，探索率: {self.exploration_rate:.4f}")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    def train(self, episodes=1000, max_steps=1000, save_interval=100, target_update_interval=10):
        env = GoGameEnv(self.board_size)
        scores = []
        losses = []
        rewards = []
        
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            episode_loss = []
            
            for step in range(max_steps):
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                action = self.choose_action(state, valid_actions)
                next_state, reward, done, info = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # 记录结果
            scores.append(total_reward)
            if episode_loss:
                losses.append(np.mean(episode_loss))
            rewards.append(total_reward)
            
            # 衰减探索率
            self.decay_exploration()
            
            # 更新目标网络
            if episode % target_update_interval == 0:
                self.update_target_model()
            
            # 保存模型
            if episode % save_interval == 0:
                self.save_model()
            
            # 打印训练信息
            if episode % 10 == 0:
                print(f"Episode: {episode}, Score: {total_reward:.2f}, Loss: {np.mean(episode_loss) if episode_loss else 0:.4f}, Exploration Rate: {self.exploration_rate:.4f}")
        
        return scores, losses, rewards
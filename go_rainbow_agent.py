import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import math
from go_agent import GoQLearningAgent
from go_game_env import GoGameEnv

# 定义经验元组
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class GoRainbowDQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, atom_size=51, support_min=-10, support_max=10, noisy=True):
        super(GoRainbowDQNNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.atom_size = atom_size
        self.support = torch.linspace(support_min, support_max, atom_size)
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        conv_out_size = 64 * input_size * input_size
        
        # 根据是否使用噪声网络选择不同的线性层
        if noisy:
            self.fc1 = NoisyLinear(conv_out_size, hidden_size)
            self.fc2 = NoisyLinear(hidden_size, output_size * atom_size)
        else:
            self.fc1 = nn.Linear(conv_out_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size * atom_size)
        
        self.noisy = noisy
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, int(np.sqrt(x.size(1))), int(np.sqrt(x.size(1))))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.output_size, self.atom_size)
        x = F.softmax(x, dim=2)
        return x
    
    def reset_noise(self):
        if self.noisy:
            self.fc1.reset_noise()
            self.fc2.reset_noise()

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        experiences = [self.buffer[idx] for idx in indices]
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class GoRainbowDQNAgent(GoQLearningAgent):
    def __init__(self, board_size=19, learning_rate=0.0003, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.board_size = board_size
        self.state_size = board_size * board_size
        self.action_size = board_size * board_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.batch_size = 64
        self.model_path = 'go_rainbow_dqn_agent_model.pth'
        
        # Rainbow DQN特有参数
        self.atom_size = 51
        self.v_min = -10
        self.v_max = 10
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建网络
        self.model = GoRainbowDQNNetwork(
            self.state_size, 512, self.action_size,
            self.atom_size, self.v_min, self.v_max
        ).to(self.device)
        
        self.target_model = GoRainbowDQNNetwork(
            self.state_size, 512, self.action_size,
            self.atom_size, self.v_min, self.v_max
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 使用优先经验回放
        self.memory = PrioritizedReplayBuffer(10000)
        
        self.update_target_model()
    
    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        
        if random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.model(state_tensor)
            q_values = (dist * self.support.expand_as(dist)).sum(2).cpu().numpy()[0]
        
        masked_q_values = q_values.copy()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[i] = float('-inf')
        
        return np.argmax(masked_q_values)
    
    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        with torch.no_grad():
            next_dist = self.target_model(next_states)
            next_q = (next_dist * self.support.expand_as(next_dist)).sum(2)
            next_actions = next_q.max(1)[1]
            next_dist = next_dist[range(self.batch_size), next_actions]
        
        t_z = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * self.discount_factor * self.support.unsqueeze(0)
        t_z = t_z.clamp(min=self.v_min, max=self.v_max)
        b = (t_z - self.v_min) / ((self.v_max - self.v_min) / (self.atom_size - 1))
        l = b.floor().long()
        u = b.ceil().long()
        
        offset = torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long()
        offset = offset.unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)
        
        proj_dist = torch.zeros(self.batch_size, self.atom_size, device=self.device)
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1),
            (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1)
        )
        
        dist = self.model(states)
        log_p = torch.log(dist[range(self.batch_size), actions])
        
        loss = -(proj_dist * log_p).sum(1)
        loss = (loss * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新优先级
        priorities = loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities + 1e-6)
        
        return loss.item()
    
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
                
                self.memory.add(state, action, reward, next_state, done)
                
                if len(self.memory.buffer) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            scores.append(total_reward)
            if episode_loss:
                losses.append(np.mean(episode_loss))
            rewards.append(total_reward)
            
            self.decay_exploration()
            
            if episode % target_update_interval == 0:
                self
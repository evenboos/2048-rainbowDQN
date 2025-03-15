# 2048 AI 强化学习项目

这是一个基于深度强化学习的2048游戏AI项目。项目使用DQN（深度Q网络）和Rainbow DQN两种算法来训练AI代理，使其能够自主学习并玩2048游戏。通过可视化界面，您可以直观地观察AI的决策过程。

## 特点

- 支持两种深度强化学习算法：
  - 经典DQN（深度Q网络）
  - Rainbow DQN（集成了多种DQN改进技术）
- 实时可视化界面，展示：
  - 游戏状态
  - 当前分数
  - 最大数字
  - 每个动作的Q值
- 支持三种运行模式：
  - AI训练模式
  - AI演示模式
  - 人类玩家模式

## 安装

1. 克隆项目到本地：
```bash
git clone https://github.com/evenboos/2048-rainbowDQN.git
cd RL2048
```

2. 安装依赖：
```bash
pip install torch numpy tkinter matplotlib
```

## 使用方法

项目提供三种运行模式：

### 1. 训练模式

训练AI代理（可选DQN或Rainbow DQN）：
```bash
python main.py train --episodes 1000 --agent dqn  # 训练普通DQN代理
python main.py train --episodes 1000 --agent rainbow  # 训练Rainbow DQN代理
```

### 2. AI演示模式

观看训练好的AI代理玩游戏：
```bash
python main.py play --agent dqn  # 使用DQN代理
python main.py play --agent rainbow  # 使用Rainbow DQN代理
```

### 3. 人类玩家模式

启动游戏界面供人类玩家游玩：
```bash
python main.py human
```

## 项目结构

- `main.py`: 主程序入口，处理命令行参数和游戏模式选择
- `game.py`: 2048游戏核心逻辑
- `agent.py`: DQN代理实现
- `rainbow_agent.py`: Rainbow DQN代理实现
- `visualization.py`: 游戏界面和训练过程可视化
- `constants.py`: 常量定义
- `logic.py`: 游戏逻辑辅助函数
- `puzzle.py`: 人类玩家界面实现

## 技术细节

### DQN实现
- 使用经典的深度Q学习算法
- 包含经验回放和目标网络
- 采用epsilon-greedy策略进行探索

### Rainbow DQN改进
- 集成了多种DQN改进技术：
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
  - Multi-step Learning
  - Distributional RL
  - Noisy Nets

## 贡献

欢迎提交问题和改进建议！如果您想贡献代码：

1. Fork 项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建一个Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情
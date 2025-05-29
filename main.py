import sys
import argparse
import time
from agent import QLearningAgent
from rainbow_agent import RainbowDQNAgent
from game import Game2048
from puzzle import GameGrid
from visualization import TrainingVisualizer, GameVisualizer

def train_agent(episodes=1000, agent_type='dqn', game_type='2048'):
    """训练代理（不带实时可视化，仅在训练结束时保存结果）"""
    print(f"开始训练{game_type}游戏的{agent_type.upper()}代理，计划训练{episodes}轮...")
    
    # 根据游戏类型和代理类型创建相应的代理
    if game_type.lower() == '2048':
        if agent_type.lower() == 'rainbow':
            agent = RainbowDQNAgent()
        else:  # 默认使用普通DQN
            agent = QLearningAgent()
    else:  # 围棋游戏
        if agent_type.lower() == 'rainbow':
            from go_rainbow_agent import GoRainbowDQNAgent
            agent = GoRainbowDQNAgent()
        else:  # 默认使用普通DQN
            from go_agent import GoQLearningAgent
            agent = GoQLearningAgent()
    
    # 创建训练数据收集器
    visualizer = TrainingVisualizer()
    
    # 修改agent.train方法的调用，添加数据收集回调
    def visualization_callback(episode, score, max_tile, loss=None, reward=None):
        visualizer.update(episode, score, max_tile, loss, reward)
        # 每100轮打印一次进度
        if episode % 100 == 0 or episode == episodes - 1:
            print(f"训练进度: {episode+1}/{episodes} 轮, 当前分数: {score}, 最大数字: {max_tile}")
    
    scores, max_tiles = agent.train(episodes=episodes, callback=visualization_callback)
    
    # 保存训练结果图表
    visualizer.save_figure()
    
    print(f"训练完成！最终探索率: {agent.exploration_rate:.4f}")
    print(f"最后10轮平均分数: {sum(scores[-10:]) / 10:.2f}")
    print(f"最后10轮平均最大数字: {sum(max_tiles[-10:]) / 10:.2f}")

def agent_play(agent_type='dqn', game_type='2048'):
    """让训练好的代理自动游玩（带可视化）"""
    print(f"让训练好的{game_type}游戏的{agent_type.upper()}代理自动游玩...")
    
    # 根据游戏类型和代理类型创建相应的代理
    if game_type.lower() == '2048':
        if agent_type.lower() == 'rainbow':
            agent = RainbowDQNAgent()  # 会自动加载已保存的模型
        else:  # 默认使用普通DQN
            agent = QLearningAgent()  # 会自动加载已保存的模型
    else:  # 围棋游戏
        if agent_type.lower() == 'rainbow':
            from go_rainbow_agent import GoRainbowDQNAgent
            agent = GoRainbowDQNAgent()  # 会自动加载已保存的模型
        else:  # 默认使用普通DQN
            from go_agent import GoQLearningAgent
            agent = GoQLearningAgent()  # 会自动加载已保存的模型
    
    game = Game2048()
    state = game.reset()
    total_reward = 0
    done = False
    steps = 0
    
    # 创建游戏可视化器
    visualizer = GameVisualizer()
    visualizer.update_grid_cells(state)
    visualizer.update_info(game.score, game.get_max_tile(), steps)
    
    while not done and steps < 1000:  # 设置最大步数防止无限循环
        available_actions = game.get_available_actions()
        if not available_actions:
            break
        
        # 获取动作和Q值
        action, q_values = agent.play_move(game, return_q_values=True)
        if action is None:
            break
        
        # 更新Q值显示
        visualizer.update_q_values(q_values)
        
        next_state, reward, done, info = game.move(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        # 更新可视化
        visualizer.update_grid_cells(state)
        visualizer.update_info(game.score, game.get_max_tile(), steps, action)
        visualizer.update()
        
        # 打印当前状态
        print(f"步骤 {steps}, 动作: {['上', '下', '左', '右'][action]}, 分数: {info['score']}, 最大数字: {info['max_tile']}")
        
        # 添加短暂延迟，使可视化更容易观察
        time.sleep(0.3)
    
    print(f"游戏结束！总步数: {steps}, 总分数: {game.score}, 最大数字: {game.get_max_tile()}")
    visualizer.mainloop()  # 保持窗口显示直到用户关闭

def human_play(game_type='2048'):
    """启动人类玩家界面"""
    print(f"启动{game_type}游戏的人类玩家界面...")
    if game_type.lower() == '2048':
        game_grid = GameGrid()
    else:  # 围棋游戏
        from go_game import GoGame
        game = GoGame()
        game.play()

def main():
    parser = argparse.ArgumentParser(description='2048和围棋游戏与DQN代理')
    parser.add_argument('mode', choices=['train', 'play', 'human'], 
                        help='选择模式: train(训练代理), play(代理游玩), human(人类游玩)')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='训练模式下的训练轮数，默认为1000')
    parser.add_argument('--agent', choices=['dqn', 'rainbow'], default='dqn',
                        help='选择代理类型: dqn(普通DQN), rainbow(Rainbow DQN)，默认为dqn')
    parser.add_argument('--game', choices=['2048', 'go'], default='2048',
                        help='选择游戏类型: 2048(2048游戏), go(围棋)，默认为2048')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(episodes=args.episodes, agent_type=args.agent, game_type=args.game)
    elif args.mode == 'play':
        agent_play(agent_type=args.agent, game_type=args.game)
    elif args.mode == 'human':
        human_play(game_type=args.game)

if __name__ == "__main__":
    main()
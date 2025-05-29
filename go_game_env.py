import numpy as np
from go_game import GoGame

class GoGameEnv:
    def __init__(self, board_size=19):
        self.game = GoGame()
        self.board_size = board_size
        self.action_space_size = board_size * board_size
        self.state_size = board_size * board_size
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        self.game = GoGame()
        return self.get_state()
    
    def get_state(self):
        """获取当前状态
        返回一个(N, N)的numpy数组，其中：
        0: 空位
        1: 黑子
        2: 白子
        """
        return self.game.board.copy()
    
    def step(self, action):
        """执行一步动作
        
        参数:
            action: int, 范围[0, board_size * board_size - 1]
                    表示在哪个位置落子
        
        返回:
            state: numpy array, 新的状态
            reward: float, 奖励值
            done: bool, 游戏是否结束
            info: dict, 额外信息
        """
        # 将一维动作转换为二维坐标
        x = action // self.board_size
        y = action % self.board_size
        
        # 检查动作是否有效
        if not (0 <= x < self.board_size and 0 <= y < self.board_size) or \
           self.game.board[x][y] != 0:
            return self.get_state(), -10.0, True, {"invalid_move": True}
        
        # 记录移动前的状态
        old_board = self.game.board.copy()
        old_stones = np.sum(old_board != 0)
        
        # 执行落子
        self.game.board[x][y] = self.game.current_player
        
        # 检查是否有提子
        captured = False
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.game.board[nx][ny] == (3 - self.game.current_player):
                    if self.game.get_liberties(nx, ny, self.game.board[nx][ny]) == 0:
                        self.game.remove_stones(nx, ny)
                        captured = True
        
        # 如果当前落子后没有气，则不允许落子
        if self.game.get_liberties(x, y, self.game.current_player) == 0:
            self.game.board = old_board
            return self.get_state(), -10.0, True, {"invalid_move": True}
        
        # 计算奖励
        reward = 0.0
        new_stones = np.sum(self.game.board != 0)
        
        # 基础奖励：占据位置
        reward += 0.1
        
        # 提子奖励
        if captured:
            stones_diff = old_stones - new_stones
            reward += stones_diff * 2.0
        
        # 切换玩家
        self.game.current_player = 3 - self.game.current_player
        
        # 检查游戏是否结束（这里简化处理，实际围棋规则更复杂）
        done = False
        if np.sum(self.game.board == 0) == 0:  # 棋盘已满
            done = True
        
        return self.get_state(), reward, done, {"captured": captured}
    
    def get_valid_actions(self):
        """获取当前可用的动作"""
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.game.board[i][j] == 0:
                    action = i * self.board_size + j
                    valid_actions.append(action)
        return valid_actions
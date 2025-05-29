import pygame
import numpy as np

class GoGame:
    def __init__(self):
        pygame.init()
        self.BOARD_SIZE = 19
        self.CELL_SIZE = 40
        self.MARGIN = 40
        self.STONE_SIZE = 18
        self.WINDOW_SIZE = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.MARGIN
        
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption('围棋')
        
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        self.current_player = 1  # 1为黑子，2为白子
        
    def draw_board(self):
        self.screen.fill((240, 200, 150))  # 棋盘底色
        
        # 画棋盘线
        for i in range(self.BOARD_SIZE):
            pygame.draw.line(self.screen, (0, 0, 0),
                           (self.MARGIN + i * self.CELL_SIZE, self.MARGIN),
                           (self.MARGIN + i * self.CELL_SIZE, self.WINDOW_SIZE - self.MARGIN))
            pygame.draw.line(self.screen, (0, 0, 0),
                           (self.MARGIN, self.MARGIN + i * self.CELL_SIZE),
                           (self.WINDOW_SIZE - self.MARGIN, self.MARGIN + i * self.CELL_SIZE))
        
        # 画星位
        star_points = [(3, 3), (3, 9), (3, 15),
                      (9, 3), (9, 9), (9, 15),
                      (15, 3), (15, 9), (15, 15)]
        for point in star_points:
            x = self.MARGIN + point[0] * self.CELL_SIZE
            y = self.MARGIN + point[1] * self.CELL_SIZE
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 5)
        
        # 画棋子
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i][j] == 1:  # 黑子
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                     (self.MARGIN + i * self.CELL_SIZE,
                                      self.MARGIN + j * self.CELL_SIZE),
                                     self.STONE_SIZE)
                elif self.board[i][j] == 2:  # 白子
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                     (self.MARGIN + i * self.CELL_SIZE,
                                      self.MARGIN + j * self.CELL_SIZE),
                                     self.STONE_SIZE)
    
    def get_board_position(self, pos):
        x, y = pos
        board_x = round((x - self.MARGIN) / self.CELL_SIZE)
        board_y = round((y - self.MARGIN) / self.CELL_SIZE)
        if 0 <= board_x < self.BOARD_SIZE and 0 <= board_y < self.BOARD_SIZE:
            return board_x, board_y
        return None
    
    def get_liberties(self, x, y, color, checked=None):
        if checked is None:
            checked = set()
        
        if (x, y) in checked or x < 0 or x >= self.BOARD_SIZE or y < 0 or y >= self.BOARD_SIZE:
            return 0
        
        if self.board[x][y] == 0:
            return 1
        
        if self.board[x][y] != color:
            return 0
        
        checked.add((x, y))
        liberties = 0
        
        # 检查四个方向
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            liberties += self.get_liberties(x + dx, y + dy, color, checked)
        
        return liberties
    
    def remove_stones(self, x, y):
        color = self.board[x][y]
        stones_to_remove = set()
        checked = set()
        
        def find_group(x, y):
            if (x, y) in checked or x < 0 or x >= self.BOARD_SIZE or y < 0 or y >= self.BOARD_SIZE:
                return
            if self.board[x][y] != color:
                return
            
            stones_to_remove.add((x, y))
            checked.add((x, y))
            
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                find_group(x + dx, y + dy)
        
        find_group(x, y)
        for stone_x, stone_y in stones_to_remove:
            self.board[stone_x][stone_y] = 0
    
    def place_stone(self, pos):
        board_pos = self.get_board_position(pos)
        if board_pos is None:
            return
        
        x, y = board_pos
        if self.board[x][y] != 0:
            return
        
        self.board[x][y] = self.current_player
        
        # 检查是否有提子
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.BOARD_SIZE and 0 <= ny < self.BOARD_SIZE:
                if self.board[nx][ny] == (3 - self.current_player):
                    if self.get_liberties(nx, ny, self.board[nx][ny]) == 0:
                        self.remove_stones(nx, ny)
        
        # 如果当前落子后没有气，则不允许落子
        if self.get_liberties(x, y, self.current_player) == 0:
            self.board[x][y] = 0
            return
        
        self.current_player = 3 - self.current_player
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        self.place_stone(event.pos)
            
            self.draw_board()
            pygame.display.flip()
        
        pygame.quit()

if __name__ == '__main__':
    game = GoGame()
    game.run()
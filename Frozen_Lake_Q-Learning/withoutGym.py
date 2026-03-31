import numpy as np
import random


class FrozenLake:
    def __init__(self, is_slippery=True):
        # Grid layout
        self.grid = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ]
        
        self.n_rows = 4
        self.n_cols = 4
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4  # LEFT, DOWN, RIGHT, UP
        
        self.is_slippery = is_slippery
        
        # Action mapping
        self.LEFT = 0
        self.DOWN = 1
        self.RIGHT = 2
        self.UP = 3
        
        self.reset()

    def state_to_pos(self, state):
        return divmod(state, self.n_cols)

    def pos_to_state(self, row, col):
        return row * self.n_cols + col

    def reset(self):
        self.state = 0  # Start at 'S'
        return self.state

    def step(self, action):
        # Handle slippery randomness
        if self.is_slippery:
            action = random.choice([
                action,                      # intended
                (action + 1) % 4,            # right turn
                (action - 1) % 4             # left turn
            ])

        row, col = self.state_to_pos(self.state)

        # Move
        if action == self.LEFT:
            col = max(col - 1, 0)
        elif action == self.DOWN:
            row = min(row + 1, self.n_rows - 1)
        elif action == self.RIGHT:
            col = min(col + 1, self.n_cols - 1)
        elif action == self.UP:
            row = max(row - 1, 0)

        new_state = self.pos_to_state(row, col)
        cell = self.grid[row][col]
        self.state = new_state

        # Rewards and termination
        if cell == 'H':
            return new_state, 0, True   # hole → done
        elif cell == 'G':
            return new_state, 1, True   # goal → reward
        else:
            return new_state, 0, False  # normal

    def render(self):
        row, col = self.state_to_pos(self.state)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if r == row and c == col:
                    print('A', end=' ')
                else:
                    print(self.grid[r][c], end=' ')
            print()
        print()

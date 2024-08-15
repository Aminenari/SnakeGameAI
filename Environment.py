import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

#random.seed(24) #random seed for some analysis described in the Paper

pygame.init()
font = pygame.font.Font('Font.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Color codes
WHITE = (255, 255, 255)
RED = (200,0,0)
Darkgreen = (0, 100, 0)
Lightgreen = (144, 238, 144)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 999999999999999999999999999999999999999999999999999999999999999999999999999999

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game RL Amine Nari')
        self.clock = pygame.time.Clock()
        self.reset()
    

    def reset(self):
        # initialize game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    #Function that places the food
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    #Function that plays each step
    def play_step(self, action):
        self.frame_iteration += 1
        # Makes sure that the game can be quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Performs movement based on the action decided by the agent.
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Checks if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Checks if there is an point, if so then a new food is being placed. Otherwise it performs another move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            
            self.snake.pop()
        
        # Updates the display
        self._update_display()
        self.clock.tick(SPEED)
        
        # Return information (reward, score and if the game is over)
        return reward, game_over, self.score

    #Function that checks if there is an collision
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # If it hits an boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # If it hits itself
        if pt in self.snake[1:]:
            return True

        return False

    #Updates the display
    def _update_display(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, Darkgreen, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, Lightgreen, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    #performs the move based on the action.
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)






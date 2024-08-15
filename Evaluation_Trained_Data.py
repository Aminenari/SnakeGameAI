import numpy as np
from Environment import SnakeGameAI, Direction, Point
import random
from Live_plotter import plot
import matplotlib.pyplot as plt



eps = 1.0                       # epsilon value for epsilon-greedy policy
eps_discount = 0.7              # Epsilon discount value for epsilon-greedy policy
min_eps = 0.01                  # The minimal epsilon value for the epsilon-greedy policy, hence there will always be some randomness in the agent moves
gamma = 0.9                     # Discount rate
actions = 3                     # Actions
states = 2 ** 11                # Amount of states --> Check Paper for details
alpha = 0.1                     # learning rate     
Q = np.load("Qtable_seed24_from_results.npy")    # Loads the saved Q Table


#Function that gets the state of the agent
def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
        
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
            
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
            
        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
        ]
            
    return np.array(state, dtype=int)
    

def state_to_index(state):
    index = 0
    for bit in state:
        index = (index << 1) | bit
    return index



#get action greedy --> we want to get the best action from the trained values.
def get_action(state):
    action = [0,0,0]
    index = state_to_index(state)
    move = np.argmax(Q[index])
    action[move] = 1
    return action, move

def train():
    game = SnakeGameAI()
    n_games = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    global eps
    
    while n_games < 100:
        state = get_state(game)

        action, move = get_action(state)

        reward, game_over, score = game.play_step(action)

        state_new = get_state(game)

        
        

        if game_over:
            game.reset()
            n_games += 1


            
            if score > record:
                record = score


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            
            
            plt.savefig('PerformenceTrainedQvalues')
            



train()


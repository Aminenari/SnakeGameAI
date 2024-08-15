import numpy as np
from Environment import SnakeGameAI, Direction, Point
import random
from Live_plotter import plot
import matplotlib.pyplot as plt


eps = 1.0                       # epsilon value for epsilon-greedy policy
eps_discount = 0.7              # Epsilon discount value for epsilon-greedy policy
min_eps = 0.01                  # The minimal epsilon value for the epsilon-greedy policy, hence there will always be some randomness in the agent moves
gamma = 0.9                     # Discount rate
states = 2 ** 11                # Amount of states --> Check Paper for details
alpha = 0.3                     # learning rate     
actions = 3                     # Actions
Q = np.zeros((states,actions))  # Initializes an empty Q-table.


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
        # boolean value for danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # boolean value for danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # boolean value for danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
            
        # boolean values for move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
            
        # boolean values for Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
        ]
            
    return np.array(state, dtype=int)
    
#This function is used to transform the state to an index number, so the right Qvalue can be obtained from the Q-table.
def state_to_index(state):
    index = 0
    for bit in state:
        index = (index << 1) | bit
    return index


#Get action based on the epsilon-greedy policy
"""
def get_action(state):
    # random moves: tradeoff exploration / exploitation
    action = [0,0,0]
    if random.random() < eps:
        move = random.randint(0, 2)
    else:
        #max from Q table. 
        #action depends on state
        #then max value of state 
        index = state_to_index(state)
        move = np.argmax(Q[index])
    action[move] = 1
    return action, move
"""

#Get action based on the greedy policy
def get_action(state):
    action = [0,0,0]
    index = state_to_index(state)
    move = np.argmax(Q[index])
    action[move] = 1
    return action, move

#Function that Trains the agent 
def train():
    
    game = SnakeGameAI()
    n_games = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    global eps
    
    #while loop for the amount of episodes desired
    while n_games < 500:
        # Obtain current state
        state = get_state(game)

        # Obtain the action
        action, move = get_action(state)

        # Perform the move and get reward if game over and score
        reward, game_over, score = game.play_step(action)

        #get the new state
        state_new = get_state(game)


        #Update the Q value in the Q table
        index = state_to_index(state)
        index_new = state_to_index(state_new)
        Q[index][move] = Q[index][move] + alpha * (reward + gamma * np.max(Q[index_new]) - Q[index][move])
        

        
        #Checks if the game is over to record the information and start a new episode/game
        if game_over:
            game.reset()
            n_games += 1
            #decays the eps, if the new epsilon is higher then the min_eps value
            eps = max(eps * eps_discount, min_eps)


            
            if score > record:
                record = score


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            #Save the scores, meanscore, Qtable and Figure
            #np.save("Score.npy", np.array(plot_scores))                #save Score
            #np.save("MeanScore.npy", np.array(plot_mean_scores))       #save mean score
            #plt.savefig('TrainingPerformance.png')                     #save training performance figure
            
        #np.save("Qtable.npy", Q)                                       #Save Q table 
            
            


if __name__ == "__main__":
    train()

import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.temperature = 1.0
        self.temperature_decay = 0.995
        self.min_temperature = 0.1
        self.reset()
        
        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()
        
    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
        self.steps_without_food = 0
        self.last_positions = []

    def check_collision(self, new_x, new_y, snake_body):
        if (new_x < helper.GRID_SIZE or 
            new_x >= helper.DISPLAY_SIZE - helper.GRID_SIZE or 
            new_y < helper.GRID_SIZE or 
            new_y >= helper.DISPLAY_SIZE - helper.GRID_SIZE):
            return True
        
        if (new_x, new_y) in snake_body:
            return True
            
        return False

    def get_safe_actions(self, state):
        snake_head_x, snake_head_y, snake_body, _, _ = state
        safe_actions = []
        
        for action in self.actions:
            new_x, new_y = snake_head_x, snake_head_y
            
            if action == 0:
                new_y -= helper.GRID_SIZE
            elif action == 1:
                new_y += helper.GRID_SIZE
            elif action == 2:
                new_x -= helper.GRID_SIZE
            elif action == 3:
                new_x += helper.GRID_SIZE
                
            if not self.check_collision(new_x, new_y, snake_body):
                safe_actions.append(action)
        
        return safe_actions

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    
    def helper_func(self, state):
        # print("IN helper_func")
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state

        near_wall_x = 1 if snake_head_x <= helper.GRID_SIZE * 2 else 2 if snake_head_x >= (helper.DISPLAY_SIZE - 3 * helper.GRID_SIZE) else 0
        near_wall_y = 1 if snake_head_y <= helper.GRID_SIZE * 2 else 2 if snake_head_y >= (helper.DISPLAY_SIZE - 3 * helper.GRID_SIZE) else 0
        
        food_dir_x = 1 if food_x < snake_head_x else 2 if food_x > snake_head_x else 0
        food_dir_y = 1 if food_y < snake_head_y else 2 if food_y > snake_head_y else 0
        
        adjoining_top = int((snake_head_x, snake_head_y - helper.GRID_SIZE) in snake_body or 
                          (snake_head_x, snake_head_y - 2*helper.GRID_SIZE) in snake_body)
        adjoining_bottom = int((snake_head_x, snake_head_y + helper.GRID_SIZE) in snake_body or 
                             (snake_head_x, snake_head_y + 2*helper.GRID_SIZE) in snake_body)
        adjoining_left = int((snake_head_x - helper.GRID_SIZE, snake_head_y) in snake_body or 
                           (snake_head_x - 2*helper.GRID_SIZE, snake_head_y) in snake_body)
        adjoining_right = int((snake_head_x + helper.GRID_SIZE, snake_head_y) in snake_body or 
                            (snake_head_x + 2*helper.GRID_SIZE, snake_head_y) in snake_body)
        
        return (near_wall_x, near_wall_y, food_dir_x, food_dir_y, 
                adjoining_top, adjoining_bottom, adjoining_left, adjoining_right) 

    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1
        
    def get_food_direction_reward(self, state, action):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        current_distance = abs(snake_head_x - food_x) + abs(snake_head_y - food_y)
        
        new_x, new_y = snake_head_x, snake_head_y
        if action == 0:
            new_y -= helper.GRID_SIZE
        elif action == 1:
            new_y += helper.GRID_SIZE
        elif action == 2:
            new_x -= helper.GRID_SIZE
        elif action == 3:
            new_x += helper.GRID_SIZE

        if self.check_collision(new_x, new_y, snake_body):
            return -1.0
            
        new_distance = abs(new_x - food_x) + abs(new_y - food_y)
        
        if new_distance < current_distance:
            reward = 0.3
        elif new_distance > current_distance:
            reward = -0.2
        else:
            reward = 0

        num_safe_moves = len(self.get_safe_actions((new_x, new_y, snake_body, food_x, food_y)))
        safety_reward = 0.1 * num_safe_moves
        
        return reward + safety_reward

    def softmax_exploration(self, q_values, temperature):
        q_values = q_values - np.max(q_values)
        exp_values = np.exp(q_values / temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probabilities)
            
    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.

    def agent_action(self, state, points, dead):
        # print("IN AGENT_ACTION")
        s_prime = self.helper_func(state)
        safe_actions = self.get_safe_actions(state)
        
        if not safe_actions:
            safe_actions = self.actions
        
        if self._train:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
            q_values = self.Q[tuple(s_prime)].copy()
            
            for action in self.actions:
                if action in safe_actions:
                    q_values[action] += self.get_food_direction_reward(state, action)
                else:
                    q_values[action] = float('-inf')
            
            if random.random() < self.epsilon:
                if len(safe_actions) > 0:
                    safe_q_values = np.array([q_values[a] for a in safe_actions])
                    safe_action_idx = self.softmax_exploration(safe_q_values, self.temperature)
                    action = safe_actions[safe_action_idx]
                else:
                    action = random.choice(self.actions)
            else:
                action = np.argmax(q_values)
            
            reward = self.compute_reward(points, dead)
            if self.s is not None:
                lr = self.LPC / (self.LPC + self.N[tuple(self.s)][self.a])
                next_safe_actions = self.get_safe_actions(state)
                if next_safe_actions:
                    max_next_q = max([self.Q[tuple(s_prime)][a] for a in next_safe_actions])
                else:
                    max_next_q = np.max(self.Q[tuple(s_prime)])
                
                self.Q[tuple(self.s)][self.a] += lr * (
                    reward + 
                    self.gamma * max_next_q - 
                    self.Q[tuple(self.s)][self.a]
                )
                self.N[tuple(self.s)][self.a] += 1
            
            self.s = s_prime
            self.a = action
            self.points = points
            
        else:
            q_values = self.Q[tuple(s_prime)].copy()
            for action in self.actions:
                if action in safe_actions:
                    q_values[action] += self.get_food_direction_reward(state, action)
                else:
                    q_values[action] = float('-inf')
            
            action = np.argmax(q_values)
        
        return action
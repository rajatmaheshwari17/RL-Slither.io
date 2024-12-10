
# RL-Slither.io

Welcome to **RL-Slither.io**, a dynamic and fun implementation of the classic Snake game powered by Reinforcement Learning (RL). The project uses **Q-Learning** with added exploration strategies, safety mechanisms, and rewards to teach the snake agent how to navigate the environment, avoid obstacles, and optimize for long-term survival.

## Features

1.  **Q-Learning Agent**:
    
    -   Trained using a Q-Table to learn state-action values.
    -   Balances exploitation and exploration with **ε-greedy** and **Softmax exploration** strategies.
    -   Dynamic exploration decay for optimized training.
2.  **Safety Mechanisms**:
    
    -   Actions are filtered to avoid collisions with walls or the snake's body.
    -   Reward adjustments to prioritize survival and safe moves.
3.  **Reward System**:
    
    -   Rewards for eating food (+1).
    -   Penalties for collisions (-1).
    -   Small penalties for idle moves (-0.1).
    -   Additional reward for moving closer to food.
4.  **Exploration Strategies**:
    
    -   **Softmax Exploration**: Action probabilities are adjusted based on the Q-values and temperature for better exploration of the environment.
    -   **State Exploration**: Tracks unvisited tiles to encourage the agent to explore the board comprehensively.
5.  **Adaptive Learning Rates**:
    
    -   Learning rate dynamically adjusts based on the number of times a state-action pair has been visited.
6.  **Training, Testing, and Display Modes**:
    
    -   Train the agent for a specified number of games.
    -   Test its performance across unseen games.
    -   Watch the trained snake play in a graphical display.


## Files and Structure

### **Core Files**

1.  **`snake_agent.py`**:
    
    -   Implements the `SnakeAgent` class, which houses all logic for Q-Learning, exploration strategies, reward calculations, and safe actions.
2.  **`game.py`**:
    
    -   Contains the game loop and manages training, testing, and display modes.
    -   Interfaces with `SnakeAgent` and the game environment.
3.  **`board.py`**:
    
    -   Defines the `BoardEnv` and `Snake` classes for managing the game board and the snake's movements.
    -   Handles collision detection, food placement, and scoring.
4.  **`helper.py`**:
    
    -   Includes utility functions for initializing Q-tables, saving/loading models, and other shared constants (like grid size, colors, and display settings).



## How It Works

### **Q-Learning**

The agent uses Q-Learning to update the Q-table based on the reward for its actions: Q(s,a)←Q(s,a)+α(r+γmax⁡a′Q(s′,a′)−Q(s,a))Q(s, a) \gets Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)

-   **States (`s`)**: Encoded as tuples capturing the snake's position, proximity to walls, food direction, and body segments.
-   **Actions (`a`)**: Four possible moves (up, down, left, right).
-   **Rewards (`r`)**: Based on eating food, moving closer to food, and avoiding unsafe moves.

### **Exploration**

The agent balances **exploitation** (choosing the best-known action) with **exploration** (trying new actions). Strategies used:

-   **ε-Greedy**: A decaying epsilon controls the exploration probability.
-   **Softmax Exploration**: Action probabilities are calculated based on Q-values and a temperature parameter.

### **Safety First**

Before taking an action, the agent checks for safety:

-   **Collision Avoidance**: Ensures moves don't lead into walls or the snake's body.
-   **Safe Actions**: Limits choices to non-lethal moves.



## Usage

### **Installation**

1.  Clone the repository:
    
    ```bash
    git clone git@github.com:rajatmaheshwari17/RL-Slither.io.git
    cd RL-Slither.io
    
    ```
    
2.  Install dependencies:
    
    ```bash
    pip install pygame numpy
    ```
    

### **Run the Game**

1.  **Train the Agent**:
    
    ```bash
    python3 game.py --NTRI 10000
    ```
    
    Train for 10,000 iterations.
    
2.  **Test the Agent**:
    
    ```bash
    python3 game.py --NTEI 1000
    ```
    
    Test for 1,000 iterations.
    
3.  **Display Games**:
    
    ```bash
    python3 game.py --DISP 5
    ```
    
    Watch 5 games played by the trained agent.
    
4.  Combine all:
    
    ```bash
    python3 game.py --NTRI 10000 --NTEI 1000 --DISP 5
    ```
    



## Enhancements

### Current Improvements

-   Adaptive exploration strategies using ε-greedy and Softmax.
-   Reward structure tailored for safety and food collection.
-   Safety checks for collision-free actions.




#
_This README is a part of the RL-Slither.io Project by Rajat Maheshwari._

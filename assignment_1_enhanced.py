import copy
import sys
import time
import csv
import os
import random
import math
from turtle import width
from utils import *

from search import Problem, best_first_graph_search, breadth_first_graph_search,recursive_best_first_search, compare_searchers, depth_first_graph_search, depth_limited_search, uniform_cost_search,astar_search, Node, memoize
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from agents import *

import random
from agents import *
# Rule Class


######################### To ACTIVATE Random wall placement, go to line 238 and uncomment that section, please comment back before testing 2#############
class Rule:
    def __init__(self, state, action):
        self.state = state  # The expected state (percept)
        self.action = action  # The corresponding action to take
    
    def matches(self, state):
        # Check if the percept (state) matches the rule's state
        return self.state == state

# Function to match the current state with a rule
def interpret_input(state):
    return state

class ReachabilityProblem(Problem):
    def __init__(self, initial, goal, grid, width, height):
        super().__init__(initial, goal)
        self.grid = grid
        self.width = width
        self.height = height

    def actions(self, state):
        """Return the list of possible actions (up, down, left, right)."""
        x, y = state
        actions = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Down, Up
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                if self.grid[new_x][new_y] != 'wall':  # Only move to non-wall spaces
                    actions.append((new_x, new_y))

        return actions


    def result(self, state, action):
        """Return the new state after taking an action."""
        return action

    def goal_test(self, state):
        """Check if the current state is the goal state."""
        return state == self.goal


def is_reachable_bfs(start, goal, grid, width, height):
    """Use AIMA BFS to check if the goal (coin) is reachable from the start (agent's position)."""
    problem = ReachabilityProblem(start, goal, grid, width, height)
    solution = breadth_first_graph_search(problem)
    return solution is not None  # Return True if a solution (path) was found, False otherwise




# Treasure Hunt Environment Class

class treasure_hunt_environment(Environment):
    def __init__(self, width, height, coins_num, wall_num):
        super().__init__()
        self.width = width
        self.height = height
        self.coins_num = coins_num
        self.wall_num = wall_num
        self.coins = []
        self.things = []  
        self.agents = [] 
        self.grid = [['empty' for _ in range(width)] for _ in range(height)]
        self.walls = []
        self.time_taken = 0

        self.setup_environment()
    
    def percept(self, agent):
        """
        The environment provides the agent with a percept based on its current location.
        """
        x, y = agent.x, agent.y
        
        # Ensure agent stays within bounds
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return 'out_of_bounds'
        
        # Return the correct percept based on the agent's current tile
        if self.grid[y][x] == 'coin':
            return 'coin'
        elif self.grid[y][x] == 'wall':
            return 'hit_wall'
        else:
            return 'empty'
    def move_walls_safely(self):
        """Move walls dynamically and ensure they don't block access to coins."""
        new_walls = set()

        for (x, y) in self.walls:
            self.grid[x][y] = 'empty'  # Remove the wall from its current position
            
            # Find new valid positions for the walls
            valid_moves = []
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Down, Up

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    if self.grid[new_x][new_y] == 'empty' and (new_x, new_y) not in new_walls and (self.agents[0].x != new_x and self.agents[0].y != new_y):
                        valid_moves.append((new_x, new_y))
            
            if valid_moves:
                # Randomly choose a valid new position for the wall
                new_wall_position = random.choice(valid_moves)
                # Temporarily place the wall
                self.grid[new_wall_position[0]][new_wall_position[1]] = 'wall'
                # Ensure the agent can still reach all coins
                if self.all_coins_reachable((self.agents[0].x, self.agents[0].y), self.coins, self.grid, self.width, self.height):
                    new_walls.add(new_wall_position)  # Keep the new position
                else:
                    # If the new wall blocks access, revert it
                    self.grid[new_wall_position[0]][new_wall_position[1]] = 'empty'
                    new_walls.add((x, y))  # Keep the wall in its original position
                    self.grid[x][y] = 'wall'
            else:
                # If no valid move, leave the wall in its current position
                new_walls.add((x, y))
                self.grid[x][y] = 'wall'
        
        self.walls = list(new_walls)  # Update the walls list with the new positions

    def random_empty_position(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.grid[x][y] == 'empty':
                return x, y
            
    def all_coins_reachable(self,agent_start, coin_positions, grid, width, height):
        """Check if all coins are reachable from the agent's starting position."""
        for coin in coin_positions:
            if not is_reachable_bfs(agent_start, coin, grid, width, height):
                return False  # Return False if any coin is not reachable
        return True  # Return True if all coins are reachable

            
    def place_coins_after_checking_reachability(self):
        """Place all coins in reachable positions after walls have been placed, retrying if necessary."""
        while True:
            # Step 1: Find all reachable positions from the agent's starting position using AIMA BFS
            reachable_positions = []
            for x in range(self.width):
                for y in range(self.height):
                    if self.grid[x][y] == 'empty':  # Consider only empty spaces
                        if is_reachable_bfs((0, 0), (x, y), self.grid, self.width, self.height):
                            reachable_positions.append((x, y))

            # Step 2: Check if enough positions are reachable
            if len(reachable_positions) >= self.coins_num:
                # Step 3: Randomly place coins in the reachable positions
                coin_positions = random.sample(reachable_positions, self.coins_num)
                for (x, y) in coin_positions:
                    self.grid[x][y] = 'coin'
                    self.coins.append((x, y))
                break  # Exit the loop when all coins are placed and reachable
            else:
                # Step 4: Retry if not enough positions are reachable
                print('HERE STUCK------------')
                self.reduce_walls()
                

    def reduce_walls(self):
        """Reduce the number of walls dynamically to increase the number of reachable positions."""
        if self.wall_num > 0:
            # Remove a percentage of walls
            for _ in range(int(self.wall_num / 2)):
                x, y = self.walls.pop()  # Remove walls from the list
                print(self.grid[x][y])
                print(f"Removed {self.grid[x][y]} walls to increase reachable positions.")
                self.grid[x][y] = 'empty'

    def setup_environment(self):
        """Set up the environment by placing walls first, then coins."""
        # Step 1: Place walls randomly
        self.place_walls_randomized()
        
        # Step 2: Place coins only in reachable positions using AIMA BFS
        self.place_coins_after_checking_reachability()

    def add_agent(self, x, y):
        self.grid[x][y] = 'agent'
            
    def place_walls_randomized(self):
            """Place walls randomly, ensuring at least one empty tile per column."""
            total_walls = 0
            placed_walls = set()  # Track placed walls to avoid duplicates

            # Ensure at least one empty tile per column
            empty_y_per_column = [random.randint(0, self.height - 1) for _ in range(self.width)]

            while total_walls < self.wall_num:
                # Randomly pick a position for a wall
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)

                # Make sure the position isn't where we want an empty space in the column
                if x != empty_y_per_column[y] and (x, y) not in placed_walls and self.grid[x][y] == 'empty':
                    if x == 0 and y == 0 or x == 0 and y == 1:
                        continue
                    self.grid[x][y] = 'wall'
                    placed_walls.add((x, y))
                    self.walls.append((x, y))
                    total_walls += 1


    

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        start_time = time.time()
        for step in range(steps):
        #   # Dynamically move walls every 20 steps, Keep commented for Q2
            # if step % 20 == 0:
            #     print(f"Step {step}: Moving walls dynamically...")
            #     self.move_walls_safely()
            #     self.print_grid(self.agents[0])
            if(step == (steps - 1 )and self.time_taken == 0):
                if self.time_taken == 0:
                    self.time_taken = time.time() - start_time
            if self.coins_num == self.agents[0].collected_coins:
                self.time_taken = time.time() - start_time
            if self.is_done():
                return
            self.step()
            


    def percept(self, agent):
        # Return the percept at the agent's current location
        if agent.x < 0 or agent.y < 0 or agent.x >= self.width or agent.y >= self.height:
            return 'out_of_bounds'
        
        if self.grid[agent.x][agent.y] == 'wall':
            return 'hit_wall'
        # If the agent is on a coin, return 'coin'
        if self.grid[agent.x][agent.y] == 'coin':
            return 'coin'
        else:
            return 'empty'  # Otherwise, return 'empty'    

    def print_grid(self, agent=None):
        """Prints the grid with walls, coins, and the agent's position."""
        for x in range(self.width):
            row = []
            for y in range(self.height):
                if (agent is not None and agent.x == x and agent.y == y):
                    row.append('A')  # 'A' for the agent
                elif self.grid[x][y] == 'coin':
                    row.append('C')  # 'C' for coin
                elif self.grid[x][y] == 'wall':
                    row.append('W')  # 'W' for wall
                elif self.grid[x][y] == 'A':
                    row.append('A') 
                elif self.grid[x][y] == 'P':
                    row.append('P')
                else:
                    row.append('.')  # '.' for empty space
            print(' | '.join(row))
        print('\n')
            
    def execute_action(self, agent, action):
        if action == 'move_randomly':
            agent.random_move()
            agent.performance -= 1
        if action == 'remember_wall':
            agent.walls_hit.add((agent.x, agent.y))
            agent.explore_new_tiles()
        elif action == 'explore_new_tiles':
            agent.explore_new_tiles()
        elif action == 'fetch':
            if self.grid[self.agents[0].x][self.agents[0].y] == 'coin':
                agent.fetch()
                self.grid[self.agents[0].x][self.agents[0].y] = 'empty'  # Remove the coin
                agent.explore_new_tiles()
        elif action == 'back_track':
            agent.performance -= 2
            agent.x = agent.prev_x
            agent.y = agent.prev_y
            agent.explore_new_tiles()
        else:
            agent.move(action)
        if agent.collected_coins != self.coins_num:
            agent.update_cost_path()

# Base class for all treasure hunt agents
# Base class for all treasure hunt agents
class BaseTreasureHuntAgent(Agent):
    def __init__(self, environment = None, program=None):
        super().__init__(program)
        self.environment = environment
        self.x = 0
        self.y = 0
        self.prev_x = 0
        self.prev_y = 0
        self.collected_coins = 0
        self.performance = 0
        self.path = []
    
    def fetch(self):
            self.collected_coins += 1
            self.performance += 100
            print(f"Coin collected at ({self.x}, {self.y}). Total coins collected: {self.collected_coins}")
    
    def move(self, direction):
        self.prev_x = self.x
        self.prev_y = self.y
        if direction == 'move_right' and self.x + 1:
            self.x += 1
        elif direction == 'move_left' and self.x - 1 >= 0:
            self.x -= 1
        elif direction == 'move_up' and self.y - 1 >= 0:
            self.y -= 1
        elif direction == 'move_down' and self.y + 1:
            self.y += 1
        
    def update_cost_path(self):
        self.path.append((self.x, self.y))

    
    def random_move(self):
        possible_moves = ['move_right', 'move_left', 'move_up', 'move_down']
        random.shuffle(possible_moves)  # Shuffle to randomize the order of moves

        for move in possible_moves:
            new_x, new_y = self.predict_move(self.x, self.y, move)
            # if self.is_within_bounds(new_x, new_y):
            self.move(move)
            return
        
    def predict_move(self, x, y, direction):
        """Predict the new position based on the move direction."""
        if direction == 'move_right':
            return x + 1, y
        elif direction == 'move_left':
            return x - 1, y
        elif direction == 'move_up':
            return x, y - 1
        elif direction == 'move_down':
            return x, y + 1
        print(' no movement')
        return x, y  # No movement
    

    

def display_solution_path_on_grid(environment, path):
    # Iterate over the path and mark the positions on the grid
    for step in path:
        (agent_position, _) = step  # Extract the agent's position (x, y) from the step
        x, y = agent_position
        # if environment.grid[x][y] == 'P':
        #     environment.grid[y][x] = '.'

        # Mark the path on the grid (e.g., 'P' for path)
        environment.grid[x][y] = 'P'

    # Display the grid with the path
    environment.print_grid()


# Simple Reflex Agent uses random directions to move
class TreasureHuntSimpleReflexAgent(BaseTreasureHuntAgent):
    def __init__(self, program=None):
        self.rules = [
            Rule("coin", "fetch"), 
            Rule("empty", 'move_randomly'), 
            Rule("hit_wall", "back_track"),
            Rule("out_of_bounds", 'back_track')
        ]
        if program is None:
            program = SimpleReflexAgentProgram(self.rules, interpret_input)
        super().__init__(None,program)
    
    def move_randomly(self):
        self.random_move()
        
    def explore_new_tiles(self):
        self.move_randomly()

# Goal-Based Agent
class GoalBasedAgent(BaseTreasureHuntAgent):
    def __init__(self, environment, program=None):
        self.visited_tiles = set()
        self.possible_moves = ['move_right', 'move_left', 'move_up', 'move_down']
        self.treasure_found = False
        self.rules = [
            Rule("coin", "fetch"),
            Rule("empty", "explore_new_tiles"),
            
        ]
        if program is None:
            program = ModelBasedReflexAgentProgram(self.rules, self.update_state, None)
        super().__init__(environment, program)

    def update_state(self, state, action, percept, model):
        return percept

    def explore_new_tiles(self):
        if self.collected_coins == self.environment.coins_num:
            self.treasure_found = True
            return 'NOOP'
        for move in self.possible_moves:
            new_x, new_y = self.predict_move(self.x, self.y, move)
            if (new_x, new_y) not in self.visited_tiles and self.is_within_bounds(new_x, new_y) and self.environment.grid[new_x][new_y] != 'wall':
                self.move(move)
                self.performance -= 1
                self.visited_tiles.add((new_x, new_y))
                return move
        return self.random_move()
    
    def is_within_bounds(self, x, y):
    #     """Check if the predicted position is within the environment bounds."""
         return 0 <= x < self.environment.width and 0 <= y < self.environment.height

    def random_move(self):
        possible_moves = ['move_right', 'move_left', 'move_up', 'move_down']
        random.shuffle(possible_moves)  # Shuffle to randomize the order of moves

        for move in possible_moves:
            new_x, new_y = self.predict_move(self.x, self.y, move)
            if self.is_within_bounds(new_x, new_y) and self.environment.grid[new_x][new_y] != 'wall':
                # Move only if the new position is not a wall
                self.move(move)
                return

# Model-Based Agent
class ModelBasedAgent(BaseTreasureHuntAgent):
    def __init__(self,  program=None):
        self.visited_tiles = set()  # Only store visited locations
        self.x = 0  # Current x position
        self.y = 0  # Current y position
        self.prev_x = 0  # Previous x position
        self.prev_y = 0  # Previous y position
        self.walls_hit = set()
        self.rules = [
            Rule("coin", "fetch"), 
            Rule("empty", 'move_randomly'), 
            Rule("out_of_bounds", 'back_track'),
            Rule("hit_wall", "remember_wall")
        ]
        if program is None:
            program = ModelBasedReflexAgentProgram(self.rules, self.update_state, None)
        super().__init__(None, program)

    def update_state(self, state, action, percept, model):
        """Update the state with the current perception. In this simplified version, we don't maintain a full model."""
        return percept

    def explore_new_tiles(self):
        """Explore unvisited tiles."""
        possible_moves = ['move_up', 'move_down', 'move_right', 'move_left']
        
        # Check for unvisited tiles around the agent
        for move in possible_moves:
            new_x, new_y = self.predict_move(self.x, self.y, move)
            if (new_x, new_y) not in self.visited_tiles and (new_x, new_y) not in self.walls_hit:
                # Found an unvisited tile; move there
                self.visited_tiles.add((new_x, new_y))
                self.move(move)  # Perform the movement
                return move
        
        # If all surrounding tiles are visited, choose a random move (fallback)
        return self.random_move()


    def random_move(self):
        """Random fallback move."""
        possible_moves = ['move_right', 'move_left', 'move_up', 'move_down']
        random.shuffle(possible_moves)  # Shuffle to randomize the order of moves

        for move in possible_moves:
            new_x, new_y = self.predict_move(self.x, self.y, move)
            self.move(move)  # Move if within bounds
            return move



class TreasureHuntProblem(Problem):
    def __init__(self, initial_state, width, height, env):
        """
        initial_state: ((x, y), dirt_locations)
        width, height: dimensions of the grid
        env: reference to the environment
        """
        self.width = width
        self.height = height
        self.env = env
        super().__init__(initial_state)
    
    def actions(self, state):
        """
        Returns the possible actions from the current state considering walls.
        """
        (x, y), dirt_locations = state  # Unpack the agent's position and dirt locations
        actions = []

        # Define possible movement directions (left, right, up, down)
        directions = {
            'Left': (x - 1, y),
            'Right': (x + 1, y),
            'Up': (x, y - 1),
            'Down': (x, y + 1)
        }

        # Check if the neighboring cells are within bounds and not walls
        for action, (new_x, new_y) in directions.items():
            if 0 <= new_x < self.width and 0 <= new_y < self.height:  # Ensure within grid bounds
                if self.env.grid[new_x][new_y] != 'wall':  # Avoid walls
                    actions.append(action)


        if (x, y) in dirt_locations:
            actions.append('fetch')

        return actions

    def result(self, state, action):
        """
        Returns the new state after performing the given action.
        """
        (x, y), dirt_locations = state  # Unpack the current state
        dirt_locations = set(dirt_locations)  # Convert to set for easy manipulation

        # Define the result of each possible action
        if action == 'Left':
            return ((x - 1, y), tuple(dirt_locations))
        elif action == 'Right':
            return ((x + 1, y), tuple(dirt_locations))
        elif action == 'Up':
            return ((x, y - 1), tuple(dirt_locations))
        elif action == 'Down':
            return ((x, y + 1), tuple(dirt_locations))
        elif action == 'fetch':
            # Remove coin at the current position if present
            if (x, y) in dirt_locations:
                dirt_locations.remove((x, y))
                # print(f' Lets check if its really a coin {self.env.grid[x][y]} x , y {x} , {y}')
                self.env.grid[x][y] = 'empty'
            return ((x, y), tuple(dirt_locations))
        

    def goal_test(self, state):
        """
        The goal is to collect all coins.
        """
        _, dirt_locations = state
        return len(dirt_locations) == 0

    def step_cost(self, state, action, result):
        """
        Define the cost of each action.
        """
        if action == 'fetch':
            return -100  # Negative cost to represent a reward for cleaning
        else:
            return 1  # Positive cost for movement

    # use MST with a star
    

    def manhattan_distance_to_nearest_coin(self, node):
        """Heuristic: Manhattan distance to the nearest coin."""
        (agent_position, coin_locations) = node.state  # Access the state of the node, not the problem
        
        if not coin_locations:  # No coins left
            return 0

        # Find the nearest coin by Manhattan distance
        x1, y1 = agent_position
        nearest_distance = float('inf')

        for goal in coin_locations:
            x2, y2 = goal
            distance = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
            nearest_distance = min(nearest_distance, distance)

        return nearest_distance


    def greedy_heuristic(self,node, *args):
        """Heuristic function for Greedy Best-First Search: 
        Euclidean distance to the nearest coin."""
        (agent_position, coin_locations) = node.state  # Access the state of the node, not the problem
        if len(coin_locations) == 0:
            return 0  # No more coins, goal state
        return min(self.manhattan_distance_to_all_goals(agent_position, coin) for coin in coin_locations)
    
    def manhattan_distance_to_all_goals(self,node):
        """Heuristic: Sum of Manhattan distances from the agent to all remaining coins."""
        # Unpack the agent's location and the remaining coins' locations
        agent_location, remaining_goals = node.state

        x1, y1 = agent_location  # Agent's current position
        total_distance = 0
        
        # Calculate the Manhattan distance from the agent to each remaining coin
        for goal in remaining_goals:
            x2, y2 = goal  # Coin's location
            total_distance += abs(x2 - x1) + abs(y2 - y1)
        
        return total_distance


def euclidean_distance(first_location, second_location):
    # print('first_location ',first_location, '' )
    """The Euclidean distance between two (x, y) points."""
    val = math.sqrt(pow(second_location[0] - first_location[0],2) + 
          pow(second_location[1] - first_location[1],2))
    return val


# Step 4: Automatically Generate the Initial State from the Environment
def get_initial_state_from_env(env):
    print(' inside get_initial_state_from_env')
    """Extract the agent's position and dirt locations from the environment."""
    agent_location = None
    dirt_locations = set()
    # Iterate over all things in the environment
    for thing in env.coins:
        dirt_locations.add(thing)
        # Step 2: Convert the set to a list
    my_list = list(dirt_locations)

    return ((env.agents[0].x, env.agents[0].y), tuple(my_list))  # Return as a tuple for hashability

def print_search_algorithms_results(uninformed_search_results, environment, file_name="search_results_informed.txt"):
    # Prepare header and divider
    header = f"{'Algorithm':<35} {'Solution Steps':<30} {'Path':<50} {'Total Cost':<10} {'Search Time (s)':<15}\n"
    divider = "-" * 150 + "\n"

    # Start with the header
    output = header + divider

    # Loop through each dictionary in the list of results
    for result in uninformed_search_results:
        for algorithm, solution_data in result.items():
            if isinstance(solution_data, dict):  # Check if it's a dictionary containing both solution and time
                solution_node = solution_data.get('solution_node')
                search_time = solution_data.get('search_time', "N/A")
                if solution_node:
                    full_path = [node.state for node in solution_node.path()]  # Get the full path
                    solution_steps = str(solution_node.solution()[:20])+ "..." if len(str(solution_node.solution())) > 28 else str(solution_node.solution())
                    path = str([node.state for node in solution_node.path()])[:10] + "..." if len(str([node.state for node in solution_node.path()])) > 48 else str([node.state for node in solution_node.path()])
                    total_cost = solution_node.path_cost

                else:
                    solution_steps = "No solution"
                    path = "N/A"
                    total_cost = "N/A"

        
                # uncomment to Display the solution path based on actions
                if solution_node:  # Only display if there is a valid solution
                    display_solution_path_on_grid(environment, full_path)
               
                # Format the row with the data, including the search time
                output += f"{algorithm:<35} {solution_steps:<30}  {total_cost:<10} {search_time:<15}\n"

    # Output to the console or write to a file
    if file_name:
        with open(file_name, 'a') as file:
            file.write(output)
    else:
        print(output)

def run_search_methods(searchers, searchProblem, environment):
        searchers_results = []
        # Iterate over each search algorithm and append the result
        for search_name, search_function in searchers:
            start_time = time.time()
            solution_node = search_function(searchProblem)
            search_time = time.time() - start_time
            searchers_results.append({search_name: {'solution_node': solution_node, 'search_time': search_time}})
            # solution_steps = solution_node.solution()  # Get the full solution steps
            # print(f'is it sol {solution_steps}')
        print_search_algorithms_results(searchers_results, environment)


def compare_uninformed_searchers(environment, agent):
        initial_state = get_initial_state_from_env(environment)
        treasure_hunt_problem = TreasureHuntProblem(initial_state, environment.width, environment.height, environment)
        searchers = [
            
           ("Breadth-First Search", breadth_first_graph_search),
           ("Depth-First Search", depth_first_graph_search),
           ("uniform_cost_search", uniform_cost_search)
        ]
        environment.print_grid()
        run_search_methods(searchers,treasure_hunt_problem, environment)

        # Print the results for this agent
def run_informed_search_methods(searchers, searchProblem, environment):
    searchers_results = []
    for search_name, search_function,heuristic in searchers:
        start_time = time.time()
        solution_node = search_function(searchProblem, heuristic)
        search_time = time.time() - start_time
        searchers_results.append({search_name: {'solution_node': solution_node, 'search_time': search_time}})
    print_search_algorithms_results(searchers_results, environment)



def compare_informed_searchers(environment, agent):
        initial_state = get_initial_state_from_env(environment)

        # Define the treasure hunt problem for the search
        treasure_hunt_problem = TreasureHuntProblem(initial_state, environment.width, environment.height, environment)

        searchers = [
           ("astar_search", astar_search, treasure_hunt_problem.manhattan_distance_to_all_goals),
            ("best_first_graph_search", best_first_graph_search, treasure_hunt_problem.manhattan_distance_to_nearest_coin),
        #    ("recursive_best_first_search", recursive_best_first_search, treasure_hunt_problem.manhattan_distance_to_all_goals)
        ]
        
        run_informed_search_methods(searchers,treasure_hunt_problem, environment)


def run_agents(treasure_hunt_env, agent_classes) :
  
    agents_performance = {}
    for agent_class, agent_name in agent_classes:
        treasure_hunt_env_copy = copy.deepcopy(treasure_hunt_env)
        # treasure_hunt_env_copy.print_grid()
        # Instantiate the agent
        if agent_name == 'Goal-Based Agent':
            agent = agent_class(treasure_hunt_env_copy)
        else: 
            agent = agent_class()
        
        # Add the agent to the environment
        treasure_hunt_env_copy.add_thing(agent)
        treasure_hunt_env.print_grid(agent)

        # Run the agent in the environment until completion
        treasure_hunt_env_copy.run()

        # Store the agent's performance
        agents_performance[agent_name] = agent.performance

        # Print the performance after the agent finishes
        print(f"{agent_name} Performance: {agent.performance}\n  collected coins {agent.collected_coins} \n time taken {treasure_hunt_env_copy.time_taken} \n path cost {agent.path}")
        if agent_name == 'Random Agent':
           space_complexity = 1
        elif agent_name == 'Model-Based Agent':
            space_complexity = len(agent.visited_tiles) + len(agent.walls_hit)
        else:
            space_complexity = len(agent.visited_tiles)

        # Optionally, print the grid after each agent completes
        treasure_hunt_env_copy.print_grid(agent)
        headers = ['Width', 'Height', 'Coins', 'Walls', 'steps', 'Agent', 'Collected Coins', 'Performance', 'Path Cost', 'Space Complexity', 'Time Taken']
        results = []
        results.append({
            'Agent': agent_name,
            'Performance': agent.performance,
            'Collected Coins': agent.collected_coins,
            'Time Taken':  treasure_hunt_env_copy.time_taken,
            'Path Cost': len(agent.path),
            'Space Complexity': space_complexity,
            'Width': 7 ,
            'Height': 7,
            'Coins': 10,
            'Walls': 10,
            'steps': 800
        })

        with open('agents_results_comparision_final.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            # Write the header
            writer.writeheader()
            for result in results:
                writer.writerow(result)


    return agents_performance

def main():

    ############################# Q1 uncomment to run ##########################
    # Define the environment size and number of coins
    width, height, coins_num, wall_num = 10, 10, 5 , 6
    # treasure_hunt_env = treasure_hunt_environment(width, height, coins_num,wall_num)
    # # Create and test each agent
    # agents = [
    #       (TreasureHuntSimpleReflexAgent, "Random Agent"),
    #        (GoalBasedAgent, "Goal-Based Agent"),
    #        (ModelBasedAgent, "Model-Based Agent")
    # ]

    # run_agents(treasure_hunt_env, agents)
 
    # treasure_hunt_env.print_grid()

##################### Q2 ################################################
    # Search Algorithmst uncomment to run
    # treasure_hunt_env = treasure_hunt_environment(width, height, coins_num, wall_num)
    # goal_based_agent_rlrmrnt = GoalBasedAgent(treasure_hunt_env)
    # treasure_hunt_env.add_thing(goal_based_agent_rlrmrnt, (0, 0))
    
    #compare_uninformed_searchers(treasure_hunt_env, goal_based_agent_rlrmrnt)
    # compare_informed_searchers(treasure_hunt_env, goal_based_agent_rlrmrnt)

if __name__ == "__main__":
    main()
       
       

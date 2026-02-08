###Treasure Hunt Agents and Search Algorithms

This project implements a dynamic 2D treasure hunt environment designed to evaluate intelligent agents and classical search algorithms. The game is set in a partially observable grid world where an agent must navigate through tiles to collect randomly placed coins while avoiding dynamically changing walls.

The environment is stochastic: walls are randomly placed and may move during execution, yet the system guarantees that a valid path to every coin always exists. To enforce this constraint, a reachability validation mechanism is implemented using Breadth-First Search (BFS). After each wall placement (and during dynamic updates), BFS ensures that all coin positions remain reachable from the agent’s starting state. If not, wall density is adjusted to maintain solvability.

Multiple agent architectures were implemented and compared, including a simple reflex agent, a goal-based agent, and a model-based agent. Each agent was evaluated using performance score, path cost, time taken, number of collected coins, and space complexity.

The environment was also reformulated as a formal search problem, where the objective is to collect all coins with minimal path cost. Both uninformed (BFS, DFS, Uniform Cost Search) and informed search algorithms (Greedy Best-First Search, A*, Recursive Best-First Search) were implemented and analyzed. Heuristics based on Manhattan distance were used to guide informed search methods, with A* achieving the most optimal paths in larger grids.

This project demonstrates the relationship between agent design, search strategy, heuristic guidance, computational trade-offs, and optimality.

This game and analysis were developed as part of the Knowledge Representation course.

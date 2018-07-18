# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #Define the start node
    start_node = problem.getStartState()
    #Keep track of the nodes visited
    visited_nodes = []

    #If the start is the end we are already done.
    if problem.isGoalState(start_node):
        return []

    #Create a stack for open set
    nodes = util.Stack()
    #Update open set with start node
    nodes.push((start_node, []))

    #Traverse over nodes until game is not over and open set is not empty
    while not nodes.isEmpty() and not problem.isGoalState(start_node):
        #Get the current node 
        current_node, actions = nodes.pop()
        #Since we visited it; update 
        visited_nodes.append(current_node)
		
        for (successor, action, stepCost) in problem.getSuccessors(current_node):
            if successor not in visited_nodes:
                start_node = successor
                moves = actions + [action]
                nodes.push((successor, moves))

    return moves
    util.raiseNotDefined()

  
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
	#Define the start node 
    start_node = problem.getStartState()
    #Keep track of the nodes visited
    visited_nodes = []

	#Since we are starting at this node, we are also visiting it
    visited_nodes.append(start_node)
	
	#If we are already at the goal then STOP. 
    if problem.isGoalState(start_node):
        return []

    #Create a queue the open set
    nodes = util.Queue()
	#Add start node to open set
    nodes.push((start_node, []))

	
    while not nodes.isEmpty():
		#Visit each node
        current_node, actions = nodes.pop()
		#Check if its the goal
        if problem.isGoalState(current_node):
            return actions
		#It's not the goal, proceed	
        for (successor, action, stepCost) in problem.getSuccessors(current_node):
			#If we haven't visited this node already
            if successor not in visited_nodes:
				#Now we have visited it, so update the list
                visited_nodes.append(successor)
                moves = actions + [action]
                nodes.push((successor, moves))

    return actions
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
	#Get the start node and set up empty list of visited nodes. 
    start_node = problem.getStartState()
    visited_nodes = []

	#If we are already at the goal then we're done. 
    if problem.isGoalState(start_node):
        return []

	#Get the priority queue for open set
    nodes = util.PriorityQueue()
    nodes.push((start_node, []), 0)

    while not nodes.isEmpty():
        #Visit each node
        current_node, actions = nodes.pop()
        #Check if its the goal
        if problem.isGoalState(current_node):
            return actions
        #If we haven't visited this node already
        if current_node not in visited_nodes:
            for (successor, action, stepCost) in problem.getSuccessors(current_node):
                #If we haven't visited the successor already
                if successor not in visited_nodes:
                    moves = actions + [action]
                    nodes.push((successor, moves), problem.getCostOfActions(moves))
        #Now that we have visited, update it
        visited_nodes.append(current_node)

    return actions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
	#Get the start node and set up empty list of visited nodes.
    start_node = problem.getStartState()
    visited_nodes = []
	
	#If we are already at the goal then quit.
    if problem.isGoalState(start_node):
        return []

    #Get the priority queue for open set
    nodes = util.PriorityQueue()
    nodes.push((start_node, []), nullHeuristic(start_node, problem))

    while not nodes.isEmpty():
		#Visit each node
        current_node, actions = nodes.pop()
        #Check if it's goal
        if problem.isGoalState(current_node):
            return actions
        if current_node not in visited_nodes:
            for (successor, action, stepCost) in problem.getSuccessors(current_node):
                if successor not in visited_nodes:
					#Update the moves list
                    moves = actions + [action]
					#Get the total cost for this
                    total_cost = problem.getCostOfActions(moves) + heuristic(successor, problem)
                    nodes.push((successor, moves), total_cost)
        visited_nodes.append(current_node)

    return actions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

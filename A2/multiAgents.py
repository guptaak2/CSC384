# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        #Get a list of the food in the current state.
        currentFoodList = currentGameState.getFood()
        
        #Loop through each food in the list to determine the closest one
        #First define an arbitrary distance to compare against
        minimumFoodDistance = 100
        i=0
        for currentFoodIterator in currentFoodList.asList():
            #Check if distance to item is closer than previous min
            #Use manhattan distance from utils
            foodDistance = manhattanDistance(currentFoodIterator, newPos)
            if foodDistance < minimumFoodDistance:
                minimumFoodDistance = foodDistance
        score = -minimumFoodDistance
        #Now find out how far is the nearest ghost
        #Define an arbitrary minimum distance between pacman and ghost
        minimumGhostDistance = 100
        pacmanGhostDistance = 0 #Move without any effect from ghost
        #Loop through each ghost position to determine how close it is
        for ghost in newGhostStates:
            ghostDistance = manhattanDistance(ghost.getPosition(), newPos)
            #Check to see if ghost is really close (1 away)
            if ghostDistance <= 1:
               #It is close, update minimum distance
               pacmanGhostDistance = minimumGhostDistance
        #Return an updated score using the minimum food and ghost distance
        score = score - pacmanGhostDistance
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        #Use helper function action cost to determine the best action for next move. 
        cost = self.actionCost(gameState, 0)
        action = cost[0]
        return action
        
    def actionCost(self, gameState, depth):
        """
            Returns the best action at this state and the cost for that action. 
        """
        #Check to see if the game has not ended or what the depth is
        gameEndedFlag = gameState.isWin() or gameState.isLose()
        if depth == self.depth * gameState.getNumAgents() or gameEndedFlag == True:
            #Game ended
            return (None, self.evaluationFunction(gameState))
        elif depth % gameState.getNumAgents() == 0:
            #Max agent
            return self.performMiniMax(gameState, depth, True)
        else:
            #Min agent
            return self.performMiniMax(gameState, depth, False)
    
    def performMiniMax(self, gameState, depth, isMax):
        """
            Performs the minimax algorithm. Returns the best action and cost.
            If isMax is true it performs algorithm for a max agent. Min otherwise. 
        """
        #Get the legal actions 
        if isMax == True: 
            agentIndex = 0;
            legalActions = gameState.getLegalActions(agentIndex)
        else:
            agentIndex = depth % gameState.getNumAgents()
            legalActions = gameState.getLegalActions(agentIndex)
        
        #If there are no actions, just quit. 
        if len(legalActions) != 0:
            #Pick an arbitrary min/max value to compare against
            if isMax == True:
                valueToCompareAgainst = (None, -float("inf"))
            else:
                valueToCompareAgainst = (None, float("inf"))
            
            #Loop over each action
            for action in legalActions:
                currentSuccessor = gameState.generateSuccessor(agentIndex, action)
                #Get cost for this action
                cost = self.actionCost(currentSuccessor, depth + 1) 
                #Compare the cost with the previous max/min value
                #Update Accordingly
                if isMax == True:
                    if cost[1] > valueToCompareAgainst[1]:
                        #Have a new max. Update
                        valueToCompareAgainst = (action, cost[1])
                else:
                    if cost[1] < valueToCompareAgainst[1]:
                        #Have a new min. Update. 
                        valueToCompareAgainst = (action, cost[1])
            return valueToCompareAgainst
        else:
            #No actions
            #Return (No Action, Value)
            return (None, self.evaluationFunction(gameState))
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
    #Use helper function action cost to determine the best action for next move. 
        cost = self.actionCost(gameState, 0, -float("inf"), float("inf"))
        action = cost[0]
        return action

    def actionCost(self, gameState, depth, alpha, beta):
        """
            Returns the best action at this state and the cost for that action. 
        """

    #Check to see if the game has not ended or what the depth is
        gameEndedFlag = gameState.isWin() or gameState.isLose()
        if depth == self.depth * gameState.getNumAgents() or gameEndedFlag == True:
            #Game ended
            return (None, self.evaluationFunction(gameState))
        elif depth % gameState.getNumAgents() == 0:
            #Max agent
            return self.performAlphaBeta(gameState, depth, True, alpha, beta)
        else:
            #Min agent
            return self.performAlphaBeta(gameState, depth, False, alpha, beta)

    def performAlphaBeta(self, gameState, depth, isMax, alpha, beta):
        """
            Performs the minimax algorithm. Returns the best action and cost.
            If isMax is true it performs algorithm for a max agent. Min otherwise. 
        """
        #Get the legal actions 
        if isMax == True: 
            agentIndex = 0;
            legalActions = gameState.getLegalActions(agentIndex)
        else:
            agentIndex = depth % gameState.getNumAgents()
            legalActions = gameState.getLegalActions(agentIndex)
        
        #If there are no actions, just quit. 
        if len(legalActions) != 0:
            #Pick an arbitrary min/max value to compare against
            if isMax == True:
                valueToCompareAgainst = (None, -float("inf"))
            else:
                valueToCompareAgainst = (None, float("inf"))
            
            #Loop over each action
            for action in legalActions:
                currentSuccessor = gameState.generateSuccessor(agentIndex, action)
                #Get cost for this action
                cost = self.actionCost(currentSuccessor, depth + 1, alpha, beta) 
                #Compare the cost with the previous max/min value
                #Update Accordingly
                if isMax == True:
                    if cost[1] > valueToCompareAgainst[1]:
                        #Have a new max. Update
                        valueToCompareAgainst = (action, cost[1])
                        #Need to compare with beta
                    if valueToCompareAgainst[1] >= beta:
                        return valueToCompareAgainst

                    alpha = max(alpha, valueToCompareAgainst[1])
                else:
                    if cost[1] < valueToCompareAgainst[1]:
                        #Have a new min. Update. 
                        valueToCompareAgainst = (action, cost[1])
                        #Need to compare with alpha
                    if valueToCompareAgainst[1] <= alpha:
                        return valueToCompareAgainst

                    beta = min(beta, valueToCompareAgainst[1])

            return valueToCompareAgainst
        else:
            #No actions
            #Return (No Action, Value)
            return (None, self.evaluationFunction(gameState))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
    #Use helper function action cost to determine the best action for next move. 
        cost = self.actionCost(gameState, 0)
        action = cost[0]
        return action

    def actionCost(self, gameState, depth):
        """
            Returns the best action at this state and the cost for that action. 
        """
    #Check to see if the game has not ended or what the depth is
        gameEndedFlag = gameState.isWin() or gameState.isLose()
        if depth == self.depth * gameState.getNumAgents() or gameEndedFlag == True:
            #Game ended
            return (None, self.evaluationFunction(gameState))
        elif depth % gameState.getNumAgents() == 0:
            #Max agent
            return self.performExpectimax(gameState, depth, True)
        else:
            #Min agent
            return self.performExpectimax(gameState, depth, False)

    def performExpectimax(self, gameState, depth, isMax):
        """
            Performs the minimax algorithm. Returns the best action and cost.
            If isMax is true it performs algorithm for a max agent. Min otherwise. 
        """
        #Get the legal actions 
        if isMax == True: 
            agentIndex = 0;
            legalActions = gameState.getLegalActions(agentIndex)
        else:
            agentIndex = depth % gameState.getNumAgents()
            legalActions = gameState.getLegalActions(agentIndex)
        
        #If there are no actions, just quit. 
        if len(legalActions) != 0:
            #Pick an arbitrary min/max value to compare against
            if isMax == True:
                valueToCompareAgainst = (None, -float("inf"))
            else:
                valueToCompareAgainst = (None, 0)
                succCosts = []
                
            #Loop over each action
            for action in legalActions:
                currentSuccessor = gameState.generateSuccessor(agentIndex, action)
                #Get cost for this action
                cost = self.actionCost(currentSuccessor, depth + 1) 
                #Compare the cost with the previous max/min value
                #Update Accordingly
                if isMax == True:
                    if cost[1] > valueToCompareAgainst[1]:
                        #Have a new max. Update
                        valueToCompareAgainst = (action, cost[1])
                else:
                    succCosts.append(cost[1])
                    valueToCompareAgainst = (None, sum(succCosts) / float(len(succCosts)))

            return valueToCompareAgainst
        else:
            #No actions
            #Return (No Action, Value)
            return (None, self.evaluationFunction(gameState))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    The following excerpt is from our Assignment handout:
    "One way you might want to write your evaluation function is to use a linear combination of features.
    That is, compute values for features about the state that you think are important, and then combine
    those features by multiplying them by different values and adding the results together.  
    You might decide what to multiply each feature by based on how important you think it is."
    That's exactly what I did, I used a linear combination of features such as ghosts, food and pellets.
    I learnt about these features from currentGameState (q1). the score from food is implemented from A1. We use 
    manhattanDistance to calculate the distance between pacman and ghost/food/pellet. All blocks are
    nearly identical to each other.
    """
    "*** YOUR CODE HERE ***"

    def getScore(pos_pacman, all_ghosts, all_food_lst, all_pellets):   
        # get current game score
        score = currentGameState.getScore() 
        
        # get score from ghosts
        score_ghost = 0
        for ghost in all_ghosts:
            dist_ghost = manhattanDistance(pos_pacman, ghost.getPosition())
        if ghost.scaredTimer > 0:
          score_ghost += pow(max(8 - dist_ghost, 0), 2)
        else:
          score_ghost -= pow(max(7 - dist_ghost, 0), 2)

        # get score from food
        score_food = 0
        dist_food = []
        for food in all_food_lst:
            dist_food.append(1.0 / manhattanDistance(pos_pacman, food))
        if len(dist_food) > 0:
            score_food = max(dist_food)
        else:
            score_food = 0

        # get score from pellets
        score_pellet = 0
        score_pellet_lst = []
        for pellet in all_pellets:
            score_pellet_lst.append(50.0 / manhattanDistance(pos_pacman, pellet))
        if len(score_pellet_lst) > 0:
            score_pellet = max(score_pellet_lst)
        else:
            score_pellet = 0

        return score, score_ghost, score_food, score_pellet

    # get pacman's position
    pos_pacman = currentGameState.getPacmanPosition()
    # get all ghosts from the current game state
    all_ghosts = currentGameState.getGhostStates()
    # get location of all food from the current game state
    all_food_lst = currentGameState.getFood().asList()
    # get location of all pellets from the current game state
    all_pellets = currentGameState.getCapsules()

    score, score_ghost, score_food, score_pellet = getScore(pos_pacman, all_ghosts, all_food_lst, all_pellets)
    # return linear combination of features
    return score + score_ghost + score_food + score_pellet

# Abbreviation
better = betterEvaluationFunction

